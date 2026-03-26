"""
Ensemble Model Framework for Genomic Variant Classification
============================================================
Implements 8 base classifiers + 1 stacking meta-learner.

Base classifiers:
  1. Random Forest         (sklearn)
  2. XGBoost               (xgboost)
  3. LightGBM              (lightgbm)
  4. SVM (RBF kernel)      (sklearn)
  5. Logistic Regression   (sklearn)
  6. Gradient Boosting     (sklearn)
  7. 1D-CNN                (TensorFlow/Keras)  — sequence-based
  8. Feedforward NN        (TensorFlow/Keras)  — tabular features

Meta-learner:
  Logistic Regression stacker trained on OOF predictions

CHANGES FROM PHASE 1:
  - Consolidated src/models/ensemble.py + src/models/variant_ensemble.py
    into this single file (Issue A).
  - Removed unused `k` parameter from encode_sequence; docstring fixed (Bug 7).
  - codon_position removed from TABULAR_FEATURES — it was always 0 (Issue P).
  - base_estimators cleared after fitting to avoid double-memory (Issue H).
  - from __future__ import annotations added (Issue N).
  - Module-level logging.basicConfig removed (Issue L).
  - VariantEnsemble.fit() / evaluate() now robust when CNN/NN are excluded.
  # CHANGES IN PHASE 2:
#   - alphamissense_score and GTEx features promoted to TABULAR_FEATURES.
#   - splice_ai_score remains in PHASE_2_FEATURES (SpliceAI connector pending).
  # CHANGES IN PHASE 4:
#   - splice_ai_score and eve_score promoted to TABULAR_FEATURES (functional scores).
#   - codon_position, dbsnp_af, omim_n_diseases, omim_is_autosomal_dominant,
#     clingen_validity_score, hgmd_is_disease_mutation, hgmd_n_reports added.
#   - PHASE_2_FEATURES cleared (all planned features now promoted).
#   - Total: 55 features.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import xgboost as xgb
import lightgbm as lgb

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Feature definitions  (55 features — must match DataPrepPipeline._engineer_features)
# ---------------------------------------------------------------------------

# VEP consequence → ordinal severity (shared with real_data_prep.py)
CONSEQUENCE_SEVERITY: dict[str, int] = {
    "transcript_ablation":                  10,
    "splice_acceptor_variant":               9,
    "splice_donor_variant":                  9,
    "stop_gained":                           9,
    "frameshift_variant":                    8,
    "stop_lost":                             8,
    "start_lost":                            8,
    "transcript_amplification":              7,
    "inframe_insertion":                     6,
    "inframe_deletion":                      6,
    "missense_variant":                      5,
    "protein_altering_variant":              5,
    "splice_region_variant":                 4,
    "incomplete_terminal_codon_variant":     3,
    "start_retained_variant":                3,
    "stop_retained_variant":                 3,
    "synonymous_variant":                    2,
    "coding_sequence_variant":               2,
    "5_prime_UTR_variant":                   2,
    "3_prime_UTR_variant":                   2,
    "non_coding_transcript_exon_variant":    1,
    "intron_variant":                        1,
    "NMD_transcript_variant":                1,
    "upstream_gene_variant":                 0,
    "downstream_gene_variant":               0,
    "intergenic_variant":                    0,
}

TABULAR_FEATURES = [
    # Allele frequency (6)
    "af_raw",               # gnomAD v4 allele frequency (raw)
    "af_log10",             # log10(af + 1e-8)
    "af_is_absent",         # 1 if AF == 0 (absent from gnomAD)
    "af_is_ultra_rare",     # 1 if AF < 0.0001
    "af_is_rare",           # 1 if 0.0001 ≤ AF < 0.001
    "af_is_common",         # 1 if AF ≥ 0.01
    # Variant type (7)
    "ref_len",              # length of reference allele
    "alt_len",              # length of alternate allele
    "len_diff",             # |alt_len − ref_len|
    "is_snv",               # 1 if single nucleotide variant
    "is_insertion",         # 1 if alt longer than ref
    "is_deletion",          # 1 if ref longer than alt
    "is_indel",             # 1 if insertion or deletion
    # Consequence (6)
    "consequence_severity", # ordinal: 0 (intergenic) → 10 (transcript ablation)
    "is_loss_of_function",  # stop_gained / frameshift / splice_donor|acceptor / start|stop_lost
    "is_missense",          # missense_variant
    "is_synonymous",        # synonymous_variant
    "is_splice",            # any splice consequence
    "in_coding",            # missense / synonymous / stop / frameshift / inframe / splice
    # Functional scores (9)
    "cadd_phred",           # CADD PHRED (higher = more deleterious); default 15.0
    "sift_score",           # SIFT (lower = more deleterious); default 0.5
    "polyphen2_score",      # PolyPhen-2 HDIV score; default 0.5
    "revel_score",          # REVEL ensemble score; default 0.5
    "phylop_score",         # phyloP conservation; default 0.0
    "gerp_score",           # GERP++ RS; default 0.0
    "alphamissense_score",  # AlphaMissense pathogenicity; 0.5 = not covered
    "splice_ai_score",      # SpliceAI max delta score; 0.0 = no splice disruption
    "eve_score",            # EVE evolutionary model; 0.5 = not covered/ambiguous
    # Binary score flags + meta-score (5)
    "cadd_high",                    # cadd_phred ≥ 20
    "sift_deleterious",             # sift_score < 0.05
    "polyphen_probably_damaging",   # polyphen2_score ≥ 0.908
    "revel_pathogenic",             # revel_score ≥ 0.5
    "n_tools_pathogenic",           # count of the 4 binary flags above
    # Gene-level (4)
    "gene_constraint_oe",           # gnomAD pLoF observed/expected ratio
    "gene_is_constrained",          # gene_constraint_oe < 0.35
    "n_pathogenic_in_gene",         # ClinVar pathogenic variant count in this gene
    "gene_has_known_disease",       # n_pathogenic_in_gene > 0
    # Protein features (UniProt) (2)
    "has_uniprot_annotation",               # gene has any UniProt annotation
    "n_known_pathogenic_protein_variants",  # known pathogenic protein variants
    # Expression / regulatory (GTEx v8) (6)
    "gtex_max_tpm",             # max median TPM across tissues (gene level)
    "gtex_n_tissues_expressed", # tissues with median TPM ≥ 1.0
    "gtex_tissue_specificity",  # 1 − mean_tpm/max_tpm
    "gtex_is_eqtl",             # 1 if significant eQTL in any tissue
    "gtex_min_eqtl_pval",       # max −log10(p) eQTL across tissues
    "gtex_max_abs_effect",      # max |beta| eQTL effect size
    # Variant coding context (2)
    "codon_position",           # position within codon (1, 2, 3); 0 = non-coding
    "dbsnp_af",                 # dbSNP AF supplement for variants absent from gnomAD
    # Gene-disease annotation (3)
    "omim_n_diseases",              # number of OMIM phenotype entries for the gene
    "omim_is_autosomal_dominant",   # 1 if gene has autosomal dominant OMIM phenotype
    "clingen_validity_score",       # ClinGen Gene Validity score (0-5)
    # HGMD (2)
    "hgmd_is_disease_mutation",     # 1 if classified DM in HGMD
    "hgmd_n_reports",               # number of HGMD records for this variant
    # Chromosome context (3)
    "is_autosome",              # chrom in 1–22
    "is_sex_chrom",             # chrom in X, Y
    "is_mitochondrial",         # chrom in MT, M
    # GNN-derived score (1) — optional; default 0.5 (ambiguous) when GNN absent
    "gnn_score",                # Graph Attention Network pathogenicity score (0–1)
    # RNA splice-context features (4) — Phase 6.1; gated by is_splice / severity ≥ 7
    "maxentscan_score",         # MaxEntScan splice site strength (log-odds); 0.0 = non-splice
    "dist_to_splice_site",      # distance in bp to nearest splice site; 50 = at boundary
    "exon_number",              # VEP exon number; 0 = non-exonic / unknown
    "is_canonical_splice",      # 1 if variant at canonical GT-AG dinucleotide
    # Protein structure features (4) — Phase 6.2; gated by is_missense
    "alphafold_plddt",          # AlphaFold pLDDT at mutated residue (0–100); 50 = unknown
    "solvent_accessibility",    # relative solvent accessibility (0–1); 0.5 = unknown
    "secondary_structure_context",  # 0=loop, 1=helix, 2=sheet; 0 = unknown
    "dist_to_active_site",      # Cα distance to nearest active site (Å); 100 = unknown
]
# Total: 6+7+6+9+5+4+2+6+2+3+2+3+1+4+4 = 64

# All planned features have been promoted — PHASE_2_FEATURES is now empty
PHASE_2_FEATURES: list[str] = []

SEQUENCE_FEATURES = ["fasta_seq"]   # 101 bp context window, one-hot encoded


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------
@dataclass
class EnsembleConfig:
    n_folds:        int   = 5
    random_state:   int   = 42
    calibrate:      bool  = True         # Platt scaling on base models
    class_weight:   str   = "balanced"   # handles class imbalance
    n_jobs:         int   = -1
    model_dir:      Path  = Path("models/ensemble")

    def __post_init__(self) -> None:
        self.model_dir = Path(self.model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive the 56 tabular features from a raw variant DataFrame.

    Mirrors DataPrepPipeline._engineer_features() in src/data/real_data_prep.py.
    All missing columns are filled with population-median defaults so the
    function is safe to call on partially-populated DataFrames.

    Input columns consumed (all optional with safe defaults):
        allele_freq, ref, alt, consequence,
        cadd_phred, sift_score, polyphen2_score, revel_score,
        phylop_score, gerp_score, alphamissense_score,
        splice_ai_score, eve_score,
        gene_constraint_oe, n_pathogenic_in_gene,
        has_uniprot_annotation, n_known_pathogenic_protein_variants,
        gtex_max_tpm, gtex_n_tissues_expressed, gtex_tissue_specificity,
        gtex_is_eqtl, gtex_min_eqtl_pval, gtex_max_abs_effect,
        codon_position, dbsnp_af,
        omim_n_diseases, omim_is_autosomal_dominant, clingen_validity_score,
        hgmd_is_disease_mutation, hgmd_n_reports,
        gnn_score,   # GNN pathogenicity score; default 0.5 if GNN not run
        chrom
    """
    feats = pd.DataFrame(index=df.index)

    # --- Allele frequency (6 features) ---
    af = df.get("allele_freq", pd.Series([0.0] * len(df), index=df.index)).fillna(0.0).astype(float).clip(lower=0)
    feats["af_raw"]           = af
    feats["af_log10"]         = np.log10(af + 1e-8)
    feats["af_is_absent"]     = (af == 0).astype(int)
    feats["af_is_ultra_rare"] = (af < 0.0001).astype(int)
    feats["af_is_rare"]       = ((af >= 0.0001) & (af < 0.001)).astype(int)
    feats["af_is_common"]     = (af >= 0.01).astype(int)

    # --- Variant type (7 features) ---
    ref = df.get("ref", pd.Series(["A"] * len(df), index=df.index)).fillna("A")
    alt = df.get("alt", pd.Series(["A"] * len(df), index=df.index)).fillna("A")
    ref_len = ref.str.len().clip(lower=1)
    alt_len = alt.str.len().clip(lower=1)
    feats["ref_len"]      = ref_len
    feats["alt_len"]      = alt_len
    feats["len_diff"]     = (alt_len - ref_len).abs()
    feats["is_snv"]       = ((ref_len == 1) & (alt_len == 1)).astype(int)
    feats["is_insertion"] = (alt_len > ref_len).astype(int)
    feats["is_deletion"]  = (ref_len > alt_len).astype(int)
    feats["is_indel"]     = (feats["is_insertion"] | feats["is_deletion"]).astype(int)

    # --- Consequence (6 features) ---
    consequence = df.get("consequence", pd.Series([""] * len(df), index=df.index)).fillna("")
    feats["consequence_severity"] = consequence.map(
        lambda c: max(
            (CONSEQUENCE_SEVERITY.get(term, 0) for term in str(c).split("&")),
            default=0,
        )
    )
    feats["is_loss_of_function"] = consequence.str.contains(
        "stop_gained|frameshift|splice_donor|splice_acceptor|start_lost|stop_lost",
        case=False, na=False,
    ).astype(int)
    feats["is_missense"]   = consequence.str.contains("missense",   case=False, na=False).astype(int)
    feats["is_synonymous"] = consequence.str.contains("synonymous", case=False, na=False).astype(int)
    feats["is_splice"]     = consequence.str.contains("splice",     case=False, na=False).astype(int)
    feats["in_coding"]     = consequence.str.contains(
        "missense|synonymous|stop|frameshift|inframe|splice", case=False, na=False,
    ).astype(int)

    # --- Functional scores (9 features) ---
    score_defaults = {
        "cadd_phred":         15.0,
        "sift_score":          0.5,
        "polyphen2_score":     0.5,
        "revel_score":         0.5,
        "phylop_score":        0.0,
        "gerp_score":          0.0,
        "alphamissense_score": 0.5,
        "splice_ai_score":     0.0,
        "eve_score":           0.5,
    }
    for col, default in score_defaults.items():
        feats[col] = df.get(col, pd.Series([default] * len(df), index=df.index)).fillna(default).astype(float)

    # --- Binary score flags + meta-score (5 features) ---
    feats["cadd_high"]                  = (feats["cadd_phred"]     >= 20).astype(int)
    feats["sift_deleterious"]           = (feats["sift_score"]      < 0.05).astype(int)
    feats["polyphen_probably_damaging"] = (feats["polyphen2_score"] >= 0.908).astype(int)
    feats["revel_pathogenic"]           = (feats["revel_score"]     >= 0.5).astype(int)
    feats["n_tools_pathogenic"] = (
        feats["cadd_high"] + feats["sift_deleterious"] +
        feats["polyphen_probably_damaging"] + feats["revel_pathogenic"]
    )

    # --- Gene-level (4 features) ---
    feats["gene_constraint_oe"]  = df.get("gene_constraint_oe",  pd.Series([1.0] * len(df), index=df.index)).fillna(1.0)
    feats["gene_is_constrained"] = (feats["gene_constraint_oe"] < 0.35).astype(int)
    feats["n_pathogenic_in_gene"]  = df.get("n_pathogenic_in_gene", pd.Series([0] * len(df), index=df.index)).fillna(0).astype(int)
    feats["gene_has_known_disease"] = (feats["n_pathogenic_in_gene"] > 0).astype(int)

    # --- Protein features (2 features) ---
    feats["has_uniprot_annotation"] = df.get("has_uniprot_annotation", pd.Series([0] * len(df), index=df.index)).fillna(0).astype(int)
    feats["n_known_pathogenic_protein_variants"] = df.get("n_known_pathogenic_protein_variants", pd.Series([0] * len(df), index=df.index)).fillna(0).astype(int)

    # --- GTEx expression / regulatory (6 features) ---
    gtex_defaults = {
        "gtex_max_tpm":             0.0,
        "gtex_n_tissues_expressed": 0,
        "gtex_tissue_specificity":  0.0,
        "gtex_is_eqtl":             0,
        "gtex_min_eqtl_pval":       0.0,
        "gtex_max_abs_effect":      0.0,
    }
    for col, default in gtex_defaults.items():
        feats[col] = df.get(col, pd.Series([default] * len(df), index=df.index)).fillna(default)
    for col in ["gtex_n_tissues_expressed", "gtex_is_eqtl"]:
        feats[col] = feats[col].astype(int)

    # --- Variant coding context (2 features) ---
    feats["codon_position"] = df.get("codon_position", pd.Series([0] * len(df), index=df.index)).fillna(0).astype(int)
    feats["dbsnp_af"] = df.get("dbsnp_af", pd.Series([0.0] * len(df), index=df.index)).fillna(0.0).astype(float).clip(lower=0)

    # --- Gene-disease annotation (3 features) ---
    feats["omim_n_diseases"]            = df.get("omim_n_diseases",            pd.Series([0] * len(df), index=df.index)).fillna(0).astype(int)
    feats["omim_is_autosomal_dominant"] = df.get("omim_is_autosomal_dominant", pd.Series([0] * len(df), index=df.index)).fillna(0).astype(int)
    feats["clingen_validity_score"]     = df.get("clingen_validity_score",     pd.Series([0] * len(df), index=df.index)).fillna(0).astype(int)

    # --- HGMD (2 features) ---
    feats["hgmd_is_disease_mutation"] = df.get("hgmd_is_disease_mutation", pd.Series([0] * len(df), index=df.index)).fillna(0).astype(int)
    feats["hgmd_n_reports"]           = df.get("hgmd_n_reports",           pd.Series([0] * len(df), index=df.index)).fillna(0).astype(int)

    # --- Chromosome context (3 features) ---
    chrom = df.get("chrom", pd.Series(["0"] * len(df), index=df.index)).fillna("0").astype(str)
    feats["is_autosome"]      = chrom.isin([str(i) for i in range(1, 23)]).astype(int)
    feats["is_sex_chrom"]     = chrom.isin(["X", "Y"]).astype(int)
    feats["is_mitochondrial"] = chrom.isin(["MT", "M"]).astype(int)

    # --- GNN-derived score (1 feature) — 0.5 = no GNN / ambiguous ---
    feats["gnn_score"] = (
        df.get("gnn_score", pd.Series([0.5] * len(df), index=df.index))
        .fillna(0.5)
        .astype(float)
        .clip(0.0, 1.0)
    )

    # --- RNA splice-context features (4 features) — Phase 6.1 ---
    # Defaults: 0.0 / 50 / 0 / 0 for non-splice variants
    feats["maxentscan_score"] = (
        df.get("maxentscan_score", pd.Series([0.0] * len(df), index=df.index))
        .fillna(0.0).astype(float)
    )
    feats["dist_to_splice_site"] = (
        df.get("dist_to_splice_site", pd.Series([50] * len(df), index=df.index))
        .fillna(50).astype(int)
    )
    feats["exon_number"] = (
        df.get("exon_number", pd.Series([0] * len(df), index=df.index))
        .fillna(0).astype(int)
    )
    feats["is_canonical_splice"] = (
        df.get("is_canonical_splice", pd.Series([0] * len(df), index=df.index))
        .fillna(0).astype(int)
    )

    # --- Protein structure features (4 features) — Phase 6.2 ---
    # Defaults: 50.0 / 0.5 / 0 / 100.0 for non-missense variants
    feats["alphafold_plddt"] = (
        df.get("alphafold_plddt", pd.Series([50.0] * len(df), index=df.index))
        .fillna(50.0).astype(float).clip(0.0, 100.0)
    )
    feats["solvent_accessibility"] = (
        df.get("solvent_accessibility", pd.Series([0.5] * len(df), index=df.index))
        .fillna(0.5).astype(float).clip(0.0, 1.0)
    )
    feats["secondary_structure_context"] = (
        df.get("secondary_structure_context", pd.Series([0] * len(df), index=df.index))
        .fillna(0).astype(int).clip(0, 2)
    )
    feats["dist_to_active_site"] = (
        df.get("dist_to_active_site", pd.Series([100.0] * len(df), index=df.index))
        .fillna(100.0).astype(float).clip(lower=0.0)
    )

    n_nan = feats.isnull().sum().sum()
    if n_nan > 0:
        logger.warning("%d NaN values in feature matrix — filling with 0.", n_nan)
        feats = feats.fillna(0.0)

    feats = feats[TABULAR_FEATURES]
    assert list(feats.columns) == TABULAR_FEATURES, (
        f"Feature column mismatch.\nExpected: {TABULAR_FEATURES}\nGot: {list(feats.columns)}"
    )
    return feats.reset_index(drop=True)


def encode_sequence(seq: str, window: int = 101) -> np.ndarray:
    """
    One-hot encode a nucleotide sequence for CNN input.

    Args:
        seq:    Raw nucleotide string (any length; will be padded/trimmed to `window`).
        window: Context window size in base pairs. Default 101 bp.

    Returns:
        np.ndarray of shape (window, 4), dtype float32.
        Axis 1 encodes [A, C, G, T]. Ambiguous bases (N) are encoded as all zeros.

    CHANGE: Removed unused `k` k-mer parameter. The function only performs
    one-hot encoding. k-mer frequency encoding will be added in Phase 2
    as a separate `kmer_encode_sequence()` function. (Bug 7)
    """
    BASES = "ACGT"
    base_map = {b: i for i, b in enumerate(BASES)}

    # Normalise, pad/trim to window
    seq = seq.upper()[:window].ljust(window, "A")

    one_hot = np.zeros((window, len(BASES)), dtype=np.float32)
    for i, nuc in enumerate(seq):
        if nuc in base_map:
            one_hot[i, base_map[nuc]] = 1.0
    return one_hot


# ---------------------------------------------------------------------------
# Sklearn-compatible 1D-CNN wrapper (sequence input)
# ---------------------------------------------------------------------------
class CNN1DClassifier(BaseEstimator, ClassifierMixin):
    """
    1D Convolutional neural network for variant sequence classification.
    Wraps Keras for sklearn cross_val_predict compatibility.
    Input: pd.Series of nucleotide strings (length ≥ window bp).
    """

    def __init__(
        self,
        window: int = 101,
        filters: int = 64,
        kernel_size: int = 7,
        dropout: float = 0.3,
        epochs: int = 30,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        random_state: int = 42,
    ) -> None:
        self.window        = window
        self.filters       = filters
        self.kernel_size   = kernel_size
        self.dropout       = dropout
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.learning_rate = learning_rate
        self.random_state  = random_state
        self.model_: Optional[object] = None
        self.classes_      = np.array([0, 1])

    def _build_model(self):
        import tensorflow as tf
        from tensorflow.keras import layers, models as km

        tf.random.set_seed(self.random_state)
        inp = layers.Input(shape=(self.window, 4))
        x   = layers.Conv1D(self.filters,     self.kernel_size, activation="relu", padding="same")(inp)
        x   = layers.MaxPooling1D(2)(x)
        x   = layers.Conv1D(self.filters * 2, self.kernel_size, activation="relu", padding="same")(x)
        x   = layers.GlobalMaxPooling1D()(x)
        x   = layers.Dense(128, activation="relu")(x)
        x   = layers.Dropout(self.dropout)(x)
        out = layers.Dense(1, activation="sigmoid")(x)
        model = km.Model(inp, out)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            loss="binary_crossentropy",
            metrics=["AUC"],
        )
        return model

    def _encode_X(self, X) -> np.ndarray:
        if isinstance(X, pd.Series):
            seqs = X.fillna("A" * self.window)
        elif isinstance(X, pd.DataFrame) and "fasta_seq" in X.columns:
            seqs = X["fasta_seq"].fillna("A" * self.window)
        else:
            seqs = pd.Series(X).fillna("A" * self.window)
        return np.stack([encode_sequence(s, window=self.window) for s in seqs])

    def fit(self, X, y):
        import tensorflow as tf
        self.model_ = self._build_model()
        self.model_.fit(
            self._encode_X(X), y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
            ],
            verbose=0,
        )
        return self

    def predict_proba(self, X) -> np.ndarray:
        proba = self.model_.predict(self._encode_X(X), verbose=0).flatten()
        return np.column_stack([1 - proba, proba])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


# ---------------------------------------------------------------------------
# Sklearn-compatible feedforward NN (tabular input)
# ---------------------------------------------------------------------------
class TabularNNClassifier(BaseEstimator, ClassifierMixin):
    """Feedforward neural network for tabular variant features."""

    def __init__(
        self,
        hidden_dims: tuple = (256, 128, 64),
        dropout: float = 0.3,
        epochs: int = 50,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        random_state: int = 42,
    ) -> None:
        self.hidden_dims   = hidden_dims
        self.dropout       = dropout
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.learning_rate = learning_rate
        self.random_state  = random_state
        self.model_: Optional[object]  = None
        self.scaler_       = StandardScaler()
        self.classes_      = np.array([0, 1])

    def _build_model(self, input_dim: int):
        import tensorflow as tf
        from tensorflow.keras import layers, models as km, regularizers

        tf.random.set_seed(self.random_state)
        inp = layers.Input(shape=(input_dim,))
        x   = inp
        for dim in self.hidden_dims:
            x = layers.Dense(dim, activation="relu",
                             kernel_regularizer=regularizers.l2(1e-4))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(self.dropout)(x)
        out = layers.Dense(1, activation="sigmoid")(x)
        model = km.Model(inp, out)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            loss="binary_crossentropy",
            metrics=["AUC"],
        )
        return model

    def fit(self, X, y):
        import tensorflow as tf
        X_scaled = self.scaler_.fit_transform(X)
        self.model_ = self._build_model(X_scaled.shape[1])
        self.model_.fit(
            X_scaled, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)
            ],
            verbose=0,
        )
        return self

    def predict_proba(self, X) -> np.ndarray:
        proba = self.model_.predict(self.scaler_.transform(X), verbose=0).flatten()
        return np.column_stack([1 - proba, proba])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


# ---------------------------------------------------------------------------
# Ensemble orchestrator
# ---------------------------------------------------------------------------
class VariantEnsemble:
    """
    Orchestrates training and evaluation of all base classifiers
    plus a stacking meta-learner (Logistic Regression on OOF predictions).

    Usage:
        ensemble = VariantEnsemble()
        ensemble.fit(X_tab_train, X_seq_train, y_train)
        results  = ensemble.evaluate(X_tab_test, X_seq_test, y_test)
        ensemble.save(Path("models/v1"))
    """

    def __init__(self, config: Optional[EnsembleConfig] = None) -> None:
        self.config = config or EnsembleConfig()
        self._build_estimators()

    def _build_estimators(self) -> None:
        cfg = self.config
        self.base_estimators: dict = {
            "random_forest": RandomForestClassifier(
                n_estimators=500, max_features="sqrt",
                class_weight=cfg.class_weight,
                n_jobs=cfg.n_jobs, random_state=cfg.random_state,
            ),
            "xgboost": xgb.XGBClassifier(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=10,
                eval_metric="auc",
                n_jobs=cfg.n_jobs, random_state=cfg.random_state, verbosity=0,
            ),
            "lightgbm": lgb.LGBMClassifier(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                class_weight=cfg.class_weight,
                n_jobs=cfg.n_jobs, random_state=cfg.random_state, verbose=-1,
            ),
            "svm": CalibratedClassifierCV(
                SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced"),
                cv=3,
            ),
            "logistic_regression": LogisticRegression(
                C=0.1, max_iter=1000, class_weight=cfg.class_weight,
                n_jobs=cfg.n_jobs, random_state=cfg.random_state,
            ),
            "gradient_boosting": GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=cfg.random_state,
            ),
            "tabular_nn": TabularNNClassifier(random_state=cfg.random_state),
            "cnn_1d":     CNN1DClassifier(random_state=cfg.random_state),
        }
        self.meta_learner = LogisticRegression(
            C=0.1, max_iter=1000, random_state=cfg.random_state
        )
        # Populated during fit(); kept separate from base_estimators
        self.trained_models_: dict = {}

    def fit(
        self,
        X_tab: pd.DataFrame,
        X_seq: pd.Series,
        y: pd.Series,
    ) -> "VariantEnsemble":
        """
        Two-phase training:
          Phase 1 — cross-val all base models → OOF predictions
          Phase 2 — train meta-learner on OOF predictions

        CHANGE: base_estimators is cleared after fitting; trained_models_
        is the single source of truth for fitted models. (Issue H)
        """
        y_arr = np.asarray(y)
        logger.info(
            "Training ensemble: %d samples, %d pathogenic.",
            len(y_arr), int(y_arr.sum()),
        )
        cv = StratifiedKFold(
            n_splits=self.config.n_folds, shuffle=True,
            random_state=self.config.random_state,
        )

        oof_preds = np.zeros((len(y_arr), len(self.base_estimators)))

        for model_idx, (name, model) in enumerate(self.base_estimators.items()):
            logger.info("  Training %s ...", name)
            X_input = X_seq if name == "cnn_1d" else X_tab.values

            try:
                oof = cross_val_predict(
                    model, X_input, y_arr,
                    cv=cv, method="predict_proba", n_jobs=1,
                )[:, 1]
            except Exception as exc:
                logger.error("  %s OOF failed: %s — skipping.", name, exc)
                oof_preds[:, model_idx] = 0.5
                continue

            oof_preds[:, model_idx] = oof
            model.fit(X_input, y_arr)
            self.trained_models_[name] = model
            logger.info("  %s OOF AUROC: %.4f", name, roc_auc_score(y_arr, oof))

        # Remove columns for skipped models
        valid_cols = [
            i for i, name in enumerate(self.base_estimators)
            if name in self.trained_models_
        ]
        oof_preds = oof_preds[:, valid_cols]

        logger.info("Training meta-learner on %d base-model OOF columns ...", len(valid_cols))
        self.meta_learner.fit(oof_preds, y_arr)
        self.feature_names_ = list(self.trained_models_.keys())

        # Free memory from unfitted duplicates (Issue H)
        self.base_estimators.clear()
        return self

    def predict_proba(
        self,
        X_tab: pd.DataFrame,
        X_seq: pd.Series,
    ) -> np.ndarray:
        if not self.trained_models_:
            raise RuntimeError("Call fit() before predict_proba().")
        base_preds = np.zeros((len(X_tab), len(self.trained_models_)))
        for i, (name, model) in enumerate(self.trained_models_.items()):
            X_input = X_seq if name == "cnn_1d" else X_tab.values
            base_preds[:, i] = model.predict_proba(X_input)[:, 1]
        return self.meta_learner.predict_proba(base_preds)

    def predict(self, X_tab: pd.DataFrame, X_seq: pd.Series) -> np.ndarray:
        return (self.predict_proba(X_tab, X_seq)[:, 1] > 0.5).astype(int)

    def evaluate(
        self,
        X_tab: pd.DataFrame,
        X_seq: pd.Series,
        y: pd.Series,
    ) -> pd.DataFrame:
        """
        Full per-model + ensemble evaluation.
        Returns a DataFrame sorted by AUROC (descending).
        """
        y_arr = np.asarray(y)
        results: dict[str, dict] = {}

        for name, model in self.trained_models_.items():
            X_input = X_seq if name == "cnn_1d" else X_tab.values
            proba   = model.predict_proba(X_input)[:, 1]
            preds   = (proba > 0.5).astype(int)
            results[name] = {
                "auroc":       roc_auc_score(y_arr, proba),
                "auprc":       average_precision_score(y_arr, proba),
                "f1_macro":    f1_score(y_arr, preds, average="macro",    zero_division=0),
                "f1_weighted": f1_score(y_arr, preds, average="weighted", zero_division=0),
                "mcc":         matthews_corrcoef(y_arr, preds),
                "brier":       brier_score_loss(y_arr, proba),
            }

        ens_proba = self.predict_proba(X_tab, X_seq)[:, 1]
        ens_preds = (ens_proba > 0.5).astype(int)
        results["ENSEMBLE_STACKER"] = {
            "auroc":       roc_auc_score(y_arr, ens_proba),
            "auprc":       average_precision_score(y_arr, ens_proba),
            "f1_macro":    f1_score(y_arr, ens_preds, average="macro",    zero_division=0),
            "f1_weighted": f1_score(y_arr, ens_preds, average="weighted", zero_division=0),
            "mcc":         matthews_corrcoef(y_arr, ens_preds),
            "brier":       brier_score_loss(y_arr, ens_proba),
        }

        df = pd.DataFrame(results).T.round(4)
        df = df.sort_values("auroc", ascending=False)
        logger.info("\n%s", df.to_string())
        return df

    def save(self, path: Optional[Path] = None) -> None:
        import joblib
        path = Path(path or self.config.model_dir / "ensemble.joblib")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Ensemble saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "VariantEnsemble":
        import joblib
        return joblib.load(path)
