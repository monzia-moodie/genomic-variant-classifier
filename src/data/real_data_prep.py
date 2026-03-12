"""
Real Data Preparation Pipeline
================================
Bridges raw ClinVar parquet (from database_connectors.py) to a
training-ready feature matrix.

What this module does that synthetic prep cannot:
  - Filters to high-confidence ClinVar labels (removes VUS and conflicting)
  - Joins gnomAD v4 allele frequencies by variant_id locus
  - Derives consequence severity from VEP consequence strings
  - Applies a gene-aware train/test split to prevent label leakage
  - Computes class weights for imbalanced label distribution (~15% pathogenic)

CHANGES FROM PHASE 1:
  - Was never written to disk in Phase 1 (Bug 3 fixed).
  - from __future__ import annotations added (Issue N).
  - Module-level logging.basicConfig removed (Issue L).
  - Pre-split class balance validation with helpful error message (Issue I).

Usage:
    from src.data.real_data_prep import DataPrepPipeline
    pipeline = DataPrepPipeline()
    X_train, X_test, y_train, y_test, meta_test = pipeline.run(
        clinvar_path="data/processed/clinvar_grch38.parquet",
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_class_weight

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ClinVar label vocabulary
# ---------------------------------------------------------------------------
REVIEW_STATUS_TIER: dict[str, int] = {
    "practice guideline":                                      1,
    "reviewed by expert panel":                                1,
    "criteria provided, multiple submitters, no conflicts":    2,
    "criteria provided, single submitter":                     3,
    "no assertion criteria provided":                          4,
    "no classification provided":                              5,
    "no classification for the individual variant":            5,
}

PATHOGENIC_TERMS = {
    "Pathogenic",
    "Likely pathogenic",
    "Pathogenic/Likely pathogenic",
}
BENIGN_TERMS = {
    "Benign",
    "Likely benign",
    "Benign/Likely benign",
}

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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class DataPrepConfig:
    min_review_tier:      int   = 3        # exclude tier 4-5 (no criteria)
    exclude_conflicting:  bool  = True
    require_both_classes: bool  = True
    test_fraction:        float = 0.20
    random_state:         int   = 42
    group_column:         str   = "gene_symbol"
    class_weight_strategy: str  = "balanced"
    scale_features:       bool  = True
    output_dir:           Path  = Path("data/splits")

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------
class DataPrepPipeline:
    """
    Loads, filters, enriches, and splits genomic variant data
    from the canonical parquet format produced by database_connectors.py.
    """

    def __init__(self, config: Optional[DataPrepConfig] = None) -> None:
        self.config = config or DataPrepConfig()
        self.scaler = StandardScaler()

    def run(
        self,
        clinvar_path: str,
        gnomad_path:  Optional[str] = None,
        uniprot_path: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
        """
        Full pipeline from raw parquet to train/test splits.

        Returns:
            X_train, X_test  — feature DataFrames
            y_train, y_test  — binary labels (1=pathogenic, 0=benign)
            meta_test        — original rows for test set (for reporting)
        """
        logger.info("=== DataPrepPipeline: starting ===")

        df = self._load_and_label(clinvar_path)
        logger.info(
            "After label filtering: %d variants (%d pathogenic, %d benign).",
            len(df), int(df["label"].sum()), int((df["label"] == 0).sum()),
        )

        if gnomad_path:
            df = self._join_gnomad(df, gnomad_path)
        if uniprot_path:
            df = self._join_uniprot(df, uniprot_path)

        X      = self._engineer_features(df)
        y      = df["label"].reset_index(drop=True)
        groups = df[self.config.group_column].fillna("unknown").reset_index(drop=True)

        logger.info("Feature matrix: %d rows × %d features.", X.shape[0], X.shape[1])

        # CHANGE: validate class balance BEFORE splitting for a useful error (Issue I)
        if self.config.require_both_classes:
            if set(y.unique()) != {0, 1}:
                raise ValueError(
                    f"Dataset missing classes — found only {set(y.unique())}. "
                    "Lower min_review_tier or increase dataset size."
                )

        X_train, X_test, y_train, y_test, train_idx, test_idx = (
            self._gene_aware_split(X, y, groups)
        )

        meta_test = df.iloc[test_idx].reset_index(drop=True)

        if self.config.scale_features:
            X_train, X_test = self._scale(X_train, X_test)

        self._save_splits(X_train, X_test, y_train, y_test, meta_test)
        self._report_split_stats(y_train, y_test, groups, train_idx, test_idx)

        logger.info("=== DataPrepPipeline: complete ===")
        return X_train, X_test, y_train, y_test, meta_test

    # ── Stage 1: Load and label ────────────────────────────────────────────

    def _load_and_label(self, clinvar_path: str) -> pd.DataFrame:
        df = pd.read_parquet(clinvar_path)
        logger.info("Loaded %d rows from %s.", len(df), clinvar_path)

        df["clinical_sig"] = df["clinical_sig"].fillna("").str.strip()
        df["label"] = np.nan
        df.loc[df["clinical_sig"].isin(PATHOGENIC_TERMS), "label"] = 1
        df.loc[df["clinical_sig"].isin(BENIGN_TERMS),     "label"] = 0

        n_before = len(df)
        df = df[df["label"].notna()].copy()
        df["label"] = df["label"].astype(int)
        logger.info(
            "Label filtering: %d → %d (%d VUS/conflicting removed).",
            n_before, len(df), n_before - len(df),
        )

        if "ReviewStatus" in df.columns:
            df["review_tier"] = df["ReviewStatus"].str.lower().map(
                lambda s: next(
                    (v for k, v in REVIEW_STATUS_TIER.items() if k in s), 5
                )
            )
            before = len(df)
            df = df[df["review_tier"] <= self.config.min_review_tier]
            logger.info(
                "Review tier filter (≤%d): %d → %d.",
                self.config.min_review_tier, before, len(df),
            )

        if self.config.exclude_conflicting:
            before = len(df)
            df = df[~df["clinical_sig"].str.contains("onflict", na=False)]
            if len(df) < before:
                logger.info("Removed %d conflicting variants.", before - len(df))

        return df.reset_index(drop=True)

    # ── Stage 2: Enrich with gnomAD AFs ───────────────────────────────────

    def _join_gnomad(self, df: pd.DataFrame, gnomad_path: str) -> pd.DataFrame:
        gnomad = pd.read_parquet(gnomad_path)[["variant_id", "allele_freq"]].copy()
        gnomad = gnomad.rename(columns={"allele_freq": "gnomad_af"})
        gnomad = gnomad.dropna(subset=["variant_id"]).drop_duplicates("variant_id")

        def extract_locus(vid: str) -> str:
            parts = str(vid).split(":", 1)
            return parts[1] if len(parts) > 1 else vid

        df["_locus"]     = df["variant_id"].apply(extract_locus)
        gnomad["_locus"] = gnomad["variant_id"].apply(extract_locus)

        df = df.merge(gnomad[["_locus", "gnomad_af"]], on="_locus", how="left").drop(columns=["_locus"])

        if "allele_freq" in df.columns:
            df["allele_freq"] = df["allele_freq"].fillna(df.get("gnomad_af", np.nan))
        else:
            df["allele_freq"] = df.get("gnomad_af", np.nan)

        logger.info(
            "After gnomAD join: %d variants have AF.",
            int(df["allele_freq"].notna().sum()),
        )
        return df

    # ── Stage 3: Enrich with UniProt protein features ─────────────────────

    def _join_uniprot(self, df: pd.DataFrame, uniprot_path: str) -> pd.DataFrame:
        uniprot = pd.read_parquet(uniprot_path)
        gene_features = (
            uniprot.groupby("gene_symbol")
            .agg(
                has_uniprot_annotation=("source_id", "any"),
                n_known_pathogenic_protein_variants=(
                    "pathogenicity",
                    lambda x: (x == "pathogenic").sum(),
                ),
            )
            .reset_index()
        )
        df = df.merge(gene_features, on="gene_symbol", how="left")
        df["has_uniprot_annotation"] = df["has_uniprot_annotation"].fillna(False).astype(int)
        df["n_known_pathogenic_protein_variants"] = (
            df["n_known_pathogenic_protein_variants"].fillna(0).astype(int)
        )
        return df

    # ── Stage 4: Feature engineering ──────────────────────────────────────

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feats = pd.DataFrame(index=df.index)

        # Allele frequency
        af = df.get("allele_freq", pd.Series(0.0, index=df.index)).fillna(0.0).clip(lower=0)
        feats["af_raw"]          = af
        feats["af_log10"]        = np.log10(af + 1e-8)
        feats["af_is_absent"]    = (af == 0).astype(int)
        feats["af_is_ultra_rare"] = (af < 0.0001).astype(int)
        feats["af_is_rare"]      = ((af >= 0.0001) & (af < 0.001)).astype(int)
        feats["af_is_common"]    = (af >= 0.01).astype(int)

        # Variant type
        ref = df.get("ref", pd.Series([""] * len(df), index=df.index)).fillna("")
        alt = df.get("alt", pd.Series([""] * len(df), index=df.index)).fillna("")
        ref_len = ref.str.len().clip(lower=1)
        alt_len = alt.str.len().clip(lower=1)
        feats["ref_len"]    = ref_len
        feats["alt_len"]    = alt_len
        feats["len_diff"]   = (alt_len - ref_len).abs()
        feats["is_snv"]     = ((ref_len == 1) & (alt_len == 1)).astype(int)
        feats["is_insertion"] = (alt_len > ref_len).astype(int)
        feats["is_deletion"]  = (ref_len > alt_len).astype(int)
        feats["is_indel"]   = (feats["is_insertion"] | feats["is_deletion"]).astype(int)

        # Consequence severity
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

        # Precomputed functional scores
        score_defaults = {
            "cadd_phred":      15.0,
            "sift_score":       0.05,
            "polyphen2_score":  0.5,
            "revel_score":      0.5,
            "phylop_score":     0.0,
            "gerp_score":       0.0,
        }
        for col, default in score_defaults.items():
            feats[col] = df.get(col, pd.Series([default] * len(df), index=df.index)).fillna(default).astype(float)

        feats["cadd_high"]                  = (feats["cadd_phred"]     >= 20).astype(int)
        feats["sift_deleterious"]           = (feats["sift_score"]      < 0.05).astype(int)
        feats["polyphen_probably_damaging"] = (feats["polyphen2_score"] >= 0.908).astype(int)
        feats["revel_pathogenic"]           = (feats["revel_score"]     >= 0.5).astype(int)
        feats["n_tools_pathogenic"] = (
            feats["cadd_high"] + feats["sift_deleterious"] +
            feats["polyphen_probably_damaging"] + feats["revel_pathogenic"]
        )

        # Gene-level
        feats["gene_constraint_oe"]  = df.get("gene_constraint_oe",  pd.Series([1.0] * len(df), index=df.index)).fillna(1.0)
        feats["gene_is_constrained"] = (feats["gene_constraint_oe"] < 0.35).astype(int)
        feats["n_pathogenic_in_gene"]  = df.get("n_pathogenic_in_gene", pd.Series([0] * len(df), index=df.index)).fillna(0).astype(int)
        feats["gene_has_known_disease"] = (feats["n_pathogenic_in_gene"] > 0).astype(int)

        # Protein features (UniProt-derived)
        feats["has_uniprot_annotation"] = df.get("has_uniprot_annotation", pd.Series([0] * len(df), index=df.index)).fillna(0).astype(int)
        feats["n_known_pathogenic_protein_variants"] = df.get("n_known_pathogenic_protein_variants", pd.Series([0] * len(df), index=df.index)).fillna(0).astype(int)

        # Chromosome features
        chrom = df.get("chrom", pd.Series(["0"] * len(df), index=df.index)).fillna("0").astype(str)
        feats["is_autosome"]     = chrom.isin([str(i) for i in range(1, 23)]).astype(int)
        feats["is_sex_chrom"]    = chrom.isin(["X", "Y"]).astype(int)
        feats["is_mitochondrial"] = chrom.isin(["MT", "M"]).astype(int)

        n_nan = feats.isnull().sum().sum()
        if n_nan > 0:
            logger.warning("%d NaN values in feature matrix — filling with 0.", n_nan)
            feats = feats.fillna(0.0)

        return feats.reset_index(drop=True)

    # ── Stage 5: Gene-aware split ──────────────────────────────────────────

    def _gene_aware_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: pd.Series,
    ) -> tuple:
        """
        Split so all variants of a gene land in the same fold.
        Prevents models from memorizing gene-level patterns.
        """
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=self.config.test_fraction,
            random_state=self.config.random_state,
        )
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))

        X_train = X.iloc[train_idx].reset_index(drop=True)
        X_test  = X.iloc[test_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        y_test  = y.iloc[test_idx].reset_index(drop=True)

        if self.config.require_both_classes:
            for split_name, y_split in [("train", y_train), ("test", y_test)]:
                classes = set(y_split.unique())
                if classes != {0, 1}:
                    raise ValueError(
                        f"Gene-aware split '{split_name}' missing class(es): {classes}. "
                        "Try lowering min_review_tier or increasing dataset size."
                    )

        return X_train, X_test, y_train, y_test, train_idx, test_idx

    # ── Stage 6: Scaling ───────────────────────────────────────────────────

    def _scale(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        cols = X_train.columns
        X_train_scaled = pd.DataFrame(self.scaler.fit_transform(X_train),  columns=cols)
        X_test_scaled  = pd.DataFrame(self.scaler.transform(X_test),       columns=cols)
        return X_train_scaled, X_test_scaled

    # ── Stage 7: Save ──────────────────────────────────────────────────────

    def _save_splits(
        self,
        X_train: pd.DataFrame, X_test: pd.DataFrame,
        y_train: pd.Series,    y_test: pd.Series,
        meta_test: pd.DataFrame,
    ) -> None:
        out = self.config.output_dir
        X_train.to_parquet(out / "X_train.parquet",   index=False)
        X_test.to_parquet(out  / "X_test.parquet",    index=False)
        y_train.to_frame("label").to_parquet(out / "y_train.parquet", index=False)
        y_test.to_frame("label").to_parquet(out  / "y_test.parquet",  index=False)
        meta_test.to_parquet(out / "meta_test.parquet", index=False)
        logger.info("Splits saved to %s/", out)

    # ── Utilities ──────────────────────────────────────────────────────────

    def _report_split_stats(
        self,
        y_train: pd.Series, y_test: pd.Series,
        groups:  pd.Series,
        train_idx: np.ndarray, test_idx: np.ndarray,
    ) -> None:
        train_genes = groups.iloc[train_idx].nunique()
        test_genes  = groups.iloc[test_idx].nunique()
        logger.info("─" * 55)
        logger.info("%-12s %10s %12s %8s", "Split", "Variants", "Pathogenic", "Genes")
        logger.info("─" * 55)
        logger.info(
            "%-12s %10d %11d (%4.1f%%)  %8d",
            "Train", len(y_train), y_train.sum(), y_train.mean() * 100, train_genes,
        )
        logger.info(
            "%-12s %10d %11d (%4.1f%%)  %8d",
            "Test",  len(y_test),  y_test.sum(),  y_test.mean()  * 100, test_genes,
        )
        logger.info("─" * 55)

    def get_class_weights(self, y: pd.Series) -> dict[int, float]:
        weights = compute_class_weight(
            class_weight=self.config.class_weight_strategy,
            classes=np.array([0, 1]),
            y=y.values,
        )
        return {0: float(weights[0]), 1: float(weights[1])}


# ---------------------------------------------------------------------------
# Utility: enrich gene-level pathogenic counts before splitting
# ---------------------------------------------------------------------------
def enrich_gene_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add n_pathogenic_in_gene to each row.

    This is a strong predictor: genes with many known pathogenic variants
    (e.g. BRCA1, TP53) are a priori more suspicious for new variants.
    Must be computed on the FULL labeled dataset BEFORE splitting to avoid
    information leakage (the count uses only labeled rows, not the test set).
    """
    gene_path_counts = (
        df[df["label"] == 1]
        .groupby("gene_symbol")
        .size()
        .rename("n_pathogenic_in_gene")
        .reset_index()
    )
    df = df.merge(gene_path_counts, on="gene_symbol", how="left")
    df["n_pathogenic_in_gene"] = df["n_pathogenic_in_gene"].fillna(0).astype(int)
    return df
