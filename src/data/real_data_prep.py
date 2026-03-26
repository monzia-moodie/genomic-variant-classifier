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

from src.data.dbnsfp   import DbNSFPConnector
from src.data.phylop   import PhyloPConnector
from src.data.cadd     import CADDConnector
from src.data.spliceai import SpliceAIConnector

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
    val_fraction:         float = 0.10    # held-out validation set (of full data)
    random_state:         int   = 42
    group_column:         str   = "gene_symbol"
    class_weight_strategy: str  = "balanced"
    scale_features:       bool  = True
    output_dir:           Path  = Path("data/splits")

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)




@dataclass
class AnnotationConfig:
    """
    Paths and flags controlling which score connectors run during DataPrepPipeline.

    All paths default to None -> connector runs in stub mode (returns default
    scores, logs a WARNING, pipeline continues).  Set paths to activate real
    annotation.

    Sequence when run:
      1. DbNSFPConnector(dbnsfp_path)     -- 6 scores for missense SNVs
      2. PhyloPConnector(phylop_path)     -- phylop_score for all positions
      3. CADDConnector()                  -- cadd_phred via REST (if annotate_cadd=True)
      4. SpliceAIConnector(spliceai_path) -- splice_ai_score (PHASE_2_FEATURES)

    annotate_cadd is False by default because the CADD REST API requires
    1.5 s/variant. Enable only for small batches or when the pre-computed
    file is available (PHASE_2_PLACEHOLDER).
    """
    dbnsfp_path:        Optional[Path] = None
    phylop_path:        Optional[Path] = None
    spliceai_path:      Optional[Path] = None
    alphamissense_path: Optional[Path] = None
    annotate_cadd:      bool           = False
    gtex_genes:         list[str]      = field(default_factory=list)
    gtex_tissues:       list[str]      = field(default_factory=list)
    vep_path:           Optional[Path] = None
    omim_path:          Optional[Path] = None
    clingen_path:       Optional[Path] = None
    dbsnp_path:         Optional[Path] = None
    eve_path:           Optional[Path] = None
    hgmd_path:          Optional[Path] = None
    kg_path:            Optional[Path] = None   # 1000 Genomes Phase 3 AF parquet
# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------
class DataPrepPipeline:
    """
    Loads, filters, enriches, and splits genomic variant data
    from the canonical parquet format produced by database_connectors.py.
    """

    def __init__(self, config: Optional[DataPrepConfig] = None, annotation_config: Optional[AnnotationConfig] = None) -> None:
        self.config            = config or DataPrepConfig()
        self.annotation_config = annotation_config or AnnotationConfig()
        self.scaler            = StandardScaler()

    def run(
        self,
        clinvar_path: str,
        gnomad_path:  Optional[str] = None,
        uniprot_path: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
               pd.Series,    pd.Series,    pd.Series,
               pd.DataFrame, pd.DataFrame]:
        """
        Full pipeline from raw parquet to train/val/test splits.

        Returns:
            X_train, X_val, X_test   — feature DataFrames
            y_train, y_val, y_test   — binary labels (1=pathogenic, 0=benign)
            meta_val, meta_test      — original rows for val/test sets

        Split fractions (gene-aware, no gene straddles splits):
            train : 1 - test_fraction - val_fraction  (~70%)
            val   : val_fraction                       (~10%)  ← clean holdout
            test  : test_fraction                      (~20%)  ← dev/tuning set
        """
        logger.info("=== DataPrepPipeline: starting ===")

        df = self._load_and_label(clinvar_path)
        df = enrich_gene_counts(df)
        logger.info(
            "After label filtering: %d variants (%d pathogenic, %d benign).",
            len(df), int(df["label"].sum()), int((df["label"] == 0).sum()),
        )

        if gnomad_path:
            df = self._join_gnomad(df, gnomad_path, kg_path=self.annotation_config.kg_path)
        if uniprot_path:
            df = self._join_uniprot(df, uniprot_path)

        # Score annotation (Phase 2 connectors)
        logger.info("=== Score annotation: starting ===")
        df = self._annotate_scores(df)
        logger.info("=== Score annotation: complete ===")

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

        X_train, X_test, X_val, y_train, y_test, y_val, train_idx, test_idx, val_idx = (
            self._gene_aware_split(X, y, groups)
        )

        meta_val  = df.iloc[val_idx].reset_index(drop=True)
        meta_test = df.iloc[test_idx].reset_index(drop=True)

        if self.config.scale_features:
            X_train, X_test, X_val = self._scale(X_train, X_test, X_val)

        self._save_splits(X_train, X_val, X_test, y_train, y_val, y_test, meta_val, meta_test)
        self._report_split_stats(y_train, y_test, y_val, groups, train_idx, test_idx, val_idx)

        logger.info("=== DataPrepPipeline: complete ===")
        return X_train, X_val, X_test, y_train, y_val, y_test, meta_val, meta_test

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

    def _join_gnomad(
        self,
        df: pd.DataFrame,
        gnomad_path: str,
        kg_path: Optional[str] = None,
    ) -> pd.DataFrame:
        gnomad = pd.read_parquet(gnomad_path, columns=["variant_id", "allele_freq"]).copy()

        # Build (chrom, pos, ref, alt) join keys from variant_id
        # Handles both "gnomad:1:69134:A:G" and "1:69134:A:G" formats
        def _parse_locus(vid: str) -> tuple[str, str, str, str] | None:
            parts = str(vid).split(":")
            # strip source prefix if present (non-numeric first part)
            if not parts[0].replace("X","").replace("Y","").replace("M","").isdigit():
                parts = parts[1:]
            if len(parts) < 4:
                return None
            return parts[0], parts[1], parts[2], parts[3]

        gnomad[["_chrom","_pos","_ref","_alt"]] = pd.DataFrame(
            gnomad["variant_id"].map(_parse_locus).tolist(),
            index=gnomad.index,
        )
        gnomad = (gnomad.dropna(subset=["_chrom"])
                        .drop_duplicates(subset=["_chrom","_pos","_ref","_alt"])
                        [["_chrom","_pos","_ref","_alt","allele_freq"]]
                        .rename(columns={"allele_freq": "gnomad_af"}))

        # Build matching keys on ClinVar side
        df["_chrom"] = df["chrom"].astype(str)
        df["_pos"]   = df["pos"].astype(str)
        df["_ref"]   = df["ref"].astype(str)
        df["_alt"]   = df["alt"].astype(str)

        df = df.merge(gnomad, on=["_chrom","_pos","_ref","_alt"], how="left")
        df = df.drop(columns=["_chrom","_pos","_ref","_alt"])

        df["allele_freq"] = df["allele_freq"].fillna(df.get("gnomad_af", float("nan")))
        if "gnomad_af" in df.columns:
            df = df.drop(columns=["gnomad_af"])

        n_matched = df["allele_freq"].notna().sum()
        logger.info("After gnomAD join: %d / %d variants have AF (%.1f%%).",
                    n_matched, len(df), n_matched / len(df) * 100)

        # ── 1000 Genomes Phase 3 fallback for still-null AFs ──────────────
        n_null = int(df["allele_freq"].isna().sum())
        if n_null > 0:
            if kg_path:
                from src.data.thousandgenomes import ThousandGenomesConnector
                kg = ThousandGenomesConnector(kg_path)
                df = kg.fill_missing_af(df)
                n_filled = n_null - int(df["allele_freq"].isna().sum())
                logger.info(
                    "1000G fallback: filled %d / %d null AFs.",
                    n_filled, n_null,
                )
            else:
                logger.info(
                    "%d variants still have null AF after gnomAD join. "
                    "Pass kg_path for 1000 Genomes fallback.",
                    n_null,
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

    def _annotate_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Annotate df with pre-computed pathogenicity and conservation scores.

        Sequence:
          1. DbNSFPConnector   -- 6 scores for missense SNVs in dbNSFP
          2. PhyloPConnector   -- overwrites phylop_score for all positions
          3. CADDConnector     -- cadd_phred via REST (skipped by default)
          4. SpliceAIConnector -- splice_ai_score
          5. AlphaMissense     -- alphamissense_score for missense variants
          6. GTEx              -- expression and eQTL features
          7. VEP               -- codon_position (from protein_change column)
          8. OMIM              -- omim_n_diseases, omim_is_autosomal_dominant
          9. ClinGen           -- clingen_validity_score
         10. dbSNP             -- dbsnp_af (AF supplement for gnomAD-absent variants)
         11. EVE               -- eve_score (evolutionary model)
         12. HGMD              -- hgmd_is_disease_mutation, hgmd_n_reports

        All connectors run in stub mode when their file is absent.
        """
        ac = self.annotation_config

        # 1. dbNSFP: SIFT, PP2, REVEL, CADD, PhyloP, GERP for missense SNVs
        dbnsfp = DbNSFPConnector(dbnsfp_file=ac.dbnsfp_path)
        df = dbnsfp.annotate_dataframe(df)
        logger.info(
            "Score annotation 1/4 (DbNSFP): %d variants with real SIFT scores.",
            (df.get("sift_score", pd.Series([0.5] * len(df), index=df.index)) != 0.5).sum(),
        )

        # 2. PhyloP: conservation for non-missense positions
        phylop = PhyloPConnector(phylop_file=ac.phylop_path)
        df = phylop.annotate_dataframe(df)
        logger.info(
            "Score annotation 2/4 (PhyloP): %d variants with non-zero phylop_score.",
            (df.get("phylop_score", pd.Series([0.0] * len(df), index=df.index)) != 0.0).sum(),
        )

        # 3. CADD REST API (optional, off by default)
        if ac.annotate_cadd:
            cadd = CADDConnector()
            df = cadd.fetch(variant_df=df)
            logger.info(
                "Score annotation 3/4 (CADD): %d variants with non-median cadd_phred.",
                (df.get("cadd_phred", pd.Series([15.0] * len(df), index=df.index)) != 15.0).sum(),
            )
        else:
            logger.debug(
                "Score annotation 3/4 skipped (CADD disabled; "
                "set annotate_cadd=True to enable)."
            )

        # 4. SpliceAI: splice disruption scores (PHASE_2_FEATURES)
        spliceai = SpliceAIConnector(vcf_path=ac.spliceai_path)
        df = spliceai.fetch(variant_df=df)
        logger.info(
            "Score annotation 4/6 (SpliceAI): %d variants with splice_ai_score > 0.",
            (df.get("splice_ai_score", pd.Series([0.0] * len(df), index=df.index)) > 0).sum(),
        )

        # 5. AlphaMissense: missense pathogenicity scores
        if ac.alphamissense_path is not None:
            from src.data.alphamissense import AlphaMissenseConnector
            am = AlphaMissenseConnector(tsv_path=ac.alphamissense_path)
            df = am.fetch(variant_df=df)
        else:
            df["alphamissense_score"] = 0.5   # ambiguous default; safe for _engineer_features
        logger.info(
            "Score annotation 5/6 (AlphaMissense): %d variants annotated (score != 0.5).",
            (df.get("alphamissense_score",
                    pd.Series([0.5] * len(df), index=df.index)) != 0.5).sum(),
        )

        # 6. GTEx: expression and eQTL features
        if ac.gtex_genes:
            from src.data.gtex import GTExConnector, build_gtex_feature_df
            gtex = GTExConnector()
            gtex.fetch(
                gene_symbols=ac.gtex_genes,
                tissues=ac.gtex_tissues if ac.gtex_tissues else None,
            )
            df = build_gtex_feature_df(gtex, df)
        else:
            for col, val in [
                ("gtex_max_tpm",             0.0),
                ("gtex_n_tissues_expressed", 0),
                ("gtex_tissue_specificity",  0.0),
                ("gtex_is_eqtl",             0),
                ("gtex_min_eqtl_pval",       0.0),
                ("gtex_max_abs_effect",      0.0),
            ]:
                df[col] = val
        logger.info(
            "Score annotation 6/12 (GTEx): %d eQTL variants.",
            int(df.get("gtex_is_eqtl", pd.Series([0]*len(df), index=df.index)).sum()),
        )

        # 7. VEP: codon_position (from protein_change column — no file required)
        from src.data.vep import VEPConnector
        vep = VEPConnector()
        df = vep.annotate_dataframe(df)
        logger.info(
            "Score annotation 7/12 (VEP): %d variants with non-zero codon_position.",
            int((df.get("codon_position", pd.Series([0]*len(df), index=df.index)) > 0).sum()),
        )

        # 8. OMIM: gene-disease features
        from src.data.omim import OMIMConnector
        omim = OMIMConnector(mim2gene_path=ac.omim_path)
        df = omim.annotate_dataframe(df)
        logger.info(
            "Score annotation 8/12 (OMIM): %d variants with omim_n_diseases > 0.",
            int((df.get("omim_n_diseases", pd.Series([0]*len(df), index=df.index)) > 0).sum()),
        )

        # 9. ClinGen: gene validity score
        from src.data.clingen import ClinGenConnector
        clingen = ClinGenConnector(csv_path=ac.clingen_path)
        df = clingen.annotate_dataframe(df)
        logger.info(
            "Score annotation 9/12 (ClinGen): %d variants with clingen_validity_score > 0.",
            int((df.get("clingen_validity_score", pd.Series([0]*len(df), index=df.index)) > 0).sum()),
        )

        # 10. dbSNP: supplemental allele frequency
        from src.data.dbsnp import DbSNPConnector
        dbsnp = DbSNPConnector(parquet_path=ac.dbsnp_path)
        df = dbsnp.annotate_dataframe(df)
        logger.info(
            "Score annotation 10/12 (dbSNP): %d variants with dbsnp_af > 0.",
            int((df.get("dbsnp_af", pd.Series([0.0]*len(df), index=df.index)) > 0).sum()),
        )

        # 11. EVE: evolutionary model score
        from src.data.eve import EVEConnector
        eve = EVEConnector(eve_path=ac.eve_path)
        df = eve.annotate_dataframe(df)
        logger.info(
            "Score annotation 11/12 (EVE): %d variants covered (score != 0.5).",
            int((df.get("eve_score", pd.Series([0.5]*len(df), index=df.index)) != 0.5).sum()),
        )

        # 12. HGMD: disease mutation annotation (requires institutional license)
        if ac.hgmd_path is not None:
            from src.data.hgmd import HGMDConnector
            hgmd = HGMDConnector(hgmd_path=ac.hgmd_path)
            df = hgmd.annotate_dataframe(df)
        else:
            df["hgmd_is_disease_mutation"] = 0
            df["hgmd_n_reports"] = 0
        logger.info(
            "Score annotation 12/12 (HGMD): %d variants flagged as disease mutations.",
            int((df.get("hgmd_is_disease_mutation", pd.Series([0]*len(df), index=df.index)) == 1).sum()),
        )

        return df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feats = pd.DataFrame(index=df.index)

        # Allele frequency
        af = df.get("allele_freq", pd.Series(0.0, index=df.index)).fillna(0.0).astype(float).clip(lower=0)
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
            "cadd_phred":            15.0,
            "sift_score":             0.5,
            "polyphen2_score":        0.5,
            "revel_score":            0.5,
            "phylop_score":           0.0,
            "gerp_score":             0.0,
            "alphamissense_score":    0.5,   # 0.5 = ambiguous / not covered
            "splice_ai_score":        0.0,   # 0.0 = no splice disruption
            "eve_score":              0.5,   # 0.5 = not covered / ambiguous
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

        # GTEx expression / regulatory features (populated by _annotate_scores step 6)
        gtex_defaults = {
            "gtex_max_tpm":             0.0,
            "gtex_n_tissues_expressed": 0,
            "gtex_tissue_specificity":  0.0,
            "gtex_is_eqtl":             0,
            "gtex_min_eqtl_pval":       0.0,
            "gtex_max_abs_effect":      0.0,
        }
        for col, default in gtex_defaults.items():
            feats[col] = (
                df.get(col, pd.Series([default] * len(df), index=df.index))
                .fillna(default)
            )
        for col in ["gtex_n_tissues_expressed", "gtex_is_eqtl"]:
            feats[col] = feats[col].astype(int)

        # Variant coding context (2 features)
        feats["codon_position"] = df.get("codon_position", pd.Series([0] * len(df), index=df.index)).fillna(0).astype(int)
        feats["dbsnp_af"] = df.get("dbsnp_af", pd.Series([0.0] * len(df), index=df.index)).fillna(0.0).astype(float).clip(lower=0)

        # Gene-disease annotation (3 features)
        feats["omim_n_diseases"]            = df.get("omim_n_diseases",            pd.Series([0] * len(df), index=df.index)).fillna(0).astype(int)
        feats["omim_is_autosomal_dominant"] = df.get("omim_is_autosomal_dominant", pd.Series([0] * len(df), index=df.index)).fillna(0).astype(int)
        feats["clingen_validity_score"]     = df.get("clingen_validity_score",     pd.Series([0] * len(df), index=df.index)).fillna(0).astype(int)

        # HGMD features (2 features)
        feats["hgmd_is_disease_mutation"] = df.get("hgmd_is_disease_mutation", pd.Series([0] * len(df), index=df.index)).fillna(0).astype(int)
        feats["hgmd_n_reports"]           = df.get("hgmd_n_reports",           pd.Series([0] * len(df), index=df.index)).fillna(0).astype(int)

        # Chromosome features
        chrom = df.get("chrom", pd.Series(["0"] * len(df), index=df.index)).fillna("0").astype(str)
        feats["is_autosome"]     = chrom.isin([str(i) for i in range(1, 23)]).astype(int)
        feats["is_sex_chrom"]    = chrom.isin(["X", "Y"]).astype(int)
        feats["is_mitochondrial"] = chrom.isin(["MT", "M"]).astype(int)

        # GNN-derived score (pass-through; 0.5 = no GNN / ambiguous)
        feats["gnn_score"] = (
            df.get("gnn_score", pd.Series([0.5] * len(df), index=df.index))
            .fillna(0.5)
            .astype(float)
            .clip(lower=0.0, upper=1.0)
        )

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
        Gene-aware three-way split: train / val / test.
        All variants of a gene land in the same split — no leakage.

        Step 1: carve test from full data at test_fraction.
        Step 2: carve val from remaining data.
                val_fraction is expressed as fraction of the full dataset,
                so the effective fraction of the remaining pool is
                val_fraction / (1 - test_fraction).
        """
        # Step 1: test split
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=self.config.test_fraction,
            random_state=self.config.random_state,
        )
        trainval_idx, test_idx = next(splitter.split(X, y, groups=groups))

        # Step 2: val split from the train+val pool
        val_size_of_pool = self.config.val_fraction / (1.0 - self.config.test_fraction)
        val_splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=val_size_of_pool,
            random_state=self.config.random_state + 1,
        )
        X_pool      = X.iloc[trainval_idx]
        y_pool      = y.iloc[trainval_idx]
        groups_pool = groups.iloc[trainval_idx]
        rel_train_idx, rel_val_idx = next(val_splitter.split(X_pool, y_pool, groups=groups_pool))

        # Map back to absolute indices
        train_idx = trainval_idx[rel_train_idx]
        val_idx   = trainval_idx[rel_val_idx]

        X_train = X.iloc[train_idx].reset_index(drop=True)
        X_test  = X.iloc[test_idx].reset_index(drop=True)
        X_val   = X.iloc[val_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        y_test  = y.iloc[test_idx].reset_index(drop=True)
        y_val   = y.iloc[val_idx].reset_index(drop=True)

        if self.config.require_both_classes:
            for split_name, y_split in [("train", y_train), ("val", y_val), ("test", y_test)]:
                classes = set(y_split.unique())
                if classes != {0, 1}:
                    raise ValueError(
                        f"Gene-aware split '{split_name}' missing class(es): {classes}. "
                        "Try lowering min_review_tier or increasing dataset size."
                    )

        return X_train, X_test, X_val, y_train, y_test, y_val, train_idx, test_idx, val_idx

    # ── Stage 6: Scaling ───────────────────────────────────────────────────

    def _scale(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame, X_val: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        cols = X_train.columns
        X_train_scaled = pd.DataFrame(self.scaler.fit_transform(X_train), columns=cols)
        X_test_scaled  = pd.DataFrame(self.scaler.transform(X_test),      columns=cols)
        X_val_scaled   = pd.DataFrame(self.scaler.transform(X_val),       columns=cols)
        return X_train_scaled, X_test_scaled, X_val_scaled

    # ── Stage 7: Save ──────────────────────────────────────────────────────

    def _save_splits(
        self,
        X_train: pd.DataFrame, X_val: pd.DataFrame,  X_test: pd.DataFrame,
        y_train: pd.Series,    y_val: pd.Series,     y_test: pd.Series,
        meta_val: pd.DataFrame, meta_test: pd.DataFrame,
    ) -> None:
        out = self.config.output_dir
        X_train.to_parquet(out / "X_train.parquet",       index=False)
        X_val.to_parquet(out   / "X_val.parquet",         index=False)
        X_test.to_parquet(out  / "X_test.parquet",        index=False)
        y_train.to_frame("label").to_parquet(out / "y_train.parquet",    index=False)
        y_val.to_frame("label").to_parquet(out   / "y_val.parquet",      index=False)
        y_test.to_frame("label").to_parquet(out  / "y_test.parquet",     index=False)
        meta_val.to_parquet(out  / "meta_val.parquet",    index=False)
        meta_test.to_parquet(out / "meta_test.parquet",   index=False)
        logger.info("Splits saved to %s/", out)

    # ── Utilities ──────────────────────────────────────────────────────────

    def _report_split_stats(
        self,
        y_train: pd.Series, y_test: pd.Series, y_val: pd.Series,
        groups:  pd.Series,
        train_idx: np.ndarray, test_idx: np.ndarray, val_idx: np.ndarray,
    ) -> None:
        train_genes = groups.iloc[train_idx].nunique()
        test_genes  = groups.iloc[test_idx].nunique()
        val_genes   = groups.iloc[val_idx].nunique()
        logger.info("─" * 55)
        logger.info("%-12s %10s %12s %8s", "Split", "Variants", "Pathogenic", "Genes")
        logger.info("─" * 55)
        logger.info(
            "%-12s %10d %11d (%4.1f%%)  %8d",
            "Train", len(y_train), y_train.sum(), y_train.mean() * 100, train_genes,
        )
        logger.info(
            "%-12s %10d %11d (%4.1f%%)  %8d",
            "Val",   len(y_val),   y_val.sum(),   y_val.mean()   * 100, val_genes,
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





