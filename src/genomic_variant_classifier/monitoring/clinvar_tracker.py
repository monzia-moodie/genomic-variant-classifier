"""
src/monitoring/clinvar_tracker.py
===================================
ClinVar temporal reclassification tracker.

Tracks variant pathogenicity reclassifications across ClinVar releases and
quantifies label drift — the most clinically dangerous form of drift because
the model learns the wrong ground truth.

Key facts about ClinVar dynamics:
  - ClinVar releases monthly; major re-curation events happen annually
  - ~1–3% of variants are reclassified each year
  - VUS → Pathogenic and Likely Pathogenic → Benign are the most impactful
    reclassifications (model must unlearn incorrect signal)
  - Tier-2 curations (expert panel reviewed) are extremely stable;
    tier-3 (single submitter) have >10× the reclassification rate

This module:
  1. Diffs two ClinVar parquets (old vs. new release)
  2. Identifies variants in your training set that were reclassified
  3. Quantifies the clinical impact by severity (P→B is worse than VUS→LP)
  4. Determines if the flip rate exceeds the retraining trigger threshold
  5. Generates a temporal cohort for external validation
  6. Exports a reclassification manifest for audit trail

Usage:
    from src.monitoring.clinvar_tracker import ClinVarTracker

    tracker = ClinVarTracker(training_variant_ids=set(X_train_ids))
    result = tracker.compare(
        old_path="data/processed/clinvar_grch38_2024_01.parquet",
        new_path="data/processed/clinvar_grch38_2024_07.parquet",
    )
    if result.should_retrain:
        run_retraining()
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Clinical impact weights for different reclassification directions
# Higher = more harmful to model quality if the training set contains them
RECLASSIFICATION_WEIGHTS: dict[tuple[str, str], float] = {
    # From → To : impact weight
    ("Pathogenic",          "Benign"):                3.0,  # Worst: trained on wrong signal
    ("Pathogenic",          "Likely benign"):          2.5,
    ("Likely pathogenic",   "Benign"):                2.5,
    ("Likely pathogenic",   "Likely benign"):          2.0,
    ("Benign",              "Pathogenic"):             3.0,  # Also very bad
    ("Benign",              "Likely pathogenic"):      2.5,
    ("Likely benign",       "Pathogenic"):             2.5,
    ("Likely benign",       "Likely pathogenic"):      2.0,
    ("Uncertain significance", "Pathogenic"):          1.5,  # Promotion — helpful
    ("Uncertain significance", "Likely pathogenic"):   1.0,
    ("Uncertain significance", "Benign"):              1.5,
    ("Uncertain significance", "Likely benign"):       1.0,
    ("Pathogenic",          "Uncertain significance"): 1.5,  # Demotion — confusing
    ("Benign",              "Uncertain significance"): 1.5,
}

# Trigger thresholds
FLIP_RATE_MONITOR  = 0.005   # 0.5% of training set reclassified → increase monitoring
FLIP_RATE_RETRAIN  = 0.010   # 1.0% → trigger full retraining
FLIP_RATE_URGENT   = 0.025   # 2.5% → urgent retraining + alert

# Weighted impact threshold (accounts for severity of reclassification)
WEIGHTED_IMPACT_RETRAIN = 0.015


@dataclass
class ReclassifiedVariant:
    variant_id:     str
    gene_symbol:    str
    chrom:          str
    pos:            int
    ref:            str
    alt:            str
    old_class:      str
    new_class:      str
    impact_weight:  float
    in_training_set: bool
    in_val_set:     bool
    in_test_set:    bool
    tier_old:       int
    tier_new:       int


@dataclass
class LabelDriftReport:
    """Complete label drift analysis across a ClinVar version update."""
    old_release:          str
    new_release:          str
    n_variants_old:       int
    n_variants_new:       int
    n_new_variants:       int          # in new but not old
    n_removed_variants:   int          # in old but not new
    n_reclassified_total: int          # any classification change
    n_reclassified_training: int       # in your training set
    n_reclassified_val:   int
    n_reclassified_test:  int
    flip_rate_training:   float        # n_reclassified_training / n_training
    weighted_impact:      float        # impact-weighted flip rate
    direction_breakdown:  dict[str, int] = field(default_factory=dict)
    reclassified:         list[ReclassifiedVariant] = field(default_factory=list)
    should_monitor:       bool = False
    should_retrain:       bool = False
    urgency:              str  = "none"  # "none" | "monitor" | "retrain" | "urgent"
    new_cohort_path:      Optional[str] = None  # path to temporal holdout set
    summary:              str = ""

    def to_json(self, path: str | Path) -> None:
        import dataclasses
        d = dataclasses.asdict(self)
        Path(path).write_text(json.dumps(d, indent=2, default=str), encoding="utf-8")

    def print_summary(self) -> None:
        print(f"\n{'='*60}")
        print(f"  CLINVAR LABEL DRIFT REPORT")
        print(f"  {self.old_release}  →  {self.new_release}")
        print(f"{'='*60}")
        print(f"  Total variants old release:  {self.n_variants_old:,}")
        print(f"  Total variants new release:  {self.n_variants_new:,}")
        print(f"  New variants (unseen):       {self.n_new_variants:,}")
        print(f"  Reclassified (total):        {self.n_reclassified_total:,}")
        print(f"  Reclassified in training:    {self.n_reclassified_training:,}")
        print(f"  Training flip rate:          {self.flip_rate_training:.4%}")
        print(f"  Weighted impact:             {self.weighted_impact:.4%}")
        print(f"\n  Direction breakdown:")
        for direction, count in sorted(
            self.direction_breakdown.items(), key=lambda x: -x[1]
        )[:10]:
            print(f"    {direction}: {count}")
        print(f"\n  URGENCY: {self.urgency.upper()}")
        print(f"  {self.summary}")
        print(f"{'='*60}\n")


class ClinVarTracker:
    """
    Compares two ClinVar release parquets and quantifies label drift.

    Parameters
    ----------
    training_variant_ids : set of variant_id strings in your training set
    val_variant_ids : set of variant_id strings in your validation set
    test_variant_ids : set of variant_id strings in your test set
    """

    # ClinVar 5-tier expert review mapping
    REVIEW_TIER: dict[str, int] = {
        "practice guideline":                                   1,
        "reviewed by expert panel":                             1,
        "criteria provided, multiple submitters, no conflicts": 2,
        "criteria provided, single submitter":                  3,
        "no assertion criteria provided":                       4,
        "no classification provided":                           5,
    }

    def __init__(
        self,
        training_variant_ids: set[str],
        val_variant_ids:      Optional[set[str]] = None,
        test_variant_ids:     Optional[set[str]] = None,
    ) -> None:
        self.training_ids = training_variant_ids
        self.val_ids      = val_variant_ids or set()
        self.test_ids     = test_variant_ids or set()
        logger.info(
            "ClinVarTracker: tracking %d training, %d val, %d test variants.",
            len(self.training_ids), len(self.val_ids), len(self.test_ids),
        )

    def compare(
        self,
        old_path:       str | Path,
        new_path:       str | Path,
        output_dir:     Optional[str | Path] = None,
        old_release:    str = "previous",
        new_release:    str = "current",
    ) -> LabelDriftReport:
        """
        Compare two ClinVar releases and generate a label drift report.

        Parameters
        ----------
        old_path : path to the previous ClinVar parquet
        new_path : path to the new ClinVar parquet
        output_dir : if provided, writes the reclassification manifest and
                     the new temporal cohort parquet here
        """
        cols_needed = ["variant_id", "clinical_sig", "review_status",
                       "gene_symbol", "chrom", "pos", "ref", "alt"]

        old_df = pd.read_parquet(old_path, columns=cols_needed).copy()
        new_df = pd.read_parquet(new_path, columns=cols_needed).copy()

        logger.info(
            "ClinVar diff: %d variants (old) → %d variants (new)",
            len(old_df), len(new_df),
        )

        # Normalise clinical significance
        old_df["_cls"] = old_df["clinical_sig"].apply(self._normalise_sig)
        new_df["_cls"] = new_df["clinical_sig"].apply(self._normalise_sig)

        old_df["_tier"] = old_df["review_status"].apply(self._tier)
        new_df["_tier"] = new_df["review_status"].apply(self._tier)

        # Find variants present in both releases
        merged = old_df.merge(
            new_df,
            on=["variant_id", "chrom", "pos", "ref", "alt"],
            suffixes=("_old", "_new"),
            how="inner",
        )
        merged = merged.rename(columns={
            "gene_symbol_old": "gene_symbol",
            "_cls_old":  "old_class",
            "_cls_new":  "new_class",
            "_tier_old": "tier_old",
            "_tier_new": "tier_new",
        })

        # Identify reclassifications (classification changed)
        reclassified = merged[
            (merged["old_class"] != merged["new_class"]) &
            merged["old_class"].notna() &
            merged["new_class"].notna()
        ].copy()

        logger.info(
            "Reclassified variants: %d / %d (%.2f%%)",
            len(reclassified), len(merged), 100 * len(reclassified) / max(len(merged), 1),
        )

        # Build direction breakdown
        direction_breakdown: dict[str, int] = {}
        for _, row in reclassified.iterrows():
            key = f"{row['old_class']} → {row['new_class']}"
            direction_breakdown[key] = direction_breakdown.get(key, 0) + 1

        # Build per-variant records
        records: list[ReclassifiedVariant] = []
        for _, row in reclassified.iterrows():
            vid    = str(row["variant_id"])
            weight = RECLASSIFICATION_WEIGHTS.get(
                (row["old_class"], row["new_class"]), 1.0
            )
            records.append(ReclassifiedVariant(
                variant_id     = vid,
                gene_symbol    = str(row.get("gene_symbol", "")),
                chrom          = str(row["chrom"]),
                pos            = int(row["pos"]),
                ref            = str(row["ref"]),
                alt            = str(row["alt"]),
                old_class      = str(row["old_class"]),
                new_class      = str(row["new_class"]),
                impact_weight  = weight,
                in_training_set = vid in self.training_ids,
                in_val_set     = vid in self.val_ids,
                in_test_set    = vid in self.test_ids,
                tier_old       = int(row.get("tier_old", 5)),
                tier_new       = int(row.get("tier_new", 5)),
            ))

        training_records = [r for r in records if r.in_training_set]
        val_records      = [r for r in records if r.in_val_set]
        test_records     = [r for r in records if r.in_test_set]

        # Flip rate
        n_training = max(len(self.training_ids), 1)
        flip_rate  = len(training_records) / n_training

        # Weighted impact (severity-weighted flip rate)
        weighted_impact = sum(r.impact_weight for r in training_records) / n_training

        # Determine urgency
        if weighted_impact >= FLIP_RATE_URGENT or flip_rate >= FLIP_RATE_URGENT:
            urgency = "urgent"
        elif weighted_impact >= WEIGHTED_IMPACT_RETRAIN or flip_rate >= FLIP_RATE_RETRAIN:
            urgency = "retrain"
        elif flip_rate >= FLIP_RATE_MONITOR:
            urgency = "monitor"
        else:
            urgency = "none"

        # New variants (in new release but not old) → temporal holdout cohort
        old_ids = set(old_df["variant_id"])
        new_ids = set(new_df["variant_id"])
        genuinely_new = new_ids - old_ids
        new_cohort_path = None

        if output_dir and genuinely_new:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            new_cohort = new_df[new_df["variant_id"].isin(genuinely_new)].copy()
            new_cohort_path = str(output_dir / f"clinvar_new_variants_{new_release}.parquet")
            new_cohort.to_parquet(new_cohort_path, index=False)
            logger.info(
                "Temporal holdout cohort: %d new variants → %s",
                len(new_cohort), new_cohort_path,
            )

            # Write reclassification manifest
            manifest = pd.DataFrame([
                {
                    "variant_id": r.variant_id,
                    "gene_symbol": r.gene_symbol,
                    "old_class": r.old_class,
                    "new_class": r.new_class,
                    "impact_weight": r.impact_weight,
                    "in_training": r.in_training_set,
                    "in_val": r.in_val_set,
                }
                for r in records
            ])
            manifest_path = output_dir / f"reclassification_manifest_{new_release}.csv"
            manifest.to_csv(manifest_path, index=False)
            logger.info("Reclassification manifest → %s", manifest_path)

        summary = (
            f"{len(training_records):,} training variants reclassified "
            f"(flip rate {flip_rate:.3%}, weighted impact {weighted_impact:.3%}). "
            f"Urgency: {urgency}. "
            f"{len(genuinely_new):,} new variants available for temporal validation."
        )

        report = LabelDriftReport(
            old_release           = old_release,
            new_release           = new_release,
            n_variants_old        = len(old_df),
            n_variants_new        = len(new_df),
            n_new_variants        = len(genuinely_new),
            n_removed_variants    = len(old_ids - new_ids),
            n_reclassified_total  = len(records),
            n_reclassified_training = len(training_records),
            n_reclassified_val    = len(val_records),
            n_reclassified_test   = len(test_records),
            flip_rate_training    = round(flip_rate, 6),
            weighted_impact       = round(weighted_impact, 6),
            direction_breakdown   = direction_breakdown,
            reclassified          = records,
            should_monitor        = urgency in ("monitor", "retrain", "urgent"),
            should_retrain        = urgency in ("retrain", "urgent"),
            urgency               = urgency,
            new_cohort_path       = new_cohort_path,
            summary               = summary,
        )
        report.print_summary()
        return report

    def generate_temporal_split(
        self,
        clinvar_path:   str | Path,
        cutoff_date:    str,          # ISO format: "2024-01-01"
        output_dir:     str | Path,
    ) -> dict[str, str]:
        """
        Split ClinVar by submission date to create a temporal train/test split.

        All variants submitted before cutoff_date → training
        All variants submitted after cutoff_date → temporal test set

        This is a stronger external validation than gene-stratified splitting
        because it tests whether the model generalises to genuinely new
        biological discoveries rather than unseen variants from the same era.
        """
        df = pd.read_parquet(clinvar_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Try to find a date column
        date_cols = [c for c in df.columns if "date" in c.lower() or "submitted" in c.lower()]
        if not date_cols:
            raise ValueError(
                f"No date column found in {clinvar_path}. "
                f"Available columns: {list(df.columns)}"
            )

        date_col = date_cols[0]
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        cutoff = pd.to_datetime(cutoff_date)

        train_mask = df[date_col] <= cutoff
        test_mask  = df[date_col] > cutoff

        train_path = str(output_dir / f"clinvar_temporal_train_before_{cutoff_date}.parquet")
        test_path  = str(output_dir / f"clinvar_temporal_test_after_{cutoff_date}.parquet")

        df[train_mask].to_parquet(train_path, index=False)
        df[test_mask].to_parquet(test_path, index=False)

        logger.info(
            "Temporal split at %s: %d train, %d temporal test",
            cutoff_date, train_mask.sum(), test_mask.sum(),
        )
        return {"train": train_path, "temporal_test": test_path}

    # ── Internal helpers ───────────────────────────────────────────────────

    @staticmethod
    def _normalise_sig(sig: str | None) -> Optional[str]:
        """Map raw ClinVar clinical_sig to a canonical 5-class string."""
        if sig is None or (isinstance(sig, float) and np.isnan(sig)):
            return None
        sig = str(sig)
        for term in ("Pathogenic/Likely pathogenic", "Pathogenic"):
            if term in sig:
                return "Pathogenic"
        if "Likely pathogenic" in sig:
            return "Likely pathogenic"
        for term in ("Benign/Likely benign", "Benign"):
            if term in sig and "pathogenic" not in sig.lower():
                return "Benign"
        if "Likely benign" in sig:
            return "Likely benign"
        if "Uncertain" in sig or "VUS" in sig:
            return "Uncertain significance"
        return None

    @staticmethod
    def _tier(status: str | None) -> int:
        if not status or not isinstance(status, str):
            return 5
        status_lower = status.lower()
        for k, v in ClinVarTracker.REVIEW_TIER.items():
            if k in status_lower:
                return v
        return 5