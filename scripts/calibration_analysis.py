"""
scripts/calibration_analysis.py
=================================
Phase 8.2 — Calibration analysis: ECE, per-consequence-class curves,
reliability diagrams.

Reads the validation split predictions (from run_phase2_eval.py output) and
produces a detailed calibration report broken down by:
  - Overall
  - Consequence class (missense, LoF, splice, synonymous, other)
  - ACMG tier operating point

Output files (--output-dir)
----------------------------
  calibration_overall.parquet    Reliability diagram data (overall)
  calibration_by_consequence.parquet  Per-class calibration bins
  calibration_metrics.json       ECE/MCE for each consequence group
  calibration_report.txt         Human-readable summary

Usage
-----
  python scripts/calibration_analysis.py \
      --pipeline    models/phase2_pipeline.joblib \
      --val-parquet data/splits/val.parquet \
      --output-dir  outputs/calibration
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Consequence → coarse class mapping
_CONSEQUENCE_CLASS: dict[str, str] = {
    "missense_variant":             "missense",
    "protein_altering_variant":     "missense",
    "stop_gained":                  "lof",
    "frameshift_variant":           "lof",
    "stop_lost":                    "lof",
    "start_lost":                   "lof",
    "transcript_ablation":          "lof",
    "splice_donor_variant":         "splice",
    "splice_acceptor_variant":      "splice",
    "splice_region_variant":        "splice",
    "synonymous_variant":           "synonymous",
    "coding_sequence_variant":      "synonymous",
}


def _map_consequence_class(consequence: str) -> str:
    return _CONSEQUENCE_CLASS.get(str(consequence).strip(), "other")


def calibration_bins(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> pd.DataFrame:
    """Return reliability diagram data as a DataFrame."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    n    = len(y_true)
    ece  = 0.0
    mce  = 0.0

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        cnt  = int(mask.sum())
        if cnt == 0:
            rows.append({
                "bin_lower": round(lo, 4), "bin_upper": round(hi, 4),
                "mean_prob": round((lo + hi) / 2, 4), "fraction_pos": float("nan"),
                "count": 0,
            })
            continue
        frac_pos  = float(y_true[mask].mean())
        mean_prob = float(y_prob[mask].mean())
        gap       = abs(mean_prob - frac_pos)
        ece      += gap * cnt / n
        mce       = max(mce, gap)
        rows.append({
            "bin_lower":    round(lo, 4),
            "bin_upper":    round(hi, 4),
            "mean_prob":    round(mean_prob, 4),
            "fraction_pos": round(frac_pos, 4),
            "count":        cnt,
        })

    df = pd.DataFrame(rows)
    df["ece"] = round(ece, 4)
    df["mce"] = round(mce, 4)
    return df


def _calibration_summary(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    df = calibration_bins(y_true, y_prob)
    return {
        "n":   len(y_true),
        "ece": float(df["ece"].iloc[0]),
        "mce": float(df["mce"].iloc[0]),
        "mean_predicted_prob": round(float(y_prob.mean()), 4),
        "prevalence":          round(float(y_true.mean()), 4),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Calibration analysis for the variant pathogenicity pipeline."
    )
    parser.add_argument("--pipeline",    required=True, help="Path to InferencePipeline .joblib")
    parser.add_argument("--val-parquet", required=True, help="Validation split parquet with 'label' column")
    parser.add_argument("--output-dir",  default="outputs/calibration")
    parser.add_argument("--n-bins",      type=int, default=15)
    parser.add_argument("--log-level",   default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load ---
    logger.info("Loading pipeline from %s …", args.pipeline)
    from src.api.pipeline import InferencePipeline
    pipeline = InferencePipeline.load(args.pipeline)

    logger.info("Loading validation split from %s …", args.val_parquet)
    df = pd.read_parquet(args.val_parquet)

    if "label" not in df.columns:
        logger.error("Validation parquet must contain a 'label' column.")
        return 1

    y_true = df["label"].astype(int).values

    # --- Score ---
    logger.info("Scoring %d validation variants …", len(df))
    y_prob = pipeline.predict_proba(df)

    # --- Overall calibration ---
    logger.info("Computing overall calibration …")
    overall_df = calibration_bins(y_true, y_prob, n_bins=args.n_bins)
    overall_df.to_parquet(output_dir / "calibration_overall.parquet", index=False)

    # --- Per-consequence calibration ---
    if "consequence" in df.columns:
        df["_consequence_class"] = df["consequence"].map(_map_consequence_class).fillna("other")
        groups = df["_consequence_class"].unique()

        class_rows = []
        class_metrics = {}
        for grp in sorted(groups):
            mask = df["_consequence_class"] == grp
            if mask.sum() < 50:
                continue
            grp_true = y_true[mask.values]
            grp_prob = y_prob[mask.values]
            grp_df   = calibration_bins(grp_true, grp_prob, n_bins=args.n_bins)
            grp_df["consequence_class"] = grp
            class_rows.append(grp_df)
            class_metrics[grp] = _calibration_summary(grp_true, grp_prob)
            logger.info(
                "  %-12s  n=%5d  ECE=%.4f  MCE=%.4f",
                grp, class_metrics[grp]["n"],
                class_metrics[grp]["ece"], class_metrics[grp]["mce"],
            )

        if class_rows:
            pd.concat(class_rows, ignore_index=True).to_parquet(
                output_dir / "calibration_by_consequence.parquet", index=False
            )
    else:
        class_metrics = {}
        logger.warning("No 'consequence' column — skipping per-class calibration.")

    # --- Write metrics JSON ---
    metrics = {
        "overall": _calibration_summary(y_true, y_prob),
        "by_consequence": class_metrics,
    }
    with open(output_dir / "calibration_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # --- Human-readable report ---
    overall_ece = metrics["overall"]["ece"]
    overall_mce = metrics["overall"]["mce"]
    lines = [
        "Calibration Analysis Report",
        "=" * 44,
        f"Validation variants : {metrics['overall']['n']:,}",
        f"Prevalence          : {metrics['overall']['prevalence']:.3f}",
        f"Mean predicted prob : {metrics['overall']['mean_predicted_prob']:.3f}",
        f"ECE (overall)       : {overall_ece:.4f}",
        f"MCE (overall)       : {overall_mce:.4f}",
        "",
        "Per-consequence ECE",
        "-" * 44,
    ]
    for grp, m in class_metrics.items():
        lines.append(f"  {grp:<14}  n={m['n']:>6,}  ECE={m['ece']:.4f}  MCE={m['mce']:.4f}")

    report_text = "\n".join(lines)
    (output_dir / "calibration_report.txt").write_text(report_text + "\n", encoding="utf-8")
    print(report_text)

    logger.info("Outputs written to %s/", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
