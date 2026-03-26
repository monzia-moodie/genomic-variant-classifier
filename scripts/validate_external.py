"""
scripts/validate_external.py
==============================
Phase 8.1 — External validation cohort evaluation.

Runs the trained InferencePipeline against an external validation cohort
(ClinVar holdout, gnomAD de-novo, or a custom labelled VCF) and writes a
comprehensive performance report.

Metrics computed
----------------
  AUROC          Area under ROC curve
  AUPRC          Area under precision-recall curve
  ECE            Expected calibration error (15 equal-width bins)
  MCE            Maximum calibration error
  Sensitivity    At each ACMG-tier operating point
  Specificity    At each ACMG-tier operating point
  PPV / NPV      Positive / negative predictive value
  F1             At each operating point
  Confusion matrix (normalised and absolute)

Output files (--output-dir)
----------------------------
  metrics.json          All scalar metrics
  roc_curve.parquet     fpr, tpr, threshold columns
  pr_curve.parquet      precision, recall, threshold columns
  calibration.parquet   mean_prob, fraction_positives per bin
  predictions.parquet   variant_id, label, score, classification

Usage
-----
  python scripts/validate_external.py \
      --pipeline   models/phase2_pipeline.joblib \
      --cohort     data/external/validation_cohort.parquet \
      --output-dir outputs/external_validation

  # cohort parquet must have columns:
  #   - All feature columns accepted by /predict (chrom, pos, ref, alt, …)
  #   - label  (1=pathogenic, 0=benign)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------

def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> tuple[float, float, pd.DataFrame]:
    """
    Compute ECE and MCE using equal-width probability bins.

    Returns
    -------
    ece : float
    mce : float
    cal_df : pd.DataFrame  columns: bin_lower, bin_upper, mean_prob, fraction_pos, count
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    rows = []
    n   = len(y_true)

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        frac_pos  = float(y_true[mask].mean())
        mean_prob = float(y_prob[mask].mean())
        cnt       = int(mask.sum())
        gap       = abs(mean_prob - frac_pos)
        ece      += gap * cnt / n
        mce       = max(mce, gap)
        rows.append({
            "bin_lower":       round(lo, 4),
            "bin_upper":       round(hi, 4),
            "mean_prob":       round(mean_prob, 4),
            "fraction_pos":    round(frac_pos, 4),
            "count":           cnt,
        })

    return round(ece, 4), round(mce, 4), pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-threshold metrics
# ---------------------------------------------------------------------------

def _threshold_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: dict[str, float],
) -> dict[str, dict[str, float]]:
    results = {}
    for tier, thr in thresholds.items():
        y_pred = (y_prob >= thr).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv  = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        f1   = f1_score(y_true, y_pred, zero_division=0)

        results[tier] = {
            "threshold":   round(thr, 4),
            "sensitivity": round(sens, 4),
            "specificity": round(spec, 4),
            "ppv":         round(ppv, 4),
            "npv":         round(npv, 4),
            "f1":          round(f1, 4),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        }
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="External cohort validation for the variant pathogenicity pipeline."
    )
    parser.add_argument("--pipeline",   required=True, help="Path to InferencePipeline .joblib")
    parser.add_argument("--cohort",     required=True, help="Labelled cohort parquet (must have 'label' column)")
    parser.add_argument("--output-dir", default="outputs/external_validation",
                        help="Directory for output files (default: outputs/external_validation)")
    parser.add_argument("--batch-size", type=int, default=2048,
                        help="Scoring batch size (default: 2048)")
    parser.add_argument("--log-level",  default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load pipeline ---
    logger.info("Loading pipeline from %s …", args.pipeline)
    from src.api.pipeline import InferencePipeline
    pipeline = InferencePipeline.load(args.pipeline)

    # --- Load cohort ---
    logger.info("Loading cohort from %s …", args.cohort)
    df = pd.read_parquet(args.cohort)
    if "label" not in df.columns:
        logger.error("Cohort parquet must contain a 'label' column (1=pathogenic, 0=benign).")
        return 1

    y_true = df["label"].astype(int).values
    n_pos  = int(y_true.sum())
    n_neg  = int((y_true == 0).sum())
    logger.info("Cohort: %d variants (%d pathogenic, %d benign).", len(df), n_pos, n_neg)

    if set(np.unique(y_true)) != {0, 1}:
        logger.error("'label' column must be binary (0/1). Found: %s", np.unique(y_true).tolist())
        return 1

    # --- Score in batches ---
    logger.info("Scoring %d variants (batch_size=%d) …", len(df), args.batch_size)
    scores = []
    for start in range(0, len(df), args.batch_size):
        chunk = df.iloc[start : start + args.batch_size]
        proba = pipeline.predict_proba(chunk)
        scores.append(proba)
    y_prob = np.concatenate(scores)

    # --- AUROC / AUPRC ---
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    logger.info("AUROC=%.4f  AUPRC=%.4f", auroc, auprc)

    # --- Calibration ---
    ece, mce, cal_df = expected_calibration_error(y_true, y_prob)
    logger.info("ECE=%.4f  MCE=%.4f", ece, mce)

    # --- Load thresholds (from calibrated file or defaults) ---
    from src.api.schemas import CLASSIFICATION_THRESHOLDS
    threshold_metrics = _threshold_metrics(y_true, y_prob, CLASSIFICATION_THRESHOLDS)

    # --- ROC / PR curves ---
    fpr, tpr, roc_thr = roc_curve(y_true, y_prob)
    prec, rec, pr_thr = precision_recall_curve(y_true, y_prob)

    # --- Predictions table ---
    from src.api.schemas import score_to_classification
    pred_df = pd.DataFrame({
        "variant_id":    df.get("variant_id", pd.Series(
            [f"{r.chrom}:{r.pos}:{r.ref}:{r.alt}" for _, r in df.iterrows()],
            index=df.index,
        )),
        "label":         y_true,
        "score":         y_prob.round(4),
        "classification": [score_to_classification(float(p))[0] for p in y_prob],
    })

    # --- Write outputs ---
    metrics = {
        "n_variants":    len(df),
        "n_pathogenic":  n_pos,
        "n_benign":      n_neg,
        "auroc":         round(auroc, 4),
        "auprc":         round(auprc, 4),
        "ece":           ece,
        "mce":           mce,
        "threshold_metrics": threshold_metrics,
        "pipeline_val_auroc": pipeline.metadata.val_auroc,
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": np.append(roc_thr, 1.0)}).to_parquet(
        output_dir / "roc_curve.parquet", index=False
    )
    pd.DataFrame({"precision": prec, "recall": rec, "threshold": np.append(pr_thr, 1.0)}).to_parquet(
        output_dir / "pr_curve.parquet", index=False
    )
    cal_df.to_parquet(output_dir / "calibration.parquet", index=False)
    pred_df.to_parquet(output_dir / "predictions.parquet", index=False)

    logger.info("Outputs written to %s/", output_dir)
    logger.info(
        "Summary: AUROC=%.4f  AUPRC=%.4f  ECE=%.4f  n=%d",
        auroc, auprc, ece, len(df),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
