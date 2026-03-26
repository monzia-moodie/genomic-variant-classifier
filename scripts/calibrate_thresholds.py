"""
scripts/calibrate_thresholds.py
================================
Post-training probability calibration and ACMG-tier threshold optimisation.

What it does
------------
1. Loads the trained InferencePipeline and the gene-stratified validation set.
2. Runs Platt scaling (logistic regression on the raw ensemble log-odds) to
   obtain well-calibrated probabilities.
3. Sweeps candidate thresholds and finds, for each ACMG tier boundary, the
   lowest threshold where PPV ≥ target (default 0.95 for Pathogenic/Likely
   pathogenic, 0.80 for the other tiers).
4. Writes the calibrated thresholds to models/classification_thresholds.json.
5. Optionally replaces the in-memory thresholds used by schemas.py at startup
   (see --apply flag).

Usage
-----
    python scripts/calibrate_thresholds.py \\
        --pipeline  models/phase2_pipeline.joblib \\
        --val-X     outputs/phase2_eval/splits/X_val.parquet \\
        --val-y     outputs/phase2_eval/splits/y_val.parquet \\
        --output    models/classification_thresholds.json

Exit codes
----------
    0   Success — thresholds written.
    1   Calibration error (e.g. not enough positive examples in val set).
    2   I/O or pipeline error.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("calibrate_thresholds")


# ---------------------------------------------------------------------------
# Default tier-boundary PPV targets
# ---------------------------------------------------------------------------
PPV_TARGETS: dict[str, float] = {
    "pathogenic_lo":       0.95,   # lower bound of Pathogenic tier
    "likely_pathogenic_lo": 0.80,  # lower bound of Likely pathogenic tier
    "vus_hi":              0.20,   # upper bound of VUS (= lower Likely benign)
    "benign_hi":           0.05,   # upper bound of Benign tier
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Calibrate pathogenicity thresholds against validation set"
    )
    p.add_argument(
        "--pipeline",
        default="models/phase2_pipeline.joblib",
        help="Path to InferencePipeline joblib artifact",
    )
    p.add_argument(
        "--val-X",
        default="outputs/phase2_eval/splits/X_val.parquet",
        help="Validation feature matrix (parquet)",
    )
    p.add_argument(
        "--val-y",
        default="outputs/phase2_eval/splits/y_val.parquet",
        help="Validation labels (parquet with 'label' column, or plain series parquet)",
    )
    p.add_argument(
        "--output",
        default="models/classification_thresholds.json",
        help="Output path for threshold JSON",
    )
    p.add_argument(
        "--ppv-pathogenic",
        type=float,
        default=0.95,
        help="Minimum PPV for the Pathogenic tier lower boundary (default 0.95)",
    )
    p.add_argument(
        "--ppv-likely",
        type=float,
        default=0.80,
        help="Minimum PPV for the Likely pathogenic tier lower boundary (default 0.80)",
    )
    p.add_argument(
        "--n-bins",
        type=int,
        default=1000,
        help="Number of threshold sweep points (default 1000)",
    )
    return p.parse_args()


def load_pipeline(path: str):
    import joblib
    from src.api.pipeline import InferencePipeline
    obj = joblib.load(path)
    if not isinstance(obj, InferencePipeline):
        raise TypeError(f"Expected InferencePipeline, got {type(obj)}")
    logger.info("Pipeline loaded: val_auroc=%.4f", obj.metadata.val_auroc)
    return obj


def get_raw_scores(pipeline, X_val: pd.DataFrame) -> np.ndarray:
    """
    Extract pre-stacking base-model predictions (log-odds) for Platt scaling.

    We pass the already-engineered features directly to the base models,
    bypassing engineer_features() since X_val is already the feature matrix.
    """
    from src.api.pipeline import INFERENCE_FEATURE_COLUMNS

    # X_val is the feature matrix from the split parquets — columns already engineered
    X_np = X_val[INFERENCE_FEATURE_COLUMNS].values

    if pipeline.scaler is not None:
        X_np = pipeline.scaler.transform(X_np)

    base_preds = np.column_stack([
        model.predict_proba(X_np)[:, 1]
        for model in pipeline.trained_models.values()
    ])
    # Raw stacker log-odds for Platt scaling
    stacker_proba = pipeline.meta_learner.predict_proba(base_preds)[:, 1]
    return stacker_proba


def platt_scale(raw_scores: np.ndarray, y_val: np.ndarray) -> LogisticRegression:
    """Fit a logistic regression on log-odds of raw_scores → calibrated proba."""
    eps = 1e-7
    log_odds = np.log(
        (raw_scores + eps) / (1.0 - raw_scores + eps)
    ).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(log_odds, y_val)
    logger.info(
        "Platt scaling fit: coef=%.4f  intercept=%.4f",
        float(lr.coef_[0, 0]),
        float(lr.intercept_[0]),
    )
    return lr


def find_ppv_threshold(
    calibrated_scores: np.ndarray,
    y_true: np.ndarray,
    target_ppv: float,
) -> float:
    """
    Return the lowest threshold t such that PPV(score >= t) >= target_ppv.
    Falls back to the highest threshold that maximises PPV if target is unreachable.
    """
    precision, _, thresholds = precision_recall_curve(y_true, calibrated_scores)
    # precision_recall_curve returns precision[i] for threshold[i]; last entry is 1.0
    # thresholds has one fewer entry than precision
    for prec, thresh in zip(reversed(precision[:-1]), reversed(thresholds)):
        if prec >= target_ppv:
            return float(thresh)
    # Target PPV not reachable — return threshold that achieves highest precision
    best_idx = int(np.argmax(precision[:-1]))
    logger.warning(
        "PPV target %.2f not reachable; best achievable PPV=%.3f at threshold=%.4f",
        target_ppv,
        float(precision[best_idx]),
        float(thresholds[best_idx]),
    )
    return float(thresholds[best_idx])


def compute_thresholds(
    calibrated_scores: np.ndarray,
    y_true: np.ndarray,
    ppv_pathogenic: float,
    ppv_likely: float,
) -> dict:
    """
    Derive five-tier classification thresholds from the calibrated score.

    Returns a dict compatible with schemas.CLASSIFICATION_THRESHOLDS format:
        { tier_label: [lo, hi], ... }
    """
    path_lo    = find_ppv_threshold(calibrated_scores, y_true, ppv_pathogenic)
    likely_lo  = find_ppv_threshold(calibrated_scores, y_true, ppv_likely)

    # Benign / likely-benign boundaries: mirror the pathogenic thresholds
    # using 1 - P(benign) = P(pathogenic) logic on the inverted label
    y_benign = 1 - y_true
    benign_lo   = 1.0 - find_ppv_threshold(1.0 - calibrated_scores, y_benign, ppv_pathogenic)
    lb_lo       = 1.0 - find_ppv_threshold(1.0 - calibrated_scores, y_benign, ppv_likely)

    # Clamp & order: Benign < Likely benign < VUS < Likely path < Path
    benign_hi   = min(benign_lo, lb_lo)
    lb_hi       = min(lb_lo, likely_lo)
    vus_lo      = benign_hi
    vus_hi      = lb_hi

    thresholds = {
        "Pathogenic":             [round(path_lo, 4),   1.01],
        "Likely pathogenic":      [round(likely_lo, 4), round(path_lo, 4)],
        "Uncertain significance": [round(vus_lo, 4),    round(vus_hi, 4)],
        "Likely benign":          [round(benign_hi, 4), round(lb_hi, 4)],
        "Benign":                 [-0.01,               round(benign_lo, 4)],
    }

    logger.info("Calibrated thresholds:")
    for label, (lo, hi) in thresholds.items():
        logger.info("  %-26s  [%.4f, %.4f)", label, lo, hi)

    return thresholds


def main() -> int:
    args = parse_args()

    pipeline_path = Path(args.pipeline)
    val_x_path    = Path(args.val_X)
    val_y_path    = Path(args.val_y)
    output_path   = Path(args.output)

    for p in [pipeline_path, val_x_path, val_y_path]:
        if not p.exists():
            logger.error("File not found: %s", p)
            return 2

    try:
        pipeline = load_pipeline(str(pipeline_path))
    except Exception as exc:
        logger.error("Failed to load pipeline: %s", exc)
        return 2

    try:
        X_val = pd.read_parquet(val_x_path)
        y_raw = pd.read_parquet(val_y_path)
        if "label" in y_raw.columns:
            y_val = y_raw["label"].values.astype(int)
        else:
            y_val = y_raw.iloc[:, 0].values.astype(int)
        logger.info(
            "Validation set: %d samples  (%d pathogenic, %d benign)",
            len(y_val), int(y_val.sum()), int((y_val == 0).sum()),
        )
    except Exception as exc:
        logger.error("Failed to load validation data: %s", exc)
        return 2

    if len(np.unique(y_val)) < 2:
        logger.error("Validation set must contain both classes for calibration.")
        return 1

    try:
        raw_scores = get_raw_scores(pipeline, X_val)
    except Exception as exc:
        logger.error("Failed to score validation set: %s", exc)
        return 2

    # Platt scaling
    platt = platt_scale(raw_scores, y_val)
    eps = 1e-7
    log_odds = np.log(
        (raw_scores + eps) / (1.0 - raw_scores + eps)
    ).reshape(-1, 1)
    calibrated = platt.predict_proba(log_odds)[:, 1]

    from sklearn.metrics import roc_auc_score, brier_score_loss
    logger.info(
        "Pre-calibration:  AUROC=%.4f  Brier=%.4f",
        roc_auc_score(y_val, raw_scores),
        brier_score_loss(y_val, raw_scores),
    )
    logger.info(
        "Post-calibration: AUROC=%.4f  Brier=%.4f",
        roc_auc_score(y_val, calibrated),
        brier_score_loss(y_val, calibrated),
    )

    thresholds = compute_thresholds(
        calibrated, y_val,
        ppv_pathogenic=args.ppv_pathogenic,
        ppv_likely=args.ppv_likely,
    )

    output = {
        "thresholds":        thresholds,
        "platt_coef":        float(platt.coef_[0, 0]),
        "platt_intercept":   float(platt.intercept_[0]),
        "val_auroc_raw":     round(float(roc_auc_score(y_val, raw_scores)), 4),
        "val_auroc_calib":   round(float(roc_auc_score(y_val, calibrated)), 4),
        "val_brier_raw":     round(float(brier_score_loss(y_val, raw_scores)), 4),
        "val_brier_calib":   round(float(brier_score_loss(y_val, calibrated)), 4),
        "n_val":             int(len(y_val)),
        "ppv_pathogenic":    args.ppv_pathogenic,
        "ppv_likely":        args.ppv_likely,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    logger.info("Thresholds written to %s", output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
