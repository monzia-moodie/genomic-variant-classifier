"""
scripts/conformal_prediction.py
=================================
Phase 8.3 — Conformal prediction intervals for /predict responses.

Implements split conformal prediction (Papadopoulos et al. 2002 / Venn–Abers
variant) to produce calibrated prediction sets with guaranteed marginal
coverage at a user-specified error level α.

Algorithm (split conformal)
---------------------------
1. Score a calibration split with the trained pipeline → s_i = f(x_i).
2. Define non-conformity scores as residuals:
     r_i = |y_i - s_i|        for regression-style intervals
   or for classification:
     r_i = 1 - s_i  if y_i = 1   (pathogenic)
     r_i = s_i      if y_i = 0   (benign)
3. Compute the (1-α)(1 + 1/n) quantile q̂ of {r_i}.
4. At test time, the prediction set is:
     Ŷ = {y : r(x, y) ≤ q̂}
   which for binary labels collapses to a score interval:
     score ∈ [s - q̂,  s + q̂]  clipped to [0, 1]

Coverage guarantee: P(y_true ∈ Ŷ) ≥ 1 - α for exchangeable data.

Output
------
  conformal_config.json   q̂ values for α ∈ {0.01, 0.05, 0.10, 0.20}
                          + empirical coverage on the calibration set

  The API can load conformal_config.json at startup and attach
  prediction_interval_low / prediction_interval_high to /predict responses.

Usage
-----
  python scripts/conformal_prediction.py \
      --pipeline  models/phase2_pipeline.joblib \
      --cal-split data/splits/val.parquet \
      --output    models/conformal_config.json \
      --alpha     0.05 0.10

  # Add --apply to score a test parquet and write intervals:
  python scripts/conformal_prediction.py \
      --pipeline  models/phase2_pipeline.joblib \
      --cal-split data/splits/val.parquet \
      --apply     data/splits/test.parquet \
      --output    outputs/test_with_intervals.parquet
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

_DEFAULT_ALPHAS = [0.01, 0.05, 0.10, 0.20]


# ---------------------------------------------------------------------------
# Core conformal calibration
# ---------------------------------------------------------------------------

def calibrate(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    alpha: float,
) -> float:
    """
    Compute the conformal quantile q̂ for a given significance level α.

    Non-conformity score:
        r_i = 1 - y_prob_i  if y_i = 1
        r_i = y_prob_i      if y_i = 0

    The quantile is the ceil((n+1)(1-α))/n -th empirical quantile of {r_i},
    which ensures marginal coverage ≥ 1-α.

    Parameters
    ----------
    y_true  : binary labels (0/1), shape (n,)
    y_prob  : predicted probabilities, shape (n,)
    alpha   : target error level ∈ (0, 1)

    Returns
    -------
    q_hat : float — conformal quantile
    """
    n = len(y_true)
    scores = np.where(y_true == 1, 1.0 - y_prob, y_prob)
    # Adjusted quantile level for finite-sample coverage
    level  = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    q_hat  = float(np.quantile(scores, level))
    return q_hat


def prediction_interval(
    y_prob: np.ndarray,
    q_hat: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (lower, upper) prediction interval for an array of probabilities.

    Intervals are clipped to [0, 1].
    """
    lo = np.clip(y_prob - q_hat, 0.0, 1.0)
    hi = np.clip(y_prob + q_hat, 0.0, 1.0)
    return lo, hi


def empirical_coverage(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    q_hat: float,
) -> float:
    """
    Empirical coverage = fraction of calibration samples where the true label
    is inside the prediction interval.
    """
    lo, hi = prediction_interval(y_prob, q_hat)
    # A label is "inside" the interval if:
    #   pathogenic (y=1): upper bound ≥ threshold to call positive
    #   benign (y=0):     lower bound ≤ threshold to call negative
    # Equivalently, the non-conformity score ≤ q_hat
    scores   = np.where(y_true == 1, 1.0 - y_prob, y_prob)
    covered  = (scores <= q_hat).mean()
    return float(covered)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Conformal prediction interval calibration."
    )
    parser.add_argument("--pipeline",  required=True, help="InferencePipeline .joblib")
    parser.add_argument("--cal-split", required=True, help="Calibration parquet (must have 'label' column)")
    parser.add_argument("--output",    default="models/conformal_config.json",
                        help="Output JSON path (default: models/conformal_config.json)")
    parser.add_argument("--alpha",     type=float, nargs="+", default=_DEFAULT_ALPHAS,
                        help="Error levels to calibrate (default: 0.01 0.05 0.10 0.20)")
    parser.add_argument("--apply",     default=None,
                        help="Optional: path to test parquet to annotate with intervals")
    parser.add_argument("--apply-alpha", type=float, default=0.05,
                        help="α level to use when --apply is set (default: 0.05)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Load pipeline ---
    logger.info("Loading pipeline from %s …", args.pipeline)
    from src.api.pipeline import InferencePipeline
    pipeline = InferencePipeline.load(args.pipeline)

    # --- Load calibration split ---
    logger.info("Loading calibration split from %s …", args.cal_split)
    cal_df  = pd.read_parquet(args.cal_split)
    if "label" not in cal_df.columns:
        logger.error("Calibration parquet must have a 'label' column.")
        return 1

    y_true  = cal_df["label"].astype(int).values
    y_prob  = pipeline.predict_proba(cal_df)
    n_cal   = len(y_true)
    logger.info("Calibration set: %d variants (%d pathogenic).", n_cal, int(y_true.sum()))

    # --- Calibrate for each α ---
    config: dict = {"n_calibration": n_cal, "alphas": {}}

    for alpha in sorted(args.alpha):
        q_hat    = calibrate(y_true, y_prob, alpha)
        coverage = empirical_coverage(y_true, y_prob, q_hat)
        config["alphas"][str(alpha)] = {
            "q_hat":             round(q_hat, 6),
            "empirical_coverage": round(coverage, 4),
            "target_coverage":   round(1 - alpha, 4),
        }
        logger.info(
            "α=%.2f  q̂=%.4f  empirical_coverage=%.4f  (target=%.4f)",
            alpha, q_hat, coverage, 1 - alpha,
        )

    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info("Conformal config written to %s", output_path)

    # --- Optional: annotate a test set ---
    if args.apply:
        apply_alpha = args.apply_alpha
        q_hat = config["alphas"].get(str(apply_alpha), {}).get("q_hat")
        if q_hat is None:
            logger.error("α=%.2f not in calibrated alphas.", apply_alpha)
            return 1

        logger.info("Annotating %s with α=%.2f intervals …", args.apply, apply_alpha)
        test_df = pd.read_parquet(args.apply)
        test_prob = pipeline.predict_proba(test_df)
        lo, hi = prediction_interval(test_prob, q_hat)

        test_df = test_df.copy()
        test_df["pathogenicity_score"] = test_prob.round(4)
        test_df["interval_low"]        = lo.round(4)
        test_df["interval_high"]       = hi.round(4)
        test_df["interval_width"]      = (hi - lo).round(4)

        apply_out = Path(args.apply).with_suffix("").name + "_with_intervals.parquet"
        apply_path = output_path.parent / apply_out
        test_df.to_parquet(apply_path, index=False)
        logger.info("Annotated test set written to %s", apply_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
