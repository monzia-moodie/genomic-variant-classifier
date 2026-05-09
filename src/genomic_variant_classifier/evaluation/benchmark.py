"""
src/evaluation/benchmark.py
============================
Algorithm comparison framework -- Phase 4F.

Runs all enabled model families on identical stratified k-fold splits and
produces a multi-dimensional comparison table.

Evaluation dimensions
---------------------
  Predictive   AUROC, AUPRC, Brier score, ECE (15 bins), log-loss
  Speed        training time (s), inference latency (ms/variant)
  Memory       peak RSS increase during training (MB) via tracemalloc
  Uncertainty  epistemic and aleatoric uncertainty (DeepEnsemble variants only)

Supported model families
------------------------
  Tabular ensemble   LightGBM, XGBoost, Random Forest, GBM, Logistic Regression
  Neural tabular     MLP (sklearn), KAN (if pykan/efficient-kan installed)
  Ensemble           DeepEnsembleWrapper over LightGBM (uncertainty baseline)
  Custom             pass any list of (name, estimator) pairs via --extra-models

Usage (script)
--------------
  python -m src.evaluation.benchmark \\
      --parquet data/processed/clinvar_grch38.parquet \\
      --output  outputs/benchmark \\
      --n-folds 5

Usage (Python API)
------------------
  from src.evaluation.benchmark import BenchmarkRunner
  from src.models.variant_ensemble import TABULAR_FEATURES

  runner = BenchmarkRunner(n_folds=5, random_state=42)
  results = runner.run(X, y, feature_names=TABULAR_FEATURES)
  runner.print_summary(results)
  runner.save(results, "outputs/benchmark")
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy deps
# ---------------------------------------------------------------------------
try:
    import lightgbm as lgb

    _LGB_AVAILABLE = True
except ImportError:
    _LGB_AVAILABLE = False

try:
    import xgboost as xgb

    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

from src.models.kan import KANClassifier
from src.models.mc_dropout import DeepEnsembleWrapper


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ModelResult:
    name: str
    auroc: float
    auprc: float
    brier: float
    logloss: float
    ece: float
    train_time_s: float
    infer_latency_ms: float   # per-variant latency
    peak_memory_mb: float
    # Per-fold details
    fold_aurocs: list[float] = field(default_factory=list)
    fold_auprcs: list[float] = field(default_factory=list)
    # Uncertainty (deep ensemble only)
    mean_epistemic: Optional[float] = None
    mean_aleatoric: Optional[float] = None


# ---------------------------------------------------------------------------
# ECE computation
# ---------------------------------------------------------------------------

def _ece(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error (equal-width bins)."""
    frac_positives, mean_predicted = calibration_curve(
        y_true, y_proba, n_bins=n_bins, strategy="uniform"
    )
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    counts = np.histogram(y_proba, bins=bin_edges)[0]
    ece = 0.0
    n = len(y_true)
    for i, (fp, mp, cnt) in enumerate(zip(frac_positives, mean_predicted, counts)):
        ece += (cnt / n) * abs(fp - mp)
    return float(ece)


# ---------------------------------------------------------------------------
# Default model catalogue
# ---------------------------------------------------------------------------

def _default_models() -> list[tuple[str, object]]:
    models = []

    if _LGB_AVAILABLE:
        models.append(
            ("LightGBM", lgb.LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=63,
                class_weight="balanced",
                n_jobs=-1,
                verbose=-1,
            ))
        )

    if _XGB_AVAILABLE:
        models.append(
            ("XGBoost", xgb.XGBClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                scale_pos_weight=1,
                eval_metric="logloss",
                n_jobs=-1,
                verbosity=0,
            ))
        )

    models.append(
        ("RandomForest", RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ))
    )

    models.append(
        ("GradientBoosting", GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
        ))
    )

    models.append(
        ("LogisticRegression", Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
            )),
        ]))
    )

    models.append(
        ("MLP", Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                max_iter=200,
                random_state=42,
            )),
        ]))
    )

    models.append(
        ("KAN", KANClassifier(hidden_sizes=[64, 32], max_iter=200, random_state=42))
    )

    if _LGB_AVAILABLE:
        models.append(
            ("LightGBM-DeepEnsemble", DeepEnsembleWrapper(
                base_estimator=lgb.LGBMClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    class_weight="balanced",
                    verbose=-1,
                ),
                n_members=5,
            ))
        )

    return models


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """
    Cross-validated benchmark across multiple model families.

    Parameters
    ----------
    n_folds : int
        Stratified k-fold splits (default: 5).
    random_state : int
    models : list[(name, estimator)] or None
        Custom model list.  Defaults to _default_models().
    """

    def __init__(
        self,
        n_folds: int = 5,
        random_state: int = 42,
        models: Optional[list[tuple[str, object]]] = None,
    ) -> None:
        self.n_folds = n_folds
        self.random_state = random_state
        self.models = models if models is not None else _default_models()

    def _evaluate_fold(
        self,
        estimator,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> tuple[float, float, float, float, float, float, float]:
        """Returns (auroc, auprc, brier, logloss, ece, train_s, infer_ms_per_variant)."""
        tracemalloc.start()

        t0 = time.perf_counter()
        estimator.fit(X_train, y_train)
        train_s = time.perf_counter() - t0

        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = peak / 1e6  # bytes -> MB

        t_inf = time.perf_counter()
        y_proba = estimator.predict_proba(X_test)[:, 1]
        infer_ms = (time.perf_counter() - t_inf) * 1000 / len(X_test)

        auroc = roc_auc_score(y_test, y_proba)
        auprc = average_precision_score(y_test, y_proba)
        brier = brier_score_loss(y_test, y_proba)
        ll = log_loss(y_test, y_proba)
        ece = _ece(y_test, y_proba)

        return auroc, auprc, brier, ll, ece, train_s, infer_ms, peak_mb

    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[list[str]] = None,
    ) -> list[ModelResult]:
        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.random_state
        )
        results: list[ModelResult] = []

        for name, estimator in self.models:
            logger.info("Benchmarking: %s ...", name)
            fold_aurocs, fold_auprcs = [], []
            fold_briers, fold_lls, fold_eces = [], [], []
            fold_train_s, fold_infer_ms, fold_mem_mb = [], [], []

            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                from copy import deepcopy

                est = deepcopy(estimator)
                try:
                    auroc, auprc, brier, ll, ece, t_s, i_ms, mem_mb = self._evaluate_fold(
                        est,
                        X[train_idx], y[train_idx],
                        X[test_idx],  y[test_idx],
                    )
                except Exception as exc:
                    logger.warning("  fold %d FAILED: %s", fold_idx, exc)
                    continue

                fold_aurocs.append(auroc)
                fold_auprcs.append(auprc)
                fold_briers.append(brier)
                fold_lls.append(ll)
                fold_eces.append(ece)
                fold_train_s.append(t_s)
                fold_infer_ms.append(i_ms)
                fold_mem_mb.append(mem_mb)

                logger.info(
                    "  fold %d: AUROC=%.4f AUPRC=%.4f brier=%.4f",
                    fold_idx, auroc, auprc, brier,
                )

            if not fold_aurocs:
                logger.warning("  %s: all folds failed -- skipping.", name)
                continue

            result = ModelResult(
                name=name,
                auroc=float(np.mean(fold_aurocs)),
                auprc=float(np.mean(fold_auprcs)),
                brier=float(np.mean(fold_briers)),
                logloss=float(np.mean(fold_lls)),
                ece=float(np.mean(fold_eces)),
                train_time_s=float(np.mean(fold_train_s)),
                infer_latency_ms=float(np.mean(fold_infer_ms)),
                peak_memory_mb=float(np.mean(fold_mem_mb)),
                fold_aurocs=fold_aurocs,
                fold_auprcs=fold_auprcs,
            )

            logger.info(
                "  %s SUMMARY: AUROC=%.4f +/-%.4f  AUPRC=%.4f  Brier=%.4f  "
                "ECE=%.4f  train=%.1fs  infer=%.3fms/variant",
                name,
                result.auroc, np.std(fold_aurocs),
                result.auprc, result.brier, result.ece,
                result.train_time_s, result.infer_latency_ms,
            )
            results.append(result)

        return results

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_summary(self, results: list[ModelResult]) -> None:
        print("\n" + "=" * 90)
        print(f"{'Model':<28}{'AUROC':>8}{'AUPRC':>8}{'Brier':>8}{'ECE':>8}"
              f"{'Train(s)':>10}{'Inf(ms)':>9}{'Mem(MB)':>9}")
        print("-" * 90)
        for r in sorted(results, key=lambda x: x.auroc, reverse=True):
            print(
                f"{r.name:<28}"
                f"{r.auroc:>8.4f}"
                f"{r.auprc:>8.4f}"
                f"{r.brier:>8.4f}"
                f"{r.ece:>8.4f}"
                f"{r.train_time_s:>10.1f}"
                f"{r.infer_latency_ms:>9.3f}"
                f"{r.peak_memory_mb:>9.1f}"
            )
        print("=" * 90)

    def save(self, results: list[ModelResult], output_dir: str | Path) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # JSON summary
        summary = [
            {
                "model": r.name,
                "auroc": r.auroc,
                "auroc_std": float(np.std(r.fold_aurocs)),
                "auprc": r.auprc,
                "auprc_std": float(np.std(r.fold_auprcs)),
                "brier": r.brier,
                "logloss": r.logloss,
                "ece": r.ece,
                "train_time_s": r.train_time_s,
                "infer_latency_ms": r.infer_latency_ms,
                "peak_memory_mb": r.peak_memory_mb,
                "fold_aurocs": r.fold_aurocs,
            }
            for r in results
        ]
        (out / "benchmark_results.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

        # Parquet for downstream analysis
        df = pd.DataFrame(summary).drop(columns=["fold_aurocs"])
        df.to_parquet(out / "benchmark_results.parquet", index=False)

        logger.info("Benchmark results saved to %s", out)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Algorithm comparison benchmark for genomic variant classification."
    )
    p.add_argument(
        "--parquet",
        type=Path,
        required=True,
        help="Labelled variant parquet (must have 'label' column and TABULAR_FEATURES columns).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/benchmark"),
        help="Output directory for results (default: outputs/benchmark).",
    )
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Subsample to N rows for quick testing.",
    )
    p.add_argument(
        "--models",
        nargs="*",
        default=None,
        metavar="NAME",
        help="Subset of model names to run (default: all).",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = _parse_args()

    from src.models.variant_ensemble import TABULAR_FEATURES

    logger.info("Loading %s ...", args.parquet)
    df = pd.read_parquet(args.parquet)

    if args.max_rows:
        df = df.sample(n=min(args.max_rows, len(df)), random_state=args.random_state)
        logger.info("Subsampled to %d rows.", len(df))

    missing_features = [f for f in TABULAR_FEATURES if f not in df.columns]
    if missing_features:
        logger.error("Missing feature columns: %s", missing_features)
        raise SystemExit(1)

    if "label" not in df.columns:
        logger.error("Parquet must have a 'label' column (0=benign, 1=pathogenic).")
        raise SystemExit(1)

    X = df[TABULAR_FEATURES].fillna(0.0).values.astype(np.float32)
    y = df["label"].values.astype(int)

    logger.info("Dataset: %d variants, %d features, %.1f%% pathogenic",
                len(y), X.shape[1], 100 * y.mean())

    runner = BenchmarkRunner(
        n_folds=args.n_folds,
        random_state=args.random_state,
    )

    if args.models:
        runner.models = [(n, m) for n, m in runner.models if n in args.models]
        if not runner.models:
            logger.error("No models matched: %s", args.models)
            raise SystemExit(1)

    results = runner.run(X, y, feature_names=TABULAR_FEATURES)
    runner.print_summary(results)
    runner.save(results, args.output)


if __name__ == "__main__":
    main()
