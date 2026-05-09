"""
scripts/run_benchmark.py
=========================
Run the Phase 4 algorithm comparison benchmark -- Step 1.

Uses the existing gene-stratified train/val/test split from Phase 2 training
so results are directly comparable to the published AUROC 0.9847 baseline.

Each model is trained on X_train and evaluated on X_val.
No cross-validation inside this script -- the held-out val set is the
evaluation set, matching exactly how the production ensemble was evaluated.

Output
------
  outputs/benchmark/benchmark_results.json   -- metrics for all models
  outputs/benchmark/benchmark_results.parquet
  outputs/benchmark/benchmark_comparison.png  -- bar chart (if matplotlib available)

Usage
-----
  .venv\\Scripts\\python scripts\\run_benchmark.py \\
      --splits-dir outputs/phase2_with_gnomad/splits \\
      --output     outputs/benchmark

  # Subset of models for a quick test
  .venv\\Scripts\\python scripts\\run_benchmark.py \\
      --splits-dir outputs/phase2_with_gnomad/splits \\
      --output     outputs/benchmark \\
      --models LightGBM XGBoost LogisticRegression
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import tracemalloc
from copy import deepcopy
from pathlib import Path

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
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_benchmark")


def _ece(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 15) -> float:
    frac_pos, mean_pred = calibration_curve(
        y_true, y_proba, n_bins=n_bins, strategy="uniform"
    )
    bins   = np.linspace(0, 1, n_bins + 1)
    counts = np.histogram(y_proba, bins=bins)[0]
    return float(sum(
        (c / len(y_true)) * abs(fp - mp)
        for fp, mp, c in zip(frac_pos, mean_pred, counts)
    ))


def _build_models():
    models = {}

    try:
        import lightgbm as lgb
        models["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05, num_leaves=63,
            class_weight="balanced", n_jobs=-1, verbose=-1,
        )
    except ImportError:
        pass

    try:
        import xgboost as xgb
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            eval_metric="logloss", n_jobs=-1, verbosity=0,
        )
    except ImportError:
        pass

    models["RandomForest"] = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=42,
    )
    models["GradientBoosting"] = GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42,
    )
    models["LogisticRegression"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, class_weight="balanced",
                                   max_iter=1000, random_state=42)),
    ])
    models["MLP"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(128, 64, 32),
                              max_iter=300, random_state=42)),
    ])

    try:
        from genomic_variant_classifier.models.kan import KANClassifier
        models["KAN"] = KANClassifier(hidden_sizes=[64, 32], max_iter=200, random_state=42)
    except Exception:
        pass

    try:
        import lightgbm as lgb
        from genomic_variant_classifier.models.mc_dropout import DeepEnsembleWrapper
        models["DeepEnsemble-LGB"] = DeepEnsembleWrapper(
            base_estimator=lgb.LGBMClassifier(
                n_estimators=300, learning_rate=0.05,
                class_weight="balanced", verbose=-1,
            ),
            n_members=5,
        )
    except Exception:
        pass

    return models


def _evaluate(name, model, X_train, y_train, X_val, y_val) -> dict:
    tracemalloc.start()
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_s = time.perf_counter() - t0
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    t_inf = time.perf_counter()
    y_proba = model.predict_proba(X_val)[:, 1]
    infer_ms = (time.perf_counter() - t_inf) * 1000 / len(X_val)

    auroc = roc_auc_score(y_val, y_proba)
    auprc = average_precision_score(y_val, y_proba)
    brier = brier_score_loss(y_val, y_proba)
    ll    = log_loss(y_val, y_proba)
    ece   = _ece(y_val, y_proba)

    logger.info(
        "  %-24s AUROC=%.4f  AUPRC=%.4f  Brier=%.4f  ECE=%.4f  "
        "train=%.1fs  infer=%.3fms/var  mem=%.0fMB",
        name, auroc, auprc, brier, ece,
        train_s, infer_ms, peak_mem / 1e6,
    )

    return {
        "model": name,
        "auroc": auroc,
        "auprc": auprc,
        "brier": brier,
        "logloss": ll,
        "ece": ece,
        "train_time_s": train_s,
        "infer_latency_ms": infer_ms,
        "peak_memory_mb": peak_mem / 1e6,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Algorithm benchmark on Phase 2 train/val split.")
    p.add_argument("--splits-dir", type=Path,
                   default=Path("outputs/phase2_with_gnomad/splits"))
    p.add_argument("--output",     type=Path, default=Path("outputs/benchmark"))
    p.add_argument("--models",     nargs="*", default=None,
                   help="Subset of model names to run.")
    args = p.parse_args()

    # -----------------------------------------------------------------------
    # Load splits
    # -----------------------------------------------------------------------
    logger.info("Loading splits from %s ...", args.splits_dir)
    X_train = pd.read_parquet(args.splits_dir / "X_train.parquet")
    y_train = pd.read_parquet(args.splits_dir / "y_train.parquet").squeeze()
    X_val   = pd.read_parquet(args.splits_dir / "X_val.parquet")
    y_val   = pd.read_parquet(args.splits_dir / "y_val.parquet").squeeze()

    from genomic_variant_classifier.models.variant_ensemble import TABULAR_FEATURES
    feat_cols = [c for c in TABULAR_FEATURES if c in X_train.columns]
    missing   = [c for c in TABULAR_FEATURES if c not in X_train.columns]
    if missing:
        logger.warning("Features absent from splits (will be zero-filled): %s", missing)
        for c in missing:
            X_train[c] = 0.0
            X_val[c]   = 0.0

    X_tr = X_train[TABULAR_FEATURES].fillna(0.0).values.astype(np.float32)
    y_tr = y_train.values.astype(int)
    X_vl = X_val[TABULAR_FEATURES].fillna(0.0).values.astype(np.float32)
    y_vl = y_val.values.astype(int)

    logger.info(
        "Train: %d variants (%.1f%% pathogenic)  "
        "Val: %d variants (%.1f%% pathogenic)  "
        "Features: %d",
        len(y_tr), 100 * y_tr.mean(),
        len(y_vl), 100 * y_vl.mean(),
        X_tr.shape[1],
    )

    # -----------------------------------------------------------------------
    # Run benchmark
    # -----------------------------------------------------------------------
    all_models = _build_models()
    if args.models:
        all_models = {k: v for k, v in all_models.items() if k in args.models}
        if not all_models:
            logger.error("No models matched: %s. Available: %s", args.models, list(_build_models()))
            raise SystemExit(1)

    logger.info("Running benchmark across %d models ...", len(all_models))
    results = []
    for name, model in all_models.items():
        logger.info("Training %s ...", name)
        try:
            r = _evaluate(name, deepcopy(model), X_tr, y_tr, X_vl, y_vl)
            results.append(r)
        except Exception as exc:
            logger.error("  %s FAILED: %s", name, exc)

    # -----------------------------------------------------------------------
    # Save + print
    # -----------------------------------------------------------------------
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "benchmark_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    df = pd.DataFrame(results)
    df.to_parquet(args.output / "benchmark_results.parquet", index=False)

    print("\n" + "=" * 96)
    print(f"{'Model':<26}{'AUROC':>8}{'AUPRC':>8}{'Brier':>8}{'ECE':>8}"
          f"{'Train(s)':>10}{'Inf(ms)':>9}{'Mem(MB)':>9}")
    print("-" * 96)
    for r in sorted(results, key=lambda x: x["auroc"], reverse=True):
        print(
            f"{r['model']:<26}"
            f"{r['auroc']:>8.4f}"
            f"{r['auprc']:>8.4f}"
            f"{r['brier']:>8.4f}"
            f"{r['ece']:>8.4f}"
            f"{r['train_time_s']:>10.1f}"
            f"{r['infer_latency_ms']:>9.3f}"
            f"{r['peak_memory_mb']:>9.1f}"
        )
    print("=" * 96)

    # Optional: comparison chart
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        df_sorted = df.sort_values("auroc", ascending=True)
        metrics = [("auroc", "AUROC (higher = better)"),
                   ("brier", "Brier Score (lower = better)"),
                   ("infer_latency_ms", "Inference Latency ms/variant (lower = better)")]
        colors = ["steelblue", "coral", "seagreen"]
        for ax, (col, title), color in zip(axes, metrics, colors):
            ax.barh(df_sorted["model"], df_sorted[col], color=color, alpha=0.8)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel(col)
        plt.tight_layout()
        plt.savefig(args.output / "benchmark_comparison.png", dpi=150, bbox_inches="tight")
        logger.info("Saved benchmark_comparison.png")
    except Exception as exc:
        logger.debug("matplotlib chart skipped: %s", exc)

    logger.info("Benchmark complete. Results in %s", args.output)


if __name__ == "__main__":
    main()
