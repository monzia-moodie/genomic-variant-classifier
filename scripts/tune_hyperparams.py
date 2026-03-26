"""
scripts/tune_hyperparams.py
============================
Optuna-based hyperparameter tuning for LightGBM and XGBoost base classifiers.

Reads the gene-stratified training splits produced by run_phase2_eval.py and
optimises the chosen model's hyperparameters by maximising val-set AUROC.
The best parameters are written to a JSON file that run_phase2_eval.py (or
custom training code) can load to override the classifier defaults.

Usage
-----
    # Tune LightGBM (100 trials, ~20 min):
    python scripts/tune_hyperparams.py \\
        --model     lgbm \\
        --train-X   outputs/phase2_eval/splits/X_train.parquet \\
        --train-y   outputs/phase2_eval/splits/y_train.parquet \\
        --val-X     outputs/phase2_eval/splits/X_val.parquet \\
        --val-y     outputs/phase2_eval/splits/y_val.parquet \\
        --n-trials  100 \\
        --output    models/best_lgbm_params.json

    # Tune XGBoost (50 trials):
    python scripts/tune_hyperparams.py --model xgboost --n-trials 50

Exit codes
----------
    0   Success — best params written.
    1   Optimisation error.
    2   I/O error.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tune_hyperparams")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optuna hyperparameter tuning for LightGBM or XGBoost"
    )
    p.add_argument(
        "--model",
        choices=["lgbm", "xgboost"],
        default="lgbm",
        help="Model to tune (default: lgbm)",
    )
    p.add_argument(
        "--train-X",
        default="outputs/phase2_eval/splits/X_train.parquet",
        help="Training feature matrix (parquet)",
    )
    p.add_argument(
        "--train-y",
        default="outputs/phase2_eval/splits/y_train.parquet",
        help="Training labels (parquet with 'label' column)",
    )
    p.add_argument(
        "--val-X",
        default="outputs/phase2_eval/splits/X_val.parquet",
        help="Validation feature matrix (parquet)",
    )
    p.add_argument(
        "--val-y",
        default="outputs/phase2_eval/splits/y_val.parquet",
        help="Validation labels (parquet with 'label' column)",
    )
    p.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of Optuna trials (default: 100)",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Maximum wall-clock seconds for optimisation (default: unlimited)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: models/best_<model>_params.json)",
    )
    p.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel workers for each model fit (default: 1; use -1 for all cores)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    p.add_argument(
        "--max-train",
        type=int,
        default=None,
        help="Subsample training rows for faster trials (e.g. 50000)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_splits(
    train_x: str,
    train_y: str,
    val_x: str,
    val_y: str,
    max_train: int | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train = pd.read_parquet(train_x)
    X_val   = pd.read_parquet(val_x)

    y_raw = pd.read_parquet(train_y)
    y_train = (y_raw["label"] if "label" in y_raw.columns else y_raw.iloc[:, 0]).values.astype(int)

    y_raw = pd.read_parquet(val_y)
    y_val = (y_raw["label"] if "label" in y_raw.columns else y_raw.iloc[:, 0]).values.astype(int)

    if max_train and len(y_train) > max_train:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(y_train), max_train, replace=False)
        X_train = X_train.iloc[idx].reset_index(drop=True)
        y_train = y_train[idx]
        logger.info("Subsampled training set to %d", max_train)

    logger.info(
        "Splits: train=%d  val=%d  features=%d",
        len(y_train), len(y_val), X_train.shape[1],
    )
    return X_train.values, y_train, X_val.values, y_val


# ---------------------------------------------------------------------------
# Objective functions
# ---------------------------------------------------------------------------

def _lgbm_objective(
    trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_jobs: int,
    seed: int,
) -> float:
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    params = {
        "objective":        "binary",
        "metric":           "auc",
        "verbosity":        -1,
        "boosting_type":    "gbdt",
        "n_jobs":           n_jobs,
        "random_state":     seed,
        "n_estimators":     trial.suggest_int("n_estimators", 200, 2000, step=100),
        "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "num_leaves":       trial.suggest_int("num_leaves", 20, 300),
        "max_depth":        trial.suggest_int("max_depth", 3, 12),
        "min_child_samples":trial.suggest_int("min_child_samples", 5, 100),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "class_weight":     "balanced",
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    proba = model.predict_proba(X_val)[:, 1]
    return float(roc_auc_score(y_val, proba))


def _xgb_objective(
    trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_jobs: int,
    seed: int,
) -> float:
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    scale_pos_weight = float((y_train == 0).sum()) / max(float(y_train.sum()), 1.0)

    params = {
        "objective":        "binary:logistic",
        "eval_metric":      "auc",
        "verbosity":        0,
        "nthread":          n_jobs,
        "seed":             seed,
        "n_estimators":     trial.suggest_int("n_estimators", 200, 2000, step=100),
        "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "max_depth":        trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma":            trial.suggest_float("gamma", 1e-8, 5.0, log=True),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "scale_pos_weight": scale_pos_weight,
    }

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
        early_stopping_rounds=50,
    )
    proba = model.predict_proba(X_val)[:, 1]
    return float(roc_auc_score(y_val, proba))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    for path in [args.train_X, args.train_y, args.val_X, args.val_y]:
        if not Path(path).exists():
            logger.error("File not found: %s", path)
            return 2

    try:
        import optuna
    except ImportError:
        logger.error(
            "optuna is not installed.  Run: pip install optuna"
        )
        return 2

    try:
        X_train, y_train, X_val, y_val = load_splits(
            args.train_X, args.train_y,
            args.val_X,   args.val_y,
            max_train=args.max_train,
            seed=args.seed,
        )
    except Exception as exc:
        logger.error("Failed to load data: %s", exc)
        return 2

    objective_fn = _lgbm_objective if args.model == "lgbm" else _xgb_objective

    def objective(trial) -> float:
        return objective_fn(
            trial, X_train, y_train, X_val, y_val,
            n_jobs=args.n_jobs, seed=args.seed,
        )

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    logger.info(
        "Starting %d-trial %s optimisation (timeout=%s s) ...",
        args.n_trials, args.model.upper(), args.timeout or "∞",
    )
    try:
        study.optimize(
            objective,
            n_trials=args.n_trials,
            timeout=args.timeout,
            show_progress_bar=False,
        )
    except Exception as exc:
        logger.error("Optimisation failed: %s", exc)
        return 1

    best = study.best_trial
    logger.info(
        "Best trial #%d  AUROC=%.4f",
        best.number, best.value,
    )
    logger.info("Best params:")
    for k, v in best.params.items():
        logger.info("  %-25s %s", k + ":", v)

    output_path = Path(
        args.output or f"models/best_{args.model}_params.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "model":        args.model,
        "best_val_auroc": round(best.value, 6),
        "n_trials":     len(study.trials),
        "best_params":  best.params,
    }
    output_path.write_text(json.dumps(result, indent=2))
    logger.info("Best params written to %s", output_path)

    # Print top-10 trials for quick review
    top = sorted(study.trials, key=lambda t: t.value or 0.0, reverse=True)[:10]
    print(f"\n{'─'*54}")
    print(f"  Top-10 {args.model.upper()} trials")
    print(f"{'─'*54}")
    for i, t in enumerate(top, 1):
        print(f"  #{i:2d}  AUROC={t.value:.4f}  trial={t.number}")
    print(f"{'─'*54}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
