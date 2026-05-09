"""
src/models/catboost_wrapper.py
================================
Sklearn-compatible CatBoost wrapper for the genomic variant ensemble.

Why CatBoost is architecturally distinct from XGBoost and LightGBM
-------------------------------------------------------------------
XGBoost and LightGBM are both leaf-wise (best-first) gradient boosting
implementations. Their tree structures and regularisation mechanisms are
similar enough that they tend to make correlated errors on the same variants.

CatBoost uses *ordered boosting* with *symmetric (oblivious) trees*:

  Ordered boosting: each tree is trained on a permutation of the data where
  the target statistics for example i are computed only from examples that
  appear before i in the permutation. This eliminates the prediction shift
  bias that arises when you use the same examples to build and evaluate trees,
  which is particularly important for rare-variant classification where ClinVar
  pathogenic labels are sparse and label leakage is easy to introduce.

  Symmetric trees: every split in a given depth uses the same feature and
  threshold across all leaves at that depth. This is less flexible than
  leaf-wise growth but dramatically reduces overfitting on the
  tens-of-thousands scale that expert-reviewed ClinVar data occupies.

  Native categorical handling: CatBoost encodes categorical features using
  ordered target statistics — the mean label for that category, computed
  only on examples seen earlier in the training permutation. This avoids
  the target leakage that arises from standard label encoding + mean-target
  encoding on imbalanced clinical data (12% pathogenic).

Genomic categorical columns used natively
------------------------------------------
  gene_symbol    — CatBoost learns gene-level pathogenicity priors
                   beyond what n_pathogenic_in_gene captures linearly.
                   Rare disease genes with few training examples benefit most.
  consequence    — Raw VEP consequence string, not the integer severity.
                   CatBoost can learn non-linear interactions: e.g. splice_donor
                   in a constrained gene is more pathogenic than the same
                   consequence in an unconstrained gene.
  chrom          — Chromosome-level effects (chrX hemizygous, chrMT mitochondrial
                   inheritance) that are not fully captured by the is_autosome flags.
  review_status  — ClinVar review tier as a category: CatBoost can learn
                   that a variant from an expert panel with tier-1 status
                   labelled "Benign" is much more trustworthy than tier-3.

Integration with existing pipeline
------------------------------------
The wrapper is sklearn-compatible and drops into the base_estimators dict in
VariantEnsemble with one change: the training loop must pass X_tab (DataFrame)
rather than X_tab.values (numpy) to CatBoost so that column names are visible.
The loop in variant_ensemble.py already has a CNN special-case; CatBoost is
added as a second special case in the same pattern.

SHAP compatibility
-------------------
CatBoost's get_feature_importance(type='ShapValues') returns an (n, n_features+1)
array where the last column is the expected value. This is compatible with the
existing SHAP pipeline in InferencePipeline.feature_importances().

Calibration
-----------
CatBoost probabilities are already well-calibrated relative to XGBoost/LightGBM
due to ordered boosting, but this wrapper still accepts an optional calibration
mode ('platt' | 'isotonic' | None) for consistency with the rest of the ensemble.

Usage:
    from src.models.catboost_wrapper import CatBoostVariantClassifier

    clf = CatBoostVariantClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        task_type="CPU",    # "GPU" when GPU available
        cat_feature_names=["gene_symbol", "consequence", "chrom"],
    )
    clf.fit(X_tab_df, y)             # X_tab_df is a DataFrame
    proba = clf.predict_proba(X_tab_df)   # shape (n, 2)
    shap_vals = clf.shap_values(X_tab_df) # shape (n, n_features)
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)

# Categorical columns expected to be present in X_tab when available.
# These must be string dtype — numeric encodings get no benefit from
# CatBoost's ordered target statistics.
DEFAULT_CAT_FEATURE_NAMES: list[str] = [
    "gene_symbol",       # HGNC symbol — gene-level pathogenicity prior
    "consequence",       # Raw VEP consequence term (e.g. "missense_variant")
    "chrom",             # Chromosome (string: "1"…"22", "X", "Y", "MT")
    "review_status",     # ClinVar expert review tier string
]

# CatBoost default hyperparameters tuned for genomic variant data.
# These are starting points; Optuna tuning should be run on the full
# 64-feature dataset before publication.
_CB_DEFAULTS = dict(
    iterations          = 1000,
    learning_rate       = 0.05,
    depth               = 6,
    l2_leaf_reg         = 3.0,
    random_strength     = 1.0,
    bagging_temperature = 1.0,
    border_count        = 128,
    loss_function       = "Logloss",
    eval_metric         = "AUC",
    auto_class_weights  = "Balanced",   # handles 12% pathogenic imbalance
    bootstrap_type      = "Bayesian",   # best for GPU; 'MVS' for CPU
    random_seed         = 42,
    verbose             = 0,
)


class CatBoostVariantClassifier(BaseEstimator, ClassifierMixin):
    """
    Sklearn-compatible CatBoost classifier for genomic variant pathogenicity.

    Accepts either a pandas DataFrame or a numpy array.  When a DataFrame is
    supplied, any columns listed in cat_feature_names that are present and
    have object/string dtype are used as native CatBoost categorical features.
    If those columns are absent or the input is a numpy array, the model falls
    back to treating all features as numerical — still valid, just without the
    native-categorical advantage.

    Parameters
    ----------
    iterations        : number of boosting rounds
    learning_rate     : shrinkage rate
    depth             : symmetric tree depth (6 is CatBoost's sweet spot)
    task_type         : "CPU" or "GPU"
    cat_feature_names : list of column names to treat as categorical
    calibrate         : None | "platt" | "isotonic" — post-hoc calibration
    early_stopping_rounds : stop if eval metric doesn't improve (requires eval_set)
    **kwargs          : passed directly to CatBoostClassifier
    """

    def __init__(
        self,
        iterations:             int           = 1000,
        learning_rate:          float         = 0.05,
        depth:                  int           = 6,
        l2_leaf_reg:            float         = 3.0,
        random_strength:        float         = 1.0,
        bagging_temperature:    float         = 1.0,
        border_count:           int           = 128,
        auto_class_weights:     str           = "Balanced",
        task_type:              str           = "CPU",
        cat_feature_names:      Optional[list[str]] = None,
        calibrate:              Optional[str] = None,
        early_stopping_rounds:  Optional[int] = 50,
        random_seed:            int           = 42,
        verbose:                int           = 0,
        **kwargs,
    ) -> None:
        self.iterations           = iterations
        self.learning_rate        = learning_rate
        self.depth                = depth
        self.l2_leaf_reg          = l2_leaf_reg
        self.random_strength      = random_strength
        self.bagging_temperature  = bagging_temperature
        self.border_count         = border_count
        self.auto_class_weights   = auto_class_weights
        self.task_type            = task_type
        self.cat_feature_names    = cat_feature_names if cat_feature_names is not None \
                                    else DEFAULT_CAT_FEATURE_NAMES
        self.calibrate            = calibrate
        self.early_stopping_rounds = early_stopping_rounds
        self.random_seed          = random_seed
        self.verbose              = verbose
        self.kwargs               = kwargs

        self.classes_             = np.array([0, 1])
        self._model               = None
        self._calibrator          = None
        self._cat_indices:        list[int] = []
        self._feature_names:      list[str] = []

    # ── Fit ────────────────────────────────────────────────────────────────

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray | pd.Series,
        eval_set: Optional[tuple] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> CatBoostVariantClassifier:
        """
        Fit the CatBoost model.

        Parameters
        ----------
        X            : DataFrame (preferred) or numpy array of shape (n, p)
        y            : binary labels (1 = pathogenic, 0 = benign)
        eval_set     : optional (X_val, y_val) tuple for early stopping
        sample_weight: optional per-sample weights (EWC / LSIF weights)
        """
        try:
            from catboost import CatBoostClassifier, Pool
        except ImportError:
            raise ImportError(
                "catboost is not installed. Run: pip install catboost"
            )

        X, cat_indices, feature_names = self._prepare_input(X)
        self._cat_indices   = cat_indices
        self._feature_names = feature_names

        y_arr = np.asarray(y, dtype=np.int32)

        train_pool = Pool(
            data           = X,
            label          = y_arr,
            cat_features   = cat_indices,
            feature_names  = feature_names,
            weight         = sample_weight,
        )

        eval_pool = None
        if eval_set is not None:
            X_eval, y_eval = eval_set
            X_eval, _, _ = self._prepare_input(X_eval)
            eval_pool = Pool(
                data          = X_eval,
                label         = np.asarray(y_eval, dtype=np.int32),
                cat_features  = cat_indices,
                feature_names = feature_names,
            )

        cb_params = dict(
            iterations         = self.iterations,
            learning_rate      = self.learning_rate,
            depth              = self.depth,
            l2_leaf_reg        = self.l2_leaf_reg,
            random_strength    = self.random_strength,
            bagging_temperature= self.bagging_temperature,
            border_count       = self.border_count,
            loss_function      = "Logloss",
            eval_metric        = "AUC",
            auto_class_weights = self.auto_class_weights,
            task_type          = self.task_type,
            random_seed        = self.random_seed,
            verbose            = self.verbose,
            **self.kwargs,
        )
        if self.early_stopping_rounds and eval_pool is not None:
            cb_params["early_stopping_rounds"] = self.early_stopping_rounds

        self._model = CatBoostClassifier(**cb_params)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.fit(
                train_pool,
                eval_set=eval_pool,
                verbose=self.verbose,
            )

        logger.info(
            "CatBoost fit: best_iteration=%d, categorical_features=%d/%d",
            self._model.best_iteration_ or self.iterations,
            len(cat_indices),
            len(feature_names),
        )

        # Optional post-hoc calibration
        if self.calibrate:
            self._fit_calibrator(train_pool, y_arr)

        return self

    # ── Predict ────────────────────────────────────────────────────────────

    def predict_proba(
        self, X: pd.DataFrame | np.ndarray
    ) -> np.ndarray:
        """Returns shape (n, 2): columns are [P(benign), P(pathogenic)]."""
        check_is_fitted(self, "_model")
        X_prep, _, _ = self._prepare_input(X)

        try:
            from catboost import Pool
            pool  = Pool(data=X_prep, cat_features=self._cat_indices,
                         feature_names=self._feature_names)
            proba = self._model.predict_proba(pool)
        except Exception:
            proba = self._model.predict_proba(X_prep)

        if self._calibrator is not None:
            p_pos     = self._calibrator.transform(proba[:, 1].clip(1e-7, 1 - 1e-7))
            proba     = np.column_stack([1 - p_pos, p_pos])

        return proba

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    # ── SHAP / feature importance ───────────────────────────────────────────

    def shap_values(
        self, X: pd.DataFrame | np.ndarray
    ) -> np.ndarray:
        """
        Return SHAP values for the pathogenic class (column 1).

        CatBoost's native get_feature_importance(type='ShapValues') returns
        shape (n, n_features + 1) where the last column is the bias term.
        We strip the bias and return shape (n, n_features).
        """
        check_is_fitted(self, "_model")
        X_prep, _, _ = self._prepare_input(X)

        try:
            from catboost import Pool
            pool = Pool(data=X_prep, cat_features=self._cat_indices,
                        feature_names=self._feature_names)
            shap = self._model.get_feature_importance(pool, type="ShapValues")
        except Exception as e:
            logger.warning("CatBoost SHAP computation failed: %s", e)
            return np.zeros((len(X_prep), len(self._feature_names)))

        # shap shape: (n, n_features + 1) — last col is bias
        return shap[:, :-1]

    def feature_importances_(self) -> np.ndarray:
        """Model-level feature importances (PredictionValuesChange)."""
        check_is_fitted(self, "_model")
        return self._model.get_feature_importance()

    def top_shap_features(
        self, X: pd.DataFrame | np.ndarray, n: int = 10
    ) -> list[tuple[str, float]]:
        """Return top-n features by mean absolute SHAP value."""
        shap = self.shap_values(X)
        mean_abs = np.abs(shap).mean(axis=0)
        order    = np.argsort(mean_abs)[::-1][:n]
        return [(self._feature_names[i], float(mean_abs[i])) for i in order]

    # ── Serialisation ───────────────────────────────────────────────────────

    def save_catboost_model(self, path: str | Path) -> None:
        """Save the native CatBoost model file (independent of joblib)."""
        check_is_fitted(self, "_model")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(path))
        logger.info("CatBoost native model saved to %s", path)

    @classmethod
    def load_catboost_model(
        cls, path: str | Path, **kwargs
    ) -> CatBoostVariantClassifier:
        """Load from a native CatBoost model file."""
        from catboost import CatBoostClassifier
        inst = cls(**kwargs)
        inst._model = CatBoostClassifier()
        inst._model.load_model(str(path))
        inst._feature_names = inst._model.feature_names_ or []
        inst._cat_indices    = []
        return inst

    # ── Internal helpers ───────────────────────────────────────────────────

    def _prepare_input(
        self, X: pd.DataFrame | np.ndarray
    ) -> tuple[pd.DataFrame | np.ndarray, list[int], list[str]]:
        """
        Identify categorical feature indices and ensure string dtype.

        If X is a numpy array: no categoricals, return as-is.
        If X is a DataFrame: any column in cat_feature_names that is
        present and has object/string dtype is treated as categorical.
        Columns in cat_feature_names that are numeric are NOT forced to
        categorical — CatBoost will use their numerical values.
        """
        if not isinstance(X, pd.DataFrame):
            n_feats = X.shape[1] if X.ndim == 2 else 1
            feature_names = (
                self._feature_names if len(self._feature_names) == n_feats
                else [f"f{i}" for i in range(n_feats)]
            )
            return X, [], feature_names

        feature_names = list(X.columns)
        cat_indices   = []

        for i, col in enumerate(feature_names):
            if col not in self.cat_feature_names:
                continue
            if X[col].dtype == object or pd.api.types.is_string_dtype(X[col]):
                # Fill NA with a sentinel string so CatBoost doesn't error
                if X[col].isna().any():
                    X = X.copy()
                    X[col] = X[col].fillna("__missing__")
                cat_indices.append(i)

        return X, cat_indices, feature_names

    def _fit_calibrator(self, train_pool, y: np.ndarray) -> None:
        """Fit isotonic or Platt calibrator on training predictions."""
        raw_proba = self._model.predict_proba(train_pool)[:, 1]
        if self.calibrate == "isotonic":
            from sklearn.isotonic import IsotonicRegression
            self._calibrator = IsotonicRegression(out_of_bounds="clip")
            self._calibrator.fit(raw_proba, y)
        elif self.calibrate == "platt":
            from sklearn.linear_model import LogisticRegression
            self._calibrator = LogisticRegression(C=1.0, max_iter=1000)
            self._calibrator.fit(raw_proba.reshape(-1, 1), y)
            # Wrap to match IsotonicRegression interface
            _inner = self._calibrator
            class _PlattWrapper:
                def transform(self, p):
                    return _inner.predict_proba(
                        np.asarray(p).reshape(-1, 1)
                    )[:, 1]
            self._calibrator = _PlattWrapper()
        logger.info("CatBoost calibrator fitted (%s).", self.calibrate)


# ---------------------------------------------------------------------------
# Optuna hyperparameter search for CatBoost
# ---------------------------------------------------------------------------

def catboost_optuna_search(
    X_train: pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    n_trials:    int  = 50,
    n_folds:     int  = 3,
    task_type:   str  = "CPU",
    cat_feature_names: Optional[list[str]] = None,
    random_state: int = 42,
) -> dict:
    """
    Optuna hyperparameter search for CatBoostVariantClassifier.

    Searches the parameter space most impactful for genomic variant data:
      depth, learning_rate, l2_leaf_reg, random_strength, bagging_temperature

    Parameters
    ----------
    X_train : feature DataFrame (with categorical columns present)
    y_train : binary labels
    n_trials : number of Optuna trials
    n_folds  : stratified k-fold for CV
    task_type: "CPU" or "GPU"

    Returns
    -------
    best_params dict suitable for CatBoostVariantClassifier(**best_params)
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError("optuna is not installed. Run: pip install optuna")

    from sklearn.model_selection import StratifiedKFold, cross_val_score

    cat_feature_names = cat_feature_names or DEFAULT_CAT_FEATURE_NAMES
    y_arr = np.asarray(y_train)

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            iterations         = trial.suggest_int("iterations", 300, 2000, step=100),
            learning_rate      = trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
            depth              = trial.suggest_int("depth", 4, 10),
            l2_leaf_reg        = trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            random_strength    = trial.suggest_float("random_strength", 0.1, 5.0),
            bagging_temperature= trial.suggest_float("bagging_temperature", 0.0, 2.0),
            border_count       = trial.suggest_categorical("border_count", [64, 128, 254]),
            task_type          = task_type,
            cat_feature_names  = cat_feature_names,
            verbose            = 0,
            random_seed        = random_state,
        )
        clf = CatBoostVariantClassifier(**params)
        cv  = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        scores = cross_val_score(clf, X_train, y_arr, cv=cv, scoring="roc_auc", n_jobs=1)
        return scores.mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    best["task_type"]          = task_type
    best["cat_feature_names"]  = cat_feature_names
    best["verbose"]            = 0
    best["random_seed"]        = random_state
    logger.info("Optuna best AUROC: %.4f — params: %s", study.best_value, best)
    return best

