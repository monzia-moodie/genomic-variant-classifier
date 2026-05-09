"""
src/models/mc_dropout.py
========================
MC Dropout / Deep Ensemble uncertainty wrapper -- Phase 4C.

Produces calibrated probability estimates with epistemic and aleatoric
uncertainty decomposition for any sklearn-compatible classifier.

Two complementary approaches are implemented:

1. MCDropoutWrapper  (neural networks with dropout)
   Wraps a TF/Keras or PyTorch model that has dropout layers.  Keeps dropout
   active at inference time and runs T stochastic forward passes.
   -- Produces:
     mean_prob           -- mean of T softmax outputs (better calibrated)
     uncertainty_epistemic -- variance of predictions (model uncertainty; high
                             when the model is uncertain due to lack of training data)
     uncertainty_aleatoric -- mean binary entropy of predictions (data uncertainty;
                             high for inherently ambiguous variants regardless of data)

2. DeepEnsembleWrapper (any sklearn estimator)
   Trains M independently-initialised copies of a base estimator.  At
   inference, aggregates their predict_proba outputs.  Achieves identical
   uncertainty decomposition to MC Dropout with no architectural changes.
   Recommended when the base model does not use dropout (e.g. GBT, RF).

Usage
-----
    from src.models.mc_dropout import DeepEnsembleWrapper
    from lightgbm import LGBMClassifier

    ens = DeepEnsembleWrapper(
        base_estimator=LGBMClassifier(n_estimators=300),
        n_members=5,
    )
    ens.fit(X_train, y_train)

    mean_p, epistemic, aleatoric = ens.predict_with_uncertainty(X_test)
    # epistemic[i] high  => model is uncertain (consider retraining with more data)
    # aleatoric[i] high  => variant is intrinsically ambiguous (VUS)

Both wrappers expose a sklearn-compatible predict_proba() that returns the
mean probability, allowing them to be used as drop-in replacements in the
VariantEnsemble stacking pipeline.
"""

from __future__ import annotations

import logging
import warnings
from copy import deepcopy
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared uncertainty decomposition
# ---------------------------------------------------------------------------

def _decompose_uncertainty(
    probs_stack: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decompose predictive uncertainty from a stack of probability estimates.

    Parameters
    ----------
    probs_stack : ndarray of shape (T, n_samples)
        T probability estimates per sample (from T passes or T ensemble members).

    Returns
    -------
    mean_prob : (n_samples,)
    epistemic : (n_samples,)  -- variance across passes (reducible uncertainty)
    aleatoric : (n_samples,)  -- mean binary entropy (irreducible uncertainty)
    """
    mean_prob = probs_stack.mean(axis=0)
    epistemic = probs_stack.var(axis=0)

    eps = 1e-8
    clipped = np.clip(probs_stack, eps, 1.0 - eps)
    entropy_per_pass = -(clipped * np.log(clipped) + (1 - clipped) * np.log(1 - clipped))
    aleatoric = entropy_per_pass.mean(axis=0)

    return mean_prob, epistemic, aleatoric


# ---------------------------------------------------------------------------
# Deep Ensemble Wrapper
# ---------------------------------------------------------------------------

class DeepEnsembleWrapper(BaseEstimator, ClassifierMixin):
    """
    Deep Ensemble over M independently-trained sklearn-compatible estimators.

    Trains M clones of *base_estimator* with different random seeds and
    aggregates their predict_proba() at inference time.

    Parameters
    ----------
    base_estimator : sklearn estimator
        Any estimator with fit() and predict_proba().
    n_members : int
        Number of ensemble members (default: 5).  5-10 is sufficient for
        good uncertainty estimates; beyond 10 shows diminishing returns.
    random_state : int
        Base random seed.  Member k uses random_state + k.
    """

    def __init__(
        self,
        base_estimator: BaseEstimator,
        n_members: int = 5,
        random_state: int = 42,
    ) -> None:
        self.base_estimator = base_estimator
        self.n_members = n_members
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_kwargs) -> "DeepEnsembleWrapper":
        self.classes_ = np.unique(y)
        self.members_: list[BaseEstimator] = []

        for k in range(self.n_members):
            member = clone(self.base_estimator)
            # Inject random state if the estimator supports it
            if hasattr(member, "random_state"):
                member.set_params(random_state=self.random_state + k)
            elif hasattr(member, "seed"):
                member.set_params(seed=self.random_state + k)

            logger.info("DeepEnsemble: fitting member %d/%d ...", k + 1, self.n_members)
            member.fit(X, y, **fit_kwargs)
            self.members_.append(member)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return mean pathogenicity probabilities, shape (n_samples, 2)."""
        mean_p, _, _ = self.predict_with_uncertainty(X)
        return np.column_stack([1.0 - mean_p, mean_p])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def predict_with_uncertainty(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns
        -------
        mean_prob   : (n_samples,)  mean pathogenicity probability
        epistemic   : (n_samples,)  variance across members (model uncertainty)
        aleatoric   : (n_samples,)  mean entropy per member (data uncertainty)
        """
        check_is_fitted(self, "members_")
        probs_stack = np.array(
            [m.predict_proba(X)[:, 1] for m in self.members_]
        )  # (n_members, n_samples)
        return _decompose_uncertainty(probs_stack)

    def uncertainty_summary(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Return a dict with all uncertainty components for easy inspection."""
        mean_p, epistemic, aleatoric = self.predict_with_uncertainty(X)
        return {
            "mean_prob": mean_p,
            "epistemic_uncertainty": epistemic,
            "aleatoric_uncertainty": aleatoric,
            "total_uncertainty": epistemic + aleatoric,
        }


# ---------------------------------------------------------------------------
# MC Dropout Wrapper (neural networks)
# ---------------------------------------------------------------------------

class MCDropoutWrapper(BaseEstimator, ClassifierMixin):
    """
    MC Dropout wrapper for neural network classifiers with dropout layers.

    Wraps any estimator that exposes a ``_predict_proba_single_pass()``
    method (or falls back to predict_proba() with dropout toggled on when
    possible).  If the underlying model does not support dropout-at-inference,
    this wrapper degrades to a single deterministic pass with zero uncertainty.

    For most use cases, prefer DeepEnsembleWrapper which works with any
    sklearn estimator without requiring dropout layers.

    Parameters
    ----------
    base_estimator : estimator with predict_proba
    n_passes : int
        Number of stochastic forward passes (default: 50).
    random_state : int
    """

    def __init__(
        self,
        base_estimator: BaseEstimator,
        n_passes: int = 50,
        random_state: int = 42,
    ) -> None:
        self.base_estimator = base_estimator
        self.n_passes = n_passes
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_kwargs) -> "MCDropoutWrapper":
        self.classes_ = np.unique(y)
        self.estimator_ = clone(self.base_estimator)
        self.estimator_.fit(X, y, **fit_kwargs)
        self._supports_mc = hasattr(self.estimator_, "_predict_proba_single_pass")
        if not self._supports_mc:
            logger.warning(
                "MCDropoutWrapper: %s does not expose _predict_proba_single_pass(). "
                "Uncertainty estimates will be zero. "
                "Consider using DeepEnsembleWrapper instead.",
                type(self.estimator_).__name__,
            )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        mean_p, _, _ = self.predict_with_uncertainty(X)
        return np.column_stack([1.0 - mean_p, mean_p])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def predict_with_uncertainty(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        check_is_fitted(self, "estimator_")

        if not self._supports_mc:
            proba = self.estimator_.predict_proba(X)[:, 1]
            zeros = np.zeros_like(proba)
            return proba, zeros, zeros

        passes = []
        rng = np.random.default_rng(self.random_state)
        seeds = rng.integers(0, 2**31, size=self.n_passes)

        for seed in seeds:
            p = self.estimator_._predict_proba_single_pass(X, seed=int(seed))
            passes.append(p if p.ndim == 1 else p[:, 1])

        probs_stack = np.array(passes)  # (n_passes, n_samples)
        return _decompose_uncertainty(probs_stack)

    def uncertainty_summary(self, X: np.ndarray) -> dict[str, np.ndarray]:
        mean_p, epistemic, aleatoric = self.predict_with_uncertainty(X)
        return {
            "mean_prob": mean_p,
            "epistemic_uncertainty": epistemic,
            "aleatoric_uncertainty": aleatoric,
            "total_uncertainty": epistemic + aleatoric,
        }


# ---------------------------------------------------------------------------
# Convenience: annotate a DataFrame with uncertainty columns
# ---------------------------------------------------------------------------

def annotate_uncertainty(
    wrapper: DeepEnsembleWrapper | MCDropoutWrapper,
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Add uncertainty columns to a variant DataFrame in-place.

    Adds:
      pathogenicity_mean        -- mean ensemble probability
      uncertainty_epistemic     -- model (reducible) uncertainty
      uncertainty_aleatoric     -- data (irreducible) uncertainty
      uncertainty_total         -- sum of both
      uncertainty_flag          -- 'high' / 'medium' / 'low' for triage

    Parameters
    ----------
    wrapper : DeepEnsembleWrapper or MCDropoutWrapper (already fitted)
    df : DataFrame with *feature_cols* present
    feature_cols : list of feature column names
    """
    import pandas as pd  # local import to avoid circular at module level

    X = df[feature_cols].values.astype(np.float32)
    summary = wrapper.uncertainty_summary(X)

    df = df.copy()
    df["pathogenicity_mean"] = summary["mean_prob"]
    df["uncertainty_epistemic"] = summary["epistemic_uncertainty"]
    df["uncertainty_aleatoric"] = summary["aleatoric_uncertainty"]
    df["uncertainty_total"] = summary["total_uncertainty"]

    # Simple triage flag based on total uncertainty
    total = summary["total_uncertainty"]
    flags = np.where(total > 0.05, "high", np.where(total > 0.02, "medium", "low"))
    df["uncertainty_flag"] = flags

    return df


# avoid module-level pandas import (heavy) -- only needed in annotate_uncertainty
try:
    import pandas as pd
except ImportError:
    pass
