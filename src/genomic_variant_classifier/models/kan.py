"""
src/models/kan.py
=================
KAN (Kolmogorov-Arnold Network) classifier -- Phase 4B.

KANs replace fixed activation functions with learnable univariate spline
functions on the *edges* of the network.  Each edge learns its own smooth
nonlinear transformation, making the learned functions directly visualisable
and extending the SHAP feature attribution story: instead of SHAP values
telling you *how much* each feature contributes, the edge functions show
*how* they contribute (monotonic, U-shaped, threshold, etc.).

Reference: Liu et al., 2024 -- "KAN: Kolmogorov-Arnold Networks"
           https://arxiv.org/abs/2404.19756

Backend priority
----------------
1. pykan  -- original implementation from MIT CSAIL
   Install: pip install pykan
2. efficient-kan  -- faster GPU-friendly re-implementation
   Install: pip install efficient-kan
3. MLP fallback -- sklearn MLPClassifier; no splines but same interface

Architecture (default)
----------------------
Input (n_features) -> KAN(64, 32) -> output (1)
Spline degree: 3 (cubic), grid size: 5
Chosen to match the parameter count of the MLP fallback for fair comparison.

Usage
-----
    from src.models.kan import KANClassifier

    clf = KANClassifier(hidden_sizes=[64, 32], max_iter=200)
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)

    # Visualise learned edge functions (requires pykan)
    clf.plot_edge_functions()
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------
_KAN_BACKEND: Optional[str] = None

try:
    from kan import KAN as _PyKAN  # type: ignore[import]

    _KAN_BACKEND = "pykan"
    logger.debug("KAN backend: pykan")
except ImportError:
    pass

if _KAN_BACKEND is None:
    try:
        from efficient_kan import KAN as _EfficientKAN  # type: ignore[import]

        _KAN_BACKEND = "efficient-kan"
        logger.debug("KAN backend: efficient-kan")
    except ImportError:
        pass

if _KAN_BACKEND is None:
    logger.info(
        "KAN: neither 'pykan' nor 'efficient-kan' installed. "
        "Falling back to sklearn MLPClassifier. "
        "Install: pip install pykan"
    )


# ---------------------------------------------------------------------------
# KANClassifier
# ---------------------------------------------------------------------------

class KANClassifier(BaseEstimator, ClassifierMixin):
    """
    Sklearn-compatible KAN classifier for tabular genomic data.

    Falls back to an MLP (identical topology) when neither pykan nor
    efficient-kan is installed, ensuring the model is always usable.

    Parameters
    ----------
    hidden_sizes : list[int]
        Hidden layer widths, e.g. [64, 32].
    spline_degree : int
        Polynomial degree of the B-spline basis functions (default: 3 = cubic).
        Only used with pykan/efficient-kan backend.
    grid_size : int
        Number of spline grid intervals per edge (default: 5).
        Higher values = more expressive but slower to train.
    max_iter : int
        Maximum training epochs (default: 200).
    learning_rate : float
        Adam learning rate (default: 1e-3).
    batch_size : int
        Mini-batch size (default: 256).
    random_state : int
    scale : bool
        Standardise features before fitting (recommended; default: True).
    """

    def __init__(
        self,
        hidden_sizes: list[int] | None = None,
        spline_degree: int = 3,
        grid_size: int = 5,
        max_iter: int = 200,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        random_state: int = 42,
        scale: bool = True,
        max_fit_samples: int = 100_000,
    ) -> None:
        self.hidden_sizes = hidden_sizes or [64, 32]
        self.spline_degree = spline_degree
        self.grid_size = grid_size
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.random_state = random_state
        self.scale = scale
        self.max_fit_samples = max_fit_samples

    # ------------------------------------------------------------------
    # Internal fit helpers
    # ------------------------------------------------------------------

    def _fit_pykan(self, X: np.ndarray, y: np.ndarray) -> None:
        import torch

        self._backend_used = "pykan"

        # pykan materialises [n_samples, grid_size, n_features] in one shot.
        # At 1.2M samples this exceeds 17 GB. Subsample to max_fit_samples
        # using stratified sampling to preserve class balance.
        if X.shape[0] > self.max_fit_samples:
            rng = np.random.default_rng(self.random_state)
            pos_idx = np.where(y == 1)[0]
            neg_idx = np.where(y == 0)[0]
            n_pos = int(self.max_fit_samples * len(pos_idx) / len(y))
            n_neg = self.max_fit_samples - n_pos
            chosen = np.concatenate([
                rng.choice(pos_idx, min(n_pos, len(pos_idx)), replace=False),
                rng.choice(neg_idx, min(n_neg, len(neg_idx)), replace=False),
            ])
            rng.shuffle(chosen)
            X, y = X[chosen], y[chosen]
            logger.info(
                "KAN (pykan): subsampled %d → %d samples to avoid OOM "
                "(max_fit_samples=%d). Peak RAM ≈ %.1f GB.",
                len(chosen) + (X.shape[0] - len(chosen)),
                len(chosen),
                self.max_fit_samples,
                len(chosen) * self.grid_size * X.shape[1] * 4 / 1e9,
            )

        n_features = X.shape[1]
        widths = [n_features] + list(self.hidden_sizes) + [1]

        self._kan = _PyKAN(
            width=widths,
            grid=self.grid_size,
            k=self.spline_degree,
            seed=self.random_state,
        )

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)

        # pykan >= 0.2.x requires both train and test splits in the dataset dict
        dataset = {
            "train_input": X_t,
            "train_label": y_t,
            "test_input":  X_t,
            "test_label":  y_t,
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._kan.fit(
                dataset,
                opt="Adam",
                lr=self.learning_rate,
                steps=self.max_iter,
                loss_fn=torch.nn.BCEWithLogitsLoss(),
                metrics=None,
            )

    def _fit_efficient_kan(self, X: np.ndarray, y: np.ndarray) -> None:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset

        self._backend_used = "efficient-kan"
        n_features = X.shape[1]
        widths = [n_features] + list(self.hidden_sizes) + [1]

        self._kan = _EfficientKAN(widths, grid_size=self.grid_size, spline_order=self.spline_degree)
        optimizer = optim.Adam(self._kan.parameters(), lr=self.learning_rate)
        criterion = nn.BCEWithLogitsLoss()

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True)

        self._kan.train()
        for _ in range(self.max_iter):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(self._kan(xb), yb)
                loss.backward()
                optimizer.step()
        self._kan.eval()

    def _fit_mlp(self, X: np.ndarray, y: np.ndarray) -> None:
        self._backend_used = "mlp"
        self._mlp = MLPClassifier(
            hidden_layer_sizes=tuple(self.hidden_sizes),
            max_iter=self.max_iter,
            learning_rate_init=self.learning_rate,
            batch_size=self.batch_size,
            random_state=self.random_state,
        )
        self._mlp.fit(X, y)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KANClassifier":
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        if self.scale:
            self.scaler_ = StandardScaler()
            X = self.scaler_.fit_transform(X)
        else:
            self.scaler_ = None

        if _KAN_BACKEND == "pykan":
            self._fit_pykan(X, y)
        elif _KAN_BACKEND == "efficient-kan":
            self._fit_efficient_kan(X, y)
        else:
            self._fit_mlp(X, y)

        logger.info("KANClassifier fitted via %s backend.", getattr(self, "_backend_used", "mlp"))
        return self

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Return raw logits or probabilities, shape (n_samples,)."""
        if self.scale and self.scaler_ is not None:
            X = self.scaler_.transform(X)

        backend = getattr(self, "_backend_used", "mlp")

        if backend == "mlp":
            return self._mlp.predict_proba(X)[:, 1]

        import torch

        self._kan.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            logits = self._kan(X_t).squeeze(-1).numpy()
        # sigmoid to get probabilities
        return 1.0 / (1.0 + np.exp(-logits))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "classes_")
        p = self._predict_raw(X)
        return np.column_stack([1.0 - p, p])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def plot_edge_functions(self, **kwargs) -> None:
        """
        Visualise the learned spline edge functions.

        Only available with the pykan backend.  Calls kan.plot() which
        produces a matplotlib figure with one subplot per edge.
        """
        check_is_fitted(self, "classes_")
        if getattr(self, "_backend_used", "mlp") != "pykan":
            logger.warning("plot_edge_functions() is only available with the pykan backend.")
            return
        self._kan.plot(**kwargs)
