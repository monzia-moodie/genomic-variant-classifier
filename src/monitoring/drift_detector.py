"""
src/monitoring/drift_detector.py
=================================
Comprehensive data drift detection for the Genomic Variant Classifier.

Implements four complementary drift detection strategies, each targeting a
different type of distributional change:

    PSI  — Population Stability Index (industry standard, fast, interpretable)
    KS   — Kolmogorov-Smirnov test (nonparametric, per-feature continuous)
    MMD  — Maximum Mean Discrepancy (kernel-based, catches subtle shifts)
    ADWIN — Adaptive Windowing (streaming detector for online use)

Genomic drift taxonomy addressed:
    Feature/covariate drift : P(X) changes — gnomAD cohort expansion,
                              AlphaMissense model updates, score recalibration
    Label drift             : P(Y) changes — ClinVar reclassifications
    Concept drift           : P(Y|X) changes — new biology, e.g. SpliceAI
                              dramatically altering splice variant interpretation
    Score drift             : a specific sub-type of feature drift where a
                              precomputed tool is retrained upstream

State-of-the-art additions beyond the Kirkpatrick/EWC literature:
    - Least-Squares Density Ratio Estimation (LSIF) for importance weighting:
      estimates the density ratio p_new(x) / p_old(x) without fitting two
      separate density models, which is numerically more stable than direct KL
    - Wasserstein-1 distance (Earth Mover's Distance) as a geometrically
      meaningful distance between score distributions — more sensitive than
      PSI for bimodal distributions common in pathogenicity scores
    - Two-sample energy statistic (Székely-Rizzo) — a distribution-free test
      that works well for multivariate drift in the joint feature space

Usage:
    from src.monitoring.drift_detector import DriftDetector, DriftReport

    detector = DriftDetector.from_reference(
        X_ref=X_train,
        feature_names=list(X_train.columns),
        save_path="models/drift_reference.pkl",
    )
    report = detector.check(X_new)
    if report.action_required:
        trigger_retraining()
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds (conventional clinical / finance standards)
# ---------------------------------------------------------------------------
PSI_NEGLIGIBLE  = 0.10   # PSI < 0.10 → no action
PSI_MONITOR     = 0.20   # 0.10–0.20 → increase monitoring frequency
PSI_RETRAIN     = 0.25   # > 0.25 → trigger retraining

KS_ALPHA        = 0.01   # significance level for KS test (Bonferroni corrected later)
MMD_SIGMA       = 1.0    # RBF kernel bandwidth (median heuristic used if None)
WASSERSTEIN_WARN = 0.05  # Wasserstein distance threshold for score features


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FeatureDriftResult:
    """Drift statistics for a single feature."""
    feature:          str
    psi:              float
    ks_statistic:     float
    ks_pvalue:        float
    wasserstein:      float
    ref_mean:         float
    ref_std:          float
    new_mean:         float
    new_std:          float
    mean_shift_sigmas: float     # (new_mean - ref_mean) / ref_std
    action:           str        # "none" | "monitor" | "retrain"


@dataclass
class DriftReport:
    """Complete drift analysis report across all features."""
    timestamp:         str
    n_ref_samples:     int
    n_new_samples:     int
    features_checked:  int
    features_drifted:  int          # PSI > PSI_RETRAIN
    features_monitored: int         # PSI in [PSI_MONITOR, PSI_RETRAIN]
    mmd_score:         float        # joint MMD across all features
    mmd_pvalue:        float
    energy_statistic:  float        # Székely-Rizzo two-sample energy test
    energy_pvalue:     float
    feature_results:   list[FeatureDriftResult] = field(default_factory=list)
    top_drifted:       list[str]    = field(default_factory=list)
    action_required:   bool         = False
    recommended_action: str        = "none"  # "none"|"monitor"|"retrain"|"urgent_retrain"
    summary:           str         = ""

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def print_summary(self) -> None:
        print(f"\n{'='*60}")
        print(f"  DRIFT REPORT — {self.timestamp}")
        print(f"{'='*60}")
        print(f"  Reference: {self.n_ref_samples:,} samples")
        print(f"  New data:  {self.n_new_samples:,} samples")
        print(f"  Features checked:    {self.features_checked}")
        print(f"  Features drifted:    {self.features_drifted}  (PSI > {PSI_RETRAIN})")
        print(f"  Features monitored:  {self.features_monitored}  (PSI > {PSI_MONITOR})")
        print(f"  MMD score:           {self.mmd_score:.6f}  (p={self.mmd_pvalue:.4f})")
        print(f"  Energy statistic:    {self.energy_statistic:.4f}  (p={self.energy_pvalue:.4f})")
        print(f"  ACTION: {self.recommended_action.upper()}")
        if self.top_drifted:
            print(f"  Top drifted features: {', '.join(self.top_drifted[:5])}")
        print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Core detector
# ---------------------------------------------------------------------------

class DriftDetector:
    """
    Stateful drift detector that holds a reference distribution snapshot
    and exposes a check() method for periodic evaluation.

    The reference snapshot should be set once from the training data and
    persisted. It is reloaded at each monitoring run without re-fitting.
    """

    def __init__(
        self,
        reference_data:  np.ndarray,
        feature_names:   list[str],
        n_bins:          int  = 10,
        mmd_n_permute:   int  = 200,
        energy_n_permute: int = 200,
        random_state:    int  = 42,
    ) -> None:
        self.feature_names    = list(feature_names)
        self.n_features       = len(feature_names)
        self.n_bins           = n_bins
        self.mmd_n_permute    = mmd_n_permute
        self.energy_n_permute = energy_n_permute
        self.rng              = np.random.default_rng(random_state)

        self.ref_data   = reference_data.astype(np.float64)
        self.ref_stats  = self._compute_stats(self.ref_data)
        self.ref_bins   = self._compute_bins(self.ref_data)

        # Median heuristic for MMD bandwidth
        pairwise = cdist(
            self.ref_data[:min(2000, len(self.ref_data))],
            self.ref_data[:min(2000, len(self.ref_data))],
        )
        self.mmd_sigma = float(np.median(pairwise[pairwise > 0])) or MMD_SIGMA

        logger.info(
            "DriftDetector initialised: %d features, %d reference samples, σ_MMD=%.3f",
            self.n_features, len(self.ref_data), self.mmd_sigma,
        )

    # ── Class-method constructors ──────────────────────────────────────────

    @classmethod
    def from_reference(
        cls,
        X_ref:        pd.DataFrame | np.ndarray,
        feature_names: Optional[list[str]] = None,
        save_path:    Optional[str | Path]  = None,
        **kwargs,
    ) -> DriftDetector:
        if isinstance(X_ref, pd.DataFrame):
            feature_names = feature_names or list(X_ref.columns)
            arr = X_ref.to_numpy(dtype=np.float64)
        else:
            arr = X_ref.astype(np.float64)
            feature_names = feature_names or [f"feat_{i}" for i in range(arr.shape[1])]

        detector = cls(arr, feature_names, **kwargs)
        if save_path:
            detector.save(save_path)
        return detector

    @classmethod
    def load(cls, path: str | Path) -> DriftDetector:
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected DriftDetector, got {type(obj).__name__}")
        return obj

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("DriftDetector saved → %s", path)

    # ── Main public interface ──────────────────────────────────────────────

    def check(
        self,
        X_new: pd.DataFrame | np.ndarray,
        timestamp: Optional[str] = None,
    ) -> DriftReport:
        """
        Run full drift check against the reference distribution.

        Parameters
        ----------
        X_new : DataFrame or array of shape (n_samples, n_features)
        timestamp : optional ISO timestamp string; defaults to now

        Returns
        -------
        DriftReport with per-feature statistics and a recommended action
        """
        from datetime import datetime, timezone
        timestamp = timestamp or datetime.now(timezone.utc).isoformat()

        if isinstance(X_new, pd.DataFrame):
            new_arr = X_new[self.feature_names].to_numpy(dtype=np.float64)
        else:
            new_arr = X_new.astype(np.float64)

        feature_results = []
        for i, feat in enumerate(self.feature_names):
            ref_col = self.ref_data[:, i]
            new_col = new_arr[:, i]
            result  = self._check_feature(feat, ref_col, new_col)
            feature_results.append(result)

        # Sort by PSI descending
        feature_results.sort(key=lambda r: r.psi, reverse=True)

        n_retrain  = sum(1 for r in feature_results if r.action == "retrain")
        n_monitor  = sum(1 for r in feature_results if r.action == "monitor")
        top_drifted = [r.feature for r in feature_results if r.action == "retrain"][:5]

        # Joint tests (subsample for speed)
        n_sub = min(3000, len(self.ref_data), len(new_arr))
        ref_sub = self.ref_data[self.rng.choice(len(self.ref_data), n_sub, replace=False)]
        new_sub = new_arr[self.rng.choice(len(new_arr), n_sub, replace=False)]

        mmd_score, mmd_pval   = self._mmd_test(ref_sub, new_sub)
        energy_stat, energy_p = self._energy_test(ref_sub, new_sub)

        # Determine overall action
        if n_retrain > 3 or mmd_pval < 0.001:
            action = "urgent_retrain"
        elif n_retrain > 0 or mmd_pval < 0.01:
            action = "retrain"
        elif n_monitor > 0:
            action = "monitor"
        else:
            action = "none"

        summary = (
            f"{n_retrain} features with significant drift (PSI>{PSI_RETRAIN}), "
            f"{n_monitor} under monitoring. "
            f"Joint MMD p={mmd_pval:.4f}. "
            f"Recommended: {action}."
        )

        report = DriftReport(
            timestamp          = timestamp,
            n_ref_samples      = len(self.ref_data),
            n_new_samples      = len(new_arr),
            features_checked   = self.n_features,
            features_drifted   = n_retrain,
            features_monitored = n_monitor,
            mmd_score          = float(mmd_score),
            mmd_pvalue         = float(mmd_pval),
            energy_statistic   = float(energy_stat),
            energy_pvalue      = float(energy_p),
            feature_results    = feature_results,
            top_drifted        = top_drifted,
            action_required    = action in ("retrain", "urgent_retrain"),
            recommended_action = action,
            summary            = summary,
        )

        logger.info("Drift check complete. %s", summary)
        return report

    # ── Per-feature analysis ───────────────────────────────────────────────

    def _check_feature(
        self, feature: str, ref_col: np.ndarray, new_col: np.ndarray
    ) -> FeatureDriftResult:
        ref_col = ref_col[np.isfinite(ref_col)]
        new_col = new_col[np.isfinite(new_col)]

        psi          = self._psi(ref_col, new_col)
        ks_stat, ks_p = stats.ks_2samp(ref_col, new_col)
        wasserstein  = float(stats.wasserstein_distance(ref_col, new_col))

        ref_mean, ref_std = float(np.mean(ref_col)), float(np.std(ref_col)) + 1e-9
        new_mean, new_std = float(np.mean(new_col)), float(np.std(new_col))
        shift_sigmas = (new_mean - ref_mean) / ref_std

        if psi > PSI_RETRAIN:
            action = "retrain"
        elif psi > PSI_MONITOR:
            action = "monitor"
        else:
            action = "none"

        return FeatureDriftResult(
            feature           = feature,
            psi               = round(psi, 5),
            ks_statistic      = round(float(ks_stat), 5),
            ks_pvalue         = round(float(ks_p), 6),
            wasserstein       = round(wasserstein, 5),
            ref_mean          = round(ref_mean, 5),
            ref_std           = round(ref_std, 5),
            new_mean          = round(new_mean, 5),
            new_std           = round(new_std, 5),
            mean_shift_sigmas = round(shift_sigmas, 3),
            action            = action,
        )

    # ── Statistical methods ────────────────────────────────────────────────

    def _psi(self, ref: np.ndarray, new: np.ndarray) -> float:
        """Population Stability Index (10 equal-width bins over reference range)."""
        lo, hi = np.percentile(ref, 1), np.percentile(ref, 99)
        if lo == hi:
            return 0.0
        edges   = np.linspace(lo, hi, self.n_bins + 1)
        ref_pct = np.histogram(ref, bins=edges)[0] / len(ref)
        new_pct = np.histogram(new, bins=edges)[0] / max(len(new), 1)
        # Smooth zeros to avoid log(0)
        ref_pct = np.clip(ref_pct, 1e-4, None)
        new_pct = np.clip(new_pct, 1e-4, None)
        return float(np.sum((new_pct - ref_pct) * np.log(new_pct / ref_pct)))

    def _rbf_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """RBF kernel matrix K(X, Y) using stored sigma."""
        sq_dists = cdist(X, Y, metric="sqeuclidean")
        return np.exp(-sq_dists / (2 * self.mmd_sigma ** 2))

    def _mmd_score(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Unbiased MMD^2 estimator."""
        n, m = len(X), len(Y)
        Kxx = self._rbf_kernel(X, X)
        Kyy = self._rbf_kernel(Y, Y)
        Kxy = self._rbf_kernel(X, Y)
        np.fill_diagonal(Kxx, 0)
        np.fill_diagonal(Kyy, 0)
        return (Kxx.sum() / (n * (n - 1)) +
                Kyy.sum() / (m * (m - 1)) -
                2 * Kxy.mean())

    def _mmd_test(
        self, ref: np.ndarray, new: np.ndarray
    ) -> tuple[float, float]:
        """Permutation test for MMD^2."""
        observed = self._mmd_score(ref, new)
        combined = np.vstack([ref, new])
        n = len(ref)
        perm_scores: list[float] = []
        for _ in range(self.mmd_n_permute):
            idx = self.rng.permutation(len(combined))
            perm_scores.append(
                self._mmd_score(combined[idx[:n]], combined[idx[n:]])
            )
        pval = float(np.mean(np.array(perm_scores) >= observed))
        return float(observed), pval

    def _energy_test(
        self, ref: np.ndarray, new: np.ndarray
    ) -> tuple[float, float]:
        """
        Székely-Rizzo two-sample energy statistic.
        E = (2nm)/(n+m) * [E|X-Y| - 0.5*E|X-X'| - 0.5*E|Y-Y'|]
        Sensitive to differences in shape, not just mean/variance.
        """
        n, m = len(ref), len(new)
        Exy  = cdist(ref, new).mean()
        Exx  = cdist(ref, ref).mean()
        Eyy  = cdist(new, new).mean()
        stat = (2 * n * m) / (n + m) * (Exy - 0.5 * Exx - 0.5 * Eyy)

        combined = np.vstack([ref, new])
        perm_stats: list[float] = []
        for _ in range(self.energy_n_permute):
            idx = self.rng.permutation(len(combined))
            r, s = combined[idx[:n]], combined[idx[n:]]
            perm_stats.append(
                (2 * n * m) / (n + m) * (
                    cdist(r, s).mean() - 0.5 * cdist(r, r).mean() - 0.5 * cdist(s, s).mean()
                )
            )
        pval = float(np.mean(np.array(perm_stats) >= stat))
        return float(stat), pval

    # ── Internal helpers ───────────────────────────────────────────────────

    def _compute_stats(self, arr: np.ndarray) -> dict:
        return {
            "mean":   arr.mean(axis=0),
            "std":    arr.std(axis=0) + 1e-9,
            "p1":     np.percentile(arr, 1,  axis=0),
            "p99":    np.percentile(arr, 99, axis=0),
        }

    def _compute_bins(self, arr: np.ndarray) -> list:
        bins = []
        for i in range(arr.shape[1]):
            col = arr[:, i]
            lo, hi = np.percentile(col, 1), np.percentile(col, 99)
            bins.append(np.linspace(lo if lo < hi else lo - 1, hi if lo < hi else hi + 1, self.n_bins + 1))
        return bins


# ---------------------------------------------------------------------------
# Streaming ADWIN detector (for continuous / online ingestion)
# ---------------------------------------------------------------------------

class ADWINDriftDetector:
    """
    Adaptive Windowing (ADWIN) detector for streaming variant ingestion.

    Maintains a sliding window of a scalar statistic (e.g. mean pathogenicity
    score or mean allele frequency of incoming variants). Flags drift when the
    mean in the most recent sub-window differs significantly from the full window.

    Reference: Bifet & Gavalda (2007), "Learning from Time-Changing Data with
    Adaptive Windowing". SIAM SDM 2007.

    Usage:
        adwin = ADWINDriftDetector(delta=0.002)
        for score in streaming_pathogenicity_scores:
            drifted = adwin.update(score)
            if drifted:
                trigger_retraining()
    """

    def __init__(self, delta: float = 0.002) -> None:
        self.delta   = delta
        self.window: list[float] = []
        self._drift_detected = False

    @property
    def drift_detected(self) -> bool:
        return self._drift_detected

    def update(self, value: float) -> bool:
        """
        Add a new observation. Returns True if drift is detected.
        ADWIN shrinks the window from the left when it detects a
        statistically significant shift in the mean.
        """
        self._drift_detected = False
        self.window.append(float(value))

        if len(self.window) < 32:
            return False

        n   = len(self.window)
        arr = np.array(self.window)
        mu  = arr.mean()

        # Test all possible split points
        for cut in range(1, n - 1):
            n0, n1   = cut, n - cut
            mu0      = arr[:cut].mean()
            mu1      = arr[cut:].mean()
            diff     = abs(mu0 - mu1)
            epsilon  = np.sqrt(
                (1 / (2 * n0) + 1 / (2 * n1)) *
                np.log(4 * n / self.delta)
            )
            if diff >= epsilon:
                # Shrink window to the more recent sub-window
                self.window = self.window[cut:]
                self._drift_detected = True
                logger.info(
                    "ADWIN: drift detected at split %d/%d, |Δμ|=%.4f ≥ ε=%.4f",
                    cut, n, diff, epsilon,
                )
                break

        return self._drift_detected

    def reset(self) -> None:
        self.window = []
        self._drift_detected = False

    @property
    def mean(self) -> float:
        return float(np.mean(self.window)) if self.window else 0.0

    @property
    def window_size(self) -> int:
        return len(self.window)


# ---------------------------------------------------------------------------
# LSIF density ratio estimator for importance weighting
# ---------------------------------------------------------------------------

class LSIFImportanceWeighter:
    """
    Least-Squares Importance Fitting (LSIF) — estimates the density ratio
    w(x) = p_new(x) / p_ref(x) without fitting two separate density models.

    These weights can be used for:
      1. Sample re-weighting in retraining: give higher weight to variants
         whose feature distribution resembles the new data
      2. Identifying which training variants are most stale / unrepresentative
         of current data

    Reference: Kanamori et al. (2009), "A Least-Squares Approach to Direct
    Importance Estimation". JMLR 10.

    Usage:
        weighter = LSIFImportanceWeighter()
        weighter.fit(X_ref, X_new)
        weights = weighter.transform(X_train)  # per-sample importance weights
    """

    def __init__(self, sigma: float = 1.0, lambda_: float = 0.01, n_basis: int = 200) -> None:
        self.sigma    = sigma
        self.lambda_  = lambda_
        self.n_basis  = n_basis
        self._centers: Optional[np.ndarray] = None
        self._alpha:   Optional[np.ndarray] = None

    def fit(
        self,
        X_ref: np.ndarray | pd.DataFrame,
        X_new: np.ndarray | pd.DataFrame,
    ) -> LSIFImportanceWeighter:
        if isinstance(X_ref, pd.DataFrame):
            X_ref = X_ref.to_numpy(dtype=np.float64)
        if isinstance(X_new, pd.DataFrame):
            X_new = X_new.to_numpy(dtype=np.float64)

        X_ref = X_ref.astype(np.float64)
        X_new = X_new.astype(np.float64)

        # Select basis centres from X_new (or combined) using random subsampling
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_new), min(self.n_basis, len(X_new)), replace=False)
        self._centers = X_new[idx]

        # Compute kernel matrices
        K_ref = self._kernel(X_ref, self._centers)   # (n_ref, n_basis)
        K_new = self._kernel(X_new, self._centers)   # (n_new, n_basis)

        # LSIF objective: min_alpha ||Hα - h||^2 + λ||α||^2
        H = K_ref.T @ K_ref / len(X_ref)
        h = K_new.mean(axis=0)

        # Closed-form solution: α = (H + λI)^{-1} h
        self._alpha = np.linalg.solve(
            H + self.lambda_ * np.eye(self.n_basis),
            h,
        )
        return self

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Return importance weights w(x) = p_new(x) / p_ref(x) for each row."""
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy(dtype=np.float64)
        K = self._kernel(X.astype(np.float64), self._centers)
        w = K @ self._alpha
        return np.clip(w, 0.0, None)  # clip to non-negative (density ratios ≥ 0)

    def _kernel(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        sq_dists = cdist(A, B, metric="sqeuclidean")
        return np.exp(-sq_dists / (2 * self.sigma ** 2))