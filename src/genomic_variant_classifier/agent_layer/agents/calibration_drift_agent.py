from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from logging import Logger
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CalibrationResult:
    timestamp: str
    ece_top_label: float
    mce_top_label: float
    per_class_ece: dict[str, float]
    delta_ece_vs_baseline: float
    severity: str
    n_samples: int
    bins_used: int


@dataclass
class CalibrationDriftAgent:
    """Expected/Maximum Calibration Error monitoring.

    Naeini, Cooper & Hauskrecht, "Obtaining Well Calibrated Probabilities
    Using Bayesian Binning", AAAI 2015. Guo, Pleiss, Sun & Weinberger,
    "On Calibration of Modern Neural Networks", ICML 2017 (15-bin ECE).
    """

    classes: tuple[str, ...]
    baseline_ece: float
    output_dir: Path
    n_bins: int = 15
    ece_amber: float = 0.02
    ece_red: float = 0.05
    mce_red: float = 0.20
    per_class_red: float = 0.10
    logger: Optional[Logger] = field(default=None, repr=False)

    def _binned_calibration(
        self, conf: np.ndarray, correct: np.ndarray
    ) -> tuple[float, float]:
        edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        ece = 0.0
        mce = 0.0
        n = len(conf)
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (
                (conf > lo) & (conf <= hi) if hi < 1.0 else (conf >= lo) & (conf <= hi)
            )
            count = int(mask.sum())
            if count == 0:
                continue
            acc = float(correct[mask].mean())
            avg_conf = float(conf[mask].mean())
            gap = abs(acc - avg_conf)
            ece += (count / n) * gap
            mce = max(mce, gap)
        return float(ece), float(mce)

    def detect(self, labeled_predictions: pd.DataFrame) -> CalibrationResult:
        """Compute ECE/MCE on a labeled chunk.

        Required columns:
          - 'true_class': str in self.classes
          - 'predicted_class': str in self.classes
          - 'p_<class>': posterior probability for each class
        """
        n = len(labeled_predictions)
        top_conf = (
            labeled_predictions[[f"p_{c}" for c in self.classes]].to_numpy().max(axis=1)
        )
        correct_top = (
            labeled_predictions["predicted_class"] == labeled_predictions["true_class"]
        ).to_numpy(dtype=float)
        ece_top, mce_top = self._binned_calibration(top_conf, correct_top)
        per_class: dict[str, float] = {}
        for c in self.classes:
            p = labeled_predictions[f"p_{c}"].to_numpy()
            y = (labeled_predictions["true_class"] == c).to_numpy(dtype=float)
            ece_c, _ = self._binned_calibration(p, y)
            per_class[c] = float(ece_c)
        delta = ece_top - self.baseline_ece
        if (
            ece_top >= self.ece_red
            or mce_top >= self.mce_red
            or any(v >= self.per_class_red for v in per_class.values())
        ):
            severity = "red"
        elif delta >= self.ece_amber:
            severity = "amber"
        else:
            severity = "green"
        return CalibrationResult(
            timestamp=datetime.now(timezone.utc).isoformat(),
            ece_top_label=ece_top,
            mce_top_label=mce_top,
            per_class_ece=per_class,
            delta_ece_vs_baseline=delta,
            severity=severity,
            n_samples=n,
            bins_used=self.n_bins,
        )
