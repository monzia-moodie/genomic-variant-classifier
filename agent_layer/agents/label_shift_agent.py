from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from logging import Logger
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class LabelShiftResult:
    """Output of one BBSE run."""

    timestamp: str
    estimated_p_test: dict[str, float]
    p_train: dict[str, float]
    chi_square_stat: float
    chi_square_p_value: float
    max_abs_class_shift: float
    rlls_used: bool
    severity: str  # green | amber | red
    n_predictions: int


@dataclass
class LabelShiftAgent:
    """BBSE label-shift detection without test labels.

    Lipton, Wang & Smola, "Detecting and Correcting for Label Shift
    with Black Box Predictors", ICML 2018, arXiv:1802.03916.
    Regularized variant (RLLS): Azizzadenesheli et al., ICLR 2019.
    """

    reference_confusion: np.ndarray  # shape (K, K), rows=true, cols=pred
    classes: tuple[str, ...]  # ("B", "LB", "VUS", "LP", "P")
    p_train: np.ndarray  # shape (K,)
    output_dir: Path
    chi2_alpha: float = 0.01
    abs_shift_amber: float = 0.05
    abs_shift_red: float = 0.10
    rlls_lambda: float = 1e-3
    logger: Optional[Logger] = field(default=None, repr=False)

    def estimate_test_marginal(
        self, predicted_labels: np.ndarray
    ) -> tuple[np.ndarray, bool]:
        """Solve Ĉ p̂(y) = q̂(ŷ); fall back to RLLS if Ĉ near-singular."""
        k = len(self.classes)
        q_hat = np.bincount(predicted_labels, minlength=k).astype(float)
        q_hat = q_hat / max(q_hat.sum(), 1.0)
        c = self.reference_confusion.astype(float)
        cond = np.linalg.cond(c)
        if cond < 1e6:
            try:
                p_hat = np.linalg.solve(c, q_hat)
                return (
                    np.clip(p_hat, 1e-9, None) / np.clip(p_hat.sum(), 1e-9, None),
                    False,
                )
            except np.linalg.LinAlgError:
                pass
        # RLLS regularized
        ctc = c.T @ c + self.rlls_lambda * np.eye(k)
        p_hat = np.linalg.solve(ctc, c.T @ q_hat)
        return np.clip(p_hat, 1e-9, None) / np.clip(p_hat.sum(), 1e-9, None), True

    def detect(self, prediction_log: pd.DataFrame) -> LabelShiftResult:
        """Run BBSE on a window of production predictions.

        Expects columns: ['predicted_class'] with values in self.classes.
        """
        idx = {c: i for i, c in enumerate(self.classes)}
        preds = prediction_log["predicted_class"].map(idx).to_numpy(dtype=int)
        p_hat, used_rlls = self.estimate_test_marginal(preds)
        # χ² test on observed-vs-expected predicted marginals under H0: p_test = p_train.
        n = len(preds)
        expected = (self.reference_confusion @ self.p_train) * n
        observed = np.bincount(preds, minlength=len(self.classes)).astype(float)
        expected = np.where(expected < 1e-6, 1e-6, expected)
        chi2 = float(((observed - expected) ** 2 / expected).sum())
        p_value = float(1.0 - stats.chi2.cdf(chi2, df=len(self.classes) - 1))
        max_shift = float(np.max(np.abs(p_hat - self.p_train)))
        if p_value < self.chi2_alpha and max_shift >= self.abs_shift_red:
            severity = "red"
        elif p_value < self.chi2_alpha and max_shift >= self.abs_shift_amber:
            severity = "amber"
        else:
            severity = "green"
        return LabelShiftResult(
            timestamp=datetime.now(timezone.utc).isoformat(),
            estimated_p_test=dict(zip(self.classes, map(float, p_hat))),
            p_train=dict(zip(self.classes, map(float, self.p_train))),
            chi_square_stat=chi2,
            chi_square_p_value=p_value,
            max_abs_class_shift=max_shift,
            rlls_used=used_rlls,
            severity=severity,
            n_predictions=int(n),
        )

    def emit_hypothesis(self, result: LabelShiftResult, hypothesis_dir: Path) -> Path:
        """Write a hypothesis stub when severity != green; never auto-retrain."""
        if result.severity == "green":
            return Path()
        date = datetime.now(timezone.utc).strftime("%Y%m%d")
        path = hypothesis_dir / f"HYP_drift_{date}_label_marginal.md"
        body = (
            f"# HYP_drift_{date}_label_marginal\n\n"
            f"- Severity: **{result.severity}**\n"
            f"- BBSE estimated p_test: `{result.estimated_p_test}`\n"
            f"- p_train: `{result.p_train}`\n"
            f"- χ² stat: {result.chi_square_stat:.4f}, p-value: {result.chi_square_p_value:.6f}\n"
            f"- Max |Δp(y)|: {result.max_abs_class_shift:.4f}\n"
            f"- RLLS regularization used: {result.rlls_used}\n"
            f"- N predictions in window: {result.n_predictions}\n\n"
            f"## Suggested action\n- Inspect cohort selection and ClinVar release deltas.\n"
            f"- Re-run NannyML CBPE under BBSE-corrected weights before declaring concept drift.\n"
        )
        hypothesis_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
        return path
