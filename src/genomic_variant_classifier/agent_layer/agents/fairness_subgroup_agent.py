from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from logging import Logger
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ANCESTRY_AXES: tuple[str, ...] = (
    "afr",
    "ami",
    "amr",
    "asj",
    "eas",
    "fin",
    "mid",
    "nfe",
    "sas",
    "remaining",
)


@dataclass(frozen=True)
class SubgroupMetric:
    axis: str
    stratum: str
    n: int
    auroc_estimated: float
    ece: float
    psi_vs_train: float
    tpr: Optional[float]
    fpr: Optional[float]


@dataclass(frozen=True)
class FairnessResult:
    timestamp: str
    metrics: tuple[SubgroupMetric, ...]
    max_eod: float
    max_dpd_change: float
    severity: str
    high_priority_strata_flags: tuple[str, ...]


@dataclass
class FairnessSubgroupAgent:
    """Per-axis equalized-odds and CBPE-based fairness monitoring.

    Hardt, Price & Srebro, "Equality of Opportunity in Supervised Learning",
    NeurIPS 2016. Stratification axes per project-specific framing:
      - gnomAD ancestry (afr/ami/amr/asj/eas/fin/mid/nfe/sas/remaining)
      - Variant type (SNV, indel, missense, LoF, synonymous, splice)
      - ClinGen gene class (definitive..refuted)
      - Submitting lab (top-20 + long-tail)
    """

    classes: tuple[str, ...]
    p_train_per_stratum: dict[tuple[str, str], np.ndarray]
    output_dir: Path
    eod_amber: float = 0.05
    auroc_below_overall_sigma: float = 2.0
    ece_red: float = 0.05
    high_priority_strata: frozenset[str] = frozenset({"afr", "sas", "amr"})
    logger: Optional[Logger] = field(default=None, repr=False)

    @staticmethod
    def _psi(p: np.ndarray, q: np.ndarray, eps: float = 1e-6) -> float:
        p = np.clip(p, eps, None) / np.clip(p.sum(), eps, None)
        q = np.clip(q, eps, None) / np.clip(q.sum(), eps, None)
        return float(np.sum((p - q) * np.log(p / q)))

    def _stratum_metric(
        self,
        axis: str,
        stratum: str,
        sub: pd.DataFrame,
        overall_auroc: float,
        overall_sigma: float,
    ) -> SubgroupMetric:
        n = len(sub)
        # Confidence-based estimated AUROC (NannyML-style proxy for unlabeled chunks).
        top_conf = sub[[f"p_{c}" for c in self.classes]].to_numpy().max(axis=1)
        auroc_est = float(
            np.clip(top_conf.mean(), 0.5, 1.0)
        )  # placeholder; wire to nannyml.CBPE
        # Per-stratum top-label ECE.
        if "true_class" in sub.columns:
            correct = (sub["predicted_class"] == sub["true_class"]).to_numpy(
                dtype=float
            )
            edges = np.linspace(0, 1, 16)
            ece = 0.0
            for lo, hi in zip(edges[:-1], edges[1:]):
                mask = (top_conf > lo) & (top_conf <= hi)
                if mask.sum():
                    ece += (mask.sum() / max(n, 1)) * abs(
                        correct[mask].mean() - top_conf[mask].mean()
                    )
        else:
            ece = float("nan")
        # PSI on predicted-class marginal vs. training stratum.
        observed = np.array(
            [(sub["predicted_class"] == c).sum() for c in self.classes], dtype=float
        )
        psi = self._psi(
            observed, self.p_train_per_stratum.get((axis, stratum), observed)
        )
        # TPR/FPR for binary collapse (P/LP vs. else) when labels available.
        tpr: Optional[float] = None
        fpr: Optional[float] = None
        if "true_class" in sub.columns:
            yhat_pos = sub["predicted_class"].isin(("P", "LP"))
            ytrue_pos = sub["true_class"].isin(("P", "LP"))
            tp = int((yhat_pos & ytrue_pos).sum())
            fn = int((~yhat_pos & ytrue_pos).sum())
            fp = int((yhat_pos & ~ytrue_pos).sum())
            tn = int((~yhat_pos & ~ytrue_pos).sum())
            tpr = tp / max(tp + fn, 1)
            fpr = fp / max(fp + tn, 1)
        return SubgroupMetric(axis, stratum, n, auroc_est, float(ece), psi, tpr, fpr)

    def detect(self, predictions: pd.DataFrame, axes: dict[str, str]) -> FairnessResult:
        """`axes` maps axis_name -> column name in `predictions`.
        E.g., {"ancestry": "gnomad_pop", "variant_type": "vtype", "lab": "submitter_id"}.
        """
        overall_top = (
            predictions[[f"p_{c}" for c in self.classes]].to_numpy().max(axis=1)
        )
        overall_auroc = float(np.clip(overall_top.mean(), 0.5, 1.0))
        overall_sigma = float(overall_top.std() / np.sqrt(max(len(overall_top), 1)))
        metrics: list[SubgroupMetric] = []
        tpr_overall = float("nan")
        fpr_overall = float("nan")
        if "true_class" in predictions.columns:
            yhat_pos = predictions["predicted_class"].isin(("P", "LP"))
            ytrue_pos = predictions["true_class"].isin(("P", "LP"))
            tp = int((yhat_pos & ytrue_pos).sum())
            fn = int((~yhat_pos & ytrue_pos).sum())
            fp = int((yhat_pos & ~ytrue_pos).sum())
            tn = int((~yhat_pos & ~ytrue_pos).sum())
            tpr_overall = tp / max(tp + fn, 1)
            fpr_overall = fp / max(fp + tn, 1)
        for axis_label, col in axes.items():
            for stratum, sub in predictions.groupby(col, sort=False):
                metrics.append(
                    self._stratum_metric(
                        axis_label, str(stratum), sub, overall_auroc, overall_sigma
                    )
                )
        max_eod = 0.0
        for m in metrics:
            if m.tpr is not None and m.fpr is not None:
                max_eod = max(
                    max_eod, abs(m.tpr - tpr_overall) + abs(m.fpr - fpr_overall)
                )
        priority_flags = tuple(
            sorted(
                {
                    m.stratum
                    for m in metrics
                    if m.stratum in self.high_priority_strata
                    and (
                        m.ece >= self.ece_red
                        or m.auroc_estimated < overall_auroc - 0.03
                    )
                }
            )
        )
        if any(m.ece >= self.ece_red for m in metrics) or priority_flags:
            severity = "red"
        elif max_eod >= self.eod_amber:
            severity = "amber"
        else:
            severity = "green"
        return FairnessResult(
            timestamp=datetime.now(timezone.utc).isoformat(),
            metrics=tuple(metrics),
            max_eod=float(max_eod),
            max_dpd_change=0.0,  # wire from training-time DPD baseline
            severity=severity,
            high_priority_strata_flags=priority_flags,
        )
