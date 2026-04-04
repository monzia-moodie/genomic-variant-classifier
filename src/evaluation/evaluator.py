"""
Clinical Evaluator
==================
Comprehensive evaluation framework for variant pathogenicity classifiers.

Goes beyond basic AUROC to measure what matters in clinical genomics:
  - Calibration quality (are predicted probabilities trustworthy?)
  - Performance at clinically-relevant operating points (90%/95% sensitivity,
    ≥80% PPV for confident reporting to clinicians)
  - Per-consequence-class breakdown (LoF vs. missense vs. synonymous)
  - Bootstrap confidence intervals on all primary metrics
  - Gene-level error analysis (which genes drive the most errors?)

CHANGES FROM PHASE 1:
  - Was a bare string literal (Bug 3 fixed — now a real .py file).
  - _gene_error_analysis used itertuples() and then **row._asdict().
    This is fragile: pandas renames columns whose names conflict with
    Python keywords or NamedTuple internals (e.g., "index", "_fields").
    Fixed by using DataFrame.to_dict(orient="records") which returns
    plain dicts that unpack cleanly with ** (Issue S).
  - Module-level logging.basicConfig removed (Issue L).
  - from __future__ import annotations added (Issue N).

Usage:
    from src.evaluation.evaluator import ClinicalEvaluator

    evaluator = ClinicalEvaluator()
    report = evaluator.evaluate(
        y_true=y_test,
        y_proba=ensemble_proba,
        meta=meta_test,
        model_name="EnsembleStacker",
    )
    evaluator.print_report(report)
    evaluator.save_report(report, path="models/v1/eval_report.json")
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class OperatingPoint:
    """Model performance at a specific probability threshold."""

    threshold:    float
    sensitivity:  float   # TPR / recall
    specificity:  float   # TNR
    ppv:          float   # precision / positive predictive value
    npv:          float   # negative predictive value
    f1:           float
    n_flagged:    int     # total predicted positive (TP + FP)
    n_tp:         int
    n_fp:         int
    n_fn:         int
    n_tn:         int


@dataclass
class ConsequenceBreakdown:
    """Per-consequence-class performance metrics."""

    consequence:  str
    n_total:      int
    n_pathogenic: int
    auroc:        float
    auprc:        float
    prevalence:   float


@dataclass
class GeneErrorAnalysis:
    """Error analysis for a single gene."""

    gene_symbol:        str
    n_variants:         int
    n_false_positives:  int
    n_false_negatives:  int
    total_errors:       int
    error_rate:         float


@dataclass
class EvaluationReport:
    """Full evaluation report for one model."""

    model_name:   str
    n_samples:    int
    n_pathogenic: int
    n_benign:     int
    prevalence:   float

    # Core discriminative metrics
    auroc:       float
    auroc_ci_lo: float
    auroc_ci_hi: float
    auprc:       float
    auprc_ci_lo: float
    auprc_ci_hi: float
    mcc:         float
    brier_score: float

    # Calibration
    calibration_ece: float  # Expected Calibration Error
    calibration_mce: float  # Maximum Calibration Error

    # Clinical operating points
    at_sensitivity_90: Optional[OperatingPoint] = None
    at_sensitivity_95: Optional[OperatingPoint] = None
    at_high_ppv:       Optional[OperatingPoint] = None

    # Breakdowns
    consequence_breakdown: list = field(default_factory=list)
    gene_errors:           list = field(default_factory=list)

    # Curves for downstream plotting
    fpr_curve:               list = field(default_factory=list)
    tpr_curve:               list = field(default_factory=list)
    precision_curve:         list = field(default_factory=list)
    recall_curve:            list = field(default_factory=list)
    calibration_frac_pos:    list = field(default_factory=list)
    calibration_mean_pred:   list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core evaluator
# ---------------------------------------------------------------------------
class ClinicalEvaluator:
    """
    Computes a full suite of clinical evaluation metrics for a binary
    variant pathogenicity classifier.
    """

    def __init__(
        self,
        n_bootstrap: int = 1000,
        random_state: int = 42,
    ) -> None:
        self.n_bootstrap = n_bootstrap
        self.rng = np.random.default_rng(random_state)

    # ── Public entry point ─────────────────────────────────────────────────

    def evaluate(
        self,
        y_true: pd.Series | np.ndarray,
        y_proba: np.ndarray,
        meta: Optional[pd.DataFrame] = None,
        model_name: str = "model",
    ) -> EvaluationReport:
        """
        Full evaluation pipeline.

        Args:
            y_true:     Binary ground-truth labels (1=pathogenic, 0=benign).
            y_proba:    Predicted probabilities in [0, 1].
            meta:       Canonical variant DataFrame aligned with y_true/y_proba.
                        Required for per-gene and per-consequence analysis.
            model_name: Label for this model in report output.

        Returns:
            EvaluationReport with all metrics populated.
        """
        y = np.asarray(y_true)
        p = np.asarray(y_proba)
        n = len(y)
        n_pos = int(y.sum())
        n_neg = n - n_pos

        logger.info(
            "Evaluating %s: n=%d, pos=%d (%.1f%%)",
            model_name, n, n_pos, n_pos / n * 100,
        )

        # Core metrics
        auroc = roc_auc_score(y, p)
        auprc = average_precision_score(y, p)
        mcc   = matthews_corrcoef(y, (p >= 0.5).astype(int))
        brier = float(np.mean((p - y) ** 2))

        auroc_ci = self._bootstrap_ci(y, p, roc_auc_score)
        auprc_ci = self._bootstrap_ci(y, p, average_precision_score)

        # Curves
        fpr, tpr, _  = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)

        # Calibration
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ece, mce = self._calibration_error(y, p, n_bins=10)

        # Operating points
        op_90  = self._find_operating_point(y, p, target_sensitivity=0.90)
        op_95  = self._find_operating_point(y, p, target_sensitivity=0.95)
        op_ppv = self._find_high_ppv_point(y, p, min_ppv=0.80)

        # Breakdowns (require meta)
        consequence_rows: list = []
        gene_error_rows:  list = []
        if meta is not None:
            consequence_rows = self._consequence_breakdown(y, p, meta)
            gene_error_rows  = self._gene_error_analysis(y, p, meta, top_n=20)

        report = EvaluationReport(
            model_name=model_name,
            n_samples=n, n_pathogenic=n_pos, n_benign=n_neg,
            prevalence=round(n_pos / n, 4),
            auroc=round(auroc, 5),
            auroc_ci_lo=round(auroc_ci[0], 5),
            auroc_ci_hi=round(auroc_ci[1], 5),
            auprc=round(auprc, 5),
            auprc_ci_lo=round(auprc_ci[0], 5),
            auprc_ci_hi=round(auprc_ci[1], 5),
            mcc=round(mcc, 5),
            brier_score=round(brier, 5),
            calibration_ece=round(ece, 5),
            calibration_mce=round(mce, 5),
            at_sensitivity_90=op_90,
            at_sensitivity_95=op_95,
            at_high_ppv=op_ppv,
            consequence_breakdown=consequence_rows,
            gene_errors=gene_error_rows,
            fpr_curve=fpr.tolist(),
            tpr_curve=tpr.tolist(),
            precision_curve=prec.tolist(),
            recall_curve=rec.tolist(),
            calibration_frac_pos=frac_pos.tolist(),
            calibration_mean_pred=mean_pred.tolist(),
        )
        self.print_report(report)
        return report

    # ── Metric helpers ─────────────────────────────────────────────────────

    def _bootstrap_ci(
        self,
        y: np.ndarray,
        p: np.ndarray,
        metric_fn,
        ci: float = 0.95,
    ) -> tuple[float, float]:
        scores: list[float] = []
        n = len(y)
        for _ in range(self.n_bootstrap):
            idx = self.rng.integers(0, n, n)
            if len(np.unique(y[idx])) < 2:
                continue
            scores.append(metric_fn(y[idx], p[idx]))
        arr = np.array(scores)
        alpha = (1 - ci) / 2
        return (
            float(np.percentile(arr, 100 * alpha)),
            float(np.percentile(arr, 100 * (1 - alpha))),
        )

    def _calibration_error(
        self,
        y: np.ndarray,
        p: np.ndarray,
        n_bins: int = 10,
    ) -> tuple[float, float]:
        """
        Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
        Each bin's contribution to ECE is weighted by its fraction of total samples.
        """
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        mce = 0.0
        n = len(y)
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (p >= lo) & (p < hi)
            if mask.sum() == 0:
                continue
            accuracy   = float(y[mask].mean())
            confidence = float(p[mask].mean())
            bin_weight = mask.sum() / n
            err = abs(accuracy - confidence)
            ece += bin_weight * err
            mce = max(mce, err)
        return float(ece), float(mce)

    def _find_operating_point(
        self,
        y: np.ndarray,
        p: np.ndarray,
        target_sensitivity: float,
    ) -> Optional[OperatingPoint]:
        """Find the threshold closest to the target sensitivity (recall)."""
        best: Optional[OperatingPoint] = None
        best_diff = float("inf")
        for t in np.linspace(0, 1, 1000):
            preds = (p >= t).astype(int)
            tp = int(((preds == 1) & (y == 1)).sum())
            fp = int(((preds == 1) & (y == 0)).sum())
            fn = int(((preds == 0) & (y == 1)).sum())
            tn = int(((preds == 0) & (y == 0)).sum())
            n_pos = tp + fn
            n_neg = fp + tn
            if n_pos == 0:
                continue
            sensitivity = tp / n_pos
            diff = abs(sensitivity - target_sensitivity)
            if diff < best_diff:
                best_diff = diff
                specificity = tn / n_neg if n_neg > 0 else 0.0
                ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
                f1  = (2 * ppv * sensitivity / (ppv + sensitivity)
                       if (ppv + sensitivity) > 0 else 0.0)
                best = OperatingPoint(
                    threshold=round(float(t), 4),
                    sensitivity=round(sensitivity, 4),
                    specificity=round(specificity, 4),
                    ppv=round(ppv, 4),
                    npv=round(npv, 4),
                    f1=round(f1, 4),
                    n_flagged=int(tp + fp),
                    n_tp=tp, n_fp=fp, n_fn=fn, n_tn=tn,
                )
        return best

    def _find_high_ppv_point(
        self,
        y: np.ndarray,
        p: np.ndarray,
        min_ppv: float = 0.80,
    ) -> Optional[OperatingPoint]:
        """
        Highest-sensitivity threshold where PPV ≥ min_ppv.

        Walk thresholds from HIGH→LOW (conservative→liberal).
        Track the last threshold seen where ppv ≥ min_ppv — that is the
        most permissive threshold that never drops below min_ppv.
        """
        thresholds = np.sort(np.unique(p))[::-1]  # high → low
        best: Optional[OperatingPoint] = None

        for t in thresholds:
            preds = (p >= t).astype(int)
            tp = int(((preds == 1) & (y == 1)).sum())
            fp = int(((preds == 1) & (y == 0)).sum())
            fn = int(((preds == 0) & (y == 1)).sum())
            tn = int(((preds == 0) & (y == 0)).sum())
            n_pos = tp + fn
            n_neg = tp + fp  # n_flagged

            if n_neg == 0 or n_pos == 0:
                continue

            ppv = tp / n_neg
            if ppv < min_ppv:
                # Once PPV drops below target, stop — prior iteration was the best
                break

            sensitivity = tp / n_pos
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            npv         = tn / (tn + fn) if (tn + fn) > 0 else 0.0
            f1          = (2 * ppv * sensitivity / (ppv + sensitivity)
                           if (ppv + sensitivity) > 0 else 0.0)

            best = OperatingPoint(
                threshold   = round(float(t), 4),
                sensitivity = round(sensitivity, 4),
                specificity = round(specificity, 4),
                ppv         = round(ppv, 4),
                npv         = round(npv, 4),
                f1          = round(f1, 4),
                n_flagged   = int(n_neg),
                n_tp=tp, n_fp=fp, n_fn=fn, n_tn=tn,
            )

        return best

    # ── Breakdown helpers ──────────────────────────────────────────────────

    def _consequence_breakdown(
        self,
        y: np.ndarray,
        p: np.ndarray,
        meta: pd.DataFrame,
    ) -> list[ConsequenceBreakdown]:
        """AUROC and AUPRC broken down by coarsened consequence category."""
        if "consequence" not in meta.columns:
            return []

        meta = meta.reset_index(drop=True)
        consequence = meta["consequence"].fillna("unknown")

        def coarsen(c: str) -> str:
            c = str(c).lower()
            if any(t in c for t in [
                "stop_gained", "frameshift", "splice_donor",
                "splice_acceptor", "start_lost",
            ]):
                return "loss_of_function"
            if "missense"   in c: return "missense"
            if "synonymous" in c: return "synonymous"
            if "splice"     in c: return "splice_region"
            if "inframe"    in c: return "inframe_indel"
            return "other"

        consequence_coarse = consequence.map(coarsen)
        rows: list[ConsequenceBreakdown] = []

        for cat in sorted(consequence_coarse.unique()):
            mask = (consequence_coarse == cat).values
            if mask.sum() < 20 or len(np.unique(y[mask])) < 2:
                continue
            rows.append(ConsequenceBreakdown(
                consequence=cat,
                n_total=int(mask.sum()),
                n_pathogenic=int(y[mask].sum()),
                auroc=round(float(roc_auc_score(y[mask], p[mask])), 4),
                auprc=round(float(average_precision_score(y[mask], p[mask])), 4),
                prevalence=round(float(y[mask].mean()), 4),
            ))
        return rows

    def _gene_error_analysis(
        self,
        y: np.ndarray,
        p: np.ndarray,
        meta: pd.DataFrame,
        top_n: int = 20,
        threshold: float = 0.5,
    ) -> list[GeneErrorAnalysis]:
        """
        Identify genes contributing most to false positives and negatives.

        CHANGE: The original code used itertuples() and then **row._asdict().
        pandas renames columns that collide with NamedTuple reserved names
        (e.g., "index", "_fields") which causes KeyError on unpack.
        Using .to_dict(orient="records") returns plain dicts that unpack
        cleanly and are immune to column-name collisions (Issue S).
        """
        if "gene_symbol" not in meta.columns:
            return []

        meta = meta.reset_index(drop=True).copy()
        preds = (p >= threshold).astype(int)

        meta["_fp"] = ((preds == 1) & (y == 0)).astype(int)
        meta["_fn"] = ((preds == 0) & (y == 1)).astype(int)

        gene_errors = (
            meta.groupby("gene_symbol")
            .agg(
                n_variants=("_fp", "count"),
                n_false_positives=("_fp", "sum"),
                n_false_negatives=("_fn", "sum"),
            )
            .reset_index()
        )
        gene_errors["total_errors"] = (
            gene_errors["n_false_positives"] + gene_errors["n_false_negatives"]
        )
        gene_errors["error_rate"] = (
            gene_errors["total_errors"] / gene_errors["n_variants"]
        ).round(4)

        gene_errors = (
            gene_errors
            .sort_values("total_errors", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

        # CHANGE: to_dict → plain dicts → unpack cleanly (Issue S)
        return [
            GeneErrorAnalysis(**row)
            for row in gene_errors.to_dict(orient="records")
        ]

    # ── Output ─────────────────────────────────────────────────────────────

    def print_report(self, r: EvaluationReport) -> None:
        sep = "─" * 60
        print(f"\n{sep}")
        print(f"  EVALUATION REPORT: {r.model_name}")
        print(sep)
        print(
            f"  Dataset: {r.n_samples:,} variants  "
            f"({r.n_pathogenic:,} pathogenic = {r.prevalence*100:.1f}%)"
        )
        print()
        print(
            f"  AUROC   : {r.auroc:.4f}  "
            f"[95% CI: {r.auroc_ci_lo:.4f}–{r.auroc_ci_hi:.4f}]"
        )
        print(
            f"  AUPRC   : {r.auprc:.4f}  "
            f"[95% CI: {r.auprc_ci_lo:.4f}–{r.auprc_ci_hi:.4f}]"
        )
        print(f"  MCC     : {r.mcc:.4f}")
        print(
            f"  Brier   : {r.brier_score:.4f}  "
            f"(ECE: {r.calibration_ece:.4f}, MCE: {r.calibration_mce:.4f})"
        )

        for label, op in [
            ("@ Sensitivity ≥ 90%", r.at_sensitivity_90),
            ("@ Sensitivity ≥ 95%", r.at_sensitivity_95),
            ("@ PPV ≥ 80%",         r.at_high_ppv),
        ]:
            if op:
                print()
                print(f"  {label}  (threshold={op.threshold:.3f}):")
                print(
                    f"    Sens: {op.sensitivity:.3f}  Spec: {op.specificity:.3f}  "
                    f"PPV: {op.ppv:.3f}  NPV: {op.npv:.3f}  Flagged: {op.n_flagged:,}"
                )

        if r.consequence_breakdown:
            print()
            print(f"  {'Consequence':<22} {'N':>7} {'%Path':>7} {'AUROC':>8} {'AUPRC':>8}")
            print(f"  {'─'*22} {'─'*7} {'─'*7} {'─'*8} {'─'*8}")
            for cb in sorted(r.consequence_breakdown, key=lambda x: x.auroc, reverse=True):
                print(
                    f"  {cb.consequence:<22} {cb.n_total:>7,} "
                    f"{cb.prevalence*100:>6.1f}% "
                    f"{cb.auroc:>8.4f} {cb.auprc:>8.4f}"
                )

        print(sep + "\n")

    def save_report(self, report: EvaluationReport, path: str | Path) -> None:
        """Serialize the full report to JSON (curves included for downstream plotting)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(asdict(report), fh, indent=2, default=str)
        logger.info("Evaluation report saved to %s", path)


# ---------------------------------------------------------------------------
# Multi-model comparison convenience function
# ---------------------------------------------------------------------------
def compare_models(
    y_true: np.ndarray,
    model_probas: dict[str, np.ndarray],
    meta: Optional[pd.DataFrame] = None,
    n_bootstrap: int = 500,
    output_csv: str = "models/model_comparison.csv",
) -> pd.DataFrame:
    """
    Compare multiple models in one call.

    Args:
        y_true:        Ground-truth binary labels.
        model_probas:  {model_name: proba_array}.
        meta:          Optional variant metadata for consequence/gene breakdowns.
        n_bootstrap:   Bootstrap iterations for CI estimation.
        output_csv:    Where to save the comparison table.

    Returns:
        DataFrame with one row per model, sorted by AUROC descending.
    """
    evaluator = ClinicalEvaluator(n_bootstrap=n_bootstrap)
    records: list[dict] = []

    for name, proba in model_probas.items():
        r = evaluator.evaluate(y_true, proba, meta=meta, model_name=name)
        records.append({
            "model":             name,
            "auroc":             r.auroc,
            "auroc_95ci":        f"[{r.auroc_ci_lo:.4f}, {r.auroc_ci_hi:.4f}]",
            "auprc":             r.auprc,
            "mcc":               r.mcc,
            "brier":             r.brier_score,
            "ece":               r.calibration_ece,
            "sens_at_90_spec":   r.at_sensitivity_90.specificity if r.at_sensitivity_90 else None,
            "ppv_at_90_sens":    r.at_sensitivity_90.ppv         if r.at_sensitivity_90 else None,
        })

    df = pd.DataFrame(records).sort_values("auroc", ascending=False).reset_index(drop=True)

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Comparison table saved to %s", out_path)
    return df
