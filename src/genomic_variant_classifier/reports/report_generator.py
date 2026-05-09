"""
Report Generator
================
Generates per-modality HTML validation reports for the Genomic Variant
Classifier pipeline.

Modalities: "dna", "rna", "protein"

Each report contains:
  1. Input summary and QC (variant counts, AF distribution)
  2. Ensemble model performance table (AUROC, AUPRC, F1, MCC, Brier)
  3. ROC and Precision-Recall curves
  4. Calibration curve for the stacking meta-learner
  5. Variant-phenotype association statistics (Fisher, Cramér's V)
  6. Feature importance plot
  7. Top predicted pathogenic variants table

CHANGES FROM PHASE 1:
  - Module was a bare string literal (Bug 3 fixed — now a real .py file).
  - Template.globals["format_int"] = ... is invalid: Template objects do not
    have a .globals dict; that attribute lives on Environment objects.
    Fixed by using a Jinja2 Environment with a custom filter (Issue O).
  - Module-level logging.basicConfig removed (Issue L).
  - from __future__ import annotations added (Issue N).

Dependencies:
    pip install jinja2 matplotlib seaborn scipy scikit-learn
"""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from jinja2 import Environment
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})


# ---------------------------------------------------------------------------
# Validation metrics dataclass
# ---------------------------------------------------------------------------
@dataclass
class ValidationMetrics:
    """Full validation statistics for a single model or ensemble."""

    model_name:   str
    auroc:        float
    auprc:        float
    f1_macro:     float
    f1_weighted:  float
    mcc:          float
    brier:        float
    auroc_ci:     tuple[float, float] = (0.0, 0.0)
    auprc_ci:     tuple[float, float] = (0.0, 0.0)
    n_pathogenic: int = 0
    n_benign:     int = 0
    n_uncertain:  int = 0


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------
def bootstrap_metric(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> tuple[float, float]:
    """Bootstrap confidence interval for a scalar classification metric."""
    rng = np.random.default_rng(42)
    scores: list[float] = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(metric_fn(y_true[idx], y_proba[idx]))
    arr = np.array(scores)
    alpha = (1 - ci) / 2
    return float(np.percentile(arr, 100 * alpha)), float(np.percentile(arr, 100 * (1 - alpha)))


def compute_variant_phenotype_association(
    variant_presence: np.ndarray,
    phenotype: np.ndarray,
) -> dict[str, Any]:
    """
    Compute odds ratio, p-value (Fisher exact), and Cramér's V effect size
    for a variant-phenotype pair.

    Args:
        variant_presence: Binary array (1 = carries variant, 0 = does not).
        phenotype:        Binary array (1 = has disease, 0 = does not).

    Returns:
        dict with keys: odds_ratio, p_value, cramers_v, contingency_table, significant.
    """
    ct = np.array([
        [np.sum((variant_presence == 1) & (phenotype == 1)),
         np.sum((variant_presence == 1) & (phenotype == 0))],
        [np.sum((variant_presence == 0) & (phenotype == 1)),
         np.sum((variant_presence == 0) & (phenotype == 0))],
    ])
    oddsratio, pvalue = stats.fisher_exact(ct)
    n = int(ct.sum())
    chi2 = float(stats.chi2_contingency(ct)[0]) if n > 0 else 0.0
    cramers_v = float(np.sqrt(chi2 / n)) if n > 0 else 0.0

    return {
        "odds_ratio":         round(float(oddsratio), 4),
        "p_value":            round(float(pvalue), 6),
        "cramers_v":          round(cramers_v, 4),
        "contingency_table":  ct.tolist(),
        "significant":        pvalue < 0.05,
    }


# ---------------------------------------------------------------------------
# Plot helpers (return base64-encoded PNG for inline HTML embedding)
# ---------------------------------------------------------------------------
def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def plot_roc_curves(results: dict[str, tuple]) -> str:
    """ROC curves for all models. results = {name: (y_true, y_proba)}."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10.colors
    for (name, (y_true, y_proba)), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Ensemble vs. Base Models")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    return _fig_to_b64(fig)


def plot_pr_curves(results: dict[str, tuple]) -> str:
    """Precision-Recall curves. results = {name: (y_true, y_proba)}."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10.colors
    for (name, (y_true, y_proba)), color in zip(results.items(), colors):
        prec, rec, _ = precision_recall_curve(y_true, y_proba)
        ap = average_precision_score(y_true, y_proba)
        ax.step(rec, prec, label=f"{name} (AP={ap:.3f})", color=color, lw=2, where="post")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(loc="upper right", fontsize=9)
    return _fig_to_b64(fig)


def plot_allele_freq_distribution(variant_df: pd.DataFrame) -> str:
    """Population AF distribution split by pathogenicity class."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    has_af = variant_df[
        variant_df["allele_freq"].notna() & (variant_df["allele_freq"] > 0)
    ]
    ax = axes[0]
    if not has_af.empty and "pathogenicity" in has_af.columns:
        for path_class, color in [
            ("pathogenic", "#e74c3c"),
            ("benign", "#2ecc71"),
            ("uncertain", "#95a5a6"),
        ]:
            subset = has_af[has_af["pathogenicity"] == path_class]["allele_freq"]
            if not subset.empty:
                ax.hist(
                    np.log10(subset + 1e-8),
                    bins=40, alpha=0.6, label=path_class, color=color,
                )
    ax.set_xlabel("log10(Allele Frequency)")
    ax.set_ylabel("Count")
    ax.set_title("Allele Frequency Distribution")
    ax.legend()

    ax2 = axes[1]
    if "pathogenicity" in variant_df.columns:
        path_counts = variant_df["pathogenicity"].fillna("unknown").value_counts()
        bar_colors = [
            "#e74c3c" if "path" in k else
            "#2ecc71" if "benign" in k else "#95a5a6"
            for k in path_counts.index
        ]
        ax2.bar(path_counts.index, path_counts.values, color=bar_colors)
    ax2.set_title("Variant Count by Pathogenicity Class")
    ax2.set_ylabel("Count")
    plt.tight_layout()
    return _fig_to_b64(fig)


def plot_feature_importance(
    importance_df: pd.DataFrame,
    title: str = "Feature Importance",
) -> str:
    fig, ax = plt.subplots(figsize=(9, max(5, len(importance_df) * 0.35)))
    importance_df = importance_df.sort_values("importance", ascending=True)
    ax.barh(importance_df["feature"], importance_df["importance"],
            color="#3498db", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel("Mean Importance")
    plt.tight_layout()
    return _fig_to_b64(fig)


def plot_calibration(y_true, y_proba, model_name: str) -> str:
    from sklearn.calibration import calibration_curve
    fig, ax = plt.subplots(figsize=(6, 5))
    frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=10)
    ax.plot(mean_pred, frac_pos, "s-", label=model_name, color="#e74c3c")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(f"Calibration Curve — {model_name}")
    ax.legend()
    return _fig_to_b64(fig)


# ---------------------------------------------------------------------------
# Jinja2 environment with custom filter
# ---------------------------------------------------------------------------
def _make_jinja_env() -> Environment:
    """
    CHANGE: The original code called Template.globals["format_int"] = ...,
    which raises AttributeError because Template objects have no .globals.
    That attribute belongs to Environment objects (Issue O).

    The fix: create an Environment, register format_int as a filter,
    and call env.from_string(HTML_TEMPLATE) to get a Template.
    Jinja2 filters are invoked with the pipe operator: {{ value | format_int }}.
    """
    env = Environment(autoescape=False)
    env.filters["format_int"] = lambda x: f"{int(x):,}"
    return env


# ---------------------------------------------------------------------------
# HTML report template
# ---------------------------------------------------------------------------
HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>Genomic Disease Association Report — {{ modality }}</title>
<style>
  body { font-family: 'Segoe UI', Arial, sans-serif; max-width: 1100px; margin: 0 auto; padding: 30px; color: #2c3e50; background: #f8f9fa; }
  h1 { color: #1a252f; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
  h2 { color: #2c3e50; border-left: 4px solid #3498db; padding-left: 12px; margin-top: 40px; }
  h3 { color: #34495e; }
  .meta { background: #ecf0f1; padding: 15px; border-radius: 6px; margin-bottom: 30px; }
  .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin: 20px 0; }
  .metric-card { background: white; border: 1px solid #ddd; border-radius: 8px; padding: 15px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
  .metric-card .value { font-size: 2em; font-weight: bold; color: #2980b9; }
  .metric-card .label { color: #7f8c8d; font-size: 0.9em; margin-top: 4px; }
  table { width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
  th { background: #2c3e50; color: white; padding: 12px 15px; text-align: left; }
  td { padding: 10px 15px; border-bottom: 1px solid #ecf0f1; }
  tr:hover td { background: #f1f8ff; }
  .sig { color: #e74c3c; font-weight: bold; }
  .not-sig { color: #7f8c8d; }
  img { max-width: 100%; border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.15); margin: 10px 0; }
  .section { background: white; border-radius: 8px; padding: 25px; margin: 20px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
  .badge-path { background: #e74c3c; color: white; padding: 3px 8px; border-radius: 12px; font-size: 0.8em; }
  .badge-benign { background: #2ecc71; color: white; padding: 3px 8px; border-radius: 12px; font-size: 0.8em; }
  .badge-uncertain { background: #95a5a6; color: white; padding: 3px 8px; border-radius: 12px; font-size: 0.8em; }
  footer { text-align: center; color: #7f8c8d; font-size: 0.85em; margin-top: 50px; padding-top: 20px; border-top: 1px solid #ecf0f1; }
</style>
</head>
<body>

<h1>&#x1F9EC; Genomic Disease Association Report</h1>

<div class="meta">
  <strong>Modality:</strong> {{ modality }} &nbsp;|&nbsp;
  <strong>Generated:</strong> {{ generated_at }} &nbsp;|&nbsp;
  <strong>Pipeline version:</strong> {{ version }}
  {% if run_id %}&nbsp;|&nbsp; <strong>Run ID:</strong> {{ run_id }}{% endif %}
</div>

<div class="section">
  <h2>1. Input Summary &amp; QC</h2>
  <div class="metric-grid">
    <div class="metric-card"><div class="value">{{ summary.total_variants | format_int }}</div><div class="label">Total Variants</div></div>
    <div class="metric-card"><div class="value">{{ summary.n_pathogenic | format_int }}</div><div class="label">Pathogenic / Likely Pathogenic</div></div>
    <div class="metric-card"><div class="value">{{ summary.n_benign | format_int }}</div><div class="label">Benign / Likely Benign</div></div>
    <div class="metric-card"><div class="value">{{ summary.n_uncertain | format_int }}</div><div class="label">Uncertain Significance</div></div>
    <div class="metric-card"><div class="value">{{ summary.n_genes | format_int }}</div><div class="label">Unique Genes</div></div>
    <div class="metric-card"><div class="value">{{ summary.n_sources }}</div><div class="label">Source Databases</div></div>
  </div>
  {% if qc_plot %}<img src="data:image/png;base64,{{ qc_plot }}" alt="QC distribution"/>{% endif %}
</div>

<div class="section">
  <h2>2. Ensemble Prediction Performance</h2>
  <table>
    <thead><tr><th>Model</th><th>AUROC</th><th>AUPRC</th><th>F1 (macro)</th><th>MCC</th><th>Brier Score</th></tr></thead>
    <tbody>
      {% for row in model_metrics %}
      <tr>
        <td><strong>{{ row.model_name }}</strong></td>
        <td>{{ "%.4f" | format(row.auroc) }}</td>
        <td>{{ "%.4f" | format(row.auprc) }}</td>
        <td>{{ "%.4f" | format(row.f1_macro) }}</td>
        <td>{{ "%.4f" | format(row.mcc) }}</td>
        <td>{{ "%.4f" | format(row.brier) }}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
  {% if roc_plot %}<img src="data:image/png;base64,{{ roc_plot }}" alt="ROC curves"/>{% endif %}
  {% if pr_plot %}<img src="data:image/png;base64,{{ pr_plot }}" alt="PR curves"/>{% endif %}
  {% if calibration_plot %}<img src="data:image/png;base64,{{ calibration_plot }}" alt="Calibration"/>{% endif %}
</div>

<div class="section">
  <h2>3. Variant–Phenotype Association Statistics</h2>
  <p>Fisher's exact test. Effect size reported as Cramér's V.</p>
  {% if associations %}
  <table>
    <thead><tr><th>Variant</th><th>Gene</th><th>Phenotype</th><th>Odds Ratio</th><th>p-value</th><th>Cramér's V</th><th>Significant</th></tr></thead>
    <tbody>
      {% for assoc in associations %}
      <tr>
        <td>{{ assoc.variant_id }}</td>
        <td>{{ assoc.gene }}</td>
        <td>{{ assoc.phenotype }}</td>
        <td>{{ "%.3f" | format(assoc.odds_ratio) }}</td>
        <td {% if assoc.significant %}class="sig"{% else %}class="not-sig"{% endif %}>{{ "%.6f" | format(assoc.p_value) }}</td>
        <td>{{ "%.4f" | format(assoc.cramers_v) }}</td>
        <td>{% if assoc.significant %}<span class="badge-path">Yes</span>{% else %}<span class="badge-uncertain">No</span>{% endif %}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
  {% else %}
  <p>No phenotype association data available for this run.</p>
  {% endif %}
</div>

{% if feature_importance_plot %}
<div class="section">
  <h2>4. Feature Importance (Random Forest + XGBoost ensemble)</h2>
  <img src="data:image/png;base64,{{ feature_importance_plot }}" alt="Feature importance"/>
</div>
{% endif %}

<div class="section">
  <h2>5. Top Predicted Pathogenic Variants</h2>
  {% if top_variants %}
  <table>
    <thead><tr><th>Variant ID</th><th>Gene</th><th>Consequence</th><th>Protein Change</th><th>AF</th><th>Pathogenicity</th><th>Ensemble Score</th></tr></thead>
    <tbody>
      {% for v in top_variants %}
      <tr>
        <td>{{ v.variant_id }}</td>
        <td>{{ v.gene_symbol }}</td>
        <td>{{ v.consequence }}</td>
        <td>{{ v.protein_change }}</td>
        <td>{{ "%.2e" | format(v.allele_freq) if v.allele_freq != "N/A" else "N/A" }}</td>
        <td>
          {% if "path" in v.pathogenicity | lower %}<span class="badge-path">{{ v.pathogenicity }}</span>
          {% elif "benign" in v.pathogenicity | lower %}<span class="badge-benign">{{ v.pathogenicity }}</span>
          {% else %}<span class="badge-uncertain">{{ v.pathogenicity }}</span>{% endif %}
        </td>
        <td><strong>{{ "%.3f" | format(v.ensemble_score) }}</strong></td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
  {% endif %}
</div>

<footer>
  Generated by Genomic Variant Classifier v{{ version }} &bull; {{ generated_at }}<br/>
  Data sources: ClinVar, gnomAD v4, UniProt, STRING DB
</footer>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Report generator class
# ---------------------------------------------------------------------------
class ReportGenerator:
    """
    Generates per-modality HTML validation reports.

    Usage:
        gen = ReportGenerator(output_dir=Path("reports"))
        path = gen.generate(
            modality="dna",
            variant_df=df,
            model_metrics=[{"model_name": "RF", "auroc": 0.91, ...}],
        )
    """

    VERSION = "1.0.0"

    def __init__(self, output_dir: Path = Path("reports")) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # CHANGE: use Environment.from_string instead of Template() +
        # Template.globals assignment (Issue O)
        self._env = _make_jinja_env()
        self._template = self._env.from_string(HTML_TEMPLATE)

    def generate(
        self,
        modality: str,
        variant_df: pd.DataFrame,
        model_metrics: list[dict],
        roc_data: Optional[dict[str, tuple]] = None,
        associations: Optional[list[dict]] = None,
        feature_importance: Optional[pd.DataFrame] = None,
        top_n_variants: int = 25,
        run_id: Optional[str] = None,
    ) -> Path:
        """
        Render and write an HTML report to self.output_dir.

        Args:
            modality:           "dna", "rna", or "protein".
            variant_df:         Canonical variant DataFrame.
            model_metrics:      List of dicts with keys matching ValidationMetrics fields.
            roc_data:           Optional {model_name: (y_true, y_proba)} for curve plots.
            associations:       Optional list of variant-phenotype association dicts.
            feature_importance: Optional DataFrame with "feature" and "importance" columns.
            top_n_variants:     Number of top predicted variants to include in table.
            run_id:             Optional experiment identifier.

        Returns:
            Path to the written HTML file.
        """
        logger.info(
            "Generating %s report for %d variants...", modality.upper(), len(variant_df),
        )

        summary = {
            "total_variants": len(variant_df),
            "n_pathogenic": int((variant_df.get("pathogenicity", pd.Series()) == "pathogenic").sum()),
            "n_benign":     int((variant_df.get("pathogenicity", pd.Series()) == "benign").sum()),
            "n_uncertain":  int((variant_df.get("pathogenicity", pd.Series()) == "uncertain").sum()),
            "n_genes":      int(variant_df.get("gene_symbol", pd.Series()).nunique()),
            "n_sources":    int(variant_df.get("source_db", pd.Series()).nunique()),
        }

        qc_plot = (
            plot_allele_freq_distribution(variant_df)
            if "allele_freq" in variant_df.columns else None
        )

        roc_plot = pr_plot = calibration_plot = None
        if roc_data:
            roc_plot = plot_roc_curves(roc_data)
            pr_plot  = plot_pr_curves(roc_data)
            if "ENSEMBLE_STACKER" in roc_data:
                y_true, y_proba = roc_data["ENSEMBLE_STACKER"]
                calibration_plot = plot_calibration(y_true, y_proba, "Ensemble Stacker")

        fi_plot = None
        if feature_importance is not None and not feature_importance.empty:
            fi_plot = plot_feature_importance(feature_importance)

        top_variants: list[dict] = []
        if "ensemble_score" in variant_df.columns:
            cols = [
                "variant_id", "gene_symbol", "consequence",
                "protein_change", "allele_freq", "pathogenicity", "ensemble_score",
            ]
            available = [c for c in cols if c in variant_df.columns]
            top_df = variant_df.nlargest(top_n_variants, "ensemble_score")[available]
            top_variants = top_df.fillna("N/A").to_dict("records")

        html = self._template.render(
            modality=modality.upper(),
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
            version=self.VERSION,
            run_id=run_id,
            summary=summary,
            qc_plot=qc_plot,
            model_metrics=model_metrics,
            roc_plot=roc_plot,
            pr_plot=pr_plot,
            calibration_plot=calibration_plot,
            feature_importance_plot=fi_plot,
            associations=associations or [],
            top_variants=top_variants,
        )

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = self.output_dir / f"report_{modality}_{ts}.html"
        out_path.write_text(html, encoding="utf-8")
        logger.info("Report written to %s", out_path)
        return out_path
