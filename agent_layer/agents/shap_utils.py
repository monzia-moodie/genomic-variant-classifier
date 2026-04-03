"""
agents/shap_utils.py
=====================
SHAP and GradCAM utilities for the InterpretabilityAgent.

Provides
--------
compute_ensemble_shap()      — TreeExplainer SHAP values for XGB / LightGBM
compute_gradcam()            — GradCAM saliency maps for ResNet-50
rank_features()              — Mean |SHAP| ranking
measure_stability()          — Spearman rank-correlation vs previous run
detect_importance_anomalies()— Flags large per-feature deltas
detect_biological_anomalies()— Flags non-biological features in top-K
generate_html_report()       — Self-contained HTML report (no external CDN)
"""

from __future__ import annotations

import base64
import io
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SHAP — ensemble (XGBoost / LightGBM)
# ---------------------------------------------------------------------------

def compute_ensemble_shap(
    model_path: Path,
    X: np.ndarray,
    feature_names: list[str],
    model_type: str = "xgboost",   # "xgboost" | "lightgbm"
    max_samples: int = 2000,
) -> dict[str, Any]:
    """
    Compute SHAP values for a tree ensemble using TreeExplainer.

    Returns
    -------
    dict with keys:
        shap_values     : np.ndarray shape (n_samples, n_features, n_classes)
                          or (n_samples, n_features) for binary
        expected_value  : base value (float or list per class)
        feature_names   : list[str]
        mean_abs_shap   : np.ndarray (n_features,) — mean |SHAP| across classes
        class_labels    : list[str]
    """
    try:
        import shap  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "shap is required for interpretability. "
            "pip install shap --trusted-host pypi.org"
        ) from exc

    if X.shape[0] > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(X.shape[0], size=max_samples, replace=False)
        X   = X[idx]

    if model_type == "xgboost":
        try:
            import xgboost as xgb  # type: ignore
        except ImportError as exc:
            raise ImportError("xgboost required") from exc
        model = xgb.Booster()
        model.load_model(str(model_path))
        explainer = shap.TreeExplainer(model)

    elif model_type == "lightgbm":
        try:
            import lightgbm as lgb  # type: ignore
        except ImportError as exc:
            raise ImportError("lightgbm required") from exc
        model = lgb.Booster(model_file=str(model_path))
        explainer = shap.TreeExplainer(model)

    else:
        raise ValueError(f"model_type must be 'xgboost' or 'lightgbm', got '{model_type}'")

    log.info("Computing SHAP values (%s, %d samples) …", model_type, X.shape[0])
    shap_values = explainer.shap_values(X)

    # Normalise shape: ensure (n_samples, n_features, n_classes)
    if isinstance(shap_values, list):
        # lightgbm multiclass returns a list of (n_samples, n_features) per class
        shap_array = np.stack(shap_values, axis=2)   # → (n, f, c)
    elif shap_values.ndim == 2:
        shap_array = shap_values[:, :, np.newaxis]   # binary → (n, f, 1)
    else:
        shap_array = shap_values                     # already (n, f, c)

    mean_abs_shap = np.abs(shap_array).mean(axis=(0, 2))  # (n_features,)

    class_labels = [
        "Pathogenic", "Likely pathogenic", "VUS",
        "Likely benign", "Benign",
    ][:shap_array.shape[2]]

    log.info("SHAP done. Top feature: %s (mean |SHAP|=%.4f)",
             feature_names[np.argmax(mean_abs_shap)], mean_abs_shap.max())

    return {
        "shap_values":    shap_array,
        "expected_value": explainer.expected_value,
        "feature_names":  feature_names,
        "mean_abs_shap":  mean_abs_shap,
        "class_labels":   class_labels,
        "n_samples":      X.shape[0],
    }


# ---------------------------------------------------------------------------
# GradCAM — ResNet-50
# ---------------------------------------------------------------------------

def compute_gradcam(
    model_path:   Path,
    image_tensor: "torch.Tensor",   # (1, C, H, W)
    target_class: int,
    target_layer_name: str = "layer4",
    num_classes:  int = 5,
    device:       "torch.device | None" = None,
) -> np.ndarray:
    """
    Produce a GradCAM heatmap for a single image tensor.

    Returns a (H, W) numpy array with values in [0, 1].
    """
    try:
        import torch
        import torch.nn.functional as F
        from ewc_utils import build_resnet50
    except ImportError as exc:
        raise ImportError(f"PyTorch / torchvision required: {exc}") from exc

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_resnet50(num_classes, model_path, device)
    model.eval()

    # Register forward + backward hooks on the target conv layer
    target_layer = dict(model.named_modules()).get(target_layer_name)
    if target_layer is None:
        raise ValueError(f"Layer '{target_layer_name}' not found in ResNet-50.")

    activations: list[torch.Tensor] = []
    gradients:   list[torch.Tensor] = []

    def fwd_hook(_, __, output):
        activations.append(output.detach())

    def bwd_hook(_, __, grad_output):
        gradients.append(grad_output[0].detach())

    h_fwd = target_layer.register_forward_hook(fwd_hook)
    h_bwd = target_layer.register_full_backward_hook(bwd_hook)

    try:
        image_tensor = image_tensor.to(device)
        logits = model(image_tensor)
        score  = logits[0, target_class]
        model.zero_grad()
        score.backward()

        act  = activations[0].squeeze(0)   # (C, h, w)
        grad = gradients[0].squeeze(0)     # (C, h, w)

        weights = grad.mean(dim=(1, 2))    # global average pooling of gradients
        cam = (weights[:, None, None] * act).sum(dim=0)  # (h, w)
        cam = F.relu(cam)

        # Upsample to input size
        cam_up = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(image_tensor.shape[2], image_tensor.shape[3]),
            mode="bilinear",
            align_corners=False,
        ).squeeze().cpu().numpy()

        # Normalise to [0, 1]
        cam_min, cam_max = cam_up.min(), cam_up.max()
        if cam_max > cam_min:
            cam_up = (cam_up - cam_min) / (cam_max - cam_min)

        return cam_up

    finally:
        h_fwd.remove()
        h_bwd.remove()


# ---------------------------------------------------------------------------
# Feature ranking and stability
# ---------------------------------------------------------------------------

def rank_features(mean_abs_shap: np.ndarray, feature_names: list[str],
                  top_k: int) -> list[dict]:
    """
    Return top_k features sorted by mean |SHAP| descending.
    Each entry: {"rank": int, "feature": str, "mean_abs_shap": float}
    """
    order = np.argsort(mean_abs_shap)[::-1][:top_k]
    return [
        {"rank": i + 1, "feature": feature_names[idx],
         "mean_abs_shap": float(mean_abs_shap[idx])}
        for i, idx in enumerate(order)
    ]


def measure_stability(
    current_ranking:  list[dict],
    previous_ranking: list[dict] | None,
) -> dict[str, float | None]:
    """
    Compute Spearman rank correlation between two top-K feature rankings.
    Returns {"spearman_r": float | None, "n_features": int}.
    """
    if previous_ranking is None:
        return {"spearman_r": None, "n_features": len(current_ranking)}

    try:
        from scipy.stats import spearmanr  # type: ignore
    except ImportError:
        # Fallback: manual Spearman
        pass

    curr_names = [r["feature"] for r in current_ranking]
    prev_names = [r["feature"] for r in previous_ranking]
    shared     = [f for f in curr_names if f in prev_names]

    if len(shared) < 3:
        return {"spearman_r": None, "n_features": len(current_ranking),
                "note": "Too few shared features for correlation."}

    curr_ranks = {r["feature"]: r["rank"] for r in current_ranking}
    prev_ranks = {r["feature"]: r["rank"] for r in previous_ranking}

    x = [curr_ranks[f] for f in shared]
    y = [prev_ranks[f] for f in shared]

    try:
        from scipy.stats import spearmanr
        corr, pval = spearmanr(x, y)
        return {
            "spearman_r": float(corr),
            "p_value":    float(pval),
            "n_shared":   len(shared),
            "n_features": len(current_ranking),
        }
    except Exception:
        # Manual rank correlation
        n    = len(shared)
        d2   = sum((xi - yi) ** 2 for xi, yi in zip(x, y))
        corr = 1.0 - (6.0 * d2) / (n * (n ** 2 - 1))
        return {"spearman_r": float(corr), "n_shared": n,
                "n_features": len(current_ranking)}


def detect_importance_anomalies(
    current_ranking:  list[dict],
    previous_ranking: list[dict] | None,
    delta_threshold:  float,
) -> list[dict]:
    """
    Flag features whose mean |SHAP| has changed by more than delta_threshold
    (fractional) since the previous run.
    """
    if previous_ranking is None:
        return []

    prev_map = {r["feature"]: r["mean_abs_shap"] for r in previous_ranking}
    anomalies = []
    for entry in current_ranking:
        feat = entry["feature"]
        if feat not in prev_map:
            continue
        prev_val = prev_map[feat]
        if prev_val == 0:
            continue
        delta = abs(entry["mean_abs_shap"] - prev_val) / prev_val
        if delta >= delta_threshold:
            anomalies.append({
                "feature":         feat,
                "current_shap":    entry["mean_abs_shap"],
                "previous_shap":   prev_val,
                "fractional_delta": delta,
                "severity":        "high" if delta > delta_threshold * 2 else "medium",
            })
    return sorted(anomalies, key=lambda a: a["fractional_delta"], reverse=True)


def detect_biological_anomalies(
    current_ranking:             list[dict],
    expected_high_importance:    list[str],
    top_k_bio_check:             int = 10,
) -> list[dict]:
    """
    Flag features in the top-K that are not in the expected biological feature set.
    This helps catch data leakage, spurious correlations, or engineering errors.
    """
    expected_set = set(expected_high_importance)
    return [
        {
            "feature":  entry["feature"],
            "rank":     entry["rank"],
            "mean_abs_shap": entry["mean_abs_shap"],
            "issue":    "Feature not in expected high-importance biological set.",
        }
        for entry in current_ranking[:top_k_bio_check]
        if entry["feature"] not in expected_set
    ]


# ---------------------------------------------------------------------------
# HTML report generation
# ---------------------------------------------------------------------------

def _fig_to_b64(fig) -> str:
    """Encode a matplotlib Figure as a base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _make_bar_chart(ranking: list[dict], title: str, color: str = "#4f8ef7"):
    """Horizontal bar chart of top-K SHAP importances."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    n    = len(ranking)
    vals = [r["mean_abs_shap"] for r in reversed(ranking)]
    labs = [r["feature"] for r in reversed(ranking)]

    fig, ax = plt.subplots(figsize=(9, max(4, n * 0.32)))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    bars = ax.barh(range(n), vals, color=color, alpha=0.85, height=0.65)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labs, fontsize=8.5, color="#c9d1d9")
    ax.set_xlabel("Mean |SHAP value|", color="#8b949e", fontsize=9)
    ax.set_title(title, color="#e6edf3", fontsize=11, pad=10)
    ax.tick_params(colors="#8b949e", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.xaxis.label.set_color("#8b949e")
    fig.tight_layout()
    return fig


def _make_delta_chart(anomalies: list[dict]):
    """Bar chart of importance deltas for anomalous features."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    if not anomalies:
        return None

    n     = len(anomalies)
    labs  = [a["feature"] for a in anomalies]
    delts = [a["fractional_delta"] * 100 for a in anomalies]
    cols  = ["#f85149" if a["severity"] == "high" else "#d29922" for a in anomalies]

    fig, ax = plt.subplots(figsize=(9, max(3, n * 0.45)))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")
    ax.barh(range(n), delts, color=cols, alpha=0.85, height=0.6)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labs, fontsize=8.5, color="#c9d1d9")
    ax.set_xlabel("Importance change (%)", color="#8b949e", fontsize=9)
    ax.set_title("Feature importance deltas vs previous run", color="#e6edf3",
                 fontsize=11, pad=10)
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    fig.tight_layout()
    return fig


def generate_html_report(
    run_metadata:       dict,
    xgb_result:         dict | None,
    lgb_result:         dict | None,
    stability:          dict,
    importance_anomalies: list[dict],
    biological_anomalies: list[dict],
    uniprot_context:    list[dict],
    output_path:        Path,
) -> Path:
    """
    Write a self-contained HTML report.  All charts are base64-encoded PNGs —
    no external CDN dependencies.

    Returns the path to the written file.
    """
    sections: list[str] = []

    # --- Bar charts -------------------------------------------------------
    for label, result, color in [
        ("XGBoost",  xgb_result,  "#4f8ef7"),
        ("LightGBM", lgb_result,  "#2ea043"),
    ]:
        if result and result.get("ranking"):
            fig = _make_bar_chart(result["ranking"], f"{label} — Top feature importances", color)
            if fig:
                sections.append(
                    f'<h2>{label} Feature Importances</h2>'
                    f'<img src="data:image/png;base64,{_fig_to_b64(fig)}" '
                    f'style="max-width:100%;border-radius:6px;">'
                )
                import matplotlib.pyplot as plt
                plt.close(fig)

    # --- Stability section ------------------------------------------------
    spearman = stability.get("spearman_r")
    stab_color = (
        "#2ea043" if spearman is not None and spearman >= 0.85 else
        "#d29922" if spearman is not None and spearman >= 0.70 else
        "#f85149"
    )
    stab_label = (
        "Stable" if spearman is not None and spearman >= 0.85 else
        "Moderate drift" if spearman is not None and spearman >= 0.70 else
        "Unstable / large drift" if spearman is not None else
        "No prior run for comparison"
    )
    sections.append(f"""
    <h2>Feature Importance Stability</h2>
    <table>
      <tr><td>Spearman rank correlation (top-K)</td>
          <td style="color:{stab_color};font-weight:600;">
            {f"{spearman:.4f}" if spearman is not None else "—"} &nbsp; {stab_label}
          </td></tr>
      <tr><td>Shared features in comparison</td>
          <td>{stability.get("n_shared", "—")}</td></tr>
    </table>
    """)

    # --- Importance delta chart -------------------------------------------
    if importance_anomalies:
        fig = _make_delta_chart(importance_anomalies)
        if fig:
            sections.append(
                '<h2>Importance Anomalies (vs previous run)</h2>'
                f'<img src="data:image/png;base64,{_fig_to_b64(fig)}" '
                f'style="max-width:100%;border-radius:6px;">'
            )
            import matplotlib.pyplot as plt
            plt.close(fig)

        sections.append(_render_table(
            "Feature importance deltas",
            ["Feature", "Current", "Previous", "Delta %", "Severity"],
            [
                [a["feature"],
                 f"{a['current_shap']:.5f}",
                 f"{a['previous_shap']:.5f}",
                 f"{a['fractional_delta']*100:.1f}%",
                 f'<span style="color:{"#f85149" if a["severity"]=="high" else "#d29922"}">'
                 f'{a["severity"]}</span>']
                for a in importance_anomalies
            ]
        ))

    # --- Biological plausibility ------------------------------------------
    if biological_anomalies:
        sections.append('<h2 style="color:#f85149;">⚠ Biological Plausibility Flags</h2>')
        sections.append(_render_table(
            "Features unexpected in top-K",
            ["Rank", "Feature", "Mean |SHAP|", "Issue"],
            [[str(a["rank"]), a["feature"],
              f"{a['mean_abs_shap']:.5f}", a["issue"]]
             for a in biological_anomalies],
        ))

    # --- UniProt context --------------------------------------------------
    if uniprot_context:
        rows = [
            [u.get("feature",""), u.get("gene",""), u.get("protein",""),
             u.get("function_summary","")]
            for u in uniprot_context
        ]
        sections.append(_render_table(
            "UniProt context for top features",
            ["Feature", "Gene", "Protein", "Function summary"],
            rows,
        ))

    body = "\n".join(sections)
    drift_val = run_metadata.get("drift_score", "—")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SHAP Interpretability Report — {run_metadata.get("version","")}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'IBM Plex Mono', 'Courier New', monospace;
    background: #0d1117; color: #c9d1d9;
    padding: 32px 40px; max-width: 1100px; margin: 0 auto;
  }}
  h1 {{ font-size: 22px; color: #e6edf3; font-weight: 400;
        border-bottom: 1px solid #21262d; padding-bottom: 12px; margin-bottom: 24px; }}
  h2 {{ font-size: 14px; color: #e6edf3; font-weight: 600;
        letter-spacing: 1px; margin: 32px 0 12px; text-transform: uppercase; }}
  .meta {{ display: flex; gap: 32px; flex-wrap: wrap; margin-bottom: 28px; }}
  .meta-item {{ background: #161b22; border: 1px solid #21262d;
                border-radius: 6px; padding: 10px 16px; font-size: 11px; }}
  .meta-item .label {{ color: #8b949e; margin-bottom: 4px; letter-spacing: 1px;
                       text-transform: uppercase; font-size: 9px; }}
  .meta-item .value {{ color: #e6edf3; font-size: 13px; font-weight: 600; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 11px; margin-top: 8px; }}
  th {{ background: #161b22; color: #8b949e; text-align: left;
        padding: 8px 12px; border-bottom: 1px solid #21262d;
        text-transform: uppercase; font-size: 9px; letter-spacing: 1px; }}
  td {{ padding: 7px 12px; border-bottom: 1px solid #161b22; vertical-align: top; }}
  tr:hover td {{ background: #161b22; }}
  caption {{ text-align: left; color: #8b949e; font-size: 10px;
             padding: 0 0 6px; letter-spacing: 1px; text-transform: uppercase; }}
  footer {{ margin-top: 48px; font-size: 10px; color: #3d444d;
            border-top: 1px solid #21262d; padding-top: 16px; }}
</style>
</head>
<body>
<h1>SHAP Interpretability Report</h1>
<div class="meta">
  <div class="meta-item">
    <div class="label">Run version</div>
    <div class="value">{run_metadata.get("version","—")}</div>
  </div>
  <div class="meta-item">
    <div class="label">Generated</div>
    <div class="value">{run_metadata.get("generated_at","—")}</div>
  </div>
  <div class="meta-item">
    <div class="label">Corpus drift (JS)</div>
    <div class="value">{f"{drift_val:.4f}" if isinstance(drift_val, float) else drift_val}</div>
  </div>
  <div class="meta-item">
    <div class="label">Val samples</div>
    <div class="value">{run_metadata.get("val_samples","—")}</div>
  </div>
  <div class="meta-item">
    <div class="label">Importance anomalies</div>
    <div class="value" style="color:{'#f85149' if run_metadata.get('n_importance_anomalies',0) else '#2ea043'}">
      {run_metadata.get("n_importance_anomalies", 0)}
    </div>
  </div>
  <div class="meta-item">
    <div class="label">Biological flags</div>
    <div class="value" style="color:{'#f85149' if run_metadata.get('n_biological_anomalies',0) else '#2ea043'}">
      {run_metadata.get("n_biological_anomalies", 0)}
    </div>
  </div>
</div>

{body}

<footer>
  Genomic Variant Classifier · Interpretability Agent · auto-generated
</footer>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    log.info("SHAP HTML report written → %s", output_path)
    return output_path


def _render_table(caption: str, headers: list[str], rows: list[list]) -> str:
    th_html = "".join(f"<th>{h}</th>" for h in headers)
    row_html = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f"""<table>
  <caption>{caption}</caption>
  <thead><tr>{th_html}</tr></thead>
  <tbody>{row_html}</tbody>
</table>"""
