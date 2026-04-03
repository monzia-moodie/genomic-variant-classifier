"""
agents/interpretability_agent.py
==================================
Interpretability Agent — runs SHAP analysis after every training update,
audits feature stability, checks biological plausibility, and generates
a self-contained HTML report.

Pipeline
--------
1. Load current model checkpoint from SharedState
2. Load validation parquet  →  X_val, y_val, feature_names
3. _analyze_ensemble()
      a. XGBoost  TreeExplainer  → SHAP values, ranking, stability delta
      b. LightGBM TreeExplainer  → same
4. _analyze_resnet()
      GradCAM on a stratified sample of validation histopathology images
5. _check_biological_plausibility()
      Compare top-K features against EXPECTED_HIGH_IMPORTANCE_FEATURES
      Optional: fetch UniProt protein context for top tabular features
6. _detect_and_flag_anomalies()
      Importance instability + biological flags → pending_review items
7. _generate_report()
      Self-contained HTML saved to SHAP_REPORT_DIR and mirrored to GDrive
8. Update SharedState
      shap_report_path, shap_top_features, shap_last_run
"""

from __future__ import annotations

import json
import logging
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_AL = Path(__file__).resolve().parent.parent
for _p in (str(_AL), str(_AL / "agents")):
    if _p not in sys.path: sys.path.insert(0, _p)

from base_agent import AgentResult, BaseAgent
from shap_utils import (
    compute_ensemble_shap,
    detect_biological_anomalies,
    detect_importance_anomalies,
    generate_html_report,
    measure_stability,
    rank_features,
)
from config import (
    CHECKPOINT_DIR,
    ENSEMBLE_SUBDIR,
    EXPECTED_HIGH_IMPORTANCE_FEATURES,
    GDRIVE_CHECKPOINT_DIR,
    GTEX_API_BASE,
    PROCESSED_DATA_DIR,
    RESNET_NUM_CLASSES,
    RESNET_SUBDIR,
    SHAP_IMPORTANCE_DELTA,
    SHAP_REPORT_DIR,
    SHAP_STABILITY_THRESHOLD,
    SHAP_TOP_K,
    SHAP_VAL_SAMPLES,
    UNIPROT_API_BASE,
    VAL_PARQUET,
)
from shared_state import SharedState

log = logging.getLogger("InterpretabilityAgent")


class InterpretabilityAgent(BaseAgent):
    """
    Post-training SHAP audit for the genomic variant classifier.
    Runs after TrainingLifecycleAgent; safe to run independently.
    """

    def __init__(self, state: SharedState, dry_run: bool = False):
        super().__init__(state, dry_run)
        SHAP_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        return "InterpretabilityAgent"

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(self) -> AgentResult:
        results: dict[str, Any] = {
            "xgb_analyzed":           False,
            "lgb_analyzed":           False,
            "resnet_analyzed":        False,
            "importance_anomalies":   [],
            "biological_anomalies":   [],
            "stability_xgb":          {},
            "stability_lgb":          {},
            "report_path":            None,
            "errors":                 [],
        }

        # ---- 1. Locate checkpoint ---------------------------------------
        checkpoint_ref = self.state.get("model_checkpoint_ref")
        if not checkpoint_ref:
            msg = "No model_checkpoint_ref in shared state — nothing to analyse."
            self.log.warning(msg)
            return AgentResult(success=False, action="analyze", errors=[msg])

        checkpoint_dir  = Path(checkpoint_ref)
        ensemble_dir    = checkpoint_dir / ENSEMBLE_SUBDIR
        resnet_dir      = checkpoint_dir / RESNET_SUBDIR

        # ---- 2. Load validation data ------------------------------------
        X_val, y_val, feature_names = self._load_val_data()
        if X_val is None:
            msg = "Could not load validation data — skipping SHAP analysis."
            self.log.warning(msg)
            return AgentResult(success=False, action="analyze", errors=[msg])

        # Previous rankings for stability comparison
        prev_rankings = self._load_previous_rankings()

        # ---- 3. Ensemble SHAP -------------------------------------------
        xgb_result = self._analyze_model(
            model_path=ensemble_dir / "xgb_model.ubj",
            X_val=X_val,
            feature_names=feature_names,
            model_type="xgboost",
            previous_ranking=prev_rankings.get("xgboost"),
        )
        results["xgb_analyzed"]       = xgb_result.get("analyzed", False)
        results["stability_xgb"]      = xgb_result.get("stability", {})
        results["importance_anomalies"] += xgb_result.get("importance_anomalies", [])
        results["biological_anomalies"] += xgb_result.get("biological_anomalies", [])
        if xgb_result.get("error"):
            results["errors"].append(f"XGBoost SHAP: {xgb_result['error']}")

        lgb_result = self._analyze_model(
            model_path=ensemble_dir / "lgb_model.txt",
            X_val=X_val,
            feature_names=feature_names,
            model_type="lightgbm",
            previous_ranking=prev_rankings.get("lightgbm"),
        )
        results["lgb_analyzed"]         = lgb_result.get("analyzed", False)
        results["stability_lgb"]        = lgb_result.get("stability", {})
        results["importance_anomalies"] += lgb_result.get("importance_anomalies", [])
        results["biological_anomalies"] += lgb_result.get("biological_anomalies", [])
        if lgb_result.get("error"):
            results["errors"].append(f"LightGBM SHAP: {lgb_result['error']}")

        # ---- 4. ResNet GradCAM ------------------------------------------
        resnet_result = self._analyze_resnet(resnet_dir)
        results["resnet_analyzed"] = resnet_result.get("analyzed", False)
        if resnet_result.get("error"):
            results["errors"].append(f"GradCAM: {resnet_result['error']}")

        # ---- 5. UniProt context -----------------------------------------
        top_features = self._top_features_union(xgb_result, lgb_result, n=10)
        uniprot_ctx  = self._fetch_uniprot_context(top_features)

        # ---- 6. Flag anomalies ------------------------------------------
        self._flag_anomalies(results)

        # ---- 7. Generate report -----------------------------------------
        if self.dry_run:
            self._dry_run_log("Would generate SHAP HTML report.")
        else:
            try:
                report_path = self._generate_report(
                    xgb_result, lgb_result, results, uniprot_ctx
                )
                results["report_path"] = str(report_path)
                self._mirror_report(report_path)
            except Exception as exc:
                msg = f"Report generation failed: {exc}"
                self.log.error("%s\n%s", msg, traceback.format_exc())
                results["errors"].append(msg)

        # ---- 8. Persist rankings + metadata to state --------------------
        self._update_state(xgb_result, lgb_result, results)

        success = (results["xgb_analyzed"] or results["lgb_analyzed"]) \
                  and not results["errors"]
        return AgentResult(
            success=success,
            action="analyze",
            details=results,
            errors=results["errors"],
        )

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_val_data(self) -> tuple[np.ndarray | None, np.ndarray | None, list[str]]:
        try:
            import pandas as pd
        except ImportError:
            self.log.error("pandas not available.")
            return None, None, []

        if not VAL_PARQUET.exists():
            self.log.warning("val.parquet not found at %s", VAL_PARQUET)
            return None, None, []

        try:
            df         = pd.read_parquet(VAL_PARQUET)
            label_col  = "pathogenicity_class"
            if label_col not in df.columns:
                self.log.error("Label column '%s' missing from val.parquet.", label_col)
                return None, None, []

            feature_cols = [c for c in df.columns if c != label_col]
            X = df[feature_cols].values.astype(np.float32)
            y = df[label_col].values
            self.log.info("Validation data loaded: %d samples, %d features.",
                          X.shape[0], X.shape[1])
            return X, y, feature_cols
        except Exception as exc:
            self.log.error("Failed to load val.parquet: %s", exc)
            return None, None, []

    def _load_previous_rankings(self) -> dict[str, list[dict]]:
        raw = self.state.get("shap_top_features") or {}
        return raw if isinstance(raw, dict) else {}

    # ------------------------------------------------------------------
    # Single-model SHAP analysis
    # ------------------------------------------------------------------

    def _analyze_model(
        self,
        model_path:       Path,
        X_val:            np.ndarray,
        feature_names:    list[str],
        model_type:       str,
        previous_ranking: list[dict] | None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "analyzed":            False,
            "ranking":             [],
            "stability":           {},
            "importance_anomalies": [],
            "biological_anomalies": [],
            "error":               None,
        }

        if not model_path.exists():
            result["error"] = f"{model_path.name} not found."
            return result

        if self.dry_run:
            self._dry_run_log(f"Would compute {model_type} SHAP values.")
            return result

        try:
            shap_data = compute_ensemble_shap(
                model_path=model_path,
                X=X_val,
                feature_names=feature_names,
                model_type=model_type,
                max_samples=SHAP_VAL_SAMPLES,
            )
        except Exception as exc:
            result["error"] = str(exc)
            self.log.error("SHAP computation failed for %s: %s", model_type, exc)
            return result

        ranking = rank_features(
            shap_data["mean_abs_shap"], feature_names, SHAP_TOP_K
        )
        result["ranking"]  = ranking
        result["analyzed"] = True

        # Stability
        result["stability"] = measure_stability(ranking, previous_ranking)
        spearman = result["stability"].get("spearman_r")
        if spearman is not None and spearman < SHAP_STABILITY_THRESHOLD:
            self.log.warning(
                "%s SHAP stability low: Spearman r=%.3f < threshold %.3f",
                model_type, spearman, SHAP_STABILITY_THRESHOLD,
            )

        # Per-feature importance deltas
        result["importance_anomalies"] = detect_importance_anomalies(
            ranking, previous_ranking, SHAP_IMPORTANCE_DELTA
        )

        # Biological plausibility
        result["biological_anomalies"] = detect_biological_anomalies(
            ranking, EXPECTED_HIGH_IMPORTANCE_FEATURES, top_k_bio_check=10
        )

        self.log.info(
            "%s SHAP: top=%s (%.5f)  stability=%.3f  imp_anomalies=%d  bio_flags=%d",
            model_type,
            ranking[0]["feature"] if ranking else "—",
            ranking[0]["mean_abs_shap"] if ranking else 0,
            spearman if spearman is not None else float("nan"),
            len(result["importance_anomalies"]),
            len(result["biological_anomalies"]),
        )
        return result

    # ------------------------------------------------------------------
    # ResNet GradCAM
    # ------------------------------------------------------------------

    def _analyze_resnet(self, resnet_dir: Path) -> dict[str, Any]:
        """
        Run GradCAM on a stratified sample of validation histopathology images.
        Saves per-class average heatmaps as PNG files in resnet_dir/gradcam/.
        """
        result: dict[str, Any] = {"analyzed": False, "error": None}
        model_path = resnet_dir / "model.pt"

        if not model_path.exists():
            result["error"] = "ResNet model.pt not found — skipping GradCAM."
            return result

        val_img_dir = PROCESSED_DATA_DIR / "images" / "val"
        if not val_img_dir.exists():
            result["error"] = "Validation image directory not found."
            return result

        if self.dry_run:
            self._dry_run_log("Would run GradCAM on validation images.")
            return result

        try:
            import torch
            from torchvision import datasets, transforms  # type: ignore
            from shap_utils import compute_gradcam

            tfm = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225],
                ),
            ])
            device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dataset = datasets.ImageFolder(str(val_img_dir), transform=tfm)

            out_dir = resnet_dir / "gradcam"
            out_dir.mkdir(exist_ok=True)

            # Accumulate average heatmaps per class
            class_heatmaps: dict[int, list[np.ndarray]] = {}
            n_samples_per_class = 5

            for class_idx in range(RESNET_NUM_CLASSES):
                class_indices = [
                    i for i, (_, lbl) in enumerate(dataset.samples)
                    if lbl == class_idx
                ][:n_samples_per_class]

                heatmaps = []
                for idx in class_indices:
                    img_tensor, _ = dataset[idx]
                    img_tensor = img_tensor.unsqueeze(0)
                    try:
                        cam = compute_gradcam(
                            model_path, img_tensor, class_idx,
                            num_classes=RESNET_NUM_CLASSES, device=device,
                        )
                        heatmaps.append(cam)
                    except Exception as exc:
                        self.log.warning("GradCAM failed for sample %d: %s", idx, exc)

                if heatmaps:
                    avg_heatmap = np.mean(heatmaps, axis=0)
                    self._save_heatmap(
                        avg_heatmap,
                        out_dir / f"class_{class_idx}_avg_gradcam.png",
                        title=f"Class {class_idx} — avg GradCAM ({len(heatmaps)} samples)",
                    )

            result["analyzed"]    = True
            result["gradcam_dir"] = str(out_dir)
            self.log.info("GradCAM saved to %s", out_dir)

        except ImportError as exc:
            result["error"] = f"PyTorch / torchvision not available: {exc}"
        except Exception as exc:
            result["error"] = f"GradCAM failed: {exc}"
            self.log.error("%s\n%s", result["error"], traceback.format_exc())

        return result

    @staticmethod
    def _save_heatmap(heatmap: np.ndarray, path: Path, title: str = "") -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_facecolor("#0f1117")
            ax.set_facecolor("#0f1117")
            im = ax.imshow(heatmap, cmap="inferno", vmin=0, vmax=1)
            plt.colorbar(im, ax=ax)
            ax.set_title(title, color="#e6edf3", fontsize=9)
            ax.axis("off")
            fig.tight_layout()
            fig.savefig(path, dpi=90, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            plt.close(fig)
        except Exception as exc:
            log.warning("Could not save heatmap: %s", exc)

    # ------------------------------------------------------------------
    # Biological plausibility — UniProt context
    # ------------------------------------------------------------------

    def _fetch_uniprot_context(self, features: list[str]) -> list[dict]:
        """
        For features whose names contain a gene symbol (e.g. "BRCA1_conservation"),
        fetch a brief UniProt summary.  Fails gracefully — a network error here
        should never abort the agent.
        """
        import re
        import requests

        ctx = []
        seen_genes: set[str] = set()

        # Extract likely gene names from feature names
        gene_pattern = re.compile(r"\b([A-Z][A-Z0-9]{1,9})\b")

        for feat in features:
            genes = gene_pattern.findall(feat)
            for gene in genes:
                if gene in seen_genes or len(gene) < 3:
                    continue
                seen_genes.add(gene)

                try:
                    resp = requests.get(
                        f"{UNIPROT_API_BASE}/search",
                        params={
                            "query": f"gene:{gene} AND organism_id:9606 AND reviewed:true",
                            "fields": "gene_names,protein_name,function",
                            "format": "json",
                            "size":   1,
                        },
                        timeout=8,
                    )
                    resp.raise_for_status()
                    results = resp.json().get("results", [])
                    if not results:
                        continue

                    entry  = results[0]
                    pnames = entry.get("proteinDescription", {})
                    rec_name = (
                        pnames.get("recommendedName", {})
                              .get("fullName", {})
                              .get("value", "")
                    )
                    func_comments = [
                        c.get("texts", [{}])[0].get("value", "")
                        for c in entry.get("comments", [])
                        if c.get("commentType") == "FUNCTION"
                    ]
                    func_summary = func_comments[0][:200] if func_comments else ""

                    ctx.append({
                        "feature":          feat,
                        "gene":             gene,
                        "protein":          rec_name,
                        "function_summary": func_summary,
                    })
                    self.log.info("UniProt context fetched for gene: %s", gene)

                except requests.RequestException as exc:
                    self.log.warning("UniProt lookup failed for %s: %s", gene, exc)

        return ctx

    # ------------------------------------------------------------------
    # Anomaly flagging
    # ------------------------------------------------------------------

    def _flag_anomalies(self, results: dict) -> None:
        # Stability flags
        for model_key, stab_key in [("xgb_analyzed", "stability_xgb"),
                                     ("lgb_analyzed", "stability_lgb")]:
            if not results.get(model_key):
                continue
            stab  = results.get(stab_key, {})
            spear = stab.get("spearman_r")
            if spear is not None and spear < SHAP_STABILITY_THRESHOLD:
                model_label = "XGBoost" if "xgb" in stab_key else "LightGBM"
                self.state.add_pending_review({
                    "reason":  f"{model_label} SHAP ranking instability detected",
                    "spearman_r": spear,
                    "threshold":  SHAP_STABILITY_THRESHOLD,
                    "agent":   self.name,
                    "action_required": (
                        f"{model_label} feature importance ranking has shifted "
                        f"significantly (Spearman r={spear:.3f}). "
                        "Review whether this reflects genuine data change or a "
                        "training artefact before deploying the model."
                    ),
                })

        # Biological anomaly flags
        for anom in results.get("biological_anomalies", []):
            self.state.add_pending_review({
                "reason":  "Biologically unexpected feature in top-K SHAP ranking",
                "feature": anom["feature"],
                "rank":    anom["rank"],
                "mean_abs_shap": anom["mean_abs_shap"],
                "agent":   self.name,
                "action_required": (
                    f"Feature '{anom['feature']}' (rank {anom['rank']}) is not in "
                    "the expected high-importance biological feature set. "
                    "Investigate for data leakage, label leakage, or feature "
                    "engineering error before deployment."
                ),
            })

        # High-severity importance delta flags
        for anom in results.get("importance_anomalies", []):
            if anom.get("severity") == "high":
                self.state.add_pending_review({
                    "reason":  "Large feature importance shift detected",
                    "feature": anom["feature"],
                    "delta_pct": f"{anom['fractional_delta']*100:.1f}%",
                    "agent":   self.name,
                    "action_required": (
                        f"Feature '{anom['feature']}' importance changed by "
                        f"{anom['fractional_delta']*100:.1f}% since the last run. "
                        "Verify that upstream data for this feature is consistent."
                    ),
                })

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def _generate_report(
        self,
        xgb_result:  dict,
        lgb_result:  dict,
        results:     dict,
        uniprot_ctx: list[dict],
    ) -> Path:
        now = datetime.now(timezone.utc)
        version_tag = now.strftime("%Y%m%dT%H%M%SZ")

        run_meta = {
            "version":                version_tag,
            "generated_at":           now.isoformat(),
            "drift_score":            self.state.get("drift_score"),
            "val_samples":            SHAP_VAL_SAMPLES,
            "n_importance_anomalies": len(results.get("importance_anomalies", [])),
            "n_biological_anomalies": len(results.get("biological_anomalies", [])),
        }

        report_path = SHAP_REPORT_DIR / f"shap_report_{version_tag}.html"
        return generate_html_report(
            run_metadata=run_meta,
            xgb_result=xgb_result if xgb_result.get("analyzed") else None,
            lgb_result=lgb_result if lgb_result.get("analyzed") else None,
            stability=results.get("stability_xgb", {}),
            importance_anomalies=results.get("importance_anomalies", []),
            biological_anomalies=results.get("biological_anomalies", []),
            uniprot_context=uniprot_ctx,
            output_path=report_path,
        )

    def _mirror_report(self, report_path: Path) -> None:
        try:
            gdrive_reports = GDRIVE_CHECKPOINT_DIR.parent / "reports" / "shap"
            if gdrive_reports.parent.parent.exists():
                gdrive_reports.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copy2(report_path, gdrive_reports / report_path.name)
                self.log.info("Report mirrored to GDrive → %s", gdrive_reports)
        except Exception as exc:
            self.log.warning("GDrive report mirror failed: %s", exc)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _top_features_union(xgb: dict, lgb: dict, n: int) -> list[str]:
        """Return the top-n features by mean |SHAP| across both models."""
        scores: dict[str, float] = {}
        for result in (xgb, lgb):
            for entry in result.get("ranking", [])[:n]:
                feat = entry["feature"]
                scores[feat] = max(scores.get(feat, 0), entry["mean_abs_shap"])
        return [k for k, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)][:n]

    def _update_state(self, xgb_result: dict, lgb_result: dict, results: dict) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.state.set("shap_last_run", now)

        if results.get("report_path"):
            self.state.set("shap_report_path", results["report_path"])

        # Persist top feature rankings for stability comparison next run
        rankings: dict[str, list[dict]] = {}
        if xgb_result.get("ranking"):
            rankings["xgboost"]  = xgb_result["ranking"]
        if lgb_result.get("ranking"):
            rankings["lightgbm"] = lgb_result["ranking"]
        if rankings:
            self.state.set("shap_top_features", rankings)

        self.log.info(
            "State updated: shap_last_run=%s  report=%s",
            now, results.get("report_path", "—"),
        )
