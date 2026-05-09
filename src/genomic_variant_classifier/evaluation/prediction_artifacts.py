"""
src/evaluation/prediction_artifacts.py
=======================================
RunArtifactWriter — one-stop shop for emitting the full artefact set
defined in docs/RUN9_SCIENTIFIC_DESIGN.md §Rule 5.

Design principles:
  - Local-first: every write lands on disk before any upload.
  - Atomic: writes go to .tmp then rename; a failed write never corrupts
    an existing artefact.
  - Cheap: uses parquet (columnar, compressed) for tables, JSON for
    config/graph stats. Never pickle (not portable, not auditable).
  - No logging.basicConfig at module level (library-module rule).
  - from __future__ import annotations (standing rule #N).

Every method is callable in any order EXCEPT save_manifest() which should
be called first so downstream artefacts can reference the manifest path if
needed. upload_to_gcs() is a single final call that mirrors the entire
output directory to the configured bucket.

Integration example (see scripts/run_phase2_eval.py):
    writer = RunArtifactWriter(run_id="run9", ablation="full", output_dir=path)
    writer.save_manifest(git_sha=..., versions=..., config=...)
    writer.save_test_predictions(y_test, proba, base_probs, meta)
    writer.save_eval_report(report)
    writer.save_calibration(report)
    writer.save_shap_values(ensemble, base_probs, meta, top_k=20)
    writer.save_permutation_importance(ensemble, X_tab, X_seq, y)
    writer.save_graph_stats(stats_dict)              # GNN runs only
    writer.upload_to_gcs(bucket="genomic-variant-prod-outputs")
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------
class RunArtifactWriter:
    """
    Writes the full artefact set for a single ablation of a single run.
    All files land under `output_dir/`; `upload_to_gcs()` mirrors that
    directory into `gs://<bucket>/runs/<run_id>/<ablation>/`.
    """

    def __init__(
        self,
        run_id: str,
        ablation: str,
        output_dir: Path,
    ) -> None:
        if not run_id or not ablation:
            raise ValueError("run_id and ablation are required")
        self.run_id = run_id
        self.ablation = ablation
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._artefacts: list[str] = []

    # ── Atomic write helper ────────────────────────────────────────────────

    def _atomic_write(
        self,
        filename: str,
        write_fn,  # callable(Path) -> None that writes the content
    ) -> Path:
        """
        Write to a `.tmp` sibling, fsync, then rename. Crash-safe: either the
        old file remains (or no file), or the new file is complete.
        """
        dst = self.output_dir / filename
        tmp_fd, tmp_name = tempfile.mkstemp(
            prefix=f".{filename}.",
            suffix=".tmp",
            dir=str(self.output_dir),
        )
        os.close(tmp_fd)
        tmp = Path(tmp_name)
        try:
            write_fn(tmp)
            # Ensure data is on disk before rename
            # Windows: os.fsync requires a writable fd (_commit requires write access)
            with open(tmp, "r+b") as fh:
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, dst)
        except Exception:
            if tmp.exists():
                tmp.unlink()
            raise
        self._artefacts.append(filename)
        return dst

    # ── Manifest ───────────────────────────────────────────────────────────

    def save_manifest(
        self,
        git_sha: str,
        versions: dict[str, str],
        config: dict[str, Any],
    ) -> Path:
        manifest = {
            "run_id": self.run_id,
            "ablation": self.ablation,
            "git_sha": git_sha,
            "versions": versions,
            "config": config,
        }

        def _write(path: Path) -> None:
            path.write_text(json.dumps(manifest, indent=2, default=str))

        dst = self._atomic_write("manifest.json", _write)
        logger.info("Manifest written: %s", dst)
        return dst

    # ── Out-of-fold predictions (training) ────────────────────────────────

    def save_oof_predictions(
        self,
        oof_df: pd.DataFrame,
    ) -> Path:
        """
        oof_df: one row per training example, columns:
            variant_id, gene_symbol, fold, label,
            <base_model_1>_prob, <base_model_2>_prob, ..., ensemble_prob
        """
        required = {"variant_id", "fold", "label"}
        missing = required - set(oof_df.columns)
        if missing:
            raise ValueError(f"oof_df missing required cols: {missing}")

        def _write(path: Path) -> None:
            oof_df.to_parquet(path, index=False, compression="zstd")

        dst = self._atomic_write("oof_predictions.parquet", _write)
        logger.info(
            "OOF predictions: %d rows, %d cols -> %s", len(oof_df), oof_df.shape[1], dst
        )
        return dst

    # ── Test predictions ───────────────────────────────────────────────────

    def save_test_predictions(
        self,
        y_test: pd.Series | np.ndarray,
        proba: np.ndarray,
        base_probs: dict[str, np.ndarray],
        meta: pd.DataFrame,
    ) -> Path:
        y = np.asarray(y_test)
        n = len(y)
        if len(proba) != n:
            raise ValueError(f"len(proba)={len(proba)} != len(y_test)={n}")
        for name, vec in base_probs.items():
            if len(vec) != n:
                raise ValueError(f"len(base_probs[{name}])={len(vec)} != {n}")

        out = pd.DataFrame(
            {
                "label": y,
                "ensemble_prob": np.asarray(proba),
            }
        )
        for name, vec in base_probs.items():
            out[f"{name}_prob"] = np.asarray(vec)

        # Attach metadata columns that are useful for downstream analysis
        for col in (
            "variant_id",
            "gene_symbol",
            "consequence",
            "chrom",
            "pos",
            "ref",
            "alt",
        ):
            if col in meta.columns:
                out[col] = meta[col].reset_index(drop=True).values

        def _write(path: Path) -> None:
            out.to_parquet(path, index=False, compression="zstd")

        dst = self._atomic_write("test_predictions.parquet", _write)
        logger.info(
            "Test predictions: %d rows x %d cols -> %s", len(out), out.shape[1], dst
        )
        return dst

    # ── Evaluation report (JSON) ───────────────────────────────────────────

    def save_eval_report(self, report: Any) -> Path:
        """
        Accepts a ClinicalEvaluator EvaluationReport dataclass OR a plain dict.
        Both are serialised to JSON.
        """
        if is_dataclass(report):
            payload = asdict(report)
        elif isinstance(report, dict):
            payload = report
        else:
            raise TypeError(f"Unsupported report type: {type(report)}")

        def _write(path: Path) -> None:
            path.write_text(json.dumps(payload, indent=2, default=str))

        dst = self._atomic_write("eval_report.json", _write)
        logger.info("Eval report: %s", dst)
        return dst

    # ── Per-consequence calibration ────────────────────────────────────────

    def save_calibration(self, report: Any) -> Path:
        """
        Extracts global calibration curve + per-consequence breakdown from
        a ClinicalEvaluator EvaluationReport and writes a long-format table.

        Schema:
            scope:       'global' or consequence name
            bucket:      bin index 0..n_bins-1
            mean_pred:   mean predicted probability in bucket
            frac_pos:    empirical positive rate in bucket
            n:           samples in bucket
        """
        rows: list[dict[str, Any]] = []
        cal_frac = getattr(report, "calibration_frac_pos", None) or []
        cal_pred = getattr(report, "calibration_mean_pred", None) or []
        for i, (fp, mp) in enumerate(zip(cal_frac, cal_pred)):
            rows.append(
                {
                    "scope": "global",
                    "bucket": i,
                    "mean_pred": float(mp),
                    "frac_pos": float(fp),
                    "n": -1,
                }
            )

        # Per-consequence — if the evaluator returned a breakdown, use it
        conseq = getattr(report, "consequence_breakdown", None) or []
        for c in conseq:
            d = asdict(c) if is_dataclass(c) else c
            rows.append(
                {
                    "scope": d.get("consequence", "unknown"),
                    "bucket": -1,
                    "mean_pred": float(d.get("auroc", np.nan)),
                    "frac_pos": float(d.get("prevalence", np.nan)),
                    "n": int(d.get("n_total", -1)),
                }
            )

        df = pd.DataFrame(rows)

        def _write(path: Path) -> None:
            df.to_parquet(path, index=False, compression="zstd")

        dst = self._atomic_write("calibration.parquet", _write)
        logger.info("Calibration: %d rows -> %s", len(df), dst)
        return dst

    # ── SHAP ───────────────────────────────────────────────────────────────

    def save_shap_values(
        self,
        ensemble: Any,
        base_probs: dict[str, np.ndarray],
        meta: pd.DataFrame,
        top_k: int = 20,
    ) -> Path:
        """
        Compute SHAP at the *stacker* level: the stacker is a LogisticRegression
        over the base-model probability matrix, so LinearExplainer is exact and
        cheap. Output is per-variant top-K features by |SHAP|.

        Schema:
            variant_id, gene_symbol, rank, feature, shap_value
        """
        import shap

        if not hasattr(ensemble, "meta_learner"):
            raise AttributeError("ensemble has no meta_learner attribute")
        stacker = ensemble.meta_learner
        feature_names = [f"{n}_prob" for n in base_probs]
        X = np.column_stack([base_probs[n] for n in base_probs])

        explainer = shap.LinearExplainer(stacker, X)
        shap_vals = explainer.shap_values(X)
        # For binary logistic, shap_values is (n_samples, n_features)

        rows: list[dict[str, Any]] = []
        ids = meta.get("variant_id", pd.Series([f"row_{i}" for i in range(len(X))]))
        genes = meta.get("gene_symbol", pd.Series([""] * len(X)))
        for i in range(len(X)):
            # Top-K by |SHAP|
            vals = shap_vals[i]
            order = np.argsort(-np.abs(vals))[:top_k]
            for rank, j in enumerate(order):
                rows.append(
                    {
                        "variant_id": str(ids.iloc[i]),
                        "gene_symbol": str(genes.iloc[i]),
                        "rank": rank,
                        "feature": feature_names[j],
                        "shap_value": float(vals[j]),
                    }
                )
        df = pd.DataFrame(rows)

        def _write(path: Path) -> None:
            df.to_parquet(path, index=False, compression="zstd")

        dst = self._atomic_write("shap_values.parquet", _write)
        logger.info(
            "SHAP values: %d rows (%d variants x top-%d) -> %s",
            len(df),
            len(X),
            top_k,
            dst,
        )
        return dst

    # ── Permutation importance ─────────────────────────────────────────────

    def save_permutation_importance(
        self,
        ensemble: Any,
        X_tab_test: pd.DataFrame,
        X_seq_test: pd.Series,
        y_test: pd.Series | np.ndarray,
        n_repeats: int = 5,
        sample_size: int = 50_000,
        seed: int = 42,
    ) -> Path:
        """
        Permutation importance on the stacker's ensemble output.
        Subsamples for tractability; the unbiased estimator is the gold
        standard (vs. built-in feature_importances_ which is split-count-
        biased).

        Schema: feature, mean_drop, std_drop, n_repeats
        """
        from sklearn.metrics import roc_auc_score

        y = np.asarray(y_test)
        rng = np.random.default_rng(seed)
        n = len(y)
        if n > sample_size:
            idx = rng.choice(n, size=sample_size, replace=False)
            X_tab_sub = X_tab_test.iloc[idx].reset_index(drop=True)
            X_seq_sub = X_seq_test.iloc[idx].reset_index(drop=True)
            y_sub = y[idx]
        else:
            X_tab_sub, X_seq_sub, y_sub = X_tab_test, X_seq_test, y

        baseline = roc_auc_score(
            y_sub, ensemble.predict_proba(X_tab_sub, X_seq_sub)[:, 1]
        )
        feature_cols = list(X_tab_sub.columns)
        rows: list[dict[str, Any]] = []
        for col in feature_cols:
            drops = []
            for _ in range(n_repeats):
                X_perm = X_tab_sub.copy()
                X_perm[col] = rng.permutation(X_perm[col].values)
                auc_perm = roc_auc_score(
                    y_sub, ensemble.predict_proba(X_perm, X_seq_sub)[:, 1]
                )
                drops.append(baseline - auc_perm)
            rows.append(
                {
                    "feature": col,
                    "mean_drop": float(np.mean(drops)),
                    "std_drop": float(np.std(drops)),
                    "n_repeats": n_repeats,
                    "baseline": float(baseline),
                }
            )

        df = pd.DataFrame(rows).sort_values("mean_drop", ascending=False)

        def _write(path: Path) -> None:
            df.to_parquet(path, index=False, compression="zstd")

        dst = self._atomic_write("feature_importance.parquet", _write)
        logger.info("Permutation importance: %d features -> %s", len(df), dst)
        return dst

    # ── Graph stats (GNN only) ─────────────────────────────────────────────

    def save_graph_stats(self, stats: dict[str, Any]) -> Path:
        """Writes the graph_stats.json required for any GNN-live run."""
        required = {"node_count", "edge_count"}
        missing = required - set(stats.keys())
        if missing:
            raise ValueError(f"graph_stats missing required keys: {missing}")

        def _write(path: Path) -> None:
            path.write_text(json.dumps(stats, indent=2, default=str))

        dst = self._atomic_write("graph_stats.json", _write)
        logger.info(
            "Graph stats: %d nodes / %d edges -> %s",
            stats["node_count"],
            stats["edge_count"],
            dst,
        )
        return dst

    # ── Ablation results aggregator (written by top-level driver) ─────────

    def append_ablation_row(
        self,
        master_path: Path,
        row: dict[str, Any],
    ) -> None:
        """
        Appends a single ablation's headline metrics to a shared
        ablation_results.parquet at the RUN level (one level up from the
        ablation-specific output_dir). Idempotent: re-running the same
        ablation replaces the row for that ablation name.
        """
        row = dict(row)
        row.setdefault("ablation", self.ablation)
        new_df = pd.DataFrame([row])
        if master_path.exists():
            existing = pd.read_parquet(master_path)
            existing = existing[existing["ablation"] != self.ablation]
            merged = pd.concat([existing, new_df], ignore_index=True)
        else:
            merged = new_df
        master_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(master_path, index=False, compression="zstd")
        logger.info(
            "Ablation results appended: %s rows -> %s", len(merged), master_path
        )

    # ── Upload ─────────────────────────────────────────────────────────────

    def upload_to_gcs(
        self,
        bucket: str = "genomic-variant-prod-outputs",
        gcs_prefix: str = "runs",
    ) -> None:
        """
        Mirror the entire output_dir into
            gs://<bucket>/<gcs_prefix>/<run_id>/<ablation>/

        Uses `gcloud storage cp --recursive` (gsutil is deprecated per
        standing rule). Single call at end of run; do not stream uploads
        during training.
        """
        gcloud = shutil.which("gcloud")
        if not gcloud:
            logger.error(
                "gcloud not on PATH; cannot upload. " "Artefacts remain at %s",
                self.output_dir,
            )
            return

        dst_uri = f"gs://{bucket}/{gcs_prefix}/{self.run_id}/{self.ablation}/"
        logger.info("Uploading %s -> %s", self.output_dir, dst_uri)
        proc = subprocess.run(
            [
                gcloud,
                "storage",
                "cp",
                "--recursive",
                str(self.output_dir) + "/",
                dst_uri,
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            logger.error("GCS upload failed (rc=%d): %s", proc.returncode, proc.stderr)
            raise RuntimeError(f"gcs upload failed: {proc.stderr}")
        logger.info("Upload complete. %d artefacts uploaded.", len(self._artefacts))

    # ── Convenience ────────────────────────────────────────────────────────

    @property
    def artefacts(self) -> list[str]:
        """Filenames (relative to output_dir) written so far."""
        return list(self._artefacts)
