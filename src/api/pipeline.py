"""
src/api/pipeline.py
===================
Serialisable InferencePipeline that bundles trained base models and a
stacking meta-learner into a single joblib artifact.

Build the artifact with scripts/export_model.py after a successful
run_phase2_eval.py run.  Load it in the API with InferencePipeline.load().

Usage (inference):
    pipe = InferencePipeline.load("models/phase2_pipeline.joblib")
    result = pipe.predict_single({
        "chrom": "17", "pos": 43094692, "ref": "G", "alt": "A",
        "consequence": "missense_variant",
        "gene_symbol": "BRCA1",
        "alphamissense_score": 0.94,
        "allele_freq": 0.0,
    })
    # -> {"pathogenicity_score": 0.97, "classification": "Pathogenic", "confidence": "high"}
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.models.variant_ensemble import TABULAR_FEATURES, engineer_features

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature contract — locked to the exact 55 columns from X_train.parquet
# ---------------------------------------------------------------------------

INFERENCE_FEATURE_COLUMNS: list[str] = list(TABULAR_FEATURES)
assert len(INFERENCE_FEATURE_COLUMNS) == 64, (
    f"INFERENCE_FEATURE_COLUMNS has {len(INFERENCE_FEATURE_COLUMNS)} entries; "
    "expected 64.  Update TABULAR_FEATURES in src/models/variant_ensemble.py."
)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

@dataclass
class PipelineMetadata:
    """Immutable provenance stored alongside the model artifact."""
    created_at:    str       = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    val_auroc:     float     = 0.0
    n_train:       int       = 0
    n_features:    int       = 0
    feature_names: list[str] = field(default_factory=lambda: list(INFERENCE_FEATURE_COLUMNS))
    model_version: str       = "phase2"


# ---------------------------------------------------------------------------
# InferencePipeline
# ---------------------------------------------------------------------------

class InferencePipeline:
    """
    Wraps fitted tabular base models + stacking meta-learner for deployment.

    At inference time the pipeline:
      1. Optionally scores variants via GNNScorer → adds gnn_score column
      2. Calls engineer_features() to derive the 64 INFERENCE_FEATURE_COLUMNS
      3. Applies the StandardScaler (if present)
      4. Drives each base model with a numpy array → stacks predictions
      5. Feeds the stack to the meta-learner
      6. Returns a structured result dict with score, classification, and confidence

    Sequence-based models (CNN) are excluded at export time — they require
    FASTA context that is not available at API inference time.

    If no scaler is provided (scaler=None) step 3 is skipped — safe for
    tree-based-only ensembles where scaling has no effect.

    If no gnn_scorer is provided (gnn_scorer=None) gnn_score defaults to 0.5
    (ambiguous / not available) for all variants.
    """

    def __init__(
        self,
        trained_models: dict,
        meta_learner,
        scaler=None,
        metadata: Optional[PipelineMetadata] = None,
        gnn_scorer=None,
    ) -> None:
        self.trained_models = trained_models
        self.meta_learner   = meta_learner
        self.scaler         = scaler
        self.metadata       = metadata or PipelineMetadata()
        self.gnn_scorer     = gnn_scorer   # Optional GNNScorer instance

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_variant_ensemble(
        cls,
        ensemble,
        scaler=None,
        feature_names: Optional[list[str]] = None,
        val_auroc: float = 0.0,
        n_train: int = 0,
    ) -> "InferencePipeline":
        """
        Extract trained base models and meta-learner from a fitted VariantEnsemble.

        Sequence-based models ("cnn_1d") are excluded because they require a
        FASTA context window not available at API inference time.
        """
        trained_models = {
            name: model
            for name, model in ensemble.trained_models_.items()
            if name != "cnn_1d"
        }
        if not trained_models:
            raise ValueError(
                "No tabular models found in VariantEnsemble.trained_models_.  "
                "Ensure the ensemble was fitted before calling from_variant_ensemble()."
            )
        feature_names = feature_names or INFERENCE_FEATURE_COLUMNS
        metadata = PipelineMetadata(
            val_auroc     = val_auroc,
            n_train       = n_train,
            n_features    = len(feature_names),
            feature_names = feature_names,
        )
        return cls(
            trained_models = trained_models,
            meta_learner   = ensemble.meta_learner,
            scaler         = scaler,
            metadata       = metadata,
        )

    # ------------------------------------------------------------------
    # Public inference API
    # ------------------------------------------------------------------

    def predict_single(self, variant: dict[str, Any]) -> dict[str, Any]:
        """
        Predict pathogenicity for one variant.

        Parameters
        ----------
        variant : dict
            Raw variant fields.  At minimum supply chrom, pos, ref, alt.
            All other fields default to population-median values when absent
            (see engineer_features() in src/models/variant_ensemble.py).

        Returns
        -------
        dict with keys:
            pathogenicity_score  float in [0, 1]
            classification       5-tier ACMGish label
            confidence           "high" | "medium" | "low"
        """
        return self._predict_df(pd.DataFrame([variant]))[0]

    def predict_batch(self, variants: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Predict pathogenicity for a list of variant dicts."""
        if not variants:
            return []
        return self._predict_df(pd.DataFrame(variants))

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Return P(pathogenic) as a 1-D array of shape (n,).

        Parameters
        ----------
        df : pd.DataFrame
            Raw variant rows — same schema as predict_batch input dicts.
            If gnn_scorer is set, gnn_score is computed automatically.
            Otherwise gnn_score defaults to 0.5 (no GNN / ambiguous).
        """
        enriched = df.copy()

        # --- Optional GNN scoring (adds gnn_score column) ---
        if self.gnn_scorer is not None:
            try:
                gene_symbols = enriched.get(
                    "gene_symbol",
                    pd.Series([""] * len(enriched), index=enriched.index),
                ).fillna("")
                enriched["gnn_score"] = gene_symbols.map(
                    lambda g: self.gnn_scorer.score(g)
                )
            except Exception as exc:
                logger.warning(
                    "GNNScorer failed (%s) — defaulting gnn_score to 0.5.", exc
                )
                enriched["gnn_score"] = 0.5

        X = engineer_features(enriched)
        if self.scaler is not None:
            X = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index,
            )
        X_np = X[INFERENCE_FEATURE_COLUMNS].values
        base_preds = np.column_stack([
            model.predict_proba(X_np)[:, 1]
            for model in self.trained_models.values()
        ])
        return self.meta_learner.predict_proba(base_preds)[:, 1]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _predict_df(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        proba = self.predict_proba(df)
        return [_score_to_result(float(p)) for p in proba]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("InferencePipeline saved → %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "InferencePipeline":
        import joblib
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected InferencePipeline, got {type(obj)}")
        logger.info(
            "InferencePipeline loaded: val_auroc=%.4f  features=%d  created=%s",
            obj.metadata.val_auroc,
            obj.metadata.n_features,
            obj.metadata.created_at,
        )
        return obj


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _score_to_result(score: float) -> dict[str, Any]:
    """Convert a raw pathogenicity probability to a labelled result dict."""
    from src.api.schemas import score_to_classification
    classification, confidence = score_to_classification(score)
    return {
        "pathogenicity_score": round(score, 4),
        "classification":      classification,
        "confidence":          confidence,
    }
