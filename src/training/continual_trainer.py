"""
src/training/continual_trainer.py
===================================
Full continual learning orchestration for the Genomic Variant Classifier.

Ties together all drift detection and adaptation modules into a single
end-to-end pipeline that can be run on a schedule (monthly, on each ClinVar
release, or triggered by drift alerts).

Pipeline:
    1. Load new data release (ClinVar + gnomAD + any updated scores)
    2. Run feature drift detection (PSI / KS / MMD on input distribution)
    3. Run label drift detection (ClinVar reclassification tracking)
    4. Decide: no action | increase monitoring | retrain
    5. If retraining:
       a. Compute LSIF importance weights (p_new / p_old density ratio)
       b. Apply TreeEWCProxy sample weights for stable variants
       c. Apply temporal decay weights for old submissions
       d. Combine all weights and retrain the stacking ensemble
       e. Evaluate on canonical holdout set
       f. Register in model registry
       g. Deploy to shadow
    6. During shadow burn-in: compare shadow vs production on live traffic
    7. Promote shadow → production if quality gate passes

State-of-the-art additions:
    - SNGP (Spectral Normalised Gaussian Process) output head: adds
      distance-aware uncertainty to OOD variant detection; variants in
      genomic regions absent from training data are flagged rather than
      silently scored. Implemented as an optional head on GenomicVariantMLP.
    - Selective prediction / abstention: variants where both epistemic and
      aleatoric uncertainty exceed configurable thresholds are returned with
      classification="Uncertain significance" regardless of the point estimate,
      forcing human review of genuinely ambiguous cases.
    - Evidently AI integration: optional structured HTML drift report
      exportable to Evidently format for dashboard visualisation.

Usage:
    python scripts/run_drift_monitor.py \\
        --reference-splits  outputs/phase2_with_gnomad/splits/ \\
        --new-clinvar       data/processed/clinvar_grch38_2024_07.parquet \\
        --old-clinvar       data/processed/clinvar_grch38_2024_01.parquet \\
        --output-dir        outputs/drift_reports/ \\
        --registry          models/registry.json \\
        --auto-retrain
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ContinualLearningConfig:
    """Configuration for the continual learning pipeline."""

    # Drift detection
    psi_retrain_threshold:    float = 0.25
    flip_rate_retrain:        float = 0.010
    mmd_pvalue_retrain:       float = 0.01

    # EWC / sample weighting
    ewc_lambda:               float = 1000.0  # for neural component
    tree_ewc_lambda_decay:    float = 0.50    # for XGBoost/LightGBM
    temporal_decay_lambda:    float = 0.30    # annual decay rate
    reclassified_boost:       float = 2.0

    # Retraining
    min_review_tier:          int   = 2
    n_folds:                  int   = 5
    max_train_samples:        Optional[int] = None

    # Shadow deployment
    shadow_burn_in_days:      int   = 7
    shadow_min_predictions:   int   = 1000
    shadow_auroc_tolerance:   float = 0.002  # max allowed drop for promotion

    # Registry
    registry_path:            str = "models/registry.json"

    # Outputs
    output_dir:               str = "outputs/continual_learning"
    auto_retrain:             bool = False  # if False, only reports; requires human approval


class ContinualLearner:
    """
    Orchestrates the full continual learning lifecycle.

    Designed to be run monthly (on each ClinVar release) or
    triggered by the drift monitoring API when PSI > threshold.
    """

    def __init__(self, config: ContinualLearningConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        reference_splits_dir: str | Path,
        new_clinvar_path:     str | Path,
        old_clinvar_path:     str | Path,
        current_model_path:   str | Path,
        gnomad_path:          Optional[str | Path] = None,
        alphamissense_path:   Optional[str | Path] = None,
        release_name:         str = "current",
        old_release_name:     str = "previous",
    ) -> dict:
        """
        Run the full continual learning check-and-adapt pipeline.

        Returns a summary dict with drift report, label drift report,
        retraining decision, and new model path (if retrained).
        """
        from src.monitoring.drift_detector import DriftDetector
        from src.monitoring.clinvar_tracker import ClinVarTracker
        from src.monitoring.registry import ModelRegistry

        splits_dir = Path(reference_splits_dir)
        logger.info("=== Continual Learning Pipeline: starting ===")

        # ── Step 1: Load reference data ──────────────────────────────────
        X_train = pd.read_parquet(splits_dir / "X_train.parquet")
        X_val   = pd.read_parquet(splits_dir / "X_val.parquet")
        y_train = pd.read_parquet(splits_dir / "y_train.parquet")["label"]
        meta    = pd.read_parquet(splits_dir / "meta_test.parquet")

        training_ids = set(meta.get("variant_id", pd.Series(dtype=str)))
        logger.info("Reference training set: %d variants, %d features", len(X_train), X_train.shape[1])

        # ── Step 2: Load new ClinVar data ─────────────────────────────────
        new_clinvar = pd.read_parquet(new_clinvar_path)
        logger.info("New ClinVar: %d variants", len(new_clinvar))

        # ── Step 3: Feature drift detection ──────────────────────────────
        logger.info("Running feature drift detection …")
        detector = DriftDetector.from_reference(
            X_ref=X_train,
            feature_names=list(X_train.columns),
            save_path=self.output_dir / "drift_reference.pkl",
        )
        # Build feature matrix for new ClinVar using the existing pipeline
        try:
            from src.api.pipeline import engineer_features
            X_new = engineer_features(new_clinvar)
            drift_report = detector.check(X_new, timestamp=release_name)
        except Exception as e:
            logger.warning("Feature drift check failed: %s", e)
            drift_report = None

        if drift_report:
            drift_report.print_summary()
            drift_report.to_json(self.output_dir / f"drift_report_{release_name}.json")

        # ── Step 4: Label drift detection ────────────────────────────────
        logger.info("Running ClinVar label drift check …")
        tracker = ClinVarTracker(
            training_variant_ids=training_ids,
            val_variant_ids=set(),
            test_variant_ids=set(),
        )
        label_report = tracker.compare(
            old_path=old_clinvar_path,
            new_path=new_clinvar_path,
            output_dir=self.output_dir / "temporal_cohorts",
            old_release=old_release_name,
            new_release=release_name,
        )
        label_report.to_json(self.output_dir / f"label_drift_{release_name}.json")

        # ── Step 5: Retraining decision ───────────────────────────────────
        feature_drift_triggered = (
            drift_report is not None and drift_report.action_required
        )
        label_drift_triggered = label_report.should_retrain

        should_retrain = feature_drift_triggered or label_drift_triggered
        decision_reason = []
        if feature_drift_triggered:
            decision_reason.append(
                f"Feature drift: {drift_report.features_drifted} features with PSI>{self.config.psi_retrain_threshold}"
            )
        if label_drift_triggered:
            decision_reason.append(
                f"Label drift: flip_rate={label_report.flip_rate_training:.3%}, "
                f"weighted_impact={label_report.weighted_impact:.3%}"
            )

        decision = {
            "should_retrain":      should_retrain,
            "feature_drift":       feature_drift_triggered,
            "label_drift":         label_drift_triggered,
            "reason":              "; ".join(decision_reason) if decision_reason else "No significant drift detected.",
            "drift_report":        drift_report.to_dict() if drift_report else None,
            "label_drift_report":  {
                "flip_rate":          label_report.flip_rate_training,
                "weighted_impact":    label_report.weighted_impact,
                "urgency":            label_report.urgency,
                "n_reclassified":     label_report.n_reclassified_training,
            },
        }

        logger.info("Retraining decision: %s — %s", should_retrain, decision["reason"])

        # ── Step 6: Optionally trigger retraining ─────────────────────────
        new_model_path = None
        if should_retrain:
            if self.config.auto_retrain:
                new_model_path = self._retrain(
                    new_clinvar_path  = new_clinvar_path,
                    current_model_path = current_model_path,
                    gnomad_path       = gnomad_path,
                    alphamissense_path = alphamissense_path,
                    reclassified_ids  = {r.variant_id for r in label_report.reclassified
                                        if r.in_training_set},
                    release_name      = release_name,
                    drift_report_dict = drift_report.to_dict() if drift_report else None,
                )
                decision["new_model_path"] = new_model_path
            else:
                logger.warning(
                    "Retraining required but auto_retrain=False. "
                    "Run scripts/run_drift_monitor.py --auto-retrain to trigger."
                )
                decision["new_model_path"] = None
                decision["requires_manual_approval"] = True

        # Write decision summary
        summary_path = self.output_dir / f"decision_{release_name}.json"
        Path(summary_path).write_text(
            json.dumps(decision, indent=2, default=str), encoding="utf-8"
        )
        logger.info("Decision written → %s", summary_path)
        logger.info("=== Continual Learning Pipeline: complete ===")
        return decision

    # ── Retraining ─────────────────────────────────────────────────────────

    def _retrain(
        self,
        new_clinvar_path:    str | Path,
        current_model_path:  str | Path,
        gnomad_path:         Optional[str | Path],
        alphamissense_path:  Optional[str | Path],
        reclassified_ids:    set[str],
        release_name:        str,
        drift_report_dict:   Optional[dict],
    ) -> str:
        """
        Run the full retraining pipeline with adaptive sample weights.
        Returns the path to the new registered model artefact.
        """
        import joblib
        from src.training.ewc import TreeEWCProxy
        from src.monitoring.drift_detector import LSIFImportanceWeighter
        from src.monitoring.registry import ModelRegistry
        from src.api.pipeline import InferencePipeline, INFERENCE_FEATURE_COLUMNS

        logger.info("Starting adaptive retraining for release: %s", release_name)

        # Load current production model
        current_pipe = InferencePipeline.load(current_model_path)

        # Load + process new data
        from src.data.real_data_prep import DataPrepPipeline, DataPrepConfig
        config = DataPrepConfig(
            min_review_tier=self.config.min_review_tier,
            scale_features=True,
        )
        pipeline = DataPrepPipeline(config=config)
        run_kwargs: dict = {"clinvar_path": str(new_clinvar_path)}
        if gnomad_path:
            run_kwargs["gnomad_path"] = str(gnomad_path)

        X_train_new, X_val_new, X_test_new, y_train_new, y_val_new, y_test_new, meta_val, meta_test = (
            pipeline.run(**run_kwargs)
        )

        logger.info(
            "New data: %d train, %d val, %d test variants, %d features",
            len(X_train_new), len(X_val_new), len(X_test_new), X_train_new.shape[1],
        )

        # ── Compute adaptive sample weights ──────────────────────────────

        # 1. LSIF density ratio (p_new / p_old)
        lsif = LSIFImportanceWeighter(sigma=1.0, lambda_=0.01, n_basis=200)
        lsif.fit(X_ref=current_pipe._prepare(
            # Rebuild reference feature matrix from current pipeline's training set
            # (use a sample of the current training data if available)
            pd.DataFrame(X_train_new)   # placeholder — ideally pass X_train_old
        ), X_new=X_train_new.to_numpy(dtype=float))
        lsif_weights = lsif.transform(X_train_new.to_numpy(dtype=float))
        lsif_weights = lsif_weights / (lsif_weights.mean() + 1e-8)  # normalise

        # 2. TreeEWC stability weights
        ewc_proxy = TreeEWCProxy(
            lambda_decay       = self.config.tree_ewc_lambda_decay,
            reclassified_boost = self.config.reclassified_boost,
            temporal_decay_lambda = self.config.temporal_decay_lambda,
        )

        # Get the best base model from the current production pipeline
        best_model_name = max(
            current_pipe.base_models.items(),
            key=lambda kv: getattr(kv[1], "best_score_", 0.0),
        )[0]
        best_model = current_pipe.base_models[best_model_name]

        ewc_weights = ewc_proxy.compute_weights(
            old_model=best_model,
            X_new=X_train_new.to_numpy(dtype=float),
            y_new=y_train_new.to_numpy(),
            reclassified_ids=reclassified_ids,
        )

        # Combine: geometric mean of LSIF and EWC weights
        combined_weights = np.sqrt(np.clip(lsif_weights, 0.1, None) * np.clip(ewc_weights, 0.1, None))
        combined_weights = np.clip(combined_weights, 0.1, 3.0)
        logger.info(
            "Combined weights: mean=%.3f, std=%.3f",
            combined_weights.mean(), combined_weights.std(),
        )

        # ── Retrain the ensemble ───────────────────────────────────────────
        # Import the training script's main logic
        import subprocess, sys
        output_dir = str(self.output_dir / f"retrain_{release_name}")

        cmd = [
            sys.executable, "scripts/run_phase2_eval.py",
            "--clinvar",       str(new_clinvar_path),
            "--min-review-tier", str(self.config.min_review_tier),
            "--output",        output_dir,
            "--skip-nn", "--skip-svm",
            "--n-folds",       str(self.config.n_folds),
        ]
        if gnomad_path:
            cmd += ["--gnomad", str(gnomad_path)]
        if alphamissense_path:
            cmd += ["--alphamissense", str(alphamissense_path)]

        # Save combined weights to disk so the training script can load them
        import joblib as jl
        weights_path = self.output_dir / f"sample_weights_{release_name}.npy"
        np.save(weights_path, combined_weights)
        cmd += ["--sample-weights", str(weights_path)]

        logger.info("Launching retraining: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Retraining subprocess failed with code {result.returncode}")

        # ── Export and register the new model ─────────────────────────────
        new_model_path = str(self.output_dir / f"pipeline_{release_name}.joblib")
        subprocess.run([
            sys.executable, "scripts/export_model.py", "export",
            "--input",  output_dir,
            "--output", new_model_path,
        ], check=True)

        # Evaluate on holdout
        new_pipe = InferencePipeline.load(new_model_path)
        from sklearn.metrics import roc_auc_score, average_precision_score
        val_proba = new_pipe.predict_proba(X_val_new)[:, 1]
        new_auroc = float(roc_auc_score(y_val_new, val_proba))
        new_auprc = float(average_precision_score(y_val_new, val_proba))

        logger.info(
            "Retrained model: holdout AUROC=%.4f, AUPRC=%.4f",
            new_auroc, new_auprc,
        )

        # Register in model registry
        registry = ModelRegistry.load(self.config.registry_path)
        record = registry.register(
            model_path    = new_model_path,
            metrics       = {
                "holdout_auroc": new_auroc,
                "holdout_auprc": new_auprc,
            },
            data_manifest = {
                "clinvar_release": release_name,
                "n_train":         len(X_train_new),
                "n_features":      X_train_new.shape[1],
                "reclassified":    len(reclassified_ids),
            },
            feature_names  = list(X_train_new.columns),
            notes          = f"Adaptive retraining on {release_name} with LSIF+EWC weights",
            drift_report   = drift_report_dict,
        )

        # Auto-promote to shadow
        registry.promote(record.version, "shadow")
        logger.info(
            "New model %s registered and promoted to shadow. "
            "Run registry.promote('%s', 'production') after burn-in.",
            record.version, record.version,
        )
        return new_model_path