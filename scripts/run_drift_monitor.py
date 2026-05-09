"""
scripts/run_drift_monitor.py
==============================
Scheduled drift monitoring CLI for the Genomic Variant Classifier.

Run this script:
  - Monthly (on each ClinVar release)
  - On-demand after any upstream data source update
  - Via cron / GitHub Actions scheduled workflow

What it does:
  1. Checks feature drift (PSI / KS / MMD) against the training reference
  2. Checks label drift (ClinVar reclassifications) against old release
  3. Writes a structured JSON report
  4. Optionally triggers retraining if drift exceeds thresholds
  5. Optionally exports an Evidently AI HTML dashboard
  6. Exits with code 0 (no action) or 2 (retraining recommended) so
     CI/CD can gate on the exit code

Usage:
    python scripts/run_drift_monitor.py \\
        --reference-splits  outputs/phase2_with_gnomad/splits/ \\
        --new-clinvar       data/processed/clinvar_grch38_2024_07.parquet \\
        --old-clinvar       data/processed/clinvar_grch38_2024_01.parquet \\
        --current-model     models/phase2_pipeline.joblib \\
        --output-dir        outputs/drift_reports/2024_07/ \\
        --registry          models/registry.json \\
        --release-name      2024_07 \\
        --auto-retrain

    # Check features only (no label drift, no retraining):
    python scripts/run_drift_monitor.py \\
        --reference-splits outputs/phase2_with_gnomad/splits/ \\
        --new-data         data/processed/gnomad_v5_exomes.parquet \\
        --features-only

Exit codes:
    0 = no drift detected, no action required
    1 = monitoring recommended (PSI in yellow zone)
    2 = retraining recommended (PSI in red zone or label drift)
    3 = urgent retraining (severe drift, high weighted flip rate)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("run_drift_monitor")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Drift monitor for the Genomic Variant Classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--reference-splits", type=Path, required=True,
                   help="Directory containing X_train.parquet (reference distribution).")
    p.add_argument("--new-clinvar",  type=Path, default=None,
                   help="New ClinVar parquet to check for label drift.")
    p.add_argument("--old-clinvar",  type=Path, default=None,
                   help="Previous ClinVar parquet (for reclassification comparison).")
    p.add_argument("--new-data",     type=Path, default=None,
                   help="Any new feature parquet for covariate drift check.")
    p.add_argument("--current-model", type=Path, default=Path("models/phase2_pipeline.joblib"),
                   help="Path to the current production InferencePipeline.")
    p.add_argument("--gnomad",        type=Path, default=None)
    p.add_argument("--alphamissense", type=Path, default=None)
    p.add_argument("--output-dir",    type=Path, default=Path("outputs/drift_reports"),
                   help="Directory for reports and artefacts.")
    p.add_argument("--registry",      type=Path, default=Path("models/registry.json"),
                   help="Path to the model registry JSON.")
    p.add_argument("--release-name",  type=str,  default="latest",
                   help="Label for this release (e.g. '2024_07').")
    p.add_argument("--old-release-name", type=str, default="previous")
    p.add_argument("--auto-retrain",  action="store_true",
                   help="Automatically trigger retraining if drift is detected.")
    p.add_argument("--features-only", action="store_true",
                   help="Run feature drift check only (skip label drift).")
    p.add_argument("--evidently",     action="store_true",
                   help="Generate an Evidently AI HTML drift dashboard.")
    p.add_argument("--psi-threshold", type=float, default=0.25,
                   help="PSI threshold for retraining trigger.")
    p.add_argument("--flip-rate-threshold", type=float, default=0.010,
                   help="ClinVar flip rate threshold for retraining trigger.")
    return p


def run_feature_drift(args: argparse.Namespace) -> int:
    """Returns exit code fragment from feature drift check."""
    import pandas as pd
    from genomic_variant_classifier.monitoring.drift_detector import DriftDetector

    splits_dir = args.reference_splits
    if not (splits_dir / "X_train.parquet").exists():
        logger.error("X_train.parquet not found in %s", splits_dir)
        return 3

    X_ref = pd.read_parquet(splits_dir / "X_train.parquet")
    logger.info("Reference: %d samples × %d features", *X_ref.shape)

    detector = DriftDetector.from_reference(
        X_ref=X_ref,
        save_path=args.output_dir / "drift_reference.pkl",
    )

    # New data: prefer --new-data; fallback to building from --new-clinvar
    if args.new_data and args.new_data.exists():
        X_new = pd.read_parquet(args.new_data)
        if X_new.shape[1] != X_ref.shape[1]:
            logger.warning(
                "New data has %d features, reference has %d — aligning to reference columns.",
                X_new.shape[1], X_ref.shape[1],
            )
            X_new = X_new.reindex(columns=X_ref.columns, fill_value=0.0)
    elif args.new_clinvar and args.new_clinvar.exists():
        logger.info("Building feature matrix from new ClinVar …")
        clinvar = pd.read_parquet(args.new_clinvar)
        try:
            from genomic_variant_classifier.api.pipeline import engineer_features
            X_new = engineer_features(clinvar)
            X_new = X_new.reindex(columns=X_ref.columns, fill_value=0.0)
        except Exception as e:
            logger.error("Feature engineering failed: %s", e)
            return 3
    else:
        logger.warning("No new data provided for feature drift check — skipping.")
        return 0

    logger.info("New data: %d samples", len(X_new))
    report = detector.check(X_new, timestamp=args.release_name)
    report.to_json(args.output_dir / f"feature_drift_{args.release_name}.json")

    # Optional Evidently AI export
    if args.evidently:
        _export_evidently(X_ref, X_new, report, args.output_dir, args.release_name)

    # Map to exit code
    if report.recommended_action == "urgent_retrain":
        return 3
    if report.recommended_action == "retrain":
        return 2
    if report.recommended_action == "monitor":
        return 1
    return 0


def run_label_drift(args: argparse.Namespace) -> tuple[int, object]:
    """Returns (exit_code, label_report)."""
    import pandas as pd
    from genomic_variant_classifier.monitoring.clinvar_tracker import ClinVarTracker

    splits_dir = args.reference_splits
    meta_path = splits_dir / "meta_test.parquet"
    training_ids: set[str] = set()
    if meta_path.exists():
        meta = pd.read_parquet(meta_path)
        if "variant_id" in meta.columns:
            training_ids = set(meta["variant_id"].astype(str))
    logger.info("Label drift check: tracking %d training variant IDs.", len(training_ids))

    tracker = ClinVarTracker(training_variant_ids=training_ids)
    report  = tracker.compare(
        old_path     = args.old_clinvar,
        new_path     = args.new_clinvar,
        output_dir   = args.output_dir / "temporal_cohorts",
        old_release  = args.old_release_name,
        new_release  = args.release_name,
    )
    report.to_json(args.output_dir / f"label_drift_{args.release_name}.json")

    if report.urgency == "urgent":
        return 3, report
    if report.urgency == "retrain":
        return 2, report
    if report.urgency == "monitor":
        return 1, report
    return 0, report


def _export_evidently(
    X_ref, X_new, drift_report, output_dir: Path, release_name: str
) -> None:
    """Export Evidently AI HTML report (requires 'evidently' package)."""
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset, DataQualityPreset
        from evidently.pipeline.column_mapping import ColumnMapping

        report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
        report.run(reference_data=X_ref, current_data=X_new, column_mapping=ColumnMapping())
        html_path = output_dir / f"evidently_drift_{release_name}.html"
        report.save_html(str(html_path))
        logger.info("Evidently AI report → %s", html_path)
    except ImportError:
        logger.warning(
            "Evidently AI not installed. Run: pip install evidently"
        )
    except Exception as e:
        logger.warning("Evidently report generation failed: %s", e)


def main() -> int:
    parser = build_parser()
    args   = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    exit_codes = [0]

    # ── Feature drift ─────────────────────────────────────────────────────
    feature_code = run_feature_drift(args)
    exit_codes.append(feature_code)
    logger.info("Feature drift exit code: %d", feature_code)

    # ── Label drift ───────────────────────────────────────────────────────
    label_report = None
    if (
        not args.features_only
        and args.new_clinvar and args.new_clinvar.exists()
        and args.old_clinvar and args.old_clinvar.exists()
    ):
        label_code, label_report = run_label_drift(args)
        exit_codes.append(label_code)
        logger.info("Label drift exit code: %d", label_code)
    else:
        logger.info("Label drift check skipped (--features-only or missing paths).")

    # ── Overall decision ──────────────────────────────────────────────────
    final_code = max(exit_codes)

    if final_code >= 2 and args.auto_retrain:
        logger.info("Drift detected. Triggering continual learning pipeline …")
        if not args.current_model.exists():
            logger.error("Current model not found: %s", args.current_model)
            return 3

        from genomic_variant_classifier.training.continual_trainer import ContinualLearner, ContinualLearningConfig
        cl_config = ContinualLearningConfig(
            psi_retrain_threshold = args.psi_threshold,
            flip_rate_retrain     = args.flip_rate_threshold,
            auto_retrain          = True,
            output_dir            = str(args.output_dir / "retrain"),
            registry_path         = str(args.registry),
        )
        learner = ContinualLearner(cl_config)
        decision = learner.run(
            reference_splits_dir  = args.reference_splits,
            new_clinvar_path      = args.new_clinvar,
            old_clinvar_path      = args.old_clinvar,
            current_model_path    = args.current_model,
            gnomad_path           = args.gnomad,
            alphamissense_path    = args.alphamissense,
            release_name          = args.release_name,
            old_release_name      = args.old_release_name,
        )
        if decision.get("new_model_path"):
            logger.info("New model artefact: %s", decision["new_model_path"])
            logger.info(
                "Shadow deployment initiated. Promote to production after burn-in with:\n"
                "  python -c \"\n"
                "  from genomic_variant_classifier.monitoring.registry import ModelRegistry\n"
                "  r = ModelRegistry.load('%s')\n"
                "  r.print_summary()\n"
                "  # r.promote('v?.0.0', 'production')\n"
                "  \"", args.registry,
            )
    elif final_code >= 2:
        logger.warning(
            "Drift detected but auto_retrain=False. "
            "Re-run with --auto-retrain or review the reports in %s",
            args.output_dir,
        )

    exit_messages = {
        0: "No significant drift. No action required.",
        1: "Minor drift detected. Increase monitoring frequency.",
        2: "Significant drift detected. Retraining recommended.",
        3: "Severe drift detected. Urgent retraining required.",
    }
    logger.info("EXIT %d: %s", final_code, exit_messages.get(final_code, ""))
    return final_code


if __name__ == "__main__":
    sys.exit(main())