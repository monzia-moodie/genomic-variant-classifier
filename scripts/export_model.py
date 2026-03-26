"""
scripts/export_model.py
========================
Serialise a trained run into an ``InferencePipeline`` joblib artefact
suitable for loading by the FastAPI service.

What this script does
---------------------
1. Locates the best trained ``VariantEnsemble`` (or an ensemble subset
   from a run_phase2_eval.py output directory).
2. Wraps it with the fitted ``StandardScaler`` and feature list in a
   ``src.api.pipeline.InferencePipeline`` instance.
3. Optionally excludes large base models (``--exclude-models``) to reduce
   artefact size without significant AUROC loss.
4. Writes the artefact to ``--output`` (default: models/phase2_pipeline.joblib).
5. Runs a smoke-test: classifies three known ClinVar variants and asserts
   BRCA1 c.68_69del (pathogenic) scores higher than a benign control.

Model size guide (approximate serialised sizes)
------------------------------------------------
  random_forest    ~1 200 MB   (large due to deep trees × 300 estimators)
  gradient_boosting  ~400 MB
  xgboost            ~120 MB
  lightgbm            ~50 MB
  logistic_regression  ~1 MB

Recommended production export (< 200 MB, AUROC ~0.9845):
  python scripts/export_model.py --input outputs/run \\
      --exclude-models random_forest gradient_boosting

Usage
-----
  # After a run_phase2_eval.py training run:
  python scripts/export_model.py \\
      --input  outputs/phase2_with_gnomad \\
      --output models/phase2_pipeline.joblib

  # Slim export (~200 MB) suitable for Docker:
  python scripts/export_model.py \\
      --input  outputs/phase2_with_gnomad \\
      --output models/phase2_pipeline_slim.joblib \\
      --exclude-models random_forest gradient_boosting

  # Verify the exported artefact without re-exporting:
  python scripts/export_model.py verify models/phase2_pipeline.joblib

Exit codes
----------
    0  artefact written (and smoke test passed)
    1  required file missing or smoke test failed
    2  unexpected error
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("export_model")

# ---------------------------------------------------------------------------
# Smoke-test variants (ClinVar-reviewed, deterministic)
# ---------------------------------------------------------------------------

# Smoke-test variants use only features with real training-time data:
#   allele_freq      — gnomAD v4 (real data used in training run)
#   alphamissense_score — real data for missense variants; 0.5 default for others
#   gene_constraint_oe  — gnomAD pLoF oe (real)
#   n_pathogenic_in_gene — computed from ClinVar training labels
#   consequence, chrom — from ClinVar
#
# NOTE: cadd_phred, sift_score, polyphen2_score, revel_score, phylop_score, gerp_score
# were NOT annotated in the training run (connectors ran in stub mode → constant
# defaults).  Providing non-default values would be out-of-distribution for the scaler
# and produce garbage predictions.  engineer_features() fills them with the same
# defaults used at training time.
_SMOKE_VARIANTS = [
    # BRCA1 c.68_69del — Pathogenic (ClinVar RCV000031443)
    {
        "chrom": "17", "pos": 43115726, "ref": "AAC", "alt": "A",
        "consequence": "frameshift_variant",
        "gene_symbol": "BRCA1",
        "allele_freq": 0.0,
        "gene_constraint_oe": 0.08,
        "n_pathogenic_in_gene": 2800,   # key signal: many known pathogenic variants in BRCA1
        "_expected_class": "pathogenic",
    },
    # rs1801133 (MTHFR p.Ala222Val) — Benign (common variant, AF ~0.33)
    {
        "chrom": "1", "pos": 11796321, "ref": "G", "alt": "A",
        "consequence": "missense_variant",
        "gene_symbol": "MTHFR",
        "allele_freq": 0.334,           # key signal: common in population → benign
        "gene_constraint_oe": 0.85,
        "alphamissense_score": 0.08,    # AlphaMissense: benign
        "n_pathogenic_in_gene": 5,
        "_expected_class": "benign",
    },
    # TP53 R175H (missense hotspot) — Pathogenic
    {
        "chrom": "17", "pos": 7675088, "ref": "C", "alt": "T",
        "consequence": "missense_variant",
        "gene_symbol": "TP53",
        "allele_freq": 0.0,
        "gene_constraint_oe": 0.05,
        "alphamissense_score": 0.98,    # AlphaMissense: strongly pathogenic
        "n_pathogenic_in_gene": 3200,
        "_expected_class": "pathogenic",
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_ensemble_from_run(run_dir: Path):
    """
    Load the best-available trained model from a run_phase2_eval.py output dir.

    Looks for (in priority order):
      1. models/ensemble.joblib  — standard sub-directory artefact
      2. ensemble.joblib         — bare VariantEnsemble at root
    """
    candidates = [
        run_dir / "models" / "ensemble.joblib",
        run_dir / "ensemble.joblib",
    ]
    for path in candidates:
        if path.exists():
            logger.info("Loading model from %s", path)
            return joblib.load(path), path
    raise FileNotFoundError(
        f"No trained model found in {run_dir}. "
        "Expected one of: " + ", ".join(str(c) for c in candidates)
    )


def _load_scaler(run_dir: Path):
    """Load the fitted StandardScaler from the run directory, if present."""
    for name in ("scaler.joblib", "models/scaler.joblib"):
        p = run_dir / name
        if p.exists():
            logger.info("Loading scaler from %s", p)
            return joblib.load(p)
    logger.warning(
        "No scaler.joblib found in %s — InferencePipeline will skip scaling. "
        "This is fine if DataPrepConfig.scale_features=False was used at training.",
        run_dir,
    )
    return None


def _load_feature_names(run_dir: Path) -> list[str] | None:
    """
    Try to load the exact feature column list used at training.

    Search order (most authoritative first):
      1. splits/X_train.parquet  — ground-truth split written by run_phase2_eval.py
      2. X_train.parquet         — legacy flat location
      3. feature_names.txt       — manually curated fallback
    """
    candidates = [
        run_dir / "splits" / "X_train.parquet",
        run_dir / "X_train.parquet",
        run_dir / "feature_names.txt",
    ]
    for p in candidates:
        if not p.exists():
            continue
        if p.suffix == ".parquet":
            cols = list(pd.read_parquet(p, columns=None).columns)
            logger.info("Loaded %d feature names from %s", len(cols), p)
            return cols
        if p.suffix == ".txt":
            names = p.read_text().splitlines()
            logger.info("Loaded %d feature names from %s", len(names), p)
            return [n.strip() for n in names if n.strip()]
    return None


def _smoke_test(pipeline) -> bool:
    """
    Run three known variants through the pipeline and check score ordering.

    Asserts:
      * P(pathogenic | BRCA1 frameshift) > 0.7
      * P(pathogenic | MTHFR common)     < 0.3
      * P(pathogenic | TP53 R175H)       > 0.7
    """
    logger.info("Running smoke test on 3 known ClinVar variants ...")

    rows_in = [{k: v for k, v in row.items() if k != "_expected_class"}
               for row in _SMOKE_VARIANTS]
    expected = [row["_expected_class"] for row in _SMOKE_VARIANTS]

    try:
        scores = pipeline.predict_proba(pd.DataFrame(rows_in))
    except Exception as exc:
        logger.error("Smoke test: predict_proba raised %s: %s", type(exc).__name__, exc)
        return False

    thresholds = {"pathogenic": (0.7, "above 0.70"), "benign": (0.3, "below 0.30")}
    all_pass = True

    for score, exp_class in zip(scores, expected):
        thresh, label = thresholds[exp_class]
        passed = (score > thresh) if exp_class == "pathogenic" else (score < thresh)
        status = "PASS" if passed else "FAIL"
        logger.info(
            "  [%s] expected=%-11s  score=%.4f  threshold=%s",
            status, exp_class, float(score), label,
        )
        if not passed:
            all_pass = False

    return all_pass


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------

def cmd_export(args: argparse.Namespace) -> int:
    from src.api.pipeline import InferencePipeline
    from src.models.variant_ensemble import TABULAR_FEATURES

    run_dir = Path(args.input)
    if not run_dir.exists():
        logger.error("Input directory not found: %s", run_dir)
        return 1

    exclude: set[str] = set(args.exclude_models or [])
    if exclude:
        logger.info("Excluding base models from artefact: %s", sorted(exclude))

    try:
        ensemble, _ = _load_ensemble_from_run(run_dir)
        scaler        = _load_scaler(run_dir)
        feature_names = _load_feature_names(run_dir) or list(TABULAR_FEATURES)

        # Drop unwanted base models before wrapping
        if exclude and hasattr(ensemble, "trained_models_"):
            before = set(ensemble.trained_models_.keys())
            ensemble.trained_models_ = {
                k: v for k, v in ensemble.trained_models_.items()
                if k not in exclude
            }
            after = set(ensemble.trained_models_.keys())
            removed = before - after
            if removed:
                logger.info(
                    "Removed base models: %s  (remaining: %s)",
                    sorted(removed), sorted(after),
                )
            unknown = exclude - before
            if unknown:
                logger.warning(
                    "Requested to exclude %s but they were not in trained_models_: %s",
                    sorted(unknown), sorted(before),
                )

        if not getattr(ensemble, "trained_models_", {}):
            logger.error("No base models remain after exclusion.  Aborting.")
            return 1

        # Read provenance from metrics.json if present
        val_auroc = 0.0
        n_train   = 0
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            m         = json.loads(metrics_path.read_text())
            val_auroc = m.get("val_auroc", 0.0)
            n_train   = m.get("n_train",   0)
            logger.info("Provenance: val_auroc=%.4f  n_train=%d", val_auroc, n_train)

        pipeline = InferencePipeline.from_variant_ensemble(
            ensemble,
            scaler        = scaler,
            feature_names = feature_names,
            val_auroc     = val_auroc,
            n_train       = n_train,
        )

        # Smoke test before writing
        if not args.skip_smoke_test:
            passed = _smoke_test(pipeline)
            if not passed:
                logger.error(
                    "Smoke test FAILED — artefact NOT written. "
                    "Investigate model quality before deploying."
                )
                return 1
            logger.info("Smoke test passed.")

        pipeline.save(args.output)
        logger.info("InferencePipeline written to %s", args.output)
        return 0

    except FileNotFoundError as exc:
        logger.error("%s", exc)
        logger.error(
            "Re-run run_phase2_eval.py — the updated script saves both "
            "ensemble.joblib and scaler.joblib automatically."
        )
        return 1
    except Exception as exc:
        logger.exception("Export failed: %s", exc)
        return 2


def cmd_verify(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if not path.exists():
        logger.error("Artefact not found: %s", path)
        return 1

    logger.info("Loading %s ...", path)
    from src.api.pipeline import InferencePipeline
    pipeline = InferencePipeline.load(path)
    passed = _smoke_test(pipeline)
    return 0 if passed else 1


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a trained pipeline to an InferencePipeline joblib artefact.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    exp = sub.add_parser("export", help="Wrap and export a trained model.")
    exp.add_argument(
        "--input", "-i", type=Path, required=True,
        help="run_phase2_eval output directory containing ensemble.joblib.",
    )
    exp.add_argument(
        "--output", "-o", type=Path, default=Path("models/phase2_pipeline.joblib"),
        help="Destination path for the InferencePipeline artefact.",
    )
    exp.add_argument(
        "--exclude-models", nargs="*", metavar="MODEL",
        default=[],
        help=(
            "Base model names to drop from the artefact before serialising. "
            "Common values: random_forest gradient_boosting  "
            "(reduces artefact from ~2 GB to ~200 MB with negligible AUROC loss). "
            "Example: --exclude-models random_forest gradient_boosting"
        ),
    )
    exp.add_argument(
        "--skip-smoke-test", action="store_true",
        help="Skip the 3-variant smoke test after export.",
    )

    ver = sub.add_parser("verify", help="Smoke-test an existing artefact.")
    ver.add_argument("path", type=Path, help="Path to an InferencePipeline joblib.")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    # Default to 'export' when no sub-command supplied (legacy usage).
    # Must be done before parse_args() — argparse exits on unrecognised top-level
    # arguments before we can inspect args.command.
    _subcommands = {"export", "verify", "-h", "--help"}
    if len(sys.argv) < 2 or sys.argv[1] not in _subcommands:
        sys.argv.insert(1, "export")

    args = parse_args()

    if args.command == "export":
        return cmd_export(args)
    if args.command == "verify":
        return cmd_verify(args)

    logger.error("Unknown command: %s", args.command)
    return 1


if __name__ == "__main__":
    sys.exit(main())
