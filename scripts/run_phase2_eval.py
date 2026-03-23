"""
scripts/run_phase2_eval.py
===========================
End-to-end Phase 2 evaluation on real ClinVar data.

Exit codes:
  0  AUROC >= target (default 0.90)
  1  AUROC < target
  2  Pipeline error

Minimal usage (~15 min):
  python scripts/run_phase2_eval.py
      --clinvar data/processed/clinvar_grch38.parquet
      --skip-nn --skip-svm --n-folds 3 --output outputs/phase2_fast

Full Phase 2 usage (~45-90 min):
  python scripts/run_phase2_eval.py
      --clinvar       data/processed/clinvar_grch38.parquet
      --gnomad        data/processed/gnomad_v4_exomes.parquet
      --spliceai      data/external/spliceai/spliceai_scores.masked.snv.hg38.vcf.gz
      --alphamissense data/external/alphamissense/AlphaMissense_hg38.tsv.gz
      --gtex-genes    BRCA1 BRCA2 TP53 PTEN ATM
      --output        outputs/phase2_full
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_phase2_eval")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2 evaluation on real ClinVar data")
    p.add_argument("--clinvar",         required=True)
    p.add_argument("--gnomad",          default=None)
    p.add_argument("--spliceai",        default=None)
    p.add_argument("--alphamissense",   default=None)
    p.add_argument("--gtex-genes",      nargs="*", default=[])
    p.add_argument("--skip-nn",         action="store_true")
    p.add_argument("--skip-svm",        action="store_true",
                   help="Exclude SVM (RBF kernel is O(n²) — required at >100k samples)")
    p.add_argument("--max-train",       type=int,   default=None,
                   help="Subsample training set for fast iteration (e.g. 50000)")
    p.add_argument("--n-folds",         type=int,   default=5)
    p.add_argument("--min-review-tier", type=int,   default=3)
    p.add_argument("--auroc-target",    type=float, default=0.90)
    p.add_argument("--output",          default="outputs/phase2_eval")
    return p.parse_args()


def main() -> int:
    args   = parse_args()
    t0     = time.perf_counter()
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    clinvar = Path(args.clinvar)
    if not clinvar.exists():
        logger.error("ClinVar parquet not found: %s", clinvar)
        logger.error(
            "Build it with:\n"
            "  python -c \""
            "import logging; logging.basicConfig(level=logging.INFO); "
            "from src.data.database_connectors import ClinVarConnector; "
            "ClinVarConnector().fetch("
            "local_path='data/raw/clinvar/variant_summary.txt.gz'"
            ").to_parquet('data/processed/clinvar_grch38.parquet', index=False)"
            "\""
        )
        return 2

    logger.info("Configuration:")
    for k, v in vars(args).items():
        logger.info("  %-22s %s", k + ":", v)

    try:
        from src.data.real_data_prep import (
            AnnotationConfig, DataPrepConfig, DataPrepPipeline,
        )
        from src.models.variant_ensemble import EnsembleConfig, VariantEnsemble

        ann = AnnotationConfig(
            spliceai_path=Path(args.spliceai) if args.spliceai else None,
            alphamissense_path=Path(args.alphamissense) if args.alphamissense else None,
            gtex_genes=args.gtex_genes or [],
        )
        prep = DataPrepPipeline(
            config=DataPrepConfig(
                min_review_tier=args.min_review_tier,
                output_dir=outdir / "splits",
            ),
            annotation_config=ann,
        )
        X_train, X_test, X_val, y_train, y_test, y_val, meta = prep.run(
            clinvar_path=str(clinvar),
            gnomad_path=args.gnomad,
        )
        logger.info(
            "Data prep: train=%d val=%d test=%d features=%d",
            len(X_train), len(X_val), len(X_test), X_train.shape[1],
        )

        _poly_a = "A" * 101
        seq_tr  = pd.Series([_poly_a] * len(y_train))
        seq_te  = pd.Series([_poly_a] * len(y_test))
        seq_val = pd.Series([_poly_a] * len(y_val))

        if args.max_train and len(y_train) > args.max_train:
            idx = pd.Series(range(len(y_train))).sample(
                args.max_train, random_state=42
            ).values  # .values gives numpy array, required for .iloc
            X_train = X_train.iloc[idx].reset_index(drop=True)
            y_train = y_train.iloc[idx].reset_index(drop=True)
            seq_tr  = seq_tr.iloc[idx].reset_index(drop=True)
            logger.info("Subsampled training set to %d", args.max_train)

        ens_cfg = EnsembleConfig(n_folds=args.n_folds, model_dir=outdir / "models")
        ensemble = VariantEnsemble(ens_cfg)
        if args.skip_nn:
            ensemble.base_estimators.pop("cnn_1d",     None)
            ensemble.base_estimators.pop("tabular_nn", None)
        if args.skip_svm or len(y_train) > 100_000:
            ensemble.base_estimators.pop("svm", None)
            logger.info("SVM skipped: training set %d > 100K (O(n²) infeasible)", len(y_train))

        ensemble.fit(X_train, seq_tr, y_train)
        results     = ensemble.evaluate(X_test, seq_te, y_test)
        val_results = ensemble.evaluate(X_val,  seq_val, y_val)

        ens_row     = results.loc["ENSEMBLE_STACKER"]
        ens_val_row = val_results.loc["ENSEMBLE_STACKER"]
        elapsed = time.perf_counter() - t0
        m = {
            "auroc":           float(ens_row["auroc"]),
            "auprc":           float(ens_row["auprc"]),
            "f1":              float(ens_row["f1_macro"]),
            "mcc":             float(ens_row["mcc"]),
            "brier":           float(ens_row["brier"]),
            "val_auroc":       float(ens_val_row["auroc"]),
            "val_auprc":       float(ens_val_row["auprc"]),
            "val_f1":          float(ens_val_row["f1_macro"]),
            "val_mcc":         float(ens_val_row["mcc"]),
            "val_brier":       float(ens_val_row["brier"]),
            "elapsed_seconds": elapsed,
            "n_train":         int(len(y_train)),
            "n_val":           int(len(y_val)),
            "n_test":          int(len(y_test)),
            "n_features":      int(X_train.shape[1]),
        }

        (outdir / "metrics.json").write_text(json.dumps(m, indent=2))
        results.to_csv(outdir / "per_model_metrics.csv")
        val_results.to_csv(outdir / "per_model_metrics_val.csv")
        X_test.to_parquet(outdir / "X_test.parquet",    index=False)
        meta.to_parquet(outdir / "meta_test.parquet",   index=False)
        _save_feature_importance(ensemble, list(X_train.columns), outdir)

        auroc     = m["auroc"]
        val_auroc = m["val_auroc"]
        target    = args.auroc_target
        sep       = "-" * 52
        print(f"\n{sep}\n  Phase 2 Evaluation\n{sep}")
        print(f"  {'':8s}  {'Dev (test)':>12s}  {'Holdout (val)':>13s}")
        print(f"  {'AUROC':8s}  {auroc:>12.4f}  {val_auroc:>13.4f}  {'PASS' if val_auroc >= target else 'FAIL'}  (target >= {target})")
        print(f"  {'AUPRC':8s}  {m['auprc']:>12.4f}  {m['val_auprc']:>13.4f}")
        print(f"  {'Brier':8s}  {m['brier']:>12.4f}  {m['val_brier']:>13.4f}")
        print(f"  {'F1':8s}  {m['f1']:>12.4f}  {m['val_f1']:>13.4f}")
        print(f"  {'MCC':8s}  {m['mcc']:>12.4f}  {m['val_mcc']:>13.4f}")
        print(f"  Train: {m['n_train']:,} | Val: {m['n_val']:,} | Test: {m['n_test']:,} | Features: {m['n_features']}")
        print(f"  Time : {elapsed:.0f}s\n{sep}")

        if val_auroc >= target:
            print("\n  Holdout AUROC target met. Next: REST API + Docker deployment.\n")
        else:
            print(f"\n  Holdout AUROC below target by {target - val_auroc:.4f}. Check:")
            print("    1. --alphamissense path set? (adds signal for missense variants)")
            print("    2. --gnomad path set? (allele_freq is typically top feature)")
            print("    3. Review feature_importance.csv")
            print("    4. Try --min-review-tier 2 for expert-reviewed labels only\n")

        return 0 if val_auroc >= target else 1

    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        return 2


def _save_feature_importance(
    ensemble, feature_names: list[str], outdir: Path
) -> None:
    imps = {}
    for name, model in ensemble.trained_models_.items():
        base = getattr(model, "base_estimator", model)
        if hasattr(base, "feature_importances_"):
            imps[name] = list(base.feature_importances_)
    if not imps:
        return
    avg = (
        pd.DataFrame(imps, index=feature_names)
        .mean(axis=1)
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "feature", 0: "mean_importance"})
    )
    avg.to_csv(outdir / "feature_importance.csv", index=False)
    logger.info("Top 10 features:")
    for _, row in avg.head(10).iterrows():
        logger.info("  %-35s  %.4f", row["feature"], row["mean_importance"])


if __name__ == "__main__":
    sys.exit(main())