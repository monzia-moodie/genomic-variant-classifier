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
    p.add_argument("--kg",              default=None,
                   help="1000 Genomes Phase 3 AF parquet for gnomAD AF fallback")
    p.add_argument("--string-db",       default=None,
                   help="Enable GNN training. Value = STRING combined_score threshold "
                        "(int, default 700) or 'auto'.  Requires torch + torch_geometric.")
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
            kg_path=Path(args.kg) if args.kg else None,
        )
        prep = DataPrepPipeline(
            config=DataPrepConfig(
                min_review_tier=args.min_review_tier,
                output_dir=outdir / "splits",
            ),
            annotation_config=ann,
        )
        X_train, X_val, X_test, y_train, y_val, y_test, meta_val, meta = prep.run(
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
        ensemble.save(outdir / "models" / "ensemble.joblib")

        import joblib
        joblib.dump(prep.scaler, outdir / "scaler.joblib")
        logger.info("Scaler saved to %s/scaler.joblib", outdir)

        # ── Optional GNN training (5.2) ───────────────────────────────────
        gnn_scorer = None
        if args.string_db:
            try:
                from src.models.gnn import (
                    GNNScorer, StringDBGraph, build_pyg_dataset,
                    train_gnn_pipeline,
                )
                string_threshold = (
                    700 if args.string_db == "auto"
                    else int(args.string_db)
                )
                logger.info(
                    "GNN training: STRING threshold=%d, epochs=100", string_threshold
                )
                # Use the 55 tabular feature columns as node features
                node_feat_cols = [c for c in X_train.columns if c != "gnn_score"]

                # Build a raw training DataFrame for the GNN (needs gene_symbol + label)
                gnn_df = meta_val.iloc[:0].copy()   # schema reference
                # Attach gene_symbol from the ClinVar training rows
                # (meta_test has index aligned to test; we need train indices)
                # The simplest approach: re-build from the split parquet files if present
                split_train = outdir / "splits" / "X_train.parquet"
                if split_train.exists():
                    X_train_raw = pd.read_parquet(split_train)
                    gnn_df = X_train_raw.copy()
                    gnn_df["acmg_label"] = y_train.values
                else:
                    # Fallback: use X_train feature matrix (no gene_symbol → smaller GNN)
                    gnn_df = X_train.copy()
                    gnn_df["acmg_label"] = y_train.values

                gnn_model, gnn_trainer, gnn_history = train_gnn_pipeline(
                    variant_df=gnn_df,
                    node_feature_cols=node_feat_cols,
                    string_threshold=string_threshold,
                    test_split=0.15,
                    epochs=100,
                    batch_size=32,
                )
                joblib.dump(gnn_model, outdir / "models" / "gnn_model.joblib")

                # Build a GNNScorer for inference-time gene-level scoring
                builder = StringDBGraph(combined_score_threshold=string_threshold)
                graph = builder.build()
                full_dataset = build_pyg_dataset(gnn_df, graph, node_feat_cols)
                gnn_scorer = GNNScorer.from_trainer(gnn_trainer, full_dataset, gnn_df)
                joblib.dump(gnn_scorer, outdir / "models" / "gnn_scorer.joblib")

                # Overwrite gnn_score in feature matrices with real GNN predictions
                for split_name, split_df, X_split in [
                    ("train", gnn_df, X_train),
                    ("val",   meta_val, X_val),
                    ("test",  meta, X_test),
                ]:
                    if "gene_symbol" in split_df.columns:
                        X_split["gnn_score"] = (
                            split_df["gene_symbol"]
                            .fillna("")
                            .map(gnn_scorer.score)
                            .values
                        )
                        logger.info(
                            "GNN scores injected into %s split (mean=%.3f).",
                            split_name,
                            float(X_split["gnn_score"].mean()),
                        )
                logger.info("GNN training complete. Best val AUC: %.4f",
                            max(h["val_auc"] for h in gnn_history))
            except ImportError as exc:
                logger.warning(
                    "GNN skipped — missing dependency: %s.  "
                    "Install with: pip install torch torch-geometric", exc
                )
            except Exception as exc:
                logger.warning("GNN training failed: %s — continuing without GNN.", exc)

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
        X_test.to_parquet(outdir / "X_test.parquet",      index=False)
        meta.to_parquet(outdir / "meta_test.parquet",    index=False)
        meta_val.to_parquet(outdir / "meta_val.parquet", index=False)
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