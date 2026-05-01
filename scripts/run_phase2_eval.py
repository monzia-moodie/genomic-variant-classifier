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
    p.add_argument("--clinvar", required=True)
    p.add_argument("--gnomad", default=None)
    p.add_argument("--spliceai", default=None)
    p.add_argument("--alphamissense", default=None)
    p.add_argument("--gtex-genes", nargs="*", default=[])
    p.add_argument(
        "--kg",
        default=None,
        help="1000 Genomes Phase 3 AF parquet for gnomAD AF fallback",
    )
    p.add_argument(
        "--gnomad-constraint",
        default=None,
        help="Path to gnomAD v4.1 constraint TSV "
        "(data/external/gnomad/gnomad.v4.1.constraint_metrics.tsv)",
    )
    p.add_argument("--skip-nn", action="store_true")
    p.add_argument(
        "--string-db",
        default=None,
        help="Path to STRING DB file, or 'auto' to use config default",
    )
    p.add_argument(
        "--skip-svm",
        action="store_true",
        help="Exclude SVM (RBF kernel is O(n²) — required at >100k samples)",
    )
    p.add_argument(
        "--skip-kan",
        action="store_true",
        help="Exclude KAN (Kolmogorov-Arnold Network). KANClassifier already "
        "caps memory via a 100K-sample stratified subsample gate; this "
        "flag is provided as an optional override per ROADMAP checklist "
        "(do not hardcode removal again).",
    )
    p.add_argument(
        "--max-train",
        type=int,
        default=None,
        help="Subsample training set for fast iteration (e.g. 50000)",
    )
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--min-review-tier", type=int, default=3)
    p.add_argument("--auroc-target", type=float, default=0.90)
    p.add_argument("--output", default="outputs/phase2_eval")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    t0 = time.perf_counter()
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    clinvar = Path(args.clinvar)
    if not clinvar.exists():
        logger.error("ClinVar parquet not found: %s", clinvar)
        logger.error(
            "Build it with:\n"
            '  python -c "'
            "import logging; logging.basicConfig(level=logging.INFO); "
            "from src.data.database_connectors import ClinVarConnector; "
            "ClinVarConnector().fetch("
            "local_path='data/raw/clinvar/variant_summary.txt.gz'"
            ").to_parquet('data/processed/clinvar_grch38.parquet', index=False)"
            '"'
        )
        return 2

    logger.info("Configuration:")
    for k, v in vars(args).items():
        logger.info("  %-22s %s", k + ":", v)

    try:
        from src.data.real_data_prep import (
            AnnotationConfig,
            DataPrepConfig,
            DataPrepPipeline,
        )
        from src.models.variant_ensemble import (
            EnsembleConfig,
            VariantEnsemble,
            _write_model_manifest,
        )

        ann = AnnotationConfig(
            spliceai_path=Path(args.spliceai) if args.spliceai else None,
            alphamissense_path=Path(args.alphamissense) if args.alphamissense else None,
            gtex_genes=args.gtex_genes or [],
            kg_path=Path(args.kg) if args.kg else None,
            gnomad_constraint_path=(
                Path(args.gnomad_constraint) if args.gnomad_constraint else None
            ),
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
            len(X_train),
            len(X_val),
            len(X_test),
            X_train.shape[1],
        )

        _poly_a = "A" * 101
        seq_tr = pd.Series([_poly_a] * len(y_train))
        seq_te = pd.Series([_poly_a] * len(y_test))
        seq_val = pd.Series([_poly_a] * len(y_val))

        if args.max_train and len(y_train) > args.max_train:
            idx = (
                pd.Series(range(len(y_train)))
                .sample(args.max_train, random_state=42)
                .values
            )  # .values gives numpy array, required for .iloc
            X_train = X_train.iloc[idx].reset_index(drop=True)
            y_train = y_train.iloc[idx].reset_index(drop=True)
            seq_tr = seq_tr.iloc[idx].reset_index(drop=True)
            logger.info("Subsampled training set to %d", args.max_train)

        ens_cfg = EnsembleConfig(
            n_folds=args.n_folds,
            model_dir=outdir / "models",
            skip_kan=args.skip_kan,
        )
        _ensemble_path = outdir / "models" / "ensemble.joblib"
        if _ensemble_path.exists():
            import joblib as _jl

            logger.info("Resuming: loading existing ensemble from %s", _ensemble_path)
            ensemble = _jl.load(_ensemble_path)
        else:
            ensemble = VariantEnsemble(ens_cfg)
            if args.skip_nn:
                ensemble.base_estimators.pop("cnn_1d", None)
                ensemble.base_estimators.pop("tabular_nn", None)
            # Historical note: KAN was unconditionally removed after Run 4's
            # 17.9 GB C++ OOM at 1.2M samples (commit 2389ee2 on 2026-04-04).
            # Commit 2389ee2 shipped a 100K stratified subsample gate in
            # KANClassifier._fit_pykan that caps peak RAM at ~0.3 GB. The
            # hardcoded pop() is therefore no longer needed; skip_kan is now
            # an opt-in config flag. See docs/ROADMAP.md KAN Re-enablement
            # Checklist (items 3 and 4) and the 2026-04-20 session log.
            if args.skip_kan:
                ensemble.base_estimators.pop("kan", None)
                logger.info("KAN skipped: --skip-kan flag set.")
            if args.skip_svm or len(y_train) > 100_000:
                ensemble.base_estimators.pop("svm", None)
                logger.info(
                    "SVM skipped: training set %d > 100K (O(n²) infeasible)",
                    len(y_train),
                )
            ensemble.fit(X_train, seq_tr, y_train)
            ensemble.save(_ensemble_path)

        import joblib

        joblib.dump(prep.scaler, outdir / "scaler.joblib")
        _write_model_manifest(outdir / "scaler.joblib")
        logger.info("Scaler saved to %s/scaler.joblib", outdir)

        # ── Optional GNN training (5.2) ───────────────────────────────────
        gnn_scorer = None
        logger.info(
            "[GNN-TRACE] entry: args.string_db=%r",
            getattr(args, "string_db", None),
        )
        if args.string_db:
            try:
                logger.info("[GNN-TRACE] gate-passed: entering GNN block")
                logger.info("[GNN-TRACE] importing src.models.gnn ...")
                from src.models.gnn import (
                    GNNScorer,
                    StringDBGraph,
                    build_pyg_dataset,
                    train_gnn_pipeline,
                )
                logger.info(
                    "[GNN-TRACE] import OK: GNNScorer/StringDBGraph/build_pyg_dataset/train_gnn_pipeline resolved"
                )

                # string_db may be a file path or a threshold integer string
                _sd = args.string_db
                if _sd == "auto" or not _sd.lstrip("-").isdigit():
                    string_threshold = 700
                else:
                    string_threshold = int(_sd)
                string_db_path = None if _sd == "auto" else _sd
                logger.info(
                    "GNN training: STRING threshold=%d, epochs=100", string_threshold
                )
                # Use the 55 tabular feature columns as node features
                node_feat_cols = [c for c in X_train.columns if c != "gnn_score"]

                # Build a raw training DataFrame for the GNN (needs gene_symbol + label).
                # Patch 6b (2026-04-30): source gene_symbol from meta_train.parquet,
                # which DataPrepPipeline now persists alongside the feature matrix.
                # Previous implementation reloaded X_train.parquet (a 78-col numeric
                # matrix with NO gene_symbol) and crashed inside build_pyg_dataset.
                _meta_train_path = outdir / "splits" / "meta_train.parquet"
                if _meta_train_path.exists():
                    _meta_train = pd.read_parquet(_meta_train_path)
                    gnn_df = X_train.copy().reset_index(drop=True)
                    gnn_df["gene_symbol"] = (
                        _meta_train["gene_symbol"].fillna("").reset_index(drop=True)
                    )
                    gnn_df["acmg_label"] = y_train.values
                    logger.info(
                        "[GNN-TRACE] meta_train.parquet sourced gene_symbol "
                        "(unique_genes=%d, missing=%d)",
                        gnn_df["gene_symbol"].nunique(),
                        int((gnn_df["gene_symbol"] == "").sum()),
                    )
                else:
                    logger.warning(
                        "[GNN-TRACE] meta_train.parquet missing at %s; "
                        "GNN training cannot proceed (no gene_symbol). "
                        "Re-run DataPrepPipeline to regenerate splits.",
                        _meta_train_path,
                    )
                    raise FileNotFoundError(_meta_train_path)

                # Resolve STRING DB links file: prefer explicit --string-db path
                _local_links = (
                    Path(string_db_path)
                    if string_db_path and Path(string_db_path).exists()
                    else Path(
                        "data/external/string/9606.protein.links.detailed.v12.0.txt.gz"
                    )
                )
                _local_info = Path(
                    "data/external/string/9606.protein.info.v12.0.txt.gz"
                )
                _string_kwargs = dict(
                    combined_score_threshold=string_threshold,
                    local_links_path=_local_links if _local_links.exists() else None,
                    local_info_path=_local_info if _local_info.exists() else None,
                )
                logger.info(
                    "[GNN-TRACE] local_links exists=%s (%s)",
                    _local_links.exists(), _local_links,
                )
                logger.info(
                    "[GNN-TRACE] local_info  exists=%s (%s)",
                    _local_info.exists(), _local_info,
                )
                logger.info(
                    "[GNN-TRACE] gnn_df rows=%d cols=%d has_gene_symbol=%s",
                    len(gnn_df), len(gnn_df.columns),
                    "gene_symbol" in gnn_df.columns,
                )
                logger.info("[GNN-TRACE] train_gnn_pipeline begin")
                _gnn_t0 = time.perf_counter()

                gnn_model, gnn_trainer, gnn_history = train_gnn_pipeline(
                    variant_df=gnn_df,
                    node_feature_cols=node_feat_cols,
                    string_threshold=string_threshold,
                    test_split=0.15,
                    epochs=100,
                    batch_size=32,
                )
                logger.info(
                    "[GNN-TRACE] train_gnn_pipeline done in %.2fs",
                    time.perf_counter() - _gnn_t0,
                )
                joblib.dump(gnn_model, outdir / "models" / "gnn_model.joblib")
                _write_model_manifest(outdir / "models" / "gnn_model.joblib")

                # Build a GNNScorer for inference-time gene-level scoring
                builder = StringDBGraph(**_string_kwargs)
                graph = builder.build()
                full_dataset = build_pyg_dataset(gnn_df, graph, node_feat_cols)
                gnn_scorer = GNNScorer.from_trainer(gnn_trainer, full_dataset, gnn_df)
                logger.info(
                    "[GNN-TRACE] gnn_scorer built (type=%s); "
                    "graph_nodes=%d graph_edges=%d",
                    type(gnn_scorer).__name__,
                    int(getattr(graph, "num_nodes", -1)),
                    int(getattr(graph, "num_edges", -1)),
                )
                joblib.dump(gnn_scorer, outdir / "models" / "gnn_scorer.joblib")
                _write_model_manifest(outdir / "models" / "gnn_scorer.joblib")

                # Overwrite gnn_score in feature matrices with real GNN predictions
                for split_name, split_df, X_split in [
                    ("train", gnn_df, X_train),
                    ("val", meta_val, X_val),
                    ("test", meta, X_test),
                ]:
                    logger.info(
                        "[GNN-TRACE] split=%s split_df.cols=%d has_gene_symbol=%s",
                        split_name, len(split_df.columns),
                        "gene_symbol" in split_df.columns,
                    )
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
                        _s = X_split["gnn_score"]
                        logger.info(
                            "[GNN-TRACE] post-injection split=%s rows=%d "
                            "min=%.4f max=%.4f std=%.4f nonzero_frac=%.4f",
                            split_name, len(_s),
                            float(_s.min()), float(_s.max()),
                            float(_s.std()), float((_s != 0).mean()),
                        )
                    else:
                        logger.warning(
                            "[GNN-TRACE] split=%s MISSING gene_symbol; "
                            "gnn_score will remain at default. "
                            "split_df sample columns: %s",
                            split_name, list(split_df.columns)[:10],
                        )

                # Patch 6a — re-persist split parquets with real GNN scores so
                # the ablation harness (scripts/run9_ablations.py) reads the
                # correct gnn_score when loading splits from disk. Without
                # this, the on-disk parquets retain the default 0.0 from
                # DataPrepPipeline._engineer_features.
                _splits_dir = outdir / "splits"
                if _splits_dir.exists():
                    X_train.to_parquet(_splits_dir / "X_train.parquet", index=False)
                    X_val.to_parquet(_splits_dir / "X_val.parquet", index=False)
                    X_test.to_parquet(_splits_dir / "X_test.parquet", index=False)
                    logger.info("GNN-updated splits re-persisted to %s/", _splits_dir)
                    for _f in ("X_train.parquet", "X_val.parquet", "X_test.parquet"):
                        _p = _splits_dir / _f
                        logger.info(
                            "[GNN-TRACE] wrote %s size=%d bytes",
                            _p,
                            _p.stat().st_size if _p.exists() else -1,
                        )
                else:
                    logger.warning(
                        "[GNN-TRACE] splits_dir does not exist (%s); "
                        "re-persist SKIPPED",
                        _splits_dir,
                    )

                logger.info(
                    "GNN training complete. Best val AUC: %.4f",
                    max(h["val_auc"] for h in gnn_history),
                )
            except ImportError as exc:
                logger.warning("[GNN-TRACE] ImportError caught: %s", exc)
                logger.warning(
                    "GNN skipped — missing dependency: %s.  "
                    "Install with: pip install torch torch-geometric",
                    exc,
                )
            except Exception as exc:
                logger.warning(
                    "[GNN-TRACE] generic Exception caught: %s: %s",
                    type(exc).__name__, exc, exc_info=True,
                )
                logger.warning("GNN training failed: %s — continuing without GNN.", exc)
        else:
            logger.warning(
                "[GNN-TRACE] gate-skipped: args.string_db is falsy (%r); "
                "ENTIRE GNN BLOCK skipped",
                getattr(args, "string_db", None),
            )

        results = ensemble.evaluate(X_test, seq_te, y_test)
        val_results = ensemble.evaluate(X_val, seq_val, y_val)

        ens_row = results.loc["ENSEMBLE_STACKER"]
        ens_val_row = val_results.loc["ENSEMBLE_STACKER"]
        elapsed = time.perf_counter() - t0
        m = {
            "auroc": float(ens_row["auroc"]),
            "auprc": float(ens_row["auprc"]),
            "f1": float(ens_row["f1_macro"]),
            "mcc": float(ens_row["mcc"]),
            "brier": float(ens_row["brier"]),
            "val_auroc": float(ens_val_row["auroc"]),
            "val_auprc": float(ens_val_row["auprc"]),
            "val_f1": float(ens_val_row["f1_macro"]),
            "val_mcc": float(ens_val_row["mcc"]),
            "val_brier": float(ens_val_row["brier"]),
            "elapsed_seconds": elapsed,
            "n_train": int(len(y_train)),
            "n_val": int(len(y_val)),
            "n_test": int(len(y_test)),
            "n_features": int(X_train.shape[1]),
        }

        (outdir / "metrics.json").write_text(json.dumps(m, indent=2))
        results.to_csv(outdir / "per_model_metrics.csv")
        val_results.to_csv(outdir / "per_model_metrics_val.csv")
        X_test.to_parquet(outdir / "X_test.parquet", index=False)
        meta.to_parquet(outdir / "meta_test.parquet", index=False)
        meta_val.to_parquet(outdir / "meta_val.parquet", index=False)
        _save_feature_importance(ensemble, list(X_train.columns), outdir)

        auroc = m["auroc"]
        val_auroc = m["val_auroc"]
        target = args.auroc_target
        sep = "-" * 52
        print(f"\n{sep}\n  Phase 2 Evaluation\n{sep}")
        print(f"  {'':8s}  {'Dev (test)':>12s}  {'Holdout (val)':>13s}")
        print(
            f"  {'AUROC':8s}  {auroc:>12.4f}  {val_auroc:>13.4f}  {'PASS' if val_auroc >= target else 'FAIL'}  (target >= {target})"
        )
        print(f"  {'AUPRC':8s}  {m['auprc']:>12.4f}  {m['val_auprc']:>13.4f}")
        print(f"  {'Brier':8s}  {m['brier']:>12.4f}  {m['val_brier']:>13.4f}")
        print(f"  {'F1':8s}  {m['f1']:>12.4f}  {m['val_f1']:>13.4f}")
        print(f"  {'MCC':8s}  {m['mcc']:>12.4f}  {m['val_mcc']:>13.4f}")
        print(
            f"  Train: {m['n_train']:,} | Val: {m['n_val']:,} | Test: {m['n_test']:,} | Features: {m['n_features']}"
        )
        print(f"  Time : {elapsed:.0f}s\n{sep}")

        if val_auroc >= target:
            print("\n  Holdout AUROC target met. Next: REST API + Docker deployment.\n")
        else:
            print(f"\n  Holdout AUROC below target by {target - val_auroc:.4f}. Check:")
            print(
                "    1. --alphamissense path set? (adds signal for missense variants)"
            )
            print("    2. --gnomad path set? (allele_freq is typically top feature)")
            print("    3. Review feature_importance.csv")
            print("    4. Try --min-review-tier 2 for expert-reviewed labels only\n")

        return 0 if val_auroc >= target else 1

    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        return 2


def _save_feature_importance(ensemble, feature_names: list[str], outdir: Path) -> None:
    imps = {}
    for name, model in ensemble.trained_models_.items():
        base = getattr(model, "_base", getattr(model, "base_estimator", model))
        if hasattr(base, "feature_importances_"):
            fi = base.feature_importances_
            if callable(fi):
                fi = fi()
            imps[name] = list(fi)
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
