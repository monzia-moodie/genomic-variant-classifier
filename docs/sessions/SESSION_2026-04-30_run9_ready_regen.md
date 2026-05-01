# SESSION 2026-04-30 — run9_ready regen + GNN bug discovery

## Summary

Regenerated training splits at outputs/run9_ready/splits/ via the patched
[GNN-TRACE] instrumented run_phase2_eval.py. Wall-clock 13h 14min on CPU.
Test AUROC 0.9814, val AUROC 0.9850 — Phase 2 target met.

The [GNN-TRACE] instrumentation surfaced a previously-silent bug:
build_pyg_dataset crashes with KeyError 'gene_symbol' because gnn_df
is built from X_train.parquet (a 78-col pure numeric matrix with no
gene_symbol). Pre-instrumentation, this was caught by a generic
`except Exception` and downgraded to a warning, leaving gnn_score=0
on disk. Patch 6b drafted to fix.

## Cohort statistics

| Split | Variants | Pathogenic | Genes |
|---|---:|---:|---:|
| Train | 1,197,216 | 243,540 (20.3%) | 16,240 |
| Val   |   154,404 |  27,640 (17.9%) |  2,320 |
| Test  |   349,067 |  69,791 (20.0%) |  4,641 |

Total labeled cohort: 1,700,687 (memory entry "1.2M variants" was train fold;
true full label set is 1.7M as of ClinVar 2026-04-30).

## Headline metrics (GNN-FREE — gnn_score=0 throughout)

| Metric  | Test   | Val    |
|---------|--------|--------|
| AUROC   | 0.9814 | 0.9850 |
| AUPRC   | 0.9356 | 0.9402 |
| Brier   | 0.0482 | 0.0385 |
| F1      | 0.8912 | 0.8992 |
| MCC     | 0.7856 | 0.7992 |

OOF base-model ranking: lightgbm 0.9911 > xgboost 0.9908 > catboost 0.9900
> gradient_boosting 0.9889 > random_forest 0.9881 > deep_ensemble 0.9872
> mc_dropout 0.9870 > logistic_regression 0.9849.

Blend AUROC 0.9915 (Nelder-Mead) > LR stacker 0.9907.

Blend weights: rf=0.355, xgb=0.207, lgb=0.304, lr=0.0004, gbm=0.033,
cat=0.046, mc_dropout=0.052, deep_ensemble=0.002.

## Findings

1. **Bug 1 (Patch 6b)** — gnn_df built from X_train.parquet has no
   gene_symbol. KeyError in build_pyg_dataset:248. Fix: persist
   meta_train.parquet in DataPrepPipeline._save_splits, source
   gene_symbol from there in run_phase2_eval.py.

2. **Bug 2 (--skip-nn)** — flag does NOT skip mc_dropout/deep_ensemble.
   Cost: 10h+ of the 13h regen. Memory #17 already documented this;
   Run 9a launch must use --skip-mc-dropout + --skip-deep-ensemble
   (flags need to be added to argparse) or run on Vast.ai.

3. **Bug 3 (mc_dropout no-op)** — TabularNNClassifier doesn't expose
   _predict_proba_single_pass. Wrapper warns 3 times and produces
   zero uncertainty. Effectively reduces to a single TabularNN.
   Filed for paper P2 (5-tier ACMG calibration).

4. **Connector silent-zeros** — 30+ of 78 columns are all-zero:
   DbNSFP, PhyloP, GTEx, OMIM, ClinGen, dbSNP, EVE, HGMD, RNA splice,
   protein structure, LOVD, ESM-2 (stub), gnomAD constraint pLI.
   Surviving signal: AlphaMissense (12% covered), SpliceAI (9%),
   n_pathogenic_in_gene (importance 1213.5 = 3.3x next).

5. **Wall-clock realism** — 13h 14min is the cost of a full retrain on
   CPU with NN ensembles. Memory #19 says NEVER retrain locally; this
   run violated that rule. Not repeating.

## What worked

- [GNN-TRACE] instrumentation: 11/18 trace points fired (loop never
  reached due to upstream crash), pinpointed the bug to a single line
  in 7 seconds of GNN time.
- 78-column schema confirmed live (memory #14 superseded).
- AUROC target hit despite GNN failure — model is strong without it.

## What broke

- Patch 6a (commit abdb1d2) introduced the silent zero by reloading
  X_train.parquet for gnn_df instead of using meta_train.
- `except Exception as exc: logger.warning(...)` masked the failure
  for 14 days. This handler should be narrowed or always log
  exc_info=True (now done in [GNN-TRACE] patch).

## Lessons

- Generic exception handlers that downgrade exceptions to warnings
  are a silent-failure factory. Either narrow the except or always
  log with exc_info=True.
- Re-persist patches that depend on a successful upstream computation
  must check the upstream succeeded before re-persisting (Patch 6a
  re-persisted whatever was in memory, including the all-zero default).
- The [GNN-TRACE] pattern (logger.info at every checkpoint, never
  remove after debugging) is keeper code. It earned its commit on
  this run.

## Files written

- outputs/run9_ready/splits/X_train.parquet (1,197,216 x 78, gnn_score=0)
- outputs/run9_ready/splits/X_val.parquet (154,404 x 78, gnn_score=0)
- outputs/run9_ready/splits/X_test.parquet (349,067 x 78, gnn_score=0)
- outputs/run9_ready/splits/y_train,val,test.parquet
- outputs/run9_ready/splits/meta_val.parquet (5.85 MB, has gene_symbol)
- outputs/run9_ready/splits/meta_test.parquet (12.69 MB, has gene_symbol)
- outputs/run9_ready/models/ensemble.joblib (full 8-base + meta-learner)
- outputs/run9_ready/scaler.joblib
- outputs/run9_ready/regen.log (full 13h trace)

NOT WRITTEN (reason): outputs/run9_ready/splits/meta_train.parquet
— DataPrepPipeline did not yet persist this. Patch 6b adds it.

## Next session

1. Apply Patch 6b (scripts/apply_patch_6b.py)
2. Synthetic 5K probe (scripts/probe_patch_6b.py)
3. Commit [GNN-TRACE] instrumentation + Patch 6b on branch
4. Open PR, merge, tag run9a-baseline (NOT run9a-ready — GNN still
   needs end-to-end verification on Vast.ai)
5. Defer full GNN-injected splits to Vast.ai Run 9a launch
