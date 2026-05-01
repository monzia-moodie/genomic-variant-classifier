"""Generates SESSION + INCIDENT + CHANGELOG entries for 2026-04-30."""

from __future__ import annotations
from pathlib import Path
from datetime import date

today = date.today().isoformat()

# ── docs/sessions/SESSION_2026-04-30_run9_ready_regen.md ──────────────────
session = f"""# SESSION {today} — run9_ready regen + GNN bug discovery

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
"""

# ── docs/incidents/INCIDENT_2026-04-30_gnn-gene-symbol-keyerror.md ────────
incident = """# INCIDENT 2026-04-30 — GNN training silent-zero (gene_symbol KeyError)

## Status
DIAGNOSED. Fix drafted as Patch 6b (scripts/apply_patch_6b.py).
NOT YET RESOLVED — fix not committed.

## Symptoms
- outputs/phase2_full/splits/X_train.parquet (4/16): gnn_score=0 std=0
- outputs/run9_fresh/splits/X_train.parquet (4/19): gnn_score=0 std=0
- outputs/run9_ready/splits/X_train.parquet (4/30): gnn_score=0 std=0
- regen.log shows GNN block enters but fails with WARNING, not ERROR
- Phase 2 evaluation reports "PASS" because AUROC target met regardless

## Root cause (confirmed via [GNN-TRACE] 2026-04-30)
scripts/run_phase2_eval.py lines 246-254:

    split_train = outdir / "splits" / "X_train.parquet"
    if split_train.exists():
        X_train_raw = pd.read_parquet(split_train)
        gnn_df = X_train_raw.copy()              # ← 78-col matrix, no gene_symbol
        gnn_df["acmg_label"] = y_train.values

DataPrepPipeline._save_splits writes X_train.parquet from the
post-_engineer_features matrix (numeric features only, gene_symbol
dropped). When run_phase2_eval reloads it for gnn_df, gene_symbol
is gone. build_pyg_dataset(src/models/gnn.py:248) calls:

    grp = variant_df.groupby("gene_symbol")[feat].mean()

and raises KeyError: 'gene_symbol'.

The `except Exception as exc:` handler at run_phase2_eval.py:333
downgrades the crash to a warning. Patch 6a (commit abdb1d2) at lines
311-321 then re-persists X_train/X_val/X_test back to disk regardless,
overwriting whatever was on disk with the same all-zero gnn_score that
DataPrepPipeline._engineer_features wrote in the first place.

Net effect: silent failure, on-disk gnn_score=0, Patch 6a's intent
defeated.

## How [GNN-TRACE] caught it
Insertion 4 (post-_string_kwargs):
    [GNN-TRACE] gnn_df rows=1197216 cols=79 has_gene_symbol=False

Insertion 9 (except Exception with exc_info=True):
    [GNN-TRACE] generic Exception caught: KeyError: 'gene_symbol'
    Traceback: src/models/gnn.py:248 grp = variant_df.groupby(...)

Without [GNN-TRACE], the only signal was the existing warning:
    GNN training failed: 'gene_symbol' — continuing without GNN.
which gives no traceback and no upstream context.

## Fix (Patch 6b)
Two atomic edits:

1. src/data/real_data_prep.py
   - _save_splits accepts new optional meta_train: pd.DataFrame
   - persists meta_train.parquet alongside meta_val.parquet/meta_test.parquet
   - run() passes df.iloc[train_idx] as meta_train

2. scripts/run_phase2_eval.py
   - replaces the X_train.parquet reload with a meta_train.parquet read
   - merges gene_symbol into gnn_df from meta_train
   - raises FileNotFoundError if meta_train.parquet missing (no silent fallback)

## Verification plan
1. scripts/apply_patch_6b.py --dry-run — anchor count = 1 for all 4 anchors
2. scripts/apply_patch_6b.py — applies edits
3. scripts/probe_patch_6b.py — 5K-row synthetic probe end-to-end
4. Full Vast.ai Run 9a — true end-to-end with GNN training

## Forbidden actions
- DO NOT retrain ensembles locally to "verify the fix end-to-end"
  (memory #19: 13h CPU cost is unacceptable)
- DO NOT modify the Patch 6a re-persist block — it's correct given
  a successful upstream training; the fix is upstream.
"""

# ── docs/CHANGELOG.md (append-only) ──────────────────────────────────────
changelog_entry = f"""

## {today}

### Attempted
- Stage 3 splits regen (run_phase2_eval.py with [GNN-TRACE]
  instrumentation, --skip-nn --skip-svm --skip-kan, --string-db auto,
  --n-folds 2, output outputs/run9_ready/)

### Failed
- GNN training: KeyError 'gene_symbol' in build_pyg_dataset.
  Caught by `except Exception` and downgraded to warning. on-disk
  gnn_score remained 0.0 across all three splits.
- --skip-nn flag did not skip mc_dropout/deep_ensemble (memory #17
  confirmed). Wall-clock cost: 10h+ of the 13h total runtime.

### Fixed (this session)
- Stage 1: .venv312 bootstrapped on Python 3.12.10. requirements.txt +
  torch 2.11.0+cpu + torch_geometric 2.7.0 installed cleanly.
  Pandas pinned to 2.3.3 (was 3.0.1).
- Stage 2: [GNN-TRACE] instrumentation patch landed in
  scripts/run_phase2_eval.py (18 logger calls, 4/4 verification gates
  green). Backup at scripts/run_phase2_eval.py.bak-gnn-trace.
- Stage 3: data prep + ensemble training completed end-to-end.
  Test AUROC 0.9814, val AUROC 0.9850.

### Drafted (committed in next session)
- Patch 6b (scripts/apply_patch_6b.py): persist meta_train.parquet
  in DataPrepPipeline._save_splits, source gene_symbol from it in
  run_phase2_eval.py for gnn_df construction.
- 5K-row synthetic probe (scripts/probe_patch_6b.py).

### Learned
- Generic `except Exception: logger.warning` masks crashes. Either
  narrow the except or use exc_info=True. [GNN-TRACE] insertion 9
  uses exc_info=True and would have surfaced this immediately on
  first run.
- Patches that re-persist on success path must verify success
  before persisting. Patch 6a re-persists regardless of whether
  gnn_scorer was built.
- Memory #19 (no local retraining) was violated this session at
  cost of 13h. Reaffirming.
- run9_ready splits are a valid GNN-FREE BASELINE for paper P4
  comparison. Don't discard.
"""

# Write
docs = Path("docs")
(docs / "sessions").mkdir(parents=True, exist_ok=True)
(docs / "incidents").mkdir(parents=True, exist_ok=True)
(docs / "sessions" / f"SESSION_{today}_run9_ready_regen.md").write_text(
    session, encoding="utf-8"
)
(docs / "incidents" / f"INCIDENT_{today}_gnn-gene-symbol-keyerror.md").write_text(
    incident, encoding="utf-8"
)

cl = docs / "CHANGELOG.md"
if cl.exists():
    cl.write_text(cl.read_text(encoding="utf-8") + changelog_entry, encoding="utf-8")
else:
    cl.write_text(f"# Changelog\n{changelog_entry}", encoding="utf-8")

print(f"OK: wrote SESSION_{today}_run9_ready_regen.md")
print(f"OK: wrote INCIDENT_{today}_gnn-gene-symbol-keyerror.md")
print(f"OK: appended to docs/CHANGELOG.md")
