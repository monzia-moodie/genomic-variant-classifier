# INCIDENT 2026-04-30 — GNN training silent-zero (gene_symbol KeyError)

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
