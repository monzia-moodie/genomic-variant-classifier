"""
apply_gnn_trace_patch.py - Stage 2 of pre-Run-9a plan.

Inserts logger.info/warning calls (all prefixed [GNN-TRACE]) into
scripts/run_phase2_eval.py at every critical path of the GNN execution
block. Pure logging additions; zero logic changes.

Em-dash-free anchors throughout to avoid Python source encoding pitfalls.
The line-334 em-dash ("continuing without GNN.") is no longer part of any
matched anchor; we anchor on lines 335 (blank) + 336 (results = ...) instead.

Idempotent: re-running on an already-patched file is a no-op.
Atomic: validates AST after patching; backup is written before TARGET.

Usage:
    python scripts/apply_gnn_trace_patch.py
    python scripts/apply_gnn_trace_patch.py --dry-run
    python scripts/apply_gnn_trace_patch.py --revert
"""

from __future__ import annotations

import argparse
import ast
import shutil
import sys
from pathlib import Path

TARGET = Path("scripts/run_phase2_eval.py")
BACKUP = Path("scripts/run_phase2_eval.py.bak-gnn-trace")

# ----- anchor-after insertions -----------------------------------------
INSERTIONS: list[tuple[str, str]] = [
    # 1. before 'if args.string_db:'
    (
        "        gnn_scorer = None\n",
        "        logger.info(\n"
        '            "[GNN-TRACE] entry: args.string_db=%r",\n'
        '            getattr(args, "string_db", None),\n'
        "        )\n",
    ),
    # 2. immediately after 'try:' inside 'if args.string_db:'
    (
        "        if args.string_db:\n            try:\n",
        '                logger.info("[GNN-TRACE] gate-passed: entering GNN block")\n'
        '                logger.info("[GNN-TRACE] importing src.models.gnn ...")\n',
    ),
    # 3. after the 'from src.models.gnn import (...)' block
    (
        "                from src.models.gnn import (\n"
        "                    GNNScorer,\n"
        "                    StringDBGraph,\n"
        "                    build_pyg_dataset,\n"
        "                    train_gnn_pipeline,\n"
        "                )\n",
        "                logger.info(\n"
        '                    "[GNN-TRACE] import OK: '
        'GNNScorer/StringDBGraph/build_pyg_dataset/train_gnn_pipeline resolved"\n'
        "                )\n",
    ),
    # 4. after _string_kwargs dict, before train_gnn_pipeline call
    (
        "                _string_kwargs = dict(\n"
        "                    combined_score_threshold=string_threshold,\n"
        "                    local_links_path=_local_links if _local_links.exists() else None,\n"
        "                    local_info_path=_local_info if _local_info.exists() else None,\n"
        "                )\n",
        "                logger.info(\n"
        '                    "[GNN-TRACE] local_links exists=%s (%s)",\n'
        "                    _local_links.exists(), _local_links,\n"
        "                )\n"
        "                logger.info(\n"
        '                    "[GNN-TRACE] local_info  exists=%s (%s)",\n'
        "                    _local_info.exists(), _local_info,\n"
        "                )\n"
        "                logger.info(\n"
        '                    "[GNN-TRACE] gnn_df rows=%d cols=%d has_gene_symbol=%s",\n'
        "                    len(gnn_df), len(gnn_df.columns),\n"
        '                    "gene_symbol" in gnn_df.columns,\n'
        "                )\n"
        '                logger.info("[GNN-TRACE] train_gnn_pipeline begin")\n'
        "                _gnn_t0 = time.perf_counter()\n",
    ),
    # 5. after train_gnn_pipeline(...) closing paren
    (
        "                gnn_model, gnn_trainer, gnn_history = train_gnn_pipeline(\n"
        "                    variant_df=gnn_df,\n"
        "                    node_feature_cols=node_feat_cols,\n"
        "                    string_threshold=string_threshold,\n"
        "                    test_split=0.15,\n"
        "                    epochs=100,\n"
        "                    batch_size=32,\n"
        "                )\n",
        "                logger.info(\n"
        '                    "[GNN-TRACE] train_gnn_pipeline done in %.2fs",\n'
        "                    time.perf_counter() - _gnn_t0,\n"
        "                )\n",
    ),
    # 6. after gnn_scorer = GNNScorer.from_trainer(...)
    (
        "                gnn_scorer = GNNScorer.from_trainer(gnn_trainer, full_dataset, gnn_df)\n",
        "                logger.info(\n"
        '                    "[GNN-TRACE] gnn_scorer built (type=%s); "\n'
        '                    "graph_nodes=%d graph_edges=%d",\n'
        "                    type(gnn_scorer).__name__,\n"
        '                    int(getattr(graph, "num_nodes", -1)),\n'
        '                    int(getattr(graph, "num_edges", -1)),\n'
        "                )\n",
    ),
    # 7. after 're-persisted to' log line
    (
        '                    logger.info("GNN-updated splits re-persisted to %s/", _splits_dir)\n',
        '                    for _f in ("X_train.parquet", "X_val.parquet", "X_test.parquet"):\n'
        "                        _p = _splits_dir / _f\n"
        "                        logger.info(\n"
        '                            "[GNN-TRACE] wrote %s size=%d bytes",\n'
        "                            _p,\n"
        "                            _p.stat().st_size if _p.exists() else -1,\n"
        "                        )\n",
    ),
    # 8. after 'except ImportError as exc:'
    (
        "            except ImportError as exc:\n",
        '                logger.warning("[GNN-TRACE] ImportError caught: %s", exc)\n',
    ),
    # 9. after 'except Exception as exc:'
    (
        "            except Exception as exc:\n",
        "                logger.warning(\n"
        '                    "[GNN-TRACE] generic Exception caught: %s: %s",\n'
        "                    type(exc).__name__, exc, exc_info=True,\n"
        "                )\n",
    ),
]

# ----- block replacements ----------------------------------------------

# (a) Loop body rewrite: per-split stats + else branch on missing gene_symbol
LOOP_BLOCK_OLD = (
    "                for split_name, split_df, X_split in [\n"
    '                    ("train", gnn_df, X_train),\n'
    '                    ("val", meta_val, X_val),\n'
    '                    ("test", meta, X_test),\n'
    "                ]:\n"
    '                    if "gene_symbol" in split_df.columns:\n'
    '                        X_split["gnn_score"] = (\n'
    '                            split_df["gene_symbol"]\n'
    '                            .fillna("")\n'
    "                            .map(gnn_scorer.score)\n"
    "                            .values\n"
    "                        )\n"
    "                        logger.info(\n"
    '                            "GNN scores injected into %s split (mean=%.3f).",\n'
    "                            split_name,\n"
    '                            float(X_split["gnn_score"].mean()),\n'
    "                        )\n"
)

LOOP_BLOCK_NEW = (
    "                for split_name, split_df, X_split in [\n"
    '                    ("train", gnn_df, X_train),\n'
    '                    ("val", meta_val, X_val),\n'
    '                    ("test", meta, X_test),\n'
    "                ]:\n"
    "                    logger.info(\n"
    '                        "[GNN-TRACE] split=%s split_df.cols=%d has_gene_symbol=%s",\n'
    "                        split_name, len(split_df.columns),\n"
    '                        "gene_symbol" in split_df.columns,\n'
    "                    )\n"
    '                    if "gene_symbol" in split_df.columns:\n'
    '                        X_split["gnn_score"] = (\n'
    '                            split_df["gene_symbol"]\n'
    '                            .fillna("")\n'
    "                            .map(gnn_scorer.score)\n"
    "                            .values\n"
    "                        )\n"
    "                        logger.info(\n"
    '                            "GNN scores injected into %s split (mean=%.3f).",\n'
    "                            split_name,\n"
    '                            float(X_split["gnn_score"].mean()),\n'
    "                        )\n"
    '                        _s = X_split["gnn_score"]\n'
    "                        logger.info(\n"
    '                            "[GNN-TRACE] post-injection split=%s rows=%d "\n'
    '                            "min=%.4f max=%.4f std=%.4f nonzero_frac=%.4f",\n'
    "                            split_name, len(_s),\n"
    "                            float(_s.min()), float(_s.max()),\n"
    "                            float(_s.std()), float((_s != 0).mean()),\n"
    "                        )\n"
    "                    else:\n"
    "                        logger.warning(\n"
    '                            "[GNN-TRACE] split=%s MISSING gene_symbol; "\n'
    '                            "gnn_score will remain at default. "\n'
    '                            "split_df sample columns: %s",\n'
    "                            split_name, list(split_df.columns)[:10],\n"
    "                        )\n"
)

# (b) else branch for `if _splits_dir.exists():`
SPLITS_BLOCK_OLD = (
    "                if _splits_dir.exists():\n"
    '                    X_train.to_parquet(_splits_dir / "X_train.parquet", index=False)\n'
    '                    X_val.to_parquet(_splits_dir / "X_val.parquet", index=False)\n'
    '                    X_test.to_parquet(_splits_dir / "X_test.parquet", index=False)\n'
    '                    logger.info("GNN-updated splits re-persisted to %s/", _splits_dir)\n'
)

SPLITS_BLOCK_NEW = (
    "                if _splits_dir.exists():\n"
    '                    X_train.to_parquet(_splits_dir / "X_train.parquet", index=False)\n'
    '                    X_val.to_parquet(_splits_dir / "X_val.parquet", index=False)\n'
    '                    X_test.to_parquet(_splits_dir / "X_test.parquet", index=False)\n'
    '                    logger.info("GNN-updated splits re-persisted to %s/", _splits_dir)\n'
    "                else:\n"
    "                    logger.warning(\n"
    '                        "[GNN-TRACE] splits_dir does not exist (%s); "\n'
    '                        "re-persist SKIPPED",\n'
    "                        _splits_dir,\n"
    "                    )\n"
)

# (c) else branch for `if args.string_db:` -- em-dash-free anchor.
# We anchor on the BLANK LINE (335) + the next outer-scope statement
# (line 336: 'results = ensemble.evaluate(X_test, seq_te, y_test)').
# The em-dash on line 334 is now BEFORE the matched region, untouched.
GATE_BLOCK_OLD = (
    "\n" "\n" "        results = ensemble.evaluate(X_test, seq_te, y_test)\n"
)

GATE_BLOCK_NEW = (
    "\n"
    "        else:\n"
    "            logger.warning(\n"
    '                "[GNN-TRACE] gate-skipped: args.string_db is falsy (%r); "\n'
    '                "ENTIRE GNN BLOCK skipped",\n'
    '                getattr(args, "string_db", None),\n'
    "            )\n"
    "\n"
    "        results = ensemble.evaluate(X_test, seq_te, y_test)\n"
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--revert", action="store_true")
    args = p.parse_args()

    if not TARGET.exists():
        print(f"FAIL: {TARGET} not found (run from repo root)")
        return 1

    if args.revert:
        if not BACKUP.exists():
            print(f"FAIL: {BACKUP} not found")
            return 1
        shutil.copy(BACKUP, TARGET)
        print(f"OK: restored {TARGET} from {BACKUP}")
        return 0

    src = TARGET.read_text(encoding="utf-8")

    if "[GNN-TRACE]" in src:
        print(f"NO-OP: {TARGET} already contains [GNN-TRACE] markers")
        return 0

    anchors_to_check: list[str] = [LOOP_BLOCK_OLD, SPLITS_BLOCK_OLD, GATE_BLOCK_OLD]
    for ent in INSERTIONS:
        anchors_to_check.append(ent[0])

    missing: list[str] = []
    multi: list[tuple[str, int]] = []
    for anc in anchors_to_check:
        n = src.count(anc)
        if n == 0:
            missing.append(anc[:80])
        elif n > 1:
            multi.append((anc[:80], n))
    if missing:
        print("FAIL: anchors not found in file:")
        for a in missing:
            print(f"  {a!r}...")
        return 1
    if multi:
        print("FAIL: anchors found multiple times (ambiguous):")
        for a, n in multi:
            print(f"  ({n}x) {a!r}...")
        return 1
    print(f"OK: all {len(anchors_to_check)} anchors verified unique")

    # Apply: blocks first (larger, more sensitive), then anchor-after.
    new = src
    new = new.replace(LOOP_BLOCK_OLD, LOOP_BLOCK_NEW)
    new = new.replace(SPLITS_BLOCK_OLD, SPLITS_BLOCK_NEW)
    new = new.replace(GATE_BLOCK_OLD, GATE_BLOCK_NEW)
    for anchor, addition in INSERTIONS:
        new = new.replace(anchor, anchor + addition, 1)

    try:
        ast.parse(new)
    except SyntaxError as exc:
        print(f"FAIL: patched source has syntax error: {exc}")
        return 1
    print("OK: patched source is valid Python (ast.parse)")

    trace_count = new.count("[GNN-TRACE]")
    line_delta = new.count("\n") - src.count("\n")
    print(f"OK: [GNN-TRACE] string count = {trace_count} (expected >= 14)")
    print(f"OK: net new lines = {line_delta} (expected 45-65)")

    if trace_count < 14:
        print("FAIL: trace count below threshold")
        return 1

    if args.dry_run:
        print("DRY-RUN: no files written")
        return 0

    shutil.copy(TARGET, BACKUP)
    TARGET.write_text(new, encoding="utf-8")
    print(f"OK: wrote patched {TARGET} (backup at {BACKUP})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
