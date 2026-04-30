"""
apply_patch_6b.py — fixes the GNN gene_symbol KeyError discovered in
the 2026-04-30 run9_ready regen.

Two atomic edits, both anchor-verified, idempotent, AST-validated:

(1) src/data/real_data_prep.py
    - _save_splits signature: add meta_train: pd.DataFrame parameter
    - _save_splits body:      add meta_train.to_parquet(out / "meta_train.parquet", index=False)
    - run() body:             pass df.iloc[train_idx] as meta_train into _save_splits

(2) scripts/run_phase2_eval.py
    - Replace the X_train_raw/gnn_df reload block (lines ~241-254) with a
      version that merges gene_symbol from meta_train.parquet.

Backups written: <file>.bak-patch6b
"""

from __future__ import annotations

import argparse
import ast
import shutil
import sys
from pathlib import Path

REPO = Path(".")
PREP = REPO / "src" / "data" / "real_data_prep.py"
EVAL = REPO / "scripts" / "run_phase2_eval.py"


# ───────────────────────────────────────────────────────────────────────
# Edit 1a — _save_splits signature (real_data_prep.py)
# ───────────────────────────────────────────────────────────────────────
PREP_SIG_OLD = (
    "    def _save_splits(\n"
    "        self,\n"
    "        X_train: pd.DataFrame, X_test: pd.DataFrame,\n"
    "        y_train: pd.Series,    y_test: pd.Series,\n"
    "        meta_test: pd.DataFrame,\n"
    "    ) -> None:\n"
)

PREP_SIG_NEW = (
    "    def _save_splits(\n"
    "        self,\n"
    "        X_train: pd.DataFrame, X_test: pd.DataFrame,\n"
    "        y_train: pd.Series,    y_test: pd.Series,\n"
    "        meta_test: pd.DataFrame,\n"
    "        meta_train: pd.DataFrame | None = None,\n"
    "    ) -> None:\n"
)


# ───────────────────────────────────────────────────────────────────────
# Edit 1b — _save_splits body (real_data_prep.py)
# Insert meta_train.to_parquet right after meta_test.to_parquet
# ───────────────────────────────────────────────────────────────────────
PREP_BODY_OLD = (
    '        meta_test.to_parquet(out / "meta_test.parquet", index=False)\n'
    '        logger.info("Splits saved to %s/", out)\n'
)

PREP_BODY_NEW = (
    '        meta_test.to_parquet(out / "meta_test.parquet", index=False)\n'
    "        if meta_train is not None:\n"
    '            meta_train.to_parquet(out / "meta_train.parquet", index=False)\n'
    '        logger.info("Splits saved to %s/", out)\n'
)


# ───────────────────────────────────────────────────────────────────────
# Edit 1c — run() must pass meta_train to _save_splits
# Anchor: the `meta_test = df.iloc[test_idx].reset_index(drop=True)` line
# right before scaling. We need to materialize meta_train at the same time
# and pass it into _save_splits.
# ───────────────────────────────────────────────────────────────────────
PREP_RUN_OLD = (
    "        meta_test = df.iloc[test_idx].reset_index(drop=True)\n"
    "\n"
    "        if self.config.scale_features:\n"
    "            X_train, X_test = self._scale(X_train, X_test)\n"
    "\n"
    "        self._save_splits(X_train, X_test, y_train, y_test, meta_test)\n"
)

PREP_RUN_NEW = (
    "        meta_test  = df.iloc[test_idx].reset_index(drop=True)\n"
    "        meta_train = df.iloc[train_idx].reset_index(drop=True)\n"
    "\n"
    "        if self.config.scale_features:\n"
    "            X_train, X_test = self._scale(X_train, X_test)\n"
    "\n"
    "        self._save_splits(\n"
    "            X_train, X_test, y_train, y_test, meta_test,\n"
    "            meta_train=meta_train,\n"
    "        )\n"
)


# ───────────────────────────────────────────────────────────────────────
# Edit 2 — run_phase2_eval.py: replace gnn_df construction
# Anchor: lines 241-254 of the patched file (before [GNN-TRACE] insertions
# in this region). After the [GNN-TRACE] patch, this block was preserved
# verbatim, so the anchor still matches.
# ───────────────────────────────────────────────────────────────────────
EVAL_OLD = (
    "                # Build a raw training DataFrame for the GNN (needs gene_symbol + label)\n"
    "                gnn_df = meta_val.iloc[:0].copy()  # schema reference\n"
    "                # Attach gene_symbol from the ClinVar training rows\n"
    "                # (meta_test has index aligned to test; we need train indices)\n"
    "                # The simplest approach: re-build from the split parquet files if present\n"
    '                split_train = outdir / "splits" / "X_train.parquet"\n'
    "                if split_train.exists():\n"
    "                    X_train_raw = pd.read_parquet(split_train)\n"
    "                    gnn_df = X_train_raw.copy()\n"
    '                    gnn_df["acmg_label"] = y_train.values\n'
    "                else:\n"
    "                    # Fallback: use X_train feature matrix (no gene_symbol \u2192 smaller GNN)\n"
    "                    gnn_df = X_train.copy()\n"
    '                    gnn_df["acmg_label"] = y_train.values\n'
)

EVAL_NEW = (
    "                # Build a raw training DataFrame for the GNN (needs gene_symbol + label).\n"
    "                # Patch 6b (2026-04-30): source gene_symbol from meta_train.parquet,\n"
    "                # which DataPrepPipeline now persists alongside the feature matrix.\n"
    "                # Previous implementation reloaded X_train.parquet (a 78-col numeric\n"
    "                # matrix with NO gene_symbol) and crashed inside build_pyg_dataset.\n"
    '                _meta_train_path = outdir / "splits" / "meta_train.parquet"\n'
    "                if _meta_train_path.exists():\n"
    "                    _meta_train = pd.read_parquet(_meta_train_path)\n"
    "                    gnn_df = X_train.copy().reset_index(drop=True)\n"
    '                    gnn_df["gene_symbol"] = (\n'
    '                        _meta_train["gene_symbol"].fillna("").reset_index(drop=True)\n'
    "                    )\n"
    '                    gnn_df["acmg_label"] = y_train.values\n'
    "                    logger.info(\n"
    '                        "[GNN-TRACE] meta_train.parquet sourced gene_symbol "\n'
    '                        "(unique_genes=%d, missing=%d)",\n'
    '                        gnn_df["gene_symbol"].nunique(),\n'
    '                        int((gnn_df["gene_symbol"] == "").sum()),\n'
    "                    )\n"
    "                else:\n"
    "                    logger.warning(\n"
    '                        "[GNN-TRACE] meta_train.parquet missing at %s; "\n'
    '                        "GNN training cannot proceed (no gene_symbol). "\n'
    '                        "Re-run DataPrepPipeline to regenerate splits.",\n'
    "                        _meta_train_path,\n"
    "                    )\n"
    "                    raise FileNotFoundError(_meta_train_path)\n"
)


def apply_edit(path: Path, old: str, new: str, label: str) -> bool:
    """Apply one anchor-replace edit. Returns True if changed."""
    src = path.read_text(encoding="utf-8")
    if new in src and old not in src:
        print(f"  NO-OP: {label} already applied")
        return False
    n = src.count(old)
    if n != 1:
        print(f"  FAIL: {label} anchor count={n} (expected 1)")
        return False
    new_src = src.replace(old, new, 1)
    try:
        ast.parse(new_src)
    except SyntaxError as e:
        print(f"  FAIL: {label} produces syntax error: {e}")
        return False
    path.write_text(new_src, encoding="utf-8")
    print(f"  OK: {label} applied")
    return True


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--revert", action="store_true")
    args = p.parse_args()

    if not PREP.exists() or not EVAL.exists():
        print(f"FAIL: missing {PREP} or {EVAL} (run from repo root)")
        return 1

    bak_prep = PREP.with_suffix(".py.bak-patch6b")
    bak_eval = EVAL.with_suffix(".py.bak-patch6b")

    if args.revert:
        ok = True
        for orig, bak in [(PREP, bak_prep), (EVAL, bak_eval)]:
            if bak.exists():
                shutil.copy(bak, orig)
                print(f"OK: restored {orig} from {bak}")
            else:
                print(f"WARN: {bak} not found")
                ok = False
        return 0 if ok else 1

    # Backups (only on real run)
    if not args.dry_run:
        shutil.copy(PREP, bak_prep)
        shutil.copy(EVAL, bak_eval)
        print(f"OK: backups at {bak_prep} and {bak_eval}")

    print("\n=== Edit 1: src/data/real_data_prep.py ===")
    if args.dry_run:
        # Validate anchors exist exactly once
        s = PREP.read_text(encoding="utf-8")
        for label, anchor in [
            ("1a sig", PREP_SIG_OLD),
            ("1b body", PREP_BODY_OLD),
            ("1c run", PREP_RUN_OLD),
        ]:
            n = s.count(anchor)
            print(f"  anchor 1{label}: count={n} (expected 1)")
    else:
        apply_edit(PREP, PREP_SIG_OLD, PREP_SIG_NEW, "1a signature")
        apply_edit(PREP, PREP_BODY_OLD, PREP_BODY_NEW, "1b body")
        apply_edit(PREP, PREP_RUN_OLD, PREP_RUN_NEW, "1c run")

    print("\n=== Edit 2: scripts/run_phase2_eval.py ===")
    if args.dry_run:
        s = EVAL.read_text(encoding="utf-8")
        n = s.count(EVAL_OLD)
        print(f"  anchor 2 gnn_df build: count={n} (expected 1)")
    else:
        apply_edit(EVAL, EVAL_OLD, EVAL_NEW, "2 gnn_df build")

    print("\n=== Validation ===")
    for f in [PREP, EVAL]:
        try:
            ast.parse(f.read_text(encoding="utf-8"))
            print(f"  {f}: AST OK")
        except SyntaxError as e:
            print(f"  {f}: AST FAIL — {e}")
            return 1

    if args.dry_run:
        print("\nDRY-RUN: no files written")
    else:
        print("\nNext: re-run synthetic probe to verify Patch 6b end-to-end")
    return 0


if __name__ == "__main__":
    sys.exit(main())
