"""Complete Patch 6b: add meta_train param + materialize + pass-through.

Three str_replace edits anchored against live src/data/real_data_prep.py
verified at HEAD cb9a4cb on 2026-04-30.
"""

from __future__ import annotations

import ast
from pathlib import Path

PATH = Path("src/data/real_data_prep.py")

EDIT_1_OLD = (
    "        meta_val: pd.DataFrame,\n"
    "        meta_test: pd.DataFrame,\n"
    "    ) -> None:\n"
)
EDIT_1_NEW = (
    "        meta_val: pd.DataFrame,\n"
    "        meta_test: pd.DataFrame,\n"
    "        meta_train: pd.DataFrame | None = None,\n"
    "    ) -> None:\n"
)

EDIT_2_OLD = (
    "        meta_val = df.iloc[val_idx].reset_index(drop=True)\n"
    "        meta_test = df.iloc[test_idx].reset_index(drop=True)\n"
)
EDIT_2_NEW = (
    "        meta_val = df.iloc[val_idx].reset_index(drop=True)\n"
    "        meta_test = df.iloc[test_idx].reset_index(drop=True)\n"
    "        meta_train = df.iloc[train_idx].reset_index(drop=True)\n"
)

EDIT_3_OLD = (
    "        self._save_splits(\n"
    "            X_train, X_val, X_test, y_train, y_val, y_test, meta_val, meta_test\n"
    "        )\n"
)
EDIT_3_NEW = (
    "        self._save_splits(\n"
    "            X_train, X_val, X_test, y_train, y_val, y_test, meta_val, meta_test,\n"
    "            meta_train=meta_train,\n"
    "        )\n"
)

EDITS = [
    ("1: _save_splits signature add meta_train param", EDIT_1_OLD, EDIT_1_NEW),
    ("2: meta_train materialization in run()", EDIT_2_OLD, EDIT_2_NEW),
    ("3: _save_splits call passes meta_train", EDIT_3_OLD, EDIT_3_NEW),
]


def main() -> int:
    src = PATH.read_text(encoding="utf-8")

    print("=== Pre-flight: anchor uniqueness ===")
    fail = False
    for name, old, _new in EDITS:
        n = src.count(old)
        status = "OK  " if n == 1 else "FAIL"
        print(f"  {status} Edit {name}: count={n} (expected 1)")
        if n != 1:
            fail = True
    if fail:
        print("\nFAIL: refusing to edit. Inspect file and adjust.")
        return 1

    backup = PATH.with_suffix(PATH.suffix + ".bak-patch6b-complete")
    backup.write_text(src, encoding="utf-8")
    print(f"\nOK: backup at {backup}")

    for name, old, new in EDITS:
        src = src.replace(old, new, 1)
        print(f"  applied edit {name}")

    ast.parse(src)
    print("OK: AST validates")

    PATH.write_text(src, encoding="utf-8")
    print(f"OK: wrote {PATH}")

    print("\nNext: python scripts/probe_patch_6b.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
