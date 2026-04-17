"""
Verify SpliceAI parquet index is production-ready.
Runs before wiring SPLICEAI_PATH in .env / AnnotationConfig.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

EXPECTED_COLUMNS = {
    "chrom",
    "pos",
    "ref",
    "alt",
    "ds_ag",
    "ds_al",
    "ds_dg",
    "ds_dl",
    "splice_ai_score",
    "symbol",
}
MIN_ROWS = 10_000_000  # Expect ~18.5M at >=0.1 threshold


def verify(path: Path) -> bool:
    if not path.exists():
        print(f"FAIL: file does not exist: {path}")
        return False

    size_gb = path.stat().st_size / (1024**3)
    print(f"File: {path}")
    print(f"Size: {size_gb:.2f} GB")

    try:
        df = pd.read_parquet(
            path, columns=["chrom", "pos", "ref", "alt", "splice_ai_score"]
        )
    except Exception as exc:
        print(f"FAIL: cannot read parquet: {exc}")
        return False

    n = len(df)
    print(f"Rows: {n:,}")
    if n < MIN_ROWS:
        print(f"FAIL: expected >= {MIN_ROWS:,} rows")
        return False

    full_cols = set(pq.read_schema(path).names)
    missing_cols = EXPECTED_COLUMNS - full_cols
    if missing_cols:
        print(f"FAIL: missing columns: {missing_cols}")
        return False
    print(f"Columns OK: {sorted(full_cols & EXPECTED_COLUMNS)}")

    null_frac = df["splice_ai_score"].isna().mean()
    print(f"splice_ai_score null fraction: {null_frac:.4f}")
    if null_frac > 0.01:
        print(f"FAIL: too many null scores ({null_frac:.2%})")
        return False

    min_score = float(df["splice_ai_score"].min())
    max_score = float(df["splice_ai_score"].max())
    print(f"splice_ai_score range: [{min_score:.4f}, {max_score:.4f}]")
    if min_score < 0.0 or max_score > 1.0:
        print(f"FAIL: score out of [0, 1] range")
        return False
    if min_score < 0.095:
        print(f"WARN: min score {min_score} is below the 0.1 threshold claim")

    chroms = sorted(df["chrom"].astype(str).unique().tolist())
    expected_chroms = {str(i) for i in range(1, 23)} | {"X", "Y", "MT"}
    missing_chroms = expected_chroms - set(chroms)
    if missing_chroms:
        print(f"WARN: missing chromosomes: {missing_chroms}")

    print("PASS: SpliceAI index file is production-ready")
    return True


if __name__ == "__main__":
    target = Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else "data/external/spliceai/spliceai_index.parquet"
    )
    ok = verify(target)
    sys.exit(0 if ok else 1)
