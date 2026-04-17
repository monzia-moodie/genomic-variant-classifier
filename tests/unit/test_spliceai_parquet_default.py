"""
Regression test for SpliceAI silent-zero failure mode observed in Run 8.

If the default parquet exists on disk, the connector must produce at least
one non-zero score when it matches. If the connector falls back to all-zeros
despite the default file being present, this test fails and gates the
regression.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def test_spliceai_connector_uses_default_parquet_when_present(tmp_path, monkeypatch):
    """Build a 3-row fake parquet, point the module default at it, then
    instantiate SpliceAIConnector() with NO args. The connector must read
    the default, join it against matching variants, and return non-zero
    splice_ai_score values."""
    from src.data import spliceai as spliceai_mod
    from src.data.spliceai import SpliceAIConnector

    # Build a tiny fake parquet matching the production schema
    fake_parquet = tmp_path / "spliceai_index.parquet"
    df = pd.DataFrame(
        {
            "chrom": ["1", "17", "13"],
            "pos": [100, 200, 300],
            "ref": ["A", "G", "C"],
            "alt": ["T", "A", "T"],
            "ds_ag": [0.8, 0.0, 0.1],
            "ds_al": [0.0, 0.9, 0.0],
            "ds_dg": [0.0, 0.0, 0.0],
            "ds_dl": [0.0, 0.0, 0.0],
            "splice_ai_score": [0.8, 0.9, 0.1],
            "symbol": ["G1", "G2", "G3"],
        }
    )
    pq.write_table(pa.Table.from_pandas(df), fake_parquet)

    # Point the module constant at our fake file.
    # We also stash a fresh cache_dir under tmp_path to avoid contaminating
    # the real data/raw/cache/ with the fake lookup.
    monkeypatch.setattr(spliceai_mod, "DEFAULT_SPLICEAI_PATH", fake_parquet)

    variants = pd.DataFrame(
        {
            "chrom": ["1", "17", "13"],
            "pos": [100, 200, 300],
            "ref": ["A", "G", "C"],
            "alt": ["T", "A", "T"],
        }
    )

    from src.data.database_connectors import FetchConfig

    cfg = FetchConfig(cache_dir=tmp_path / "cache")
    connector = SpliceAIConnector(config=cfg)  # no vcf_path: uses DEFAULT_SPLICEAI_PATH

    out = connector.fetch(variant_df=variants)

    assert (
        "splice_ai_score" in out.columns
    ), "connector did not add a splice_ai_score column"
    scores = out["splice_ai_score"].to_numpy()
    assert (scores > 0).any(), (
        f"all splice_ai_score values are zero: {scores.tolist()}. "
        "This is the exact failure mode that zeroed SpliceAI in Run 8."
    )
