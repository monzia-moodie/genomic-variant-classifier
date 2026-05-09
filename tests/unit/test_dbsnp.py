"""
tests/unit/test_dbsnp.py
=========================
Unit tests for DbSNPConnector and its wiring into engineer_features.

Coverage:
  1.  Stub mode (no parquet_path) — all AF = 0.0
  2.  Empty DataFrame — empty output with column present
  3.  _annotate with a small lookup DataFrame
  4.  Missing file on disk → defaults returned
  5.  fetch() round-trip via parquet file
  6.  TABULAR_FEATURES membership — dbsnp_af IS present
  7.  engineer_features default (missing column → 0.0)
  8.  engineer_features real value passes through
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from genomic_variant_classifier.data.dbsnp import DbSNPConnector
from genomic_variant_classifier.models.variant_ensemble import TABULAR_FEATURES, engineer_features

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_variant_df(**overrides) -> pd.DataFrame:
    base = dict(
        variant_id=["clinvar:17:43071077:G:T"],
        chrom=["17"],
        pos=[43071077],
        ref=["G"],
        alt=["T"],
        gene_symbol=["TP53"],
        consequence=["missense_variant"],
        allele_freq=[0.0],
    )
    base.update({k: [v] for k, v in overrides.items()})
    return pd.DataFrame(base)


def _engineer_df(**overrides) -> pd.DataFrame:
    base = dict(
        gene_symbol=["TP53"],
        consequence=["missense_variant"],
        allele_freq=[0.001],
        ref=["G"],
        alt=["T"],
    )
    base.update({k: [v] for k, v in overrides.items()})
    return pd.DataFrame(base)


def _make_lookup(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. Stub mode
# ---------------------------------------------------------------------------

def test_stub_mode_no_path_returns_zero():
    """No parquet_path → dbsnp_af = 0.0 for all variants."""
    connector = DbSNPConnector(parquet_path=None)
    df = _minimal_variant_df()
    result = connector.annotate_dataframe(df)
    assert "dbsnp_af" in result.columns
    assert result["dbsnp_af"].iloc[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 2. Empty DataFrame
# ---------------------------------------------------------------------------

def test_empty_dataframe_returns_empty_with_column():
    connector = DbSNPConnector()
    empty = pd.DataFrame(columns=["chrom", "pos", "ref", "alt"])
    result = connector.annotate_dataframe(empty)
    assert "dbsnp_af" in result.columns
    assert len(result) == 0


# ---------------------------------------------------------------------------
# 3. _annotate with known lookup
# ---------------------------------------------------------------------------

def test_annotate_matching_variant():
    lookup = _make_lookup([
        {"variant_id": "17:43071077:G:T", "allele_freq": 0.0023},
        {"variant_id": "13:32936732:A:C", "allele_freq": 0.0007},
    ])
    connector = DbSNPConnector()
    df = _minimal_variant_df(chrom="17", pos=43071077, ref="G", alt="T")
    result = connector._annotate(df, lookup)
    assert result["dbsnp_af"].iloc[0] == pytest.approx(0.0023)


def test_annotate_no_match_returns_zero():
    lookup = _make_lookup([
        {"variant_id": "1:111111:A:C", "allele_freq": 0.05},
    ])
    connector = DbSNPConnector()
    df = _minimal_variant_df()   # 17:43071077:G:T — no match
    result = connector._annotate(df, lookup)
    assert result["dbsnp_af"].iloc[0] == pytest.approx(0.0)


def test_annotate_chr_prefix_stripped():
    """Input variant with 'chr17' chrom should still match lookup key '17:...'."""
    lookup = _make_lookup([
        {"variant_id": "17:43071077:G:T", "allele_freq": 0.0015},
    ])
    connector = DbSNPConnector()
    df = _minimal_variant_df(chrom="chr17")
    result = connector._annotate(df, lookup)
    assert result["dbsnp_af"].iloc[0] == pytest.approx(0.0015)


# ---------------------------------------------------------------------------
# 4. Missing file on disk
# ---------------------------------------------------------------------------

def test_missing_file_returns_zero(tmp_path):
    path = tmp_path / "nonexistent.parquet"
    connector = DbSNPConnector(parquet_path=path)
    df = _minimal_variant_df()
    with patch.object(connector, "_load_cache", return_value=None):
        result = connector.annotate_dataframe(df)
    assert result["dbsnp_af"].iloc[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 5. fetch() round-trip via parquet
# ---------------------------------------------------------------------------

def test_fetch_round_trip(tmp_path):
    lookup = pd.DataFrame({
        "variant_id": ["17:43071077:G:T", "13:32936732:A:C"],
        "allele_freq": [0.0023, 0.0007],
    })
    parquet_path = tmp_path / "dbsnp.parquet"
    lookup.to_parquet(parquet_path, index=False)

    connector = DbSNPConnector(parquet_path=parquet_path)
    df = pd.DataFrame({
        "chrom": ["17", "13", "1"],
        "pos":   [43071077, 32936732, 1],
        "ref":   ["G", "A", "C"],
        "alt":   ["T", "C", "G"],
        "gene_symbol": ["TP53", "BRCA2", "UNKNOWN"],
        "allele_freq": [0.0, 0.0, 0.5],
    })

    result = connector.fetch(variant_df=df)
    assert result.loc[0, "dbsnp_af"] == pytest.approx(0.0023)
    assert result.loc[1, "dbsnp_af"] == pytest.approx(0.0007)
    assert result.loc[2, "dbsnp_af"] == pytest.approx(0.0)   # no match


# ---------------------------------------------------------------------------
# 6. TABULAR_FEATURES membership
# ---------------------------------------------------------------------------

def test_dbsnp_af_in_tabular_features():
    assert "dbsnp_af" in TABULAR_FEATURES


# ---------------------------------------------------------------------------
# 7-8. engineer_features wiring
# ---------------------------------------------------------------------------

def test_engineer_features_dbsnp_default_zero_when_absent():
    df = _engineer_df()
    assert "dbsnp_af" not in df.columns
    feats = engineer_features(df)
    assert feats.loc[0, "dbsnp_af"] == pytest.approx(0.0)
    assert not feats["dbsnp_af"].isnull().any()


def test_engineer_features_dbsnp_real_value_passes_through():
    df = _engineer_df(dbsnp_af=0.0035)
    feats = engineer_features(df)
    assert feats.loc[0, "dbsnp_af"] == pytest.approx(0.0035)
