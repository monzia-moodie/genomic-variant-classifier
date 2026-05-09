"""
tests/unit/test_alphamissense.py
=================================
Unit tests for AlphaMissenseConnector and its wiring into engineer_features.

Coverage:
  1.  Stub mode (no tsv_path)          — all scores = 0.5
  2.  Explicit tsv_path=None           — all scores = 0.5
  3.  Missing file on disk             — warning logged, all scores = 0.5
  4.  Empty input DataFrame            — empty output with column present
  5.  _parse_tsv: real gzip TSV        — correct lookup_key + score
  6.  _parse_tsv: strips chr prefix    — lookup_key uses bare chrom number
  7.  _parse_tsv: deduplication        — highest score wins per locus
  8.  _annotate: matching variants     — scores joined correctly
  9.  _annotate: no matches            — fill with AM_DEFAULT_SCORE
 10.  _annotate: chr-prefixed chrom    — lookup key normalisation
 11.  fetch() round-trip               — end-to-end with tmp TSV
 12.  Score clipping [0, 1]            — out-of-range values clamped
 13.  TABULAR_FEATURES membership      — alphamissense_score present
 14.  engineer_features default 0.5   — absent column → default 0.5
 15.  engineer_features real score    — value passes through unchanged
 16.  Threshold constants             — pathogenic > benign, sane range
"""

from __future__ import annotations

import gzip
import io
import logging
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from genomic_variant_classifier.data.alphamissense import (
    AM_BENIGN_THRESHOLD,
    AM_DEFAULT_SCORE,
    AM_PATHOGENIC_THRESHOLD,
    AlphaMissenseConnector,
)
from genomic_variant_classifier.models.variant_ensemble import TABULAR_FEATURES, engineer_features

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TSV_HEADER = (
    "# AlphaMissense hg38\n"
    "# CHROM\tPOS\tREF\tALT\tgenome\tuniprot_id\ttranscript_id"
    "\tprotein_variant\tam_pathogenicity\tam_class\n"
)

_TSV_ROWS = [
    "17\t43071077\tG\tT\thg38\tP38398\tENST00000357654\tD1733Y\t0.92\tlikely_pathogenic\n",
    "13\t32936732\tA\tC\thg38\tP51587\tENST00000380152\tN1239T\t0.12\tlikely_benign\n",
    "1\t925952\tG\tA\thg38\tQ9H251\tENST00000370321\tR123Q\t0.55\tambiguous\n",
]


def _make_tsv_gz(tmp_path: Path, rows: list[str] | None = None) -> Path:
    """Write a minimal AlphaMissense-format .tsv.gz to tmp_path."""
    rows = rows if rows is not None else _TSV_ROWS
    content = _TSV_HEADER + "".join(rows)
    path = tmp_path / "AlphaMissense_hg38.tsv.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(content)
    return path


def _minimal_variant_df(**overrides) -> pd.DataFrame:
    """One-row canonical-schema DataFrame for annotation tests."""
    base = dict(
        variant_id=["clinvar:17:43071077:G:T"],
        chrom=["17"],
        pos=[43071077],
        ref=["G"],
        alt=["T"],
        gene_symbol=["BRCA1"],
        consequence=["missense_variant"],
        allele_freq=[0.0001],
    )
    base.update({k: [v] for k, v in overrides.items()})
    return pd.DataFrame(base)


def _engineer_df(**overrides) -> pd.DataFrame:
    """Minimal DataFrame for engineer_features (no chrom/pos/ref/alt needed)."""
    base = dict(
        gene_symbol=["BRCA1"],
        consequence=["missense_variant"],
        allele_freq=[0.001],
        ref=["G"],
        alt=["T"],
    )
    base.update({k: [v] for k, v in overrides.items()})
    return pd.DataFrame(base)


# ---------------------------------------------------------------------------
# 1-2. Stub / no-path mode
# ---------------------------------------------------------------------------

def test_stub_mode_no_path_returns_default():
    """No tsv_path → every row gets AM_DEFAULT_SCORE (0.5)."""
    connector = AlphaMissenseConnector(tsv_path=None)
    df = _minimal_variant_df()
    result = connector.fetch(variant_df=df)
    assert "alphamissense_score" in result.columns
    assert result["alphamissense_score"].iloc[0] == pytest.approx(AM_DEFAULT_SCORE)


def test_stub_mode_explicit_none():
    """Explicit tsv_path=None is the same as omitting the argument."""
    c1 = AlphaMissenseConnector()
    c2 = AlphaMissenseConnector(tsv_path=None)
    df = _minimal_variant_df()
    assert (
        c1.fetch(variant_df=df)["alphamissense_score"].iloc[0]
        == c2.fetch(variant_df=df)["alphamissense_score"].iloc[0]
        == pytest.approx(AM_DEFAULT_SCORE)
    )


# ---------------------------------------------------------------------------
# 3. Missing file on disk
# ---------------------------------------------------------------------------

def test_missing_file_logs_warning_and_returns_default(tmp_path, caplog):
    path = tmp_path / "nonexistent.tsv.gz"
    connector = AlphaMissenseConnector(tsv_path=path)
    df = _minimal_variant_df()
    # Patch cache so a cached lookup from another test doesn't mask the missing-file path
    with patch.object(connector, "_load_cache", return_value=None):
        with caplog.at_level(logging.WARNING, logger="genomic_variant_classifier.data.alphamissense"):
            result = connector.fetch(variant_df=df)
    assert result["alphamissense_score"].iloc[0] == pytest.approx(AM_DEFAULT_SCORE)
    assert any("not found" in r.getMessage().lower() or "tsv" in r.getMessage().lower()
               for r in caplog.records)


# ---------------------------------------------------------------------------
# 4. Empty input DataFrame
# ---------------------------------------------------------------------------

def test_empty_dataframe_returns_empty_with_column():
    connector = AlphaMissenseConnector()
    empty = pd.DataFrame(columns=["chrom", "pos", "ref", "alt", "gene_symbol"])
    result = connector.fetch(variant_df=empty)
    assert "alphamissense_score" in result.columns
    assert len(result) == 0


# ---------------------------------------------------------------------------
# 5-7. _parse_tsv internals
# ---------------------------------------------------------------------------

def test_parse_tsv_produces_lookup_key_and_score(tmp_path):
    path = _make_tsv_gz(tmp_path)
    connector = AlphaMissenseConnector(tsv_path=path)
    lookup = connector._parse_tsv(path)
    assert "lookup_key" in lookup.columns
    assert "alphamissense_score" in lookup.columns
    assert len(lookup) == 3


def test_parse_tsv_strips_chr_prefix(tmp_path):
    """Even if the source uses 'chr17', lookup_key must use bare '17'."""
    rows = [
        "chr17\t43071077\tG\tT\thg38\tP38398\tENST1\tD1733Y\t0.91\tlikely_pathogenic\n"
    ]
    path = _make_tsv_gz(tmp_path, rows)
    connector = AlphaMissenseConnector(tsv_path=path)
    lookup = connector._parse_tsv(path)
    key = lookup["lookup_key"].iloc[0]
    assert key.startswith("17:"), f"Expected bare chrom, got: {key}"


def test_parse_tsv_deduplication_keeps_highest_score(tmp_path):
    """Duplicate locus → highest alphamissense_score wins."""
    rows = [
        "17\t43071077\tG\tT\thg38\tP1\tT1\tD1Y\t0.30\tlikely_benign\n",
        "17\t43071077\tG\tT\thg38\tP2\tT2\tD1Y\t0.95\tlikely_pathogenic\n",
    ]
    path = _make_tsv_gz(tmp_path, rows)
    connector = AlphaMissenseConnector(tsv_path=path)
    lookup = connector._parse_tsv(path)
    assert len(lookup) == 1
    assert lookup["alphamissense_score"].iloc[0] == pytest.approx(0.95, abs=1e-3)


# ---------------------------------------------------------------------------
# 8-10. _annotate
# ---------------------------------------------------------------------------

def _make_lookup(rows: list[tuple]) -> pd.DataFrame:
    """Build a lookup DataFrame from (lookup_key, score) tuples."""
    return pd.DataFrame(rows, columns=["lookup_key", "alphamissense_score"])


def test_annotate_matching_variant():
    lookup = _make_lookup([("17:43071077:G:T", 0.92)])
    connector = AlphaMissenseConnector()
    df = _minimal_variant_df()
    result = connector._annotate(df, lookup)
    assert result["alphamissense_score"].iloc[0] == pytest.approx(0.92, abs=1e-3)


def test_annotate_no_match_uses_default():
    lookup = _make_lookup([("1:111111:A:C", 0.80)])
    connector = AlphaMissenseConnector()
    df = _minimal_variant_df()           # variant is 17:43071077:G:T
    result = connector._annotate(df, lookup)
    assert result["alphamissense_score"].iloc[0] == pytest.approx(AM_DEFAULT_SCORE)


def test_annotate_chr_prefixed_chrom_normalised():
    """Input variant with 'chr17' chrom should still match lookup key '17:...'."""
    lookup = _make_lookup([("17:43071077:G:T", 0.88)])
    connector = AlphaMissenseConnector()
    df = _minimal_variant_df(chrom="chr17")
    result = connector._annotate(df, lookup)
    assert result["alphamissense_score"].iloc[0] == pytest.approx(0.88, abs=1e-3)


# ---------------------------------------------------------------------------
# 11. fetch() end-to-end round-trip
# ---------------------------------------------------------------------------

def test_fetch_round_trip(tmp_path):
    from genomic_variant_classifier.data.database_connectors import FetchConfig
    path = _make_tsv_gz(tmp_path)
    # Isolate cache to tmp_path so a shared real-data cache cannot pollute the test
    connector = AlphaMissenseConnector(tsv_path=path, config=FetchConfig(cache_dir=tmp_path))
    df = pd.DataFrame({
        "chrom": ["17", "13", "99"],
        "pos":   [43071077, 32936732, 1],
        "ref":   ["G", "A", "C"],
        "alt":   ["T", "C", "G"],
        "gene_symbol": ["BRCA1", "BRCA2", "UNKNOWN"],
        "consequence": ["missense_variant"] * 3,
        "allele_freq": [0.0001, 0.0002, 0.5],
    })
    result = connector.fetch(variant_df=df)
    assert result.loc[0, "alphamissense_score"] == pytest.approx(0.92, abs=1e-3)
    assert result.loc[1, "alphamissense_score"] == pytest.approx(0.12, abs=1e-3)
    assert result.loc[2, "alphamissense_score"] == pytest.approx(AM_DEFAULT_SCORE)  # no match


# ---------------------------------------------------------------------------
# 12. Score clipping [0, 1]
# ---------------------------------------------------------------------------

def test_score_clipped_to_0_1():
    """Out-of-range scores in the lookup must be clipped to [0, 1]."""
    lookup = _make_lookup([("17:43071077:G:T", 1.5), ("13:32936732:A:C", -0.2)])
    connector = AlphaMissenseConnector()
    df = pd.DataFrame({
        "chrom": ["17", "13"],
        "pos":   [43071077, 32936732],
        "ref":   ["G", "A"],
        "alt":   ["T", "C"],
        "gene_symbol": ["BRCA1", "BRCA2"],
        "consequence": ["missense_variant"] * 2,
        "allele_freq": [0.0001, 0.0002],
    })
    result = connector._annotate(df, lookup)
    assert result["alphamissense_score"].iloc[0] <= 1.0
    assert result["alphamissense_score"].iloc[1] >= 0.0


# ---------------------------------------------------------------------------
# 13. TABULAR_FEATURES membership
# ---------------------------------------------------------------------------

def test_alphamissense_score_in_tabular_features():
    assert "alphamissense_score" in TABULAR_FEATURES


def test_alphamissense_score_not_in_phase_2():
    from genomic_variant_classifier.models.variant_ensemble import PHASE_2_FEATURES
    assert "alphamissense_score" not in PHASE_2_FEATURES


# ---------------------------------------------------------------------------
# 14-15. engineer_features wiring
# ---------------------------------------------------------------------------

def test_engineer_features_default_when_absent():
    """Missing alphamissense_score column → default 0.5 (ambiguous)."""
    df = _engineer_df()
    assert "alphamissense_score" not in df.columns
    feats = engineer_features(df)
    assert feats.loc[0, "alphamissense_score"] == pytest.approx(AM_DEFAULT_SCORE)
    assert not feats["alphamissense_score"].isnull().any()


def test_engineer_features_real_score_passes_through():
    """Explicit score in input should survive engineer_features unchanged."""
    df = _engineer_df(alphamissense_score=0.92)
    feats = engineer_features(df)
    assert feats.loc[0, "alphamissense_score"] == pytest.approx(0.92)


# ---------------------------------------------------------------------------
# 16. Threshold constants sanity
# ---------------------------------------------------------------------------

def test_threshold_constants_sane():
    assert 0.0 < AM_BENIGN_THRESHOLD < AM_PATHOGENIC_THRESHOLD < 1.0
    assert AM_DEFAULT_SCORE == pytest.approx(0.5)
