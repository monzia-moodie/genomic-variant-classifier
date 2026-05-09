"""
tests/unit/test_hgmd.py
========================
Unit tests for HGMDConnector and its wiring into engineer_features.

Coverage:
  1.  Stub mode (no hgmd_path, expected default for users without license)
  2.  Empty DataFrame — empty output with columns present
  3.  _parse_hgmd with a small tab-separated temp file
  4.  _annotate with known lookup
  5.  fetch() round-trip
  6.  TABULAR_FEATURES membership — hgmd_is_disease_mutation, hgmd_n_reports
  7.  engineer_features default (missing columns → 0)
  8.  engineer_features real values pass through
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from genomic_variant_classifier.data.hgmd import HGMDConnector, DISEASE_MUTATION_CLASSES
from genomic_variant_classifier.models.variant_ensemble import TABULAR_FEATURES, engineer_features

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TSV_CONTENT = """\
CHROM\tPOS\tREF\tALT\tCLASS
17\t43071077\tG\tT\tDM
17\t43071077\tG\tT\tDM
13\t32936732\tA\tC\tDM?
1\t925952\tG\tA\tDP
7\t117548628\tC\tT\tFP
"""


def _write_tsv(tmp_path: Path, content: str | None = None) -> Path:
    content = content if content is not None else _TSV_CONTENT
    path = tmp_path / "hgmd.txt"
    path.write_text(content, encoding="utf-8")
    return path


def _minimal_variant_df(**overrides) -> pd.DataFrame:
    base = dict(
        variant_id=["clinvar:17:43071077:G:T"],
        chrom=["17"],
        pos=[43071077],
        ref=["G"],
        alt=["T"],
        gene_symbol=["BRCA1"],
        consequence=["missense_variant"],
        allele_freq=[0.0],
    )
    base.update({k: [v] for k, v in overrides.items()})
    return pd.DataFrame(base)


def _engineer_df(**overrides) -> pd.DataFrame:
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
# 1. Stub mode (no hgmd_path)
# ---------------------------------------------------------------------------

def test_stub_mode_no_path_returns_defaults():
    """No hgmd_path → hgmd_is_disease_mutation=0, hgmd_n_reports=0."""
    connector = HGMDConnector(hgmd_path=None)
    df = _minimal_variant_df()
    result = connector.annotate_dataframe(df)
    assert "hgmd_is_disease_mutation" in result.columns
    assert "hgmd_n_reports" in result.columns
    assert result["hgmd_is_disease_mutation"].iloc[0] == 0
    assert result["hgmd_n_reports"].iloc[0] == 0


# ---------------------------------------------------------------------------
# 2. Empty DataFrame
# ---------------------------------------------------------------------------

def test_empty_dataframe_returns_empty_with_columns():
    connector = HGMDConnector()
    empty = pd.DataFrame(columns=["chrom", "pos", "ref", "alt"])
    result = connector.annotate_dataframe(empty)
    assert "hgmd_is_disease_mutation" in result.columns
    assert "hgmd_n_reports" in result.columns
    assert len(result) == 0


# ---------------------------------------------------------------------------
# 3. _parse_hgmd with temp file
# ---------------------------------------------------------------------------

def test_parse_hgmd_tab_file(tmp_path):
    path = _write_tsv(tmp_path)
    connector = HGMDConnector(hgmd_path=path)
    lookup = connector._parse_hgmd(path)

    assert "lookup_key" in lookup.columns
    assert "hgmd_is_disease_mutation" in lookup.columns
    assert "hgmd_n_reports" in lookup.columns

    # 17:43071077:G:T appears twice (both DM) → n_reports=2, is_dm=1
    row_17 = lookup[lookup["lookup_key"] == "17:43071077:G:T"]
    assert len(row_17) == 1
    assert row_17["hgmd_is_disease_mutation"].iloc[0] == 1
    assert row_17["hgmd_n_reports"].iloc[0] == 2

    # 13:32936732:A:C → DM? → is_dm=1
    row_13 = lookup[lookup["lookup_key"] == "13:32936732:A:C"]
    assert row_13["hgmd_is_disease_mutation"].iloc[0] == 1

    # 1:925952:G:A → DP → is_dm=0
    row_1 = lookup[lookup["lookup_key"] == "1:925952:G:A"]
    assert row_1["hgmd_is_disease_mutation"].iloc[0] == 0


# ---------------------------------------------------------------------------
# 4. _annotate with known lookup
# ---------------------------------------------------------------------------

def test_annotate_matching_variant():
    lookup = pd.DataFrame({
        "lookup_key":              ["17:43071077:G:T"],
        "hgmd_is_disease_mutation": [1],
        "hgmd_n_reports":           [3],
    })
    connector = HGMDConnector()
    df = _minimal_variant_df()
    result = connector._annotate(df, lookup)
    assert result["hgmd_is_disease_mutation"].iloc[0] == 1
    assert result["hgmd_n_reports"].iloc[0] == 3


def test_annotate_no_match_returns_zero():
    lookup = pd.DataFrame({
        "lookup_key":              ["1:111111:A:C"],
        "hgmd_is_disease_mutation": [1],
        "hgmd_n_reports":           [1],
    })
    connector = HGMDConnector()
    df = _minimal_variant_df()   # 17:43071077:G:T — no match
    result = connector._annotate(df, lookup)
    assert result["hgmd_is_disease_mutation"].iloc[0] == 0
    assert result["hgmd_n_reports"].iloc[0] == 0


# ---------------------------------------------------------------------------
# 5. fetch() round-trip
# ---------------------------------------------------------------------------

def test_fetch_round_trip(tmp_path):
    path = _write_tsv(tmp_path)
    connector = HGMDConnector(hgmd_path=path)
    df = _minimal_variant_df(chrom="17", pos=43071077, ref="G", alt="T")
    result = connector.fetch(variant_df=df)
    assert result["hgmd_is_disease_mutation"].iloc[0] == 1
    assert result["hgmd_n_reports"].iloc[0] == 2


# ---------------------------------------------------------------------------
# 6. TABULAR_FEATURES membership
# ---------------------------------------------------------------------------

def test_hgmd_features_in_tabular_features():
    assert "hgmd_is_disease_mutation" in TABULAR_FEATURES
    assert "hgmd_n_reports" in TABULAR_FEATURES


# ---------------------------------------------------------------------------
# 7-8. engineer_features wiring
# ---------------------------------------------------------------------------

def test_engineer_features_hgmd_default_zero_when_absent():
    df = _engineer_df()
    assert "hgmd_is_disease_mutation" not in df.columns
    assert "hgmd_n_reports" not in df.columns
    feats = engineer_features(df)
    assert feats.loc[0, "hgmd_is_disease_mutation"] == 0
    assert feats.loc[0, "hgmd_n_reports"] == 0
    assert not feats["hgmd_is_disease_mutation"].isnull().any()
    assert not feats["hgmd_n_reports"].isnull().any()


def test_engineer_features_hgmd_real_values_pass_through():
    df = _engineer_df(hgmd_is_disease_mutation=1, hgmd_n_reports=5)
    feats = engineer_features(df)
    assert feats.loc[0, "hgmd_is_disease_mutation"] == 1
    assert feats.loc[0, "hgmd_n_reports"] == 5


# ---------------------------------------------------------------------------
# DISEASE_MUTATION_CLASSES constant sanity
# ---------------------------------------------------------------------------

def test_disease_mutation_classes_correct():
    assert "DM" in DISEASE_MUTATION_CLASSES
    assert "DM?" in DISEASE_MUTATION_CLASSES
    # DP, DFP, FP, R are NOT disease mutations
    assert "DP"  not in DISEASE_MUTATION_CLASSES
    assert "FP"  not in DISEASE_MUTATION_CLASSES
