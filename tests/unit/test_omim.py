"""
tests/unit/test_omim.py
========================
Unit tests for OMIMConnector and its wiring into engineer_features.

Coverage:
  1.  Stub mode (no mim2gene_path) — all values = defaults
  2.  Empty DataFrame — empty output with columns present
  3.  _parse_mim2gene with a small in-memory temp file
  4.  annotate_dataframe join — known gene counts
  5.  fetch() round-trip
  6.  TABULAR_FEATURES membership — omim_n_diseases and omim_is_autosomal_dominant
  7.  engineer_features default (missing column → 0)
  8.  engineer_features real value passes through
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from genomic_variant_classifier.data.omim import OMIMConnector
from genomic_variant_classifier.models.variant_ensemble import TABULAR_FEATURES, engineer_features

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MIM2GENE_CONTENT = """\
# Copyright OMIM 2024
# MIM_number\tMIM_type\tEntrez_ID\tHGNC_symbol\tEnsembl_ID
100050\tphenotype\t-\t-\t-
100070\tpredominantly phenotypes\t-\t-\t-
100100\tgene\t348\tAMBP\tENSG00000106511
123500\tphenotype\t-\t-\t-
123501\tphenotype\t7157\tTP53\tENSG00000141510
123502\tphenotype\t7157\tTP53\tENSG00000141510
672\tphenotype\t672\tBRCA1\tENSG00000012048
675\tphenotype\t675\tBRCA2\tENSG00000139618
"""


def _write_mim2gene(tmp_path: Path, content: str | None = None) -> Path:
    content = content if content is not None else _MIM2GENE_CONTENT
    path = tmp_path / "mim2gene.txt"
    path.write_text(content, encoding="utf-8")
    return path


def _minimal_variant_df(**overrides) -> pd.DataFrame:
    base = dict(
        variant_id=["clinvar:17:43071077:G:T"],
        chrom=["17"],
        pos=[43071077],
        ref=["G"],
        alt=["T"],
        gene_symbol=["TP53"],
        consequence=["missense_variant"],
        allele_freq=[0.0001],
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


# ---------------------------------------------------------------------------
# 1. Stub mode
# ---------------------------------------------------------------------------

def test_stub_mode_no_path_returns_defaults():
    """No mim2gene_path → default values (0, 0)."""
    connector = OMIMConnector(mim2gene_path=None)
    df = _minimal_variant_df()
    result = connector.annotate_dataframe(df)
    assert "omim_n_diseases" in result.columns
    assert "omim_is_autosomal_dominant" in result.columns
    assert result["omim_n_diseases"].iloc[0] == 0
    assert result["omim_is_autosomal_dominant"].iloc[0] == 0


# ---------------------------------------------------------------------------
# 2. Empty DataFrame
# ---------------------------------------------------------------------------

def test_empty_dataframe_returns_empty_with_columns():
    connector = OMIMConnector()
    empty = pd.DataFrame(columns=["gene_symbol", "chrom", "pos"])
    result = connector.annotate_dataframe(empty)
    assert "omim_n_diseases" in result.columns
    assert "omim_is_autosomal_dominant" in result.columns
    assert len(result) == 0


# ---------------------------------------------------------------------------
# 3. _parse_mim2gene with temp file
# ---------------------------------------------------------------------------

def test_parse_mim2gene_counts_phenotypes(tmp_path):
    path = _write_mim2gene(tmp_path)
    connector = OMIMConnector(mim2gene_path=path)
    gene_table = connector._parse_mim2gene(path)
    assert "gene_symbol" in gene_table.columns
    assert "omim_n_diseases" in gene_table.columns
    # TP53 has 2 phenotype entries in the test data
    tp53_row = gene_table[gene_table["gene_symbol"] == "TP53"]
    assert len(tp53_row) == 1
    assert tp53_row["omim_n_diseases"].iloc[0] == 2


def test_parse_mim2gene_excludes_gene_type_rows(tmp_path):
    """MIM_type='gene' rows must not be counted."""
    path = _write_mim2gene(tmp_path)
    connector = OMIMConnector(mim2gene_path=path)
    gene_table = connector._parse_mim2gene(path)
    # AMBP has MIM_type='gene', should not appear
    ambp_rows = gene_table[gene_table["gene_symbol"] == "AMBP"]
    assert len(ambp_rows) == 0


# ---------------------------------------------------------------------------
# 4. annotate_dataframe join
# ---------------------------------------------------------------------------

def test_annotate_dataframe_known_gene(tmp_path):
    path = _write_mim2gene(tmp_path)
    connector = OMIMConnector(mim2gene_path=path)
    df = _minimal_variant_df(gene_symbol="TP53")
    result = connector.annotate_dataframe(df)
    assert result["omim_n_diseases"].iloc[0] == 2


def test_annotate_dataframe_unknown_gene_gets_zero(tmp_path):
    path = _write_mim2gene(tmp_path)
    connector = OMIMConnector(mim2gene_path=path)
    df = _minimal_variant_df(gene_symbol="UNKNOWNGENE99")
    result = connector.annotate_dataframe(df)
    assert result["omim_n_diseases"].iloc[0] == 0


# ---------------------------------------------------------------------------
# 5. fetch() round-trip
# ---------------------------------------------------------------------------

def test_fetch_round_trip(tmp_path):
    path = _write_mim2gene(tmp_path)
    connector = OMIMConnector(mim2gene_path=path)
    df = _minimal_variant_df(gene_symbol="BRCA1")
    result = connector.fetch(variant_df=df)
    assert "omim_n_diseases" in result.columns
    assert result["omim_n_diseases"].iloc[0] == 1


# ---------------------------------------------------------------------------
# 6. TABULAR_FEATURES membership
# ---------------------------------------------------------------------------

def test_omim_features_in_tabular_features():
    assert "omim_n_diseases" in TABULAR_FEATURES
    assert "omim_is_autosomal_dominant" in TABULAR_FEATURES


# ---------------------------------------------------------------------------
# 7-8. engineer_features wiring
# ---------------------------------------------------------------------------

def test_engineer_features_omim_default_zero_when_absent():
    df = _engineer_df()
    assert "omim_n_diseases" not in df.columns
    feats = engineer_features(df)
    assert feats.loc[0, "omim_n_diseases"] == 0
    assert feats.loc[0, "omim_is_autosomal_dominant"] == 0
    assert not feats["omim_n_diseases"].isnull().any()


def test_engineer_features_omim_real_value_passes_through():
    df = _engineer_df(omim_n_diseases=5, omim_is_autosomal_dominant=1)
    feats = engineer_features(df)
    assert feats.loc[0, "omim_n_diseases"] == 5
    assert feats.loc[0, "omim_is_autosomal_dominant"] == 1
