"""
tests/unit/test_vep.py
=======================
Unit tests for VEPConnector and its wiring into engineer_features.

Coverage:
  1.  Stub mode (no file needed, runs in-place on protein_change column)
  2.  Empty DataFrame returns empty output with column present
  3.  _extract_codon_position helper — known values
  4.  annotate_dataframe with known protein_change strings
  5.  fetch() round-trip
  6.  TABULAR_FEATURES membership — codon_position IS in TABULAR_FEATURES
  7.  engineer_features default (missing column → 0)
  8.  engineer_features real value passes through
"""

from __future__ import annotations

import pandas as pd
import pytest

from genomic_variant_classifier.data.vep import VEPConnector, _extract_codon_position
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
        allele_freq=[0.0001],
        protein_change=["p.Arg175His"],
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
# 1. Stub mode — basic operation (no external file)
# ---------------------------------------------------------------------------

def test_stub_mode_no_file_needed():
    """VEPConnector requires no external file for basic codon_position derivation."""
    connector = VEPConnector()
    df = _minimal_variant_df(protein_change="p.Arg175His")
    result = connector.annotate_dataframe(df)
    assert "codon_position" in result.columns
    # (175 - 1) % 3 + 1 = 174 % 3 + 1 = 0 + 1 = 1
    assert result["codon_position"].iloc[0] == 1


def test_no_protein_change_gives_zero():
    """Missing protein_change → codon_position = 0."""
    connector = VEPConnector()
    df = _minimal_variant_df(protein_change=None)
    df["protein_change"] = None
    result = connector.annotate_dataframe(df)
    assert result["codon_position"].iloc[0] == 0


# ---------------------------------------------------------------------------
# 2. Empty DataFrame
# ---------------------------------------------------------------------------

def test_empty_dataframe_returns_empty_with_column():
    connector = VEPConnector()
    empty = pd.DataFrame(columns=["chrom", "pos", "ref", "alt", "protein_change"])
    result = connector.annotate_dataframe(empty)
    assert "codon_position" in result.columns
    assert len(result) == 0


# ---------------------------------------------------------------------------
# 3. _extract_codon_position helper
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("protein_change, expected", [
    ("p.Arg175His",   1),   # (175-1)%3+1 = 1
    ("p.Gly12Val",    3),   # (12-1)%3+1  = 3
    ("p.Val600Glu",   3),   # (600-1)%3+1 = 3  (599%3=2 → +1=3)
    ("p.Lys117Asn",   3),   # (117-1)%3+1 = 3
    ("p.Pro34Ser",    1),   # (34-1)%3+1  = 1
    ("",              0),   # empty → 0
    (None,            0),   # None → 0
    ("intron",        0),   # no digits match at expected position → 0 (actually '0' since re finds none)
])
def test_extract_codon_position_known_values(protein_change, expected):
    result = _extract_codon_position(protein_change)
    assert result == expected, (
        f"_extract_codon_position({protein_change!r}) = {result}, expected {expected}"
    )


# ---------------------------------------------------------------------------
# 4. annotate_dataframe — multiple protein_change strings
# ---------------------------------------------------------------------------

def test_annotate_multiple_variants():
    connector = VEPConnector()
    df = pd.DataFrame({
        "chrom":          ["17",  "13",   "1"],
        "pos":            [43071077, 32936732, 925952],
        "ref":            ["G",   "A",    "G"],
        "alt":            ["T",   "C",    "A"],
        "gene_symbol":    ["TP53","BRCA2","GENE1"],
        "protein_change": ["p.Arg175His", "p.Asn1239Thr", None],
        "allele_freq":    [0.0001, 0.0002, 0.5],
    })
    result = connector.annotate_dataframe(df)
    assert result["codon_position"].iloc[0] == 1    # 175 → pos 1
    assert result["codon_position"].iloc[1] == 3    # 1239 → (1238%3+1)=3
    assert result["codon_position"].iloc[2] == 0    # None → 0


# ---------------------------------------------------------------------------
# 5. fetch() round-trip
# ---------------------------------------------------------------------------

def test_fetch_round_trip():
    connector = VEPConnector()
    df = _minimal_variant_df(protein_change="p.Val600Glu")
    result = connector.fetch(variant_df=df)
    assert "codon_position" in result.columns
    assert result["codon_position"].iloc[0] == 3


# ---------------------------------------------------------------------------
# 6. TABULAR_FEATURES membership
# ---------------------------------------------------------------------------

def test_codon_position_in_tabular_features():
    """codon_position must be in TABULAR_FEATURES."""
    assert "codon_position" in TABULAR_FEATURES


# ---------------------------------------------------------------------------
# 7-8. engineer_features wiring
# ---------------------------------------------------------------------------

def test_engineer_features_default_zero_when_absent():
    """Missing codon_position column → default 0."""
    df = _engineer_df()
    assert "codon_position" not in df.columns
    feats = engineer_features(df)
    assert feats.loc[0, "codon_position"] == 0
    assert not feats["codon_position"].isnull().any()


def test_engineer_features_real_codon_position_passes_through():
    """Explicit codon_position in input must survive engineer_features unchanged."""
    df = _engineer_df(codon_position=2)
    feats = engineer_features(df)
    assert feats.loc[0, "codon_position"] == 2
