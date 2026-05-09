"""
tests/unit/test_eve.py
=======================
Unit tests for EVEConnector and its wiring into engineer_features.

Coverage:
  1.  Stub mode (no eve_path) — all scores = 0.5
  2.  Empty DataFrame — empty output with column present
  3.  _hgvsp_to_eve_key helper — known conversion cases
  4.  _annotate with a small lookup DataFrame
  5.  fetch() round-trip with a temp CSV directory
  6.  TABULAR_FEATURES membership — eve_score IS present
  7.  engineer_features default (missing column → 0.5)
  8.  engineer_features real value passes through
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

from genomic_variant_classifier.data.eve import EVEConnector, _hgvsp_to_eve_key, DEFAULT_SCORE
from genomic_variant_classifier.models.variant_ensemble import TABULAR_FEATURES, engineer_features

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_eve_csv(tmp_path: Path, gene: str, rows: list[dict]) -> Path:
    """Write a minimal per-protein EVE CSV for a given gene."""
    path = tmp_path / f"{gene}_HUMAN_singles_scores.csv"
    fieldnames = [
        "mutations_protein_name", "position", "wt_aa", "mt_aa",
        "EVE_scores_ASM", "EVE_classes_25_pct_retained",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "mutations_protein_name": f"{gene}_HUMAN",
                "EVE_classes_25_pct_retained": "pathogenic",
                **row,
            })
    return path


def _minimal_variant_df(**overrides) -> pd.DataFrame:
    base = dict(
        variant_id=["clinvar:17:7674220:G:A"],
        chrom=["17"],
        pos=[7674220],
        ref=["G"],
        alt=["A"],
        gene_symbol=["TP53"],
        consequence=["missense_variant"],
        allele_freq=[0.0],
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
        alt=["A"],
    )
    base.update({k: [v] for k, v in overrides.items()})
    return pd.DataFrame(base)


# ---------------------------------------------------------------------------
# 1. Stub mode
# ---------------------------------------------------------------------------

def test_stub_mode_no_path_returns_default():
    """No eve_path → eve_score = 0.5 (not covered) for all variants."""
    connector = EVEConnector(eve_path=None)
    df = _minimal_variant_df()
    result = connector.annotate_dataframe(df)
    assert "eve_score" in result.columns
    assert result["eve_score"].iloc[0] == pytest.approx(DEFAULT_SCORE)


# ---------------------------------------------------------------------------
# 2. Empty DataFrame
# ---------------------------------------------------------------------------

def test_empty_dataframe_returns_empty_with_column():
    connector = EVEConnector()
    empty = pd.DataFrame(columns=["gene_symbol", "protein_change"])
    result = connector.annotate_dataframe(empty)
    assert "eve_score" in result.columns
    assert len(result) == 0


# ---------------------------------------------------------------------------
# 3. _hgvsp_to_eve_key helper
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("protein_change, expected", [
    ("p.Arg175His",   "R175H"),
    ("p.Gly12Val",    "G12V"),
    ("p.Val600Glu",   "V600E"),
    ("p.Lys117Asn",   "K117N"),
    ("p.Arg248Trp",   "R248W"),
    ("",              None),
    (None,            None),
    ("p.Arg175*",     None),   # stop gained → not a missense → None
    ("synonymous",    None),   # no HGVSp match
])
def test_hgvsp_to_eve_key(protein_change, expected):
    result = _hgvsp_to_eve_key(protein_change)
    assert result == expected, (
        f"_hgvsp_to_eve_key({protein_change!r}) = {result!r}, expected {expected!r}"
    )


# ---------------------------------------------------------------------------
# 4. _annotate with known lookup
# ---------------------------------------------------------------------------

def test_annotate_matching_variant():
    lookup = pd.DataFrame({
        "gene_symbol": ["TP53", "TP53"],
        "aa_change":   ["R175H", "G245S"],
        "eve_score":   [0.92, 0.78],
    })
    connector = EVEConnector()
    df = _minimal_variant_df(gene_symbol="TP53", protein_change="p.Arg175His")
    result = connector._annotate(df, lookup)
    assert result["eve_score"].iloc[0] == pytest.approx(0.92, abs=1e-3)


def test_annotate_no_match_returns_default():
    lookup = pd.DataFrame({
        "gene_symbol": ["BRCA1"],
        "aa_change":   ["R1699W"],
        "eve_score":   [0.85],
    })
    connector = EVEConnector()
    df = _minimal_variant_df(gene_symbol="TP53", protein_change="p.Arg175His")
    result = connector._annotate(df, lookup)
    assert result["eve_score"].iloc[0] == pytest.approx(DEFAULT_SCORE)


def test_annotate_wrong_gene_returns_default():
    """Same aa_change but wrong gene → no match → default."""
    lookup = pd.DataFrame({
        "gene_symbol": ["BRCA2"],
        "aa_change":   ["R175H"],
        "eve_score":   [0.88],
    })
    connector = EVEConnector()
    df = _minimal_variant_df(gene_symbol="TP53", protein_change="p.Arg175His")
    result = connector._annotate(df, lookup)
    assert result["eve_score"].iloc[0] == pytest.approx(DEFAULT_SCORE)


# ---------------------------------------------------------------------------
# 5. fetch() round-trip with temp CSV directory
# ---------------------------------------------------------------------------

def test_fetch_round_trip_csv_directory(tmp_path):
    _write_eve_csv(tmp_path, "TP53", [
        {"position": "175", "wt_aa": "R", "mt_aa": "H", "EVE_scores_ASM": "0.92"},
        {"position": "248", "wt_aa": "R", "mt_aa": "W", "EVE_scores_ASM": "0.88"},
    ])
    connector = EVEConnector(eve_path=tmp_path)
    df = pd.DataFrame({
        "chrom":          ["17",   "17",   "17"],
        "pos":            [7674220, 7674200, 7674100],
        "ref":            ["G",   "G",   "G"],
        "alt":            ["A",   "A",   "A"],
        "gene_symbol":    ["TP53", "TP53", "UNKNOWN"],
        "protein_change": ["p.Arg175His", "p.Arg248Trp", None],
        "allele_freq":    [0.0, 0.0, 0.0],
    })
    result = connector.fetch(variant_df=df)
    assert result.loc[0, "eve_score"] == pytest.approx(0.92, abs=1e-3)
    assert result.loc[1, "eve_score"] == pytest.approx(0.88, abs=1e-3)
    assert result.loc[2, "eve_score"] == pytest.approx(DEFAULT_SCORE)


# ---------------------------------------------------------------------------
# 6. TABULAR_FEATURES membership
# ---------------------------------------------------------------------------

def test_eve_score_in_tabular_features():
    assert "eve_score" in TABULAR_FEATURES


# ---------------------------------------------------------------------------
# 7-8. engineer_features wiring
# ---------------------------------------------------------------------------

def test_engineer_features_eve_default_half_when_absent():
    """Missing eve_score → default 0.5 (not covered)."""
    df = _engineer_df()
    assert "eve_score" not in df.columns
    feats = engineer_features(df)
    assert feats.loc[0, "eve_score"] == pytest.approx(DEFAULT_SCORE)
    assert not feats["eve_score"].isnull().any()


def test_engineer_features_eve_real_score_passes_through():
    """Explicit eve_score must survive engineer_features unchanged."""
    df = _engineer_df(eve_score=0.92)
    feats = engineer_features(df)
    assert feats.loc[0, "eve_score"] == pytest.approx(0.92)
