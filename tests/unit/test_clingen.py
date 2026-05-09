"""
tests/unit/test_clingen.py
===========================
Unit tests for ClinGenConnector and its wiring into engineer_features.

Coverage:
  1.  Stub mode (no csv_path) — all scores = 0
  2.  Empty DataFrame — empty output with column present
  3.  _parse_csv with a small in-memory temp file
  4.  annotate_dataframe join — known classification scores
  5.  fetch() round-trip
  6.  TABULAR_FEATURES membership — clingen_validity_score IS present
  7.  engineer_features default (missing column → 0)
  8.  engineer_features real value passes through
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from genomic_variant_classifier.data.clingen import ClinGenConnector, CLASSIFICATION_SCORE
from genomic_variant_classifier.models.variant_ensemble import TABULAR_FEATURES, engineer_features

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CSV_CONTENT = """\
GENE SYMBOL,DISEASE LABEL,MOI,SOP,CLASSIFICATION,ONLINE REPORT,GCEP,UUID
BRCA1,Hereditary Breast and Ovarian Cancer Syndrome,AD,SOP8,Definitive,https://example.com,GCEP1,uuid1
TP53,Li-Fraumeni Syndrome,AD,SOP8,Definitive,https://example.com,GCEP2,uuid2
TP53,Li-Fraumeni Syndrome 2,AD,SOP7,Strong,https://example.com,GCEP2,uuid3
BRCA2,Hereditary Breast and Ovarian Cancer Syndrome,AD,SOP7,Moderate,https://example.com,GCEP1,uuid4
MLH1,Lynch Syndrome,AD,SOP7,No Known Disease Relationship,https://example.com,GCEP3,uuid5
NEWGENE,Unknown disease,AD,SOP6,Limited,https://example.com,GCEP4,uuid6
"""


def _write_csv(tmp_path: Path, content: str | None = None) -> Path:
    content = content if content is not None else _CSV_CONTENT
    path = tmp_path / "ClinGen-Gene-Disease-Summary.csv"
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
        allele_freq=[0.0001],
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
# 1. Stub mode
# ---------------------------------------------------------------------------

def test_stub_mode_no_path_returns_zero():
    """No csv_path → all scores = 0."""
    connector = ClinGenConnector(csv_path=None)
    df = _minimal_variant_df()
    result = connector.annotate_dataframe(df)
    assert "clingen_validity_score" in result.columns
    assert result["clingen_validity_score"].iloc[0] == 0


# ---------------------------------------------------------------------------
# 2. Empty DataFrame
# ---------------------------------------------------------------------------

def test_empty_dataframe_returns_empty_with_column():
    connector = ClinGenConnector()
    empty = pd.DataFrame(columns=["gene_symbol", "chrom"])
    result = connector.annotate_dataframe(empty)
    assert "clingen_validity_score" in result.columns
    assert len(result) == 0


# ---------------------------------------------------------------------------
# 3. _parse_csv with temp file
# ---------------------------------------------------------------------------

def test_parse_csv_known_scores(tmp_path):
    path = _write_csv(tmp_path)
    connector = ClinGenConnector(csv_path=path)
    gene_table = connector._parse_csv(path)

    assert "gene_symbol" in gene_table.columns
    assert "clingen_validity_score" in gene_table.columns

    # BRCA1 → Definitive = 5
    brca1 = gene_table[gene_table["gene_symbol"] == "BRCA1"]
    assert brca1["clingen_validity_score"].iloc[0] == 5

    # TP53 has Definitive (5) and Strong (4) → max = 5
    tp53 = gene_table[gene_table["gene_symbol"] == "TP53"]
    assert tp53["clingen_validity_score"].iloc[0] == 5

    # BRCA2 → Moderate = 3
    brca2 = gene_table[gene_table["gene_symbol"] == "BRCA2"]
    assert brca2["clingen_validity_score"].iloc[0] == 3

    # MLH1 → No Known Disease Relationship = 1
    mlh1 = gene_table[gene_table["gene_symbol"] == "MLH1"]
    assert mlh1["clingen_validity_score"].iloc[0] == 1


# ---------------------------------------------------------------------------
# 4. annotate_dataframe join
# ---------------------------------------------------------------------------

def test_annotate_dataframe_definitive_gene(tmp_path):
    path = _write_csv(tmp_path)
    connector = ClinGenConnector(csv_path=path)
    df = _minimal_variant_df(gene_symbol="BRCA1")
    result = connector.annotate_dataframe(df)
    assert result["clingen_validity_score"].iloc[0] == 5


def test_annotate_dataframe_unknown_gene_gets_zero(tmp_path):
    path = _write_csv(tmp_path)
    connector = ClinGenConnector(csv_path=path)
    df = _minimal_variant_df(gene_symbol="UNKNOWNGENE999")
    result = connector.annotate_dataframe(df)
    assert result["clingen_validity_score"].iloc[0] == 0


# ---------------------------------------------------------------------------
# 5. fetch() round-trip
# ---------------------------------------------------------------------------

def test_fetch_round_trip(tmp_path):
    path = _write_csv(tmp_path)
    connector = ClinGenConnector(csv_path=path)
    df = _minimal_variant_df(gene_symbol="BRCA2")
    result = connector.fetch(variant_df=df)
    assert "clingen_validity_score" in result.columns
    assert result["clingen_validity_score"].iloc[0] == 3


# ---------------------------------------------------------------------------
# 6. TABULAR_FEATURES membership
# ---------------------------------------------------------------------------

def test_clingen_validity_score_in_tabular_features():
    assert "clingen_validity_score" in TABULAR_FEATURES


# ---------------------------------------------------------------------------
# 7-8. engineer_features wiring
# ---------------------------------------------------------------------------

def test_engineer_features_clingen_default_zero_when_absent():
    df = _engineer_df()
    assert "clingen_validity_score" not in df.columns
    feats = engineer_features(df)
    assert feats.loc[0, "clingen_validity_score"] == 0
    assert not feats["clingen_validity_score"].isnull().any()


def test_engineer_features_clingen_real_value_passes_through():
    df = _engineer_df(clingen_validity_score=5)
    feats = engineer_features(df)
    assert feats.loc[0, "clingen_validity_score"] == 5


# ---------------------------------------------------------------------------
# Score mapping constants sanity
# ---------------------------------------------------------------------------

def test_classification_score_mapping_sane():
    """Definitive must be highest, mapping must cover all tiers."""
    assert CLASSIFICATION_SCORE["definitive"] == 5
    assert CLASSIFICATION_SCORE["strong"] == 4
    assert CLASSIFICATION_SCORE["moderate"] == 3
    assert CLASSIFICATION_SCORE["limited"] == 2
    assert CLASSIFICATION_SCORE["no known disease relationship"] == 1
    # All values are in [1, 5]
    assert all(1 <= v <= 5 for v in CLASSIFICATION_SCORE.values())
