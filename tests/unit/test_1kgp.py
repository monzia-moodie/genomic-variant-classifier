"""
Unit tests for src/data/connectors/connector_1kgp.py

Run with:
    python -m pytest tests/unit/test_1kgp.py -v --tb=short
"""
from __future__ import annotations

import gzip
import io
import textwrap
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.data.connectors.connector_1kgp import (
    KGPConnector,
    KGPScores,
    MISSING_AF_DEFAULT,
    _norm_chrom,
    _parse_info,
    _parse_vcf,
    _df_to_index,
    engineer_kgp_features,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Minimal synthetic VCF content covering common edge cases
_SYNTHETIC_VCF = textwrap.dedent("""\
    ##fileformat=VCFv4.1
    ##INFO=<ID=AF_afr,Number=A,Type=Float,Description="AFR AF">
    ##INFO=<ID=AF_eur,Number=A,Type=Float,Description="EUR AF">
    ##INFO=<ID=AF_eas,Number=A,Type=Float,Description="EAS AF">
    ##INFO=<ID=AF_sas,Number=A,Type=Float,Description="SAS AF">
    ##INFO=<ID=AF_amr,Number=A,Type=Float,Description="AMR AF">
    #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO
    1\t925952\t.\tG\tA\t.\tPASS\tAF_afr=0.12;AF_eur=0.05;AF_eas=0.02;AF_sas=0.08;AF_amr=0.06
    2\t1000000\t.\tC\tT\t.\tPASS\tAF_afr=0.0;AF_eur=0.0;AF_eas=0.0;AF_sas=0.0;AF_amr=0.0
    X\t500000\t.\tA\tG\t.\tPASS\tAF_afr=0.33;AF_eur=0.21;AF_eas=0.15;AF_sas=0.19;AF_amr=0.25
    1\t200000\t.\tT\tA,C\t.\tPASS\tAF_afr=0.01,0.02;AF_eur=0.03,0.04;AF_eas=0.05,0.06;AF_sas=0.07,0.08;AF_amr=0.09,0.10
    MT\t73\t.\tA\tG\t.\tPASS\tAF_afr=0.50;AF_eur=0.40;AF_eas=0.60;AF_sas=0.55;AF_amr=0.45
""")


@pytest.fixture
def synthetic_vcf_gz(tmp_path: Path) -> Path:
    """Write the synthetic VCF to a gzipped temp file and return its path."""
    vcf_path = tmp_path / "test.vcf.gz"
    with gzip.open(vcf_path, "wt") as fh:
        fh.write(_SYNTHETIC_VCF)
    return vcf_path


@pytest.fixture
def connector(synthetic_vcf_gz: Path) -> KGPConnector:
    return KGPConnector(vcf_path=synthetic_vcf_gz)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "chrom": ["1",  "2",       "X",      "chr1",   "22"],
        "pos":   [925952, 1000000, 500000,   200000,   999],
        "ref":   ["G",  "C",       "A",      "T",      "C"],
        "alt":   ["A",  "T",       "G",      "A",      "G"],
    })


# ---------------------------------------------------------------------------
# _norm_chrom
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("1",    "1"),
    ("chr1", "1"),
    ("chrX", "X"),
    ("X",    "X"),
    ("chrM", "MT"),
    ("M",    "MT"),
    ("MT",   "MT"),
    ("22",   "22"),
])
def test_norm_chrom(raw: str, expected: str) -> None:
    assert _norm_chrom(raw) == expected


# ---------------------------------------------------------------------------
# _parse_info
# ---------------------------------------------------------------------------

def test_parse_info_key_value() -> None:
    info = _parse_info("AF_afr=0.12;AF_eur=0.05;PASS")
    assert info["AF_afr"] == "0.12"
    assert info["AF_eur"] == "0.05"
    assert info["PASS"] == "true"


def test_parse_info_empty() -> None:
    assert _parse_info("") == {}


def test_parse_info_dot_value() -> None:
    info = _parse_info("AF_afr=.;AF_eur=0.1")
    assert info["AF_afr"] == "."


# ---------------------------------------------------------------------------
# _parse_vcf
# ---------------------------------------------------------------------------

def test_parse_vcf_row_count(synthetic_vcf_gz: Path) -> None:
    df = _parse_vcf(synthetic_vcf_gz)
    # 3 biallelic + 2 from multi-allelic expansion + 1 MT = 6 rows
    assert len(df) == 6


def test_parse_vcf_columns(synthetic_vcf_gz: Path) -> None:
    df = _parse_vcf(synthetic_vcf_gz)
    for col in ["chrom", "pos", "ref", "alt"] + KGPConnector.POPULATION_COLS:
        assert col in df.columns


def test_parse_vcf_chrom_normalised(synthetic_vcf_gz: Path) -> None:
    df = _parse_vcf(synthetic_vcf_gz)
    assert "chr1" not in df["chrom"].values
    assert "1" in df["chrom"].values


def test_parse_vcf_no_nans(synthetic_vcf_gz: Path) -> None:
    df = _parse_vcf(synthetic_vcf_gz)
    for col in KGPConnector.POPULATION_COLS:
        assert df[col].isna().sum() == 0


def test_parse_vcf_af_range(synthetic_vcf_gz: Path) -> None:
    df = _parse_vcf(synthetic_vcf_gz)
    for col in KGPConnector.POPULATION_COLS:
        assert (df[col] >= 0.0).all()
        assert (df[col] <= 1.0).all()


def test_parse_vcf_multiallelic_expansion(synthetic_vcf_gz: Path) -> None:
    df = _parse_vcf(synthetic_vcf_gz)
    # chrom=1, pos=200000 should produce two rows: alt=A and alt=C
    multi = df[(df["chrom"] == "1") & (df["pos"] == 200000)]
    assert len(multi) == 2
    alts = set(multi["alt"].tolist())
    assert alts == {"A", "C"}


def test_parse_vcf_multiallelic_correct_afs(synthetic_vcf_gz: Path) -> None:
    df = _parse_vcf(synthetic_vcf_gz)
    row_a = df[(df["chrom"] == "1") & (df["pos"] == 200000) & (df["alt"] == "A")].iloc[0]
    row_c = df[(df["chrom"] == "1") & (df["pos"] == 200000) & (df["alt"] == "C")].iloc[0]
    assert pytest.approx(row_a.af_1kg_afr, abs=1e-6) == 0.01
    assert pytest.approx(row_c.af_1kg_afr, abs=1e-6) == 0.02


# ---------------------------------------------------------------------------
# KGPConnector — stub mode
# ---------------------------------------------------------------------------

def test_stub_mode_no_path_returns_defaults() -> None:
    c = KGPConnector(vcf_path=None)
    scores = c.get_scores("1", 925952, "G", "A")
    assert scores == KGPScores()
    for col in KGPConnector.POPULATION_COLS:
        assert getattr(scores, col) == MISSING_AF_DEFAULT


def test_stub_mode_annotate_fills_defaults() -> None:
    c = KGPConnector(vcf_path=None)
    df = pd.DataFrame({"chrom": ["1"], "pos": [1], "ref": ["A"], "alt": ["T"]})
    out = c.annotate(df)
    for col in KGPConnector.POPULATION_COLS:
        assert out[col].iloc[0] == MISSING_AF_DEFAULT


def test_missing_file_returns_defaults(tmp_path: Path) -> None:
    c = KGPConnector(vcf_path=tmp_path / "nonexistent.vcf.gz")
    scores = c.get_scores("1", 925952, "G", "A")
    assert scores == KGPScores()


def test_empty_dataframe_returns_empty_with_columns() -> None:
    c = KGPConnector(vcf_path=None)
    out = c.annotate(pd.DataFrame())
    for col in KGPConnector.POPULATION_COLS:
        assert col in out.columns


# ---------------------------------------------------------------------------
# KGPConnector — file-based annotation
# ---------------------------------------------------------------------------

def test_known_variant_returns_real_scores(connector: KGPConnector) -> None:
    scores = connector.get_scores("1", 925952, "G", "A")
    assert pytest.approx(scores.af_1kg_afr, abs=1e-6) == 0.12
    assert pytest.approx(scores.af_1kg_eur, abs=1e-6) == 0.05
    assert pytest.approx(scores.af_1kg_eas, abs=1e-6) == 0.02
    assert pytest.approx(scores.af_1kg_sas, abs=1e-6) == 0.08
    assert pytest.approx(scores.af_1kg_amr, abs=1e-6) == 0.06


def test_missing_variant_returns_defaults(connector: KGPConnector) -> None:
    scores = connector.get_scores("7", 999999, "A", "T")
    assert scores == KGPScores()


def test_chr_prefix_stripped_on_get_scores(connector: KGPConnector) -> None:
    # Should find the same variant whether passed as "1" or "chr1"
    s1 = connector.get_scores("1", 925952, "G", "A")
    s2 = connector.get_scores("chr1", 925952, "G", "A")
    assert s1 == s2


def test_lowercase_alleles_accepted(connector: KGPConnector) -> None:
    s1 = connector.get_scores("1", 925952, "G", "A")
    s2 = connector.get_scores("1", 925952, "g", "a")
    assert s1 == s2


def test_x_chromosome_annotated(connector: KGPConnector) -> None:
    scores = connector.get_scores("X", 500000, "A", "G")
    assert pytest.approx(scores.af_1kg_afr, abs=1e-6) == 0.33


def test_mt_chromosome_annotated(connector: KGPConnector) -> None:
    scores = connector.get_scores("MT", 73, "A", "G")
    assert pytest.approx(scores.af_1kg_eas, abs=1e-6) == 0.60


def test_annotate_adds_all_five_columns(connector: KGPConnector, sample_df: pd.DataFrame) -> None:
    out = connector.annotate(sample_df)
    for col in KGPConnector.POPULATION_COLS:
        assert col in out.columns


def test_annotate_returns_copy(connector: KGPConnector, sample_df: pd.DataFrame) -> None:
    out = connector.annotate(sample_df)
    assert out is not sample_df


def test_annotate_does_not_mutate_input(connector: KGPConnector, sample_df: pd.DataFrame) -> None:
    original_cols = list(sample_df.columns)
    connector.annotate(sample_df)
    assert list(sample_df.columns) == original_cols


def test_annotate_correct_scores_for_hit(connector: KGPConnector) -> None:
    df = pd.DataFrame({
        "chrom": ["1"], "pos": [925952], "ref": ["G"], "alt": ["A"],
    })
    out = connector.annotate(df)
    assert pytest.approx(out["af_1kg_afr"].iloc[0], abs=1e-6) == 0.12
    assert pytest.approx(out["af_1kg_eur"].iloc[0], abs=1e-6) == 0.05


def test_annotate_default_for_miss(connector: KGPConnector) -> None:
    df = pd.DataFrame({
        "chrom": ["99"], "pos": [1], "ref": ["A"], "alt": ["T"],
    })
    out = connector.annotate(df)
    for col in KGPConnector.POPULATION_COLS:
        assert out[col].iloc[0] == MISSING_AF_DEFAULT


def test_annotate_no_nans(connector: KGPConnector, sample_df: pd.DataFrame) -> None:
    out = connector.annotate(sample_df)
    for col in KGPConnector.POPULATION_COLS:
        assert out[col].isna().sum() == 0


def test_annotate_preserves_existing_columns(
    connector: KGPConnector, sample_df: pd.DataFrame
) -> None:
    out = connector.annotate(sample_df)
    for col in sample_df.columns:
        assert col in out.columns


def test_annotate_replaces_existing_1kgp_columns(connector: KGPConnector) -> None:
    df = pd.DataFrame({
        "chrom": ["1"], "pos": [925952], "ref": ["G"], "alt": ["A"],
        "af_1kg_afr": [0.999],   # stale value should be overwritten
    })
    out = connector.annotate(df)
    assert pytest.approx(out["af_1kg_afr"].iloc[0], abs=1e-6) == 0.12


def test_annotate_score_range(connector: KGPConnector, sample_df: pd.DataFrame) -> None:
    out = connector.annotate(sample_df)
    for col in KGPConnector.POPULATION_COLS:
        assert (out[col] >= 0.0).all()
        assert (out[col] <= 1.0).all()


# ---------------------------------------------------------------------------
# Parquet cache
# ---------------------------------------------------------------------------

def test_parquet_cache_written_and_reused(synthetic_vcf_gz: Path, tmp_path: Path) -> None:
    c = KGPConnector(vcf_path=synthetic_vcf_gz, cache_dir=tmp_path)
    # First call — builds and writes cache
    c.get_scores("1", 925952, "G", "A")
    cache_files = list(tmp_path.glob("*.parquet"))
    assert len(cache_files) == 1

    # Second connector instance — should load from cache, not re-parse VCF
    c2 = KGPConnector(vcf_path=synthetic_vcf_gz, cache_dir=tmp_path)
    with patch.object(
        __import__("src.data.connectors.connector_1kgp", fromlist=["_parse_vcf"]),
        "_parse_vcf",
        side_effect=AssertionError("VCF should not be re-parsed when cache exists"),
    ):
        s = c2.get_scores("1", 925952, "G", "A")
    assert pytest.approx(s.af_1kg_afr, abs=1e-6) == 0.12


# ---------------------------------------------------------------------------
# KGPScores dataclass
# ---------------------------------------------------------------------------

def test_kgp_scores_default_values() -> None:
    s = KGPScores()
    for col in KGPConnector.POPULATION_COLS:
        assert getattr(s, col) == MISSING_AF_DEFAULT


def test_kgp_scores_is_frozen() -> None:
    s = KGPScores(af_1kg_afr=0.5)
    with pytest.raises(Exception):
        s.af_1kg_afr = 0.9  # type: ignore[misc]


def test_kgp_scores_as_dict() -> None:
    s = KGPScores(af_1kg_afr=0.1, af_1kg_eur=0.2)
    d = s.as_dict()
    assert d["af_1kg_afr"] == pytest.approx(0.1)
    assert d["af_1kg_eur"] == pytest.approx(0.2)
    assert set(d.keys()) == set(KGPConnector.POPULATION_COLS)


# ---------------------------------------------------------------------------
# engineer_kgp_features()
# ---------------------------------------------------------------------------

def test_engineer_kgp_features_stub_mode() -> None:
    row = {"chrom": "1", "pos": 925952, "ref": "G", "alt": "A"}
    out = engineer_kgp_features(row, connector=None)
    for col in KGPConnector.POPULATION_COLS:
        assert out[col] == MISSING_AF_DEFAULT


def test_engineer_kgp_features_precomputed_values_passed_through() -> None:
    row = {
        "chrom": "1", "pos": 1, "ref": "A", "alt": "T",
        "af_1kg_afr": 0.77,
        "af_1kg_eur": 0.33,
        "af_1kg_eas": None,   # None should fall back to connector/default
        "af_1kg_sas": 0.11,
        "af_1kg_amr": 0.22,
    }
    out = engineer_kgp_features(row, connector=None)
    assert out["af_1kg_afr"] == pytest.approx(0.77)
    assert out["af_1kg_eur"] == pytest.approx(0.33)
    assert out["af_1kg_eas"] == MISSING_AF_DEFAULT   # None → default
    assert out["af_1kg_sas"] == pytest.approx(0.11)
    assert out["af_1kg_amr"] == pytest.approx(0.22)


def test_engineer_kgp_features_with_connector(connector: KGPConnector) -> None:
    row = {"chrom": "1", "pos": 925952, "ref": "G", "alt": "A"}
    out = engineer_kgp_features(row, connector=connector)
    assert pytest.approx(out["af_1kg_afr"], abs=1e-6) == 0.12
    assert pytest.approx(out["af_1kg_eur"], abs=1e-6) == 0.05


# ---------------------------------------------------------------------------
# TABULAR_FEATURES integration
# ---------------------------------------------------------------------------

def test_population_cols_in_tabular_features() -> None:
    """
    Smoke test: verify the 1KGP columns are declared in TABULAR_FEATURES.
    This test will FAIL until the features are added to src/api/schemas.py —
    which is the intended behaviour (test-driven integration).
    """
    from src.models.variant_ensemble import TABULAR_FEATURES
    for col in KGPConnector.POPULATION_COLS:
        assert col in TABULAR_FEATURES, (
            f"{col} is missing from TABULAR_FEATURES in src/api/schemas.py. "
            f"Add it alongside the other AF features."
        )


def test_no_basicconfig_in_module() -> None:
    """Logging discipline: connector must not call logging.basicConfig."""
    import ast
    src_path = Path("src/data/connectors/connector_1kgp.py")
    if not src_path.exists():
        pytest.skip("connector_1kgp.py not yet in src/ — copy it first")
    tree = ast.parse(src_path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "basicConfig":
                pytest.fail("connector_1kgp.py must not call logging.basicConfig()")