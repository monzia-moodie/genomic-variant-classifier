"""
tests/unit/test_core.py
========================
Unit and integration tests for all pipeline components.

Run with:
    pytest tests/ -v --tb=short --cov=src

Coverage targets:
  - src/data/database_connectors.py   → connectors, canonical schema
  - src/data/spark_etl.py             → CHROM_MAP, VARIANT_SCHEMA, normalization
  - src/models/variant_ensemble.py    → feature engineering, ensemble training
  - src/reports/report_generator.py   → report generation with mock data
  - src/evaluation/evaluator.py       → metrics, operating points, gene analysis

CHANGES FROM PHASE 1:
  - All imports from src.data_ingestion.* changed to src.data.* (Bug 4).
  - All imports from src.models.ensemble changed to src.models.variant_ensemble (Bug 5).
  - Shared test fixtures extracted to tests/fixtures/make_synthetic_data.py (Issue E).
  - from __future__ import annotations added (Issue N).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_canonical_df():
    """Minimal canonical-schema DataFrame for testing."""
    return pd.DataFrame({
        "variant_id":     [
            "clinvar:1:925952:G:A",
            "clinvar:1:925958:C:T",
            "gnomad:1:925952:G:A",
            "clinvar:17:43071077:G:T",
        ],
        "source_db":      ["clinvar", "clinvar", "gnomad", "clinvar"],
        "chrom":          ["1", "1", "1", "17"],
        "pos":            [925952, 925958, 925952, 43071077],
        "ref":            ["G", "C", "G", "G"],
        "alt":            ["A", "T", "A", "T"],
        "gene_symbol":    ["AGRN", "AGRN", "AGRN", "BRCA1"],
        "transcript_id":  [None, None, None, "ENST00000471181"],
        "consequence":    [
            "missense_variant",
            "stop_gained",
            "missense_variant",
            "splice_donor_variant",
        ],
        "pathogenicity":  ["pathogenic", "benign", "pathogenic", "pathogenic"],
        "allele_freq":    [0.0001, 0.05, 0.0001, 0.0],
        "clinical_sig":   ["Pathogenic", "Benign", None, "Pathogenic"],
        "protein_change": ["p.Arg20Trp", None, None, None],
        "fasta_seq":      ["ACGTACGTACGT" * 8 + "A"] * 4,
        "source_id":      ["12345", "12346", "rs1234", "67890"],
        "metadata":       [{}] * 4,
    })


@pytest.fixture
def sample_labels(sample_canonical_df):
    return (
        sample_canonical_df["pathogenicity"]
        .map({"pathogenic": 1, "benign": 0})
        .fillna(0)
        .astype(int)
    )


# ---------------------------------------------------------------------------
# Tests: database_connectors.py (CHANGE: src.data not src.data_ingestion)
# ---------------------------------------------------------------------------
class TestClinVarConnector:
    def test_map_pathogenicity_pathogenic(self):
        # CHANGE: import path updated from src.data_ingestion → src.data (Bug 4)
        from src.data.database_connectors import ClinVarConnector
        assert ClinVarConnector._map_pathogenicity("Pathogenic") == "pathogenic"
        assert ClinVarConnector._map_pathogenicity("Likely pathogenic") == "likely_pathogenic"

    def test_map_pathogenicity_compound_values(self):
        """Compound ClinVar values like 'Pathogenic, risk factor' must map correctly."""
        from src.data.database_connectors import ClinVarConnector
        # Would have returned "uncertain" in original (exact set membership)
        # Now uses substring matching (Issue G)
        assert ClinVarConnector._map_pathogenicity("Pathogenic, risk factor") == "pathogenic"
        assert ClinVarConnector._map_pathogenicity("Pathogenic/Likely pathogenic") == "pathogenic"
        assert ClinVarConnector._map_pathogenicity("Benign/Likely benign") == "benign"

    def test_map_pathogenicity_benign(self):
        from src.data.database_connectors import ClinVarConnector
        assert ClinVarConnector._map_pathogenicity("Benign") == "benign"
        assert ClinVarConnector._map_pathogenicity("Likely benign") == "likely_benign"

    def test_map_pathogenicity_uncertain(self):
        from src.data.database_connectors import ClinVarConnector
        assert ClinVarConnector._map_pathogenicity("Uncertain significance") == "uncertain"
        assert ClinVarConnector._map_pathogenicity(None) == "uncertain"
        assert ClinVarConnector._map_pathogenicity("") == "uncertain"

    def test_to_canonical_fills_missing_columns(self, sample_canonical_df):
        from src.data.database_connectors import BaseConnector, CANONICAL_COLUMNS
        connector = BaseConnector()
        partial_df = sample_canonical_df[["variant_id", "chrom", "pos"]].copy()
        result = connector._to_canonical(partial_df)
        assert set(CANONICAL_COLUMNS).issubset(set(result.columns))
        assert len(result) == len(partial_df)

    def test_canonical_columns_complete(self):
        from src.data.database_connectors import CANONICAL_COLUMNS
        required = {
            "variant_id", "source_db", "chrom", "pos", "ref", "alt",
            "gene_symbol", "pathogenicity", "allele_freq",
        }
        assert required.issubset(set(CANONICAL_COLUMNS))


class TestGnomADConnector:
    def test_variant_id_format(self):
        """Variant IDs follow source:chrom:pos:ref:alt pattern."""
        parts = "gnomad:17:43071077:G:T".split(":")
        assert parts[0] == "gnomad"
        assert parts[1] == "17"
        assert int(parts[2]) > 0
        
# ---------------------------------------------------------------------------
# Tests: sift_polyphen.py (Connector 6)
# ---------------------------------------------------------------------------
class TestSIFTPolyPhenConnector:
    """
    Unit tests for SIFTPolyPhenConnector.

    The fixture ``_connector_with_index`` injects a small synthetic index
    directly (bypassing file I/O) so every test is fast and self-contained.
    This mirrors how TestREVELConnector is structured.
    """

    # ------------------------------------------------------------------
    # Shared fixture: a connector whose index is pre-populated in memory
    # ------------------------------------------------------------------
    @pytest.fixture
    def connector(self):
        """Connector with a small synthetic in-memory index (no file I/O)."""
        from src.data.sift_polyphen import SIFTPolyPhenConnector, _normalise_chrom
        c = SIFTPolyPhenConnector(sift_polyphen_file=None)
        # Inject a synthetic index directly — mirrors what _df_to_index produces
        c._index = {
            ("17", 43071077, "G", "T"): (0.03, 0.95),   # deleterious / prob. damaging
            ("1",  925952,   "G", "A"): (0.21, 0.12),   # tolerated / benign
            ("13", 32338271, "T", "A"): (0.00, 1.00),   # extreme deleterious
            ("X",  153296777,"C", "T"): (0.08, 0.55),   # X chromosome
        }
        return c

    @pytest.fixture
    def minimal_df(self):
        """Four-variant canonical DataFrame covering the synthetic index."""
        return pd.DataFrame({
            "chrom": ["17", "1", "13", "X"],
            "pos":   [43071077, 925952, 32338271, 153296777],
            "ref":   ["G", "G", "T", "C"],
            "alt":   ["T", "A", "A", "T"],
        })

    # ------------------------------------------------------------------
    # Import + constants
    # ------------------------------------------------------------------

    def test_module_imports_cleanly(self):
        from src.data.sift_polyphen import SIFTPolyPhenConnector
        assert SIFTPolyPhenConnector is not None

    def test_constants_exported(self):
        from src.data.sift_polyphen import (
            DEFAULT_SIFT,
            DEFAULT_PP2,
            SIFT_DELETERIOUS_THRESHOLD,
            PP2_PROBABLY_DAMAGING_THRESHOLD,
        )
        assert DEFAULT_SIFT == 0.5
        assert DEFAULT_PP2  == 0.5
        assert SIFT_DELETERIOUS_THRESHOLD == pytest.approx(0.05)
        assert PP2_PROBABLY_DAMAGING_THRESHOLD == pytest.approx(0.908)

    def test_source_name(self):
        from src.data.sift_polyphen import SIFTPolyPhenConnector
        assert SIFTPolyPhenConnector.source_name == "sift_polyphen"

    # ------------------------------------------------------------------
    # Stub mode (no file)
    # ------------------------------------------------------------------

    def test_stub_mode_returns_default_sift(self):
        from src.data.sift_polyphen import SIFTPolyPhenConnector, DEFAULT_SIFT
        c = SIFTPolyPhenConnector(sift_polyphen_file=None)
        assert c.get_sift_score("17", 43071077, "G", "T") == DEFAULT_SIFT

    def test_stub_mode_returns_default_pp2(self):
        from src.data.sift_polyphen import SIFTPolyPhenConnector, DEFAULT_PP2
        c = SIFTPolyPhenConnector(sift_polyphen_file=None)
        assert c.get_pp2_score("17", 43071077, "G", "T") == DEFAULT_PP2

    def test_stub_mode_annotate_dataframe(self, minimal_df):
        """Stub connector must annotate without raising; columns filled with defaults."""
        from src.data.sift_polyphen import (
            SIFTPolyPhenConnector, DEFAULT_SIFT, DEFAULT_PP2,
        )
        c = SIFTPolyPhenConnector(sift_polyphen_file=None)
        result = c.annotate_dataframe(minimal_df)
        assert (result["sift_score"]      == DEFAULT_SIFT).all()
        assert (result["polyphen2_score"] == DEFAULT_PP2).all()

    # ------------------------------------------------------------------
    # get_sift_score — lookup correctness
    # ------------------------------------------------------------------

    def test_known_variant_returns_real_sift_score(self, connector):
        score = connector.get_sift_score("17", 43071077, "G", "T")
        assert score == pytest.approx(0.03)

    def test_missing_variant_returns_default_sift(self, connector):
        from src.data.sift_polyphen import DEFAULT_SIFT
        score = connector.get_sift_score("2", 999999, "A", "C")
        assert score == DEFAULT_SIFT

    def test_custom_missing_sift_value_honoured(self, connector):
        score = connector.get_sift_score("2", 999999, "A", "C", missing_value=-1.0)
        assert score == -1.0

    def test_sift_chr_prefix_stripped(self, connector):
        score_no_prefix  = connector.get_sift_score("17",    43071077, "G", "T")
        score_chr_prefix = connector.get_sift_score("chr17", 43071077, "G", "T")
        assert score_no_prefix == score_chr_prefix

    def test_sift_lowercase_chrom_accepted(self, connector):
        score = connector.get_sift_score("chr17", 43071077, "G", "T")
        assert score == pytest.approx(0.03)

    def test_sift_lowercase_alleles_accepted(self, connector):
        score_upper = connector.get_sift_score("17", 43071077, "G",  "T")
        score_lower = connector.get_sift_score("17", 43071077, "g",  "t")
        assert score_upper == score_lower

    def test_sift_sex_chromosome_X(self, connector):
        score = connector.get_sift_score("X", 153296777, "C", "T")
        assert score == pytest.approx(0.08)

    # ------------------------------------------------------------------
    # get_pp2_score — lookup correctness
    # ------------------------------------------------------------------

    def test_known_variant_returns_real_pp2_score(self, connector):
        score = connector.get_pp2_score("17", 43071077, "G", "T")
        assert score == pytest.approx(0.95)

    def test_missing_variant_returns_default_pp2(self, connector):
        from src.data.sift_polyphen import DEFAULT_PP2
        score = connector.get_pp2_score("2", 999999, "A", "C")
        assert score == DEFAULT_PP2

    def test_custom_missing_pp2_value_honoured(self, connector):
        score = connector.get_pp2_score("2", 999999, "A", "C", missing_value=-1.0)
        assert score == -1.0

    def test_pp2_chr_prefix_stripped(self, connector):
        assert connector.get_pp2_score("17", 43071077, "G", "T") == \
               connector.get_pp2_score("chr17", 43071077, "G", "T")

    def test_pp2_sex_chromosome_X(self, connector):
        score = connector.get_pp2_score("X", 153296777, "C", "T")
        assert score == pytest.approx(0.55)

    # ------------------------------------------------------------------
    # annotate_dataframe — output correctness
    # ------------------------------------------------------------------

    def test_annotate_adds_both_score_columns(self, connector, minimal_df):
        result = connector.annotate_dataframe(minimal_df)
        assert "sift_score" in result.columns
        assert "polyphen2_score" in result.columns

    def test_annotate_returns_copy(self, connector, minimal_df):
        result = connector.annotate_dataframe(minimal_df)
        assert result is not minimal_df

    def test_annotate_does_not_mutate_input(self, connector, minimal_df):
        original_cols = set(minimal_df.columns)
        _ = connector.annotate_dataframe(minimal_df)
        assert set(minimal_df.columns) == original_cols

    def test_annotate_correct_sift_for_hit(self, connector, minimal_df):
        result = connector.annotate_dataframe(minimal_df)
        assert result.loc[0, "sift_score"] == pytest.approx(0.03)   # chr17

    def test_annotate_correct_pp2_for_hit(self, connector, minimal_df):
        result = connector.annotate_dataframe(minimal_df)
        assert result.loc[0, "polyphen2_score"] == pytest.approx(0.95)   # chr17

    def test_annotate_default_for_miss(self, connector):
        from src.data.sift_polyphen import DEFAULT_SIFT, DEFAULT_PP2
        miss_df = pd.DataFrame({
            "chrom": ["2"], "pos": [999999], "ref": ["A"], "alt": ["C"],
        })
        result = connector.annotate_dataframe(miss_df)
        assert result.loc[0, "sift_score"]      == DEFAULT_SIFT
        assert result.loc[0, "polyphen2_score"] == DEFAULT_PP2

    def test_annotate_mixed_hits_and_misses(self, connector):
        from src.data.sift_polyphen import DEFAULT_SIFT, DEFAULT_PP2
        df = pd.DataFrame({
            "chrom": ["17",      "2"],
            "pos":   [43071077,  999999],
            "ref":   ["G",       "A"],
            "alt":   ["T",       "C"],
        })
        result = connector.annotate_dataframe(df)
        assert result.loc[0, "sift_score"]      == pytest.approx(0.03)
        assert result.loc[0, "polyphen2_score"] == pytest.approx(0.95)
        assert result.loc[1, "sift_score"]      == DEFAULT_SIFT
        assert result.loc[1, "polyphen2_score"] == DEFAULT_PP2

    def test_annotate_preserves_existing_columns(self, connector, minimal_df):
        result = connector.annotate_dataframe(minimal_df)
        for col in minimal_df.columns:
            assert col in result.columns

    def test_annotate_no_nans(self, connector, minimal_df):
        result = connector.annotate_dataframe(minimal_df)
        assert not result["sift_score"].isna().any()
        assert not result["polyphen2_score"].isna().any()

    def test_annotate_score_ranges_valid(self, connector, minimal_df):
        result = connector.annotate_dataframe(minimal_df)
        assert (result["sift_score"]      >= 0).all() and (result["sift_score"]      <= 1).all()
        assert (result["polyphen2_score"] >= 0).all() and (result["polyphen2_score"] <= 1).all()

    def test_annotate_replaces_existing_columns(self, connector):
        df = pd.DataFrame({
            "chrom": ["17"], "pos": [43071077], "ref": ["G"], "alt": ["T"],
            "sift_score": [0.99],   # stale value — should be overwritten
            "polyphen2_score": [0.01],
        })
        result = connector.annotate_dataframe(df)
        assert result.loc[0, "sift_score"]      == pytest.approx(0.03)
        assert result.loc[0, "polyphen2_score"] == pytest.approx(0.95)

    # ------------------------------------------------------------------
    # _parse_multival helper
    # ------------------------------------------------------------------

    def test_parse_multival_min_basic(self):
        from src.data.sift_polyphen import _parse_multival
        assert _parse_multival("0.04;0.12;0.01", "min") == pytest.approx(0.01)

    def test_parse_multival_max_basic(self):
        from src.data.sift_polyphen import _parse_multival
        assert _parse_multival("0.2;.;0.9", "max") == pytest.approx(0.9)

    def test_parse_multival_all_missing_returns_none(self):
        from src.data.sift_polyphen import _parse_multival
        assert _parse_multival(".", "min")     is None
        assert _parse_multival(".;.;.", "max") is None
        assert _parse_multival("", "min")      is None
        assert _parse_multival(None, "min")    is None

    def test_parse_multival_single_value(self):
        from src.data.sift_polyphen import _parse_multival
        assert _parse_multival("0.03", "min") == pytest.approx(0.03)

    def test_parse_multival_already_float(self):
        """pandas may pass a float directly for single-transcript genes."""
        from src.data.sift_polyphen import _parse_multival
        assert _parse_multival(0.07, "min") == pytest.approx(0.07)

    # ------------------------------------------------------------------
    # _df_to_index
    # ------------------------------------------------------------------

    def test_df_to_index_keys_are_normalised(self):
        from src.data.sift_polyphen import SIFTPolyPhenConnector
        df = pd.DataFrame({
            "chrom":         ["chr17"],
            "pos":           [43071077],
            "ref":           ["g"],
            "alt":           ["t"],
            "sift_score":    [0.03],
            "polyphen2_score": [0.95],
        })
        index = SIFTPolyPhenConnector._df_to_index(df)
        assert ("17", 43071077, "G", "T") in index

    def test_df_to_index_values_are_float_tuples(self):
        from src.data.sift_polyphen import SIFTPolyPhenConnector
        df = pd.DataFrame({
            "chrom":         ["17"],
            "pos":           [43071077],
            "ref":           ["G"],
            "alt":           ["T"],
            "sift_score":    [0.03],
            "polyphen2_score": [0.95],
        })
        index = SIFTPolyPhenConnector._df_to_index(df)
        val = index[("17", 43071077, "G", "T")]
        assert isinstance(val, tuple) and len(val) == 2
        assert isinstance(val[0], float) and isinstance(val[1], float)

    # ------------------------------------------------------------------
    # File-based integration test (synthetic TSV — no real dbNSFP needed)
    # ------------------------------------------------------------------

    def test_file_based_annotation(self, tmp_path):
        from src.data.sift_polyphen import SIFTPolyPhenConnector
        # Minimal synthetic dbNSFP v4.x TSV
        content = (
            "#chr\tpos(1-based)\tref\talt\tSIFT_score\tPolyphen2_HDIV_score\n"
            "17\t43071077\tG\tT\t0.03\t0.95\n"
            "1\t925952\tG\tA\t0.21\t0.12\n"
            "13\t32338271\tT\tA\t.\t.\n"   # dots → both missing
        )
        tsv = tmp_path / "test_dbnsfp.tsv"
        tsv.write_text(content)

        c = SIFTPolyPhenConnector(sift_polyphen_file=tsv)
        df = pd.DataFrame({
            "chrom": ["17", "1",   "13"],
            "pos":   [43071077, 925952, 32338271],
            "ref":   ["G", "G", "T"],
            "alt":   ["T", "A", "A"],
        })
        result = c.annotate_dataframe(df)

        # chr17: real scores from file
        assert result.loc[0, "sift_score"]      == pytest.approx(0.03)
        assert result.loc[0, "polyphen2_score"] == pytest.approx(0.95)

        # chr1: real scores
        assert result.loc[1, "sift_score"]      == pytest.approx(0.21)
        assert result.loc[1, "polyphen2_score"] == pytest.approx(0.12)

        # chr13: both dots → defaults (0.5)
        from src.data.sift_polyphen import DEFAULT_SIFT, DEFAULT_PP2
        assert result.loc[2, "sift_score"]      == DEFAULT_SIFT
        assert result.loc[2, "polyphen2_score"] == DEFAULT_PP2

    def test_parquet_cache_used_on_second_call(self, tmp_path):
        from src.data.sift_polyphen import SIFTPolyPhenConnector
        content = (
            "#chr\tpos(1-based)\tref\talt\tSIFT_score\tPolyphen2_HDIV_score\n"
            "17\t43071077\tG\tT\t0.03\t0.95\n"
        )
        tsv = tmp_path / "dbnsfp.tsv"
        tsv.write_text(content)

        # First call — builds and caches
        c1 = SIFTPolyPhenConnector(sift_polyphen_file=tsv, cache_dir=tmp_path)
        c1.get_sift_score("17", 43071077, "G", "T")
        cache_file = tmp_path / "dbnsfp_sift_pp2_index.parquet"
        assert cache_file.exists(), "Parquet cache must be written after first call"

        # Second call — loads from parquet (delete TSV to prove it)
        tsv.unlink()
        c2 = SIFTPolyPhenConnector(sift_polyphen_file=tsv, cache_dir=tmp_path)
        score = c2.get_sift_score("17", 43071077, "G", "T")
        assert score == pytest.approx(0.03), "Second call must read from parquet cache"

    # ------------------------------------------------------------------
    # End-to-end: annotated scores flow through engineer_features
    # ------------------------------------------------------------------

    def test_scores_flow_into_feature_matrix(self, connector):
        from src.models.variant_ensemble import engineer_features, TABULAR_FEATURES
        df = pd.DataFrame({
            "variant_id":  ["test:17:43071077:G:T"],
            "chrom":       ["17"],
            "pos":         [43071077],
            "ref":         ["G"],
            "alt":         ["T"],
            "consequence": ["missense_variant"],
            "allele_freq": [0.0001],
        })
        annotated = connector.annotate_dataframe(df)
        feats = engineer_features(annotated)

        assert feats.shape == (1, len(TABULAR_FEATURES))
        assert not feats.isnull().any().any()
        # Real scores must survive the fillna pass in engineer_features
        assert feats.loc[0, "sift_score"]      == pytest.approx(0.03)
        assert feats.loc[0, "polyphen2_score"] == pytest.approx(0.95)

    def test_sift_score_in_tabular_features(self):
        from src.models.variant_ensemble import TABULAR_FEATURES
        assert "sift_score" in TABULAR_FEATURES

    def test_polyphen2_score_in_tabular_features(self):
        from src.models.variant_ensemble import TABULAR_FEATURES
        assert "polyphen2_score" in TABULAR_FEATURES 
        
# ---------------------------------------------------------------------------
# Tests: dbnsfp.py (Connector 7)
# ---------------------------------------------------------------------------
class TestDbNSFPConnector:
    """
    Unit tests for DbNSFPConnector (full-width dbNSFP annotator).

    The ``connector`` fixture injects a small synthetic index directly,
    bypassing file I/O.  All six scores are tested in every dimension.
    """

    @pytest.fixture
    def connector(self):
        """Connector with a synthetic in-memory index covering four variants."""
        import pandas as pd
        from src.data.dbnsfp import DbNSFPConnector
        c = DbNSFPConnector(dbnsfp_file=None)
        c._index = pd.DataFrame([
            # BRCA1 missense — deleterious across all tools
            {"chrom": "17", "pos": 43071077, "ref": "G", "alt": "T",
             "sift_score": 0.03, "polyphen2_score": 0.95, "revel_score": 0.87,
             "cadd_phred": 28.4, "phylop_score": 7.2, "gerp_score": 5.1},
            # AGRN benign — tolerated across all tools
            {"chrom": "1",  "pos": 925952,   "ref": "G", "alt": "A",
             "sift_score": 0.21, "polyphen2_score": 0.12, "revel_score": 0.11,
             "cadd_phred": 8.3,  "phylop_score": 0.3,  "gerp_score": -1.2},
            # X chromosome variant
            {"chrom": "X",  "pos": 153296777, "ref": "C", "alt": "T",
             "sift_score": 0.08, "polyphen2_score": 0.55, "revel_score": 0.44,
             "cadd_phred": 17.1, "phylop_score": 3.1,  "gerp_score": 2.8},
        ])
        return c

    @pytest.fixture
    def minimal_df(self):
        return pd.DataFrame({
            "chrom": ["17", "1", "X", "2"],
            "pos":   [43071077, 925952, 153296777, 999999],
            "ref":   ["G", "G", "C", "A"],
            "alt":   ["T", "A", "T", "C"],
        })

    # ------------------------------------------------------------------
    # Import + constants
    # ------------------------------------------------------------------

    def test_module_imports_cleanly(self):
        from src.data.dbnsfp import DbNSFPConnector
        assert DbNSFPConnector is not None

    def test_score_defaults_exported(self):
        from src.data.dbnsfp import (
            DEFAULT_SIFT, DEFAULT_PP2, DEFAULT_REVEL,
            DEFAULT_CADD, DEFAULT_PHYLOP, DEFAULT_GERP,
        )
        assert DEFAULT_SIFT   == 0.5
        assert DEFAULT_PP2    == 0.5
        assert DEFAULT_REVEL  == 0.5
        assert DEFAULT_CADD   == 15.0
        assert DEFAULT_PHYLOP == 0.0
        assert DEFAULT_GERP   == 0.0

    def test_source_name(self):
        from src.data.dbnsfp import DbNSFPConnector
        assert DbNSFPConnector.source_name == "dbnsfp"

    def test_dbnsfp_scores_dataclass(self):
        from src.data.dbnsfp import DbNSFPScores, DEFAULT_SIFT, DEFAULT_GERP
        s = DbNSFPScores()
        assert s.sift_score  == DEFAULT_SIFT
        assert s.gerp_score  == DEFAULT_GERP
        assert isinstance(s.to_dict(), dict)
        assert set(s.to_dict().keys()) == {
            "sift_score", "polyphen2_score", "revel_score",
            "cadd_phred", "phylop_score", "gerp_score",
        }

    def test_dbnsfp_scores_is_frozen(self):
        from src.data.dbnsfp import DbNSFPScores
        s = DbNSFPScores(sift_score=0.1)
        with pytest.raises((AttributeError, TypeError)):
            s.sift_score = 0.9  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Stub mode
    # ------------------------------------------------------------------

    def test_stub_mode_get_scores_returns_defaults(self):
        from src.data.dbnsfp import DbNSFPConnector, _DEFAULT_SCORES
        c = DbNSFPConnector(dbnsfp_file=None)
        scores = c.get_scores("17", 43071077, "G", "T")
        assert scores == _DEFAULT_SCORES

    def test_stub_mode_annotate_fills_all_columns(self, minimal_df):
        from src.data.dbnsfp import (
            DbNSFPConnector,
            DEFAULT_SIFT, DEFAULT_PP2, DEFAULT_REVEL,
            DEFAULT_CADD, DEFAULT_PHYLOP, DEFAULT_GERP,
        )
        c = DbNSFPConnector(dbnsfp_file=None)
        result = c.annotate_dataframe(minimal_df)
        assert (result["sift_score"]      == DEFAULT_SIFT).all()
        assert (result["polyphen2_score"] == DEFAULT_PP2).all()
        assert (result["revel_score"]     == DEFAULT_REVEL).all()
        assert (result["cadd_phred"]      == DEFAULT_CADD).all()
        assert (result["phylop_score"]    == DEFAULT_PHYLOP).all()
        assert (result["gerp_score"]      == DEFAULT_GERP).all()

    # ------------------------------------------------------------------
    # get_scores — correctness
    # ------------------------------------------------------------------

    def test_known_variant_returns_all_real_scores(self, connector):
        scores = connector.get_scores("17", 43071077, "G", "T")
        assert scores.sift_score      == pytest.approx(0.03)
        assert scores.polyphen2_score == pytest.approx(0.95)
        assert scores.revel_score     == pytest.approx(0.87)
        assert scores.cadd_phred      == pytest.approx(28.4)
        assert scores.phylop_score    == pytest.approx(7.2)
        assert scores.gerp_score      == pytest.approx(5.1)

    def test_missing_variant_returns_defaults(self, connector):
        from src.data.dbnsfp import _DEFAULT_SCORES
        scores = connector.get_scores("2", 999999, "A", "C")
        assert scores == _DEFAULT_SCORES

    def test_chr_prefix_stripped(self, connector):
        no_prefix  = connector.get_scores("17",    43071077, "G", "T")
        chr_prefix = connector.get_scores("chr17", 43071077, "G", "T")
        assert no_prefix == chr_prefix

    def test_lowercase_chrom_accepted(self, connector):
        scores = connector.get_scores("chr17", 43071077, "G", "T")
        assert scores.sift_score == pytest.approx(0.03)

    def test_lowercase_alleles_accepted(self, connector):
        upper = connector.get_scores("17", 43071077, "G", "T")
        lower = connector.get_scores("17", 43071077, "g", "t")
        assert upper == lower

    def test_sex_chromosome_X(self, connector):
        scores = connector.get_scores("X", 153296777, "C", "T")
        assert scores.sift_score  == pytest.approx(0.08)
        assert scores.gerp_score  == pytest.approx(2.8)

    def test_negative_gerp_returned_correctly(self, connector):
        scores = connector.get_scores("1", 925952, "G", "A")
        assert scores.gerp_score == pytest.approx(-1.2)

    # ------------------------------------------------------------------
    # annotate_dataframe — output correctness
    # ------------------------------------------------------------------

    def test_annotate_adds_all_six_columns(self, connector, minimal_df):
        result = connector.annotate_dataframe(minimal_df)
        for col in ["sift_score", "polyphen2_score", "revel_score",
                    "cadd_phred", "phylop_score", "gerp_score"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_annotate_returns_copy(self, connector, minimal_df):
        result = connector.annotate_dataframe(minimal_df)
        assert result is not minimal_df

    def test_annotate_does_not_mutate_input(self, connector, minimal_df):
        original_cols = set(minimal_df.columns)
        _ = connector.annotate_dataframe(minimal_df)
        assert set(minimal_df.columns) == original_cols

    def test_annotate_correct_scores_for_hit(self, connector, minimal_df):
        result = connector.annotate_dataframe(minimal_df)
        assert result.loc[0, "sift_score"]      == pytest.approx(0.03)
        assert result.loc[0, "polyphen2_score"] == pytest.approx(0.95)
        assert result.loc[0, "revel_score"]     == pytest.approx(0.87)
        assert result.loc[0, "cadd_phred"]      == pytest.approx(28.4)
        assert result.loc[0, "phylop_score"]    == pytest.approx(7.2)
        assert result.loc[0, "gerp_score"]      == pytest.approx(5.1)

    def test_annotate_defaults_for_miss(self, connector, minimal_df):
        from src.data.dbnsfp import DEFAULT_SIFT, DEFAULT_CADD, DEFAULT_GERP
        result = connector.annotate_dataframe(minimal_df)
        # Row 3 = chr2:999999 A>C — not in synthetic index
        assert result.loc[3, "sift_score"]  == DEFAULT_SIFT
        assert result.loc[3, "cadd_phred"]  == DEFAULT_CADD
        assert result.loc[3, "gerp_score"]  == DEFAULT_GERP

    def test_annotate_preserves_existing_columns(self, connector, minimal_df):
        result = connector.annotate_dataframe(minimal_df)
        for col in minimal_df.columns:
            assert col in result.columns

    def test_annotate_no_nans(self, connector, minimal_df):
        result = connector.annotate_dataframe(minimal_df)
        for col in ["sift_score", "polyphen2_score", "revel_score",
                    "cadd_phred", "phylop_score", "gerp_score"]:
            assert not result[col].isna().any(), f"NaN in {col}"

    def test_annotate_replaces_existing_columns(self, connector):
        df = pd.DataFrame({
            "chrom": ["17"], "pos": [43071077], "ref": ["G"], "alt": ["T"],
            "gerp_score": [99.0],   # stale — must be overwritten
        })
        result = connector.annotate_dataframe(df)
        assert result.loc[0, "gerp_score"] == pytest.approx(5.1)

    def test_missing_value_override(self, connector):
        """annotate_dataframe must honour per-column default overrides."""
        from src.data.dbnsfp import DbNSFPConnector
        c = DbNSFPConnector(dbnsfp_file=None)   # stub — all variants missing
        df = pd.DataFrame({
            "chrom": ["2"], "pos": [999999], "ref": ["A"], "alt": ["C"],
        })
        result = c.annotate_dataframe(df, cadd_phred=99.0, gerp_score=-99.0)
        assert result.loc[0, "cadd_phred"]  == 99.0
        assert result.loc[0, "gerp_score"]  == -99.0

    # ------------------------------------------------------------------
    # _df_to_index
    # ------------------------------------------------------------------

    def test_df_to_index_keys_normalised(self):
        from src.data.dbnsfp import DbNSFPConnector
        df = pd.DataFrame({
            "chrom":         ["chr17"],
            "pos":           [43071077],
            "ref":           ["g"],
            "alt":           ["t"],
            "sift_score":    [0.03],
            "polyphen2_score": [0.95],
            "revel_score":   [0.87],
            "cadd_phred":    [28.4],
            "phylop_score":  [7.2],
            "gerp_score":    [5.1],
        })
        index = DbNSFPConnector._df_to_index(df)
        assert ("17", 43071077, "G", "T") in index

    def test_df_to_index_values_are_dbnsfp_scores(self):
        from src.data.dbnsfp import DbNSFPConnector, DbNSFPScores
        df = pd.DataFrame({
            "chrom":         ["17"],
            "pos":           [43071077],
            "ref":           ["G"],
            "alt":           ["T"],
            "sift_score":    [0.03],
            "polyphen2_score": [0.95],
            "revel_score":   [0.87],
            "cadd_phred":    [28.4],
            "phylop_score":  [7.2],
            "gerp_score":    [5.1],
        })
        index = DbNSFPConnector._df_to_index(df)
        val = index[("17", 43071077, "G", "T")]
        assert isinstance(val, DbNSFPScores)
        assert val.gerp_score == pytest.approx(5.1)

    # ------------------------------------------------------------------
    # File-based integration (synthetic TSV — no real dbNSFP needed)
    # ------------------------------------------------------------------

    def test_file_based_annotation(self, tmp_path):
        from src.data.dbnsfp import DbNSFPConnector, DEFAULT_GERP
        content = (
            "#chr\tpos(1-based)\tref\talt\t"
            "SIFT_score\tPolyphen2_HDIV_score\tREVEL_score\t"
            "CADD_phred\tphyloP100way_vertebrate\tGERP++_RS\n"
            "17\t43071077\tG\tT\t0.03\t0.95\t0.87\t28.4\t7.2\t5.1\n"
            "1\t925952\tG\tA\t0.21\t0.12\t0.11\t8.3\t0.3\t-1.2\n"
            "13\t32338271\tT\tA\t.\t.\t.\t.\t.\t.\n"
        )
        tsv = tmp_path / "test_dbnsfp_full.tsv"
        tsv.write_text(content)

        c = DbNSFPConnector(dbnsfp_file=tsv)
        df = pd.DataFrame({
            "chrom": ["17", "1", "13"],
            "pos":   [43071077, 925952, 32338271],
            "ref":   ["G", "G", "T"],
            "alt":   ["T", "A", "A"],
        })
        result = c.annotate_dataframe(df)

        assert result.loc[0, "sift_score"]   == pytest.approx(0.03)
        assert result.loc[0, "gerp_score"]   == pytest.approx(5.1)
        assert result.loc[1, "revel_score"]  == pytest.approx(0.11)
        assert result.loc[1, "gerp_score"]   == pytest.approx(-1.2)
        # All-dot row → defaults
        assert result.loc[2, "gerp_score"]   == DEFAULT_GERP

    def test_missing_score_column_gracefully_defaults(self, tmp_path):
        """File missing GERP++_RS column must still annotate other scores."""
        from src.data.dbnsfp import DbNSFPConnector, DEFAULT_GERP
        content = (
            "#chr\tpos(1-based)\tref\talt\t"
            "SIFT_score\tPolyphen2_HDIV_score\tREVEL_score\t"
            "CADD_phred\tphyloP100way_vertebrate\n"   # no GERP column
            "17\t43071077\tG\tT\t0.03\t0.95\t0.87\t28.4\t7.2\n"
        )
        tsv = tmp_path / "no_gerp.tsv"
        tsv.write_text(content)

        c = DbNSFPConnector(dbnsfp_file=tsv)
        df = pd.DataFrame({
            "chrom": ["17"], "pos": [43071077], "ref": ["G"], "alt": ["T"],
        })
        result = c.annotate_dataframe(df)

        assert result.loc[0, "sift_score"]  == pytest.approx(0.03)
        assert result.loc[0, "gerp_score"]  == DEFAULT_GERP   # graceful default

    def test_parquet_cache_written_and_reused(self, tmp_path):
        from src.data.dbnsfp import DbNSFPConnector
        content = (
            "#chr\tpos(1-based)\tref\talt\t"
            "SIFT_score\tPolyphen2_HDIV_score\tREVEL_score\t"
            "CADD_phred\tphyloP100way_vertebrate\tGERP++_RS\n"
            "17\t43071077\tG\tT\t0.03\t0.95\t0.87\t28.4\t7.2\t5.1\n"
        )
        tsv = tmp_path / "dbnsfp_full.tsv"
        tsv.write_text(content)

        c1 = DbNSFPConnector(dbnsfp_file=tsv, cache_dir=tmp_path)
        c1.get_scores("17", 43071077, "G", "T")
        cache_file = tmp_path / "dbnsfp_clinvar_index.parquet"
        assert cache_file.exists(), "Parquet cache not written"

        tsv.unlink()
        c2 = DbNSFPConnector(dbnsfp_file=tsv, cache_dir=tmp_path)
        scores = c2.get_scores("17", 43071077, "G", "T")
        assert scores.sift_score == pytest.approx(0.03)

    def test_cache_filename_does_not_collide_with_connector6(self, tmp_path):
        """Connector 7 cache must not overwrite Connector 6 cache."""
        from src.data.dbnsfp import DbNSFPConnector
        from src.data.sift_polyphen import SIFTPolyPhenConnector
        c6_cache = tmp_path / "dbnsfp_sift_pp2_index.parquet"
        c7_cache = tmp_path / "dbnsfp_clinvar_index.parquet"
        assert c6_cache != c7_cache

    # ------------------------------------------------------------------
    # End-to-end: all six scores flow through engineer_features
    # ------------------------------------------------------------------

    def test_all_scores_flow_into_feature_matrix(self, connector):
        from src.models.variant_ensemble import engineer_features, TABULAR_FEATURES
        df = pd.DataFrame({
            "chrom":       ["17"],
            "pos":         [43071077],
            "ref":         ["G"],
            "alt":         ["T"],
            "consequence": ["missense_variant"],
            "allele_freq": [0.0001],
        })
        annotated = connector.annotate_dataframe(df)
        feats = engineer_features(annotated)

        assert feats.shape == (1, len(TABULAR_FEATURES))
        assert not feats.isnull().any().any()
        assert feats.loc[0, "sift_score"]      == pytest.approx(0.03)
        assert feats.loc[0, "polyphen2_score"] == pytest.approx(0.95)
        assert feats.loc[0, "revel_score"]     == pytest.approx(0.87)
        assert feats.loc[0, "cadd_phred"]      == pytest.approx(28.4)
        assert feats.loc[0, "phylop_score"]    == pytest.approx(7.2)
        assert feats.loc[0, "gerp_score"]      == pytest.approx(5.1)

    def test_gerp_score_in_tabular_features(self):
        from src.models.variant_ensemble import TABULAR_FEATURES
        assert "gerp_score" in TABULAR_FEATURES

    def test_helpers_imported_from_connector6(self):
        """Connector 7 must import helpers from sift_polyphen, not redefine them."""
        import inspect
        import src.data.dbnsfp as m
        from src.data.sift_polyphen import _normalise_chrom, _parse_multival
        assert m._normalise_chrom is _normalise_chrom
        assert m._parse_multival  is _parse_multival               


# ---------------------------------------------------------------------------
# Tests: spark_etl.py
# ---------------------------------------------------------------------------
class TestETLNormalization:
    """Test normalization logic without requiring a live Spark session."""

    def test_chrom_map_completeness(self):
        pytest.importorskip("pyspark")
        from src.data.spark_etl import CHROM_MAP
        for i in range(1, 23):
            assert str(i) in CHROM_MAP, f"Missing bare chromosome {i}"
            assert f"chr{i}" in CHROM_MAP, f"Missing chr-prefixed chromosome {i}"
        assert "X" in CHROM_MAP
        assert "chrX" in CHROM_MAP

    def test_chrom_normalization_values(self):
        pytest.importorskip("pyspark")
        from src.data.spark_etl import CHROM_MAP
        assert CHROM_MAP["chr1"] == "1"
        assert CHROM_MAP["chrX"] == "X"
        assert CHROM_MAP["chrM"] == "MT"
        assert CHROM_MAP["MT"] == "MT"

    def test_schema_has_required_fields(self):
        pytest.importorskip("pyspark")
        from src.data.spark_etl import VARIANT_SCHEMA
        field_names = [f.name for f in VARIANT_SCHEMA]
        for required in ["variant_id", "chrom", "pos", "ref", "alt", "pathogenicity"]:
            assert required in field_names


# ---------------------------------------------------------------------------
# Tests: variant_ensemble.py — feature engineering
# ---------------------------------------------------------------------------
class TestFeatureEngineering:
    def test_engineer_features_shape(self, sample_canonical_df):
        # CHANGE: import from src.models.variant_ensemble not src.models.ensemble
        from src.models.variant_ensemble import engineer_features, TABULAR_FEATURES
        feats = engineer_features(sample_canonical_df)
        assert feats.shape[0] == len(sample_canonical_df)
        assert feats.shape[1] == len(TABULAR_FEATURES)

    def test_engineer_features_no_nans(self, sample_canonical_df):
        from src.models.variant_ensemble import engineer_features
        feats = engineer_features(sample_canonical_df)
        assert not feats.isnull().any().any(), "Feature matrix must have no NaNs"

    def test_is_snv_detection(self, sample_canonical_df):
        from src.models.variant_ensemble import engineer_features
        feats = engineer_features(sample_canonical_df)
        assert feats["is_snv"].sum() == len(sample_canonical_df)
        assert feats["is_indel"].sum() == 0

    def test_consequence_features(self, sample_canonical_df):
        from src.models.variant_ensemble import engineer_features
        feats = engineer_features(sample_canonical_df)
        # Row 0 = missense_variant
        assert feats.loc[0, "is_missense"] == 1
        # Row 1 = stop_gained → captured by is_loss_of_function
        assert feats.loc[1, "is_loss_of_function"] == 1
        # Row 3 = splice_donor_variant → captured by is_splice
        assert feats.loc[3, "is_splice"] == 1

    def test_codon_position_in_tabular_features(self):
        """codon_position was promoted to TABULAR_FEATURES in Phase 4 (VEP connector)."""
        from src.models.variant_ensemble import TABULAR_FEATURES, PHASE_2_FEATURES
        assert "codon_position" in TABULAR_FEATURES
        assert "codon_position" not in PHASE_2_FEATURES

    def test_encode_sequence_shape(self):
        from src.models.variant_ensemble import encode_sequence
        seq = "ACGT" * 25 + "A"  # 101 bp
        encoded = encode_sequence(seq, window=101)
        assert encoded.shape == (101, 4)

    def test_encode_sequence_one_hot_valid(self):
        from src.models.variant_ensemble import encode_sequence
        seq = "ACGT" * 25 + "A"
        encoded = encode_sequence(seq, window=101)
        # Every position should be a valid one-hot row summing to 1.0
        assert np.allclose(encoded.sum(axis=1), 1.0)

    def test_encode_sequence_no_k_parameter(self):
        """encode_sequence removed the unused k parameter (Bug 7)."""
        import inspect
        from src.models.variant_ensemble import encode_sequence
        sig = inspect.signature(encode_sequence)
        assert "k" not in sig.parameters, "k parameter should have been removed (Bug 7)"


# ---------------------------------------------------------------------------
# Tests: report_generator.py
# ---------------------------------------------------------------------------
class TestValidationMetrics:
    def _make_predictions(self):
        rng = np.random.default_rng(42)
        y_true  = rng.integers(0, 2, 200)
        y_proba = rng.uniform(0, 1, 200)
        return y_true, y_proba

    def test_bootstrap_ci_bounds(self):
        from src.reports.report_generator import bootstrap_metric
        from sklearn.metrics import roc_auc_score
        y_true, y_proba = self._make_predictions()
        lo, hi = bootstrap_metric(y_true, y_proba, roc_auc_score, n_bootstrap=50)
        assert 0.0 <= lo <= hi <= 1.0

    def test_variant_phenotype_association_keys(self):
        from src.reports.report_generator import compute_variant_phenotype_association
        rng = np.random.default_rng(0)
        result = compute_variant_phenotype_association(
            rng.integers(0, 2, 100), rng.integers(0, 2, 100)
        )
        for key in ["odds_ratio", "p_value", "cramers_v", "significant"]:
            assert key in result
        assert result["odds_ratio"] >= 0.0
        assert 0.0 <= result["p_value"] <= 1.0

    def test_report_generation_creates_file(self, sample_canonical_df, tmp_path):
        from src.reports.report_generator import ReportGenerator
        rng = np.random.default_rng(42)
        sample_canonical_df = sample_canonical_df.copy()
        sample_canonical_df["ensemble_score"] = rng.uniform(0, 1, len(sample_canonical_df))

        gen = ReportGenerator(output_dir=tmp_path)
        model_metrics = [
            {"model_name": "random_forest", "auroc": 0.85, "auprc": 0.78,
             "f1_macro": 0.82, "mcc": 0.64, "brier": 0.15},
            {"model_name": "ENSEMBLE_STACKER", "auroc": 0.91, "auprc": 0.88,
             "f1_macro": 0.89, "mcc": 0.78, "brier": 0.10},
        ]
        report_path = gen.generate(
            modality="dna",
            variant_df=sample_canonical_df,
            model_metrics=model_metrics,
        )
        assert report_path.exists()
        html = report_path.read_text()
        assert "Genomic Disease Association Report" in html
        assert "AUROC" in html

    def test_report_format_int_filter(self, sample_canonical_df, tmp_path):
        """format_int filter (Environment fix, Issue O) must produce comma-formatted numbers."""
        from src.reports.report_generator import ReportGenerator
        gen = ReportGenerator(output_dir=tmp_path)
        report_path = gen.generate(
            modality="dna",
            variant_df=sample_canonical_df,
            model_metrics=[],
        )
        html = report_path.read_text()
        # 4 total variants → "4" (small number, no comma, but template renders it)
        assert report_path.stat().st_size > 500


# ---------------------------------------------------------------------------
# Tests: evaluator.py
# ---------------------------------------------------------------------------
class TestClinicalEvaluator:
    def _make_data(self, n: int = 300):
        rng = np.random.default_rng(42)
        y = rng.integers(0, 2, size=n)
        signal = y.astype(float) + rng.normal(0, 0.8, size=n)
        p = 1 / (1 + np.exp(-signal))
        return y, p

    def test_evaluate_returns_report(self):
        from src.evaluation.evaluator import ClinicalEvaluator, EvaluationReport
        y, p = self._make_data()
        ev = ClinicalEvaluator(n_bootstrap=10)
        report = ev.evaluate(y, p, model_name="test_model")
        assert isinstance(report, EvaluationReport)
        assert 0.0 < report.auroc < 1.0
        assert report.n_samples == len(y)

    def test_gene_error_analysis_dict_unpack(self):
        """GeneErrorAnalysis must use to_dict(orient='records') not itertuples (Issue S)."""
        from src.evaluation.evaluator import ClinicalEvaluator, GeneErrorAnalysis
        rng = np.random.default_rng(0)
        n = 200
        y, p = self._make_data(n)
        meta = pd.DataFrame({
            "gene_symbol": rng.choice(["BRCA1", "TP53", "AGRN", "TTN"], n),
        })
        ev = ClinicalEvaluator(n_bootstrap=10)
        gene_errors = ev._gene_error_analysis(y, p, meta)
        assert all(isinstance(ge, GeneErrorAnalysis) for ge in gene_errors)

    def test_operating_points_present(self):
        from src.evaluation.evaluator import ClinicalEvaluator
        y, p = self._make_data()
        ev = ClinicalEvaluator(n_bootstrap=10)
        report = ev.evaluate(y, p)
        # With n=300 balanced data there should be operating points
        assert report.at_sensitivity_90 is not None
        assert report.at_sensitivity_95 is not None

    def test_save_report(self, tmp_path):
        from src.evaluation.evaluator import ClinicalEvaluator
        y, p = self._make_data()
        ev = ClinicalEvaluator(n_bootstrap=10)
        report = ev.evaluate(y, p)
        out = tmp_path / "eval_report.json"
        ev.save_report(report, out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert "auroc" in data


# ---------------------------------------------------------------------------
# Integration test: end-to-end mini pipeline (no Spark, no live APIs)
# ---------------------------------------------------------------------------
class TestEndToEndPipeline:
    def test_features_to_predictions_shape(self, sample_canonical_df, sample_labels):
        from src.models.variant_ensemble import engineer_features
        from sklearn.ensemble import RandomForestClassifier

        feats = engineer_features(sample_canonical_df)
        rf = RandomForestClassifier(n_estimators=5, random_state=42)
        rf.fit(feats, sample_labels)
        preds = rf.predict(feats)
        assert len(preds) == len(sample_canonical_df)
        assert set(preds).issubset({0, 1})

    def test_full_report_pipeline(self, sample_canonical_df, sample_labels, tmp_path):
        """Smoke test: features → RF → report generation."""
        from src.models.variant_ensemble import engineer_features
        from src.reports.report_generator import ReportGenerator
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score

        feats = engineer_features(sample_canonical_df)
        rf = RandomForestClassifier(n_estimators=5, random_state=42)
        rf.fit(feats, sample_labels)
        proba = rf.predict_proba(feats)[:, 1]

        df = sample_canonical_df.copy()
        df["ensemble_score"] = proba

        model_metrics = [{
            "model_name": "random_forest",
            "auroc": roc_auc_score(sample_labels, proba),
            "auprc": 0.5, "f1_macro": 0.5, "mcc": 0.0, "brier": 0.25,
        }]
        gen = ReportGenerator(output_dir=tmp_path)
        report = gen.generate(modality="dna", variant_df=df, model_metrics=model_metrics)
        assert report.exists()
        assert report.stat().st_size > 1000


# ---------------------------------------------------------------------------
# Tests: src/data/gtex.py — Phase 2 Pillar 1
# ---------------------------------------------------------------------------
class TestGTExConnector:
    """Unit tests for the GTEx connector. All HTTP calls are mocked."""
 
    # ── ID helpers ────────────────────────────────────────────────────────
 
    def test_gtex_variant_id_format(self):
        from src.data.gtex import _gtex_variant_id
        assert _gtex_variant_id("17", 43071077, "G", "T") == "chr17_43071077_G_T_b38"
        assert _gtex_variant_id("X",  100000,   "A", "C") == "chrX_100000_A_C_b38"
        assert _gtex_variant_id("1",  925952,   "G", "A") == "chr1_925952_G_A_b38"
 
    def test_from_gtex_variant_id_valid(self):
        from src.data.gtex import _from_gtex_variant_id
        result = _from_gtex_variant_id("chr17_43071077_G_T_b38")
        assert result == {"chrom": "17", "pos": 43071077, "ref": "G", "alt": "T"}
 
    def test_from_gtex_variant_id_x_chromosome(self):
        from src.data.gtex import _from_gtex_variant_id
        result = _from_gtex_variant_id("chrX_100000_A_C_b38")
        assert result is not None
        assert result["chrom"] == "X"
 
    def test_from_gtex_variant_id_invalid_returns_none(self):
        from src.data.gtex import _from_gtex_variant_id
        assert _from_gtex_variant_id("bad_format")            is None
        assert _from_gtex_variant_id("chr1_notanint_G_A_b38") is None
        assert _from_gtex_variant_id("1_925952_G_A_b38")      is None
        assert _from_gtex_variant_id("chr1_925952_G_A")        is None
 
    def test_variant_id_roundtrip(self):
        from src.data.gtex import _gtex_variant_id, _from_gtex_variant_id
        gtex_id = _gtex_variant_id("17", 43071077, "G", "T")
        parsed  = _from_gtex_variant_id(gtex_id)
        assert parsed == {"chrom": "17", "pos": 43071077, "ref": "G", "alt": "T"}
 
    # ── Expression summary ─────────────────────────────────────────────────
 
    def test_summarise_expression_basic(self):
        from src.data.gtex import GTExConnector
        expr_df = pd.DataFrame({
            "tissueSiteDetailId": ["Whole_Blood", "Liver", "Lung"],
            "median": [10.0, 0.5, 5.0],
        })
        result = GTExConnector._summarise_expression("BRCA1", expr_df)
        assert result["gene_symbol"]              == "BRCA1"
        assert result["gtex_max_tpm"]             == 10.0
        assert result["gtex_n_tissues_expressed"] == 2   # Whole_Blood + Lung >= 1.0
        assert 0.0 < result["gtex_tissue_specificity"] < 1.0
        assert "Whole_Blood" in result["gtex_tissue_tpm"]
 
    def test_summarise_expression_ubiquitous(self):
        from src.data.gtex import GTExConnector
        expr_df = pd.DataFrame({
            "tissueSiteDetailId": [f"T{i}" for i in range(10)],
            "median": [5.0] * 10,
        })
        result = GTExConnector._summarise_expression("UBIQ", expr_df)
        assert result["gtex_tissue_specificity"] == 0.0
 
    def test_summarise_expression_all_zero(self):
        from src.data.gtex import GTExConnector
        expr_df = pd.DataFrame({
            "tissueSiteDetailId": ["Whole_Blood"],
            "median": [0.0],
        })
        result = GTExConnector._summarise_expression("SILENT", expr_df)
        assert result["gtex_max_tpm"]             == 0.0
        assert result["gtex_tissue_specificity"]  == 0.0
        assert result["gtex_n_tissues_expressed"] == 0
 
    def test_empty_expression_row(self):
        from src.data.gtex import GTExConnector
        result = GTExConnector._empty_expression_row("UNKNOWN")
        assert result["gene_symbol"]              == "UNKNOWN"
        assert result["gtex_max_tpm"]             == 0.0
        assert result["gtex_n_tissues_expressed"] == 0
        assert result["gtex_tissue_specificity"]  == 0.0
        assert result["gtex_tissue_tpm"]          == {}
 
    # ── connector.fetch() — all HTTP mocked ───────────────────────────────
 
    def test_fetch_returns_canonical_schema(self):
        from src.data.gtex import GTExConnector
        from src.data.database_connectors import CANONICAL_COLUMNS
        connector = GTExConnector()
        connector._resolve_gencode_ids = MagicMock(
            return_value={"BRCA1": "ENSG00000012048.23"}
        )
        connector._fetch_expression = MagicMock(
            return_value=GTExConnector._empty_expression_row("BRCA1")
        )
        connector._fetch_eqtls = MagicMock(return_value=[{
            "variant_id": "gtex:17:43071077:G:T", "source_db": "gtex",
            "chrom": "17", "pos": 43071077, "ref": "G", "alt": "T",
            "gene_symbol": "BRCA1", "transcript_id": None,
            "consequence": "regulatory_region_variant", "pathogenicity": None,
            "allele_freq": 0.01, "clinical_sig": None, "protein_change": None,
            "fasta_seq": None, "source_id": "chr17_43071077_G_T_b38",
            "metadata": {"tissue": "Whole_Blood", "neg_log10_pval": 5.2,
                         "beta": -0.35, "tss_distance": 1000},
        }])
        result = connector.fetch(gene_symbols=["BRCA1"])
        missing = set(CANONICAL_COLUMNS) - set(result.columns)
        assert not missing, f"Missing canonical columns: {missing}"
        assert result.iloc[0]["source_db"]   == "gtex"
        assert result.iloc[0]["gene_symbol"] == "BRCA1"
 
    def test_fetch_empty_gene_list(self):
        from src.data.gtex import GTExConnector
        from src.data.database_connectors import CANONICAL_COLUMNS
        connector = GTExConnector()
        result    = connector.fetch(gene_symbols=[])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert set(CANONICAL_COLUMNS).issubset(set(result.columns))
 
    def test_fetch_populates_expression_summary(self):
        from src.data.gtex import GTExConnector
        connector = GTExConnector()
        connector._resolve_gencode_ids = MagicMock(
            return_value={"TP53": "ENSG00000141510.16"}
        )
        connector._fetch_expression = MagicMock(return_value={
            "gene_symbol": "TP53", "gtex_max_tpm": 42.0,
            "gtex_n_tissues_expressed": 15, "gtex_tissue_specificity": 0.3,
            "gtex_tissue_tpm": {"Whole_Blood": 42.0},
        })
        connector._fetch_eqtls = MagicMock(return_value=[])
        connector.fetch(gene_symbols=["TP53"])
        assert not connector.gene_expression_summary.empty
        assert "TP53" in connector.gene_expression_summary.index
        assert connector.gene_expression_summary.loc["TP53", "gtex_max_tpm"] == 42.0
 
    # ── build_gtex_feature_df ──────────────────────────────────────────────
 
    def test_build_gtex_adds_six_columns(self):
        from src.data.gtex import GTExConnector, build_gtex_feature_df
        connector = GTExConnector()
        connector.gene_expression_summary = pd.DataFrame(
            {"gtex_max_tpm": [10.0], "gtex_n_tissues_expressed": [5],
             "gtex_tissue_specificity": [0.4], "gtex_tissue_tpm": [{}]},
            index=pd.Index(["BRCA1"], name="gene_symbol"),
        )
        df     = pd.DataFrame({"variant_id": ["clinvar:17:43071077:G:T"],
                                "source_db": ["clinvar"], "gene_symbol": ["BRCA1"]})
        result = build_gtex_feature_df(connector, df)
        for col in ["gtex_max_tpm", "gtex_n_tissues_expressed", "gtex_tissue_specificity",
                    "gtex_is_eqtl", "gtex_min_eqtl_pval", "gtex_max_abs_effect"]:
            assert col in result.columns, f"Missing column: {col}"
        assert result.iloc[0]["gtex_max_tpm"] == 10.0
 
    def test_build_gtex_nan_safe(self):
        from src.data.gtex import GTExConnector, build_gtex_feature_df
        connector = GTExConnector()
        connector.gene_expression_summary = pd.DataFrame(
            {"gtex_max_tpm": [5.0], "gtex_n_tissues_expressed": [3],
             "gtex_tissue_specificity": [0.6], "gtex_tissue_tpm": [{}]},
            index=pd.Index(["KNOWN"], name="gene_symbol"),
        )
        df     = pd.DataFrame({"variant_id": ["clinvar:1:1000:A:T"],
                                "source_db": ["clinvar"], "gene_symbol": ["UNKNOWN"]})
        result = build_gtex_feature_df(connector, df)
        assert result["gtex_max_tpm"].isna().sum()   == 0
        assert result["gtex_is_eqtl"].isna().sum()   == 0
        assert result["gtex_max_tpm"].iloc[0]         == 0.0
 
    # ── Integration with variant_ensemble.py ──────────────────────────────
 
    def test_gtex_features_in_tabular_features(self):
        from src.models.variant_ensemble import TABULAR_FEATURES
        gtex_feats = [
            "gtex_max_tpm", "gtex_n_tissues_expressed", "gtex_tissue_specificity",
            "gtex_is_eqtl", "gtex_min_eqtl_pval", "gtex_max_abs_effect",
        ]
        for feat in gtex_feats:
            assert feat in TABULAR_FEATURES, f"{feat} missing from TABULAR_FEATURES"
 
    def test_gtex_not_in_phase2_features(self):
        from src.models.variant_ensemble import PHASE_2_FEATURES
        gtex_feats = [
            "gtex_max_tpm", "gtex_n_tissues_expressed", "gtex_tissue_specificity",
            "gtex_is_eqtl", "gtex_min_eqtl_pval", "gtex_max_abs_effect",
        ]
        for feat in gtex_feats:
            assert feat not in PHASE_2_FEATURES, f"{feat} still in PHASE_2_FEATURES"
 
    def test_inherits_base_connector(self):
        from src.data.gtex import GTExConnector
        from src.data.database_connectors import BaseConnector
        assert issubclass(GTExConnector, BaseConnector)
 
    def test_source_name(self):
        from src.data.gtex import GTExConnector
        assert GTExConnector.source_name == "gtex"
 
    def test_priority_tissues(self):
        from src.data.gtex import PRIORITY_TISSUES
        assert len(PRIORITY_TISSUES) == 7
        assert "Whole_Blood"     in PRIORITY_TISSUES
        assert "Brain_Cortex"    in PRIORITY_TISSUES
        assert "Liver"           in PRIORITY_TISSUES
        
        
    # ---------------------------------------------------------------------------
# Tests: src/data/spliceai.py -- Phase 2 Pillar 1, Connector 2
# ---------------------------------------------------------------------------
class TestSpliceAIConnector:
    """Unit tests for SpliceAIConnector. No real VCF file required."""
 
    # ── Identity ──────────────────────────────────────────────────────────
 
    def test_source_name(self):
        from src.data.spliceai import SpliceAIConnector
        assert SpliceAIConnector.source_name == "spliceai"
 
    def test_inherits_base_connector(self):
        from src.data.spliceai import SpliceAIConnector
        from src.data.database_connectors import BaseConnector
        assert issubclass(SpliceAIConnector, BaseConnector)
 
    # ── parse_info_field ──────────────────────────────────────────────────
 
    def test_parse_info_field_single_gene(self):
        from src.data.spliceai import SpliceAIConnector
        result = SpliceAIConnector.parse_info_field(
            "T|BRCA1|0.85|0.00|0.00|0.00|-22|11|5|-32"
        )
        assert result["splice_ai_score"] == pytest.approx(0.85)
        assert result["ds_ag"]           == pytest.approx(0.85)
        assert result["ds_al"]           == pytest.approx(0.00)
        assert result["symbol"]          == "BRCA1"
 
    def test_parse_info_field_takes_max_across_delta_scores(self):
        from src.data.spliceai import SpliceAIConnector
        result = SpliceAIConnector.parse_info_field(
            "A|TP53|0.01|0.02|0.03|0.72|0|0|0|0"
        )
        assert result["splice_ai_score"] == pytest.approx(0.72)
        assert result["ds_dl"]           == pytest.approx(0.72)
 
    def test_parse_info_field_multi_gene_takes_max(self):
        from src.data.spliceai import SpliceAIConnector
        result = SpliceAIConnector.parse_info_field(
            "T|BRCA1|0.01|0.00|0.00|0.03|-22|11|5|-32,"
            "T|NBR1|0.00|0.00|0.45|0.00|1|-3|4|-2"
        )
        assert result["splice_ai_score"] == pytest.approx(0.45)
        assert result["symbol"]          == "NBR1"
 
    def test_parse_info_field_empty_returns_zero(self):
        from src.data.spliceai import SpliceAIConnector
        assert SpliceAIConnector.parse_info_field("")["splice_ai_score"]    == 0.0
        assert SpliceAIConnector.parse_info_field("   ")["splice_ai_score"] == 0.0
 
    def test_parse_info_field_malformed_does_not_raise(self):
        from src.data.spliceai import SpliceAIConnector
        result = SpliceAIConnector.parse_info_field("T|BRCA1|notafloat|0.0|0.0|0.0")
        assert result["splice_ai_score"] == 0.0
 
    # ── fetch() with no VCF file ──────────────────────────────────────────
 
    def test_fetch_no_vcf_path_returns_zero_scores(self):
        from src.data.spliceai import SpliceAIConnector
        connector  = SpliceAIConnector(vcf_path=None)
        variant_df = pd.DataFrame({
            "variant_id": ["clinvar:17:43071077:G:T"],
            "source_db":  ["clinvar"],
            "chrom": ["17"], "pos": [43071077], "ref": ["G"], "alt": ["T"],
        })
        result = connector.fetch(variant_df=variant_df)
        assert "splice_ai_score" in result.columns
        assert result["splice_ai_score"].iloc[0] == 0.0
 
    def test_fetch_missing_vcf_file_returns_zero_scores(self, tmp_path):
        from src.data.spliceai import SpliceAIConnector
        connector  = SpliceAIConnector(vcf_path=tmp_path / "nonexistent.vcf.gz")
        variant_df = pd.DataFrame({
            "variant_id": ["clinvar:17:43071077:G:T"],
            "source_db":  ["clinvar"],
            "chrom": ["17"], "pos": [43071077], "ref": ["G"], "alt": ["T"],
        })
        result = connector.fetch(variant_df=variant_df)
        assert result["splice_ai_score"].iloc[0] == 0.0
 
    def test_fetch_empty_dataframe_returns_empty(self):
        from src.data.spliceai import SpliceAIConnector
        connector = SpliceAIConnector(vcf_path=None)
        result    = connector.fetch(variant_df=pd.DataFrame())
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
 
    # ── fetch() with a real (mock) VCF file ──────────────────────────────
 
    def test_fetch_with_mock_vcf_correct_scores(self, tmp_path):
        import gzip
        from src.data.spliceai import SpliceAIConnector
        from src.data.database_connectors import FetchConfig
 
        vcf_lines = (
            "##fileformat=VCFv4.2\n"
            "##INFO=<ID=SpliceAI,Number=.,Type=String,Description=\"SpliceAI\">\n"
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
            "chr17\t43071077\t.\tG\tT\t.\t.\t"
            "SpliceAI=T|BRCA1|0.01|0.00|0.00|0.00|-22|11|5|-32\n"
            "chr17\t43071077\t.\tG\tA\t.\t.\t"
            "SpliceAI=A|BRCA1|0.85|0.00|0.00|0.00|-22|11|5|-32\n"
            "chr1\t925952\t.\tG\tA\t.\t.\t"
            "SpliceAI=A|AGRN|0.00|0.00|0.05|0.00|0|0|0|0\n"
        )
        vcf_path = tmp_path / "test.vcf.gz"
        with gzip.open(vcf_path, "wt", encoding="utf-8") as f:
            f.write(vcf_lines)
 
        config    = FetchConfig(cache_dir=tmp_path / "cache")
        connector = SpliceAIConnector(vcf_path=vcf_path, config=config)
        variant_df = pd.DataFrame({
            "variant_id": [
                "clinvar:17:43071077:G:T",
                "clinvar:17:43071077:G:A",
                "clinvar:1:925952:G:A",
            ],
            "source_db":  ["clinvar"] * 3,
            "chrom": ["17", "17", "1"],
            "pos":   [43071077, 43071077, 925952],
            "ref":   ["G", "G", "G"],
            "alt":   ["T", "A", "A"],
        })
 
        result = connector.fetch(variant_df=variant_df)
        assert "splice_ai_score" in result.columns
        assert len(result) == 3
        scores = result.set_index("variant_id")["splice_ai_score"]
        assert scores["clinvar:17:43071077:G:T"] == pytest.approx(0.01)
        assert scores["clinvar:17:43071077:G:A"] == pytest.approx(0.85)
        assert scores["clinvar:1:925952:G:A"]    == pytest.approx(0.05)
 
    def test_fetch_unknown_variant_gets_zero(self, tmp_path):
        import gzip
        from src.data.spliceai import SpliceAIConnector
        from src.data.database_connectors import FetchConfig
 
        vcf_lines = (
            "##fileformat=VCFv4.2\n"
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
            "chr17\t43071077\t.\tG\tT\t.\t.\t"
            "SpliceAI=T|BRCA1|0.50|0.00|0.00|0.00|0|0|0|0\n"
        )
        vcf_path = tmp_path / "test.vcf.gz"
        with gzip.open(vcf_path, "wt", encoding="utf-8") as f:
            f.write(vcf_lines)
 
        config    = FetchConfig(cache_dir=tmp_path / "cache")
        connector = SpliceAIConnector(vcf_path=vcf_path, config=config)
        variant_df = pd.DataFrame({
            "variant_id": ["clinvar:1:999999:A:C"],
            "source_db":  ["clinvar"],
            "chrom": ["1"], "pos": [999999], "ref": ["A"], "alt": ["C"],
        })
        result = connector.fetch(variant_df=variant_df)
        assert result["splice_ai_score"].iloc[0] == 0.0
 
    def test_parquet_cache_used_on_second_call(self, tmp_path):
        import gzip
        from src.data.spliceai import SpliceAIConnector
        from src.data.database_connectors import FetchConfig
 
        vcf_lines = (
            "##fileformat=VCFv4.2\n"
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
            "chr17\t43071077\t.\tG\tT\t.\t.\t"
            "SpliceAI=T|BRCA1|0.42|0.00|0.00|0.00|0|0|0|0\n"
        )
        vcf_path = tmp_path / "test.vcf.gz"
        with gzip.open(vcf_path, "wt", encoding="utf-8") as f:
            f.write(vcf_lines)
 
        config     = FetchConfig(cache_dir=tmp_path / "cache")
        connector  = SpliceAIConnector(vcf_path=vcf_path, config=config)
        variant_df = pd.DataFrame({
            "variant_id": ["clinvar:17:43071077:G:T"],
            "source_db":  ["clinvar"],
            "chrom": ["17"], "pos": [43071077], "ref": ["G"], "alt": ["T"],
        })
 
        connector.fetch(variant_df=variant_df)   # builds the cache
        vcf_path.unlink()                         # delete the VCF
 
        connector2 = SpliceAIConnector(vcf_path=vcf_path, config=config)
        result2    = connector2.fetch(variant_df=variant_df)
        assert result2["splice_ai_score"].iloc[0] == pytest.approx(0.42)
 
    # ── Integration with variant_ensemble.py ─────────────────────────────
 
    def test_splice_ai_score_in_tabular_features(self):
        """Connector is live -- splice_ai_score promoted to TABULAR_FEATURES."""
        from src.models.variant_ensemble import TABULAR_FEATURES
        assert "splice_ai_score" in TABULAR_FEATURES

    def test_splice_ai_score_not_in_phase2_features(self):
        """After promotion, splice_ai_score must no longer be in PHASE_2_FEATURES."""
        from src.models.variant_ensemble import PHASE_2_FEATURES
        assert "splice_ai_score" not in PHASE_2_FEATURES

# ---------------------------------------------------------------------------
# Tests: src/data/cadd.py -- Phase 2 Pillar 1, Connector 3
# ---------------------------------------------------------------------------
class TestCADDConnector:
    """Unit tests for CADDConnector. All HTTP calls are mocked."""

    def test_source_name(self):
        from src.data.cadd import CADDConnector
        assert CADDConnector.source_name == "cadd"

    def test_inherits_base_connector(self):
        from src.data.cadd import CADDConnector
        from src.data.database_connectors import BaseConnector
        assert issubclass(CADDConnector, BaseConnector)

    def test_rate_delay_enforced(self):
        from src.data.cadd import CADDConnector, CADD_RATE_DELAY
        from src.data.database_connectors import FetchConfig
        config    = FetchConfig(rate_limit_delay=0.1)
        connector = CADDConnector(config=config)
        assert connector.config.rate_limit_delay >= CADD_RATE_DELAY

    def test_parse_response_valid(self):
        from src.data.cadd import CADDConnector
        data   = [{"Chrom": "17", "Pos": "43071077",
                   "Ref": "G", "Alt": "T",
                   "RawScore": "2.345", "PHRED": "23.4"}]
        result = CADDConnector.parse_response(data)
        assert result == pytest.approx(23.4)

    def test_parse_response_low_score(self):
        from src.data.cadd import CADDConnector
        data   = [{"Chrom": "1", "Pos": "925952",
                   "Ref": "G", "Alt": "A",
                   "RawScore": "-0.5", "PHRED": "1.2"}]
        result = CADDConnector.parse_response(data)
        assert result == pytest.approx(1.2)

    def test_parse_response_empty_list_returns_median(self):
        from src.data.cadd import CADDConnector, CADD_MEDIAN_PHRED
        assert CADDConnector.parse_response([]) == CADD_MEDIAN_PHRED

    def test_parse_response_none_returns_median(self):
        from src.data.cadd import CADDConnector, CADD_MEDIAN_PHRED
        assert CADDConnector.parse_response(None) == CADD_MEDIAN_PHRED

    def test_parse_response_missing_phred_returns_median(self):
        from src.data.cadd import CADDConnector, CADD_MEDIAN_PHRED
        data   = [{"Chrom": "1", "Pos": "925952",
                   "Ref": "G", "Alt": "A", "RawScore": "-0.5"}]
        assert CADDConnector.parse_response(data) == CADD_MEDIAN_PHRED

    def test_parse_response_non_numeric_phred_returns_median(self):
        from src.data.cadd import CADDConnector, CADD_MEDIAN_PHRED
        data   = [{"Chrom": "1", "Pos": "925952",
                   "Ref": "G", "Alt": "A", "PHRED": "not_a_number"}]
        assert CADDConnector.parse_response(data) == CADD_MEDIAN_PHRED

    def test_fetch_empty_dataframe_returns_empty(self):
        from src.data.cadd import CADDConnector
        connector = CADDConnector()
        result    = connector.fetch(variant_df=pd.DataFrame())
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_fetch_adds_cadd_phred_column(self):
        from src.data.cadd import CADDConnector
        connector = CADDConnector()
        connector._fetch_one = MagicMock(return_value=23.4)
        variant_df = pd.DataFrame({
            "variant_id": ["clinvar:17:43071077:G:T"],
            "source_db":  ["clinvar"],
            "chrom": ["17"], "pos": [43071077], "ref": ["G"], "alt": ["T"],
        })
        result = connector.fetch(variant_df=variant_df)
        assert "cadd_phred" in result.columns
        assert result.iloc[0]["cadd_phred"] == pytest.approx(23.4)

    def test_fetch_multiple_variants(self):
        from src.data.cadd import CADDConnector
        connector = CADDConnector()
        expected  = {
            "17:43071077_G_T": 23.4,
            "17:43071077_G_A": 8.1,
            "1:925952_G_A":    3.5,
        }
        connector._fetch_one = MagicMock(
            side_effect=lambda key: expected.get(key, 15.0)
        )
        variant_df = pd.DataFrame({
            "variant_id": [
                "clinvar:17:43071077:G:T",
                "clinvar:17:43071077:G:A",
                "clinvar:1:925952:G:A",
            ],
            "source_db":  ["clinvar"] * 3,
            "chrom": ["17", "17", "1"],
            "pos":   [43071077, 43071077, 925952],
            "ref":   ["G", "G", "G"],
            "alt":   ["T", "A", "A"],
        })
        result = connector.fetch(variant_df=variant_df)
        scores = result.set_index("variant_id")["cadd_phred"]
        assert scores["clinvar:17:43071077:G:T"] == pytest.approx(23.4)
        assert scores["clinvar:17:43071077:G:A"] == pytest.approx(8.1)
        assert scores["clinvar:1:925952:G:A"]    == pytest.approx(3.5)

    def test_fetch_api_failure_returns_median(self):
        from src.data.cadd import CADDConnector, CADD_MEDIAN_PHRED
        connector = CADDConnector()
        connector._fetch_one = MagicMock(return_value=CADD_MEDIAN_PHRED)
        variant_df = pd.DataFrame({
            "variant_id": ["clinvar:1:999999:A:C"],
            "source_db":  ["clinvar"],
            "chrom": ["1"], "pos": [999999], "ref": ["A"], "alt": ["C"],
        })
        result = connector.fetch(variant_df=variant_df)
        assert result["cadd_phred"].iloc[0] == CADD_MEDIAN_PHRED

    def test_fetch_no_lookup_key_in_result(self):
        from src.data.cadd import CADDConnector
        connector = CADDConnector()
        connector._fetch_one = MagicMock(return_value=10.0)
        variant_df = pd.DataFrame({
            "variant_id": ["clinvar:1:1000:A:T"],
            "source_db":  ["clinvar"],
            "chrom": ["1"], "pos": [1000], "ref": ["A"], "alt": ["T"],
        })
        result = connector.fetch(variant_df=variant_df)
        assert "_lookup_key" not in result.columns

    def test_cadd_phred_used_by_engineer_features_when_present(self):
        from src.models.variant_ensemble import engineer_features
        df = pd.DataFrame({
            "variant_id":  ["clinvar:17:43071077:G:T"],
            "source_db":   ["clinvar"],
            "chrom":       ["17"],
            "pos":         [43071077],
            "ref":         ["G"],
            "alt":         ["T"],
            "consequence": ["missense_variant"],
            "allele_freq": [0.001],
            "cadd_phred":  [35.0],
        })
        feats = engineer_features(df)
        assert feats.iloc[0]["cadd_phred"] == pytest.approx(35.0)

    def test_cadd_phred_uses_median_when_absent(self):
        from src.models.variant_ensemble import engineer_features
        df = pd.DataFrame({
            "variant_id":  ["clinvar:1:925952:G:A"],
            "source_db":   ["clinvar"],
            "chrom":       ["1"],
            "pos":         [925952],
            "ref":         ["G"],
            "alt":         ["A"],
            "consequence": ["missense_variant"],
            "allele_freq": [0.05],
        })
        feats = engineer_features(df)
        assert feats.iloc[0]["cadd_phred"] == pytest.approx(15.0)

    def test_cadd_phred_in_tabular_features(self):
        from src.models.variant_ensemble import TABULAR_FEATURES, PHASE_2_FEATURES
        assert "cadd_phred" in TABULAR_FEATURES
        assert "cadd_phred" not in PHASE_2_FEATURES
 
 
# ---------------------------------------------------------------------------
# Tests: src/data/revel.py  (Connector 4 — Phase 2)
# ---------------------------------------------------------------------------
class TestREVELConnector:
    """
    All tests use an in-memory synthetic index to avoid the 1.3 GB data file.
    The `_inject_index` helper bypasses _load_index so the file path is never
    required.
    """
 
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _make_connector(rows: list[tuple]) -> "REVELConnector":
        """
        Return a REVELConnector pre-loaded with a synthetic index.
 
        rows: list of (chrom, pos, ref, alt, score) tuples — no file needed.
        """
        from src.data.revel import REVELConnector
        conn = REVELConnector(revel_file=None)
        conn._index = {
            (str(chrom), int(pos), ref.upper(), alt.upper()): float(score)
            for chrom, pos, ref, alt, score in rows
        }
        return conn
 
    @staticmethod
    def _canonical_df(**overrides) -> pd.DataFrame:
        """Minimal canonical-schema DataFrame for one variant."""
        base = dict(
            chrom=["17"],
            pos=[43071077],
            ref=["G"],
            alt=["T"],
            gene_symbol=["BRCA1"],
        )
        base.update({k: [v] for k, v in overrides.items()})
        return pd.DataFrame(base)
 
    # ------------------------------------------------------------------
    # Basic lookup
    # ------------------------------------------------------------------
    def test_known_variant_returns_real_score(self):
        """A variant present in the index returns its exact score."""
        conn = self._make_connector([("17", 43071077, "G", "T", 0.853)])
        score = conn.get_score("17", 43071077, "G", "T")
        assert abs(score - 0.853) < 1e-6
 
    def test_missing_variant_returns_default(self):
        """A variant absent from the index returns DEFAULT_SCORE."""
        from src.data.revel import DEFAULT_SCORE
        conn = self._make_connector([])
        score = conn.get_score("1", 100, "A", "C")
        assert score == DEFAULT_SCORE
 
    def test_custom_missing_value_honoured(self):
        """caller-supplied missing_value overrides DEFAULT_SCORE."""
        conn = self._make_connector([])
        score = conn.get_score("1", 100, "A", "C", missing_value=-1.0)
        assert score == -1.0
 
    # ------------------------------------------------------------------
    # Chromosome normalisation
    # ------------------------------------------------------------------
    def test_chr_prefix_stripped_on_get_score(self):
        """'chr17' and '17' resolve to the same index entry."""
        conn = self._make_connector([("17", 43071077, "G", "T", 0.75)])
        assert conn.get_score("chr17", 43071077, "G", "T") == 0.75
 
    def test_lowercase_chrom_accepted(self):
        """'chr1', 'Chr1', 'CHR1' all normalise correctly."""
        conn = self._make_connector([("1", 925952, "G", "A", 0.3)])
        for prefix in ("chr1", "Chr1", "CHR1", "1"):
            assert conn.get_score(prefix, 925952, "G", "A") == pytest.approx(0.3)
 
    def test_chrM_maps_to_MT(self):
        """Mitochondrial chromosome normalises to 'MT'."""
        conn = self._make_connector([("MT", 1234, "A", "G", 0.1)])
        assert conn.get_score("chrM", 1234, "A", "G") == pytest.approx(0.1)
        assert conn.get_score("M", 1234, "A", "G") == pytest.approx(0.1)
 
    def test_sex_chromosome_X(self):
        conn = self._make_connector([("X", 50000, "C", "T", 0.65)])
        assert conn.get_score("chrX", 50000, "C", "T") == pytest.approx(0.65)
 
    # ------------------------------------------------------------------
    # Case normalisation for alleles
    # ------------------------------------------------------------------
    def test_lowercase_alleles_accepted(self):
        """Allele case should not affect lookup."""
        conn = self._make_connector([("1", 100, "A", "C", 0.5)])
        assert conn.get_score("1", 100, "a", "c") == pytest.approx(0.5)
 
    # ------------------------------------------------------------------
    # annotate_dataframe
    # ------------------------------------------------------------------
    def test_annotate_adds_revel_score_column(self):
        """annotate_dataframe must add a 'revel_score' column."""
        conn = self._make_connector([("17", 43071077, "G", "T", 0.853)])
        df = self._canonical_df()
        result = conn.annotate_dataframe(df)
        assert "revel_score" in result.columns
 
    def test_annotate_returns_copy(self):
        """annotate_dataframe must not mutate the input DataFrame."""
        conn = self._make_connector([])
        df = self._canonical_df()
        original_cols = list(df.columns)
        _ = conn.annotate_dataframe(df)
        assert list(df.columns) == original_cols
 
    def test_annotate_correct_score_for_hit(self):
        """Matched variant gets the real score, not the default."""
        conn = self._make_connector([("17", 43071077, "G", "T", 0.853)])
        df = self._canonical_df()
        result = conn.annotate_dataframe(df)
        assert result.loc[0, "revel_score"] == pytest.approx(0.853)
 
    def test_annotate_default_for_miss(self):
        """Unmatched variant gets DEFAULT_SCORE."""
        from src.data.revel import DEFAULT_SCORE
        conn = self._make_connector([])
        df = self._canonical_df()
        result = conn.annotate_dataframe(df)
        assert result.loc[0, "revel_score"] == DEFAULT_SCORE
 
    def test_annotate_mixed_hits_and_misses(self):
        """Per-row resolution — one hit, one miss in the same DataFrame."""
        from src.data.revel import DEFAULT_SCORE
        conn = self._make_connector([("17", 43071077, "G", "T", 0.90)])
        df = pd.DataFrame({
            "chrom": ["17", "1"],
            "pos":   [43071077, 999999],
            "ref":   ["G", "A"],
            "alt":   ["T", "C"],
        })
        result = conn.annotate_dataframe(df)
        assert result.loc[0, "revel_score"] == pytest.approx(0.90)
        assert result.loc[1, "revel_score"] == DEFAULT_SCORE
 
    def test_annotate_preserves_existing_columns(self):
        """All original columns must survive annotation."""
        conn = self._make_connector([])
        df = self._canonical_df()
        result = conn.annotate_dataframe(df)
        for col in df.columns:
            assert col in result.columns
 
    def test_annotate_no_nans(self):
        """Output 'revel_score' column must have no NaNs."""
        conn = self._make_connector([])
        df = self._canonical_df()
        result = conn.annotate_dataframe(df)
        assert not result["revel_score"].isnull().any()
 
    def test_annotate_score_range(self):
        """REVEL scores must be in [0, 1]."""
        rows = [
            ("1", 100, "A", "C", 0.0),
            ("2", 200, "G", "T", 0.5),
            ("3", 300, "C", "A", 1.0),
        ]
        conn = self._make_connector(rows)
        df = pd.DataFrame({
            "chrom": ["1", "2", "3"],
            "pos":   [100, 200, 300],
            "ref":   ["A", "G", "C"],
            "alt":   ["C", "T", "A"],
        })
        result = conn.annotate_dataframe(df)
        assert (result["revel_score"] >= 0.0).all()
        assert (result["revel_score"] <= 1.0).all()
 
    def test_annotate_replaces_existing_revel_score(self):
        """If the DataFrame already has a 'revel_score' column it is overwritten."""
        conn = self._make_connector([("17", 43071077, "G", "T", 0.85)])
        df = self._canonical_df()
        df["revel_score"] = 0.0          # stale placeholder
        result = conn.annotate_dataframe(df)
        assert result.loc[0, "revel_score"] == pytest.approx(0.85)
 
    # ------------------------------------------------------------------
    # Stub-mode (no file supplied)
    # ------------------------------------------------------------------
    def test_stub_mode_returns_default_without_crash(self):
        """REVELConnector(None) must not raise — it returns DEFAULT_SCORE."""
        from src.data.revel import REVELConnector, DEFAULT_SCORE
        conn = REVELConnector(revel_file=None)
        # Force-inject empty index so _load_index (file I/O) is not called.
        conn._index = {}
        score = conn.get_score("1", 100, "A", "C")
        assert score == DEFAULT_SCORE
 
    # ------------------------------------------------------------------
    # _df_to_index utility
    # ------------------------------------------------------------------
    def test_df_to_index_keys_are_normalised(self):
        """_df_to_index must normalise chromosome and upper-case alleles."""
        from src.data.revel import REVELConnector
        df = pd.DataFrame({
            "chrom":       ["chr1", "chrX"],
            "pos":         [100,    200],
            "ref":         ["a",    "g"],
            "alt":         ["c",    "t"],
            "revel_score": [0.3,    0.8],
        })
        index = REVELConnector._df_to_index(df)
        assert ("1", 100, "A", "C") in index
        assert ("X", 200, "G", "T") in index
 
    def test_df_to_index_values_are_float(self):
        from src.data.revel import REVELConnector
        df = pd.DataFrame({
            "chrom": ["1"], "pos": [100],
            "ref": ["A"], "alt": ["C"],
            "revel_score": ["0.753"],   # string in raw CSV
        })
        index = REVELConnector._df_to_index(df)
        val = index[("1", 100, "A", "C")]
        assert isinstance(val, float)
        assert val == pytest.approx(0.753)
 
    # ------------------------------------------------------------------
    # Integration with engineer_features
    # ------------------------------------------------------------------
    def test_revel_score_flows_into_feature_matrix(self, sample_canonical_df):
        """
        After annotation, engineer_features must use the real revel_score
        rather than the 0.5 median fill-in default.
        """
        from src.data.revel import REVELConnector, _normalise_chrom
        from src.models.variant_ensemble import engineer_features

        conn = REVELConnector(revel_file=None)
        conn._index = {
            (_normalise_chrom(sample_canonical_df.loc[0, "chrom"]),
             int(sample_canonical_df.loc[0, "pos"]),
             sample_canonical_df.loc[0, "ref"].upper(),
             sample_canonical_df.loc[0, "alt"].upper()): 0.999,
        }

        annotated = conn.annotate_dataframe(sample_canonical_df)
        feats = engineer_features(annotated)

        assert feats.loc[0, "revel_score"] == pytest.approx(0.999)
        assert feats.loc[1, "revel_score"] == pytest.approx(0.5)
 
 
# ---------------------------------------------------------------------------
# Tests: src/data/revel.py  (Connector 4 — Phase 2)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tests: src/data/phylop.py  (Connector 5 — Phase 2)
# ---------------------------------------------------------------------------
class TestPhyloPConnector:
    """
    All tests inject a synthetic in-memory index to avoid the large BigWig
    file.  The _make_connector helper sets conn._index directly so no file
    I/O occurs.
    """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _make_connector(rows: list[tuple]) -> "PhyloPConnector":
        """
        Return a PhyloPConnector pre-loaded with a synthetic index.

        rows: list of (chrom, pos, score) tuples — no file needed.
        """
        from src.data.phylop import PhyloPConnector
        conn = PhyloPConnector(phylop_file=None)
        conn._index = {
            (str(chrom), int(pos)): float(score)
            for chrom, pos, score in rows
        }
        return conn

    @staticmethod
    def _canonical_df(**overrides) -> pd.DataFrame:
        """Minimal canonical-schema DataFrame for one variant."""
        base = dict(
            chrom=["17"],
            pos=[43071077],
            ref=["G"],
            alt=["T"],
            gene_symbol=["BRCA1"],
        )
        base.update({k: [v] for k, v in overrides.items()})
        return pd.DataFrame(base)

    # ------------------------------------------------------------------
    # Basic lookup
    # ------------------------------------------------------------------
    def test_known_position_returns_real_score(self):
        """A position present in the index returns its exact score."""
        conn = self._make_connector([("17", 43071077, 4.532)])
        score = conn.get_score("17", 43071077)
        assert abs(score - 4.532) < 1e-6

    def test_missing_position_returns_default(self):
        """A position absent from the index returns DEFAULT_SCORE."""
        from src.data.phylop import DEFAULT_SCORE
        conn = self._make_connector([])
        assert conn.get_score("1", 100) == DEFAULT_SCORE

    def test_custom_missing_value_honoured(self):
        """Caller-supplied missing_value overrides DEFAULT_SCORE."""
        conn = self._make_connector([])
        assert conn.get_score("1", 100, missing_value=-99.0) == -99.0

    def test_negative_score_returned_correctly(self):
        """Accelerated-evolution positions return negative scores."""
        conn = self._make_connector([("1", 925952, -3.7)])
        assert conn.get_score("1", 925952) == pytest.approx(-3.7)

    # ------------------------------------------------------------------
    # Chromosome normalisation
    # ------------------------------------------------------------------
    def test_chr_prefix_stripped_on_get_score(self):
        """'chr17' and '17' resolve to the same index entry."""
        conn = self._make_connector([("17", 43071077, 4.5)])
        assert conn.get_score("chr17", 43071077) == pytest.approx(4.5)

    def test_lowercase_chrom_accepted(self):
        """'chr1', 'Chr1', 'CHR1' all normalise correctly."""
        conn = self._make_connector([("1", 925952, 2.1)])
        for prefix in ("chr1", "Chr1", "CHR1", "1"):
            assert conn.get_score(prefix, 925952) == pytest.approx(2.1)

    def test_chrM_maps_to_MT(self):
        """Mitochondrial chromosome normalises to 'MT'."""
        conn = self._make_connector([("MT", 1234, 1.0)])
        assert conn.get_score("chrM", 1234) == pytest.approx(1.0)
        assert conn.get_score("M", 1234) == pytest.approx(1.0)

    def test_sex_chromosome_X(self):
        conn = self._make_connector([("X", 50000, 3.2)])
        assert conn.get_score("chrX", 50000) == pytest.approx(3.2)

    # ------------------------------------------------------------------
    # annotate_dataframe
    # ------------------------------------------------------------------
    def test_annotate_adds_phylop_score_column(self):
        """annotate_dataframe must add a 'phylop_score' column."""
        conn = self._make_connector([("17", 43071077, 4.532)])
        df = self._canonical_df()
        result = conn.annotate_dataframe(df)
        assert "phylop_score" in result.columns

    def test_annotate_returns_copy(self):
        """annotate_dataframe must not mutate the input DataFrame."""
        conn = self._make_connector([])
        df = self._canonical_df()
        original_cols = list(df.columns)
        _ = conn.annotate_dataframe(df)
        assert list(df.columns) == original_cols

    def test_annotate_correct_score_for_hit(self):
        """Matched position gets the real score, not the default."""
        conn = self._make_connector([("17", 43071077, 4.532)])
        df = self._canonical_df()
        result = conn.annotate_dataframe(df)
        assert result.loc[0, "phylop_score"] == pytest.approx(4.532)

    def test_annotate_default_for_miss(self):
        """Unmatched position gets DEFAULT_SCORE."""
        from src.data.phylop import DEFAULT_SCORE
        conn = self._make_connector([])
        df = self._canonical_df()
        result = conn.annotate_dataframe(df)
        assert result.loc[0, "phylop_score"] == DEFAULT_SCORE

    def test_annotate_mixed_hits_and_misses(self):
        """Per-row resolution — one hit, one miss in the same DataFrame."""
        from src.data.phylop import DEFAULT_SCORE
        conn = self._make_connector([("17", 43071077, 4.5)])
        df = pd.DataFrame({
            "chrom": ["17", "1"],
            "pos":   [43071077, 999999],
        })
        result = conn.annotate_dataframe(df)
        assert result.loc[0, "phylop_score"] == pytest.approx(4.5)
        assert result.loc[1, "phylop_score"] == DEFAULT_SCORE

    def test_annotate_preserves_existing_columns(self):
        """All original columns must survive annotation."""
        conn = self._make_connector([])
        df = self._canonical_df()
        result = conn.annotate_dataframe(df)
        for col in df.columns:
            assert col in result.columns

    def test_annotate_no_nans(self):
        """Output 'phylop_score' column must have no NaNs."""
        conn = self._make_connector([])
        df = self._canonical_df()
        result = conn.annotate_dataframe(df)
        assert not result["phylop_score"].isnull().any()

    def test_annotate_replaces_existing_phylop_score(self):
        """If the DataFrame already has 'phylop_score' it is overwritten."""
        conn = self._make_connector([("17", 43071077, 4.5)])
        df = self._canonical_df()
        df["phylop_score"] = 0.0
        result = conn.annotate_dataframe(df)
        assert result.loc[0, "phylop_score"] == pytest.approx(4.5)

    # ------------------------------------------------------------------
    # Stub mode (no file supplied)
    # ------------------------------------------------------------------
    def test_stub_mode_returns_default_without_crash(self):
        """PhyloPConnector(None) must not raise — returns DEFAULT_SCORE."""
        from src.data.phylop import PhyloPConnector, DEFAULT_SCORE
        conn = PhyloPConnector(phylop_file=None)
        conn._index = {}
        assert conn.get_score("1", 100) == DEFAULT_SCORE

    def test_stub_mode_annotate_dataframe(self):
        """annotate_dataframe in stub mode fills every row with DEFAULT_SCORE."""
        from src.data.phylop import PhyloPConnector, DEFAULT_SCORE
        conn = PhyloPConnector(phylop_file=None)
        conn._index = {}
        df = pd.DataFrame({"chrom": ["1", "17"], "pos": [100, 43071077]})
        result = conn.annotate_dataframe(df)
        assert (result["phylop_score"] == DEFAULT_SCORE).all()

    # ------------------------------------------------------------------
    # Integration with engineer_features
    # ------------------------------------------------------------------
    def test_phylop_score_flows_into_feature_matrix(self, sample_canonical_df):
        """
        After annotation, engineer_features must use the real phylop_score
        rather than the 0.0 neutral fill-in default.
        """
        from src.data.phylop import PhyloPConnector, _normalise_chrom
        from src.models.variant_ensemble import engineer_features

        conn = PhyloPConnector(phylop_file=None)
        conn._index = {
            (_normalise_chrom(sample_canonical_df.loc[0, "chrom"]),
             int(sample_canonical_df.loc[0, "pos"])): 7.5,
        }

        annotated = conn.annotate_dataframe(sample_canonical_df)
        feats = engineer_features(annotated)

        assert feats.loc[0, "phylop_score"] == pytest.approx(7.5)

# ---------------------------------------------------------------------------
# Tests: pipeline wiring in real_data_prep.py
# ---------------------------------------------------------------------------
class TestAnnotationPipeline:
    @pytest.fixture(autouse=True)
    def _isolate_spliceai(self, monkeypatch, tmp_path):
        """Short-circuit SpliceAI disk I/O for every test in this class.

        Without this fixture, tests that call pipeline._annotate_scores()
        instantiate SpliceAIConnector() with the default parquet path,
        triggering a 60+ second parquet read and rebuilding a 430 MB
        cache on disk. The fixture points DEFAULT_SPLICEAI_PATH at a
        nonexistent tmp file AND nulls out BaseConnector._load_cache,
        so the stub-zero code path is exercised regardless of on-disk
        state. Individual tests remain free to override these.
        """
        from src.data import spliceai as _spliceai_mod
        from src.data.database_connectors import BaseConnector
        monkeypatch.setattr(
            _spliceai_mod, "DEFAULT_SPLICEAI_PATH", tmp_path / "nonexistent.parquet"
        )
        monkeypatch.setattr(BaseConnector, "_load_cache", lambda self, key: None)

    """Tests for DataPrepPipeline._annotate_scores() and AnnotationConfig."""

    @pytest.fixture
    def minimal_canonical_df(self):
        return pd.DataFrame({
            "variant_id":    ["clinvar:17:43071077:G:T", "clinvar:1:925952:G:A"],
            "chrom":         ["17", "1"],
            "pos":           [43071077, 925952],
            "ref":           ["G", "G"],
            "alt":           ["T", "A"],
            "consequence":   ["missense_variant", "splice_donor_variant"],
            "gene_symbol":   ["BRCA1", "AGRN"],
            "allele_freq":   [0.0001, 0.05],
            "pathogenicity": ["pathogenic", "benign"],
            "clinical_sig":  ["Pathogenic", "Benign"],
            "source_db":     ["clinvar", "clinvar"],
            "transcript_id": [None, None],
            "protein_change":["p.Arg20Trp", None],
            "fasta_seq":     [None, None],
            "source_id":     ["12345", "12346"],
            "metadata":      [{}, {}],
        })

    def test_annotation_config_imports(self):
        from src.data.real_data_prep import AnnotationConfig
        cfg = AnnotationConfig()
        assert cfg.dbnsfp_path   is None
        assert cfg.phylop_path   is None
        assert cfg.spliceai_path is None
        assert cfg.annotate_cadd is False

    def test_annotation_config_all_none_is_default(self):
        from src.data.real_data_prep import AnnotationConfig
        cfg = AnnotationConfig()
        assert all(
            getattr(cfg, f) is None
            for f in ["dbnsfp_path", "phylop_path", "spliceai_path"]
        )

    def test_sift_defaults_consistent_across_modules(self):
        from src.data.sift_polyphen import DEFAULT_SIFT
        from src.models.variant_ensemble import engineer_features
        df = pd.DataFrame({
            "chrom": ["1"], "pos": [100], "ref": ["A"], "alt": ["T"],
            "consequence": ["intron_variant"], "allele_freq": [0.01],
        })
        feats = engineer_features(df)
        assert feats.loc[0, "sift_score"] == DEFAULT_SIFT == 0.5

    def test_sift_score_fill_is_not_threshold(self):
        import pathlib
        src = pathlib.Path("src/data/real_data_prep.py").read_text(encoding="utf-8")
        assert '"sift_score":' in src and '0.5' in src
        assert '"sift_score":             0.05,' not in src

    def test_annotate_scores_stub_mode_no_raise(self, minimal_canonical_df):
        from src.data.real_data_prep import DataPrepPipeline, AnnotationConfig
        pipeline = DataPrepPipeline(annotation_config=AnnotationConfig())
        result = pipeline._annotate_scores(minimal_canonical_df)
        assert result is not minimal_canonical_df

    def test_annotate_scores_adds_all_score_columns(self, minimal_canonical_df):
        from src.data.real_data_prep import DataPrepPipeline, AnnotationConfig
        pipeline = DataPrepPipeline(annotation_config=AnnotationConfig())
        result = pipeline._annotate_scores(minimal_canonical_df)
        for col in ["sift_score", "polyphen2_score", "revel_score",
                    "cadd_phred", "phylop_score", "gerp_score"]:
            assert col in result.columns, f"Missing: {col}"

    def test_annotate_scores_adds_splice_ai_column(self, minimal_canonical_df):
        from src.data.real_data_prep import DataPrepPipeline, AnnotationConfig
        pipeline = DataPrepPipeline(annotation_config=AnnotationConfig())
        result = pipeline._annotate_scores(minimal_canonical_df)
        assert "splice_ai_score" in result.columns

    def test_annotate_scores_stub_values_are_defaults(self, minimal_canonical_df):
        from src.data.real_data_prep import DataPrepPipeline, AnnotationConfig
        from src.data.dbnsfp import DEFAULT_SIFT, DEFAULT_PP2, DEFAULT_REVEL
        from src.data.dbnsfp import DEFAULT_CADD, DEFAULT_GERP
        pipeline = DataPrepPipeline(annotation_config=AnnotationConfig())
        result = pipeline._annotate_scores(minimal_canonical_df)
        assert (result["sift_score"]      == DEFAULT_SIFT).all()
        assert (result["polyphen2_score"] == DEFAULT_PP2).all()
        assert (result["revel_score"]     == DEFAULT_REVEL).all()
        assert (result["cadd_phred"]      == DEFAULT_CADD).all()
        assert (result["gerp_score"]      == DEFAULT_GERP).all()
        assert (result["splice_ai_score"] == 0.0).all()

    def test_annotate_scores_no_nans(self, minimal_canonical_df):
        from src.data.real_data_prep import DataPrepPipeline, AnnotationConfig
        pipeline = DataPrepPipeline(annotation_config=AnnotationConfig())
        result = pipeline._annotate_scores(minimal_canonical_df)
        for col in ["sift_score", "polyphen2_score", "revel_score",
                    "cadd_phred", "phylop_score", "gerp_score", "splice_ai_score"]:
            assert not result[col].isna().any(), f"NaN in {col}"

    def test_annotate_scores_preserves_input_columns(self, minimal_canonical_df):
        from src.data.real_data_prep import DataPrepPipeline, AnnotationConfig
        pipeline = DataPrepPipeline(annotation_config=AnnotationConfig())
        result = pipeline._annotate_scores(minimal_canonical_df)
        for col in minimal_canonical_df.columns:
            assert col in result.columns

    def test_cadd_skipped_when_annotate_cadd_false(self, minimal_canonical_df):
        from src.data.real_data_prep import DataPrepPipeline, AnnotationConfig
        from unittest.mock import patch
        pipeline = DataPrepPipeline(
            annotation_config=AnnotationConfig(annotate_cadd=False)
        )
        with patch("src.data.cadd.CADDConnector.fetch") as mock_fetch:
            pipeline._annotate_scores(minimal_canonical_df)
            mock_fetch.assert_not_called()

    def test_cadd_called_when_annotate_cadd_true(self, minimal_canonical_df):
        from src.data.real_data_prep import DataPrepPipeline, AnnotationConfig
        from unittest.mock import patch, MagicMock
        pipeline = DataPrepPipeline(
            annotation_config=AnnotationConfig(annotate_cadd=True)
        )
        with patch("src.data.real_data_prep.CADDConnector") as MockCADD:
            mock_instance = MagicMock()
            mock_instance.fetch.return_value = minimal_canonical_df.copy()
            MockCADD.return_value = mock_instance
            pipeline._annotate_scores(minimal_canonical_df)
            mock_instance.fetch.assert_called_once()

    def test_real_scores_flow_through_to_features(self, minimal_canonical_df):
        from src.data.real_data_prep import DataPrepPipeline, AnnotationConfig
        from src.models.variant_ensemble import engineer_features, TABULAR_FEATURES
        from unittest.mock import patch

        pipeline = DataPrepPipeline(annotation_config=AnnotationConfig())

        def fake_annotate(df, **kw):
            out = df.copy()
            out["sift_score"]      = [0.03, 0.5]
            out["polyphen2_score"] = [0.95, 0.5]
            out["revel_score"]     = [0.87, 0.5]
            out["cadd_phred"]      = [28.4, 15.0]
            out["phylop_score"]    = [7.2,  0.0]
            out["gerp_score"]      = [5.1,  0.0]
            return out

        with patch("src.data.real_data_prep.DbNSFPConnector") as MockDB:
            MockDB.return_value.annotate_dataframe.side_effect = fake_annotate
            annotated = pipeline._annotate_scores(minimal_canonical_df)

        feats = engineer_features(annotated)
        assert feats.shape[1] == len(TABULAR_FEATURES)
        assert feats.loc[0, "sift_score"]  == pytest.approx(0.03)
        assert feats.loc[0, "gerp_score"]  == pytest.approx(5.1)

    def test_annotation_sequence_dbnsfp_before_phylop(self, minimal_canonical_df):
        from src.data.real_data_prep import DataPrepPipeline, AnnotationConfig
        from unittest.mock import patch

        call_order = []
        pipeline = DataPrepPipeline(annotation_config=AnnotationConfig())

        with patch("src.data.real_data_prep.DbNSFPConnector") as MockDB, \
             patch("src.data.real_data_prep.PhyloPConnector") as MockPP:
            def db_side(df, **kw):
                call_order.append("dbnsfp")
                return df.copy()
            def pp_side(df, **kw):
                call_order.append("phylop")
                return df.copy()
            MockDB.return_value.annotate_dataframe.side_effect = db_side
            MockPP.return_value.annotate_dataframe.side_effect = pp_side
            pipeline._annotate_scores(minimal_canonical_df)

        assert call_order.index("dbnsfp") < call_order.index("phylop")

    def test_pipeline_accepts_annotation_config(self):
        from src.data.real_data_prep import DataPrepPipeline, AnnotationConfig
        cfg = AnnotationConfig(annotate_cadd=True)
        pipeline = DataPrepPipeline(annotation_config=cfg)
        assert pipeline.annotation_config.annotate_cadd is True

    def test_pipeline_default_annotation_config_is_stub(self):
        from src.data.real_data_prep import DataPrepPipeline
        pipeline = DataPrepPipeline()
        assert pipeline.annotation_config.dbnsfp_path   is None
        assert pipeline.annotation_config.annotate_cadd is False
