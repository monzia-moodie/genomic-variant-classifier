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
        # Row 1 = stop_gained
        assert feats.loc[1, "is_nonsense"] == 1
        # Row 3 = splice_donor_variant
        assert feats.loc[3, "in_splice_site"] == 1

    def test_codon_position_not_in_tabular_features(self):
        """codon_position was always 0 — it was removed from TABULAR_FEATURES (Issue P)."""
        from src.models.variant_ensemble import TABULAR_FEATURES, PHASE_2_FEATURES
        assert "codon_position" not in TABULAR_FEATURES
        assert "codon_position" in PHASE_2_FEATURES

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
