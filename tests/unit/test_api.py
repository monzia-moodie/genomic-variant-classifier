"""
tests/unit/test_api.py
=======================
Unit tests for the Phase 3 REST API layer.

Coverage
--------
  src/api/schemas.py           → VariantRequest validation, score_to_classification
  src/models/variant_ensemble  → engineer_features, TABULAR_FEATURES
  src/api/pipeline.py          → InferencePipeline.predict_single / predict_batch
  src/api/main.py              → /health, /info, /predict, /batch endpoints

Run with:
    pytest tests/unit/test_api.py -v --tb=short

All tests run without a real trained model on disk: the ensemble and scaler
are replaced by mocks / minimal fakes.

PHASE_2_FEATURES note
---------------------
  codon_position requires VEP/HGVSc and is not in TABULAR_FEATURES.
  splice_ai_score was promoted to TABULAR_FEATURES in commit 66b2740.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock
from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_variant_row() -> dict:
    """Bare-minimum variant dict (only required fields)."""
    return {
        "chrom": "17",
        "pos":   41276045,
        "ref":   "C",
        "alt":   "T",
    }


@pytest.fixture
def full_variant_row() -> dict:
    """Variant dict with all optional fields populated."""
    return {
        "chrom":              "17",
        "pos":                41276045,
        "ref":                "C",
        "alt":                "T",
        "allele_freq":        0.0,
        "consequence":        "missense_variant",
        "gene_symbol":        "BRCA1",
        "cadd_phred":         32.0,
        "sift_score":         0.02,
        "polyphen2_score":    0.99,
        "revel_score":        0.88,
        "phylop_score":       6.5,
        "gerp_score":         5.2,
        "alphamissense_score":  0.92,
        "gene_constraint_oe":  0.08,
        "n_pathogenic_in_gene": 150,
        "has_uniprot_annotation": 1,
        "n_known_pathogenic_protein_variants": 12,
    }


def _make_mock_trained_models(proba: float = 0.85) -> dict:
    """Return mock base models that always predict `proba` for the positive class."""
    model = MagicMock()
    model.predict_proba.side_effect = lambda X: np.array([[1 - proba, proba]] * len(X))
    return {"lightgbm": model, "xgboost": model}


def _make_mock_meta_learner(proba: float = 0.85):
    """Return a mock meta-learner that always returns `proba`."""
    ml = MagicMock()
    ml.predict_proba.side_effect = lambda X: np.array([[1 - proba, proba]] * len(X))
    return ml


def _make_pipeline(proba: float = 0.85):
    from src.api.pipeline import InferencePipeline, PipelineMetadata
    return InferencePipeline(
        trained_models = _make_mock_trained_models(proba),
        meta_learner   = _make_mock_meta_learner(proba),
        scaler         = None,
        metadata       = PipelineMetadata(
            val_auroc     = 0.9847,
            n_train       = 1_197_216,
            n_features    = 64,
            model_version = "phase2",
        ),
    )


# ---------------------------------------------------------------------------
# 1. Schema validation — VariantRequest
# ---------------------------------------------------------------------------

class TestVariantRequestSchema:

    def test_minimal_valid_input(self):
        from src.api.schemas import VariantRequest
        v = VariantRequest(chrom="17", pos=43094692, ref="G", alt="A")
        assert v.chrom == "17"
        assert v.ref   == "G"

    def test_chr_prefix_stripped(self):
        from src.api.schemas import VariantRequest
        v = VariantRequest(chrom="chr17", pos=1, ref="A", alt="T")
        assert v.chrom == "17"

    def test_chrM_normalised_to_MT(self):
        from src.api.schemas import VariantRequest
        v = VariantRequest(chrom="chrM", pos=1, ref="A", alt="G")
        assert v.chrom == "MT"

    def test_M_normalised_to_MT(self):
        from src.api.schemas import VariantRequest
        v = VariantRequest(chrom="M", pos=1, ref="A", alt="G")
        assert v.chrom == "MT"

    def test_alleles_uppercased(self):
        from src.api.schemas import VariantRequest
        v = VariantRequest(chrom="1", pos=100, ref="aac", alt="a")
        assert v.ref == "AAC"
        assert v.alt == "A"

    def test_pos_must_be_positive(self):
        from pydantic import ValidationError
        from src.api.schemas import VariantRequest
        with pytest.raises(ValidationError):
            VariantRequest(chrom="1", pos=0, ref="A", alt="T")

    def test_optional_scores_default_none(self):
        from src.api.schemas import VariantRequest
        v = VariantRequest(chrom="1", pos=1, ref="A", alt="T")
        assert v.alphamissense_score is None
        assert v.allele_freq is None
        assert v.n_pathogenic_in_gene is None
        assert v.has_uniprot_annotation is None

    def test_allele_freq_bounds(self):
        from pydantic import ValidationError
        from src.api.schemas import VariantRequest
        with pytest.raises(ValidationError):
            VariantRequest(chrom="1", pos=1, ref="A", alt="T", allele_freq=1.5)

    def test_variant_id_derived(self):
        from src.api.schemas import VariantRequest
        v = VariantRequest(chrom="17", pos=43094692, ref="G", alt="A")
        assert v._variant_id == "17:43094692:G:A"

    def test_batch_input_min_one(self):
        from pydantic import ValidationError
        from src.api.schemas import BatchPredictRequest
        with pytest.raises(ValidationError):
            BatchPredictRequest(variants=[])

    def test_batch_input_max_size(self):
        from pydantic import ValidationError
        from src.api.schemas import BatchPredictRequest, MAX_BATCH_SIZE
        variants = [{"chrom": "1", "pos": i + 1, "ref": "A", "alt": "T"}
                    for i in range(MAX_BATCH_SIZE + 1)]
        with pytest.raises(ValidationError):
            BatchPredictRequest(variants=variants)

    def test_model_dump_roundtrip(self):
        from src.api.schemas import VariantRequest
        v = VariantRequest(
            chrom="17", pos=43094692, ref="G", alt="A",
            consequence="missense_variant",
            alphamissense_score=0.94,
            allele_freq=0.0,
        )
        d = v.model_dump()
        assert d["chrom"] == "17"
        assert d["alphamissense_score"] == 0.94

    def test_all_optional_fields_accepted(self, full_variant_row):
        from src.api.schemas import VariantRequest
        v = VariantRequest(**full_variant_row)
        assert v.alphamissense_score             == pytest.approx(0.92)
        assert v.gene_constraint_oe              == pytest.approx(0.08)
        assert v.n_pathogenic_in_gene            == 150
        assert v.has_uniprot_annotation          == 1
        assert v.n_known_pathogenic_protein_variants == 12


# ---------------------------------------------------------------------------
# 2. BatchPredictRequest validation
# ---------------------------------------------------------------------------

class TestBatchPredictRequest:

    def test_empty_list_rejected(self):
        from pydantic import ValidationError
        from src.api.schemas import BatchPredictRequest
        with pytest.raises(ValidationError):
            BatchPredictRequest(variants=[])

    def test_over_limit_rejected(self):
        from pydantic import ValidationError
        from src.api.schemas import BatchPredictRequest, MAX_BATCH_SIZE, VariantRequest
        variants = [
            VariantRequest(chrom="1", pos=i + 1, ref="A", alt="T")
            for i in range(MAX_BATCH_SIZE + 1)
        ]
        with pytest.raises(ValidationError):
            BatchPredictRequest(variants=variants)

    def test_single_variant_accepted(self):
        from src.api.schemas import BatchPredictRequest, VariantRequest
        req = BatchPredictRequest(
            variants=[VariantRequest(chrom="1", pos=100, ref="A", alt="T")]
        )
        assert len(req.variants) == 1


# ---------------------------------------------------------------------------
# 3. score_to_classification
# ---------------------------------------------------------------------------

class TestScoreToClassification:

    @pytest.mark.parametrize("score,label", [
        (0.95, "Pathogenic"),
        (0.75, "Likely pathogenic"),
        (0.50, "Uncertain significance"),
        (0.20, "Likely benign"),
        (0.05, "Benign"),
    ])
    def test_correct_label(self, score, label):
        from src.api.schemas import score_to_classification
        classification, _ = score_to_classification(score)
        assert classification == label

    def test_high_confidence_far_from_boundary(self):
        # score=0.50 is the midpoint of "Uncertain significance" (0.30–0.70)
        # dist = min(0.50-0.30, 0.70-0.50) = 0.20 → "high"
        from src.api.schemas import score_to_classification
        _, confidence = score_to_classification(0.50)
        assert confidence == "high"

    def test_low_confidence_near_boundary(self):
        from src.api.schemas import score_to_classification
        _, confidence = score_to_classification(0.71)   # just above Likely pathogenic threshold
        assert confidence == "low"


# ---------------------------------------------------------------------------
# 4. engineer_features
# ---------------------------------------------------------------------------

class TestEngineerFeatures:
    """Tests for engineer_features() in src.models.variant_ensemble."""

    def test_output_columns_match_tabular_features(self):
        from src.models.variant_ensemble import engineer_features, TABULAR_FEATURES
        df = pd.DataFrame([{"chrom": "1", "pos": 100, "ref": "A", "alt": "T"}])
        feats = engineer_features(df)
        assert list(feats.columns) == TABULAR_FEATURES

    def test_no_nans_in_output(self):
        from src.models.variant_ensemble import engineer_features
        df = pd.DataFrame([{"chrom": "1", "pos": 100, "ref": "A", "alt": "T"}])
        feats = engineer_features(df)
        assert feats.isnull().sum().sum() == 0

    def test_snv_detection(self):
        from src.models.variant_ensemble import engineer_features
        df = pd.DataFrame([{"chrom": "1", "pos": 100, "ref": "A", "alt": "T"}])
        feats = engineer_features(df)
        assert feats["is_snv"].iloc[0]   == 1
        assert feats["is_indel"].iloc[0] == 0

    def test_insertion_is_indel(self):
        from src.models.variant_ensemble import engineer_features
        df = pd.DataFrame([{"chrom": "1", "pos": 100, "ref": "A", "alt": "ACGT"}])
        feats = engineer_features(df)
        assert feats["is_snv"].iloc[0]       == 0
        assert feats["is_indel"].iloc[0]     == 1
        assert feats["is_insertion"].iloc[0] == 1
        assert feats["is_deletion"].iloc[0]  == 0

    def test_deletion_is_indel(self):
        from src.models.variant_ensemble import engineer_features
        df = pd.DataFrame([{"chrom": "1", "pos": 100, "ref": "ACGT", "alt": "A"}])
        feats = engineer_features(df)
        assert feats["is_indel"].iloc[0]     == 1
        assert feats["is_deletion"].iloc[0]  == 1
        assert feats["is_insertion"].iloc[0] == 0
        assert feats["is_snv"].iloc[0]       == 0

    def test_missense_consequence(self):
        from src.models.variant_ensemble import engineer_features
        df = pd.DataFrame([{"chrom": "1", "pos": 1, "ref": "A", "alt": "T",
                             "consequence": "missense_variant"}])
        feats = engineer_features(df)
        assert feats["is_missense"].iloc[0]          == 1
        assert feats["in_coding"].iloc[0]            == 1
        assert feats["is_loss_of_function"].iloc[0]  == 0

    def test_lof_consequence(self):
        from src.models.variant_ensemble import engineer_features
        df = pd.DataFrame([{"chrom": "1", "pos": 1, "ref": "A", "alt": "T",
                             "consequence": "stop_gained"}])
        feats = engineer_features(df)
        assert feats["is_loss_of_function"].iloc[0] == 1
        assert feats["is_missense"].iloc[0]         == 0

    def test_frameshift_is_lof(self):
        from src.models.variant_ensemble import engineer_features
        df = pd.DataFrame([{"chrom": "1", "pos": 1, "ref": "AC", "alt": "A",
                             "consequence": "frameshift_variant"}])
        feats = engineer_features(df)
        assert feats["is_loss_of_function"].iloc[0] == 1

    def test_splice_site_consequence(self):
        from src.models.variant_ensemble import engineer_features
        df = pd.DataFrame([{"chrom": "1", "pos": 1, "ref": "A", "alt": "T",
                             "consequence": "splice_donor_variant"}])
        feats = engineer_features(df)
        assert feats["is_splice"].iloc[0]            == 1
        assert feats["is_loss_of_function"].iloc[0]  == 1

    def test_alphamissense_defaults_to_05_when_missing(self):
        from src.models.variant_ensemble import engineer_features
        df = pd.DataFrame([{"chrom": "1", "pos": 1, "ref": "A", "alt": "T"}])
        feats = engineer_features(df)
        assert feats["alphamissense_score"].iloc[0] == pytest.approx(0.5)

    def test_af_raw_defaults_to_zero_when_missing(self):
        from src.models.variant_ensemble import engineer_features
        df = pd.DataFrame([{"chrom": "1", "pos": 1, "ref": "A", "alt": "T"}])
        feats = engineer_features(df)
        assert feats["af_raw"].iloc[0]        == pytest.approx(0.0)
        assert feats["af_is_absent"].iloc[0]  == 1
        assert feats["af_is_common"].iloc[0]  == 0

    def test_af_is_common_for_high_af(self):
        from src.models.variant_ensemble import engineer_features
        df = pd.DataFrame([{"chrom": "1", "pos": 1, "ref": "A", "alt": "T",
                             "allele_freq": 0.33}])
        feats = engineer_features(df)
        assert feats["af_is_absent"].iloc[0]  == 0
        assert feats["af_is_common"].iloc[0]  == 1

    def test_codon_position_in_tabular_features(self):
        """codon_position is derived by VEPConnector (Phase 4) and is now in TABULAR_FEATURES."""
        from src.models.variant_ensemble import TABULAR_FEATURES, PHASE_2_FEATURES
        assert "codon_position" in TABULAR_FEATURES, (
            "codon_position should be in TABULAR_FEATURES after Phase 4 VEP promotion."
        )
        assert "codon_position" not in PHASE_2_FEATURES

    def test_batch_consistency(self):
        """engineer_features must produce identical rows regardless of batch size."""
        from src.models.variant_ensemble import engineer_features
        single = pd.DataFrame([{"chrom": "1", "pos": 100, "ref": "A", "alt": "T",
                                 "consequence": "missense_variant", "allele_freq": 0.001}])
        batch  = pd.concat([single, single], ignore_index=True)
        feats_s = engineer_features(single)
        feats_b = engineer_features(batch)
        pd.testing.assert_frame_equal(
            feats_b.iloc[[0]].reset_index(drop=True),
            feats_s.reset_index(drop=True),
        )


# ---------------------------------------------------------------------------
# 5. InferencePipeline
# ---------------------------------------------------------------------------

class TestInferencePipeline:

    def test_predict_single_returns_keys(self):
        pipe   = _make_pipeline(proba=0.85)
        result = pipe.predict_single({"chrom": "17", "pos": 43094692, "ref": "G", "alt": "A"})
        assert set(result.keys()) == {"pathogenicity_score", "classification", "confidence"}

    def test_predict_single_high_score_pathogenic(self):
        pipe   = _make_pipeline(proba=0.95)
        result = pipe.predict_single({"chrom": "17", "pos": 1, "ref": "G", "alt": "A"})
        assert result["classification"] == "Pathogenic"

    def test_predict_single_likely_pathogenic(self):
        pipe   = _make_pipeline(proba=0.80)
        result = pipe.predict_single({"chrom": "17", "pos": 1, "ref": "G", "alt": "A"})
        assert result["classification"] == "Likely pathogenic"

    def test_predict_single_uncertain(self):
        pipe   = _make_pipeline(proba=0.50)
        result = pipe.predict_single({"chrom": "17", "pos": 1, "ref": "G", "alt": "A"})
        assert result["classification"] == "Uncertain significance"

    def test_predict_single_likely_benign(self):
        pipe   = _make_pipeline(proba=0.20)
        result = pipe.predict_single({"chrom": "17", "pos": 1, "ref": "G", "alt": "A"})
        assert result["classification"] == "Likely benign"

    def test_predict_single_benign(self):
        pipe   = _make_pipeline(proba=0.05)
        result = pipe.predict_single({"chrom": "17", "pos": 1, "ref": "G", "alt": "A"})
        assert result["classification"] == "Benign"

    def test_predict_batch_length(self):
        pipe    = _make_pipeline(proba=0.9)
        results = pipe.predict_batch([
            {"chrom": "17", "pos": 43094692, "ref": "G", "alt": "A"},
            {"chrom": "1",  "pos": 925952,   "ref": "G", "alt": "T"},
        ])
        assert len(results) == 2

    def test_predict_batch_empty_returns_empty(self):
        pipe    = _make_pipeline()
        results = pipe.predict_batch([])
        assert results == []

    def test_scaler_applied_when_present(self):
        from sklearn.preprocessing import StandardScaler
        from src.api.pipeline import InferencePipeline

        scaler = MagicMock(spec=StandardScaler)
        scaler.transform.side_effect = lambda X: X  # identity

        pipe = InferencePipeline(
            trained_models = _make_mock_trained_models(0.8),
            meta_learner   = _make_mock_meta_learner(0.8),
            scaler         = scaler,
        )
        pipe.predict_single({"chrom": "1", "pos": 1, "ref": "A", "alt": "T"})
        assert scaler.transform.called

    def test_save_and_load_roundtrip(self, tmp_path):
        """Roundtrip via from_variant_ensemble — no VariantEnsemble dependency at load time."""
        from sklearn.linear_model import LogisticRegression
        from src.api.pipeline import InferencePipeline
        from src.models.variant_ensemble import VariantEnsemble, EnsembleConfig, TABULAR_FEATURES

        cfg = EnsembleConfig(n_folds=2, model_dir=tmp_path / "models")
        ens = VariantEnsemble(cfg)
        ens.base_estimators = {"logistic_regression": LogisticRegression(max_iter=50)}

        rng   = np.random.default_rng(42)
        n     = 20
        X_tab = pd.DataFrame(rng.random((n, len(TABULAR_FEATURES))), columns=TABULAR_FEATURES)
        X_seq = pd.Series(["A" * 101] * n)
        y     = pd.Series([0] * 10 + [1] * 10)
        ens.fit(X_tab, X_seq, y)

        pipe = InferencePipeline.from_variant_ensemble(ens, val_auroc=0.9847)
        out  = tmp_path / "pipeline.joblib"
        pipe.save(out)
        loaded = InferencePipeline.load(out)
        assert loaded.metadata.val_auroc == pytest.approx(0.9847)
        result = loaded.predict_single({"chrom": "1", "pos": 1, "ref": "A", "alt": "T"})
        assert result["classification"] in {
            "Pathogenic", "Likely pathogenic", "Uncertain significance",
            "Likely benign", "Benign",
        }

    def test_load_wrong_type_raises(self, tmp_path):
        import joblib
        from src.api.pipeline import InferencePipeline
        p = tmp_path / "bad.joblib"
        joblib.dump({"not": "a pipeline"}, p)
        with pytest.raises(TypeError):
            InferencePipeline.load(p)


# ---------------------------------------------------------------------------
# 6. FastAPI endpoints
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """TestClient with the model pre-loaded as a mock pipeline."""
    from fastapi.testclient import TestClient
    import src.api.main as api_main

    pipe = _make_pipeline(proba=0.92)
    api_main._PIPELINE = pipe
    client = TestClient(api_main.app)
    yield client
    api_main._PIPELINE = None


class TestHealthEndpoint:

    def test_health_returns_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"]       == "ok"
        assert body["model_loaded"] is True
        assert "uptime_seconds"     in body

    def test_health_includes_uptime(self, client):
        r = client.get("/health")
        assert isinstance(r.json()["uptime_seconds"], float)

    def test_health_model_not_ready_when_no_artifact(self):
        from fastapi.testclient import TestClient
        import src.api.main as api_main

        api_main._PIPELINE = None
        c = TestClient(api_main.app)
        r = c.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["model_loaded"] is False
        assert body["status"]       == "degraded"


class TestInfoEndpoint:

    def test_info_returns_metadata(self, client):
        r = client.get("/info")
        assert r.status_code == 200
        body = r.json()
        assert body["holdout_auroc"]     == pytest.approx(0.9847)
        assert body["model_version"]     == "phase2-v1"
        assert body["pipeline_version"]  == "1.0.0"
        assert body["training_auroc"]    == pytest.approx(0.9780)
        assert body["n_features"]        == 64
        assert len(body["feature_names"]) == 64
        # Phase 4: all features promoted, phase2_features_remaining is now empty
        assert isinstance(body["phase2_features_remaining"], list)

    def test_info_phase2_remaining_is_empty(self, client):
        """All Phase 2 features have been promoted to TABULAR_FEATURES in Phase 4."""
        r = client.get("/info")
        remaining = r.json()["phase2_features_remaining"]
        assert isinstance(remaining, list)
        assert len(remaining) == 0, (
            f"Expected empty phase2_features_remaining, got: {remaining}"
        )


class TestPredictEndpoint:

    def test_predict_single_valid(self, client):
        payload = {
            "chrom": "17", "pos": 43094692, "ref": "G", "alt": "A",
            "consequence": "missense_variant",
            "gene_symbol": "BRCA1",
            "alphamissense_score": 0.94,
            "allele_freq": 0.0,
        }
        r = client.post("/predict", json=payload)
        assert r.status_code == 200
        body = r.json()
        assert "prediction" in body
        pred = body["prediction"]
        assert 0.0 <= pred["pathogenicity_score"] <= 1.0
        assert pred["classification"] in {
            "Pathogenic", "Likely pathogenic", "Uncertain significance",
            "Likely benign", "Benign",
        }
        assert pred["confidence"] in {"high", "medium", "low"}
        assert pred["variant_id"] == "17:43094692:G:A"

    def test_predict_returns_model_version(self, client):
        r = client.post("/predict", json={"chrom": "1", "pos": 1, "ref": "A", "alt": "T"})
        assert r.status_code == 200
        assert r.json()["model_version"] == "phase2-v1"

    def test_predict_chrom_normalised(self, client):
        payload = {"chrom": "chr17", "pos": 43094692, "ref": "g", "alt": "a"}
        r = client.post("/predict", json=payload)
        assert r.status_code == 200
        assert r.json()["prediction"]["variant_id"].startswith("17:")

    def test_predict_missing_required_field(self, client):
        r = client.post("/predict", json={"chrom": "17", "pos": 1, "ref": "G"})
        assert r.status_code == 422

    def test_predict_invalid_allele_freq(self, client):
        payload = {"chrom": "1", "pos": 1, "ref": "A", "alt": "T", "allele_freq": 2.0}
        r = client.post("/predict", json=payload)
        assert r.status_code == 422

    def test_predict_indel(self, client):
        payload = {"chrom": "17", "pos": 43115726, "ref": "AAC", "alt": "A",
                   "gene_symbol": "BRCA1"}
        r = client.post("/predict", json=payload)
        assert r.status_code == 200

    def test_predict_sex_chromosome(self, client):
        r = client.post("/predict", json={"chrom": "X", "pos": 153_000_000, "ref": "C", "alt": "T"})
        assert r.status_code == 200

    def test_predict_mitochondrial(self, client):
        r = client.post("/predict", json={"chrom": "MT", "pos": 8993, "ref": "T", "alt": "G"})
        assert r.status_code == 200

    def test_predict_503_when_no_model(self):
        from fastapi.testclient import TestClient
        import src.api.main as api_main
        original = api_main._PIPELINE
        api_main._PIPELINE = None
        try:
            c = TestClient(api_main.app, raise_server_exceptions=False)
            r = c.post("/predict", json={"chrom": "1", "pos": 1, "ref": "A", "alt": "T"})
            assert r.status_code == 503
        finally:
            api_main._PIPELINE = original


class TestBatchEndpoint:

    def test_batch_two_variants(self, client):
        payload = {"variants": [
            {"chrom": "17", "pos": 43094692, "ref": "G", "alt": "A"},
            {"chrom": "1",  "pos": 925952,   "ref": "G", "alt": "T"},
        ]}
        r = client.post("/batch", json=payload)
        assert r.status_code == 200
        body = r.json()
        assert body["n_pathogenic"] + body["n_benign"] + body["n_uncertain"] == 2
        assert len(body["predictions"]) == 2
        assert body["model_version"] == "phase2-v1"

    def test_batch_counts_correct(self, client):
        """With proba=0.92 all variants should be Pathogenic."""
        payload = {"variants": [
            {"chrom": "1", "pos": i + 1, "ref": "A", "alt": "T"}
            for i in range(5)
        ]}
        r = client.post("/batch", json=payload)
        assert r.status_code == 200
        body = r.json()
        assert body["n_pathogenic"] == 5
        assert body["n_benign"]     == 0
        assert body["n_uncertain"]  == 0

    def test_batch_empty_rejected(self, client):
        r = client.post("/batch", json={"variants": []})
        assert r.status_code == 422

    def test_batch_chrom_normalisation(self, client):
        payload = {"variants": [{"chrom": "chr1", "pos": 1, "ref": "A", "alt": "T"}]}
        r = client.post("/batch", json=payload)
        assert r.status_code == 200
        assert r.json()["predictions"][0]["variant_id"].startswith("1:")

    def test_batch_503_when_no_model(self):
        from fastapi.testclient import TestClient
        import src.api.main as api_main
        original = api_main._PIPELINE
        api_main._PIPELINE = None
        try:
            c = TestClient(api_main.app, raise_server_exceptions=False)
            r = c.post("/batch", json={"variants": [
                {"chrom": "1", "pos": 1, "ref": "A", "alt": "T"}
            ]})
            assert r.status_code == 503
        finally:
            api_main._PIPELINE = original


# ---------------------------------------------------------------------------
# 7. /rsid/{rs_id} endpoint
# ---------------------------------------------------------------------------

class TestRsidEndpoint:

    def test_rsid_unknown_when_no_index(self, client):
        """With no dbSNP index loaded, /rsid always returns known=false."""
        import src.api.main as api_main
        original = api_main._DBSNP_INDEX
        api_main._DBSNP_INDEX = None
        try:
            r = client.get("/rsid/rs12345678")
            assert r.status_code == 200
            body = r.json()
            assert body["known"] is False
            assert body["rs_id"] == "rs12345678"
        finally:
            api_main._DBSNP_INDEX = original

    def test_rsid_known_with_index(self, client):
        """With a mock dbSNP index, /rsid resolves locus and returns prediction."""
        import pandas as pd
        import src.api.main as api_main

        mock_index = pd.DataFrame([{
            "rs_id": "rs12345678",
            "chrom": "1",
            "pos": 925952,
            "ref": "G",
            "alt": "T",
        }]).set_index("rs_id")

        original = api_main._DBSNP_INDEX
        api_main._DBSNP_INDEX = mock_index
        try:
            r = client.get("/rsid/rs12345678")
            assert r.status_code == 200
            body = r.json()
            assert body["known"] is True
            assert body["chrom"] == "1"
            assert body["pos"] == 925952
            assert body["ref"] == "G"
            assert body["alt"] == "T"
            # Pipeline is loaded in the client fixture, so prediction should be present
            assert body["prediction"] is not None
            assert 0.0 <= body["prediction"]["pathogenicity_score"] <= 1.0
        finally:
            api_main._DBSNP_INDEX = original

    def test_rsid_normalises_prefix(self, client):
        """rs-ID without 'rs' prefix is accepted and normalised."""
        import src.api.main as api_main

        original = api_main._DBSNP_INDEX
        api_main._DBSNP_INDEX = None
        try:
            r = client.get("/rsid/12345678")
            assert r.status_code == 200
            body = r.json()
            assert body["rs_id"] == "rs12345678"
        finally:
            api_main._DBSNP_INDEX = original

    def test_rsid_no_prediction_when_no_model(self):
        """When pipeline is not loaded, /rsid still resolves locus but no prediction."""
        import pandas as pd
        from fastapi.testclient import TestClient
        import src.api.main as api_main

        mock_index = pd.DataFrame([{
            "rs_id": "rs99999999",
            "chrom": "2",
            "pos": 100,
            "ref": "A",
            "alt": "T",
        }]).set_index("rs_id")

        orig_pipeline = api_main._PIPELINE
        orig_index    = api_main._DBSNP_INDEX
        api_main._PIPELINE    = None
        api_main._DBSNP_INDEX = mock_index
        try:
            c = TestClient(api_main.app)
            r = c.get("/rsid/rs99999999")
            assert r.status_code == 200
            body = r.json()
            assert body["known"] is True
            assert body["prediction"] is None
        finally:
            api_main._PIPELINE    = orig_pipeline
            api_main._DBSNP_INDEX = orig_index


# ---------------------------------------------------------------------------
# 8. Auth header (X-API-Key)
# ---------------------------------------------------------------------------

class TestApiKeyAuth:

    def test_dev_mode_no_key_required(self, client):
        """When API_KEYS env var is empty, all requests pass without a key."""
        import src.api.auth as auth_module
        original = auth_module._VALID_KEYS
        auth_module._VALID_KEYS = frozenset()   # simulate dev mode
        try:
            r = client.get("/info")
            assert r.status_code == 200
        finally:
            auth_module._VALID_KEYS = original

    def test_valid_key_accepted(self, client):
        """A request with a valid X-API-Key is accepted."""
        import src.api.auth as auth_module
        original = auth_module._VALID_KEYS
        auth_module._VALID_KEYS = frozenset({"test-key"})
        try:
            r = client.get("/info", headers={"X-API-Key": "test-key"})
            assert r.status_code == 200
        finally:
            auth_module._VALID_KEYS = original

    def test_invalid_key_rejected(self, client):
        """A request with a wrong key returns 401."""
        import src.api.auth as auth_module
        original = auth_module._VALID_KEYS
        auth_module._VALID_KEYS = frozenset({"test-key"})
        try:
            r = client.get("/info", headers={"X-API-Key": "wrong-key"})
            assert r.status_code == 401
        finally:
            auth_module._VALID_KEYS = original

    def test_missing_key_rejected(self, client):
        """A request with no key returns 401 when API_KEYS is set."""
        import src.api.auth as auth_module
        original = auth_module._VALID_KEYS
        auth_module._VALID_KEYS = frozenset({"test-key"})
        try:
            r = client.get("/info")
            assert r.status_code == 401
        finally:
            auth_module._VALID_KEYS = original

    def test_health_always_open(self, client):
        """/health must not require auth even when API_KEYS is set."""
        import src.api.auth as auth_module
        original = auth_module._VALID_KEYS
        auth_module._VALID_KEYS = frozenset({"test-key"})
        try:
            r = client.get("/health")
            assert r.status_code == 200
        finally:
            auth_module._VALID_KEYS = original
