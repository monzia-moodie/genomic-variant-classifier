"""
tests/unit/test_catboost.py
============================
Tests for CatBoostVariantClassifier and its integration with VariantEnsemble
and InferencePipeline.

Run with:
    pytest tests/unit/test_catboost.py -v --tb=short

All tests operate on synthetic data. CatBoost is an optional dependency —
tests are skipped when it is not installed.

Fixture design rules (to avoid the four failure classes):
  - small_variant_df: has mixed string+numeric columns — for CatBoost-only tests
  - numeric_variant_df: all-numeric — for ensemble integration tests
    (ensemble.fit() passes X_tab.values to non-CatBoost models, which fail on strings)
  - InferencePipeline constructed with trained_models= (not base_models=)
  - task_type="CPU" everywhere — GPU tests require explicit opt-in
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

catboost = pytest.importorskip("catboost", reason="catboost not installed")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_variant_df() -> pd.DataFrame:
    """Mixed string+numeric fixture — for CatBoost-only tests."""
    rng = np.random.default_rng(42)
    n   = 200
    return pd.DataFrame({
        # Categorical — CatBoost native
        "gene_symbol":   rng.choice(["BRCA1", "TP53", "LDLR", "MTHFR", "CFTR"], n),
        "consequence":   rng.choice([
            "missense_variant", "stop_gained", "frameshift_variant",
            "synonymous_variant", "splice_donor_variant",
        ], n),
        "chrom":         rng.choice([str(i) for i in range(1, 23)] + ["X", "Y"], n),
        "review_status": rng.choice([
            "reviewed by expert panel",
            "criteria provided, multiple submitters, no conflicts",
            "criteria provided, single submitter",
        ], n),
        # Numerical
        "af_raw":              rng.uniform(0, 0.01, n),
        "af_log10":            rng.uniform(-8, 0, n),
        "af_is_absent":        rng.integers(0, 2, n).astype(float),
        "af_is_ultra_rare":    rng.integers(0, 2, n).astype(float),
        "af_is_rare":          rng.integers(0, 2, n).astype(float),
        "af_is_common":        rng.integers(0, 2, n).astype(float),
        "cadd_phred":          rng.uniform(0, 40, n),
        "alphamissense_score": rng.uniform(0, 1, n),
        "sift_score":          rng.uniform(0, 1, n),
        "polyphen2_score":     rng.uniform(0, 1, n),
        "revel_score":         rng.uniform(0, 1, n),
        "phylop_score":        rng.uniform(-10, 10, n),
        "gerp_score":          rng.uniform(-5, 10, n),
        "gene_constraint_oe":  rng.uniform(0, 2, n),
        "n_pathogenic_in_gene": rng.integers(0, 3000, n).astype(float),
        "is_missense":         rng.integers(0, 2, n).astype(float),
        "is_snv":              rng.integers(0, 2, n).astype(float),
        "consequence_severity": rng.integers(0, 11, n).astype(float),
    })


@pytest.fixture
def numeric_variant_df() -> pd.DataFrame:
    """All-numeric fixture — safe for ensemble.fit() which passes .values to non-CB models."""
    rng = np.random.default_rng(7)
    n   = 200
    return pd.DataFrame({
        "af_raw":              rng.uniform(0, 0.01, n),
        "af_log10":            rng.uniform(-8, 0, n),
        "af_is_absent":        rng.integers(0, 2, n).astype(float),
        "af_is_ultra_rare":    rng.integers(0, 2, n).astype(float),
        "af_is_rare":          rng.integers(0, 2, n).astype(float),
        "af_is_common":        rng.integers(0, 2, n).astype(float),
        "cadd_phred":          rng.uniform(0, 40, n),
        "alphamissense_score": rng.uniform(0, 1, n),
        "sift_score":          rng.uniform(0, 1, n),
        "polyphen2_score":     rng.uniform(0, 1, n),
        "revel_score":         rng.uniform(0, 1, n),
        "phylop_score":        rng.uniform(-10, 10, n),
        "gerp_score":          rng.uniform(-5, 10, n),
        "gene_constraint_oe":  rng.uniform(0, 2, n),
        "n_pathogenic_in_gene": rng.integers(0, 3000, n).astype(float),
        "is_missense":         rng.integers(0, 2, n).astype(float),
        "is_snv":              rng.integers(0, 2, n).astype(float),
        "consequence_severity": rng.integers(0, 11, n).astype(float),
    })


@pytest.fixture
def small_labels() -> np.ndarray:
    return np.random.default_rng(0).integers(0, 2, 200).astype(np.int32)


@pytest.fixture
def numeric_labels() -> np.ndarray:
    return np.random.default_rng(1).integers(0, 2, 200).astype(np.int32)


@pytest.fixture
def fitted_classifier(small_variant_df, small_labels):
    from genomic_variant_classifier.models.catboost_wrapper import CatBoostVariantClassifier
    clf = CatBoostVariantClassifier(
        iterations=50, verbose=0, task_type="CPU",
        cat_feature_names=["gene_symbol", "consequence", "chrom", "review_status"],
    )
    clf.fit(small_variant_df, small_labels)
    return clf


# ---------------------------------------------------------------------------
# Tests: CatBoostVariantClassifier
# ---------------------------------------------------------------------------

class TestCatBoostVariantClassifier:

    def test_import(self):
        from genomic_variant_classifier.models.catboost_wrapper import CatBoostVariantClassifier
        assert CatBoostVariantClassifier is not None

    def test_fit_with_dataframe(self, small_variant_df, small_labels):
        from genomic_variant_classifier.models.catboost_wrapper import CatBoostVariantClassifier
        clf = CatBoostVariantClassifier(iterations=30, verbose=0, task_type="CPU")
        clf.fit(small_variant_df, small_labels)
        assert clf._model is not None

    def test_fit_with_numpy(self, numeric_variant_df, numeric_labels):
        """CatBoost must fall back to numerical-only when passed a numpy array."""
        from genomic_variant_classifier.models.catboost_wrapper import CatBoostVariantClassifier
        clf = CatBoostVariantClassifier(iterations=30, verbose=0, task_type="CPU")
        clf.fit(numeric_variant_df.to_numpy(), numeric_labels)
        proba = clf.predict_proba(numeric_variant_df.to_numpy())
        assert proba.shape == (len(numeric_labels), 2)

    def test_predict_proba_shape(self, fitted_classifier, small_variant_df):
        proba = fitted_classifier.predict_proba(small_variant_df)
        assert proba.shape == (len(small_variant_df), 2)

    def test_predict_proba_sums_to_one(self, fitted_classifier, small_variant_df):
        proba = fitted_classifier.predict_proba(small_variant_df)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_predict_proba_in_01(self, fitted_classifier, small_variant_df):
        proba = fitted_classifier.predict_proba(small_variant_df)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_predict_returns_binary(self, fitted_classifier, small_variant_df):
        preds = fitted_classifier.predict(small_variant_df)
        assert set(preds).issubset({0, 1})

    def test_categorical_features_detected(self, fitted_classifier):
        """gene_symbol, consequence, chrom, review_status should be categorical."""
        assert len(fitted_classifier._cat_indices) > 0

    def test_missing_categorical_sentinel(self):
        """NA values in categorical columns should be filled with __missing__."""
        from genomic_variant_classifier.models.catboost_wrapper import CatBoostVariantClassifier
        rng = np.random.default_rng(1)
        df  = pd.DataFrame({
            "gene_symbol": ["BRCA1", None, "TP53"] * 20,
            "af_raw":      rng.uniform(0, 1, 60),
            "cadd_phred":  rng.uniform(0, 40, 60),
        })
        y = rng.integers(0, 2, 60).astype(np.int32)
        clf = CatBoostVariantClassifier(
            iterations=20, verbose=0, task_type="CPU",
            cat_feature_names=["gene_symbol"],
        )
        clf.fit(df, y)
        proba = clf.predict_proba(df)
        assert proba.shape == (60, 2)

    def test_categorical_not_applied_to_numeric_column(self):
        """
        A column in cat_feature_names that is numeric (int64) must NOT be
        treated as categorical — CatBoost ordered target statistics require
        string dtype. n_pathogenic_in_gene is int64 → stays numerical.

        This test uses a clean DataFrame with only one string column so that
        CatBoost doesn't encounter unexpected string values in numeric columns.
        """
        from genomic_variant_classifier.models.catboost_wrapper import CatBoostVariantClassifier
        rng = np.random.default_rng(2)
        n   = 100
        df  = pd.DataFrame({
            "gene_symbol":         rng.choice(["BRCA1", "TP53", "LDLR"], n),  # string → cat
            "n_pathogenic_in_gene": rng.integers(0, 3000, n),                 # int   → numeric
            "af_raw":              rng.uniform(0, 1, n),
            "cadd_phred":          rng.uniform(0, 40, n),
        })
        y = rng.integers(0, 2, n).astype(np.int32)

        clf = CatBoostVariantClassifier(
            iterations=20, verbose=0, task_type="CPU",
            cat_feature_names=["gene_symbol", "n_pathogenic_in_gene"],
        )
        clf.fit(df, y)  # must not raise

        # n_pathogenic_in_gene is int64 — should NOT appear in cat_indices
        cat_cols = [df.columns[i] for i in clf._cat_indices]
        assert "n_pathogenic_in_gene" not in cat_cols, (
            f"n_pathogenic_in_gene (int64) was treated as categorical. "
            f"Categorical columns detected: {cat_cols}"
        )
        # gene_symbol is string — should be categorical
        assert "gene_symbol" in cat_cols

    def test_feature_names_preserved(self, fitted_classifier, small_variant_df):
        assert fitted_classifier._feature_names == list(small_variant_df.columns)

    def test_shap_values_shape(self, fitted_classifier, small_variant_df):
        shap = fitted_classifier.shap_values(small_variant_df)
        assert shap.shape == (len(small_variant_df), len(small_variant_df.columns))

    def test_top_shap_features_returns_list(self, fitted_classifier, small_variant_df):
        top = fitted_classifier.top_shap_features(small_variant_df, n=5)
        assert len(top) == 5
        assert all(isinstance(name, str) and isinstance(val, float) for name, val in top)

    def test_sklearn_estimator_interface(self):
        from genomic_variant_classifier.models.catboost_wrapper import CatBoostVariantClassifier
        clf = CatBoostVariantClassifier(iterations=20, verbose=0, task_type="CPU")
        params = clf.get_params()
        assert "iterations" in params
        clf.set_params(iterations=30)
        assert clf.iterations == 30

    def test_isotonic_calibration(self, small_variant_df, small_labels):
        from genomic_variant_classifier.models.catboost_wrapper import CatBoostVariantClassifier
        clf = CatBoostVariantClassifier(
            iterations=30, verbose=0, task_type="CPU", calibrate="isotonic",
            cat_feature_names=["gene_symbol", "consequence", "chrom", "review_status"],
        )
        clf.fit(small_variant_df, small_labels)
        proba = clf.predict_proba(small_variant_df)
        assert proba.shape == (len(small_labels), 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-4)

    def test_platt_calibration(self, small_variant_df, small_labels):
        from genomic_variant_classifier.models.catboost_wrapper import CatBoostVariantClassifier
        clf = CatBoostVariantClassifier(
            iterations=30, verbose=0, task_type="CPU", calibrate="platt",
            cat_feature_names=["gene_symbol", "consequence", "chrom", "review_status"],
        )
        clf.fit(small_variant_df, small_labels)
        proba = clf.predict_proba(small_variant_df)
        assert proba.shape[1] == 2

    def test_sample_weight_accepted(self, small_variant_df, small_labels):
        from genomic_variant_classifier.models.catboost_wrapper import CatBoostVariantClassifier
        weights = np.random.default_rng(0).uniform(0.5, 2.0, len(small_labels))
        clf = CatBoostVariantClassifier(
            iterations=20, verbose=0, task_type="CPU",
            cat_feature_names=["gene_symbol", "consequence", "chrom", "review_status"],
        )
        clf.fit(small_variant_df, small_labels, sample_weight=weights)
        assert clf._model is not None

    def test_save_load_native(self, fitted_classifier, small_variant_df, tmp_path):
        from genomic_variant_classifier.models.catboost_wrapper import CatBoostVariantClassifier
        model_path = tmp_path / "catboost_model.cbm"
        fitted_classifier.save_catboost_model(model_path)
        assert model_path.exists()
        loaded = CatBoostVariantClassifier.load_catboost_model(model_path)
        proba_orig   = fitted_classifier.predict_proba(small_variant_df)
        proba_loaded = loaded.predict_proba(small_variant_df)
        np.testing.assert_allclose(proba_orig, proba_loaded, atol=1e-5)

    def test_not_fitted_raises(self, small_variant_df):
        from genomic_variant_classifier.models.catboost_wrapper import CatBoostVariantClassifier
        from sklearn.exceptions import NotFittedError
        clf = CatBoostVariantClassifier(iterations=10, verbose=0, task_type="CPU")
        with pytest.raises((NotFittedError, Exception)):
            clf.predict_proba(small_variant_df)

    def test_auroc_above_random(self, small_variant_df):
        """CatBoost should do better than random on a predictable feature."""
        from genomic_variant_classifier.models.catboost_wrapper import CatBoostVariantClassifier
        from sklearn.metrics import roc_auc_score
        labels = (small_variant_df["cadd_phred"] > 20).astype(int).to_numpy()
        clf = CatBoostVariantClassifier(
            iterations=100, verbose=0, task_type="CPU",
            cat_feature_names=["gene_symbol", "consequence", "chrom", "review_status"],
        )
        clf.fit(small_variant_df, labels)
        proba = clf.predict_proba(small_variant_df)[:, 1]
        auroc = roc_auc_score(labels, proba)
        assert auroc > 0.65, f"CatBoost AUROC {auroc:.3f} unexpectedly low"


# ---------------------------------------------------------------------------
# Tests: ensemble integration (numeric-only fixture — no string columns)
# ---------------------------------------------------------------------------

class TestEnsembleIntegration:

    def test_catboost_in_base_estimators(self):
        """CatBoost should appear in base_estimators when installed."""
        from genomic_variant_classifier.models.variant_ensemble import VariantEnsemble, EnsembleConfig
        ensemble = VariantEnsemble(config=EnsembleConfig(skip_catboost=False))
        assert "catboost" in ensemble.base_estimators

    def test_skip_catboost_flag(self):
        """EnsembleConfig(skip_catboost=True) should exclude CatBoost."""
        from genomic_variant_classifier.models.variant_ensemble import VariantEnsemble, EnsembleConfig
        ensemble = VariantEnsemble(config=EnsembleConfig(skip_catboost=True))
        assert "catboost" not in ensemble.base_estimators

    def test_catboost_receives_dataframe_in_fit(self, numeric_variant_df, numeric_labels):
        """
        In ensemble.fit(), CatBoost should receive a DataFrame (not numpy)
        so its saved categorical feature indices are resolvable.

        Uses numeric_variant_df so RF/XGBoost/LightGBM can handle the
        .values conversion without encountering string data.
        """
        from genomic_variant_classifier.models.catboost_wrapper import CatBoostVariantClassifier
        received_types: list[str] = []
        orig_fit = CatBoostVariantClassifier.fit

        def spy_fit(self_inner, X, y, **kwargs):
            received_types.append(type(X).__name__)
            return orig_fit(self_inner, X, y, **kwargs)

        CatBoostVariantClassifier.fit = spy_fit
        try:
            from genomic_variant_classifier.models.variant_ensemble import VariantEnsemble, EnsembleConfig
            config   = EnsembleConfig(n_folds=2, skip_catboost=False)
                
            
            ensemble = VariantEnsemble(config=config)
            seq      = pd.Series(["ACGT" * 25] * len(numeric_labels))
            ensemble.fit(numeric_variant_df, seq, pd.Series(numeric_labels))
        finally:
            CatBoostVariantClassifier.fit = orig_fit

        assert "DataFrame" in received_types, (
            "CatBoost.fit() received a numpy array instead of a DataFrame — "
            "check the X_input dispatch in variant_ensemble.fit()."
        )

    def test_ensemble_predict_proba_with_catboost(self, numeric_variant_df, numeric_labels):
        """Ensemble with CatBoost should produce valid (n, 2) probabilities."""
        from genomic_variant_classifier.models.variant_ensemble import VariantEnsemble, EnsembleConfig
        config   = EnsembleConfig(n_folds=2, skip_catboost=False)
        ensemble = VariantEnsemble(config=config)
        seq      = pd.Series(["ACGT" * 25] * len(numeric_labels))
        ensemble.fit(numeric_variant_df, seq, pd.Series(numeric_labels))
        proba = ensemble.predict_proba(numeric_variant_df, seq)
        assert proba.shape == (len(numeric_labels), 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-4)
        assert "catboost" in ensemble.trained_models_


# ---------------------------------------------------------------------------
# Tests: InferencePipeline integration
# ---------------------------------------------------------------------------

class TestInferencePipelineWithCatBoost:

    def test_catboost_routed_to_dataframe(self, numeric_variant_df, numeric_labels):
        """
        InferencePipeline.predict_proba() must route CatBoost to DataFrame,
        all other models to numpy.

        Uses trained_models= (the actual InferencePipeline kwarg, not base_models=).
        """
        from genomic_variant_classifier.api.pipeline import InferencePipeline
        from genomic_variant_classifier.models.catboost_wrapper import CatBoostVariantClassifier

        mock_cb    = MagicMock(spec=CatBoostVariantClassifier)
        mock_cb.predict_proba.return_value = np.tile([0.1, 0.9], (len(numeric_variant_df), 1))
        mock_other = MagicMock()
        mock_other.predict_proba.return_value = np.tile([0.2, 0.8], (len(numeric_variant_df), 1))
        mock_meta  = MagicMock()
        mock_meta.predict_proba.return_value = np.tile([0.1, 0.9], (len(numeric_variant_df), 1))

        pipe = InferencePipeline(
            trained_models={"catboost": mock_cb, "lightgbm": mock_other},
            meta_learner=mock_meta,
        )
        pipe.predict_proba(numeric_variant_df)

        cb_arg = mock_cb.predict_proba.call_args[0][0]
        assert isinstance(cb_arg, pd.DataFrame), (
            f"CatBoost received {type(cb_arg).__name__} — expected DataFrame."
        )

        lgbm_arg = mock_other.predict_proba.call_args[0][0]
        assert isinstance(lgbm_arg, np.ndarray), (
            f"LightGBM received {type(lgbm_arg).__name__} — expected numpy array."
        )

    def test_pipeline_without_catboost_unaffected(self, numeric_variant_df):
        """Pipeline with no CatBoost model behaves identically to before."""
        from genomic_variant_classifier.api.pipeline import InferencePipeline
        mock_lgbm = MagicMock()
        mock_lgbm.predict_proba.return_value = np.tile([0.1, 0.9], (len(numeric_variant_df), 1))
        mock_meta = MagicMock()
        mock_meta.predict_proba.return_value = np.tile([0.1, 0.9], (len(numeric_variant_df), 1))

        pipe = InferencePipeline(
            trained_models={"lightgbm": mock_lgbm},
            meta_learner=mock_meta,
        )
        proba = pipe.predict_proba(numeric_variant_df)
        assert proba.shape[0] == len(numeric_variant_df)

        lgbm_arg = mock_lgbm.predict_proba.call_args[0][0]
        assert isinstance(lgbm_arg, np.ndarray)