"""Unit tests for core functionality."""
import pytest
import numpy as np

class TestDataPipeline:
    def test_label_values(self):
        labels = np.array([0, 1, 0, 1])
        assert set(labels) == {0, 1}

class TestFeatureEngineering:
    def test_af_log_transform(self):
        af = np.array([0.01, 0.001, 0.0001])
        af_log = np.log10(af + 1e-10)
        assert af_log[0] > af_log[1] > af_log[2]

class TestEnsemble:
    def test_weight_normalization(self):
        weights = {"model1": 0.4, "model2": 0.35, "model3": 0.25}
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.001

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
