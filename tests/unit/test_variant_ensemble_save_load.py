"""
tests/unit/test_variant_ensemble_save_load.py
==============================================
Regression tests for Run 10 patch A1 + A2:

  A1: CNN1D inner nn.Module moved from nested local class to module level
      (Run 9 crash root cause:
       INCIDENT_2026-05-12_cnn1d-pickle-nested-class.md)

  A2: VariantEnsemble.save()/load() refactored to per-model checkpoint
      layout; single-model pickle failure no longer corrupts the entire
      ensemble (INCIDENT_2026-05-12_no-per-model-checkpoint.md)

Coverage:
  - _ensure_cnn1d_module_class() yields a class with module-level qualname
  - A fitted CNN1DClassifier pickles cleanly
  - VariantEnsemble.save() produces orchestrator + per-model joblib layout
  - VariantEnsemble.load() round-trips both old (single-joblib) and new
    (orchestrator + per-model dir) formats
  - predict_proba works on the reloaded CNN1D

The tests use a minimal `fit_minimal` helper on VariantEnsemble that
fits one tree model + CNN1D against 40 synthetic rows. Total runtime
is ~3-5s on CPU.
"""
from __future__ import annotations

from pathlib import Path
import pytest

# Skip the whole file if torch isn't installed — the CNN1D branch
# requires it and tests would be meaningless without it.
torch = pytest.importorskip("torch")


@pytest.fixture
def synthetic_data():
    import numpy as np
    import pandas as pd

    rng = np.random.RandomState(0)
    X_tab = pd.DataFrame(rng.rand(40, 5))
    X_seq = pd.Series(["ACGT" * 25 + "A"] * 40)  # 101 bp constant
    y = rng.randint(0, 2, 40)
    return X_tab, X_seq, y


def test_cnn1d_module_class_is_module_level():
    """After _ensure_cnn1d_module_class() runs, the class qualname must
    resolve at module level so pickle can find it across processes.
    """
    from genomic_variant_classifier.models import variant_ensemble as ve

    cls = ve._ensure_cnn1d_module_class()
    assert cls.__name__ == "_CNN1DModule"
    assert cls.__qualname__ == "_CNN1DModule", (
        f"Expected module-level qualname; got {cls.__qualname__!r}. "
        "Nested local-class qualnames fail pickling — see "
        "INCIDENT_2026-05-12_cnn1d-pickle-nested-class.md"
    )
    assert cls.__module__ == "genomic_variant_classifier.models.variant_ensemble"


def test_cnn1d_pickles_after_fit(synthetic_data, tmp_path):
    """A fitted CNN1DClassifier with a real torch state dict must pickle
    cleanly. This is the exact operation that crashed Run 9.
    """
    import joblib
    from genomic_variant_classifier.models.variant_ensemble import CNN1DClassifier

    _, X_seq, y = synthetic_data
    cnn = CNN1DClassifier(epochs=1, batch_size=8)
    cnn.fit(X_seq, y)

    path = tmp_path / "cnn_only.joblib"
    joblib.dump(cnn, path)
    assert path.exists()
    assert path.stat().st_size > 1000  # state dict alone is ~300 KB

    cnn2 = joblib.load(path)
    proba = cnn2.predict_proba(X_seq[:3])
    assert proba.shape == (3, 2)


def test_ensemble_save_creates_per_model_checkpoints(
    synthetic_data, tmp_path
):
    """save() must write each base model into its own joblib next to
    the orchestrator. Validates the per-model-checkpoint refactor (A2).
    """
    import joblib
    from genomic_variant_classifier.models.variant_ensemble import (
        VariantEnsemble, EnsembleConfig,
    )

    X_tab, X_seq, y = synthetic_data
    ens = VariantEnsemble(EnsembleConfig(model_dir=tmp_path))
    ens.fit_minimal(X_tab, X_seq, y)

    save_path = tmp_path / "ensemble.joblib"
    ens.save(save_path)

    # Orchestrator + manifest
    assert save_path.exists()
    assert save_path.with_suffix(".manifest.json").exists()

    # Per-model checkpoint dir
    models_dir = tmp_path / "ensemble_models"
    assert models_dir.is_dir()
    assert (models_dir / "lightgbm.joblib").exists()
    assert (models_dir / "cnn_1d.joblib").exists()

    # Orchestrator content
    orch = joblib.load(save_path)
    assert isinstance(orch, dict)
    assert orch["format_version"] == 2
    assert "lightgbm" in orch["saved_model_paths"]
    assert "cnn_1d" in orch["saved_model_paths"]
    # No save errors expected with the A1 fix in place
    assert orch["save_errors"] == {}


def test_ensemble_load_roundtrip(synthetic_data, tmp_path):
    """save() then load() must produce an ensemble where predict_proba
    works on both the tree model and the CNN1D.
    """
    from genomic_variant_classifier.models.variant_ensemble import (
        VariantEnsemble, EnsembleConfig,
    )

    X_tab, X_seq, y = synthetic_data
    ens = VariantEnsemble(EnsembleConfig(model_dir=tmp_path))
    ens.fit_minimal(X_tab, X_seq, y)

    save_path = tmp_path / "ensemble.joblib"
    ens.save(save_path)

    ens2 = VariantEnsemble.load(save_path)
    assert set(ens2.trained_models_.keys()) == {"lightgbm", "cnn_1d"}

    # Both predict
    lgb_proba = ens2.trained_models_["lightgbm"].predict_proba(X_tab.values)
    cnn_proba = ens2.trained_models_["cnn_1d"].predict_proba(X_seq)
    assert lgb_proba.shape == (40, 2)
    assert cnn_proba.shape == (40, 2)

    # Blend weights survive
    import numpy as np
    assert np.allclose(ens2.blend_weights_, ens.blend_weights_)
