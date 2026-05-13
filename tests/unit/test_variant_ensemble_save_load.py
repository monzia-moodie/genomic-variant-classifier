"""
Regression tests for Run 10 Phase 1 patches to variant_ensemble.py.

A1: _CNN1D lifted to module-level _CNN1DModule  (pickle fix)
A2: VariantEnsemble.save() refactored to per-model checkpoints

History:
- 2026-05-13 run10_phase1_v2.zip: original tests shipped.
- 2026-05-13 run10_phase1_5b.zip: tests 3 + 4 rewritten. The original
  pair called ens.fit_minimal(...), a helper that exists in Claude's
  sandbox draft but was never shipped to production variant_ensemble.py.
  Symptom: AttributeError: 'VariantEnsemble' object has no attribute
  'fit_minimal'. Replaced with one consolidated test that exercises a
  real fit() on a restricted 2-model subset (lightgbm + cnn_1d).

Refs:
- docs/incidents/INCIDENT_2026-05-12_cnn1d-pickle-nested-class.md
- docs/incidents/INCIDENT_2026-05-12_no-per-model-checkpoint.md
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Synthetic fixture
# ---------------------------------------------------------------------------
@pytest.fixture
def synthetic_data():
    """60-row balanced synthetic dataset.

    Sized for:
    - 5-fold StratifiedKFold (default n_folds): 12 rows/fold, 6 of each class
    - CNN1D default batch_size: ~2 batches/fold
    - Full 2-model fit (lightgbm + cnn_1d) under ~15s on CPU
    """
    rng = np.random.default_rng(42)
    n = 60
    y = pd.Series(np.array([0, 1] * (n // 2)), name="label")
    X_tab = pd.DataFrame(
        rng.random((n, 6)), columns=[f"f{i}" for i in range(6)],
    )
    bases = np.array(list("ACGT"))
    X_seq = pd.Series(
        ["".join(rng.choice(bases, 101)) for _ in range(n)],
        name="fasta_seq",
    )
    return X_tab, X_seq, y


# ---------------------------------------------------------------------------
# Test 1: A1 qualname check (unchanged from 1.5)
# ---------------------------------------------------------------------------
def test_cnn1d_module_class_is_module_level():
    """A1: _CNN1DModule must resolve at module level, not as a nested
    local class inside CNN1DClassifier._build_model. Minimal smoke test
    for the pickle fix.
    """
    pytest.importorskip("torch")
    from genomic_variant_classifier.models.variant_ensemble import (
        _ensure_cnn1d_module_class,
    )
    cls = _ensure_cnn1d_module_class()
    assert cls.__qualname__ == "_CNN1DModule", (
        f"Expected qualname='_CNN1DModule', got '{cls.__qualname__}' "
        "— the class is still nested inside _build_model.<locals>"
    )
    assert cls.__module__ == (
        "genomic_variant_classifier.models.variant_ensemble"
    ), f"Wrong __module__: {cls.__module__}"


# ---------------------------------------------------------------------------
# Test 2: A1 direct pickle round-trip (unchanged from 1.5)
# ---------------------------------------------------------------------------
def test_cnn1d_pickles_after_fit(tmp_path):
    """A1: a fit CNN1DClassifier must round-trip through joblib.dump/load
    without PicklingError. Direct regression test for the Run 9 crash.
    """
    pytest.importorskip("torch")
    import joblib
    from genomic_variant_classifier.models.variant_ensemble import (
        CNN1DClassifier,
    )
    rng = np.random.default_rng(42)
    bases = np.array(list("ACGT"))
    X = pd.Series(
        ["".join(rng.choice(bases, 101)) for _ in range(30)]
    )
    y = pd.Series(np.array([0, 1] * 15))

    cnn = CNN1DClassifier(
        filters=4, kernel_size=3, dropout=0.0,
        batch_size=8, epochs=1, random_state=42,
    )
    cnn.fit(X, y)

    p = tmp_path / "cnn1d.joblib"
    joblib.dump(cnn, p)  # Previously raised PicklingError on _CNN1D
    assert p.exists() and p.stat().st_size > 0

    cnn2 = joblib.load(p)
    proba1 = cnn.predict_proba(X)
    proba2 = cnn2.predict_proba(X)
    np.testing.assert_allclose(proba1, proba2, atol=1e-5)


# ---------------------------------------------------------------------------
# Test 3 (rewritten in 1.5b): A1 + A2 end-to-end save/load
# ---------------------------------------------------------------------------
def test_ensemble_save_load_with_cnn1d(synthetic_data, tmp_path):
    """A1 + A2 end-to-end: fit a 2-model subset (lightgbm + cnn_1d), save,
    load via the classmethod, verify predict_proba on both branches.

    Restricting base_estimators to 2 models keeps full fit() under ~15s.
    CNN1D MUST be in the subset to exercise the pickle-fix code path —
    that's the whole point of this test.

    Rewritten 2026-05-13 (run10_phase1_5b.zip) from the broken
    test_ensemble_save_creates_per_model_checkpoints +
    test_ensemble_load_roundtrip pair that referenced a non-existent
    ens.fit_minimal() helper.
    """
    pytest.importorskip("torch")
    pytest.importorskip("lightgbm")
    from genomic_variant_classifier.models.variant_ensemble import (
        VariantEnsemble, EnsembleConfig,
    )
    X_tab, X_seq, y = synthetic_data

    # Build ensemble, then restrict to 2 base models BEFORE fit().
    # The full 8/11-model ensemble on 60 rows would take 60+ seconds even
    # with n_jobs=1; the 2-model subset is the minimal exercise of A1+A2.
    cfg = EnsembleConfig(n_jobs=1)
    ens = VariantEnsemble(cfg)
    keep = {"lightgbm", "cnn_1d"}
    ens.base_estimators = {
        k: v for k, v in ens.base_estimators.items() if k in keep
    }

    ens.fit(X_tab, X_seq, y)

    # After fit(), trained_models_ should contain both
    trained = set(ens.trained_models_.keys())
    assert trained == keep, (
        f"Expected {keep}, got {trained}. Did fit() skip a model?"
    )

    # A2 save() — pass a fresh subdirectory path. The A2 patch may write
    # either an orchestrator joblib + per-model files, or a directory of
    # joblibs. The test only asserts that load() can reconstruct what
    # save() wrote, agnostic to internal layout.
    save_path = tmp_path / "ensemble"
    ens.save(save_path)

    # Reload via classmethod
    ens2 = VariantEnsemble.load(save_path)

    # Verify both models present after load
    trained2 = set(ens2.trained_models_.keys())
    assert trained2 == keep, (
        f"After load: expected {keep}, got {trained2}. "
        "A2 per-model save/load is lossy."
    )

    # predict_proba must work on both branches: tabular routing for
    # lightgbm, sequence routing for cnn_1d. The dispatch logic lives
    # inside VariantEnsemble.predict_proba.
    proba1 = ens.predict_proba(X_tab, X_seq)
    proba2 = ens2.predict_proba(X_tab, X_seq)

    assert proba1.shape == (len(X_tab), 2), f"Got {proba1.shape}"
    assert proba2.shape == (len(X_tab), 2), f"Got {proba2.shape}"

    # Valid probabilities
    assert np.all((proba1 >= 0) & (proba1 <= 1)), \
        "predict_proba returned values outside [0, 1]"
    assert np.all((proba2 >= 0) & (proba2 <= 1))

    # Determinism: same fitted state must give identical predictions
    np.testing.assert_allclose(
        proba1, proba2, atol=1e-5,
        err_msg="Loaded ensemble gives different predictions than the "
                "original — save()/load() round-trip is lossy (A2 regression)",
    )
