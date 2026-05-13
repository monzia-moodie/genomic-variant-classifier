# INCIDENT 2026-05-12 — CNN1D pickle crash on nested local class

## Status

ROOT CAUSE IDENTIFIED — RESOLUTION SHIPPING IN PHASE 1 PATCH BUNDLE
(`run10_phase1_v2.zip`, patches A1 + A2). Pending apply + Run 10 dry-run
confirmation, status moves to RESOLVED.

## Summary

`scripts/run_phase2_eval.py` crashed in `ensemble.save()` after 11.4 h of
training on Vast.ai RTX 4090 instance 36588175. `joblib.dump(self, path)`
raised `_pickle.PicklingError: Can't pickle <class 'genomic_variant_classifier.models.variant_ensemble.CNN1DClassifier._build_model.<locals>._CNN1D'>:
it's not found as genomic_variant_classifier.models.variant_ensemble.CNN1DClassifier._build_model.<locals>._CNN1D`.
The locked-test AUROC was never produced; the joblib is corrupt; the
OOF blend numbers (AUROC 0.9916) were displayed in the training log
but no per-model OOF predictions or test metrics were persisted to disk.

## Cross-references

- `docs/sessions/SESSION_2026-05-12.md` — full Run 9 chronology incl. cost
  accounting (~$9.70 total, ~$4.30 of which was idle post-crash).
- `docs/CHANGELOG.md` 2026-05-12 entry — single-line summary + commits.
- `docs/incidents/INCIDENT_2026-05-12_no-per-model-checkpoint.md` —
  complementary INCIDENT covering the architectural decision
  (single-joblib save) that turned this pickle failure into a total-loss
  failure mode rather than a recoverable partial-success.

## Timeline

- **2026-04-16 (final).** CNN1DClassifier was migrated from TF/Keras to
  PyTorch (commit `38656bc` per CHANGELOG 2026-04-16). The migration
  defined the inner `nn.Module` subclass as a nested local class inside
  `CNN1DClassifier._build_model`. No test exercised
  `joblib.dump` of a fitted CNN1DClassifier — runs 6, 7, 8 trained
  CNN1D successfully but no save attempted (per CHANGELOG: "tabular_nn,
  cnn_1d, mc_dropout, deep_ensemble all skipped — no tensorflow on
  Vast PyTorch image" for run 8; CNN1D was excluded from save scope).
- **2026-04-30.** Splits regen on local laptop did not exercise the
  save path with CNN1D (`--skip-nn` skipped CNN1D from fit).
- **2026-05-12 20:13 UTC.** Run 9 launched on Vast.ai. CNN1D was
  included as one of 11 base estimators. Training completed at
  ~07:36 UTC 2026-05-13 (11.4 h).
- **2026-05-12 ~07:36 UTC.** `ensemble.save(outdir / "ensemble.joblib")`
  crashed with PicklingError. Script exited non-zero. Trap function in
  `launch_run9_vm.sh` fired with `TRAINING_STARTED=yes`, printed the
  manual-destroy reminder.
- **2026-05-12 ~10:00 UTC.** Monzia SCP'd `outputs/run9/` back to local.
  Attempted `vastai destroy instance 36588175` from local shell; hung
  (separate INCIDENT — see INCIDENT_2026-05-12_vastai-destroy-interactive.md).
- **2026-05-12 ~12:00 UTC.** Manual destroy via Vast.ai web console.

## Evidence

### Reproduction in sandbox (2026-05-13)

A minimal sandbox reproducer fit a CNN1DClassifier on 30 synthetic
sequences (1 epoch, batch=8), then attempted `joblib.dump`. Identical
PicklingError:

```
PicklingError: Can't pickle
<class 'genomic_variant_classifier.models.variant_ensemble.CNN1DClassifier._build_model.<locals>._CNN1D'>:
it's not found as genomic_variant_classifier.models.variant_ensemble.CNN1DClassifier._build_model.<locals>._CNN1D
```

This confirms the bug is fully deterministic, independent of dataset
size, and not specific to the Vast.ai environment.

### Source of the bug

`src/genomic_variant_classifier/models/variant_ensemble.py` (current HEAD
at session start, commit `3cfc039`) line ~150–180:

```python
def _build_model(self):
    import torch
    import torch.nn as nn

    torch.manual_seed(self.random_state)

    class _CNN1D(nn.Module):           # <-- nested local class
        def __init__(self, filters, kernel_size, dropout):
            super().__init__()
            self.net = nn.Sequential( ... )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    return _CNN1D(self.filters, self.kernel_size, self.dropout)
```

The class `_CNN1D` is defined inside the `_build_model` method, giving
it the qualified name
`CNN1DClassifier._build_model.<locals>._CNN1D`. Python's pickle protocol
resolves a class by `(module, qualname)` lookup; the `<locals>` segment
makes the class unresolvable from a fresh process and the qualname is
not stable across processes anyway.

## Root cause

**Nested local classes inside methods cannot be pickled.** This is a
well-known Python pickle constraint
(https://docs.python.org/3/library/pickle.html#what-can-be-pickled-and-unpickled).
The PyTorch migration in 2026-04-16 introduced the pattern without an
accompanying pickle smoke test; runs 6/7/8 did not exercise the save
path with CNN1D (skipped at fit time), so the latent bug survived
undetected until Run 9 where CNN1D was both fit AND saved.

## Remediation (shipped in Phase 1 patch bundle, patch A1)

Two changes in `src/genomic_variant_classifier/models/variant_ensemble.py`:

1. **Lift the `_CNN1D` class to module level** as `_CNN1DModule`. Use a
   lazy-global pattern that defers the `torch.nn` import until first
   use, preserving the graceful-degradation property that
   `import variant_ensemble` works without torch installed:

   ```python
   _CNN1DModule = None  # populated by _ensure_cnn1d_module_class()

   def _ensure_cnn1d_module_class():
       global _CNN1DModule
       if _CNN1DModule is not None:
           return _CNN1DModule
       import torch.nn as nn
       class _CNN1DModule(nn.Module):
           def __init__(self, filters, kernel_size, dropout):
               super().__init__()
               self.net = nn.Sequential( ... )
           def forward(self, x):
               return self.net(x).squeeze(-1)
       _CNN1DModule.__qualname__ = "_CNN1DModule"
       _CNN1DModule.__module__ = __name__
       globals()["_CNN1DModule"] = _CNN1DModule
       return _CNN1DModule
   ```

2. **Update `CNN1DClassifier._build_model`** to use the module-level
   factory:

   ```python
   def _build_model(self):
       import torch
       torch.manual_seed(self.random_state)
       cls = _ensure_cnn1d_module_class()
       return cls(self.filters, self.kernel_size, self.dropout)
   ```

After the patch, `_CNN1DModule.__qualname__ == "_CNN1DModule"` and
`_CNN1DModule.__module__ == "genomic_variant_classifier.models.variant_ensemble"`.
Pickle resolves the class cleanly.

## Verification (sandbox, 2026-05-13)

End-to-end save/load roundtrip in sandbox confirmed pickle works:
- `joblib.dump(cnn1d_instance, ...)` produces a 309,523-byte file
- `joblib.load(...)` reconstitutes the model
- `model.predict_proba(seqs)` returns expected `(N, 2)` shape

Regression test shipped at
`tests/unit/test_variant_ensemble_save_load.py` (in
`run10_phase1_v2.zip`):

- `test_cnn1d_module_class_is_module_level` — asserts qualname fixup
- `test_cnn1d_pickles_after_fit` — fit + dump + load + predict roundtrip
- `test_ensemble_save_creates_per_model_checkpoints` — A2 layout check
- `test_ensemble_load_roundtrip` — full ensemble round-trip

## Lessons

- **Pickling is part of the API contract.** Any sklearn-style estimator
  in this codebase will be `joblib.dump`'d at training-end. If the
  estimator contains nested classes, lambdas, or closures referencing
  local variables, it will fail to pickle. The contract must be
  exercised by tests at every such estimator's introduction.
- **Lazy imports do not require nested class definitions.** The
  lazy-global pattern (module-level placeholder, populate on first
  call) gives the same import-time-deferral semantics as a nested
  class while preserving pickle resolvability. This pattern should be
  adopted for any future torch-backed nn.Module subclass in the
  codebase.
- **The PyTorch migration in 2026-04-16 introduced the bug 26 days
  before it was discovered, because the save path was never exercised
  with CNN1D in runs 6/7/8.** Coverage of the save path is now part of
  the Phase 1 regression test suite.

## Sign-off

INCIDENT moves to RESOLVED when:
- Phase 1 patch bundle (`run10_phase1_v2.zip`) is applied to HEAD
- `pytest tests/unit/test_variant_ensemble_save_load.py` passes
- A Run 10 dry-run (`--max-train 2000`) completes through
  `ensemble.save()` without PicklingError
