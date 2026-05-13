# Run 9 master log — partial recovery

**Captured:** 2026-05-13 via PowerShell SSH session into Vast.ai instance
36588175 before the instance was destroyed.

**Source command:**
```powershell
ssh -i $KEY -p 51089 root@213.181.122.2 'tail -100 /workspace/run9_master.log'
```

**Recovery context:**

The Vast.ai instance ran Run 9 ablations from 2026-05-12 04:08 UTC to
15:32 UTC (~11.4 hours). `ensemble.save()` then crashed with a
`PicklingError` on a nested local class (root cause documented in
[`docs/incidents/INCIDENT_2026-05-12_cnn1d-pickle-nested-class.md`](../../docs/incidents/INCIDENT_2026-05-12_cnn1d-pickle-nested-class.md)).
The instance remained up for an unclear interval before being destroyed.
Subsequent SCP attempts (visible in session transcript) returned
`Connection refused` on port 51089 — instance gone.

The full 273-line / 264 KB `/workspace/run9_master.log` was never
SCP'd back. The recovery below is the last 100 lines, retrieved via
`tail -100` over SSH during a still-live window. The earlier 173 lines
(04:08 UTC start through 14:53 UTC start of `deep_ensemble`) are lost.

The 1.4 GB `ensemble.joblib` written by the crashing `save()` call
(`-rw-rw-r-- 1 root root 1483762656 May 12 15:32 ...`) is also lost.

## Reconstructed timeline (from earlier SSH queries + recovery log)

| Time (UTC) | Event |
|---|---|
| 2026-05-12 04:08:00 | `==> full @` (training start) |
| ~04:59:41 | `tabular_nn OOF AUROC: 0.9870` |
| ~05:08:10 | `cnn_1d OOF AUROC: 0.5000` *(suspected silent failure — see below)* |
| ~13:29:43 | `kan OOF AUROC: 0.9855` |
| ~13:47:33 | `mc_dropout OOF AUROC: 0.9870` |
| ~14:53:34 | DeepEnsemble: fitting member 1/5 *(first round)* |
| ~15:06:54 | DeepEnsemble: fitting member 5/5 *(first round)* |
| ~15:09:00 | DeepEnsemble: fitting member 1/5 *(second round)* |
| ~15:23:31 | DeepEnsemble: fitting member 5/5 *(second round)* |
| 15:27:35 | `deep_ensemble OOF AUROC: 0.9872` |
| 15:27:35 | Training meta-learner on 11 base-model OOF columns |
| 15:27:40 | Running Nelder-Mead blend weight search |
| 15:32:06 | Blend weights computed |
| 15:32:07 | `OOF blend AUROC: 0.9916 (LR stacker: 0.9911, Δ=0.0005)` |
| 15:32:07 | `Fit complete in 684.1 min` |
| 15:32:07 | `ensemble.save()` called → `PicklingError` |
| 15:32:11 | `ABORT: full failed` |

## Per-model OOF AUROCs (from recovery + earlier SSH queries)

| Model | OOF AUROC |
|---|---|
| lightgbm | 0.9911 *(best single model, from earlier session memory)* |
| xgboost | 0.9908 *(from memory)* |
| catboost | 0.9900 *(from memory)* |
| gradient_boosting | 0.9889 *(from memory)* |
| random_forest | 0.9881 *(from memory)* |
| deep_ensemble | 0.9872 |
| tabular_nn | 0.9870 |
| mc_dropout | 0.9870 |
| kan | 0.9855 |
| logistic_regression | 0.9849 *(from memory)* |
| **cnn_1d** | **0.5000** *(no signal — pickle-bug source class also misbehaved at fit time)* |

**Headline numbers:**
- OOF blend AUROC: **0.9916**
- LR stacker AUROC: **0.9911**
- Δ (blend − LR stacker): **+0.0005** (within noise — argues against the
  stacker buying anything over a simple blend)
- Locked test AUROC: **NEVER PRODUCED** (lost to `save()` crash)
- Fit wall-clock: **684.1 min** (~11.4 hours) on Vast.ai RTX 4090
- Instance hours billed: ~16h (~$9.70, including ~$4.30 idle post-crash)

## Blend weights (from recovery log)

```python
{
    'random_forest':       0.3377,
    'xgboost':             0.0434,
    'lightgbm':            0.2933,
    'logistic_regression': 0.0,
    'gradient_boosting':   0.1175,
    'catboost':            0.0677,
    'tabular_nn':          0.0,
    'cnn_1d':              0.104,
    'kan':                 0.0,
    'mc_dropout':          0.0002,
    'deep_ensemble':       0.0364,
}
```

Note: `cnn_1d` gets weight 0.104 in the blend despite an OOF AUROC of
0.5000 (random). This is the Nelder-Mead optimiser finding diversity
value in an essentially-random column, not predictive signal. Reproduces
a known weak-base-learner pathology of weighted blending. Run 10 should
look at this critically; if cnn_1d's AUROC stays at 0.5 it should be
dropped from the blend entirely.

## Recovery log (SSH tail -100 verbatim)

```text
/workspace/genomic-variant-classifier/src/genomic_variant_classifier/models/mc_dropout.py:87: RuntimeWarning: invalid value encountered in multiply
  entropy_per_pass = -(clipped * np.log(clipped) + (1 - clipped) * np.log(1 - clipped))
2026-05-12T14:53:34 INFO    genomic_variant_classifier.models.mc_dropout: DeepEnsemble: fitting member 1/5 ...
2026-05-12T14:57:27 INFO    genomic_variant_classifier.models.mc_dropout: DeepEnsemble: fitting member 2/5 ...
2026-05-12T15:01:07 INFO    genomic_variant_classifier.models.mc_dropout: DeepEnsemble: fitting member 3/5 ...
2026-05-12T15:04:18 INFO    genomic_variant_classifier.models.mc_dropout: DeepEnsemble: fitting member 4/5 ...
2026-05-12T15:06:54 INFO    genomic_variant_classifier.models.mc_dropout: DeepEnsemble: fitting member 5/5 ...
/workspace/genomic-variant-classifier/src/genomic_variant_classifier/models/mc_dropout.py:87: RuntimeWarning: divide by zero encountered in log
  entropy_per_pass = -(clipped * np.log(clipped) + (1 - clipped) * np.log(1 - clipped))
/workspace/genomic-variant-classifier/src/genomic_variant_classifier/models/mc_dropout.py:87: RuntimeWarning: invalid value encountered in multiply
  entropy_per_pass = -(clipped * np.log(clipped) + (1 - clipped) * np.log(1 - clipped))
2026-05-12T15:09:00 INFO    genomic_variant_classifier.models.mc_dropout: DeepEnsemble: fitting member 1/5 ...
2026-05-12T15:11:35 INFO    genomic_variant_classifier.models.mc_dropout: DeepEnsemble: fitting member 2/5 ...
2026-05-12T15:17:24 INFO    genomic_variant_classifier.models.mc_dropout: DeepEnsemble: fitting member 3/5 ...
2026-05-12T15:21:08 INFO    genomic_variant_classifier.models.mc_dropout: DeepEnsemble: fitting member 4/5 ...
2026-05-12T15:23:31 INFO    genomic_variant_classifier.models.mc_dropout: DeepEnsemble: fitting member 5/5 ...
2026-05-12T15:27:35 INFO    genomic_variant_classifier.models.variant_ensemble:   deep_ensemble OOF AUROC: 0.9872
2026-05-12T15:27:35 INFO    genomic_variant_classifier.models.variant_ensemble: Training meta-learner on 11 base-model OOF columns ...
2026-05-12T15:27:40 INFO    genomic_variant_classifier.models.variant_ensemble: Running Nelder-Mead blend weight search ...
2026-05-12T15:32:06 INFO    genomic_variant_classifier.models.variant_ensemble: Blend weights: {'random_forest': 0.3377, 'xgboost': 0.0434, 'lightgbm': 0.2933, 'logistic_regression': 0.0, 'gradient_boosting': 0.1175, 'catboost': 0.0677, 'tabular_nn': 0.0, 'cnn_1d': 0.104, 'kan': 0.0, 'mc_dropout': 0.0002, 'deep_ensemble': 0.0364}
2026-05-12T15:32:07 INFO    genomic_variant_classifier.models.variant_ensemble: OOF blend AUROC: 0.9916  (LR stacker: 0.9911  Δ=0.0005)
2026-05-12T15:32:07 INFO    run9_ablations: Fit complete in 684.1 min
Traceback (most recent call last):
  File "/workspace/genomic-variant-classifier/scripts/run9_ablations.py", line 780, in <module>
    sys.exit(main())
             ^^^^^^
  File "/workspace/genomic-variant-classifier/scripts/run9_ablations.py", line 631, in main
    ensemble.save(args.output_dir / "models" / "ensemble.joblib")
  File "/workspace/genomic-variant-classifier/src/genomic_variant_classifier/models/variant_ensemble.py", line 1250, in save
    joblib.dump(self, path)
  File "/venv/main/lib/python3.12/site-packages/joblib/numpy_pickle.py", line 600, in dump
    NumpyPickler(f, protocol=protocol).dump(value)
  File "/venv/main/lib/python3.12/pickle.py", line 484, in dump
    self.save(obj)
  File "/venv/main/lib/python3.12/site-packages/joblib/numpy_pickle.py", line 395, in save
    return Pickler.save(self, obj)
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/pickle.py", line 601, in save
    self.save_reduce(obj=obj, *rv)
  File "/venv/main/lib/python3.12/pickle.py", line 715, in save_reduce
    save(state)
  File "/venv/main/lib/python3.12/site-packages/joblib/numpy_pickle.py", line 395, in save
    return Pickler.save(self, obj)
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/pickle.py", line 558, in save
    f(self, obj)  # Call unbound method with explicit self
    ^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/pickle.py", line 990, in save_dict
    self._batch_setitems(obj.items())
  File "/venv/main/lib/python3.12/pickle.py", line 1014, in _batch_setitems
    save(v)
  File "/venv/main/lib/python3.12/site-packages/joblib/numpy_pickle.py", line 395, in save
    return Pickler.save(self, obj)
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/pickle.py", line 558, in save
    f(self, obj)  # Call unbound method with explicit self
    ^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/pickle.py", line 990, in save_dict
    self._batch_setitems(obj.items())
  File "/venv/main/lib/python3.12/pickle.py", line 1014, in _batch_setitems
    save(v)
  File "/venv/main/lib/python3.12/site-packages/joblib/numpy_pickle.py", line 395, in save
    return Pickler.save(self, obj)
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/pickle.py", line 601, in save
    self.save_reduce(obj=obj, *rv)
  File "/venv/main/lib/python3.12/pickle.py", line 715, in save_reduce
    save(state)
  File "/venv/main/lib/python3.12/site-packages/joblib/numpy_pickle.py", line 395, in save
    return Pickler.save(self, obj)
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/pickle.py", line 558, in save
    f(self, obj)  # Call unbound method with explicit self
    ^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/pickle.py", line 990, in save_dict
    self._batch_setitems(obj.items())
  File "/venv/main/lib/python3.12/pickle.py", line 1014, in _batch_setitems
    save(v)
  File "/venv/main/lib/python3.12/site-packages/joblib/numpy_pickle.py", line 395, in save
    return Pickler.save(self, obj)
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/pickle.py", line 601, in save
    self.save_reduce(obj=obj, *rv)
  File "/venv/main/lib/python3.12/pickle.py", line 685, in save_reduce
    save(cls)
  File "/venv/main/lib/python3.12/site-packages/joblib/numpy_pickle.py", line 395, in save
    return Pickler.save(self, obj)
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/pickle.py", line 558, in save
    f(self, obj)  # Call unbound method with explicit self
    ^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/pickle.py", line 1172, in save_type
    return self.save_global(obj)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/pickle.py", line 1087, in save_global
    raise PicklingError(
_pickle.PicklingError: Can't pickle <class 'genomic_variant_classifier.models.variant_ensemble.CNN1DClassifier._build_model.<locals>._CNN1D'>: it's not found as genomic_variant_classifier.models.variant_ensemble.CNN1DClassifier._build_model.<locals>._CNN1D
==> full exit 1
==> ABORT: full failed at 2026-05-12 15:32:11 UTC
==> SCP outputs back from local NOW, then destroy from web console
```

## Bug filed

- [`docs/incidents/INCIDENT_2026-05-12_cnn1d-pickle-nested-class.md`](../../docs/incidents/INCIDENT_2026-05-12_cnn1d-pickle-nested-class.md)
  filed 2026-05-12
- [`docs/incidents/INCIDENT_2026-05-12_no-per-model-checkpoint.md`](../../docs/incidents/INCIDENT_2026-05-12_no-per-model-checkpoint.md)
  filed 2026-05-12 (would have salvaged 10/11 trained models if A2 had been
  in place)

## Phase 1 fixes verified locally

Commit `66593d6` (Phase 1) plus the test-bundle iterations `f64c024`,
`0178fe1`, `633e7d0`, `e07e3d8` (Phase 1.5b-e) ship and verify the
fixes. As of commit `e07e3d8` the regression suite is green:

- `tests/unit/test_variant_ensemble_save_load.py` (3 tests) — directly
  validates A1 (module-level class qualname) + A2 (per-model checkpoint
  save/load round-trip on a fitted CNN1D in a 2-model ensemble).
- `tests/unit/test_lovd_annotation_reaches_training_matrix.py` (2 tests)
  — validates B1 (LOVD annotation reaches the training matrix when
  `--lovd-path` is supplied, and is identically zero otherwise — the
  Run 9 silent-zero failure mode).
- Full unit-test sweep: 501 / 501 PASSED in 221 s on 2026-05-13.

## Open follow-up

The `cnn_1d OOF AUROC: 0.5000` line is anomalous and was not chased
during the Run 9 post-mortem. The same class (`CNN1D._build_model.<locals>._CNN1D`)
that broke pickle may also have failed silently at fit time, producing
the no-signal AUROC. Worth re-inspecting after Run 10's locked test
result confirms or refutes the same behaviour. If cnn_1d remains at
~0.5 with the A1 fix applied, the class itself has a fit-side bug.
