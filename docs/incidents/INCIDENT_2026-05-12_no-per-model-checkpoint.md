# INCIDENT 2026-05-12 — Single-joblib save architecture (no per-model checkpoint)

## Status

ROOT CAUSE IDENTIFIED — RESOLUTION SHIPPING IN PHASE 1 PATCH BUNDLE
(`run10_phase1_v2.zip`, patch A2). Pending apply + Run 10 dry-run
confirmation, status moves to RESOLVED.

## Summary

`VariantEnsemble.save()` serializes the entire ensemble as a single
`joblib.dump(self, path)` call. When that single call fails (Run 9: a
nested-class PicklingError in CNN1D — see
`INCIDENT_2026-05-12_cnn1d-pickle-nested-class.md`), the resulting
joblib is corrupt and **all base models, the meta-learner, and the
blend weights are lost** even though they were all fit successfully
during the preceding 11.4 h of GPU training.

Run 9 demonstrated this empirically: 11 base models trained
successfully, OOF blend AUROC 0.9916 was computed and logged, the
Nelder-Mead blend weights were optimized, the meta-learner was fit on
OOF predictions — and zero of those artifacts persisted to disk
because the save crashed before completing.

## Cross-references

- `docs/incidents/INCIDENT_2026-05-12_cnn1d-pickle-nested-class.md` —
  the proximate cause (the pickle failure that triggered this
  total-loss failure mode). That INCIDENT covers the bug in
  `_CNN1D`. This INCIDENT covers the architectural decision that
  turned a single-model bug into a whole-ensemble loss.
- `docs/sessions/SESSION_2026-05-12.md` §Failed and §Run 10 plan
  bullet 1f.
- `docs/CHANGELOG.md` 2026-05-12 entry.
- `docs/incidents/INCIDENT_2026-04-29_gcp-billing-deletion.md` —
  similar lesson, different domain: a single point of architecture
  (centralized GCS) that, when it failed, took the entire data flow
  with it. Resolution there was SCP-only redundancy.

## Empirical evidence (Run 9 outcome)

From `docs/sessions/SESSION_2026-05-12.md` §Results:

- OOF blend AUROC: **0.9916**
- LR stacker AUROC: 0.9911
- Best single base (lightgbm): 0.9911
- Δ blend over best single: +0.0005 — within noise floor

These numbers were displayed in the training log on Vast.ai but were
not persisted to disk. SCP-back retrieved zero usable artifacts beyond
the partial training log itself. The 11.4 h of GPU compute (~$5.40 at
$0.473/hr on RTX 4090) produced no reusable model. A follow-up run
must repeat the entire training.

(Note: per-model OOF AUROC table and Nelder-Mead blend weight dict were
displayed in chat at training time and are recoverable from chat
scrollback. Backfill pending log retrieval — see verification script in
`phase1_5/scripts/run9_outputs_audit.ps1`.)

## Recovery status (audit run 2026-05-13)

The Run 9 outputs audit recovered partial proxy data:

- **No Run 9 training log was SCP'd back from Vast.ai before instance
  destruction.** `outputs/run9/` does not exist on local disk;
  `outputs/run9_ready/` exists but is the 2026-04-30 CPU regen artifacts,
  not Run 9 (2026-05-12).
- **8 of 11 base-model OOF AUROCs are recoverable as 2026-04-30 proxies**
  from `outputs/run9_ready/regen.log` lines 82-131 — same splits, same
  base-model implementations.
- **4 base-model AUROCs are NOT recoverable from local disk** (`svm`,
  `kan`, `tabular_nn`, `cnn_1d` were skipped in the 04-30 regen by
  `--skip-nn --skip-svm --skip-kan`; they only ran on Vast.ai 05-12).
- **11-dim Nelder-Mead blend weight dict is NOT recoverable** beyond the
  qualitative statements in `SESSION_2026-05-12.md` §Scientific
  implications (kan/tabular_nn/logistic_regression at 0% weight;
  cnn_1d at ~10% weight).
- **Locked test AUROC was NEVER computed** — script crashed at
  `ensemble.save()` before reaching `evaluate()`. Patch B2 in
  `run10_phase1_v2.zip` reorders so test eval runs BEFORE save.

### Recovered 2026-04-30 proxy table

| Model               | OOF AUROC (04-30 proxy) |
|---------------------|------------------------|
| lightgbm            | 0.9911                 |
| xgboost             | 0.9908                 |
| catboost            | 0.9900                 |
| gradient_boosting   | 0.9889                 |
| random_forest       | 0.9881                 |
| deep_ensemble       | 0.9872                 |
| mc_dropout          | 0.9870                 |
| logistic_regression | 0.9849                 |
| svm                 | NOT RECOVERABLE        |
| kan                 | NOT RECOVERABLE        |
| tabular_nn          | NOT RECOVERABLE        |
| cnn_1d              | NOT RECOVERABLE        |
| **OOF blend**       | **0.9915** (04-30)     |
| **LR stacker**      | **0.9907** (04-30)     |

### Comparison: 04-30 (8 models) vs 05-12 (11 models)

|                           | 04-30 (8 models) | 05-12 (11 models, Run 9) |
|---------------------------|------------------|--------------------------|
| OOF blend AUROC           | 0.9915           | 0.9916                   |
| LR stacker AUROC          | 0.9907           | 0.9911                   |
| Best single (lightgbm)    | 0.9911           | 0.9911                   |
| Δ blend over best single  | +0.0004          | +0.0005                  |
| Models added (vs 04-30)   | —                | svm, kan, tabular_nn, cnn_1d |
| Blend delta from addition | —                | **+0.0001**              |

**Scientific finding:** the 4 added models contributed approximately
**+0.0001 AUROC** to the blend over the 8-model baseline. This is at
or below noise floor by any reasonable bootstrap criterion. Run 10
pruning decision (currently keep-all per SESSION §2 revision) should
treat this as the prior: drop kan, tabular_nn, svm unless bootstrap
CI shows a meaningful contribution. cnn_1d is a separate case (pickle
fix changes the trained state; OOF AUROC ~0.5 was placeholder-driven).

## Source of the bug

`src/genomic_variant_classifier/models/variant_ensemble.py`
(pre-Phase-1 patch, current HEAD):

```python
def save(self, path: Optional[Path] = None) -> None:
    import joblib
    path = Path(path or self.config.model_dir / "ensemble.joblib")
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(self, path)            # <-- single pickle of entire ensemble
    _write_model_manifest(path)
    logger.info("Ensemble saved to %s", path)

@classmethod
def load(cls, path: Path) -> "VariantEnsemble":
    import joblib
    return joblib.load(path)
```

If any of the 11 base models, the meta-learner, the blend weights, the
OOF predictions, or the EnsembleConfig contains an unpicklable object,
the entire save fails atomically. No partial output is produced.

## Root cause

**Single-point-of-failure architecture.** The save method was written
to optimize for the success path: one file, one call, one load. The
failure path (any one component fails to pickle) was not designed for.
This is the same architectural anti-pattern as centralized GCS for
artifact storage (resolved by SCP-only redundancy in
INCIDENT_2026-04-29).

## Remediation (shipped in Phase 1 patch bundle, patch A2)

Refactor `save()` to a two-tier layout:
1. **Per-model checkpoints** in `<ensemble_path>_models/`, one joblib per
   base model. Each is saved in its own try/except so a single failure
   does NOT block subsequent saves.
2. **Thin orchestrator joblib** at `<ensemble_path>` containing the
   meta-learner, blend weights, OOF predictions, config, and a
   `saved_model_paths` manifest that names each surviving base model.
3. **`save_errors` dict** in the orchestrator records which models
   failed to pickle and why; logged loud at save time so partial-success
   cases are visible.

`load()` is back-compatible with the pre-Phase-1 single-joblib format
(detects by inspecting the object) AND handles the new
format-version=2 orchestrator. Each base model is loaded in its own
try/except so a missing or corrupt per-model file doesn't block load
of survivors.

## Verification (sandbox, 2026-05-13)

End-to-end save/load roundtrip in sandbox confirmed the resilience
property:

- With CNN1D pickle bug NOT fixed: save() writes lightgbm.joblib
  (5,044 bytes) cleanly; cnn_1d.joblib save fails and is logged loud;
  orchestrator.joblib lists `save_errors: {"cnn_1d": "PicklingError: ..."}`;
  load() reloads the surviving lightgbm model and warns about the
  missing cnn_1d.
- With both A1 (CNN1D module-level) AND A2 (per-model save) applied:
  save() writes both lightgbm (5,044 bytes) AND cnn_1d (309,523 bytes)
  cleanly; orchestrator records both as saved; load() reloads both;
  predict_proba works on both reloaded models.

Regression test shipped at
`tests/unit/test_variant_ensemble_save_load.py` (in
`run10_phase1_v2.zip`).

## Defensive flush of scientific artifacts (Phase 1 patch B2 + B3)

Even with the per-model checkpoint save fix, a corner case remains: if
the orchestrator joblib itself fails to pickle (e.g., a future bug in
the meta-learner), the scientific artifacts (per-model AUROCs, OOF
predictions, test metrics) could still be lost. Patches B2 and B3 in
the Phase 1 bundle relocate the test-set evaluation and the
`metrics.json` + `oof_predictions.parquet` + per-model CSV flushes to
run **BEFORE** `ensemble.save()`. After Phase 1, the order is:

1. Train all base models
2. Train meta-learner
3. Fit blend weights
4. Evaluate on test + val
5. Flush `metrics.json`, OOF parquet, per-model CSVs to disk
6. **Only then** call `ensemble.save()`

This guarantees that even a catastrophic save failure leaves the
scientific record intact on disk.

## Lessons

- **Architectures with single points of failure for data are
  cost-asymmetric.** A defensive per-component save has marginal extra
  complexity but cuts the failure-mode cost from "lose 11 h of GPU
  compute" to "lose one base model's compute". For training runs that
  cost > $5, this is always the right tradeoff.
- **Order of operations matters for cost-recoverability.** Anything
  that can be flushed to disk should be flushed before anything that
  could crash. This is the same principle as journaled filesystems:
  small frequent writes beat one big write.
- **The Run 9 outcome is not a regression — the bug was always
  present.** No prior run exercised the save path with CNN1D included
  in the fit set (runs 6-8 explicitly skipped CNN1D). Run 9 was the
  first time the latent architectural fragility met a real failure
  mode. The Phase 1 patch bundle prevents the next class of failure
  from causing the same cost-asymmetric loss.

## Sign-off

INCIDENT moves to RESOLVED when:
- Phase 1 patch bundle (`run10_phase1_v2.zip`) is applied
- `pytest tests/unit/test_variant_ensemble_save_load.py::test_ensemble_save_creates_per_model_checkpoints` passes
- A Run 10 dry-run (`--max-train 2000`) produces:
  - `outputs/run10_dryrun/ensemble.joblib` (orchestrator)
  - `outputs/run10_dryrun/ensemble_models/*.joblib` (per-model dir with multiple files)
  - `outputs/run10_dryrun/metrics.json` written BEFORE the joblib timestamps
  - `outputs/run10_dryrun/oof_predictions.parquet` written
