---
date: 2026-05-16
severity: LOW (locked test AUROC already written to disk before crash)
status: OPEN
related_run: Run 10
related_code: scripts/run9_ablations.py:705
---

# INCIDENT 2026-05-16: OOF export length mismatch

## Summary

Run 10 training completed successfully (all 11 models + ensemble save +
locked test evaluation), then crashed at line 705 of `run9_ablations.py`
during the non-critical OOF training-set export step.

## Error

```
File "scripts/run9_ablations.py", line 705, in main
    oof["label"] = y_train.values
ValueError: Length of values (1197216) does not match length of index (1017633)
```

## Root cause (under investigation)

`ensemble.oof_predictions_` has 1,017,633 rows. `y_train` has 1,197,216 rows.
The ratio is 1,017,633 / 1,197,216 = 0.8500, i.e. exactly 85%.

The comment at line 691 says `oof_predictions_` "requires Patch 1
(VariantEnsemble.oof_predictions_)". This attribute is set somewhere in the
ensemble's `fit()` method. The 85% ratio suggests one of:

- **(a)** The ensemble uses an internal 85/15 train/val split for Nelder-Mead
  blend weight optimization, and `oof_predictions_` stores only the validation
  portion's predictions (not the full k-fold OOF matrix).
- **(b)** The k-fold OOF matrix is built for all rows, but some rows are
  subsequently filtered (e.g., rows where all models predicted 0.5 fallback).
- **(c)** `oof_predictions_` is set by a code path that uses `n_folds - 1`
  folds for OOF (reserving one fold for stacker validation).

**Investigation step**: Run:
```powershell
Select-String -Path "src\genomic_variant_classifier\models\variant_ensemble.py" -Pattern "oof_predictions_" -Context 5,5
```
to see where and how `oof_predictions_` is assigned.

## Impact

**LOW**. The crash occurred AFTER:
- All 11 per-model checkpoints were saved (15:38 UTC)
- `ensemble.joblib` was saved (89 MB, 15:38 UTC)
- `test_predictions.parquet` was written (349,067 rows × 20 cols, 15:50:01 UTC)
- `eval_report.json` was written (locked test AUROC 0.98163, 15:50:02 UTC)
- `calibration.parquet` was written (15:50:02 UTC)

The only missing artifact is the OOF training-set predictions parquet
(`oof_train.parquet`), which is used for post-hoc analysis but is NOT required
for the locked test AUROC (the primary Run 10 deliverable).

## Fix (proposed)

Defensive alignment at line 705:

```python
# Replace line 705:
#   oof["label"] = y_train.values
# With:
if len(oof) == len(y_train):
    oof["label"] = y_train.values
elif hasattr(ensemble, "oof_indices_"):
    oof["label"] = y_train.values[ensemble.oof_indices_]
else:
    logger.warning(
        "OOF length (%d) != y_train length (%d) and no oof_indices_ "
        "available; labeling first %d rows by position.",
        len(oof), len(y_train), len(oof),
    )
    oof["label"] = y_train.values[: len(oof)]
```

The proper fix requires understanding WHERE `oof_predictions_` is set in
the ensemble and ensuring it covers all training rows. The defensive fix
above prevents the crash while preserving whatever OOF data is available.

## Resolution criteria

- Grep confirms root cause (a), (b), or (c) above
- Fix applied and tested locally
- Run 10a completes without OOF crash
- Move to RESOLVED
