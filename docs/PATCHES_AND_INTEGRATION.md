# Patches to Existing Files — Run 9 Integration

**Status:** v2, 2026-04-19. Adds Patch 6 (gnn_score persistence) to the
five patches from v1. All patches are additive; reverting any one does
not affect the others.

Apply in order. Each patch has a verification step at the end.

---

## Patch 1 — `src/models/variant_ensemble.py`: expose OOF predictions

**Why:** the mainline `VariantEnsemble.fit()` computes OOF predictions via
`cross_val_predict`, uses them to train the stacker, and then discards
them. Run 9's Rule-5 artefact `oof_predictions.parquet` requires access
to that matrix, so we store it on `self` before the meta-learner call.

**Impact:** adds two instance attributes (`oof_predictions_`,
`oof_model_names_`). No existing caller reads these. Zero behaviour
change for code that doesn't use them.

### 1a. Find the exact anchor

```powershell
cd C:\Projects\genomic-variant-classifier
Select-String -Path src\models\variant_ensemble.py `
    -Pattern "logger\.info\(.Training meta-learner" -Context 2,2
```

Expected output (approximate line numbers — confirm against your file):

```
src\models\variant_ensemble.py:477:        oof_preds = oof_preds[:, valid_cols]
src\models\variant_ensemble.py:478:
src\models\variant_ensemble.py:479:        logger.info("Training meta-learner on %d base-model OOF columns ...", len(valid_cols))
src\models\variant_ensemble.py:480:        self.meta_learner.fit(oof_preds, y_arr)
```

### 1b. Apply the patch

In `src/models/variant_ensemble.py`, find the line:

```python
        oof_preds = oof_preds[:, valid_cols]
```

And insert FIVE lines AFTER it, BEFORE the existing
`logger.info("Training meta-learner...")`:

```python
        oof_preds = oof_preds[:, valid_cols]

        # Expose OOF matrix for Rule-5 artefacts (Run 9+). Downstream
        # writers (scripts/run9_ablations.py) read these attributes.
        self.oof_predictions_ = oof_preds.copy()
        self.oof_model_names_ = [
            n for n in self.base_estimators if n in self.trained_models_
        ]

        logger.info("Training meta-learner on %d base-model OOF columns ...", len(valid_cols))
        self.meta_learner.fit(oof_preds, y_arr)
```

### 1c. Verify

```powershell
python -c "import ast; ast.parse(open('src/models/variant_ensemble.py').read()); print('OK')"

Select-String -Path src\models\variant_ensemble.py `
    -Pattern "self\.oof_predictions_" |
    Select-Object LineNumber, Line
```

### 1d. Commit

```powershell
git add src\models\variant_ensemble.py
git commit -m "feat(ensemble): expose oof_predictions_ attribute

Stores the cross_val_predict OOF probability matrix (and the list of
contributing base-model names) as instance attributes before the meta-
learner fit. Required by scripts/run9_ablations.py to produce the
Rule-5 oof_predictions.parquet artefact.

No behaviour change for existing callers: the attributes are only
read by the Run 9 artefact writer. Copies the matrix so downstream
code cannot mutate the stacker's training data accidentally."
```

---

## Patch 2 — `src/evaluation/__init__.py`: export RunArtifactWriter

**Why:** keeps imports clean. `from src.evaluation.prediction_artifacts
import RunArtifactWriter` works regardless, but re-exporting from the
package matches the existing pattern (ClinicalEvaluator is re-exported
there).

### 2a. Inspect current `__init__`

```powershell
Get-Content src\evaluation\__init__.py
```

### 2b. Append RunArtifactWriter export

Edit `src/evaluation/__init__.py` so it ends with both imports:

```python
from src.evaluation.evaluator import ClinicalEvaluator
from src.evaluation.prediction_artifacts import RunArtifactWriter
```

### 2c. Verify

```powershell
python -c "from src.evaluation import ClinicalEvaluator, RunArtifactWriter; print('OK')"
```

### 2d. Commit

Fold into the same commit as the `prediction_artifacts.py` landing
(patch 3 below).

---

## Patch 3 — fill the 0-byte placeholder files

These four files currently exist at 0 bytes. Overwrite with the v3
content from `/mnt/user-data/outputs/run9_plan_v3/`.

| Source (absolute path in outputs) | Destination (absolute path on your disk) |
|---|---|
| `/mnt/user-data/outputs/run9_plan_v3/splits.py` | `C:\Projects\genomic-variant-classifier\src\data\splits.py` |
| `/mnt/user-data/outputs/run9_plan_v3/prediction_artifacts.py` | `C:\Projects\genomic-variant-classifier\src\evaluation\prediction_artifacts.py` |
| `/mnt/user-data/outputs/run9_plan_v3/test_splits.py` | `C:\Projects\genomic-variant-classifier\tests\unit\test_splits.py` |
| `/mnt/user-data/outputs/run9_plan_v3/test_prediction_artifacts.py` | `C:\Projects\genomic-variant-classifier\tests\unit\test_prediction_artifacts.py` |

### 3a. Verify

```powershell
$files = @(
    "src\data\splits.py",
    "src\evaluation\prediction_artifacts.py",
    "tests\unit\test_splits.py",
    "tests\unit\test_prediction_artifacts.py"
)
foreach ($f in $files) {
    if ((Get-Item $f).Length -lt 500) {
        Write-Host "TOO SMALL $f" -ForegroundColor Red
    } else {
        python -c "import ast; ast.parse(open('$($f.Replace('\', '/'))').read())"
        if ($LASTEXITCODE -eq 0) {
            Write-Host "OK $f ($((Get-Item $f).Length) bytes)" -ForegroundColor Green
        } else {
            Write-Host "SYNTAX ERR $f" -ForegroundColor Red
        }
    }
}
```

### 3b. Run unit tests

```powershell
python -m pytest tests\unit\test_splits.py tests\unit\test_prediction_artifacts.py -v
```

### 3c. Commit

```powershell
git add src\data\splits.py src\evaluation\prediction_artifacts.py `
        src\evaluation\__init__.py `
        tests\unit\test_splits.py tests\unit\test_prediction_artifacts.py
git commit -m "feat(run9): splits + RunArtifactWriter + unit tests

Adds gene-stratified and hash-stable unseen-gene splitters (src/data/
splits.py), the RunArtifactWriter that emits the Rule-5 artefact set
(src/evaluation/prediction_artifacts.py), and unit tests for both.
Re-exports RunArtifactWriter from src.evaluation for symmetry with
ClinicalEvaluator.

Hash stability of unseen_gene_holdout_split is the key invariant:
longitudinal comparisons across runs depend on the same genes always
being in the holdout set for the same (seed, holdout_frac). Tested
explicitly in test_splits.py::test_hash_stability_across_data_versions."
```

---

## Patch 4 — land the ablation harness

### 4a. Save file

Copy `/mnt/user-data/outputs/run9_plan_v3/run9_ablations.py` to
`C:\Projects\genomic-variant-classifier\scripts\run9_ablations.py`.

**Do NOT overwrite `scripts/run_phase2_eval.py`** — the ablation harness
is a separate, additive script.

### 4b. Verify

```powershell
Test-Path scripts\run_phase2_eval.py   # MUST still be True
Test-Path scripts\run9_ablations.py    # Should now be True
python -c "import ast; ast.parse(open('scripts/run9_ablations.py').read()); print('OK')"

# CLI responsiveness
python scripts\run9_ablations.py --help
```

### 4c. Smoke test

The smoke test requires fresh splits from the mainline. **Do not run
the smoke test against `outputs/phase2_full/splits/` — that parquet
is stale (46 columns, pre-schema-expansion).** Options:

- **Fast regeneration**: run mainline on a small ClinVar sample first
  to produce a fresh splits directory, then smoke-test against that.
- **Wait for full regeneration**: run mainline on full ClinVar (Patch 6
  should be applied first; see below), then smoke-test against the
  freshly produced splits.

```powershell
# After fresh splits exist at outputs\<run>\splits\ :
python scripts\run9_ablations.py `
    --splits-dir outputs\<run>\splits `
    --ablation full `
    --run-id smoke `
    --output-dir outputs\smoke\run9_ablations `
    --skip-nn --skip-svm --skip-shap --skip-permutation `
    --n-folds 2
```

### 4d. Commit

```powershell
git add scripts\run9_ablations.py
git commit -m "feat(run9): LOCO ablation harness (scripts/run9_ablations.py)

Reads precomputed splits from a prior run_phase2_eval.py output,
applies an ABLATION_MASKS mask by zeroing scaled columns, retrains
the ensemble, and writes the full Rule-5 artefact set to its own
output directory. Coexists with run_phase2_eval.py; does not
replace it.

ABLATION_MASKS calibrated against the 78-column schema confirmed by
direct _engineer_features probe on 2026-04-19. 14 ablation variants
covering allele frequency, splice (SpliceAI + mechanics), conservation,
AlphaMissense, ESM-2, EVE, GNN, AlphaFold, gnomAD constraint, GTEx,
disease databases (HGMD/OMIM/LOVD/ClinGen — flagged for label-leakage
check), and individual in-silico predictors.

Prefix-match logic warns on zero-match rather than failing, so schema
drift produces a log line. Also warns when column count < 70 (likely
stale pre-4/19 parquet)."
```

---

## Patch 5 — (OBSOLETE in v2)

The v1 version of this doc proposed adding `y_train/y_val/y_test.parquet`
persistence to `run_phase2_eval.py`. **Not needed** — verified
2026-04-19 that `DataPrepPipeline._save_splits` at
`src/data/real_data_prep.py:1207-1209` already writes all three
`y_*.parquet` files. Skip.

---

## Patch 6 — `gnn_score` persistence order (NEW in v2)

**Why:** verified 2026-04-19 via code inspection. Current flow:

1. `DataPrepPipeline.run()` calls `_engineer_features`, which at line 988
   assigns `feats["gnn_score"] = (...)` — likely a default constant
   (0.0 for every row, since no trained GNN exists at data-prep time).
2. `DataPrepPipeline._save_splits` writes the split parquets to disk,
   **including the default `gnn_score` column**.
3. `DataPrepPipeline.run()` returns X_train/X_val/X_test to the caller.
4. `scripts/run_phase2_eval.py` at line 192 starts GNN training (if
   `--string-db` is passed).
5. `scripts/run_phase2_eval.py` at line 271 overwrites `X_split["gnn_score"]`
   **in memory** with real GNN predictions — but the parquets on disk
   still contain the defaults.

Consequence for the ablation harness: when `run9_ablations.py` reads
splits from disk, `gnn_score` is a constant 0.0. The `no_gnn` ablation
then zeros an already-zero column — effectively a no-op. The `full`
baseline also trains against a constant `gnn_score`, so the GNN signal
never reaches the stacker in any ablation configuration.

**Two resolution options.** Pick one based on risk tolerance and how
much mainline change Codex is comfortable with.

### Option 6a — minimal: re-persist splits after GNN injection

In `scripts/run_phase2_eval.py`, after the GNN injection block
(currently ending around line 280), add a block that re-writes
the split parquets:

```powershell
Select-String -Path scripts\run_phase2_eval.py `
    -Pattern "GNN scores injected into" -Context 0,5
```

Find the loop at roughly lines 264-281:

```python
# Overwrite gnn_score in feature matrices with real GNN predictions
for split_name, split_df, X_split in [
    ("train", gnn_df, X_train),
    ("val", meta_val, X_val),
    ("test", meta, X_test),
]:
    if "gene_symbol" in split_df.columns:
        X_split["gnn_score"] = (
            split_df["gene_symbol"]
            .fillna("")
            .map(gnn_scorer.score)
            .values
        )
        logger.info(
            "GNN scores injected into %s split (mean=%.3f).",
            split_name,
            float(X_split["gnn_score"].mean()),
        )
```

Append these eight lines AFTER that for-loop:

```python
# Patch 6a — re-persist split parquets with real GNN scores so the
# ablation harness (scripts/run9_ablations.py) reads the correct
# gnn_score when loading splits from disk. Without this, the on-disk
# parquets retain the default 0.0 from DataPrepPipeline._engineer_features.
_splits_dir = outdir / "splits"
if _splits_dir.exists():
    X_train.to_parquet(_splits_dir / "X_train.parquet", index=False)
    X_val.to_parquet(_splits_dir / "X_val.parquet", index=False)
    X_test.to_parquet(_splits_dir / "X_test.parquet", index=False)
    logger.info("GNN-updated splits re-persisted to %s/", _splits_dir)
```

### Option 6b — proper: train GNN inside DataPrepPipeline

Move the GNN training flow from `scripts/run_phase2_eval.py` into
`src/data/real_data_prep.py` as a new stage between feature engineering
and scaling. This is the correct long-term architecture (GNN scores
become a first-class feature, not a post-hoc injection), but requires
threading STRING DB paths through `AnnotationConfig` and restructuring
the GNN training call site. **Estimated effort: 4-6 hours. Defer to
Run 10 prep.**

### Recommendation for Run 9

Apply Option 6a. It's 8 lines of additive code in `run_phase2_eval.py`
with a clear behavioural contract: "when GNN training completes, the
on-disk splits reflect the GNN's predictions." Option 6b is better but
is an architecture decision worth its own design review, not something
to fold into Run 9's prep.

### 6a verification

```powershell
# After patching, re-run the mainline on a small sample:
python scripts\run_phase2_eval.py `
    --clinvar <your clinvar parquet> `
    --string-db auto `
    --output outputs\run9_gnn_test `
    --skip-nn --skip-svm --n-folds 2

# Then confirm gnn_score has non-zero variance on disk:
python -c "
import pandas as pd
df = pd.read_parquet('outputs/run9_gnn_test/splits/X_train.parquet')
print('gnn_score min=', df['gnn_score'].min())
print('gnn_score max=', df['gnn_score'].max())
print('gnn_score std=', df['gnn_score'].std())
print('nonzero fraction=', (df['gnn_score'] != 0).mean())
"
```

Expected: min < max, std > 0, nonzero fraction > 0.5. If the column is
still all zeros, the GNN didn't actually train; investigate
`--string-db auto` resolution or check that the local STRING DB cache
exists at `data/raw/cache/string_graph_700.pkl`.

### 6a commit

```powershell
git add scripts\run_phase2_eval.py
git commit -m "fix(mainline): re-persist splits after GNN injection (Patch 6a)

Verified 2026-04-19 that DataPrepPipeline._save_splits writes
X_train/X_val/X_test.parquet BEFORE GNN training occurs in
run_phase2_eval.py. The in-memory X_split[gnn_score] overwrite at
~line 271 never reached disk, so the ablation harness (and any future
reload of the splits) sees the default constant gnn_score from
_engineer_features.

This 8-line patch re-writes the split parquets after the GNN
injection loop, ensuring on-disk splits reflect the trained GNN's
predictions. Proper long-term fix (move GNN training inside
DataPrepPipeline) deferred to Run 10 prep per
docs/PATCHES_AND_INTEGRATION.md Patch 6b."
```

---

## Final verification

```powershell
git status
git log --oneline -6
python -m pytest -q --tb=short 2>&1 | Select-Object -Last 15
git push origin main
```

Expected commit sequence after all patches land:

1. `fix(preflight): Windows subprocess via shutil.which, no shell` (if Patch 0 is applied — unchanged from v1)
2. `feat(ensemble): expose oof_predictions_ attribute` (Patch 1)
3. `feat(run9): splits + RunArtifactWriter + unit tests` (Patch 3)
4. `feat(run9): LOCO ablation harness (scripts/run9_ablations.py)` (Patch 4)
5. `fix(mainline): re-persist splits after GNN injection (Patch 6a)` (Patch 6a)

Then Run 9 is ready for the sequence in `docs/RUN9_OPERATIONS_PLAYBOOK.md`
Step 7 onward — but note that the playbook needs to be updated to reference
`run9_ablations.py` rather than `run_phase2_eval.py` where it talks about
the ablation runs. That playbook update is a separate small commit.

---

## Summary of locations

| Source in outputs | Target on your disk |
|---|---|
| `/mnt/user-data/outputs/run9_plan_v3/run9_ablations.py` | `C:\Projects\genomic-variant-classifier\scripts\run9_ablations.py` |
| `/mnt/user-data/outputs/run9_plan_v3/splits.py` | `C:\Projects\genomic-variant-classifier\src\data\splits.py` |
| `/mnt/user-data/outputs/run9_plan_v3/prediction_artifacts.py` | `C:\Projects\genomic-variant-classifier\src\evaluation\prediction_artifacts.py` |
| `/mnt/user-data/outputs/run9_plan_v3/test_splits.py` | `C:\Projects\genomic-variant-classifier\tests\unit\test_splits.py` |
| `/mnt/user-data/outputs/run9_plan_v3/test_prediction_artifacts.py` | `C:\Projects\genomic-variant-classifier\tests\unit\test_prediction_artifacts.py` |
| `/mnt/user-data/outputs/run9_plan_v3/preflight_run_helper_patch.py` | reference document — use to hand-patch `scripts/preflight_check.py`; do NOT save as a standalone script |
| `/mnt/user-data/outputs/run9_plan_v3/PATCHES_AND_INTEGRATION.md` | `C:\Projects\genomic-variant-classifier\docs\PATCHES_AND_INTEGRATION.md` |