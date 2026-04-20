# Run 9 Engineering Playbook

**Audience:** executor of the Run 9 preparation work.
**Prerequisite:** read `docs/RUN9_SCIENTIFIC_DESIGN.md` first. This playbook
is the concrete step-by-step; the design doc is the rationale.

Every step has an explicit verification command. Do not proceed to the next
step until the verification for the current step returns the expected state.
All outputs should be pasted for review, especially on `[FAIL]` outcomes.

---

## Step 0 — Session boot state check

```powershell
cd C:\Projects\genomic-variant-classifier
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "C:\Projects\genomic-variant-classifier"

# Baseline state
git status
git rev-parse HEAD
git rev-parse origin/main

# These should match what the handoff doc said
Get-Content docs\HANDOFF_run9_launch.md | Select-Object -First 80
```

**Expected**:
- HEAD = `50c0579`, origin/main = `50c0579`
- Working tree: only `scripts/gcp_run6_startup.sh` modified and `ROADMAP_PSYCH_GWAS_ENTRY.md` untracked (allowlisted carry-overs)

If HEAD drifted: reconcile before continuing. Do not apply patches against an unknown state.

---

## Step 1 — Apply the preflight Windows subprocess fix (T1)

Download `preflight_run_helper_patch.py` from this plan's outputs.
The patch body (the `run` function) replaces the existing `run` helper
in `scripts/preflight_check.py`.

```powershell
# 1a. Inspect the current run() definition so str_replace has an exact target
Get-Content scripts\preflight_check.py | Select-String -Pattern "^def run\(" -Context 0,30
```

You will see the current `run` helper signature. It was introduced during
the 2026-04-17 session. The patch replaces its body with the version that
uses `shutil.which` + no shell.

```powershell
# 1b. Apply patch. Open scripts\preflight_check.py in an editor and replace the
#     existing `def run(...)` and its body (up to the next top-level def) with
#     the body from preflight_run_helper_patch.py. Keep all other functions
#     untouched.

# 1c. Validate syntax
python -c "import ast; ast.parse(open('scripts/preflight_check.py').read()); print('OK')"

# 1d. Dry-run the preflight (expect green git/gcloud checks now)
Get-ChildItem -Recurse -Filter "__pycache__" -Directory | Remove-Item -Recurse -Force
python scripts\preflight_check.py --skip-pytest
echo "exit: $LASTEXITCODE"
```

**Expected**: the "cannot find the path specified" errors for `git status`,
`git fetch`, `gcloud storage ls` are gone. Remaining FAILs are allowed to
be about GCS auth (`gcloud auth login` fixes them) or about session work
being uncommitted (expected mid-session).

---

## Step 2 — Tighten `.gitignore` + fix SpliceAI cache leak (T2)

```powershell
# 2a. Confirm current gitignore rule
git check-ignore -v data/raw/cache/spliceai_scores_snv.parquet
# Expected output: ".gitignore:68:data/raw/..."  (rule exists, via parent match)
```

That's working but implicit. Add an explicit line so intent is clear:

```powershell
# 2b. Add explicit cache-path rule
Add-Content -Path .gitignore -Value "`n# Explicit: SpliceAI/STRING cache artefacts are ephemeral`n/data/raw/cache/"

# 2c. Verify still-covered
git check-ignore -v data/raw/cache/spliceai_scores_snv.parquet
# Should now match the new explicit rule.
```

Now the test-isolation fix. Audit tests that import the SpliceAI annotator:

```powershell
# 2d. List every test that touches the SpliceAI annotator
Get-ChildItem -Path tests -Recurse -Filter "*.py" |
    Select-String -Pattern "SpliceAI|splice_ai_|_isolate_spliceai" |
    Select-Object Path, LineNumber, Line |
    Format-Table -AutoSize -Wrap

# 2e. Delete the cache, run pytest, see if it regenerates
Remove-Item data\raw\cache\spliceai_scores_snv.parquet -ErrorAction SilentlyContinue
python -m pytest -q 2>&1 | Select-Object -Last 30
Test-Path data\raw\cache\spliceai_scores_snv.parquet
```

**Diagnostic**: if the cache file regenerates during `pytest`, at least one
test outside `TestAnnotationPipeline` is touching the connector. Find it
by grepping the test file paths from 2d for ones that are NOT
`TestAnnotationPipeline`.

**Fix**: widen the `_isolate_spliceai` fixture from `scope="class"` to
`scope="module"` and make it `autouse=True` at the module level for every
test file that appears in 2d. Alternative: introduce a session-scoped
fixture that monkey-patches the connector's cache path to a tmpdir.

Paste the pytest tail + final `Test-Path` output; together they show whether
the leak is fixed.

---

## Step 3 — Add `src/data/splits.py` (T3)

```powershell
# 3a. Download splits.py + test_splits.py to:
#     C:\Projects\genomic-variant-classifier\src\data\splits.py
#     C:\Projects\genomic-variant-classifier\tests\unit\test_splits.py

# 3b. Confirm presence and syntax
Test-Path src\data\splits.py
Test-Path tests\unit\test_splits.py
python -c "import ast; ast.parse(open('src/data/splits.py').read()); print('splits OK')"
python -c "import ast; ast.parse(open('tests/unit/test_splits.py').read()); print('test_splits OK')"

# 3c. Run the tests
python -m pytest tests\unit\test_splits.py -v
# Expected: all passing. The hash-stability test is the key invariant.
```

**Expected**: 7 tests, all green.

---

## Step 4 — Add `src/evaluation/prediction_artifacts.py` (T4)

```powershell
# 4a. Download prediction_artifacts.py + test_prediction_artifacts.py to:
#     C:\Projects\genomic-variant-classifier\src\evaluation\prediction_artifacts.py
#     C:\Projects\genomic-variant-classifier\tests\unit\test_prediction_artifacts.py

# 4b. Syntax + install shap if missing (we need it for SHAP support)
python -c "import ast; ast.parse(open('src/evaluation/prediction_artifacts.py').read()); print('OK')"
python -c "import shap; print('shap', shap.__version__)" 2>&1
# If not installed:  pip install shap

# 4c. Run unit tests
python -m pytest tests\unit\test_prediction_artifacts.py -v
# Expected: all passing (manifest, test_predictions, oof, graph_stats,
# ablation aggregator, artefact tracking).
```

---

## Step 5 — Add the ablation harness (T5)

```powershell
# 5a. Download run_phase2_eval.py to:
#     C:\Projects\genomic-variant-classifier\scripts\run_phase2_eval.py

# 5b. Syntax
python -c "import ast; ast.parse(open('scripts/run_phase2_eval.py').read()); print('OK')"

# 5c. CLI smoke test (--help only; does not touch data)
python scripts\run_phase2_eval.py --help
# Expected: argparse output listing --ablation, --run-id, --holdout, etc.

# 5d. 1 % sample smoke test (writes artefacts, does not upload to GCS)
python scripts\run_phase2_eval.py `
    --ablation full --run-id smoke `
    --sample-frac 0.01 --skip-gpu --skip-shap `
    --output-dir runs\smoke
# Expected: completes in <2 min, writes runs\smoke\smoke\full\manifest.json + test_predictions.parquet

Get-ChildItem runs\smoke -Recurse
```

If the smoke test fails, it's most likely because `data/processed/training_variants.parquet` doesn't exist or has a different schema than the harness expects. Confirm:

```powershell
Test-Path data\processed\training_variants.parquet
python -c "import pandas as pd; df=pd.read_parquet('data/processed/training_variants.parquet'); print(df.shape); print(list(df.columns)[:30])"
```

The harness requires: `gene_symbol`, `acmg_label`, plus whatever
`engineer_features` consumes (`allele_freq`, `ref`, `alt`, `consequence`,
`cadd_phred`, `sift_score`, `polyphen2_score`, `revel_score`, `phylop_score`,
`gene_constraint_oe`, `num_pathogenic_in_gene`, `in_active_site`, `in_domain`).
If columns differ, the harness needs a one-line fix before Run 9.

---

## Step 6 — Commit the infrastructure in logical chunks

Split into three commits so each is reviewable and revertable:

```powershell
# 6a. Commit 1: preflight + gitignore hygiene
git add scripts\preflight_check.py .gitignore
git status
git commit -m "fix(preflight): Windows subprocess via shutil.which, no shell

Resolves 'The system cannot find the path specified' errors on Windows
by resolving .cmd/.exe shims to absolute paths via shutil.which and
invoking without shell=True. Adds explicit /data/raw/cache/ line to
.gitignore; prior coverage via /data/raw/ was functional but implicit."

# 6b. Commit 2: splits + artefact writer
git add src\data\splits.py src\evaluation\prediction_artifacts.py `
    tests\unit\test_splits.py tests\unit\test_prediction_artifacts.py
git status
git commit -m "feat(run9): splits + RunArtifactWriter for multi-ablation runs

Adds gene-stratified and hash-stable unseen-gene holdout splitters, plus
a RunArtifactWriter that emits the Rule-5 artefact set (manifest, OOF
predictions, test predictions, calibration, SHAP, permutation importance,
graph stats). Unit tests cover no-gene-leak, hash stability across
dataset versions, atomic writes, and idempotent ablation-row appends."

# 6c. Commit 3: ablation harness + design doc
git add scripts\run_phase2_eval.py docs\RUN9_SCIENTIFIC_DESIGN.md
git status
git commit -m "feat(run9): ablation harness + scientific design charter

scripts/run_phase2_eval.py drives LOCO ablations with the RunArtifactWriter.
ABLATION_MASKS in this file is the single source of truth for feature-class
groupings; new feature classes must be added here to remain ablatable.

docs/RUN9_SCIENTIFIC_DESIGN.md formalises the 'maximum information per
GPU run' charter, adds four standing rules (5-8), defines eight Run 9
scientific outputs, and specifies the post-run documentation contract."

# 6d. Push and tag
git push origin main
git tag -a run9-infra-ready -m "Run 9 infrastructure landed (T1-T5 complete)"
git push origin run9-infra-ready
git log --oneline -5
```

---

## Step 7 — Pre-launch gate (all-green preflight + tests)

```powershell
# 7a. Clean caches
Get-ChildItem -Recurse -Filter "__pycache__" -Directory | Remove-Item -Recurse -Force

# 7b. Full test suite
python -m pytest -q --tb=short 2>&1 | Select-Object -Last 20
# Expected: all green, including new test_splits.py and test_prediction_artifacts.py

# 7c. Full preflight (includes the pytest step — ~10 min)
python scripts\preflight_check.py
echo "exit: $LASTEXITCODE"
# Expected exit: 0, OR only FAILs are transient (e.g. GCS auth, fixable
# with `gcloud auth login`).

# 7d. Confirm GCS auth (required for upload_to_gcs() at run end)
gcloud auth list
gcloud config get-value project
# Expected: active account + project=genomic-variant-prod
# If not: gcloud auth login; gcloud config set project genomic-variant-prod

# 7e. Ablation harness 10 % integration smoke
python scripts\run_phase2_eval.py `
    --ablation no_spliceai --run-id smoke_10pct `
    --sample-frac 0.10 --skip-shap `
    --output-dir runs\smoke
# Expected: completes in <15 min on CPU. Exercises the full end-to-end
# path including a real ablation and artefact writes.

Get-ChildItem runs\smoke\smoke_10pct\no_spliceai
```

**Do not proceed to Step 8 until every 7a–7e returns green.**

---

## Step 8 — Vast.ai provisioning + Run 9 launch

See `docs/RUN9_SCIENTIFIC_DESIGN.md` §5 Phase B and Phase C for the full
command sequence. Key reminders:

- Tag the commit that launches: `git tag run9-launch-<date>` before leaving
  the local shell.
- Never skip the `trap EXIT` shutdown script. VM must self-destroy even on
  crash.
- The six ablations + one unseen-gene split in §5 Phase C are a single
  logical run — if any fails, debug and re-run *only that ablation*, not
  the whole lot.
- Run `predict_vus.py` after training finishes (zero GPU) to produce the
  ranked VUS parquet.
- Upload all of `runs/run9/` in a single `gcloud storage cp --recursive`
  call at the end, then immediately `sudo shutdown -h now` (or let the
  trap handle it).

---

## Step 9 — Post-run documentation (Rule 8)

After Run 9 completes and artefacts are in GCS:

```powershell
# 9a. Pull artefacts locally
gcloud storage cp --recursive gs://genomic-variant-prod-outputs/runs/run9 runs\

# 9b. Auto-generate the report skeleton (when generate_run_report.py exists;
#     T7 task — may be deferred if manually writing Run 9's first report).
python scripts\generate_run_report.py --run-id run9 `
    --output docs\sessions\SESSION_$(Get-Date -Format yyyy-MM-dd)_run9.md

# 9c. Manually add sections 9-11 (surprises, clinical deliverable, next-run hypothesis).
#     See docs/RUN9_SCIENTIFIC_DESIGN.md §6.

# 9d. Update CHANGELOG and ROADMAP
#     Both must reference the specific artefacts and their location.

# 9e. Commit docs separately from any code changes
git add docs\sessions\SESSION_*.md docs\CHANGELOG.md docs\ROADMAP.md
git commit -m "docs(run9): session report, CHANGELOG, roadmap update"
git push
```

---

## Step 10 — Declare success or document failure

Either:

- **Success**: all eight Run-9 exit criteria in `docs/RUN9_SCIENTIFIC_DESIGN.md`
  §8 are checked. Close the session, set Run 10 milestone to "HGVSp parser".
- **Failure**: write `docs/incidents/INCIDENT_<date>_run9-<slug>.md` with
  root cause, `gcloud storage ls` receipts, and the specific next step that
  would prevent recurrence. Run 10 does not launch until the incident is
  resolved.

A run that reports AUROC but lacks the ablation decomposition, SHAP, and
VUS ranking is a *partial* run, not a successful one. Flag it as such in
the session doc.

---

## Appendix: quick verification chains

If you just want to know "is everything ready?" at any point during
Steps 1–7, run this:

```powershell
python scripts\preflight_check.py --skip-pytest && `
python -m pytest tests\unit\test_splits.py tests\unit\test_prediction_artifacts.py -q && `
python scripts\run_phase2_eval.py --help > $null && `
Write-Host "ready" -ForegroundColor Green
```

If all four segments succeed, the infrastructure side is done and the
next step is Vast.ai provisioning.