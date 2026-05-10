# HANDOFF — resume Run 9 launch

**Last session**: 2026-04-17 afternoon (this one)
**Next session**: fresh Claude session, whenever

## Where we left off

Session 2026-04-17 afternoon closed with Run 9 infrastructure fully
built and committed, plus three documented unknowns. Run 9 itself has
NOT launched. GPU has NOT been spent.

### What shipped (committed, on origin/main)

- `scripts/preflight_check.py` — local preflight (3rd revision,
  uses `shutil.which` to resolve Windows .cmd shims)
- `scripts/preflight_vm.sh` — on-VM preflight (bash)
- `tests/unit/test_esm2_activation.py` — three-test smoke module
  for ESM-2 stub-mode detection
- `scripts/run9_launch.md` — runbook (expects ESM-2 stub in Run 9)
- `docs/incidents/INCIDENT_2026-04-17_esm2-hgvsp-parser.md` — root
  cause of ESM-2 silent-zero across Runs 6/7/8
- `docs/sessions/SESSION_2026-04-17_run9-prep.md` — full session log
- `docs/CHANGELOG.md` — appended (v2 section)

### What's unresolved / on deck for the next session

1. **Validate the final preflight runs clean**. The committed version
   uses `shutil.which` to resolve executables BEFORE invoking
   subprocess (avoiding the Windows .cmd shim path-resolution bug
   that plagued the first two revisions). It has not been tested
   end-to-end post-commit. First command in the next session:
   ```powershell
   python scripts\preflight_check.py --skip-pytest
   ```
   Expected: all git checks PASS (no more "system cannot find the
   path specified"), gcloud GCS checks either PASS (auth fresh) or
   FAIL with an actionable gcloud-auth message.

2. **SpliceAI cache mystery**. The 430 MB cache at
   `data/raw/cache/spliceai_scores_snv.parquet` reappeared between
   the morning and afternoon sessions. It was deleted this session.
   Source of the regeneration is unknown. Hypothesis: a test outside
   the `TestAnnotationPipeline` class (which has the `_isolate_spliceai`
   autouse fixture) hit the connector during the 10-min full pytest
   run. To diagnose next session:
   ```powershell
   # Run pytest WITHOUT deleting the cache, then check if it reappears:
   Remove-Item data\raw\cache\spliceai_scores_snv.parquet -ErrorAction SilentlyContinue
   python -m pytest tests/unit/ -q
   Test-Path data\raw\cache\spliceai_scores_snv.parquet
   ```
   If True, bisect which test rebuilds it. Likely candidates: any
   test importing real_data_prep.py outside the fixture's scope.

3. **Run 9 launch itself**. Once preflight runs clean:
   * Provision Vast.ai instance (RTX 4090, 60 GB RAM, 200 GB disk,
     CUDA 12.x)
   * Clone repo with GITHUB_TOKEN; pull data from GCS
   * `pip install "transformers>=4.40,<5.0"` (pin matters — see
     runbook)
   * `bash scripts/preflight_vm.sh`
   * Launch training with the command in `scripts/run9_launch.md`
   * Expect ESM-2 stub mode WARNING in logs (this is documented,
     not a failure)
   * Upload outputs to GCS, verify with `gcloud storage ls`,
     destroy instance
   * Write SESSION_YYYY-MM-DD_run9.md with per-model OOF AUROC,
     GNN edge count, esm2_delta_norm zero-fraction, SpliceAI
     importance rank, total GPU cost

## Key context for the next session

### What we learned this session (important)

- **ESM-2 has been inert across Runs 6, 7, 8**. Root cause is a
  missing HGVSp parser upstream — see INCIDENT doc. Run 9 will
  ALSO have ESM-2 inert; this is known and documented. Run 10
  priority: build `src/genomic_variant_classifier/data/hgvsp_parser.py` (1-2 day task).
- **EVE is probably also inert** for the same reason. Not verified
  yet; can bundle fix with the HGVSp parser.
- **The real value of Run 9** is testing SpliceAI and GNN
  contributions — both newly active for the first time. Runs 6/7/8
  produced AUROC ~0.9862; Run 9 tests whether adding working
  SpliceAI + GNN + 5 more base models moves the needle. Even a
  flat result is informative.

### Standing rules that were reinforced this session

- GPU is mandatory (no CPU fallback, ever)
- Shutdown ordering: upload to GCS BEFORE destroying instance
- 2026-04-17 rule: verify all GCS-state claims with `gcloud storage ls`
- Pre-flight cache check: the 430 MB SpliceAI cache MUST be absent
  before any training run — its presence indicates test-isolation leak

### Windows-specific gotchas learned

- subprocess.run() cannot resolve .cmd shims via PATH on Windows.
  Use `shutil.which()` FIRST to get an absolute path, then pass that
  to subprocess. Do not use `shell=True` with `cwd=` — it's fragile.
- `[System.IO.File]::AppendAllText` does not prepend a newline. If
  the target file doesn't end with `\n`, the append will glue onto
  the final line. Either check-and-add-newline first, or include
  `\n\n` in the appended content.

## Quick-start commands for the next session

```powershell
cd C:\Projects\genomic-variant-classifier
C:\Projects\genomic-variant-classifier\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "C:\Projects\genomic-variant-classifier"

# State check
git status
git log --oneline -8
git rev-parse HEAD
git rev-parse origin/main   # should match HEAD

# Verify Run 9 infrastructure is still in place
Test-Path scripts\preflight_check.py
Test-Path scripts\preflight_vm.sh
Test-Path scripts\run9_launch.md
Test-Path tests\unit\test_esm2_activation.py
Test-Path docs\incidents\INCIDENT_2026-04-17_esm2-hgvsp-parser.md
Test-Path docs\sessions\SESSION_2026-04-17_run9-prep.md

# SpliceAI parquet still there?
Get-Item data\external\spliceai\spliceai_index.parquet | Select-Object Length, LastWriteTime

# Did the cache come back?
Test-Path data\raw\cache\spliceai_scores_snv.parquet

# Run the preflight --skip-pytest (fast), see what's left to fix
Get-ChildItem -Recurse -Filter "__pycache__" -Directory | Remove-Item -Recurse -Force
python scripts\preflight_check.py --skip-pytest
echo "exit: $LASTEXITCODE"
```

Paste that output into the new Claude session and say "resuming Run 9
launch after 2026-04-17 afternoon session". The session doc and
INCIDENT give Claude everything needed to pick up.