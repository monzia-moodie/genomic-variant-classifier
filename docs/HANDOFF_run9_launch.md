# HANDOFF — Run 9 Launch (current state, 2026-05-10)

**Last session**: 2026-05-10 evening (session wrap-up commit `899cae5`)
**Next session**: Run 9 provisioning + launch

## Current state (verified by 2026-05-10 21:22 pre-flight)

- HEAD on `origin/main`: `899cae5` — session wrap-up of the SpliceAI cache leak fix arc.
- CI green on `899cae5`, `a01eef3`, and `34eaf98` (last three commits, all completed/success).
- Working tree clean. 0 ahead / 0 behind `origin/main`.
- Active venv: `.venv312` (Python 3.12.10). `.venv` still exists on disk but is not used for Run 9.
- SpliceAI prod cache (`data/raw/cache/spliceai_scores_snv.parquet`): 451,626,904 bytes, mtime unchanged from 2026-04-19 13:56:19 — `a01eef3` path-aware conftest is holding.
- Splits ready: `outputs/run9_ready/splits/` has X_train/val/test, y_train/val/test, meta_train/val/test. All three X-parquets verified at 78 cols with `n_pathogenic_in_gene` present. Cohort 1,700,687 labeled (train 1,197,216 / val 154,404 / test 349,067).

Run 9 itself has NOT launched. GPU has NOT been spent.

## What shipped since the previous HANDOFF (2026-04-17 → 2026-05-10)

Three substantive arcs, all on `origin/main`:

### Package consolidation migration (C1 → C5.3b)

`src.*` and `agent_layer.*` namespaces consolidated into `genomic_variant_classifier.*`. Key commits:

- `431850c` C2 — replace `setup.py` with `pyproject.toml`; install in editable mode
- `66fdbfe` C3 — rewrite `src.*` and `agent_layer.*` imports
- `1ab216f` C3.1 through `e0f4c6e` C3.6 — sweep cleanup, Path-component refs, Dockerfile paths, CI `pip install -e .`, bare-import rewrite, `agent_layer/__init__.py`
- `e34ce7b` C4-prep — bind `agent_layer` in `install_compat_aliases` + add `migration_smoke` fixture
- `d7ed38e` C5.1 through `6443af7` C5.3b — README, CI, gitignore, docs sweep
- `add6ac4` — C5 session log + CHANGELOG

**Status:** C1–C3, C4-prep, and C5.1–C5.3b are landed and green. **C4 (the real ~9 GB joblib re-pickle) is NOT done** — deferred to a dedicated session (3.5–5.5 hr wall, local only, NEVER on Vast.ai). This does not block Run 9: Run 9 trains from scratch and writes a new `ensemble.joblib`; legacy pickles at `outputs/run9_ready/models/ensemble.joblib` (1478 MB) and `models/v1/ensemble_v1.joblib` are not loaded by the training path.

### Architectural cleanup (INCIDENT_2026-04-29: GCS retired)

Four-commit arc:

- `b15a625` (1/4) — INCIDENT doc formalising 2026-04-29 GCP project deletion + SCP-only pivot
- `aad8f5a` (2/4) — strip GCS from active runtime code
- `feece15` (3/4) — rewrite operational docs for SCP-only architecture
- `34eaf98` (4/4) — session log + CHANGELOG cap

**Data flow now:** Windows local source-of-truth ↔ Vast.ai GPU scratch via SCP (~1.0–1.2 GB per leg, SSH key at `C:\Users\monzi\.ssh\id_lambda_run8`) ↔ Google Drive via `rclone genvarcla:` for agent-layer durability only (event log SQLite WAL hourly sync, agent cache, drift reports). No remote object storage.

### SpliceAI fixture leak fix (the final pre-Run-9 blocker)

- `a01eef3` — move `_isolate_spliceai` fixture to `tests/unit/conftest.py` with path-aware `_load_cache`/`_save_cache` patches; blocks 430 MB cache regeneration at the prod-cache path while leaving `tmp_path` FetchConfigs unaffected. 16 pytest tests pass; cache mtime preserved.
- `899cae5` — session log + CHANGELOG entry

## What's unresolved before launch

Exactly one item, with one design question:

**Run 9 launch itself.** Sequence:

1. Provision Vast.ai instance: RTX 4090, 60 GB RAM, 200 GB disk, CUDA 12.x (PyTorch image).
2. SCP `outputs/run9_ready/splits/` + repo to the instance (~1.0–1.2 GB).
3. On-VM: `pip install -r requirements.txt` then `pip install "transformers>=4.40,<5.0"`. ESM-2 will run in STUB MODE per INCIDENT_2026-04-17 — this is documented and expected, not a failure.
4. Set up `trap '... ; sudo shutdown -h now' EXIT INT TERM` BEFORE launching training.
5. Run `bash scripts/preflight_vm.sh` (must exit 0).
6. Launch training (see design question below).
7. SCP outputs back to local Windows box BEFORE destroying the instance.
8. Destroy the instance from the Vast.ai web console (Destroy, not Stop — Stop keeps storage billing alive).
9. Confirm $0/hr at https://cloud.vast.ai/billing/.
10. Write `docs/sessions/SESSION_2026-MM-DD_run9.md` per `docs/RUN9_SCIENTIFIC_DESIGN.md` §6 (11 mandatory sections).

### Design question — training entrypoint (resolve at launch time)

Two mature scripts exist:

| Script | Behaviour | Splits |
|---|---|---|
| `scripts/run_phase2_eval.py` (22 KB) | Full pipeline; calls `prep.run(...)` which **regenerates splits** from raw ClinVar + gnomAD + … | Ignores `outputs/run9_ready/`; redoes 13h CPU regen on Vast.ai GPU |
| `scripts/run9_ablations.py` (31 KB) | LOCO ablation harness; reads `splits_dir/X_train.parquet` directly | Reuses pre-built `outputs/run9_ready/splits/` (the 2026-04-30 splits, AUROC 0.9814 baseline) |

Architecturally, `run9_ablations.py` is the right choice: it skips the costly regen, uses the splits that were already produced and verified, and produces the eight scientific artefacts mandated by `docs/RUN9_SCIENTIFIC_DESIGN.md` Rule 5 (manifest, OOF predictions, test predictions, SHAP, calibration, feature importance, ablation results, graph stats). Confirm before SSH'ing to the Vast.ai instance.

## Standing rules in effect (reconfirmed by the 2026-05-10 arc)

- **GPU mandatory.** No CPU fallback for training runs. RTX 4090 minimum.
- **SCP outputs back to local BEFORE destroying instance.** No remote storage. GCP project is gone per INCIDENT_2026-04-29.
- **Local-disk receipt required for any "RESOLVED" claim.** `Get-ChildItem -LiteralPath outputs\run9 -Recurse | Measure-Object Length -Sum` after SCP-back, recorded in the session doc.
- **Pre-flight cache invariant (post-`a01eef3`):** the SpliceAI cache at `data/raw/cache/spliceai_scores_snv.parquet` is **allowed to exist**; its mtime **must not advance during pytest**. Compare `(Get-Item ...).LastWriteTime` before and after the test run. mtime drift = regression in the path-aware conftest.
- **`trap ... EXIT INT TERM` for shutdown, not `&&` chaining.** `&&` only fires on success; `trap` fires on any exit (success, failure, crash, OOM, ^C). Billing must stop on any process exit.
- **Local file pre-flight gate:** every Run-N launch requires `scripts/preflight_check.py` to exit zero, OR every FAIL line explicitly justified in the launch session doc.

## Quick-start commands for the launch session

```powershell
cd C:\Projects\genomic-variant-classifier
C:\Projects\genomic-variant-classifier\.venv312\Scripts\Activate.ps1
$env:PYTHONPATH = "C:\Projects\genomic-variant-classifier"

# State check
git fetch origin
git status
git log --oneline -8
git rev-parse HEAD               # expected: 899cae5...
git rev-parse origin/main        # must equal HEAD

# Verify Run 9 infrastructure present
foreach ($f in @(
    'scripts\preflight_check.py',
    'scripts\preflight_vm.sh',
    'scripts\run9_launch.md',
    'scripts\run9_ablations.py',
    'scripts\run_phase2_eval.py',
    'tests\unit\conftest.py',
    'tests\unit\test_esm2_activation.py',
    'docs\incidents\INCIDENT_2026-04-17_esm2-hgvsp-parser.md',
    'docs\incidents\INCIDENT_2026-04-29_gcp-billing-deletion.md',
    'docs\sessions\SESSION_2026-05-10_spliceai-cache-fix.md',
    'docs\RUN9_OPERATIONS_PLAYBOOK.md',
    'docs\RUN9_SCIENTIFIC_DESIGN.md'
)) {
    $mark = if (Test-Path -LiteralPath $f) {'[OK]'} else {'[FAIL]'}
    '{0,-6} {1}' -f $mark, $f
}

# Splits state (must be untouched since 2026-04-30)
Get-Item outputs\run9_ready\splits\X_train.parquet | Select-Object Length, LastWriteTime

# SpliceAI prod cache mtime invariant (must equal 04/19/2026 13:56:19)
Get-Item data\raw\cache\spliceai_scores_snv.parquet | Select-Object Length, LastWriteTime

# Clean pyc, then run preflight
Get-ChildItem -Recurse -Filter "__pycache__" -Directory | Remove-Item -Recurse -Force
python scripts\preflight_check.py --skip-pytest
echo "preflight exit: $LASTEXITCODE"
```

Paste the output into the launch session, confirm the training-entrypoint decision, then proceed with the runbook at `scripts/run9_launch.md`.

## Cross-references

- **Runbook**: `scripts/run9_launch.md` — operational launch sequence (Vast.ai provisioning, on-VM setup, success grep checklist, shutdown ordering, post-run docs)
- **Operations playbook**: `docs/RUN9_OPERATIONS_PLAYBOOK.md` — step-by-step engineering tasks (T1–T7); much of T1–T5 is now done; revisit T6 (`predict_vus.py`) and T7 (`generate_run_report.py`) post-launch
- **Scientific design charter**: `docs/RUN9_SCIENTIFIC_DESIGN.md` — Rule 5 (artefact set), Rule 6 (six ablations), Rule 7 (preserve predictions), Rule 8 (publishable narrative); §6 has the mandatory session-doc structure
- **SpliceAI cache fix session log**: `docs/sessions/SESSION_2026-05-10_spliceai-cache-fix.md`
- **Architecture pivot incident**: `docs/incidents/INCIDENT_2026-04-29_gcp-billing-deletion.md`
- **ESM-2 silent-zero incident** (Run 10 priority): `docs/incidents/INCIDENT_2026-04-17_esm2-hgvsp-parser.md`
