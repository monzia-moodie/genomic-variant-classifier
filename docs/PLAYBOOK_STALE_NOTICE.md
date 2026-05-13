# `docs/RUN9_OPERATIONS_PLAYBOOK.md` — STALE NOTICE (2026-05-13)

**Status: SUPERSEDED. Do not follow this playbook as-is.**

The playbook was authored before the C5 layout-migration session
(2026-05-09, commits `d7ed38e`, `4eb1205`, `6a38ee3`, `6443af7`). After
C5, every reference to `src/<subpkg>/` in the playbook is wrong. After
the 2026-05-12 Run 9 outcome, the playbook's launch-step rationale also
diverges from the patched launch script.

## Specific divergences

| Section | Playbook says | Actual current state | Fix |
|---|---|---|---|
| Step 0 | `.\.venv\Scripts\Activate.ps1` | `.\.venv312\Scripts\Activate.ps1` (Python 3.12.10) | Update playbook venv path |
| Step 0 | Expected HEAD = `50c0579` | HEAD = `3cfc039` (after `docs(session): Run 9 ...` push 2026-05-12) | Update or remove the HEAD pin |
| Step 3 | `src\data\splits.py` | `src\genomic_variant_classifier\data\splits.py` if it exists; needs verification | Rewrite all `src\<subpkg>\` paths |
| Step 4 | `src\evaluation\prediction_artifacts.py` | `src\genomic_variant_classifier\evaluation\prediction_artifacts.py` if it exists; needs verification | Same |
| Step 5 | `python scripts\run_phase2_eval.py --ablation full --run-id smoke --sample-frac 0.01 --skip-gpu --skip-shap --output-dir runs\smoke` | Current `run_phase2_eval.py` has none of these flags; ablation harness is `scripts/run9_ablations.py` per CHANGELOG 2026-04-20 entry | Rewrite Step 5 to invoke `run9_ablations.py` |
| Step 5 | `data/processed/training_variants.parquet` | `data/processed/clinvar_grch38.parquet` per `scripts/train.py` | Update reference |
| Step 8 | `runs/run9/` output directory | `outputs/run9/` per current convention | Update reference |
| Step 9b | `python scripts\generate_run_report.py` | Existence not verified; may not exist | Verify before using |

## Verification commands

```powershell
cd C:\Projects\genomic-variant-classifier

# Does the playbook's referenced splits.py exist anywhere?
Get-ChildItem -Recurse -Filter "splits.py" -ErrorAction SilentlyContinue |
    Select-Object FullName

# Same for prediction_artifacts.py
Get-ChildItem -Recurse -Filter "prediction_artifacts.py" -ErrorAction SilentlyContinue |
    Select-Object FullName

# Same for generate_run_report.py
Get-ChildItem -Recurse -Filter "generate_run_report.py" -ErrorAction SilentlyContinue |
    Select-Object FullName

# Does run_phase2_eval.py have --ablation / --run-id flags?
python scripts\run_phase2_eval.py --help 2>&1 | Select-String -Pattern "--ablation|--run-id|--sample-frac|--skip-gpu"

# Does run9_ablations.py exist?
Test-Path scripts\run9_ablations.py
```

## What to use instead

Until the playbook is rewritten post-C5 + post-Phase-1, follow this canonical sequence for Run 10:

1. **Apply Phase 1.5 (this bundle).**
2. **Run `scripts/run9_outputs_audit.ps1`** to recover Run 9 per-model
   AUROCs if possible.
3. **Apply Phase 1 patch bundle** (`run10_phase1_v2.zip`) per its `README.md`.
4. **Verify locally** with `pytest tests/unit/test_variant_ensemble_save_load.py -v`.
5. **(Optional) Write `scripts/launch_run10_vm.sh`** as a clean replacement
   for `scripts/launch_run9_vm.sh` (see
   `INCIDENT_2026-05-12_vastai-destroy-interactive.md`,
   `INCIDENT_2026-05-12_launch-path-inconsistency.md`, and chat-side
   evaluation of `launch_run9_vm.sh` for the C1/C2/C3/H1–H4 patch list).
6. **Re-regen splits on Vast.ai** with LOVD/DbNSFP/FinnGen wired
   (Phase 1 patch B1 now passes these to `AnnotationConfig`).
7. **Launch Run 10** with the new launch script + corrected splits.

## Recommended action

Either:
- **(a)** Rewrite `docs/RUN9_OPERATIONS_PLAYBOOK.md` in a post-Run-10 session
  as `docs/RUN10_OPERATIONS_PLAYBOOK.md` with all current paths and the
  Phase 1 changes baked in. Mark the old playbook as superseded in its
  own front-matter.
- **(b)** Delete the stale playbook entirely after Run 10 launches
  successfully. The CHANGELOG + INCIDENT docs + this notice + Phase 1
  README cover the operational details.

Option (a) is preferred. Run 10 will need a runbook too.
