#!/bin/bash
# scripts/launch_run10_vm.sh
# Run 10 training launch — produces the locked test AUROC that Run 9 failed
# to deliver (Run 9 lost it to ensemble.save() PicklingError; Phase 1 A1+A2
# fix that).
#
# Run 10 scope (minimal — fastest path to a locked test number):
#   - Uses existing run9_ready/ splits unchanged
#   - LOVD/DbNSFP silent-zero columns remain as in Run 9 (B1 wiring is in
#     code but exercised only when splits are regenerated with new flags)
#   - Single 'full' ablation; no ablation matrix
#   - Phase 1 fixes that are exercised automatically because they live in
#     variant_ensemble.py and evaluator.py:
#       A1: _CNN1D lifted to module-level _CNN1DModule (pickle resolves)
#       A2: per-model joblib checkpoint layout (single-model crash recoverable)
#       A3: evaluate() CatBoost DataFrame dispatch fix
#
# Run 10a will re-regen splits via run_phase2_eval.py with --lovd-path +
# --dbnsfp-path to populate the 4 silent-zero columns and measure the
# delta.  Run 10b will add FinnGen after pre-indexing the 30 GB TSV
# down to a ClinVar-intersected parquet.
#
# Required env vars (inject via SCP-up step in run10_launch.md):
#   GITHUB_TOKEN  - for git operations on the VM
#   VAST_API_KEY  - to authorise self-destroy
#   INSTANCE_ID   - the vast.ai instance ID to destroy on failure
#
# Auto-destroy semantics (Phase 1.7 improvements over Run 9):
#   - Pre-training failure: vastai destroy with `echo y |` (non-interactive).
#     vastai 1.0.12 destroy is interactive; without the pipe it would hang.
#   - Once TRAINING_STARTED=yes: NO auto-destroy. Manual SCP-back + destroy
#     from web console. Preserves INCIDENT_2026-04-29's local-landing-receipt
#     rule.
#
# With VAST_API_KEY/INSTANCE_ID unset, auto-destroy prints a manual
# instruction instead of calling the CLI, so the script is safe to dry-run.

set -e

TRAINING_STARTED=no

cleanup_if_setup_failed() {
    local rc=$?
    if [ "$rc" -ne 0 ] && [ "$TRAINING_STARTED" != "yes" ]; then
        echo ""
        echo "============================================================"
        echo "[auto-destroy] Setup/preflight failed with exit $rc"
        echo "[auto-destroy] No training started; no artefacts to lose"
        echo "============================================================"
        if command -v vastai >/dev/null 2>&1 && [ -n "${VAST_API_KEY:-}" ] && [ -n "${INSTANCE_ID:-}" ]; then
            vastai set api-key "$VAST_API_KEY" >/dev/null 2>&1 || true
            echo "[auto-destroy] Calling: echo y | vastai destroy instance $INSTANCE_ID"
            # Phase 1.7 fix: vastai 1.0.12 destroy is interactive. Pipe 'y'
            # so the call returns instead of hanging on the y/N prompt.
            echo y | vastai destroy instance "$INSTANCE_ID" 2>&1 || \
                echo "[auto-destroy] CLI destroy FAILED -- destroy from web console NOW: https://cloud.vast.ai/instances/"
        else
            echo "[auto-destroy] vastai CLI or VAST_API_KEY/INSTANCE_ID not available"
            echo "[auto-destroy] DESTROY MANUALLY: https://cloud.vast.ai/instances/"
        fi
    elif [ "$rc" -ne 0 ] && [ "$TRAINING_STARTED" = "yes" ]; then
        echo ""
        echo "============================================================"
        echo "[no-auto-destroy] Training had started; artefacts may exist"
        echo "[no-auto-destroy] SCP outputs back FIRST, then destroy manually"
        echo "[no-auto-destroy] Per-model checkpoints (Phase 1 A2) under"
        echo "[no-auto-destroy]   /workspace/outputs/run10/full/models/"
        echo "[no-auto-destroy] may be salvageable even on save() crash."
        echo "============================================================"
    fi
}
trap cleanup_if_setup_failed EXIT INT TERM

# ---- Activate venv if present (vast.ai PyTorch image convention) ----
source /venv/main/bin/activate 2>/dev/null || true

cd /workspace/genomic-variant-classifier

echo "=== HEAD ==="
git log -1 --oneline
echo "    expected post-Phase-1.5e: e07e3d8 or descendant"

echo "=== pip install -e . ==="
pip install -e .

echo "=== import + cuda sanity ==="
python -c "import genomic_variant_classifier, torch; \
    assert torch.cuda.is_available(), 'CUDA not available'; \
    print('OK', torch.cuda.get_device_name(0))"

echo "=== preflight (Run 10 connector checks + sklearn/lightgbm smoke fit) ==="
bash scripts/preflight_vm.sh

# ============================================================
# Training begins now -- auto-destroy is DISABLED past this line
# ============================================================
TRAINING_STARTED=yes
echo "=== TRAINING_STARTED=yes -- auto-destroy disabled. SCP outputs back manually after training. ==="
mkdir -p logs

# Run 10 paths: existing run9_ready splits, new run10 output dir
SPLITS_DIR=/workspace/outputs/run9_ready/splits
OUT_BASE=/workspace/outputs/run10
RUN_ID=run10
SEED=42
N_FOLDS=5

# Run 10 launches a single 'full' ablation to produce the locked test
# AUROC. Run 10a will extend this loop with the 13-ablation matrix.
for ABL in full ; do
    echo "============================================================"
    echo "==> $ABL @ $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
    echo "============================================================"
    python scripts/run9_ablations.py \
        --splits-dir "$SPLITS_DIR" \
        --ablation "$ABL" \
        --run-id "$RUN_ID" \
        --output-dir "$OUT_BASE/$ABL" \
        --seed "$SEED" \
        --n-folds "$N_FOLDS" \
        --log-level INFO \
        2>&1 | tee "logs/run10_${ABL}.log"
    rc=${PIPESTATUS[0]}
    echo "==> $ABL exit $rc"
    if [ "$rc" -ne 0 ]; then
        echo "==> ABORT: $ABL failed at $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
        echo "==> SCP outputs back from local NOW, then destroy from web console"
        echo "==> Per-model checkpoints under $OUT_BASE/$ABL/models/ may be salvageable"
        break
    fi
done

echo "============================================================"
echo "==> ALL DONE @ $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
echo "==> SCP outputs back from local NOW, then destroy from web console"
echo "==> Expected outputs under $OUT_BASE/full/ :"
echo "==>   metrics.json                (test + val AUROC)"
echo "==>   per_model_metrics.csv       (locked test, 11 base models)"
echo "==>   per_model_metrics_val.csv   (val for stacker reference)"
echo "==>   oof_test.parquet            (predictions on locked test)"
echo "==>   meta_val.parquet            (OOF-stacked val for blender)"
echo "==>   models/<name>.joblib        (per-model checkpoints, Phase 1 A2)"
echo "==>   models/orchestrator.joblib  (thin save state, Phase 1 A2)"
echo "============================================================"
