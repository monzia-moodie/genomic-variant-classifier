#!/bin/bash
# scripts/launch_run9_vm.sh
# Run 9 training launch with cost-safety auto-destroy on preflight failure.
#
# Required env vars (inject via SCP-up step in run9_launch.md):
#   GITHUB_TOKEN  - for git operations on the VM
#   VAST_API_KEY  - to authorise self-destroy
#   INSTANCE_ID   - the vast.ai instance ID to destroy on failure
#
# Auto-destroy semantics (CRITICAL):
#   - Pre-training failure (preflight, pip, import sanity): instance self-destroys.
#     No artefacts exist yet so nothing is lost. Stops the billing leak that
#     cost ~$1.00 on 2026-05-11.
#   - Once TRAINING_STARTED=yes is set (just before the ablation loop): NO auto-destroy.
#     SCP outputs back manually then destroy from web console. This preserves
#     INCIDENT_2026-04-29's local-landing-receipt rule.
#
# Test before paying for a VM:
#   - With VAST_API_KEY/INSTANCE_ID unset, the script still runs; auto-destroy
#     just prints a manual instruction instead of calling the CLI.

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
            echo "[auto-destroy] Calling: vastai destroy instance $INSTANCE_ID"
            vastai destroy instance "$INSTANCE_ID" 2>&1 || \
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
        echo "============================================================"
    fi
}
trap cleanup_if_setup_failed EXIT INT TERM

# ---- Activate venv if present (vast.ai PyTorch image convention) ----
source /venv/main/bin/activate 2>/dev/null || true

cd /workspace/genomic-variant-classifier

echo "=== HEAD ==="
git log -1 --oneline

echo "=== pip install -e . ==="
pip install -e .

echo "=== import + cuda sanity ==="
python -c "import genomic_variant_classifier, torch; \
    assert torch.cuda.is_available(), 'CUDA not available'; \
    print('OK', torch.cuda.get_device_name(0))"

echo "=== preflight (workflow-aware) ==="
bash scripts/preflight_vm.sh

# ============================================================
# Training begins now -- auto-destroy is DISABLED past this line
# ============================================================
TRAINING_STARTED=yes
echo "=== TRAINING_STARTED=yes -- auto-destroy disabled. SCP outputs back manually after training. ==="
mkdir -p logs

SPLITS_DIR=/workspace/outputs/run9_ready/splits
OUT_BASE=/workspace/outputs/run9
RUN_ID=run9
SEED=42
N_FOLDS=5

for ABL in full no_spliceai no_gnn no_alphamissense no_conservation no_population_af ; do
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
        2>&1 | tee "logs/run9_${ABL}.log"
    rc=${PIPESTATUS[0]}
    echo "==> $ABL exit $rc"
    if [ "$rc" -ne 0 ]; then
        echo "==> ABORT: $ABL failed at $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
        echo "==> SCP outputs back from local NOW, then destroy from web console"
        break
    fi
done

echo "============================================================"
echo "==> ALL DONE @ $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
echo "==> SCP outputs back from local NOW, then destroy from web console"
echo "============================================================"
