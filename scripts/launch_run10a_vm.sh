#!/usr/bin/env bash
# launch_run10a_vm.sh — Run 10a: regen splits with LOVD + DbNSFP wired, then train
# Expected wall-clock: ~12-13 hr (15 min regen + 12 hr training)
# Expected cost: ~$7-9 at $0.473/hr (RTX 4090)
#
# This script runs ON the Vast.ai instance after SCP upload.
# It uses run_phase2_eval.py (all-in-one: data prep → annotation → split → train → evaluate).
#
# SCP upload plan (~1.3 GB total, run from local PowerShell BEFORE launching):
#   See the companion PowerShell block at the bottom of this file.
set -euo pipefail

LOG=/workspace/run10a_master.log
OUTDIR=/workspace/outputs/run10a/full
REPO=/workspace/genomic-variant-classifier
DATA=/workspace/data

# ── Trap: banner on exit ──────────────────────────────────────────────────
cleanup() {
    local rc=$?
    echo "============================================================" | tee -a "$LOG"
    if [ $rc -eq 0 ]; then
        echo "==> run_phase2_eval.py exit 0 (success)" | tee -a "$LOG"
    else
        echo "==> run_phase2_eval.py exit $rc" | tee -a "$LOG"
        echo "==> ABORT: failed at $(date -u +'%Y-%m-%d %H:%M:%S') UTC" | tee -a "$LOG"
        echo "==> SCP outputs back from local NOW, then destroy from web console" | tee -a "$LOG"
        echo "==> Per-model checkpoints under $OUTDIR/models/ may be salvageable" | tee -a "$LOG"
    fi
    echo "============================================================" | tee -a "$LOG"
    echo "==> ALL DONE @ $(date -u +'%Y-%m-%d %H:%M:%S') UTC" | tee -a "$LOG"
    echo "==> SCP outputs back from local NOW, then destroy from web console" | tee -a "$LOG"
    echo "==> Expected outputs under $OUTDIR/ :" | tee -a "$LOG"
    echo "==>   metrics.json                (test + val AUROC)" | tee -a "$LOG"
    echo "==>   per_model_metrics.csv       (locked test, all base models)" | tee -a "$LOG"
    echo "==>   per_model_metrics_val.csv   (val for stacker reference)" | tee -a "$LOG"
    echo "==>   splits/                     (regenerated with LOVD+DbNSFP)" | tee -a "$LOG"
    echo "==>   models/ensemble.joblib      (full ensemble)" | tee -a "$LOG"
    echo "==>   models/ensemble_models/     (per-model checkpoints)" | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"
}
trap cleanup EXIT

echo "==> Run 10a launch @ $(date -u +'%Y-%m-%d %H:%M:%S') UTC" | tee "$LOG"

# ── Preflight ─────────────────────────────────────────────────────────────
echo "==> Preflight checks" | tee -a "$LOG"

FAIL=0
for f in \
    "$DATA/processed/clinvar_grch38.parquet" \
    "$DATA/processed/gnomad_v4_exomes.parquet" \
    "$DATA/external/spliceai/spliceai_index.parquet" \
    "$DATA/external/alphamissense/AlphaMissense_hg38.tsv.gz" \
    "$DATA/external/lovd/lovd_all_variants.parquet" \
    "$DATA/external/dbnsfp/dbnsfp_clinvar_index.parquet" \
; do
    if [ ! -f "$f" ]; then
        echo "==> MISSING: $f" | tee -a "$LOG"
        FAIL=1
    else
        SZ=$(stat -c%s "$f" 2>/dev/null || echo 0)
        echo "==> OK: $f ($(( SZ / 1048576 )) MB)" | tee -a "$LOG"
    fi
done

if [ $FAIL -eq 1 ]; then
    echo "==> ABORT: missing data files. SCP them first." | tee -a "$LOG"
    exit 2
fi

# Repo
if [ ! -d "$REPO/.git" ]; then
    echo "==> Cloning repo ..." | tee -a "$LOG"
    cd /workspace
    git clone https://github.com/monzia-moodie-repo-projects/genomic-variant-classifier.git
fi
cd "$REPO"
git pull origin main 2>&1 | tee -a "$LOG"
echo "==> HEAD: $(git rev-parse --short HEAD)" | tee -a "$LOG"

# Dependencies
pip install pykan 2>&1 | tail -1 | tee -a "$LOG"
pip install -e . 2>&1 | tail -1 | tee -a "$LOG"

# STRING DB (optional, for GNN)
STRING_LINKS="$DATA/external/string/9606.protein.links.detailed.v12.0.txt.gz"
STRING_INFO="$DATA/external/string/9606.protein.info.v12.0.txt.gz"
STRING_ARG=""
if [ -f "$STRING_LINKS" ] && [ -f "$STRING_INFO" ]; then
    # Symlink into repo data/ so the script finds them at the default path
    mkdir -p data/external/string
    ln -sf "$STRING_LINKS" data/external/string/
    ln -sf "$STRING_INFO" data/external/string/
    STRING_ARG="--string-db auto"
    echo "==> STRING DB wired (GNN enabled)" | tee -a "$LOG"
else
    echo "==> STRING DB not found (GNN skipped)" | tee -a "$LOG"
fi

# ── Launch ────────────────────────────────────────────────────────────────
echo "==> Starting run_phase2_eval.py @ $(date -u +'%Y-%m-%d %H:%M:%S') UTC" | tee -a "$LOG"
mkdir -p "$OUTDIR"

python scripts/run_phase2_eval.py \
    --clinvar       "$DATA/processed/clinvar_grch38.parquet" \
    --gnomad        "$DATA/processed/gnomad_v4_exomes.parquet" \
    --spliceai      "$DATA/external/spliceai/spliceai_index.parquet" \
    --alphamissense "$DATA/external/alphamissense/AlphaMissense_hg38.tsv.gz" \
    --lovd-path     "$DATA/external/lovd/lovd_all_variants.parquet" \
    --dbnsfp-path   "$DATA/external/dbnsfp/dbnsfp_clinvar_index.parquet" \
    --gtex-genes    BRCA1 BRCA2 TP53 PTEN ATM \
    $STRING_ARG \
    --output        "$OUTDIR" \
    2>&1 | tee -a "$LOG"

echo "==> run_phase2_eval.py finished @ $(date -u +'%Y-%m-%d %H:%M:%S') UTC" | tee -a "$LOG"

# Post-condition: verify key outputs exist
for f in "$OUTDIR/metrics.json" "$OUTDIR/per_model_metrics.csv"; do
    if [ -f "$f" ]; then
        echo "==> VERIFIED: $f" | tee -a "$LOG"
    else
        echo "==> WARNING: expected output missing: $f" | tee -a "$LOG"
    fi
done

# Print headline AUROC from metrics.json
if [ -f "$OUTDIR/metrics.json" ]; then
    python -c "
import json
m = json.load(open('$OUTDIR/metrics.json'))
print(f\"==> TEST AUROC: {m.get('auroc', 'N/A')}\")
print(f\"==> VAL  AUROC: {m.get('val_auroc', 'N/A')}\")
print(f\"==> Run 10 baseline: 0.98163 (compare ΔAUROC)\")
" 2>&1 | tee -a "$LOG"
fi

# ========================================================================
# COMPANION: PowerShell SCP upload commands (run BEFORE launching this script)
# ========================================================================
# $KEY  = "$env:USERPROFILE\.ssh\id_lambda_run8"
# $PORT = <instance_port>
# $HOST = "<instance_ip>"
# $R    = "root@$HOST"
#
# # Create directory structure on VM
# ssh -i $KEY -p $PORT -T $R 'mkdir -p /workspace/data/processed /workspace/data/external/spliceai /workspace/data/external/alphamissense /workspace/data/external/lovd /workspace/data/external/dbnsfp /workspace/data/external/string'
#
# # SCP all 8 data files (~1.3 GB total, ~3-5 min)
# scp -i $KEY -P $PORT "data\processed\clinvar_grch38.parquet"                           "${R}:/workspace/data/processed/"
# scp -i $KEY -P $PORT "data\processed\gnomad_v4_exomes.parquet"                         "${R}:/workspace/data/processed/"
# scp -i $KEY -P $PORT "data\external\spliceai\spliceai_index.parquet"                   "${R}:/workspace/data/external/spliceai/"
# scp -i $KEY -P $PORT "data\external\alphamissense\AlphaMissense_hg38.tsv.gz"           "${R}:/workspace/data/external/alphamissense/"
# scp -i $KEY -P $PORT "data\external\lovd\lovd_all_variants.parquet"                    "${R}:/workspace/data/external/lovd/"
# scp -i $KEY -P $PORT "data\external\dbnsfp\dbnsfp_clinvar_index.parquet"               "${R}:/workspace/data/external/dbnsfp/"
# scp -i $KEY -P $PORT "data\external\string\9606.protein.links.detailed.v12.0.txt.gz"   "${R}:/workspace/data/external/string/"
# scp -i $KEY -P $PORT "data\external\string\9606.protein.info.v12.0.txt.gz"             "${R}:/workspace/data/external/string/"
#
# # SCP the launch script
# scp -i $KEY -P $PORT "scripts\launch_run10a_vm.sh" "${R}:/workspace/"
#
# # Launch in tmux
# ssh -i $KEY -p $PORT -T $R 'chmod +x /workspace/launch_run10a_vm.sh && tmux new-session -d -s run10a "bash /workspace/launch_run10a_vm.sh"'
#
# # Monitor
# ssh -i $KEY -p $PORT -T $R 'tail -c 500 /workspace/run10a_master.log'
