# Run 9 Launch Runbook

**Entrypoint**: `scripts/run9_ablations.py` (LOCO ablation harness over pre-built splits).
**NOT** `scripts/run_phase2_eval.py` — that one regenerates splits from raw ClinVar (13h CPU; would burn GPU budget for no gain).

**Target**: Beat Run 8's AUROC 0.9863 with the six-ablation LOCO decomposition per `docs/RUN9_SCIENTIFIC_DESIGN.md` §3.2.

**Known state**: ESM-2 will run in STUB MODE per `docs/incidents/INCIDENT_2026-04-17_esm2-hgvsp-parser.md`. Stub-mode log lines are documented and expected, not failures. Run 10 priority is the HGVSp parser.

## Architecture (post-INCIDENT_2026-04-29)

- Data flow: Windows local source-of-truth ↔ Vast.ai GPU scratch via SCP (~1.0–1.2 GB per leg).
- No remote object storage. GCP project `genomic-variant-prod` was deleted 2026-04-29.
- Splits already built at `outputs/run9_ready/splits/` (78 cols, 1,700,687 labeled variants, train 1.2M / val 154K / test 349K, AUROC 0.9814 baseline from 2026-04-30 CPU regen).

## Pre-launch gate (local, before spending any money)

```powershell
cd C:\Projects\genomic-variant-classifier
C:\Projects\genomic-variant-classifier\.venv312\Scripts\Activate.ps1
$env:PYTHONPATH = "C:\Projects\genomic-variant-classifier"

# Clear pyc caches (standing practice)
Get-ChildItem -Recurse -Filter "__pycache__" -Directory | Remove-Item -Recurse -Force

# Run the scripted preflight. Fails loudly on any gap.
python scripts\preflight_check.py
echo "preflight exit: $LASTEXITCODE"

# Fast iteration mode (skips the 5-10 min pytest):
#   python scripts\preflight_check.py --skip-pytest
```

**Do not proceed if exit code is non-zero.** Fix every FAIL line before continuing. The preflight gates GPU spending; bypassing it is how Run 8 ended up with silent-zero features.

## Vast.ai instance spec

- GPU: **RTX 4090** (24 GB VRAM)
- RAM: **>= 60 GB** (dbNSFP + SpliceAI lookups are memory-heavy)
- Disk: **>= 200 GB** (parquets + caches + per-ablation model artefacts)
- CUDA: **12.x** PyTorch image
- SSH key: `C:\Users\monzi\.ssh\id_lambda_run8` (verified present; pubkey present; no-2FA API key authenticated 2026-05-10 via `vastai search offers`)

Budget: 6 ablations × ~7000s each on RTX 4090 @ ~$0.27/hr + SHAP + permutation importance overhead ≈ **$5–10 total**. Destroy the instance the moment the final ablation prints.

## SCP data up — FROM local, BEFORE training

Set the connection variables once after provisioning (Vast.ai gives you the SSH command in the web UI):

```powershell
# Adjust per the Vast.ai instance details
$VAST_HOST = "ssh<N>.vast.ai"
$VAST_PORT = <port>
$KEY = "$env:USERPROFILE\.ssh\id_lambda_run8"
$REMOTE = "vastuser@${VAST_HOST}"

# 1. Send pre-built splits (~70 MB; the whole splits/ subdir plus the scaler)
scp -i $KEY -P $VAST_PORT -r outputs\run9_ready ${REMOTE}:/workspace/outputs/

# 2. Send external data
scp -i $KEY -P $VAST_PORT data\external\spliceai\spliceai_index.parquet ${REMOTE}:/workspace/data/external/spliceai/
scp -i $KEY -P $VAST_PORT data\external\string\9606.protein.links.detailed.v12.0.txt.gz ${REMOTE}:/workspace/data/external/string/
scp -i $KEY -P $VAST_PORT data\external\string\9606.protein.info.v12.0.txt.gz ${REMOTE}:/workspace/data/external/string/
scp -i $KEY -P $VAST_PORT data\external\alphamissense\AlphaMissense_hg38.tsv.gz ${REMOTE}:/workspace/data/external/alphamissense/
scp -i $KEY -P $VAST_PORT data\external\gnomad\gnomad.v4.1.constraint_metrics.tsv ${REMOTE}:/workspace/data/external/gnomad/
scp -i $KEY -P $VAST_PORT data\external\dbnsfp\dbnsfp_full_index.parquet ${REMOTE}:/workspace/data/external/dbnsfp/

# 3. Verify total SCP-up size (~1.0-1.2 GB)
ssh -i $KEY -p $VAST_PORT $REMOTE "du -sh /workspace/outputs /workspace/data"
```

If any path doesn't exist locally, check `data\external\` inventory before launching. Missing inputs cause silent-zero features (see Run 8 lessons).

## On-VM setup (after SSH)

```bash
# 1. Clone repo and confirm HEAD
cd /workspace
git clone https://${GITHUB_TOKEN}@github.com/monzia-moodie-repo-projects/genomic-variant-classifier.git
cd genomic-variant-classifier
git log -1 --oneline   # must match the HEAD you launched from locally (899cae5 or newer)

# 2. Install requirements. Editable install required post-C2 (pyproject.toml replaces setup.py).
python -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
pip install -e .                            # editable install (genomic_variant_classifier namespace)
pip install "transformers>=4.40,<5.0"       # ESM-2 backend (stub mode in Run 9)
pip install pykan                           # KAN base estimator (memory: reinstated for Vast.ai GPU runs)

# 3. Verify data staged
ls -lh /workspace/outputs/run9_ready/splits/X_train.parquet  # expect ~5 MB (1.2M rows, 78 cols)
ls -lh /workspace/data/external/spliceai/spliceai_index.parquet  # expect ~336 MB
ls -lh /workspace/data/external/string/9606.protein.links.detailed.v12.0.txt.gz

# 4. Trap-EXIT shutdown handler — MUST run before training
#    Fires on ANY exit: success, failure, crash, OOM, Ctrl-C.
#    Note: destroy via web console at the end -- this trap is for abnormal exit only.
trap 'echo "[trap] training process exited; SCP your outputs back from local NOW before destroying"' EXIT INT TERM

# 5. VM preflight (must exit 0 before training)
bash scripts/preflight_vm.sh
echo "vm preflight exit: $?"
```

## Launch training — six-ablation LOCO sequence

Per `docs/RUN9_SCIENTIFIC_DESIGN.md` §3.2: `full` + 5 LOCO ablations on the same fixed splits, same seed, so deltas are directly comparable.

```bash
# Repo root, on the VM, inside the venv
mkdir -p logs

SPLITS_DIR=/workspace/outputs/run9_ready/splits
OUT_BASE=/workspace/outputs/run9
RUN_ID=run9
SEED=42
N_FOLDS=5

for ABL in full no_spliceai no_gnn no_alphamissense no_conservation no_population_af ; do
    echo "============================================================"
    echo "==> Starting ablation: $ABL at $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
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
    echo "==> Done: $ABL (exit $rc)"
    if [ $rc -ne 0 ]; then
        echo "==> ABORT: $ABL failed. Inspect logs/run9_${ABL}.log before deciding to continue."
        break
    fi
done
```

Each ablation produces its own artefact set in `outputs/run9/<ABL>/` per RUN9_SCIENTIFIC_DESIGN.md Rule 5 (manifest, oof_predictions, test_predictions, shap_values, calibration, feature_importance, ablation_results, graph_stats).

## Success grep checklist — run per ablation

```bash
# Per-ablation diagnostic sweep
for ABL in full no_spliceai no_gnn no_alphamissense no_conservation no_population_af ; do
    echo "=========================================="
    echo "=== $ABL"
    echo "=========================================="
    # GNN on GPU (not CPU fallback) -- only relevant for ablations that include GNN
    grep -E "GNN device: cuda|GNN.*cuda:0" "logs/run9_${ABL}.log" | head -2
    # STRING DB loaded with non-zero edges
    grep -E "STRING DB.*loaded [1-9][0-9]* edges|STRING.*edges: [1-9]" "logs/run9_${ABL}.log" | head -2
    # SpliceAI annotated non-zero variants -- only in ablations that include SpliceAI
    grep -E "SpliceAI.*[1-9][0-9]+ variants|splice.*annotated.*[1-9]" "logs/run9_${ABL}.log" | head -2
    # AlphaMissense regression check (Run 8 fix)
    grep -E "AlphaMissense.*[1-9][0-9]+ variants" "logs/run9_${ABL}.log" | head -2
    # ESM-2 stub-mode expected line (1+ in every ablation; absence = different bug)
    grep -i "ESM-2 stub mode" "logs/run9_${ABL}.log" | head -1
    # Final holdout AUROC
    grep -E "holdout.*AUROC|AUROC:.*0\.9[89]" "logs/run9_${ABL}.log" | tail -3
done
```

Expected pattern:
- `full`: GNN on cuda, STRING edges > 100K, SpliceAI > 0 variants, AlphaMissense > 0 variants, ESM-2 stub line present, AUROC > 0.985
- `no_spliceai`: SpliceAI columns zeroed; ΔAUROC vs `full` = SpliceAI's unique contribution
- `no_gnn`: GNN head disabled; ΔAUROC vs `full` = GNN's unique contribution
- `no_alphamissense` / `no_conservation` / `no_population_af`: same pattern, different feature class

## Post-training artefact sanity check

```bash
# Each ablation should have produced the Rule-5 artefact set
for ABL in full no_spliceai no_gnn no_alphamissense no_conservation no_population_af ; do
    echo "=== $ABL artefacts ==="
    ls -la "/workspace/outputs/run9/${ABL}/" 2>/dev/null
done

# Cross-ablation roll-up (each ablation_results.parquet typically aggregates all configurations)
python -c "
import pandas as pd
from pathlib import Path

base = Path('/workspace/outputs/run9')
for abl in ['full','no_spliceai','no_gnn','no_alphamissense','no_conservation','no_population_af']:
    p = base / abl / 'ablation_results.parquet'
    if p.exists():
        df = pd.read_parquet(p)
        print(f'{abl}: rows={df.shape[0]} cols={df.shape[1]}')
    else:
        print(f'{abl}: MISSING ablation_results.parquet')
"

# ESM-2 stub-status check on the 'full' ablation's test_predictions
python -c "
import pandas as pd
df = pd.read_parquet('/workspace/outputs/run9/full/test_predictions.parquet')
if 'esm2_delta_norm' in df.columns:
    d = df['esm2_delta_norm']
    print(f'esm2_delta_norm: n={len(d)} zero_frac={(d==0).mean():.4f} '
          f'mean={d.mean():.6f} max={d.max():.6f}')
else:
    print('esm2_delta_norm not in test_predictions (expected if harness drops zero-variance cols)')
"
# Expected: zero_frac == 1.0000 (ESM-2 stub mode). Run 10 + HGVSp parser flips this.
```

## Shutdown — SCP outputs back BEFORE destroying

This order is non-negotiable. Destroy = data loss. SCP first, verify locally, then destroy.

```powershell
# Run on local Windows box
$VAST_HOST = "ssh<N>.vast.ai"
$VAST_PORT = <port>
$KEY = "$env:USERPROFILE\.ssh\id_lambda_run8"
$REMOTE = "vastuser@${VAST_HOST}"

# 1. SCP outputs back
scp -i $KEY -P $VAST_PORT -r ${REMOTE}:/workspace/outputs/run9 outputs\
scp -i $KEY -P $VAST_PORT -r ${REMOTE}:/workspace/genomic-variant-classifier/logs outputs\run9\

# 2. Verify local landing (INCIDENT_2026-04-29: local listing IS the receipt)
Get-ChildItem -LiteralPath outputs\run9 -Recurse | Measure-Object Length -Sum
Get-ChildItem -LiteralPath outputs\run9 -Recurse -Directory | Select-Object FullName

# 3. Confirm per-ablation artefact directories exist
foreach ($ABL in 'full','no_spliceai','no_gnn','no_alphamissense','no_conservation','no_population_af') {
    $p = "outputs\run9\$ABL"
    if (Test-Path -LiteralPath $p) {
        $sz = (Get-ChildItem -LiteralPath $p -Recurse -File | Measure-Object Length -Sum).Sum
        '{0,-22} [OK] {1:N1} MB' -f $ABL, ($sz / 1MB)
    } else {
        '{0,-22} [FAIL] missing' -f $ABL
    }
}
```

**Destroy the Vast.ai instance** — web console → **Destroy** (NOT Stop; Stop keeps storage billing alive).
**Confirm $0/hr** at https://cloud.vast.ai/billing/.

## Post-run documentation (mandatory, per standing rule)

Per `docs/RUN9_SCIENTIFIC_DESIGN.md` §6, the session doc has 11 mandatory sections. Auto-generate skeleton via `scripts/generate_run_report.py` (T7 task — if not yet built, write manually for Run 9 and ship the generator before Run 10).

1. `docs/sessions/SESSION_YYYY-MM-DD_run9.md` — full session record:
   - Run metadata (git SHA, date, wall time, GPU cost, instance spec)
   - Baseline comparison table (Run 8 vs Run 9 full, per-ablation)
   - Per-base-model OOF AUROC (all base estimators active in each ablation)
   - LOCO ablation table (6 rows, ΔAUROC vs full, bootstrap CI)
   - Per-consequence breakdown
   - Top-20 features by permutation importance
   - Graph stats (GNN: node/edge count, degree distribution)
   - `esm2_delta_norm` zero-fraction (should be 1.0; will flip in Run 10 with HGVSp parser)
   - SpliceAI feature importance rank
   - Total GPU cost and wall time
   - Local-disk listing receipt for `outputs/run9/` (INCIDENT_2026-04-29 verification rule)

2. `docs/CHANGELOG.md` — append `## YYYY-MM-DD — Run 9` entry.

3. `docs/ROADMAP.md` — update:
   - Bump "Last updated" line
   - Check off Phase 4D items (GNN with real STRING edges; SpliceAI activation)
   - ESM-2 still OPEN, blocked on HGVSp parser (Run 10 headline)
   - Formally schedule Spectral Path Regression (Coombs 2025) eval for Run 10 alongside the HGVSp parser

4. Commit docs as a SEPARATE commit from any code changes (clean diffs).

## What to do if it fails partway

If any ablation crashes or a feature (other than ESM-2) is silently zero:

1. **Do not retry from a re-provisioned instance.** Inspect logs first.
2. Destroy the instance anyway — debugging doesn't require GPU billing.
3. SCP all logs to local disk:
   ```powershell
   scp -i $KEY -P $VAST_PORT -r ${REMOTE}:/workspace/genomic-variant-classifier/logs outputs\run9_partial\
   ```
4. Write `docs/incidents/INCIDENT_YYYY-MM-DD_run9-<slug>.md` with:
   - Exact error (stack trace if Python crash; grep for `Error|FAIL|Traceback` in logs)
   - Which ablation(s) completed vs failed
   - Root cause (if determined) or "unknown, see logs"
   - Local-disk listing receipt for the partial artefacts
   - The fix plan before re-running

INCIDENT_2026-04-29 verification rule: no "RESOLVED" without a local-disk listing receipt. Partial runs are still informative — if `full` completed but a LOCO ablation failed, you have one new data point relative to Run 8 and the missing ablation can be re-run separately on a new instance with the same splits.
