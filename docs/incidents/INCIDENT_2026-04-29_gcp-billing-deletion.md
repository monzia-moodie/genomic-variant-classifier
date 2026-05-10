# INCIDENT 2026-04-29: GCP project deletion + architectural pivot to SCP-only

**Severity**: HIGH (architectural change — removes a major dependency)
**Status**: RESOLVED — pivoted to SCP-only Vast.ai workflow
**Date opened**: 2026-04-29
**Date resolved**: 2026-05-01 (architectural redesign committed in chat 820fbc38)
**Date documented**: 2026-05-10 (this incident; previously implicit in session record only)

## Summary

Between approximately 2026-04-19 and 2026-04-29 the GCP project
`genomic-variant-prod` accrued charges for crashed Run 8 retraining attempts
that produced no useful outputs. To stop further charges, the entire GCP
project was destroyed on or about 2026-04-29. This terminated all GCS-resident
artifacts (`gs://genomic-variant-prod-outputs/`, `gs://genomic-classifier-data/`)
and triggered an architectural pivot away from any third-party object storage
for training-data movement.

The decision was finalized and documented in chat 820fbc38 ("Comprehensive
project status assessment") on 2026-05-01 as part of Run 9a launch preparation.
A subsequent session on 2026-05-04 (chat ac25e111) captured a transitional
intent to keep "5 TB Google Cloud" for non-GPU storage; that intent was
superseded by the full deletion. Memory and operational documentation
continued to reference GCS until the architecture audit on 2026-05-10
(this session).

## Root cause

GCP billing for crashed runs combined with retry-loop overhead on Run 8
attempts. The mechanism by which Monzia was billed despite no useful output
was not investigated further before the deletion decision; stopping the bleed
took priority over root-causing the billing model.

## Decision

Pivot to **SCP-only Vast.ai workflow**:

| Layer | New target |
|---|---|
| Source of truth | Local Windows machine (`C:\Projects\genomic-variant-classifier\`) |
| GPU compute (scratch only) | Vast.ai instance (SSH'd via `id_lambda_run8`) |
| Transport | `scp` between local ↔ Vast.ai |
| Agent-layer durability | Drive via rclone `genvarcla:` (drift_reports/, events/, litcache/, modelscout/, trainlifecycle/ — NOT run data) |
| Cloud object storage | NONE |

## Impact

### Lost (in-GCS only — no on-disk equivalent)
- `gs://genomic-variant-prod-outputs/` — Run 6/7/8 model checkpoints,
  ablation parquets, eval reports, drift HTML
- `gs://genomic-classifier-data/` — possibly the SpliceAI raw VCF mirror

### Preserved
- **Local disk**: all training data (`data\external\`, `data\raw\`,
  `outputs\run9_ready\`)
- **Drive (`genvarcla:`)**: agent-layer outputs (drift_reports/, events/,
  litcache/, modelscout/, trainlifecycle/) — confirmed via `rclone lsf` on
  2026-05-10
- **GitHub**: full code history, all Run 8 results in CHANGELOG / SESSION docs

**Run 9 is NOT BLOCKED** by this incident. All inputs needed for Run 9
(splits, SpliceAI parquet, ClinVar VCF, AlphaMissense, STRING DB, gnomAD
constraints) are present on local disk per `agent_data\gcs_state_diagnose.out.txt`.

## Forward-looking architecture

See `docs/RUN9_OPERATIONS_PLAYBOOK.md` (post-2026-05-10 update) for the full
sequence. Summary:

1. Local pre-flight (`scripts/preflight_check.py`, GCS checks removed)
2. Provision Vast.ai instance (RTX 4090, 60 GB RAM, 200 GB disk, CUDA 12.x)
3. SSH in via `id_lambda_run8`
4. SCP from local → instance (~1.0–1.2 GB total, 10–30 min on home upstream)
5. On-VM pre-flight (`scripts/preflight_vm.sh`, gcloud calls removed)
6. Train per `scripts/run9_launch.md`
7. SCP results back to local
8. **Destroy instance immediately upon training completion** (standing rule)

## Cross-references

- chat `820fbc38` — 2026-05-01 23:43 UTC, decision finalization
- chat `ac25e111` — 2026-05-04 07:10 UTC, transitional state (superseded)
- `docs/CHANGELOG.md` — 2026-05-10 entry records the GCS dependency removal commits
- `docs/HANDOFF_run9_launch.md` — operational handoff (post-update)
- `scripts/preflight_check.py` — GCS checks removed (commit 1 of cleanup batch)