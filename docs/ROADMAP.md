---

## Deferred Item: OpenMed Psychiatric GWAS Dataset Integration

**Logged:** 2026-04-08
**Status:** DEFERRED — do not implement before Phase 3 Polars milestone
**Target milestone:** Phase 3 (Polars evaluation) — see rationale below

---

### Background

OpenMed released a Hugging Face dataset of **1.14 billion rows** of
standardised psychiatric genetics summary statistics, consolidating every
GWAS meta-analysis ever published by the Psychiatric Genomics Consortium
(PGC). Covers 12 conditions across 52 landmark studies:

- ADHD, Depression, Schizophrenia, Bipolar disorder, PTSD, OCD, Autism,
  Anxiety, Tourette syndrome, Eating disorders, and additional sub-phenotypes.

Each row is a single variant-phenotype association test containing:
`rsID`, `CHR/POS`, `A1/A2`, `BETA`/`OR`, `SE`, `P`, `INFO`, `FRQ/MAF`,
`N`, `Nca`, `Nco`.

---

### Clinical Relevance Assessment

**Biological mismatch (key constraint):** The PGC GWAS studies are powered
almost entirely by common variants (MAF > 1–5%) with small effect sizes
(OR 1.05–1.3). The classifier's primary signal comes from rare, large-effect
Mendelian variants (MAF < 0.1%, OR >> 5). Feature signal for the primary
pathogenicity task is expected to be **weak to marginal** overall.

**Legitimate value identified:**

1. **Pleiotropy annotation for psychiatric gene panels.** Genes such as
   *CACNA1C*, *SCN2A*, *NRXN1*, *SHANK3* carry both rare pathogenic alleles
   and common GWAS signals. A locus-level enrichment feature could improve VUS
   interpretation specifically for neuro/psychiatric panel variants.

2. **Functional region flagging.** High-density GWAS signal (many variants
   p < 1×10⁻⁵) is an indirect proxy for regulatory/functional importance.

3. **Cross-disorder pleiotropy score.** The multi-condition architecture of
   this dataset uniquely enables a meta-feature: number of independent
   psychiatric conditions with genome-wide significant signal near a locus.

4. **Gap addressed.** The current 78-feature set has no psychiatric /
   neurodevelopmental phenotypic annotation. For *de novo* neurodevelopmental
   disease genes, this fills a real gap.

**Label leakage status:** No direct leakage path — GWAS summary stats are
population-level aggregates, not clinical assertions. Audit recommended to
ensure no ClinVar P/LP labels for psychiatric variants were derived from GWAS
evidence at ingestion time.

---

### Planned Feature Schema (5 new features)

```python
# connector: src/data/database_connectors.py — PsychGWASConnector
# DEFAULT_SCORE = 0.0 (no signal — correct default for rare pathogenic variants)

gwas_psych_min_pval           # min p-value across all 52 studies
gwas_psych_hit_count          # number of studies with p < 5×10⁻⁸
gwas_psych_disorder_breadth   # number of distinct conditions significant
gwas_psych_max_neg_log10p     # strongest single-study signal
gwas_psych_is_lead_snp        # bool: is this a GWAS lead SNP
```

Follows existing `annotate_dataframe()` connector pattern with stub mode.

---

### Preprocessing Requirements (Critical)

Raw dataset is ~80–150 GB on disk. **Must filter before any downstream use:**

```python
# Step 1: Filter to genome-wide significant hits only
# Reduces 1.14B rows → ~2–5M rows across 52 studies

df_filtered = df.filter(pl.col("P") < 5e-8)   # Polars — see rationale below

# Step 2: Aggregate per rsID into a summary lookup table
# Output: ~200–500 MB Parquet — compatible with existing connector pattern

lookup = (
    df_filtered
    .group_by("rsID")
    .agg([
        pl.col("P").min().alias("gwas_psych_min_pval"),
        pl.col("STUDY_ID").n_unique().alias("gwas_psych_hit_count"),
        pl.col("CONDITION").n_unique().alias("gwas_psych_disorder_breadth"),
        (-pl.col("P").log(base=10)).max().alias("gwas_psych_max_neg_log10p"),
        pl.col("IS_LEAD_SNP").any().alias("gwas_psych_is_lead_snp"),
    ])
)
```

---

### Why This Is the Phase 3 Polars Benchmark Candidate

The preprocessing pipeline above (filter 1.14B rows → aggregate per rsID)
is an ideal real-world benchmark for the **Phase 3 Polars evaluation**:

- Demonstrates Polars throughput advantage vs. pandas at scale
- Solves the storage/preprocessing problem as a by-product of the evaluation
- No model changes required — only a new connector + 5 feature columns
- Feature count assertions will need updating: search
  `Get-ChildItem -Path "src","tests" -Recurse -Filter "*.py" | Select-String -Pattern "== N"`
  and increment by 5 before running the test suite

**Proposed sequence at Phase 3 milestone:**
1. Run Polars bottleneck evaluation on existing pandas operations (primary goal)
2. Use this GWAS preprocessing as the scale benchmark case
3. If Polars is adopted, implement `PsychGWASConnector` using Polars throughout
4. Add 5 features; update all hardcoded feature count assertions
5. Retrain; evaluate AUROC delta — expect marginal improvement for global
   classifier, potentially larger for psychiatric gene panel sub-cohort

---

### Data Source

```
Dataset:    OpenMed Psychiatric Genomics Consortium (standardised)
Location:   Hugging Face — https://huggingface.co/datasets/openmed/pgc-gwas
Conditions: ADHD, Depression, Schizophrenia, Bipolar, PTSD, OCD, Autism,
            Anxiety, Tourette, Eating disorders (12 total)
Studies:    52 PGC meta-analyses
Rows:       ~1.14 billion
Format:     Parquet (recommended download format)
Local path: G:\My Drive\genomic-variant-data\external\pgc_gwas_summary\
```

---

### Do Not Implement Until

- [x] Run 6 (GCP) training is complete with KAN removed and
      `gnomad_constraint_path` wired
      (Runs 6-8 completed without KAN; KAN reinstated for Run 9 — see
      2026-04-20 session. Commits b1c1150, 8f9eb60, 128331f.)
- [ ] Phase 3 Polars bottleneck evaluation is scheduled
- [ ] Feature count assertions audited and documented

---

---

## Run 6 — Completed 2026-04-08

**Holdout AUROC: 0.9862** | 78 features | 345K variants | 4,441 genes

### Activated in Run 6
- gnomAD v4.1 gene constraint: `pLI`, `LOEUF`, `syn_z`, `mis_z` (wired via `--gnomad-constraint`)
- KAN removed from Runs 6-8 (C++ OOM at scale was uncatchable without
  the 100K subsample gate from commit 2389ee2). Reinstated for Run 9
  after the subsample gate + Vast.ai GPU access made it tractable.
  See 2026-04-20 session; commits b1c1150, 8f9eb60, 128331f.
- AlphaMissense active (5.2GB TSV)
- Ensemble resume logic added to `run_phase2_eval.py`

### Deferred to Run 7
- GNN: torch-geometric missing on VM → zero GNN contribution
- SpliceAI: `spliceai_index.parquet` corrupt (29GB raw dump) → omitted
- ESM-2: verify non-stub on VM

### Infrastructure issues resolved
- gcloud working (SDK 564.0, project `genomic-variant-prod`)
- Correct GCS bucket: `gs://genomic-classifier-data/`
- VM startup script attachment bug identified (never ran)

### Run 7 priorities
1. Install torch-geometric on VM → activate GNN
2. Fix startup script attachment in `gcloud compute instances create`
3. Regenerate `spliceai_index.parquet` from raw VCF
4. Evaluate Polars to replace bottleneck pandas ops (Phase 3 milestone)

## Phase 4 Agenda Items

### LiteratureScoutAgent Implementation
Deferred from Phase 3. When implemented, configure the following watch targets:

**pykan memory fix (KAN re-enablement trigger) — RESOLVED 2026-04-20**

Historical context preserved; no longer active:

- Original plan: watch https://github.com/KindXiaoming/pykan releases
  for memory optimisation or OOM fixes, then remove unconditional
  `pop("kan", None)` from `run_phase2_eval.py`.
- What actually happened: the OOM was fixed in-tree by commit 2389ee2
  (2026-04-04) via a 100K-sample stratified subsample gate in
  `KANClassifier._fit_pykan`, which caps peak RAM at ~0.3 GB (from
  17.9 GB) regardless of pykan upstream behaviour. The
  `pop("kan", None)` remained hardcoded for Runs 6-8 anyway as
  belt-and-braces caution.
- Triggering condition changed: the combination of (a) the 2389ee2
  subsample gate already shipped and (b) Vast.ai GPU access for Run 9
  made the re-enablement checklist actionable without waiting for
  a pykan upstream fix.
- Resolution: 2026-04-20 session executed the re-enablement checklist.
  See commits b1c1150, 8f9eb60, 128331f.

Monitoring continues via `LiteratureScoutAgent`
(`agent_layer/agents/version_monitor_agent.py`, committed a95c9db)
for any future pykan improvements that might allow removing the
100K subsample gate or increasing its threshold.

**Database version monitoring (existing scope, document here for completeness)**
- ClinVar: monthly release cadence, watch for schema changes
- gnomAD: v4.2+ release, watch for constraint metrics column changes
- AlphaMissense: watch for hg38 TSV format updates
- STRING DB: v13 release monitoring

### DataFreshnessAgent Extension
Add pykan version check hook alongside existing ClinVar/gnomAD freshness checks:
- Check `pip index versions pykan` against installed version on each run
- Log new versions to agent state; surface in DataFreshnessAgent report
- This covers the gap until LiteratureScoutAgent is implemented

### KAN Re-enablement Checklist — COMPLETED 2026-04-20
- [x] `pip install pykan --upgrade` in local venv
      (pykan 0.2.8 installed locally prior to session; verified working)
- [x] Run `python -c "import kan; m = kan.KAN([78,64,1])"` smoke test
      (2026-04-20 500-row synthetic probe via `VariantEnsemble.fit`
      trained KAN successfully in ~90 seconds, pykan backend active)
- [x] Train on 150K stratified subsample locally, confirm no OOM
      (superseded: commit 2389ee2 already caps subsample at 100K with
      peak RAM ~0.3 GB; 150K manual test not needed)
- [x] Remove `ensemble.base_estimators.pop("kan", None)` from
      `run_phase2_eval.py` (commit 8f9eb60, 2026-04-20)
- [x] Add `--skip-kan` flag as optional override — do not hardcode
      removal again (commit 8f9eb60, 2026-04-20)
- [ ] Re-run full training on GCP, compare AUROC delta vs Run 7/8
      baseline (scheduled for Run 9 on Vast.ai; user action pending)

---

## Infrastructure: GPU-Enabled VMs (Run 8+)

**Problem:** n2-highmem-32 is CPU-only. Run 7 trains without GPU acceleration,
defeating the primary reason for GCP migration.

**Fix for Run 8:** Always include a GPU accelerator in the instance create command.
The correct flag is `--accelerator` with `--maintenance-policy=TERMINATE`
(required for GPU instances — MIGRATE is incompatible).

**Verified available in us-central1-a:** nvidia-tesla-t4, nvidia-tesla-t4-vws
Use `nvidia-tesla-t4` (not vws) for training workloads.

**Run 8 instance create command:**
gcloud compute instances create genomic-run8 --zone=us-central1-a --machine-type=n1-standard-8 --accelerator=type=nvidia-tesla-t4,count=1 --maintenance-policy=TERMINATE --restart-on-failure --boot-disk-size=200GB --boot-disk-type=pd-ssd --image-project=deeplearning-platform-release --image-family=pytorch-2-7-cu128-ubuntu-2404-nvidia-570 --scopes=cloud-platform --metadata="install-nvidia-driver=True" --service-account=genomic-classifier-sa@genomic-variant-prod.iam.gserviceaccount.com

**GPU options by cost/performance (us-central1-a):**
- T4 (16GB VRAM): ~$0.35/hr — sufficient for GNN + CatBoost, good value
- L4 (24GB VRAM): ~$0.70/hr — faster, fits larger batches
- A100 40GB: ~$2.50/hr — overkill for current ensemble, useful for ESM-2 t12 upgrade

**T4 is the right choice for Run 8.** It will accelerate GNN training and
ESM-2 inference significantly. n1-standard-8 + T4 costs less per hour than
n2-highmem-32 and is faster for GPU workloads.

**Note:** `--metadata="install-nvidia-driver=True"` is required on Deep Learning
VM images to trigger automatic NVIDIA driver installation on first boot.

---

## Run 7 — Completed 2026-04-08

**Holdout AUROC: 0.9862** | 78 features | CPU-only (n2-highmem-32, no GPU)

### Results
- Holdout AUROC: 0.9862 | AUPRC: 0.9460 | F1: 0.9224 | MCC: 0.8478
- Train: 1,197,216 | Val: 154,404 | Test: 349,067 | Time: 198s
- Models saved to: gs://genomic-variant-prod-outputs/run7/models/v1/

### Top 10 Features (vs Run 6)
1. n_pathogenic_in_gene  568.3  (unchanged #1)
2. loeuf                 418.2  (gnomAD constraint — new high impact)
3. syn_z                 370.5  (gnomAD constraint — new high impact)
4. mis_z                 352.4  (gnomAD constraint — new high impact)
5. consequence_severity  242.7
6. pli_score             218.4  (gnomAD constraint)
7. alphamissense_score   189.7  (confirmed meaningful contribution)
8. af_raw                173.5
9. af_log10              105.5
10. len_diff              86.5

gnomAD v4.1 constraint features now occupy 4 of the top 6 positions.

### Why AUROC Unchanged from Run 6 (0.9862)
- GNN still did not contribute — CPU-only VM, no GPU acceleration
- Same 78 features, same data, same ensemble weights expected

### Infrastructure Issues (documented for Run 8)
- n2-highmem-32 is CPU-only — GPU must be explicitly attached at create time
- venv torch install fails with libcusparseLt.so.0 missing on this image
- Fix: remove pip-installed torch from venv, bridge to system torch via .pth
- Startup script git pull fails as root — requires git safe.directory fix first

### Run 8 Priorities
1. Use T4 GPU instance (command in Infrastructure section above)
2. GNN training will activate with GPU — expect AUROC improvement
3. Add git safe.directory fix to startup script before git pull
4. Add .pth bridge to startup script to avoid torch/venv conflict
5. Budget: at 50% monthly GCP budget after Run 7 — stop VM immediately after training

---

## Standing Infrastructure Rules (Corrected 2026-04-08)

Non-negotiable. Applies to every future run without exception.

### Rule 1: GPU Required
- Always use a GPU-enabled VM. T4 minimum.
- Never train on CPU regardless of circumstances.
- If T4 unavailable in us-central1-a, try us-central1-b, us-east1-b, us-west1-b.
- Never create an instance without --accelerator.
- Verify GPU before training: python3 -c "import torch; assert torch.cuda.is_available()"
- If cuda: False — stop immediately, do not train.

### Rule 2: VM Shuts Down After Training Stops for ANY Reason
- The VM must stop billing whenever the training process exits,
  whether by success, failure, crash, OOM, or any other cause.
- Implement via bash trap, not && chaining.
- && chaining only shuts down on success — this is insufficient.
- Correct pattern for every startup script and tmux launch:

  trap 'gsutil -m cp -r models/v1/ gs://genomic-variant-prod-outputs/runN/models/ 2>/dev/null; sudo shutdown -h now' EXIT
  python scripts/run_phase2_eval.py [args] 2>&1 | tee logs/training.log

- The trap fires on ANY exit from the script, uploading whatever
  models exist at that point, then shutting down immediately.

### Rule 3: Models Uploaded Before Shutdown
- The trap always attempts model upload before shutdown.
- Upload failure does not block shutdown — billing stops regardless.
- Verify upload succeeded after VM stops:
  gcloud storage ls gs://genomic-variant-prod-outputs/runN/models/

---

## Session Lessons — 2026-04-08 (Runs 6 & 7)

These are standing lessons that apply to ALL future runs and ALL projects.
They are captured here because they were learned through real failures and
real money spent. They must be consulted before every GCP training run.

### GCP Infrastructure
- NEVER fall back to CPU. If GPU zone is exhausted, try other zones/regions.
  Zones to try in order: us-central1-a, us-central1-b, us-east1-b, us-west1-b.
  Never create an instance without --accelerator.
- ALWAYS attach startup script at create time via --metadata-from-file.
  add-metadata on a running VM does NOT re-execute the startup script.
- ALWAYS verify cuda: True before training starts.
  If cuda: False — abort immediately, do not train.
- VM must shut down on ANY exit (success, failure, crash, OOM).
  Use bash trap, not && chaining. && only fires on success.
  Pattern: trap 'upload && shutdown' EXIT
- ALWAYS disable parallel composite uploads before large GCS transfers:
  gcloud config set storage/parallel_composite_upload_enabled False
  Parallel uploads cause 401 auth failures on large files when token expires.
- ALWAYS use gcloud storage CLI, never gsutil for new operations.
  gsutil does not read project from gcloud config set project.
- gcloud storage cp -r adds an extra directory nesting level when destination
  exists. Use individual file copies for critical paths.
- ALWAYS verify models are in GCS before stopping or deleting a VM.
  gcloud storage ls gs://genomic-variant-prod-outputs/runN/models/
- NEVER delete a VM before confirming model upload. Run 6 models were lost
  this way.
- Correct data bucket: gs://genomic-variant-prod-outputs/
  Not gs://genomic-variant-prod-data/ (does not exist).

### Python Environment on Deep Learning VMs
- System PyTorch lives at /usr/local/lib/python3.12/dist-packages/torch.
  The venv does NOT have torch by default.
- Installing torch into the venv fails with libcusparseLt.so.0 missing.
  Fix: remove pip-installed torch from venv, bridge via .pth file:
    echo "/usr/local/lib/python3.12/dist-packages" >> ~/venv/lib/python3.12/site-packages/system.pth
    echo "/usr/lib/python3/dist-packages" >> ~/venv/lib/python3.12/site-packages/system.pth
- torch-geometric must be installed system-wide:
    sudo pip3 install torch-geometric torch-scatter torch-sparse \
      -f https://data.pyg.org/whl/torch-2.7.0+cu128.html \
      --break-system-packages --quiet
- sudo pip3 install requires --break-system-packages on Ubuntu 24.04.
- Do NOT open tmux and immediately run a large pip install — segfaults.
  Open tmux first, then run commands inside it.

### Startup Script
- git pull as root fails with "dubious ownership" when repo was cloned as
  another user. Fix before any git commands:
    git config --global --add safe.directory /path/to/repo
- set -euo pipefail causes silent exit on any error. Wrap risky commands
  in || true or fix root cause before set -e line.
- Always set PYTHONPATH explicitly in startup script and launch commands.

### Training
- GNN contributed zero in Runs 6 and 7 — no GPU available.
  Run 8 is the first run where GNN will actually train.
- gnomAD v4.1 constraint features (loeuf, syn_z, mis_z, pli_score) are
  now positions 2-5 in feature importance. Highest-impact feature addition
  since n_pathogenic_in_gene.
- ESM-2 has run in stub mode (all-zero esm2_delta_norm) in all runs.
  Must verify transformers is installed before training.
- spliceai_index.parquet in GCS is a 29GB raw VCF dump, not a proper index.
  Omitted from Runs 6 and 7 with no AUROC impact. Needs regeneration.
- KAN reinstated for Run 9 (2026-04-20). Original 17.9 GB C++ OOM at
  >100K samples was fixed in commit 2389ee2 (2026-04-04) via a 100K-
  sample stratified subsample gate in `KANClassifier._fit_pykan`
  capping peak RAM at ~0.3 GB. The hardcoded `pop("kan", None)` in
  `scripts/run_phase2_eval.py` remained through Runs 6-8 as belt-and-
  braces caution and was removed in commit 8f9eb60. KAN is now in
  the ensemble by default; `--skip-kan` opts out. See commits b1c1150,
  8f9eb60, 128331f and docs/sessions/SESSION_2026-04-20.md.
  LiteratureScoutAgent continues to monitor pykan releases for future
  improvements to the subsample gate or threshold.
- n_pathogenic_in_gene is #1 feature by large margin in every run.
  Must always be present in VariantRequest.

---

## GPU Quota — Status 2026-04-08

GPU quota request (GPUS_ALL_REGIONS = 1) was denied on 2026-04-08.
Root cause: new account with insufficient billing history.

Action: reapply on or after 2026-04-15 with detailed justification.
Justification to use:
  "Production genomic variant pathogenicity classification research.
   Active billing history from CPU runs on 2026-04-08 (genomic-variant-prod).
   Requesting 1 GPU for ML ensemble training (CatBoost, LightGBM, XGBoost,
   PyTorch Geometric GNN). CPU runs take 3-4 hrs; GPU target <1 hr per run."

Run 8 is fully staged and ready — single command to launch once quota approved.
create_run8.cmd contains the L4 instance create command (g2-standard-8).

---

## Phase 3 Milestone: Polars Integration — Approved 2026-04-09

Benchmark result (500K variants, 20K genes, 3-run average):
  Pandas merge: 403.4 ms
  Polars join:  123.1 ms
  Speedup:      3.3x

Decision: Polars integration approved. Exceeds 3x threshold.

Target operations (in priority order):
  1. gnomAD constraint join in real_data_prep.py
  2. ClinVar annotation merge
  3. Feature engineering fillna/astype chain in engineer_features()

Implementation notes:
  - Use pl.from_pandas() / .to_pandas() at ETL boundaries
  - Keep pandas for sklearn/model interfaces (no conversion overhead in hot path)
  - Benchmark script: scripts/benchmark_polars.py

---

## Spectral Path Regression — Candidate Ensemble Member (Run 9)

Paper: "Spectral Path Regression: Directional Chebyshev Harmonics for
Interpretable Tabular Learning" (Coombs, 2025)
Code: https://github.com/MiloCoombs2002/spectral-paths

Relevance: Cancer Drug Response benchmark (N=475, D=698) is directly
analogous to our ClinVar/gnomAD/CADD/AlphaMissense tabular feature set.
Result: R²=0.438 vs XGBoost 0.331 — 33% relative improvement in the
high-dimensional regime. Analytic sensitivity complements SHAP.

Implementation plan (Run 9 prep):
  1. Clone spectral-paths, audit code quality
  2. Write src/models/spectral_path_classifier.py — sklearn wrapper
     with predict_proba via Platt scaling on regression output
  3. Add "spectral_path" to base_estimators in run_phase2_eval.py
  4. Benchmark against 78-feature set on synthetic data first
  5. Evaluate AUROC delta vs 0.9862 Run 7/8 baseline

Key technical note: paper uses tanh((x-c)/s) robust preprocessing
before arccos — this must be preserved in the wrapper. Classification
extension is listed in paper's Further Work, so predict_proba requires
care around the calibration boundary.

---

## Phase 4 Foundation — Inter-Agent Message Bus (Completed 2026-04-09)

Implemented an OpenClaw-inspired agent-to-agent communication layer that allows
the four agents to coordinate semi-autonomously when an update in one affects
the behaviour or output of another. This was a planned Phase 4/5 deliverable.

### Architecture

A typed, persistent MessageBus backed by SharedState JSON (atomic-write, crash-safe).
Four canonical message subjects:

  DATA_UPDATED            DataFreshness  → TrainingLifecycle
  CHECKPOINT_READY        TrainingLifecycle → Interpretability
  FEATURE_INSTABILITY     Interpretability  → TrainingLifecycle
  FEATURE_CANDIDATE_ADDED LiteratureScout   → TrainingLifecycle

Messages requiring consequential downstream actions (DATA_UPDATED, CHECKPOINT_READY)
gate on REQUIRE_HUMAN_APPROVAL before the receiving agent acts. FEATURE_INSTABILITY
and FEATURE_CANDIDATE_ADDED are informational (no approval gate).

### Files added / modified

  agent_layer/message_bus.py              NEW — core bus (send/approve/reject/history)
  agent_layer/shared_state.py             MOD — agent_messages key + migration
  agent_layer/orchestrator.py             MOD — pre/post-run message logging + delegates
  agent_layer/run_agents.py               MOD — --inbox/--pending-msgs/--approve-msg/--reject-msg/--msg-history
  agent_layer/agents/base_agent.py        MOD — send_message/read_inbox/get_actionable inherited
  agent_layer/agents/data_freshness_agent.py      MOD — emits DATA_UPDATED on change
  agent_layer/agents/training_lifecycle_agent.py  MOD — receives all 3 inbound subjects; emits CHECKPOINT_READY
  agent_layer/agents/interpretability_agent.py    MOD — receives CHECKPOINT_READY; emits FEATURE_INSTABILITY
  agent_layer/agents/literature_scout_agent.py    MOD — emits FEATURE_CANDIDATE_ADDED per new candidate
  agent_layer/test_message_bus.py         NEW — 34-test suite, 34/34 passing on Python 3.14.3

### Verified live

Full pipeline (`--pipeline full --dry-run`) runs all 4 agents cleanly. DataFreshness
connects to real ClinVar FTP, gnomAD GCS, and AlphaMissense endpoints.

### Remaining for full Phase 4 activation

  - ewc_utils importable from agent_layer/ (copy or path addition)
  - feedparser installed (`pip install feedparser --trusted-host pypi.org`)
  - Real training checkpoint in models/checkpoints/ (from next Run 9 training run)

---

## SpliceAI Index — Completed 2026-04-09

Build stats:
  Input:    data/external/spliceai/spliceai_scores.masked.snv.hg38.vcf.gz (28.8 GB)
  Lines:    3,458,738,010 (full hg38 unmasked VCF — not just masked SNVs)
  Written:  45,549,300 variants (score >= 0.1)
  Size:     336.8 MB parquet (snappy compressed) / 28.7 GiB on GCS
  Runtime:  440 min (7h 20min)
  Chroms:   1-22, X, Y (complete)
  Score:    0.1 - 1.0
  GCS:      gs://genomic-variant-prod-outputs/run6/data/data/processed/spliceai_index.parquet

Note: File was misnamed "masked.snv" — it is the full genome-wide unmasked
VCF including indels. This is better: Run 8 will have splice site scoring
for all variant types across all chromosomes.
## Run 10 — LOVD silent-zero remediation (prerequisite) + originally-scoped expansion

**Prerequisite (LOVD silent-zero — see `INCIDENT_2026-05-02_lovd-silent-zero.md`):**

- **R10-A** — Grep `outputs/run9_ready/regen.log` for `"Score annotation 15/16 (LOVD)"`. The integer in that line distinguishes:
  - non-zero (~5,500): Cause 1 (downstream overwrite in `_engineer_features`/`_scale`)
  - zero: Cause 2 (upstream coordinate transformation by one of steps 1–14)

  Two-minute task. Result narrows the fix to one of two specific files.

- **R10-B** — Patch identified cause. Add unit test asserting `(df["lovd_variant_class"] > 0).sum() > 0` after the LOVD step on a 3-row LOVD × 5-row ClinVar fixture with 1 expected match. Pattern modeled on `tests/unit/test_spliceai_parquet_default.py` (commit 9ba3127) and `tests/unit/test_esm2_activation.py` (2026-04-17 session).

- **R10-C** — Re-regen splits on Vast.ai with LOVD live (no local retraining per standing rule #19). Post-condition assertion: roughly 4,500–5,500 of the 5,553 inner-join matches reach the train set, depending on gene-aware split distribution. If R10-C produces 0 again, R10-B is incomplete and R10-A may need to be re-checked under the new run's regen log.

**Original Run 10 work (deferred from 2026-05-02 session):**

- **R10-D** — Expand gene scope (Paths 1+2: LOVD raw downloads + gnomAD/UniProt per-gene query list). Manual browser only per LOVD admin emails of 2026-04-01. Discipline rules:
  - One gene per browser tab, opened by hand, saved as `data/external/lovd/raw/{GENE}_variants.tsv` (or `.txt` per existing convention).
  - Spaced over time — admin can see request inter-arrival times.
  - `?format=tab` view only. No `modified_since` looping. No `/shared/genes/{GENE}` or `/shared/view/{GENE}` (human interface). No scripted fetcher.
  - Prefer `https://databases.lovd.nl/shared/download/all/gene/{GENE}` if the curator has enabled it.

  Rebuild the merged parquet via `scripts/build_lovd_index.py` (the live merge script — not the dead `scripts/process_lovd.py`). Re-regen splits on Vast.ai a second time.

**Cleanup (post-Run-9, low priority):**

- Remove `scripts/process_lovd.py` (dead code; live merge is `scripts/build_lovd_index.py`).
- Remove orphaned `data/external/lovd/lovd_variants.parquet` (output of the dead script; not consumed by any active code path).
- Remove `diag_lovd_join.py` from repo root (created during 2026-05-02 session as a one-shot diagnostic; useful as R10-B verification then removable).
- Audit other connectors on the 30+ all-zero list from `SESSION_2026-04-30.md` Finding #4 for the same silent-zero pattern. Extends the 2026-04-17 audit recommendation (EVE, AlphaMissense, CADD).

