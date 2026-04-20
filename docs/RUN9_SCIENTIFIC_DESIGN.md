# Run 9 Scientific Design & Future-Run Operating Charter

**Status:** DRAFT — for Codex review
**Author:** planning session, 2026-04-19
**Applies to:** Run 9 and every subsequent training run
**Supersedes:** implicit "turn on SpliceAI + GNN, check AUROC" scope from prior sessions

---

## 0. Executive summary

Runs 6 → 7 → 8 returned AUROC 0.9862 / 0.9862 / 0.9863. That is not measurement
noise within a single run, it is three independent runs failing to distinguish
themselves — which means AUROC has **saturated against the current feature
set**, and additional GPU spend will not move it without a change of kind, not
degree. This session also confirmed three features (ESM-2, EVE, GNN-on-real-edges)
that everyone assumed were contributing are in fact silent zeros for plumbing
reasons, not modelling reasons. This is the first session in which that gap
was identified as a *measurement* problem rather than a *modelling* one.

The user has stated the correct strategic frame: every GPU run must be treated
as a scientific experiment that produces publishable-quality artefacts, not a
leaderboard score. This document formalises that frame, redesigns Run 9 around
it, and specifies the engineering prerequisites, ablations, artefact format,
and exit criteria for every run from Run 9 forward.

The total new engineering scope is:

1. Four new standing rules (§2) that apply to every run from 9 onward.
2. A Run 9 scope that produces eight distinct scientific outputs (§3), not one.
3. Four prerequisite modules to build before launch (§4):
   `scripts/run_phase2_eval.py` (ablation harness),
   `src/evaluation/prediction_artifacts.py` (OOF + SHAP + calibration dump),
   `src/data/splits.py` (unseen-gene holdout),
   preflight subprocess fix.
4. A post-run documentation contract (§6) that is checked in CI, not trusted.

Run 9 remains fundable (~$5–10, 4–8 h wall) but now returns a per-database
contribution decomposition, per-consequence calibration curves, unseen-gene
generalisation numbers, SHAP explanations for all clinically interesting cases
in the test set, and a ranked VUS prediction list. That is the minimum output
for any future run.

---

## 1. Current state assessment (what's actually true right now)

### 1.1 Repo / deployment state

| Item | State | Notes |
|---|---|---|
| HEAD | `50c0579` on `origin/main` | pushed, matches remote |
| Working tree | clean modulo two allowlisted carry-overs | `scripts/gcp_run6_startup.sh`, `ROADMAP_PSYCH_GWAS_ENTRY.md` |
| Tests | 466 passing, 3 skipped, 0 failed | from end of prior session |
| Local SpliceAI parquet | present, 336.8 MB | `data/external/spliceai/spliceai_index.parquet` |
| GCS SpliceAI parquet | present (confirmed via `gcloud storage ls` in prior session) | same path |
| SpliceAI test cache | deleted locally, regenerates on full pytest | fixture-scope gap |
| STRING DB cache | fixed in commit `0a02e5d` | but never actually exercised on GPU in Run 8 |
| ESM-2 connector | `src/data/esm2.py`, `transformers`-backed, scalar output `esm2_delta_norm` | **silent zero** in Runs 6/7/8 |
| EVE connector | `src/data/eve.py`, needs `wt_aa`/`mt_aa`/`position`/`mutations_protein_name` | **almost certainly silent zero**, unverified |
| PyTorch NN migration | complete, commit `38656bc` | 4 NNs off TensorFlow |
| Vast.ai | destroyed, ~$23.52 credit remaining | no live GPU |

### 1.2 Run 8 reality vs. assumption

Run 8's AUROC of 0.9863 was produced by **5 of 10 base models actually
contributing**:

| Feature / model | Assumed in Run 8 | Actually active |
|---|---|---|
| CatBoost / LightGBM / XGBoost tabular | yes | yes |
| Random Forest / GBM tabular | yes | yes |
| Tabular NN (PyTorch) | yes | yes (after `38656bc`) |
| 1D-CNN on sequence | yes | yes |
| GNN (GAT over STRING DB) | yes | **no** — zero edges in the data object (no GPU → PyG fell through) |
| ESM-2 delta-norm | yes | **no** — pipeline never populated `wt_aa`/`mut_aa`/`protein_pos` |
| EVE scores | yes | **no** (same plumbing gap, unverified) |
| SpliceAI `ds_max` | yes | **no** — default path bug, fixed `9ba3127`+`8b12f76`+`d01f2e1` |
| AlphaMissense | yes | **yes** — ranked 7th of 78 features |

So Run 8's result is effectively *the tabular AlphaMissense + dbNSFP +
gnomAD-AF + ClinVar-adjacent ensemble*. It is strong (0.9863) but does not
reflect the advertised architecture. This is material for the paper: we will
be able to say exactly what fraction of discrimination comes from each feature
class.

### 1.3 Known open engineering items

Four items must be resolved or consciously deferred before Run 9 launch:

1. **Preflight Windows subprocess bug.** `git` and `gcloud` both fail with
   "The system cannot find the path specified" even with `shutil.which`. The
   symptom and fix belong in `scripts/preflight_check.py`, not in any launch
   path. Deferred acceptable: VM preflight catches the real bugs.

2. **SpliceAI cache fixture-scope leak.** The `_isolate_spliceai` fixture is
   class-scoped, protecting only `TestAnnotationPipeline`. Other tests that
   import or touch the annotator rebuild the 430 MB cache. Needs module-scoped
   or session-scoped isolation with an `autouse=True` fixture that also
   guarantees cleanup.

3. **HGVSp parser missing.** Pipeline never emits `wt_aa`/`mut_aa`/`protein_pos`
   so ESM-2 (and almost certainly EVE) return a constant zero feature.
   Full incident doc: `docs/incidents/INCIDENT_2026-04-17_esm2-hgvsp-parser.md`.
   Out of scope for Run 9 — a correct HGVSp parser is 1–2 days and Run 10's
   headline item.

4. **`.gitignore` rule visibility.** `data/raw/` catches the cache via parent
   matching, but an explicit `data/raw/cache/` rule would be clearer and
   survives future refactors of the broader rule.

---

## 2. Standing rules addendum (applies to Run 9 and all future runs)

The existing charter has four standing rules (exhaustive preflight, versioned
roadmap maintenance, docs/logs filing system, technology phase triggers).
Adding four more, numbered 5–8 to extend the existing list:

### Rule 5 — Every run produces scientific artefacts, not just metrics

A run's success is defined by what *remains* after the VM is destroyed.
Specifically the following artefacts are REQUIRED in GCS for every run, no
exceptions:

- `runs/run{N}/oof_predictions.parquet` — one row per training example, one
  column per base model + ensemble, plus fold index and label. Enables
  post-hoc analysis of stacker weights, per-fold variance, and model
  correlation without re-training.
- `runs/run{N}/test_predictions.parquet` — one row per test example, columns
  for each base model's probability and the stacker's probability, plus all
  canonical metadata (variant_id, gene_symbol, consequence, acmg_label).
- `runs/run{N}/shap_values.parquet` — ensemble-level SHAP for the test set,
  at least top-20 features per variant. Wraps the trained stacker over the
  base-model probability matrix.
- `runs/run{N}/calibration.parquet` — per-consequence bucket: predicted
  probability bin, empirical fraction positive, n. Feeds calibration curves.
- `runs/run{N}/feature_importance.parquet` — permutation-based (gold
  standard), not built-in `feature_importances_` which is biased toward
  high-cardinality features.
- `runs/run{N}/ablation_results.parquet` — rows for each LOCO configuration
  (see Rule 6) with full metric tuple + bootstrap CI.
- `runs/run{N}/graph_stats.json` — for any run that touches the GNN: node
  count, edge count, degree distribution percentiles, connected components.
  Non-negotiable; "did the GNN see edges" is the single most important
  question for every GPU run.
- `runs/run{N}/manifest.json` — git SHA, config snapshot, random seeds,
  transformers/torch/PyG/CUDA versions, exact feature-column names in the
  order used by each base model, hash of the training parquet.

A run that fails to upload the manifest is considered a failed run regardless
of metric, because it is not replicable and not analysable.

### Rule 6 — Every run answers multiple hypotheses, not one

Every run includes at least three ablation configurations beyond "full model":

- Leave-One-Class-Out (LOCO): disable one feature *class* (e.g. SpliceAI,
  GNN, ESM-2, AlphaMissense, conservation, allele-freq) and retrain. Quantifies
  unique contribution per class.
- Unseen-gene holdout: 20 % of genes held out from training entirely
  (see §4.3). Tests biological generalisation vs. memorisation.
- Stratified sensitivity-at-operating-point breakdown: report metrics at
  Sens ≥ 0.90, Sens ≥ 0.95, PPV ≥ 0.80, PPV ≥ 0.95. Clinicians do not care
  about AUROC; they care about PPV at a specific threshold.

The existing `ClinicalEvaluator` already returns the operating points; what's
missing is the LOCO harness and the unseen-gene split. Both specified in §4.

### Rule 7 — Preserve raw predictions, not just summaries

Aggregated metrics lose information. A `runs/run{N}/test_predictions.parquet`
enables every one of the following to be answered *without* re-running GPU:

- Which specific BRCA1 variants does the model disagree with ClinVar on?
- Is the stacker over-confident on synonymous variants?
- Which pairs of base models are most correlated in error? (→ ensemble
  redundancy detection)
- Does the calibration hold within each consequence class?
- How does performance stratify by MAF bucket?

Saving a 100k-row × 15-col parquet is cheap (~5 MB). Re-running Run 8 to
answer these questions later is $5 and 6 h. Predictions must be preserved.

### Rule 8 — Every run contributes to a publishable narrative

The project has at least four plausible papers in it:

- **P1:** Empirical decomposition of clinical variant predictor contributions.
  Which of the ~20 canonical resources (ClinVar, gnomAD, CADD, SpliceAI,
  AlphaMissense, ESM-2, REVEL, PhyloP, STRING DB, GTEx, etc.) provides
  *unique* discriminative signal, at what cost per unit signal, and at what
  calibration cost? This paper writes itself from Rule 6 ablations.
- **P2:** Five-tier ACMG/AMP ML classifier with calibrated probabilities.
  Existing tools predict binary; VUS is where clinicians actually struggle.
  A well-calibrated five-class predictor is genuinely novel.
- **P3:** Gene-stratified generalisation of pathogenicity predictors. Can
  these models predict in genes not seen in training? Does transfer work
  within gene families?
- **P4:** STRING DB graph attention for variant pathogenicity. First real
  GPU run with real edges makes this tractable; the GAT's attention weights
  over neighbour genes are interpretable and novel.

Every run from 9 forward should be scoped to "this is the run that produces
figure/table X of paper Y" — not to an AUROC target.

---

## 3. Run 9 scope — eight scientific outputs

Under the new charter, Run 9's purpose is to produce the **baseline empirical
decomposition** that later runs will extend. It is the first run with both
SpliceAI and real-graph GNN live, so it is uniquely positioned to answer
"what does each signal class buy us?" on a single consistent split.

### 3.1 Core training run

Full ensemble with SpliceAI ON, GNN ON (on GPU with real STRING edges), all
eight base models plus stacker. ESM-2 and EVE remain stub for this run; they
will be listed in the manifest as `esm2_delta_norm=stub, eve_score=stub` and
excluded from the permutation-importance computation to avoid noise.

Split: gene-stratified `GroupShuffleSplit` (60/20/20 train/val/test) — same
seed as Run 8 for direct comparability. The test set is fixed across all
ablations in the run so numbers are directly comparable.

### 3.2 LOCO ablations (Rule 6)

Five leave-one-out configurations, all trained on the same training split,
all evaluated on the same test split:

| Configuration | Features disabled | Hypothesis |
|---|---|---|
| `full` | none | baseline |
| `no_spliceai` | `splice_ai_*` columns zeroed | H1: SpliceAI adds unique splice signal |
| `no_gnn` | `gnn_embed_*` columns dropped / GNN head disabled | H2: Network context adds unique signal |
| `no_alphamissense` | `alphamissense_*` zeroed | H3: AlphaMissense is the dominant current contributor (expected yes from Run 8 feature importance) |
| `no_conservation` | `phylop_score`, `phastcons_*`, dbNSFP conservation stack zeroed | H4: Conservation-only baseline is still strong |
| `no_population_af` | `allele_freq`, `gnomad_*` zeroed | H5: AF is a label-leaking feature (pathogenic variants are rare) |

Six runs × 8 base models × 5 folds is expensive on CPU but most features are
tabular; the ablations are fast (~15–30 min each on a 24-core box, only the
GNN+CNN require GPU). All six fit within the Run 9 window.

### 3.3 Unseen-gene holdout

Hold out 20 % of *genes* entirely (not variants in those genes). Train on
remaining 80 %, evaluate on held-out genes. Answers: does the model learn
biology or gene-identity? This is the cleanest generalisation test and
clinicians will ask for it.

### 3.4 Per-consequence calibration curves

ClinicalEvaluator already does per-consequence breakdown. Extend to per-bucket
calibration curves: for each of {missense, nonsense/LoF, splice, synonymous,
inframe indel, frameshift}, save a calibration curve table. A miscalibrated
model even at high AUROC is clinically dangerous.

### 3.5 Operating-point table

At Sens ≥ 0.90, Sens ≥ 0.95, PPV ≥ 0.80, PPV ≥ 0.95 compute: threshold,
n_flagged, TP, FP, TN, FN, calibration error in that slice. This is the
actual clinical-decision table.

### 3.6 SHAP decomposition on test set

Use `shap.LinearExplainer` (the stacker is a `LogisticRegression` over base
probabilities — SHAP is closed-form and cheap). Save top-20 mean |SHAP| per
variant plus the global ranking. Do this over the base-model probability
matrix — the stacker's coefficients are the direct per-model contribution,
and SHAP gives per-variant attributions.

### 3.7 Graph statistics

For the GAT's PyG dataset: node count, edge count, median and 99th-percentile
node degree, number of connected components, fraction of test variants whose
gene is a hub (degree > 100) vs. leaf (degree < 5). If the edge count is 0
(Run 8's regression), the run is flagged as GNN-inactive in its manifest and
the GNN columns are excluded from ablation 3.2.

### 3.8 Ranked VUS prediction list

Apply the trained stacker to all ClinVar variants currently labelled
`Uncertain significance` or `Conflicting interpretations` within the gene set
used in training. Rank by predicted pathogenicity. This produces a clinically
*usable* output from the run — specific VUS variants where the model has a
high-confidence prediction, which clinical collaborators can take to a
functional assay or a follow-up literature review.

This is the single deliverable that moves the project from "ML benchmark" to
"produces novel bioinformatics output", and it requires zero additional
training — it reuses the same trained stacker from §3.1.

---

## 4. Pre-launch engineering work

Four modules/scripts to build. Each is scoped tightly: not every Phase-2
feature has to land before Run 9, only the ones that unblock Rule 5 artefacts
and Rule 6 ablations.

### 4.1 `scripts/run_phase2_eval.py` — the ablation harness

Thin wrapper over existing training pipeline that accepts `--ablation` and
drives LOCO:

```python
# Signature (abridged):
parser.add_argument("--ablation", default="full",
    choices=["full", "no_spliceai", "no_gnn", "no_alphamissense",
             "no_conservation", "no_population_af"])
parser.add_argument("--run-id", required=True)     # e.g. "run9"
parser.add_argument("--holdout", default="variant",
    choices=["variant", "unseen_gene"])            # Rule 6, clause 2
parser.add_argument("--output-dir", default="runs/")
```

Starter code in `scripts/run_phase2_eval.py` (delivered alongside this doc).
The ablation is implemented by a **feature mask** applied to the engineered
DataFrame before model fitting — we do *not* retrain feature extraction.
Zeroing out the column is the cleanest ablation for tree models (the column
becomes constant so the split-info is 0); for the GNN ablation, we drop the
GNN column entirely because the stacker should not see it at all.

### 4.2 `src/evaluation/prediction_artifacts.py` — the saver

A single class `RunArtifactWriter` that the training script instantiates
once, hands to the ensemble, and calls at every natural checkpoint:

```python
writer = RunArtifactWriter(run_id="run9", output_dir="runs/", gcs_bucket=GCS_BUCKET)
writer.save_manifest(config=cfg, git_sha=GIT_SHA, versions=VERSIONS)
writer.save_oof_predictions(oof_df)                          # per-fold
writer.save_test_predictions(test_df, y_test, meta_test)     # after final fit
writer.save_shap_values(shap_df)                             # per-variant top-K
writer.save_calibration(per_consequence_calibration_df)
writer.save_feature_importance(permutation_importance_df)
writer.save_graph_stats(gnn_graph_stats_dict)                # JSON
writer.upload_to_gcs()                                       # atomic via .tmp suffix
```

Design rules: every write is local-first, then a single final
`upload_to_gcs()` call at shutdown so the VM can be destroyed the instant
the upload completes. Uses `gcloud storage cp --recursive` (not the
deprecated `gsutil` per the standing rule).

Starter code delivered alongside this doc.

### 4.3 `src/data/splits.py` — the splitters

Single module with two functions:

```python
def gene_stratified_split(df, test_frac=0.2, val_frac=0.2,
                          seed=42) -> tuple[Idx, Idx, Idx]:
    """Existing Run 8 behaviour — GroupShuffleSplit by gene_symbol."""

def unseen_gene_holdout_split(df, holdout_frac=0.2, seed=42) -> tuple[Idx, Idx]:
    """Holds out entire genes. Returns (train_idx, unseen_gene_test_idx).
    Genes selected by hashing gene_symbol so split is deterministic and
    stable across runs — the same genes are always unseen, enabling
    longitudinal analysis."""
```

The hash-based gene selection is important: it means "unseen gene" is a
stable concept across runs, so comparing Run 9's unseen-gene numbers against
Run 15's is meaningful.

### 4.4 Preflight Windows subprocess fix

Targeted fix, not a rewrite. Replace the `shutil.which` + `shell=True`
combination with Python `subprocess.run()` using the absolute path returned
by `shutil.which`, with `cwd` and `env` passed explicitly and *no* `shell`
kwarg:

```python
def _run_exe(cmd: list[str], cwd: Path, env: dict = None) -> CompletedProcess:
    exe = shutil.which(cmd[0])
    if not exe:
        return CompletedProcess(args=cmd, returncode=127,
                                stdout="", stderr=f"{cmd[0]} not on PATH")
    return subprocess.run([exe, *cmd[1:]], cwd=str(cwd), env=env,
                          capture_output=True, text=True, shell=False)
```

Why this works and the prior shell=True didn't: `shell=True` with a list
argv on Windows triggers an argv-reassembly path inside CPython that mangles
backslash-laden cwd; absolute-exe with no shell bypasses it entirely.

This is a ~20-line change, not a new module.

---

## 5. Ordered launch sequence (the operational checklist)

Assuming §4 work is landed, committed, pushed, and tagged. Do not launch
before every item in each phase reports green.

### Phase A — local verification (no GPU cost)

```powershell
cd C:\Projects\genomic-variant-classifier
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "C:\Projects\genomic-variant-classifier"

# A1. Ensure HEAD matches expected Run 9 commit
git rev-parse HEAD
git rev-parse origin/main    # must match

# A2. Tests green
Get-ChildItem -Recurse -Filter "__pycache__" -Directory | Remove-Item -Recurse -Force
python -m pytest -q --tb=short
# Expected: 466+ passing, 0 failed

# A3. Ablation harness smoke test (1% sample, single ablation)
python scripts\run_phase2_eval.py --ablation full --run-id smoke `
    --sample-frac 0.01 --skip-gpu --output-dir runs\smoke
# Expected: completes in <2 min, writes manifest.json + oof_predictions.parquet

# A4. Artifact writer unit tests
python -m pytest tests\unit\test_prediction_artifacts.py -v

# A5. Splits unit tests (especially the hash-stability property)
python -m pytest tests\unit\test_splits.py -v

# A6. Preflight, with --skip-pytest because we just ran it
python scripts\preflight_check.py --skip-pytest
echo "preflight exit: $LASTEXITCODE"
# Expected: exit 0 or only FAIL is "GCS auth" (refresh with gcloud auth login)
```

Do not proceed to Phase B until A1–A6 are green.

### Phase B — Vast.ai provisioning

```bash
# B1. Provision instance. Minimum spec:
#       - 1× RTX 4090 (24 GB VRAM)
#       - 60 GB system RAM
#       - 200 GB disk
#       - CUDA 12.x image with PyTorch preinstalled (vast.ai "PyTorch" stack)
# Copy the SSH command printed by Vast.ai.

# B2. SSH in and immediately set up trap-EXIT shutdown:
ssh -i ~/.ssh/vastai vastuser@<host>
cat >/home/ubuntu/shutdown_on_exit.sh <<'EOF'
#!/bin/bash
set +e
cd /home/ubuntu/genomic-variant-classifier
# Upload any artifacts present, even partial ones
if [ -d runs/run9 ]; then
    gcloud storage cp --recursive runs/run9 gs://genomic-variant-prod-outputs/runs/ || true
fi
sudo shutdown -h now
EOF
chmod +x /home/ubuntu/shutdown_on_exit.sh
trap '/home/ubuntu/shutdown_on_exit.sh' EXIT INT TERM

# B3. Clone repo, install deps, pull data
git clone https://github.com/monzia-moodie-repo-projects/genomic-variant-classifier.git
cd genomic-variant-classifier
git checkout <run9-tag>
python -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
pip install "transformers>=4.40,<5.0"    # pinned per INCIDENT 2026-04-17
pip install "torch-geometric>=2.5"

# B4. Pull input data from GCS
mkdir -p data/external/spliceai data/raw
gcloud storage cp gs://genomic-variant-prod-outputs/spliceai_index.parquet \
    data/external/spliceai/spliceai_index.parquet
gcloud storage cp -r gs://genomic-variant-prod-outputs/string_db/ data/raw/cache/

# B5. VM preflight (catches path mismatches before training starts)
bash scripts/preflight_vm.sh
# Expected: exit 0; all critical imports succeed; GPU visible
```

### Phase C — the run itself

```bash
# C1. Full Run 9 — sequential ablations into a single output directory
for ABL in full no_spliceai no_gnn no_alphamissense no_conservation no_population_af ; do
    python scripts/run_phase2_eval.py \
        --ablation "$ABL" \
        --run-id run9 \
        --holdout variant \
        --output-dir runs/ \
        2>&1 | tee "logs/run9_${ABL}.log"
done

# C2. Unseen-gene holdout with full feature set (single extra run)
python scripts/run_phase2_eval.py \
    --ablation full \
    --run-id run9 \
    --holdout unseen_gene \
    --output-dir runs/ \
    2>&1 | tee logs/run9_unseen_gene.log

# C3. VUS prediction pass (reuses trained full stacker, zero GPU needed)
python scripts/predict_vus.py \
    --model-dir runs/run9/full \
    --output runs/run9/vus_predictions.parquet \
    --min-rank 1000   # top 1000 VUS by predicted pathogenicity

# C4. Upload and shutdown
gcloud storage cp --recursive runs/run9 gs://genomic-variant-prod-outputs/runs/
# trap EXIT handles shutdown; or explicitly:
sudo shutdown -h now
```

### Phase D — post-run documentation (Rule 8, non-negotiable)

Before closing the session:

```powershell
# D1. Pull artefacts down
gcloud storage cp --recursive gs://genomic-variant-prod-outputs/runs/run9 runs/

# D2. Auto-generate the session doc from the artefacts
python scripts/generate_run_report.py --run-id run9 \
    --output docs/sessions/SESSION_$(Get-Date -Format yyyy-MM-dd)_run9.md

# D3. Human-written section: what we learned, what surprised us, what's
#     the next run. See §6 for the template.

# D4. Commit
git add docs/sessions/SESSION_*_run9.md docs/CHANGELOG.md docs/ROADMAP.md
git commit -m "docs(run9): session report + CHANGELOG + roadmap update"
git push
```

---

## 6. Post-run documentation contract

Every run produces `docs/sessions/SESSION_<date>_run<N>.md` with these
mandatory sections, in this order. Missing sections = documentation
regression = Rule 8 violation, flagged in CI.

1. **Run metadata** — git SHA, date, wall time, GPU cost, instance spec.
2. **Baseline comparison table** — metric | prior-best | this-run | delta | 95%-CI-overlap.
3. **Per-base-model OOF AUROC** — all 10 (or however many are active), with
   a column noting whether each contributed non-trivially (|coef| > 0.05 in
   stacker).
4. **LOCO ablation table** — six rows (full + five ablations), metric tuple,
   ΔAUROC vs. full, bootstrap CI on the delta.
5. **Unseen-gene holdout metrics** — AUROC, AUPRC, MCC, calibration ECE.
   A large gap vs. the random-variant split indicates memorisation.
6. **Per-consequence breakdown** — the ClinicalEvaluator table.
7. **Top-20 features by permutation importance** — plus a note on whether
   the top-5 matches Run 8's.
8. **Graph stats** — if GNN live, node/edge counts and degree distribution.
9. **Surprises / deviations from plan** — anything that did not match
   pre-launch predictions. This is the most valuable section; future runs'
   designs draw from it.
10. **Clinical deliverable** — for Run 9, a link to `vus_predictions.parquet`
    and a short commentary on the top-5 highest-confidence novel calls.
11. **Next-run hypothesis** — what does the next run test? Must be specific
    and falsifiable.

Sections 1–8 are auto-generated from artefacts by `scripts/generate_run_report.py`.
Sections 9–11 are human-written.

---

## 7. Prioritised pre-Run-9 task list (for Codex review)

Ordered by dependency; each is an atomic pull request sized for one review
cycle. Estimated effort in parentheses.

1. **T1** (1 h): Fix preflight Windows subprocess — `_run_exe` with
   absolute-path `shutil.which` and no shell. `scripts/preflight_check.py`
   only. Test: local preflight completes without crashing on a clean
   Windows checkout.
2. **T2** (1 h): Add explicit `data/raw/cache/` line to `.gitignore`. Audit
   all tests that touch SpliceAI connector; widen `_isolate_spliceai`
   fixture scope from class to module. Test: `pytest` run does not
   regenerate `spliceai_scores_snv.parquet`.
3. **T3** (3–4 h): `src/data/splits.py` with `gene_stratified_split` and
   `unseen_gene_holdout_split`. Hash-stable gene selection. Unit tests
   covering: no gene leak between sets, holdout stability across seeds,
   fraction constraint within 1 %.
4. **T4** (4–6 h): `src/evaluation/prediction_artifacts.py`
   (`RunArtifactWriter`). Unit tests covering: atomic-write semantics,
   manifest schema, re-run idempotency (same run id → same output).
5. **T5** (6–8 h): `scripts/run_phase2_eval.py` ablation harness. Uses T3
   splits and T4 writer. CLI smoke test on 1 % sample must complete in
   <2 min. Full integration test on 10 % sample completing in <15 min on
   CPU gates Run 9 launch.
6. **T6** (2–3 h): `scripts/predict_vus.py` — stateless script, no training,
   reuses a saved stacker. Input: path to `runs/run9/full`. Output:
   ranked VUS parquet.
7. **T7** (3 h): `scripts/generate_run_report.py` — reads the artefact set,
   emits the auto-generated sections of the SESSION doc. CI check: every
   session doc that mentions a run ID has a corresponding artefact
   directory in GCS.
8. **T8** (1 h): Tag `run9-ready` on the commit that lands T1–T7. Launch
   pre-flight gates on this tag, not on `main`, to decouple launch from
   unrelated mainline work.

Total: 21–28 hours of engineering before Run 9 launches. Call it 3–4
focused sessions.

---

## 8. Exit criteria — Run 9 is "successful" if and only if

A run is declared successful when ALL of the following are true, not when
AUROC improves. Codex should reject any post-run doc that claims success
without all of these checked.

- [ ] All seven GCS artefacts from Rule 5 present and parseable.
- [ ] `graph_stats.json` shows `edge_count > 100000` (confirms GNN saw real
      STRING edges — Run 8 had 0).
- [ ] `ablation_results.parquet` has exactly six rows matching §3.2.
- [ ] Unseen-gene AUROC is within 0.03 of variant-holdout AUROC, OR the gap
      is explicitly called out as the run's primary finding (a large gap is
      scientifically interesting, not a failure).
- [ ] SHAP top-20 is computed and saved.
- [ ] `vus_predictions.parquet` has ≥ 1000 ranked VUS variants.
- [ ] `docs/sessions/SESSION_<date>_run9.md` has all 11 sections.
- [ ] Two-sentence summary of findings is pushed to `docs/ROADMAP.md` with
      a specific hypothesis for Run 10.

A run that hits 0.99 AUROC but fails to upload `vus_predictions.parquet` is
*not* successful under this charter. A run that stays at 0.986 AUROC but
produces a clean six-row ablation decomposition and 1000 ranked VUS *is*
successful.

---

## 9. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Ablation harness bug changes the test set between ablations | low | high | §3.1 requires fixed seed + split across ablations; add an assertion that the test-set hash is identical in every ablation's manifest |
| GNN edge count is still 0 (Run 8 regression repeats) | medium | high | VM preflight check (already exists in `scripts/preflight_vm.sh`) must be extended to assert `edge_count > 100000` before the run starts, not after |
| Permutation importance is prohibitively slow on ~1.2M samples | medium | medium | Sample down to 50k for permutation importance only (other artefacts use full data); documented in manifest |
| SpliceAI annotation is the bottleneck | low | medium | Already a 45.5M-row parquet lookup, <10 min on our hardware; skip if run time exceeds 2 h for annotation alone |
| HGVSp parser gap limits ESM-2/EVE contribution for Run 10 as well | high | medium | INCIDENT doc is the tracking ticket; Run 10 is explicitly "HGVSp parser run" |
| Python 3.14 / scientific stack incompatibility discovered at VM launch | low | high | Pin versions in `requirements-vm.txt`; VM preflight imports every critical module before training begins |

---

## 10. Glossary of files created / modified by this plan

| Path | Status | Purpose |
|---|---|---|
| `docs/RUN9_SCIENTIFIC_DESIGN.md` | **new** (this file) | strategic charter |
| `scripts/run_phase2_eval.py` | **new** | ablation harness |
| `scripts/predict_vus.py` | **new** | VUS ranking, no-GPU |
| `scripts/generate_run_report.py` | **new** | auto-populate session doc |
| `src/evaluation/prediction_artifacts.py` | **new** | `RunArtifactWriter` |
| `src/data/splits.py` | **new** | gene-stratified + unseen-gene |
| `tests/unit/test_prediction_artifacts.py` | **new** | writer unit tests |
| `tests/unit/test_splits.py` | **new** | splitter unit tests |
| `tests/unit/test_run_phase2_eval.py` | **new** | ablation harness smoke tests |
| `scripts/preflight_check.py` | **modify** | Windows subprocess fix |
| `.gitignore` | **modify** | add explicit `data/raw/cache/` |
| `docs/CHANGELOG.md` | **append** | entry per commit |
| `docs/ROADMAP.md` | **modify** | Run 9 exit criteria + Run 10 scope |

---

_End of design. Hand to Codex for review; do not begin T1–T8 until review
notes have been addressed._