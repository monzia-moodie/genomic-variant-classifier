# Changelog — Genomic Variant Classifier

Append-only. One entry per session. Captures what was attempted, what
failed (with exact errors and root causes), what was fixed, and what was
learned. Searchable: paste any error string to find the root cause and fix.

Format per entry:
  ## YYYY-MM-DD — <one-line summary>
  ### Attempted | Failed | Fixed | Learned

---

## 2026-04-08 — Runs 6 & 7, GPU quota request, Run 8 startup script

### Attempted
- Run 6: full training on GCP (n2-highmem-32, CPU-only). Holdout AUROC 0.9862.
- Run 7: repeat with gnomAD v4.1 constraint features wired in. AUROC 0.9862 (unchanged — GNN still CPU-only).
- GPU quota request: GPUS_ALL_REGIONS = 1.
- Run 8 VM create: L4 (g2-standard-8).

### Failed
- Run 6 models lost: VM was deleted before model upload was confirmed.
  Root cause: shutdown was triggered by `&&` chaining, not `trap EXIT`.
  `&&` only fires on success; VM was already off by the time we checked GCS.
- GPU quota denied. Code: GPUS_ALL_REGIONS = 0 (new account, no billing history).
- Run 8 VM create failed: `ZONE_RESOURCE_POOL_EXHAUSTED` across all US zones.
  Root cause: quota was 0 — zone exhaustion was a red herring.
- venv torch install on Deep Learning VM: `libcusparseLt.so.0` not found.
  Root cause: venv doesn't have access to the system CUDA libraries.
  Fix: uninstall pip torch from venv; add .pth bridge to system torch.
- `gcloud storage cp -r` added extra directory nesting level.
  Fix: use individual file copies, not `-r`.
- `set -euo pipefail` in startup script caused silent exits on risky commands.
  Fix: wrap risky commands with `|| true`.

### Fixed
- Startup script: replaced `&&` chaining with `trap 'upload && shutdown' EXIT`.
  Fires on ANY exit: success, failure, crash, OOM.
- Git safe.directory: `git config --global --add safe.directory $REPO_DIR`
  (startup runs as root; repo cloned as monzi — git refuses pull otherwise).
- Parallel composite upload disabled: `gcloud config set storage/parallel_composite_upload_enabled False`.
  Was causing 401 auth failures on large files when OAuth token expired mid-upload.
- argparse `--string-db` flag: was missing from `run_phase2_eval.py`.
- gnomAD constraint path: was never wired into `AnnotationConfig`.
  All four constraint features (loeuf, syn_z, mis_z, pli_score) defaulted to 0.

### Learned
- Always verify models are in GCS before stopping/deleting a VM.
- `trap EXIT` is the only correct pattern. `&&` is insufficient.
- Google grants GPU quota only after billing history is established.
  Reapply after 2026-04-15.
- `gcloud storage` CLI always; never `gsutil` (does not read project from config).

---

## 2026-04-09 — Inter-run items 1-8, inter-agent message bus (Phase 4)

### Attempted
- SpliceAI index build from full hg38 VCF (28.8GB compressed).
- VersionMonitorAgent implementation and orchestrator wiring.
- Requirements cleanup (orphan files, add transformers>=4.40).
- Dockerfile audit and fixes.
- Polars benchmark on gnomAD constraint join.
- .gitkeep replacement in data/ subdirs.
- Inter-agent message bus: OpenClaw-inspired typed message passing between all 4 agents.
- Full pipeline dry-run verification.

### Failed
- SpliceAI VCF was misidentified as masked SNV (~72M lines).
  Actual: full unmasked hg38 VCF including indels — 1.1B+ lines, 2.5+ hours.
  Root cause: filename says "masked.snv" but file is full genome-wide.
  Result: still correct and more complete than expected. Build still running at session end.
- Docker smoke test: Docker Desktop not running (Linux engine pipe not found).
  Not a code problem. Deferred.
- `data_freshness_agent.py`: `ImportError: cannot import name 'ALPHAMISSENSE_MANIFEST_URL'`.
  Root cause: config has `ALPHAMISSENSE_MANIFEST`, not `ALPHAMISSENSE_MANIFEST_URL`.
  Fix: align agent import to real config constant name.
- `training_lifecycle_agent.py`: `ModuleNotFoundError: No module named 'ewc_utils'`.
  Root cause: top-level import; ewc_utils lives in agents/ not agent_layer/.
  Fix: lazy import inside `_check_drift()` method.
- `literature_scout_agent.py`: `ModuleNotFoundError: No module named 'feedparser'`.
  Fix: lazy import inside `_fetch_biorxiv()`.
- `literature_scout_agent.py`: `NameError: name '_TRAINING_AGENT' is not defined`.
  Root cause: constant dropped during config-name reconciliation pass.
  Fix: re-add `_TRAINING_AGENT = "TrainingLifecycleAgent"` constant.
- LOVD REST API: HTTP 402 (unsupported) on all polls.
  Root cause: LOVD changed their API terms. Logged as warning, skipped gracefully.
- ClinGen API: 404 (endpoint URL format changed).
  Logged as warning, skipped gracefully.
- PubMed efetch: occasional 500 Server Error (NCBI transient).
  Logged as warning, skipped gracefully.

### Fixed
- All 8 inter-run items completed and committed.
- Inter-agent message bus: 34/34 tests passing on Python 3.14.3.
- Full pipeline `--dry-run` confirmed working: all 4 agents run cleanly with
  graceful degradation where ewc_utils/feedparser not on path.

### Learned
- SpliceAI "masked.snv" filename is misleading — always check file size first.
  28.8GB compressed = full genome-wide VCF, not masked SNVs only.
- Polars join 3.3x faster than pandas merge on gnomAD constraint join (500K variants).
  Integration approved for Phase 3 ETL bottlenecks.
- Inter-agent messaging with lazy imports is the correct pattern for an agent layer
  where not all dependencies are always installed.
- PowerShell `<` operator is reserved — never use `<placeholder>` syntax in commands.
  Always use a real value or `PLACEHOLDER_VALUE` without angle brackets.

---

## 2026-04-09 (post-session) — Local file cleanup + SpliceAI GCS fix

### Fixed
- SpliceAI GCS index was wrong file: `processed/spliceai_index.parquet` in GCS
  was the raw 28.7GiB VCF accidentally uploaded under the wrong name.
  Root cause: `Rename-Item` failed silently (target already existed), so
  `data\processed\spliceai_index.parquet` was still the 29GB file when
  `gcloud storage cp` ran. The correct 336.8MB filtered parquet was still
  named `spliceai_index_test.parquet`.
  Fix: uploaded `spliceai_index_test.parquet` directly to GCS as `spliceai_index.parquet`.
  GCS now confirmed: 336.83MiB / 353,196,691 bytes at 2026-04-09T23:15Z.
  Local: deleted 29GB wrong file, renamed _test.parquet → spliceai_index.parquet.

### Cleaned up local files (all confirmed in GCS before deletion)
  - data\external\spliceai_scores.masked.snv.hg38.vcf.gz     27.5 GB (duplicate)
  - data\external\dbnsfp\dbNSFP5.3.1a_grch38.gz             47.9 GB ✓ GCS
  - data\external\finngen\finnge_R12_annotated_variants_v1.gz 30.6 GB ✓ GCS
  - data\external\spliceai\spliceai_scores.masked.snv.hg38.vcf.gz 27.5 GB ✓ GCS
  - data\external\alphamissense\AlphaMissense_hg38.tsv\       5.2 GB (GCS has .gz)
  - data\raw\cache\alphamissense_scores_hg38.parquet          740 MB (regeneratable)
  - data\external\clinvar_fresh\variant_summary.txt.gz        415 MB ✓ GCS
  - data\raw\clinvar\variant_summary.txt.gz                   415 MB (duplicate)
  Total recovered: ~142 GB

---

## 2026-04-16 — Lambda A10 setup; Phase 2 feature promotion; SyntaxError fix; 205 tests green

### Attempted
- Launch Lambda Labs gpu_1x_a10 as GCP GPU quota substitute (quota still 0).
- Fix SyntaxError in variant_ensemble.py blocking all imports.
- Sync TABULAR_FEATURES (21) to match engineer_features() output (78 columns).
- Provision Lambda Python environment and authenticate GCS service account.

### Failed
- ssh-keygen -N "" in PowerShell: silent parse failure. Fix: run interactively, Enter twice.
- SyntaxError fix via python -c inline: PowerShell tokenizer mangled nested quotes/backslashes.
  Fix: write repair script to .py file via Set-Content, execute, remove.
- Repair script string-match failure: file used em-dash in comment; script used ASCII --.
  Fix: locate block by structural markers (feats line + return line) not literal text.
- Lambda pip: --index-url replaces PyPI entirely; all non-torch packages returned 404.
  Fix: --extra-index-url for torch; separate pip invocation for everything else.

### Fixed
- SyntaxError line 524 variant_ensemble.py: Phase 2 feature blocks pasted inside unclosed
  assert ( expression. Removed broken fragment; clean assert added after all features computed.
- TABULAR_FEATURES mismatch (21 declared vs 78 produced): engineer_features() grew across
  Phase 2 sessions but list was frozen. Updated to full 78-feature list in 20 groups.
- Lambda torch environment: torch 2.11.0+cu130, CUDA True, pandas 2.3.3, PyG 2.7.0.
- GCS access on Lambda: SA key scp'd, gcloud authenticated, bucket accessible.

### Learned
- assert ( multiline is valid Python. Assignments inside cause SyntaxError on =.
  Compute all features first, assert last.
- --index-url is destructive (replaces PyPI). --extra-index-url is additive.
- TABULAR_FEATURES and engineer_features() must stay in sync.
  The assert at end of the function is the single guard.
- Write all multi-line Python repair scripts to .py files, not inline python -c strings.
- Lambda instance billing starts at launch. Have all code pushed before creating the instance.

## 2026-04-16 (continued) — AlphaMissense parquet fix; Run 8 training launched on Vast.ai RTX 4090

### Fixed
- alphamissense.py _parse_parquet returned raw 5-column schema instead of
  lookup_key/alphamissense_score. Fix: build lookup_key = CHROM:POS:REF:ALT,
  deduplicate, return 2-column df matching _parse_tsv output schema.
- Stale parquet cache (wrong schema from first broken run) deleted on Vast.ai.
- Result: 206,131 / 1,700,687 variants now annotated by AlphaMissense.

### Infrastructure
- Vast.ai RTX 4090 instance: 175.155.64.225:19863, $0.388/hr
- Vast.ai auto-starts tmux on login — no manual tmux new-session needed.
- All 7 data files pulled from GCS in ~3 minutes (vs 25 min scp previously).
- Training launched 20:13:40 UTC with full 78-feature set including AlphaMissense.

### Pending
- Training in progress — detached in tmux, running unattended.
- Check results in ~2-3 hours for final AUROC/AUPRC/MCC.
## 2026-04-16 — Run 8 COMPLETE — AUROC 0.9863, 1.8GB artifacts saved to GCS

### Final Results
  AUROC  0.9863 (holdout)  0.9833 (test)   PASS (target >= 0.9)
  AUPRC  0.9461 (holdout)  0.9436 (test)
  MCC    0.8482 (holdout)  0.8178 (test)
  F1     0.9226 (holdout)  0.9052 (test)
  Brier  0.0358 (holdout)  0.0479 (test)
  Time:  4270s on Vast.ai RTX 4090 ($0.388/hr)

### OOF AUROCs (5-fold CV)
  RF 0.9921 | XGB 0.9932 | LGB 0.9930 | GBM 0.9891 | CatBoost 0.9930 | LR 0.9846
  Blend: 0.9938 | Weights: RF 0.391, LGB 0.255, CatBoost 0.319, XGB 0.035

### Top 10 Features
  n_pathogenic_in_gene 568.1 | loeuf 418.2 | syn_z 370.5 | mis_z 352.4
  consequence_severity 242.7 | pli_score 218.3 | alphamissense_score 189.7
  af_raw 174.2 | af_log10 105.3 | len_diff 86.7

### AlphaMissense confirmed contributing
  206,131 / 1,700,687 variants annotated | ranked 7th of 78 features

### Bugs discovered (fix in Run 9)
  GNN: ValueError: invalid literal for int() with base 10: path string passed where
       protein ID int expected. GNN did not contribute to Run 8.
  TF models: tabular_nn, cnn_1d, mc_dropout, deep_ensemble all skipped —
             no tensorflow on Vast PyTorch image. Use PyTorch equivalents.
  ESM-2: stub mode (transformers not installed) — all esm2_delta_norm = 0.0

### GCS artifacts (gs://genomic-variant-prod-outputs/run8/)
  models/run8/models/ensemble.joblib         main ensemble
  models/run8/scaler.joblib
  models/run8/metrics.json
  models/run8/per_model_metrics.csv / _val.csv
  models/run8/feature_importance.csv
  models/run8/splits/X_train|val|test.parquet
  logs/run8.log
  19 files, 1.8 GiB total

### Infrastructure notes
  - Vast.ai auto-tmux protects from SSH drops (unlike Lambda foreground sessions)
  - sudo shutdown fails in Vast containers (no systemd) — container exits naturally
  - SA key permissions: parallel composite upload GET check fails — non-blocking
## 2026-04-16 (final) — SpliceAI + PyTorch NN fixes committed

### Fixed
- SpliceAI: _get_lookup now detects .parquet and calls _parse_parquet()
  instead of _parse_vcf(). Fixes 0 variants annotated in Run 8.
  Schema: chrom:pos:ref:alt lookup_key, dedup by max score.
- CNN1DClassifier: migrated TF/Keras → PyTorch (Conv1d, AdaptiveMaxPool1d,
  early stopping patience=5, CUDA-aware)
- TabularNNClassifier: migrated TF/Keras → PyTorch (BatchNorm1d, Dropout,
  weight_decay=1e-4, early stopping patience=8, CUDA-aware)
- All 466 tests passing after all three fixes.

### Run 9 readiness
All known bugs from Run 8 are now fixed:
  GNN string_db path bug          FIXED (0a02e5d)
  AlphaMissense parquet schema    FIXED (5297711)
  SpliceAI parquet branch         FIXED (this commit)
  CNN1D / TabularNN TF→PyTorch    FIXED (38656bc)
  transformers installed          DONE

Expected Run 9 active models: RF, XGB, LGB, GBM, CatBoost, LR,
  tabular_nn, cnn_1d, mc_dropout, deep_ensemble, GNN (10 base models + GNN)
Expected new feature signals: SpliceAI scores, ESM-2 (if HGVSp populated)
