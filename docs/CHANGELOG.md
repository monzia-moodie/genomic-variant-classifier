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
---

## 2026-04-17 — SpliceAI silent-zero fix, test isolation, GCS audit

### Attempted
- Verify Run 8 SpliceAI parquet was actually in GCS (could not be
  confirmed from prior sessions because gsutil kept returning 401).
- Patch `SpliceAIConnector` to default to the production parquet
  instead of silently returning 0.0 for all variants.
- Add regression test and confirm no regressions across the unit
  suite.

### Failed
- gsutil returned `401 Anonymous caller` on every GCS list attempt.
  Root cause: gsutil and `gcloud storage` have separate credential
  stores; gsutil's were stale. The SpliceAI parquet was in fact in
  GCS the whole time (since 2026-04-09). This cost multiple sessions
  of uncertainty.
- v1 test patch monkeypatched `FetchConfig.cache_dir` as a class
  attribute, which has no effect on dataclass instance fields.
  Individual test appeared to pass in 61s but sibling tests rebuilt
  the 430 MB production cache on the next `TestAnnotationPipeline`
  run.
- v2 test patch (short-circuiting `_load_cache` for one test) didn't
  cover the other 15 tests in the class. Full class run hit a
  5-minute timeout mid-import while building the cache.

### Fixed
- `src/data/spliceai.py`: renamed `DEFAULT_VCF_PATH` to
  `DEFAULT_SPLICEAI_PATH` pointing at
  `data/external/spliceai/spliceai_index.parquet`. `__init__` now
  falls through to this default when `vcf_path=None` is passed. This
  closes the Run 8 silent-zero failure mode - the connector no
  longer returns 0.0 for all variants when `AnnotationConfig()` is
  constructed with defaults.
- `tests/unit/test_spliceai_parquet_default.py`: new regression test
  (~3-7s runtime) that builds a 3-row synthetic parquet, instantiates
  `SpliceAIConnector()` with no args, and asserts at least one
  non-zero `splice_ai_score`.
- `tests/unit/test_core.py`: added class-scoped `autouse=True`
  fixture `_isolate_spliceai` at the top of `TestAnnotationPipeline`.
  Monkeypatches `DEFAULT_SPLICEAI_PATH` (nonexistent tmp file) and
  `BaseConnector._load_cache` (returns None), short-circuiting
  SpliceAI disk I/O for all 16 tests in the class. Full class runs in
  2:28 instead of timing out.
- `scripts/verify_spliceai_index.py`: parquet integrity/schema/null
  checks. Used at session start to confirm the production parquet
  (45,549,300 rows, 10 columns, no nulls outside of MT chromosome).
- `docs/CHANGELOG.md`: deduplicated the triplicated
  `## 2026-04-16 (final)` heading caused by PowerShell heredoc
  collision on session close. Net -46 lines.

### Learned
- Silent-zero connector fallbacks are bugs, not features. Future
  connectors should assert file existence at startup, not silently
  return defaults at runtime.
- `gsutil` is deprecated and has a separate credential store from
  `gcloud`. Use `gcloud storage ls` exclusively for authoritative
  GCS-state checks. Never trust `gsutil 401` as evidence of absence.
- Dataclass fields cannot be monkeypatched via
  `setattr(Class, "field", value)` - patching has no effect on new
  instances. Patch instance methods or module constants instead.
- Class-scoped `autouse=True` fixtures are the right tool for
  preventing disk-I/O side effects across every test in a class,
  including future tests that don't yet exist.
- Run scoped tests (`pytest path::Class::test -v --timeout=N`) before
  full suites when iterating on fixes. A 20-minute suite is the
  worst feedback loop.
- PowerShell heredocs (`@'...'@ | Add-Content`) corrupt reliably when
  content contains triple-quoted Python or literal commit messages.
  Use standalone `.py` files instead.
- `Get-Content | Add-Content` can silently fail with empty pipelines
  or encoding conflicts on existing files. For reliable appends,
  read and write in a single .NET call via
  `[System.IO.File]::AppendAllText`.

### Commits
- `9ba3127` feat(spliceai): default to parquet index; add regression
  tests; dedupe changelog (5 files changed, 191 insertions, 50
  deletions).
- `8b12f76` docs: session 2026-04-17 - SpliceAI default path fix
  (session doc only; CHANGELOG append failed silently and was
  applied in a follow-up commit).



## 2026-04-17 (afternoon, take 2) --- Run 9 infra + ESM-2 silent-zero discovery

(Note: the earlier afternoon CHANGELOG entry was draft; this supersedes
it. Kept in-place because the ESM-2 discovery materially changed the
story and file contents.)

### Added

- `scripts/preflight_check.py` (local, pre-launch gate): scripted
  enforcement of standing rule #1. Checks git tree, HEAD == origin/main,
  full pytest suite, local data files, GCS objects via `gcloud storage
  ls` (2026-04-17 rule), GITHUB_TOKEN from .env/session/Windows-User-env,
  transformers+torch importable, no tensorflow, SpliceAI test-cache
  absence. Allowlists two pre-existing carry-overs
  (`scripts/gcp_run6_startup.sh`, `ROADMAP_PSYCH_GWAS_ENTRY.md`).
  Supports `--skip-pytest` and `--skip-gcs` flags for fast iteration.
  Three revisions this session to work around Windows `.cmd` shim
  handling in subprocess.

- `scripts/preflight_vm.sh` (on-VM, post-SSH gate): checks nvidia-smi,
  `torch.cuda.is_available()`, data-file presence on container FS,
  transformers>=4.40, git HEAD, and all critical Python imports.

- `tests/unit/test_esm2_activation.py`: three-test regression module
  for ESM-2 stub-mode detection. Skipped on machines without
  transformers. When transformers is present: gates API drift, gates
  the real-mode path (passes when all four required columns are
  present and backend+network available), and explicitly documents
  the current stub-mode expected-behavior via a separate test that
  fails loud if the connector ever starts silently inferring the
  parsed columns.

- `scripts/run9_launch.md`: operational runbook for Run 9. Updated
  to explicitly expect ESM-2 stub mode in training logs per the
  INCIDENT doc. Pins `transformers>=4.40,<5.0` on Vast.ai installs.

- `docs/incidents/INCIDENT_2026-04-17_esm2-hgvsp-parser.md`: full
  root-cause record for the ESM-2 silent-zero that affected Runs
  6-8. The training pipeline never populated `wt_aa`/`mut_aa`/
  `protein_pos` (grep of `src/` returned only esm2.py as reader,
  nothing as writer); the connector logged an INFO message and
  returned all zeros. Remediation plan: add `src/data/hgvsp_parser.py`
  in Run 10.

### Discovered

- **ESM-2 has been inert in Runs 6, 7, and 8**. Root cause: pipeline
  does not populate the four columns the connector requires
  (`gene_symbol`, `protein_pos`, `wt_aa`, `mut_aa`). Connector emits
  an INFO-level log ("columns missing -- defaulting to 0.0") that was
  not being grepped. Feature-importance rankings showed ESM-2 below
  top 20, which was indistinguishable from "feature contributes
  literally zero" vs "feature contributes weakly".

- **EVE is almost certainly in the same state**. Same column-pattern:
  `eve.py:232` reads `wt_aa`/`mt_aa`/`position`/`mutations_protein_name`;
  none written by pipeline. Full diagnosis deferred to Run 10, when
  the HGVSp parser can populate both ESM-2 and EVE inputs.

### Design notes

- **Dual-layer preflight** (local + on-VM) is the minimum correctness
  boundary for Run 9, not redundancy.

- **Connector fallbacks with INFO logs are silent**. For any connector
  with a graceful fallback path, preflight should test that the
  fallback fails loud. SpliceAI got this in commit 9ba3127; ESM-2 got
  it in this session. Audit other connectors (EVE, AlphaMissense,
  CADD) for the same pattern as a Run 10 prerequisite.

- **Zero-fraction audit belongs in the agent layer**. Feature-importance
  alone cannot distinguish "weak feature" from "inert feature".
  Planned: nightly job that prints zero-fraction per feature per
  dataset and alerts when a feature flips to 1.0 zero-fraction.

### Learned

- Read connector source before writing its test. First ESM-2 test
  draft assumed 1280-dim embedding columns; actual API is a scalar.
- Windows gcloud subprocess requires `shell=True` when the cmd token
  is a bare name without explicit path or `.exe`. subprocess cannot
  resolve `.cmd` shims via `CreateProcess`.
- `[System.IO.File]::AppendAllText` does not add a separator before
  the appended content. If the target file doesn't end with `\n`, the
  append gets concatenated onto the final line. Fix: include `\n\n`
  prefix in the appended content, or check-and-add-newline first.

### Commits queued

- `feat(run9): scripts/preflight_check.py + scripts/preflight_vm.sh + ESM-2 smoke test + launch runbook`
- `docs(run9): INCIDENT for missing HGVSp parser + session doc + CHANGELOG`

### Run 9 readiness after this session

- [x] local preflight script on disk (3rd revision, all bugs fixed)
- [x] VM preflight script on disk
- [x] ESM-2 smoke test on disk (matches actual connector schema)
- [x] launch runbook on disk (expects ESM-2 stub)
- [x] INCIDENT doc filed
- [ ] Vast.ai instance provisioned (user action)
- [ ] on-VM preflight passes (requires live instance)
- [ ] training launched and final metrics captured

## 2026-04-20 — KAN reinstatement, ensemble OOF fix, CI recovery

Entered session investigating a CI failure (`pytest (3.11)` red since
2026-04-19). The failing test surfaced a pre-existing bug in
`VariantEnsemble.fit` that was simultaneously blocking Run 9's ablation
harness at ~10 hours of CPU time. Fix verified with a 500-row synthetic
probe in under 2 minutes. Separately, investigation of the local
`skip_kan` behaviour during that probe revealed the `KAN unconditionally
removed` status was 15 days out of date — the underlying OOM was
fixed in commit 2389ee2 on 2026-04-04. With Vast.ai GPU access for
Run 9, the remaining reason to keep KAN disabled evaporated. Three
atomic commits shipped, all CI green.

### Changed

- `src/models/variant_ensemble.py` (b1c1150): removed stale duplicate
  `self.meta_learner.fit(oof_preds, y_arr)` call at line 1159. The
  correct call one block below used `y_fit` (length 0.85 × N, matching
  `oof_preds`) but never ran because the stale call crashed first with
  `ValueError: Found input variables with inconsistent numbers of
  samples: [N*0.85, N]`. Pre-existing bug from a botched earlier
  patch; not introduced by Patch 1 (8a7e2da). Fix is `-7/+1` lines
  and unblocks both CI and the Run 9 ablation harness.

- `scripts/run_phase2_eval.py` (8f9eb60): added `--skip-kan` argparse
  flag, threaded through `EnsembleConfig(skip_kan=args.skip_kan)`,
  and replaced the unconditional
  `ensemble.base_estimators.pop("kan", None)` with a
  `if args.skip_kan:` gate. Default behaviour change: **KAN is now in
  the ensemble by default**. Pass `--skip-kan` to opt out. Matches
  items 3 and 4 of the ROADMAP KAN Re-enablement Checklist. Side
  effect: fixes the broken Dockerfile trainer CMD (see INCIDENT below).

### Added

- `scripts/run9_ablations.py` (128331f, 780 lines, new file): LOCO
  ablation harness for Run 9+ with 14 ablation targets. Coexists with
  `run_phase2_eval.py`; reads already-scaled splits from
  `<run>/splits/` and applies feature-prefix ablations by zeroing
  matching columns. Handles the 78-column schema confirmed on
  2026-04-19. Includes `--skip-kan` and `--skip-mc-dropout` CLI flags,
  a `no_kan` MODEL-level ablation, and a runtime guard that errors
  exit 2 if `--ablation no_kan` is passed without `--skip-kan`
  (preventing silent no-op runs).

- `docs/sessions/SESSION_2026-04-20.md`: session record covering the
  OOF bug diagnosis, KAN history reconstruction, reversal decision,
  and three-commit shipping sequence.

- `docs/incidents/INCIDENT_2026-04-20_dockerfile-trainer-skip-kan.md`:
  documents the Dockerfile trainer CMD passing a non-existent argparse
  flag from 2026-04-09 through 2026-04-20. Resolved as a side-effect
  of commit 8f9eb60 adding the flag.

### Discovered

- **The KAN "unconditionally removed" status was 15 days out of date.**
  Commit 2389ee2 (2026-04-04) added a 100K-sample stratified subsample
  gate in `KANClassifier._fit_pykan` that caps peak RAM at ~0.3 GB
  (from 17.9 GB). The hardcoded `pop("kan", None)` in
  `run_phase2_eval.py` was added in Run 6 prep (commit a0a732d on
  2026-04-05) as belt-and-braces caution and outlived its
  justification. ROADMAP had a documented re-enablement checklist
  (`docs/ROADMAP.md` lines 206-212) that was actionable-but-unactioned.
  `LiteratureScoutAgent` (`agent_layer/agents/version_monitor_agent.py`,
  commit a95c9db) already monitors pykan PyPI releases programmatically.

- **The Dockerfile trainer CMD has been broken since 2026-04-09.**
  Commit 671e48d added `--skip-kan` to the `scripts/run_phase2_eval.py`
  invocation at Dockerfile line 166. Until today, `run_phase2_eval.py`
  did not accept that flag — argparse would have errored with
  `unrecognized arguments: --skip-kan` and exit 2. Undetected for 11
  days because Runs 6-8 used startup scripts on GCP/Lambda/Vast.ai
  (`scripts/gcp_run6_startup.sh` etc.), not Docker. The trainer
  container was never invoked after 2026-04-09. Commit 8f9eb60
  incidentally fixes this by making the flag exist.

- **CI has been red since at least 2026-04-19** on the same OOF bug
  that blocked Run 9. Test
  `tests/unit/test_api.py::TestInferencePipeline::test_save_and_load_roundtrip`
  was failing at 20-sample scale with the identical `[N*0.85, N]`
  inconsistency that the Run 9 ablation harness hit at 1.2M-sample
  scale after ~10 hours of training. Commit b1c1150 fixes both.

- **Dockerfile is CPU-only multi-stage.** All three stages (builder,
  api, trainer) use `python:3.11-slim-bookworm`. No CUDA runtime, no
  GPU base image. GPU training happens via startup scripts on
  Vast.ai/Lambda/GCP, not via Docker. No change needed for Run 9.

### Design notes

- **500-row synthetic probes are fast enough to be a pre-commit gate.**
  Exercising the full `VariantEnsemble.fit()` code path with tree
  models + KAN took ~90 seconds on the CPU-only laptop, compared to
  22+ hours on the same hardware at real 1.2M-row scale (which
  crashed before meta-learner fit regardless). Used this session to
  verify the OOF fix before committing, then again to verify v4.1/v4.2
  skip-flag semantics. Standard pattern going forward: any change to
  `VariantEnsemble.fit` or `_build_estimators` gets a synthetic probe
  before any attempt at scaled training.

- **`no_kan` is a model-level ablation, not a feature-level one.**
  KAN uses the same 78 input features as every other base estimator,
  so there are no feature columns to zero. The harness handles this
  by adding `no_kan` to `ABLATION_MASKS` with an empty prefix list
  and gating execution on both `--ablation no_kan` AND `--skip-kan`.
  Without the runtime guard, `--ablation no_kan` alone would zero
  zero columns and train KAN anyway — a silent ~10-hour no-op on a
  GPU instance. The guard returns exit 2 with an explanatory message.

- **The KAN Re-enablement Checklist in ROADMAP.md was the right spec.**
  Every item on the checklist mapped cleanly to one of the commits
  shipped today. This is a data point for the value of maintaining
  forward-looking checklists in ROADMAP.md: when a condition changes
  (OOM fix + GPU access) that triggers the checklist, the work is
  already scoped.

### Learned

- **Read the notes before enforcing the decision.** Entering the
  session, memory note said "KAN unconditionally removed pending
  pykan memory fix" and I began to enforce that rule. User pushed
  back and asked me to investigate the history. The investigation
  took ~20 minutes of grep over `docs/`, `logs/`, code files, and
  git log and surfaced that (a) the OOM was fixed 15 days ago, (b)
  Vast.ai GPU access changes the calculus anyway, (c) there's a
  documented re-enablement checklist waiting to be executed. Had I
  proceeded without the investigation, KAN would still be absent.
  Standing rule #13 exists for exactly this class of error.

- **Failing-loud beats failing-silent at every scale.** The
  `--ablation no_kan` guard that returns exit 2 when `--skip-kan` is
  absent is a small amount of code (six lines) that prevents a
  ~10-hour silent no-op run on a GPU instance. Mirrors the SpliceAI
  fail-loud fix from commit 9ba3127 and the ESM-2 stub-detection
  test from 2026-04-17. Pattern: if a feature or model can be
  silently absent, add a loud check that forces the absence to
  announce itself.

- **Grep before inferring.** Initial plan for this session
  extrapolated `LiteratureScoutAgent` as a planning abstraction
  from a one-line memory note. Grep surfaced a committed
  `agent_layer/agents/version_monitor_agent.py` (commit a95c9db)
  that already does exactly what was planned. Default to reading
  the repo over reading the notes about the repo.

### Commits shipped this session

- `b1c1150 fix(ensemble): meta-learner fit uses y_fit to match oof_preds length`
- `8f9eb60 feat(ensemble): add --skip-kan CLI flag, remove hardcoded KAN removal`
- `128331f feat(run9): KAN as first-class ablation target; --skip-mc-dropout flag`

All three green on CI (pytest 3.11, pytest 3.12, lockfile drift check,
Docker build smoke test).

### Deferred to post-Run-9 cleanup

- **CI dependency conflict:** `requirements.txt` pins
  `starlette==1.0.0` but `prometheus-fastapi-instrumentator==7.1.0`
  (pinned by `requirements-api.lock` transitively) requires
  `starlette<1.0`. Pip emits a non-fatal ERROR during CI install and
  the installed env has the incompatible combination. Test suite
  passes because no current test imports Prometheus
  instrumentation, but runtime behaviour when instantiating the
  FastAPI app is untested. Fix: upgrade
  `prometheus-fastapi-instrumentator` to a version supporting
  starlette ≥1.0, or pin `starlette<1.0` in `requirements.txt`.
  File INCIDENT after Run 9 completes per user instruction.

### Run 9 readiness after this session

- [x] ensemble meta-learner fit bug fixed (b1c1150)
- [x] `--skip-kan` CLI available in `run_phase2_eval.py` (8f9eb60)
- [x] `scripts/run9_ablations.py` on disk with 14 ablation targets (128331f)
- [x] `no_kan` ablation first-class with runtime guard
- [x] CI green on main
- [x] KAN Re-enablement Checklist items 207-211 complete (ROADMAP.md)
- [ ] splits regenerated against current 78-col schema (user action)
- [ ] Vast.ai instance provisioned (user action)
- [ ] Python 3.12 venv locally (optional; deferred — Vast.ai handles its own Python)
- [ ] Step C: verify Patch 6a `--string-db auto` branch triggers GNN injection
- [ ] on-VM preflight passes (requires live instance)
- [ ] KAN scalability pre-flight at 10K and 100K rows on GPU before full run
- [ ] training launched and final metrics captured


## 2026-04-30

### Attempted
- Stage 3 splits regen (run_phase2_eval.py with [GNN-TRACE]
  instrumentation, --skip-nn --skip-svm --skip-kan, --string-db auto,
  --n-folds 2, output outputs/run9_ready/)

### Failed
- GNN training: KeyError 'gene_symbol' in build_pyg_dataset.
  Caught by `except Exception` and downgraded to warning. on-disk
  gnn_score remained 0.0 across all three splits.
- --skip-nn flag did not skip mc_dropout/deep_ensemble (memory #17
  confirmed). Wall-clock cost: 10h+ of the 13h total runtime.

### Fixed (this session)
- Stage 1: .venv312 bootstrapped on Python 3.12.10. requirements.txt +
  torch 2.11.0+cpu + torch_geometric 2.7.0 installed cleanly.
  Pandas pinned to 2.3.3 (was 3.0.1).
- Stage 2: [GNN-TRACE] instrumentation patch landed in
  scripts/run_phase2_eval.py (18 logger calls, 4/4 verification gates
  green). Backup at scripts/run_phase2_eval.py.bak-gnn-trace.
- Stage 3: data prep + ensemble training completed end-to-end.
  Test AUROC 0.9814, val AUROC 0.9850.

### Drafted (committed in next session)
- Patch 6b (scripts/apply_patch_6b.py): persist meta_train.parquet
  in DataPrepPipeline._save_splits, source gene_symbol from it in
  run_phase2_eval.py for gnn_df construction.
- 5K-row synthetic probe (scripts/probe_patch_6b.py).

### Learned
- Generic `except Exception: logger.warning` masks crashes. Either
  narrow the except or use exc_info=True. [GNN-TRACE] insertion 9
  uses exc_info=True and would have surfaced this immediately on
  first run.
- Patches that re-persist on success path must verify success
  before persisting. Patch 6a re-persists regardless of whether
  gnn_scorer was built.
- Memory #19 (no local retraining) was violated this session at
  cost of 13h. Reaffirming.
- run9_ready splits are a valid GNN-FREE BASELINE for paper P4
  comparison. Don't discard.

---

## 2026-05-02 — Gene-scope expansion deferred to Run 10; LOVD silent-zero confirmed

### Attempted
- Review of request to add additional gene variants beyond the canonical
  10 (BRCA1, BRCA2, MLH1, MSH2, MSH6, APC, NF1, TP53, PTEN, RB1) before
  Run 9, with two LOVD admin emails attached as context.
- Investigation of LOVD subsystem state (connector wiring, on-disk data,
  trained feature matrix) to scope the integration work properly.
- Three-stage diagnostic: schema check on `lovd_all_variants.parquet`,
  value_counts on trained matrix, structural merge replicating the
  connector's logic in isolation.

### Failed
- LOVD `lovd_variant_class` is identically `0` across all 1,197,216 rows
  in `outputs/run9_ready/splits/X_train.parquet` despite:
  - LOVD parquet on disk being structurally healthy (18,006 rows, 10
    genes, joinable schema).
  - LOVDConnector being unconditionally invoked at
    `src/data/real_data_prep.py:738` with return value assigned.
  - Diagnostic merge (replicating the connector's exact key construction
    against `models/v1/clinvar_enriched.parquet`) yielding 5,553 inner-
    join matches in isolation.
  Root cause is at one of the runtime join boundaries inside the ETL —
  either Cause 1 (downstream column overwrite) or Cause 2 (upstream
  coordinate transformation by one of the 14 prior `annotate_dataframe`
  steps). Distinguished by the integer in the log line at
  `real_data_prep.py:740–748` (`"Score annotation 15/16 (LOVD): %d
  variants with lovd_variant_class > 0."`); resolution deferred to R10-A.
  Full record: `docs/incidents/INCIDENT_2026-05-02_lovd-silent-zero.md`.
- Initial hypothesis (float→str trailing `.0` on the `pos` join key)
  falsified by direct dtype check: `pos` is int64, conversion is clean.

### Fixed
- Nothing patched this session. All identified work moved to Run 10.

### Learned
- LOVD label-quality is functional-translated-to-clinical, not clinical.
  Per LOVD admin's 2026-04-01 second email: clinical classification
  field intentionally withheld from API pending ACMG v4. API exposes
  `effect_reported`/`effect_concluded` (functional). Per ACMG/AMP 2015
  framework, functional evidence (PS3/BS3) is one input to a clinical
  classification combining multiple categories, not the classification
  itself. ClinVar tier-2 → LOVD-API-derived is a label-quality
  downgrade. Earlier-session "30× more rows" framing was rhetorical
  and was flagged as such mid-session.
- Silent-zero discovery requires checking the *trained* feature matrix
  value distribution, not just connector logs. Connector logged the
  zero count at INFO level once during the 13h regen and the line was
  lost in training output. Recommend post-ETL assertion that any
  feature with single-source contribution must have `nunique() > 1` in
  the training matrix, with clear failure on zero variance. Extends
  the 2026-04-17 audit recommendation (EVE, AlphaMissense, CADD) to
  LOVD; same pattern likely affects other connectors on the 30+
  all-zero list from `SESSION_2026-04-30.md` Finding #4.
- `scripts/process_lovd.py` is dead code. Live LOVD merge is
  `scripts/build_lovd_index.py` → `lovd_all_variants.parquet`. The
  schema mismatch between the two scripts (`lovd_variants.parquet` vs
  `lovd_all_variants.parquet`, `pathogenicity` vs `classification_raw`)
  is a dead-code artifact, not a live bug. Cleanup candidate for a
  separate post-Run-9 commit, low priority.
- `outputs/run9_ready/splits/` is not `data/splits/`. `DataPrepConfig`
  default and the run9 launch path differ. `docs/HANDOFF_run9_launch.md`
  and the Vast.ai onstart script must reference the actual
  `outputs/run9_ready/` path before Vast.ai launch.
- 4/1 raw LOVD download integrity confirmed against admin's logged
  ban window. TP53/PTEN/RB1 `.txt` files at 5:38–5:39 AM Eastern are
  genuine; BRCA1/BRCA2/APC/MLH1/MSH2/MSH6/NF1 `.txt` files at the
  same time are 96–98 byte error pages contemporaneous with the ban
  (`[01/Apr/2026:10:53–12:34 +0200]` → 4:53–6:34 AM Eastern). 6:56 PM
  `.json` files are post-unblock manual saves of
  `?format=application/json` views, currently unconsumed by
  `build_lovd_index.py`.
- rclone Drive remote renamed `gvc` → `genvarcla`. `agent_data/`
  namespace recreated on Drive with 5 subfolders (events, litcache,
  drift_reports, modelscout, trainlifecycle). Local `agent_data/`
  directory created. Smoke test (21-byte file round-trip) clean.
- **Process violations (this session, all recorded in SESSION doc):**
  `PASTE_FULL_PATH_HERE` placeholder in copy-pasteable command;
  bash heredoc syntax in PowerShell context (already covered by
  Windows-platform standing rule on file); loose grep regex framed as
  decisive. Pattern across all three: confident framing on
  under-constrained tooling. Recorded for future-self correction.

### Run 10 sequencing (revised)
- **R10-A:** Grep `outputs/run9_ready/regen.log` for the LOVD annotation
  count line. Distinguishes Cause 1 (downstream overwrite) vs Cause 2
  (upstream coordinate transformation).
- **R10-B:** Patch identified cause. Add unit test asserting
  `(df["lovd_variant_class"] > 0).sum() > 0` after the LOVD step on a
  3×5 fixture with 1 expected match. Pattern modeled on
  `tests/unit/test_spliceai_parquet_default.py` (commit 9ba3127) and
  `tests/unit/test_esm2_activation.py` (2026-04-17).
- **R10-C:** Re-regen splits on Vast.ai with LOVD live (no local
  retraining per standing rule #19). Post-condition: ~4,500–5,500 of
  5,553 inner-join matches in train.
- **R10-D:** Originally-requested gene scope expansion (Paths 1+2: LOVD
  raw + gnomAD/UniProt per-gene). Manual browser only per LOVD admin
  emails of 2026-04-01.
- Cleanup (low priority, post-Run-9): remove `scripts/process_lovd.py`
  and orphaned `data/external/lovd/lovd_variants.parquet`.

### Run 9 readiness after this session
- Run 9 launch path **unaffected**. Run 9 inherits the same silent-zero
  baseline as run9_ready (Test AUROC 0.9814, Val AUROC 0.9850). Adding
  this INCIDENT as a known-pending item before Run 9 launch but not as
  a launch blocker.
- All four files for this session committed in a single commit:
  `docs(session): 2026-05-02 — gene-scope expansion deferred; LOVD silent-zero INCIDENT`.

## 2026-05-09: C3.6 hotfix + C4-prep complete

### Attempted
- Pre-condition audit for C4 pickle migration (Stage 1)
- Spec compliance audit of `scripts/migrate_pickles.py` (Stage 2)
- Functional smoke of `install_compat_aliases` (Stage 2 D)
- L119 patch for AttributeError on `_new_root.agent_layer` (Stage 2.5b)
- Diagnose namespace-vs-regular package status of `agent_layer` (Stage 2.5c)
- Add `agent_layer/__init__.py` and re-test alias count (Stage 2.5d)
- C3.6 hotfix: sweep bare imports of `agents`/`config`/`message_bus`/`shared_state` (Stage 2.5e)
- Build `tests/fixtures/migration_smoke.parquet` (Stage 3)
- Final readiness check (Stage 4)
- Two-commit push: C3.6 hotfix + C4-prep (Stage A + B)

### Failed
- Initial `install_compat_aliases` smoke threw `AttributeError: module
  'genomic_variant_classifier' has no attribute 'agent_layer'` at L122.
  Bare `import genomic_variant_classifier as _new_root` does not bind subpackage
  attributes; explicit `import genomic_variant_classifier.agent_layer` needed.
- First `__init__.py` retry showed only 22/28 `agent_layer.*` aliases registered;
  6 walk_failures from bare imports in `base_agent`, `data_freshness_agent`,
  `interpretability_agent`, `literature_scout_agent`, `training_lifecycle_agent`,
  and `orchestrator`. C3 regex sweep had missed these.
- Stage 3 reported `WARN: column count 81 != 78` — false alarm; PowerShell `-match`
  against a multi-line string array does NOT populate `$Matches`; stale value from
  prior smoke test `SRC=81` capture was used. Fixture is verified 78 cols by the
  python output itself (`COLS=78`).

### Fixed
- `src/genomic_variant_classifier/agent_layer/__init__.py` created (empty;
  promotes namespace -> regular package; C1 sweep miss resolved).
- `scripts/migrate_pickles.py` L119: explicit
  `import genomic_variant_classifier.agent_layer` added before
  `_new_root.agent_layer` access; C2-spec docstring still aligns.
- 8 files in `src/genomic_variant_classifier/agent_layer/` rewritten to
  fully-qualified imports (44 lines, +1716 bytes total): `agents/base_agent.py`,
  `agents/data_freshness_agent.py`, `agents/interpretability_agent.py`,
  `agents/literature_scout_agent.py`, `agents/training_lifecycle_agent.py`,
  `orchestrator.py`, `run_agents.py`, `test_message_bus.py`.
- `tests/fixtures/migration_smoke.parquet` committed (force-added; 8 x 78,
  48830 bytes, deterministic `df.head(8)` from
  `outputs/run9_ready/splits/X_test.parquet` head; live 78-col schema).

### Learned
- `pkgutil.walk_packages` does NOT recurse into PEP 420 namespace packages by
  default. Empty `__init__.py` converts namespace -> regular package, enabling
  walk recursion. Future migration sweeps should add post-condition tests that
  walk the full module tree.
- C3 regex patterns 6 and 7 (per spec) lacked `\b` word boundaries, allowing
  over-match against names like `agents_helper`. The C3.6 sweep script added
  `\b` as defensive hardening. No actual collisions in current codebase, but
  `\b` is now the preferred pattern for any future migration sweeps.
- PowerShell `-match` against an array filters but does NOT populate `$Matches`.
  To extract groups from multi-line `python -c` stdout, either `-join "n"` to
  collapse first, or use `Where-Object { $_ -match ... }` in pipeline. Bug in
  Stage 3 column-count check was benign (false WARN) but worth fixing in
  future scripts.
- Pre-migration `find_packages()` at repo root discovered `agent_layer/` AND its
  subpackages as TOP-LEVEL packages. So bare `from agents import X` worked
  because `agents` was on `sys.path`. After C1 nested it under
  `genomic_variant_classifier/`, those bare names broke. C3 regex sweep should
  have caught all instances; missed 8 files. Root cause for the miss is not
  fully diagnosed (C3 spec patterns are correct; possibly file-glob omission or
  later re-introduction during C3.x hotfixes — neither verified).

### Refs
- Commits: `e0f4c6e` (C3.6 hotfix), `e34ce7b` (C4-prep)
- HEAD before session: `fc7f63a`
- HEAD after session: `e34ce7b`
- INCIDENT: `docs/incidents/INCIDENT_2026-05-09_c1-c3-sweep-misses.md`
- Session: `docs/sessions/SESSION_2026-05-09.md`
- Spec: `docs/hypotheses/HYP_consolidate-package-layout.md` (C1, C3, C4 sections)
- Operational tooling (in `agent_data/`, NOT in repo):
  `c4_fix_install_compat.py`, `c4_diagnose_walk.py`, `c4_fix_bare_imports.py`,
  `c4_batch_C36_through_4.ps1`, `c4_batch_commits.ps1`


## 2026-05-09 (continuation) — C5 layout-migration cleanup

### Attempted
- C5.1: rewrite stale `src/X` refs in README L196/L223, ci.yml L77, narrow .gitignore cleanup
- C5.2: rewrite stale `src.*` / `src/` refs in 7 active operational docs
- C5.3 discovery: full-repo audit (369 hits across 71 files)
- C5.3a v1: full-repo sweep of 55 files / 83 expected substitutions (Bucket 3)
- C5.3a v2: same scope after regex fix
- C5.3b: remove 8 stale `.gitignore` rules

### Failed
- **C5.3a v1** (Stage 3, no commit, recovered): post-apply stale-ref count 9 ≠ 4 expected. Path-style regex `src/(SUBPKG)/` required trailing slash; missed 5 line-level hits where slash was absent (`src/api + src/models` in Dockerfile L10, bare `src/evaluation`/`src/reports`/`src/utils` at end of L2 in three `__init__.py` files, bare `src/` in `test_1kgp.py` L409). Working tree dirty with 51 partial writes; recovered via `git checkout -- .`.

### Fixed
- **C5.3a v2:** loosened path-style regex to `src/(SUBPKG)(?![A-Za-z0-9_])` (word-boundary lookahead instead of required slash). Catches all 5 v1-missed hits except bare-`src/` in test_1kgp.py L409 (intentional incidental).
- **Stage 1 arithmetic-sanity check** added to v2 batch: parses helper output and asserts `actual_substitutions == baseline_lines - deliberate_skip_lines - incidental_lines + multi_match_extras` (C5.3a v2: `83 == 87 - 4 - 1 + 1`, where `+1` is Dockerfile L10's multi-match adjustment) BEFORE Stage 2 apply. Catches the v1 class of regex-undershoot at dry-run time. See SESSION_2026-05-09_C5.md §Lesson 1 for the full term-by-term derivation.

### Learned
- **STANDING RULE — apply-batch arithmetic sanity:** every mechanical-rewrite batch must assert at Stage 1 (dry-run) that `actual_substitutions == expected_substitutions`, where `expected = baseline_lines - deliberate_skip_lines - incidental_lines + multi_match_extras` (the last term reconciles match-count vs line-count: each non-skipped line with N>1 matches contributes N-1 extras). Without this check, a too-strict regex undershoots silently; the failure surfaces only at Stage 3 post-apply verification, after partial writes. Codify in every future apply helper template.
- **Path-style regex form:** `src/(SUBPKG)(?![A-Za-z0-9_])` (word-boundary lookahead) is more robust than `src/(SUBPKG)/` (required slash).
- **Recovery enforced by pre-flight:** apply batches' pre-flight rejects dirty working trees, ensuring `git checkout -- .` recovery happens before any retry.
- **Substitutions ≠ line-level diff:** helper substitution count and git diff stat can differ when a single line has multiple substitutions (Dockerfile L10: 2 substitutions, +1/-1 in diff).

### Commits
- `d7ed38e` — C5.1
- `4eb1205` — C5.2
- `6a38ee3` — C5.3a (v2): 55 files, 83 substitutions, +82/-82
- `6443af7` — C5.3b: 8 .gitignore deletions

### Refs
- `agent_data/c5_3_discovery.ps1`
- `agent_data/c5_3a_apply_full_sweep.py` (v2)
- `agent_data/c5_3a_batch.ps1` (v2 with Stage 1 arithmetic-sanity)
- `agent_data/c5_3b_apply_gitignore_cleanup.py`
- `agent_data/c5_3b_batch.ps1`
- Session doc: `docs/sessions/SESSION_2026-05-09_C5.md`

---

## 2026-05-10 — Architectural cleanup: GCS retirement (Commits 1-4 of cleanup arc)

### Attempted
- Complete the SCP-only architectural pivot started by the 2026-04-29 GCP project deletion (`INCIDENT_2026-04-29_gcp-billing-deletion.md`). Required four ordered commits: incident formalization, runtime GCS strip, operational docs rewrite, and session log + CHANGELOG cap.

### Failed
- **Stage 3 batch parser** (parse-time, no writes, recovered): PowerShell `$p:` in double-quoted strings parsed as scope/drive prefix. Anchor at L162:33 reported `Variable reference is not valid. ':' was not followed by a valid variable name character.` Fixed by wrapping in `${p}:` form.
- **Stage 3 P1.6 dry-run** (anchor not found, no writes, recovered): anchor at `scripts/run9_launch.md:200-201` had a trailing `\n` but the file ends at L201 without a terminal newline. Fixed by removing the trailing newline from the P1.6 anchor and replacement (matches both `receipt.` and `receipt.\n[more]` cases via `text.count(old)`).
- **Save procedure silent failure** (state corruption, recovered): `Move-Item -Force` from `~\Downloads` to `agent_data\` removes the source. Subsequent re-attempts find the source missing and silently no-op, leaving `agent_data\` with no file at all. Fixed by adding a `Test-Path` source check BEFORE removing the destination.

### Fixed
- **Commit 1/4 (`b15a625`)** — `docs(incident): formalize 2026-04-29 GCP project deletion + SCP-only architectural pivot`. Created `docs/incidents/INCIDENT_2026-04-29_gcp-billing-deletion.md` (4065 bytes); deleted stale `secrets/gcp-sa-key.json`.
- **Commit 2/4 (`aad8f5a`)** — `chore(arch): strip GCS from active runtime code`. Removed `upload_to_gcs()` (`prediction_artifacts.py`), `gcloud auth` block (`preflight_check.py`), GCS bucket config (`agent_layer/config.py`), GCS-mode pytest assertions (`agent_layer/test_message_bus.py`). 4 files, 5 insertions, 90 deletions. Live `upload_to_gcs` callers post-strip: 0.
- **Commit 3/4 (`feece15`)** — `docs(arch): rewrite operational docs for SCP-only architecture`. 4 files, 20 atomic patches, 30 GCS hit-lines removed: `scripts/run9_launch.md` (11), `docs/HANDOFF_run9_launch.md` (2), `docs/RUN9_OPERATIONS_PLAYBOOK.md` (9), `docs/RUN9_SCIENTIFIC_DESIGN.md` (8). 62 insertions, 62 deletions (balanced textual rewrite). Post-patch GCS hit count across all four files: 0.
- **Commit 4/4 (this commit)** — session log + CHANGELOG cap.

### Learned
- **STANDING RULE — PowerShell variable-colon hazard:** in double-quoted strings, `"$varname:..."` parses as scope/drive prefix (matches `$env:`, `$global:`, `$script:` family). Use `"${varname}:..."` when followed by a literal colon. Add the brace-delimited form to the standing-rules list of PowerShell hygiene patterns.
- **STANDING RULE — EOF-newline anchor:** multi-line `replace` anchors at or near EOF must not include a terminal `\n`. The anchor without trailing newline matches both `text.` (EOF) and `text.\n[more]` cases via Python's `str.count(old)`. P1.6's failure proved this empirically; the file ends without a trailing newline.
- **STANDING RULE — Move-Item is destructive:** Windows `Move-Item -Force` removes the source after the move. Save procedures must `Test-Path` the source BEFORE removing the destination. Pattern: verify Downloads has the file → only then delete `agent_data\` → then move.
- **STANDING RULE — SHA-256 fingerprint verification:** byte-count alone can miss "downloaded the cached pre-fix version" failures (two file versions can share a byte count by coincidence). Each chat-delivered file should carry a SHA-256 fingerprint the user verifies before save.
- Helper writes with `newline="\n"` for deterministic LF output; Git `core.autocrlf=true` on Windows produces benign `LF will be replaced by CRLF` warnings at staging. Repo content remains LF-normalized; the warnings have no functional impact.
- Architectural state after cleanup arc: GCP project `genomic-variant-prod` permanently destroyed; no remote object storage; data flow is local Windows source-of-truth ↔ Vast.ai GPU scratch (SCP via `id_lambda_run8`) ↔ Drive via rclone `genvarcla:` for agent-layer durability only. `INCIDENT_2026-04-29` is the canonical verification-rule supersession of the 2026-04-17 GCS-receipt rule.

### Commits
- `b15a625` — Commit 1/4: incident formalization (4065 bytes of incident doc, secret deleted)
- `aad8f5a` — Commit 2/4: runtime GCS strip (4 files, +5/-90)
- `feece15` — Commit 3/4: operational docs rewrite (4 files, +62/-62)
- (this commit) — Commit 4/4: session log + CHANGELOG cap

### Refs
- `agent_data/arch_cleanup_stage3_discovery.ps1` (5266 bytes)
- `agent_data/arch_cleanup_stage3_code.py` (21838 bytes; SHA `154884df6e976e1614c43c879e7dd71bbcdb1222ce61f277dd379fdd0b33fc1f`)
- `agent_data/arch_cleanup_stage3_batch.ps1` (8991 bytes; SHA `952daab6457d22c9459c5fe9288030eb9f117c776ba57ac769e8957ecf5c1fae`)
- `agent_data/arch_cleanup_stage4_code.py` (this commit's helper)
- `agent_data/arch_cleanup_stage4_batch.ps1` (this commit's batch)
- Session doc: `docs/sessions/SESSION_2026-05-10_arch-cleanup.md`
- Incident doc: `docs/incidents/INCIDENT_2026-04-29_gcp-billing-deletion.md`

## 2026-05-10 — SpliceAI cache leak fix (path-aware conftest.py)

### Attempted
- Move class-scoped `_isolate_spliceai` fixture from `TestAnnotationPipeline` (test_core.py L2167) to a module-scoped autouse fixture in `tests/unit/conftest.py`, add `_save_cache` patch to plug the 430 MB `data/raw/cache/spliceai_scores_snv.parquet` regeneration leak.

### Failed
- **Attempt 1** (Stage 2 abort, no commit): helper's in-line post-apply check used a loose grep `if "_isolate_spliceai" in final_tc` that false-positived on the NEW class docstring's legitimate cross-reference to the new fixture location. Same-pattern-bug as the batch verification fix moments earlier — fixed one location, missed the identical pattern in the other.
- **Attempt 2** (Stage 3b abort, no commit): fixture's UNCONDITIONAL `_save_cache → no-op` blocked the legitimate cache write in `test_parquet_cache_used_on_second_call`, which uses `FetchConfig(cache_dir=tmp_path / "cache")` — a tmp-scoped cache that does NOT touch the production dir. Test failed `assert score == 0.42 → got 0.0`. Cache mtime UNCHANGED throughout (leak prevention was working; over-blocking was the issue).
- **Pre-check B** (non-fatal): Python helper structural validation via `& python -c @"..."@` errored on `f'{\"X\" if ok else \"Y\"}'` — PS here-strings pass `\"` literally; backslashes inside Python f-string `{expr}` are forbidden. Other pre-checks confirmed file state independently.

### Fixed
- **Attempt 3 commit `a01eef3`**: path-aware fixture design. New `_is_prod_cache_path(cache_path)` helper resolves the cache target and tests `relative_to(_PROD_CACHE_DIR.resolve())`; load/save are blocked only when path resolves under `data/raw/cache/`. tmp_path-scoped FetchConfigs are unaffected and exercise the real load→save→load flow. `_orig_load_cache` and `_orig_save_cache` captured before patch, called for non-prod paths.
- Helper's in-line post-apply check tightened to `def _isolate_spliceai(` (the method definition) instead of the bare name (which legitimately appears in the new docstring's cross-reference).

### Verified
- 16 pytest tests pass in 58.90s (including `test_parquet_cache_used_on_second_call`, the regression test that exposed Attempt 2's over-blocking).
- Cache mtime IDENTICAL pre/post pytest: `04/19/2026 13:56:19`.
- Cache size IDENTICAL pre/post pytest: 451,626,904 bytes.
- CI green on `a01eef3` (4 min runtime).

### Learned
- **Autouse + unconditional patching is dangerous.** Fixtures that null out shared infrastructure must be conditional/path-aware, not blanket no-ops. Cost of over-blocking: silent test failures that look like real bugs.
- **Same-pattern-bug-different-location.** When fixing a pattern, grep the entire change-set for similar instances. Fixing the batch verification but missing the identical helper internal check cost an iteration.
- **CRLF/UTF-8 byte-delta surprises.** Disk byte delta differs from Python char delta by `num_CRLF_lines + 2*multibyte_chars`. Existing `[WARN] -500 to -1500` bounds in the batch are tight; should widen to roughly `python_char_delta − num_lines_with_CRLF + 2*multibyte_char_count` in future batches.
- **PS here-string + Python f-string interaction.** Inside `@"..."@`, `\"` is passed literally; backslashes in Python f-string `{expr}` are syntax errors. Use single quotes inside double-quoted f-strings.

### Commits
- `a01eef3` — `test(spliceai): move _isolate_spliceai fixture to conftest.py and add _save_cache patch to prevent 430 MB cache regeneration`

### Refs
- Helper: `agent_data/spliceai_cache_fix_code.py` (SHA `3ca0cca1cddaea0b0f46ec56be012482dae3fe8448875ad36cdc8b00b36d5d1e`)
- Batch: `agent_data/spliceai_cache_fix_batch.ps1` (SHA `4d7023a9424f9b54a4e4fce0360bde0fa496736a7da1c1051c5bf6ba80a1491e`)
- Session doc: `docs/sessions/SESSION_2026-05-10_spliceai-cache-fix.md`
- New conftest: `tests/unit/conftest.py`
- Prior session (arch cleanup, same day): `docs/sessions/SESSION_2026-05-10_arch-cleanup.md`
## 2026-05-12 — Run 9: 11.4h training on Vast.ai RTX 4090, ensemble.save() crash, no test AUROC

### Attempted
- Launch Run 9 as 6-ablation suite (`full + 5 feature-group ablations`)
  on Vast.ai RTX 4090 (instance 36588175, $0.473/hr).
- Auto-destroy on preflight failure via vastai CLI `cleanup_if_setup_failed`
  trap function in `scripts/launch_run9_vm.sh`.
- Pickle entire fitted ensemble as a single joblib via
  `joblib.dump(self, path)` in `VariantEnsemble.save()`.

### Failed
- **4 failed launch attempts** before successful launch (~10 min debug each):
  - Attempt 1: workflow-aware preflight bugs (ClinVar VCF,
    torch_geometric). Resolved by commits `8a3785a` + `bd75ed5`.
  - Attempt 2: data SCP'd to repo-relative paths
    (`/workspace/genomic-variant-classifier/data/...`) but
    `launch_run9_vm.sh` uses `/workspace/{data,outputs}/` absolute paths.
  - Attempt 3: training script used absolute paths while preflight
    used repo-relative. Operator added symlinks ad-hoc.
  - Attempt 4: `ln -s /workspace/genomic-variant-classifier/data
    /workspace/data` placed symlink INSIDE the existing
    `/workspace/data/` directory created in attempt 3 instead of
    replacing it (silent Unix `ln` behaviour on existing-target).
    `rm -rf` of the destination required before `ln -s`.
- **Auto-destroy broken** in vastai CLI 1.0.12: interactive
  `input()` confirmation fails under `nohup` with `OSError: Bad file
  descriptor`. Manual destroy via Vast.ai web console at
  https://cloud.vast.ai/instances/ after ~9h idle billing.
- **`ensemble.save()` PicklingError** at end of 11.4h training:
  `_CNN1D` defined inside `CNN1DClassifier._build_model.<locals>`
  is not pickle-able. `joblib.dump()` crashed with
  `_pickle.PicklingError: Can't pickle <class
  'genomic_variant_classifier.models.variant_ensemble.CNN1DClassifier._build_model.<locals>._CNN1D'>:
  it's not found as ...<locals>._CNN1D`. Joblib is corrupt; no
  per-model checkpoints exist; locked test AUROC never produced.

### Fixed (this session)
- Workflow-aware preflight (commits `8a3785a` + `bd75ed5`) — landed
  before final launch attempt.
- Path mismatch — manual `mv` data into repo + symlink
  `/workspace/{data,outputs}` → repo paths (workaround; canonical fix
  deferred to Phase 1.5 launch-script unified patch).
- Symlink trap — `rm -rf` before `ln -s` when destination might be
  recreated as directory.

### Drafted (shipped in 2026-05-13 follow-up session as `run10_phase1_v2.zip`)
- Patch A1: `_CNN1D` lifted to module-level `_CNN1DModule` via lazy-
  global with qualname fixup. Fixes pickle.
- Patch A2: `VariantEnsemble.save()` refactored to per-model joblib
  checkpoints (`<ensemble>_models/<model_name>.joblib`) + thin
  orchestrator joblib. Single-model pickle failure no longer poisons
  whole ensemble. `load()` back-compat with legacy single-joblib format.
- Patch A3: `evaluate()` CatBoost dispatch fix (was missing the
  DataFrame branch that `fit`/`predict_proba` correctly include).
- Patch B1: `scripts/run_phase2_eval.py` — added `--lovd-path`,
  `--dbnsfp-path`, `--finngen-path` CLI args + `AnnotationConfig`
  wiring (mirrors `scripts/train.py:167-172`). Closes the
  silent-zero gap for three connectors that were unknowingly absent
  from Run 9 alongside LOVD. Supersedes R10-A of
  `INCIDENT_2026-05-02_lovd-silent-zero.md` (see
  `INCIDENT_2026-05-02_lovd-silent-zero_AMENDMENT.md`).
- Patch B2 + B3: test-set evaluation + OOF parquet + `metrics.json`
  flushed BEFORE `ensemble.save()` so a save crash never loses
  scientific artifacts.
- Regression tests: `tests/unit/test_variant_ensemble_save_load.py`
  (4 tests) + `tests/unit/test_lovd_annotation_reaches_training_matrix.py`
  (2 tests with importskip guard).

### Results
- OOF blend AUROC: **0.9916**
- LR stacker AUROC: 0.9911
- Best single base (lightgbm): 0.9911
- **Δ blend over best single: +0.0005 — within noise floor** pending
  bootstrap CI per `SESSION_2026-05-12.md` Run 10 plan §3.
- No test-set AUROC: script crashed at save before test evaluation
  ran. Phase 1 patch B2 moves test eval before save to prevent
  recurrence.
- **Per-model OOF AUROC table (2026-05-13 partial recovery via
  `scripts/run9_outputs_audit.ps1`):** 8 of 11 base models recovered as
  04-30 proxies (lightgbm 0.9911, xgboost 0.9908, catboost 0.9900,
  gradient_boosting 0.9889, random_forest 0.9881, deep_ensemble 0.9872,
  mc_dropout 0.9870, logistic_regression 0.9849). 4 NOT recoverable:
  svm, kan, tabular_nn, cnn_1d (skipped in 04-30 regen). 11-dim
  Nelder-Mead weight dict NOT recoverable beyond qualitative statement
  (kan/tabular_nn/logistic_regression 0%, cnn_1d ~10%). See
  `INCIDENT_2026-05-12_no-per-model-checkpoint.md` §Recovery status.
- **Scientific finding from proxy comparison:** 04-30 8-model blend
  was 0.9915 vs Run 9 11-model blend 0.9916. Adding 4 models
  (svm/kan/tabular_nn/cnn_1d) moved blend by **+0.0001** — at or below
  noise floor. Supports the §2 keep-all decision being conditional on
  bootstrap CI.

### Scientific implications (preliminary; full analysis in Run 10)
- The 11-model ensemble adds essentially nothing over a single tuned
  lightgbm in OOF blend. Δ=+0.0005 must be confirmed via bootstrap CI
  before any pruning decision.
- KAN (8h compute) received 0% blend weight. Drop candidate for
  Run 10, deferred pending bootstrap CI per SESSION §2 amendment.
- tabular_nn and logistic_regression received 0% blend weight.
- cnn_1d received ~10% blend weight despite OOF AUROC ~0.5 (broken
  signal — fed placeholder sequences per
  `INCIDENT_2026-05-12_cnn1d-pickle-nested-class.md`). Investigate
  whether this generalizes after pickle fix; Sequence Branch
  (real FASTA) wiring deferred to Run 11.
- Standing concern about gene-prevalence + external-score
  memorization remains unresolved.

### Learned (7 new standing rules — see SESSION doc §Learned)
1. Vast.ai SCP destinations must be repo-relative or include explicit
   symlink step in runbook.
2. `vastai destroy ≥1.0.12` is interactive; auto-destroy in scripts
   MUST pipe `yes` or `echo y`.
3. `ln -s` does NOT replace existing real directories; use `rm -rf`
   first when destination may have been recreated between fix attempts.
4. PowerShell strips inner `"..."` from ssh command args — use single
   quotes ONLY inside ssh wrappers, never double quotes.
5. STOP putting bash code inside `ssh ... '<bash>'` from PowerShell.
   Use `@'...'@ | ssh ... bash -s` with `-replace "`r`n", "`n"` to
   strip CRLF.
6. PowerShell `@'...'@` heredocs preserve `\r\n` line endings; always
   `-replace "`r`n", "`n"` before piping to remote bash.
7. Vast.ai 2026 PyTorch images auto-tmux + auto-activate `/venv/main`.
   SCP destinations MUST be inside the cloned repo. Subprocess can
   still use `/usr/local/bin/python` symlinks for non-activated calls.

### Costs
- Instance 36588175, Vast.ai RTX 4090, $0.473/hr
- ~20.5h total wall-clock = **~$9.70**
- ~9h of that was idle post-crash because auto-destroy was broken
- Productive: ~$5.40 | Idle: ~$4.30

### Commits
- `3cfc039` — `docs(session): Run 9 launch, training, pickle crash, results`

### Refs
- Session doc: `docs/sessions/SESSION_2026-05-12.md`
  (amended 2026-05-13 — §2 of Run 10 plan revised to keep-all; OOF
  AUROC/blend-weight placeholders annotated with recovery pointer)
- INCIDENTs (filed in 2026-05-13 follow-up session):
  - `docs/incidents/INCIDENT_2026-05-12_cnn1d-pickle-nested-class.md`
  - `docs/incidents/INCIDENT_2026-05-12_vastai-destroy-interactive.md`
  - `docs/incidents/INCIDENT_2026-05-12_launch-path-inconsistency.md`
  - `docs/incidents/INCIDENT_2026-05-12_no-per-model-checkpoint.md`
- LOVD INCIDENT 2026-05-13 amendment: launch-script wiring gap
  identified as actual root cause; supersedes Cause 1 + Cause 2
  candidates. See `INCIDENT_2026-05-02_lovd-silent-zero.md`
  §"2026-05-13 Update".
- Phase 1 patch bundle: `run10_phase1_v2.zip` (shipped 2026-05-13)
- Run 9 outputs audit: `scripts/run9_outputs_audit.ps1` (placed
  2026-05-13)

# Phase 1.5b CHANGELOG entry

Append this block to `docs/CHANGELOG.md` (after the existing
`## 2026-05-12 — Run 9:` entry).

---

## 2026-05-13 (post-1.5) — Phase 1.5b: test fixes + FinnGen wiring correction

### Test fixes — commit 66593d6 shipped 2 broken tests

The Phase 1 patch bundle (`run10_phase1_v2.zip`, commit 66593d6) shipped 4
regression tests with 2 sandbox-only assumptions that broke under production
pytest:

**1.** `tests/unit/test_variant_ensemble_save_load.py::test_ensemble_save_creates_per_model_checkpoints`
and `::test_ensemble_load_roundtrip` called `ens.fit_minimal(X_tab, X_seq, y)` —
a helper method that exists in Claude's sandbox draft but was never shipped to
production `variant_ensemble.py`.

```
AttributeError: 'VariantEnsemble' object has no attribute 'fit_minimal'
```

**Fix (1.5b):** rewritten as one consolidated test `test_ensemble_save_load_with_cnn1d`
that restricts `ens.base_estimators` to `{"lightgbm", "cnn_1d"}` BEFORE
calling `ens.fit()`, then exercises the full save/load/predict_proba round
trip on a 60-row balanced synthetic dataset. CNN1D is in the restricted set
specifically to exercise the A1 pickle-fix code path.

**2.** `tests/unit/test_lovd_annotation_reaches_training_matrix.py::test_lovd_annotation_reaches_training_matrix`
and `::test_lovd_annotation_silent_zero_when_path_omitted` used a 5-row gene
fixture (TP53×2, GENE_X, BRCA2, APC) that `GroupShuffleSplit` cannot partition
into class-balanced train/val/test splits.

```
ValueError: Gene-aware split 'train' missing class(es): {np.int64(1)}.
Try lowering min_review_tier or increasing dataset size.
```

**Fix (1.5b):** added `require_both_classes=False` to both tests' `DataPrepConfig`.
The class-balance constraint is for production training; the LOVD column-
propagation check these tests target doesn't need it.

Tests 1 and 2 (`test_cnn1d_module_class_is_module_level` and
`test_cnn1d_pickles_after_fit`) passed in production unchanged. Those tests
directly validate the A1 pickle fix and remain the most important regression
guards.

### FinnGen wiring — commit 66593d6 message was incorrect

The 66593d6 commit message stated:

> NOTE: FinnGen wiring is partial. B1 sets AnnotationConfig.finngen_path
> but real_data_prep.py annotate chain does not invoke FinnGenConnector
> (regen.log shows no FinnGen step). Phase 1.6 will add the connector
> invocation. LOVD and DbNSFP are fully fixed.

This is **incorrect** and was based on a false inference that "no FinnGen
entries in regen.log" implied "FinnGen connector not wired". Empirical
verification on 2026-05-13 via direct grep of `src/genomic_variant_classifier/data/real_data_prep.py`:

```
185:    finngen_path: Optional[Path] = None  # FinnGen R10 annotated variants TSV
418:    # FinnGen R10: third-tier AF fallback after gnomAD and 1KGP
419:    if self.annotation_config.finngen_path:
420:        from genomic_variant_classifier.data.finngen import FinnGenConnector
422:        finngen = FinnGenConnector(tsv_path=self.annotation_config.finngen_path)
423:        df = finngen.annotate(df)
425:    else:
427:        for col in FINNGEN_COLUMNS:
430:        df["finngen_enrichment"] = 1.0
```

**Phase 1 B1 IS sufficient for FinnGen.** Passing `--finngen-path` to
`scripts/run_phase2_eval.py` sets `AnnotationConfig.finngen_path`, which
satisfies the line 419 conditional and invokes `FinnGenConnector.annotate()`
at line 422. Same fix shape as LOVD and DbNSFP.

The reason no FinnGen entries appear in Run 9's `outputs/run9_ready/regen.log`
is **NOT** a wiring gap — it's that the `else` branch at line 425-430 silently
fills defaults (`finngen_af_fin=0`, `finngen_af_nfsee=0`, `finngen_enrichment=1`)
with **no log emission at all**. This is a *worse* silent-zero pattern than
LOVD or DbNSFP (which at least emit a WARNING that audit greps catch).

FinnGen is wired into the **AF-fallback** stage (line ~418, third tier after
gnomAD and 1KGP) — NOT into the **score-annotation** stage (line 504+). The
"Score annotation N/M" log series covers the 17 score connectors only.
That's why `Select-String "Score annotation"` against `real_data_prep.py`
shows 17 score steps with FinnGen absent — that absence is structural, not
a bug.

**Phase 1.6 follow-up (deferred, optional):** add an `INFO` log to the
FinnGen `else` branch so silent-zero is detectable in `regen.log` audits.
Small code-hygiene patch, can ride with `sequence_context.py` stub work.

### Phase 1 commit message accuracy

The 66593d6 commit message will remain as-is (git history rewrite not worth
the risk on `main`). The correction lives here and will be referenced by any
future audit. Future commit messages should phrase FinnGen as "fully wired"
alongside LOVD and DbNSFP.

