# SESSION 2026-05-10 — Architectural cleanup: GCS retirement

**Scope:** complete the SCP-only architectural pivot. Four ordered commits, three diagnostic failures, all caught at dry-run / parse-time / save-time before writes. Repo state preserved at every failure boundary.

**Prior state:** GCP project `genomic-variant-prod` deleted 2026-04-29. Repository still referenced GCS in runtime code (4 files) and operational documentation (4 files).

**Final state:** operational doc set is SCP-aligned; zero `gs://`, `gcloud storage`, `gcloud auth`, or `genomic-variant-prod` references in the 4 patched docs (post-commit verified).

## The four commits

### Commit 1/4 — `b15a625` — Incident formalization
`docs(incident): formalize 2026-04-29 GCP project deletion + SCP-only architectural pivot`

- Created `docs/incidents/INCIDENT_2026-04-29_gcp-billing-deletion.md` (4065 bytes) documenting the architectural pivot and the verification-rule supersession (2026-04-17 GCS-receipt rule → INCIDENT_2026-04-29 SCP-only rule).
- Deleted stale `secrets/gcp-sa-key.json`.
- CI: green (run 25638482305).

### Commit 2/4 — `aad8f5a` — Runtime GCS strip
`chore(arch): strip GCS from active runtime code (Commit 2/4 of cleanup)`

4 files, 5 insertions, 90 deletions:
- `scripts/preflight_check.py` (~42 lines): removed `gcloud auth` block and `--skip-gcs` CLI flag.
- `src/genomic_variant_classifier/evaluation/prediction_artifacts.py` (~47 lines): removed `upload_to_gcs()` method.
- `src/genomic_variant_classifier/agent_layer/config.py` (~3 lines): removed GCS bucket configuration.
- `src/genomic_variant_classifier/agent_layer/test_message_bus.py` (~2 lines): removed GCS-mode pytest assertions.

Verification: live `upload_to_gcs` callers post-strip: 0. Preflight goes from 11 PASS/0 FAIL to 10 PASS/1 FAIL (only the SpliceAI cache leak remains).

CI: green (run 25640716959).

### Commit 3/4 — `feece15` — Operational docs rewrite
`docs(arch): rewrite operational docs for SCP-only architecture (Commit 3/4 of cleanup)`

4 files, 20 atomic patches, 30 GCS hit-lines removed: balanced 62/62 rewrite.

**Per-file breakdown:**
- `scripts/run9_launch.md` (11 hits, 6 patches P1.1-P1.6): `--skip-gcs` flag block, "Pull data from GCS" block, save+verify GCS shutdown block, deliverables receipt line, INCIDENT template confirmation line, closing INCIDENT-verification paragraph.
- `docs/HANDOFF_run9_launch.md` (2 hits, 2 patches P2.1-P2.2): upload+verify+destroy bullet, shutdown-ordering + verification-rule standing-rules bullets.
- `docs/RUN9_OPERATIONS_PLAYBOOK.md` (9 hits, 5 patches P3.1-P3.5): expected-preflight-output paragraph, step 7c+7d block (GCS auth → SCP key check), upload bullet, step 9a pull artefacts, failure-path INCIDENT template.
- `docs/RUN9_SCIENTIFIC_DESIGN.md` (8 hits, 7 patches P4.1-P4.7): GCS SpliceAI parquet table row, `upload_to_gcs()` example + design rules paragraph, preflight expected output comment, shutdown trap upload block, B4 input data pull GCS → verify pre-staged, C4 upload+shutdown, D1 pull artefacts via SCP.

All 4 files post-patch GCS hit count: 0 (verified). Method `upload_to_gcs()` references in code examples replaced with the SCP-back workflow narrative.

CI: green (run 25642115566).

### Commit 4/4 — (this commit) — Session log + CHANGELOG cap
`docs(session): 2026-05-10 architectural cleanup session log + CHANGELOG cap (Commit 4/4)`

- Created this session doc.
- Appended new 2026-05-10 entry to `docs/CHANGELOG.md`.

## Three diagnostic failures (all caught before writes)

The Stage 3 batch v1 (8987 bytes) and helper v1 (21493 bytes) failed in three distinct ways across three rounds. The safety net (parse-time check → dry-run check → SHA verification → repo state preservation) caught every failure cleanly with zero writes.

### Failure 1: PowerShell `$p:` parser hazard (parse-time)

**Symptom (batch v1, 8987 bytes):**

```
ParserError: arch_cleanup_stage3_batch.ps1:162:33
 if ($hit) { "    [OK]   $p: '$probe' present" }
 | Variable reference is not valid. ':' was not followed by a valid variable name character.
```

**Root cause:** PowerShell double-quoted string parser greedily extends `$name` to `$name:` because `name:` matches the scope-prefix family (`$env:`, `$global:`, `$script:`). When the character after `:` is not a valid name-character, the parser fails.

**Fix (batch v2, 8991 bytes):** wrap the variable in braces — `${p}:` — to terminate the variable name explicitly. Two lines updated in the batch's post-edit verification section.

**Generalization:** added to standing rules as "PowerShell variable-colon hazard."

### Failure 2: P1.6 EOF-newline anchor mismatch (dry-run-time)

**Symptom (helper v1, 21493 bytes):**

```
[FAIL] scripts/run9_launch.md :: P1.6 run9_launch L200-201: closing INCIDENT-verification paragraph
       anchor not found
       anchor (first 80 chars): 'The 2026-04-17 INCIDENT-verification standing rule applies: no\n"RESOLVED" withou'
```

**Root cause:** P1.6 anchor included a trailing `\n` after `receipt.`, but `scripts/run9_launch.md` ends at L201 with no final newline.

**Fix (helper v2, 21838 bytes):** removed terminal `\n` from both P1.6 anchor and replacement. The anchor without trailing newline matches both `receipt.` (EOF) and `receipt.\n[more content]` cases via `text.count(old)`.

**Generalization:** added to standing rules as "EOF-newline anchor."

### Failure 3: Move-Item silent source-consumption (save-time)

**Symptom (during re-download after batch v1 abort):** user attempted to re-download the fixed batch. `Move-Item -Force` reported `Cannot move item because the item at '~\Downloads\arch_cleanup_stage3_batch.ps1' does not exist.` `agent_data\` ended up with no batch at all.

**Root cause:** Windows `Move-Item` removes the source after a successful move. A `Remove-Item agent_data\<file>` + `Move-Item Downloads\<file> → agent_data\` sequence works exactly once. If repeated, the second iteration deletes the destination but finds no source to move.

**Fix:** defensive save procedure that `Test-Path`s the source FIRST and aborts loudly if absent. Pattern: check `Downloads` has both files → only then remove from `agent_data\` → then move.

**Generalization:** added to standing rules as "Move-Item is destructive."

## What this run produced

**Files modified in the architectural-cleanup arc (commits 1-3):**
- `docs/incidents/INCIDENT_2026-04-29_gcp-billing-deletion.md` (created, 4065 bytes)
- `secrets/gcp-sa-key.json` (deleted)
- `scripts/preflight_check.py` (~42 lines removed)
- `src/genomic_variant_classifier/evaluation/prediction_artifacts.py` (~47 lines removed)
- `src/genomic_variant_classifier/agent_layer/config.py` (~3 lines modified)
- `src/genomic_variant_classifier/agent_layer/test_message_bus.py` (~2 lines modified)
- `scripts/run9_launch.md` (37 lines diffed)
- `docs/HANDOFF_run9_launch.md` (8 lines diffed)
- `docs/RUN9_OPERATIONS_PLAYBOOK.md` (39 lines diffed)
- `docs/RUN9_SCIENTIFIC_DESIGN.md` (40 lines diffed)

**Files modified in this commit (4/4):**
- `docs/CHANGELOG.md` (appended new 2026-05-10 entry, ~4500 bytes)
- `docs/sessions/SESSION_2026-05-10_arch-cleanup.md` (this file, created)

**Total GCS hits across operational doc set:** 30 → 0 (verified).

## Next steps

**Stage 5 (post-Commit 4/4):** memory updates via `memory_user_edits` (no batch needed). Four GCS-stale entries to replace with SCP-only equivalents:

1. `gs://genomic-variant-prod-outputs/` references in compute/storage memory.
2. `gcloud-storage-ls` verification rule in standing rules.
3. SpliceAI parquet GCS-mirror entry.
4. Agent-layer GCS claim.

Plus three new standing rules learned today:

1. PowerShell variable-colon hazard (`${var}:`).
2. EOF-newline anchor pattern.
3. Move-Item destructiveness.

Plus the SHA-256 fingerprint verification pattern for chat-delivered files.

**Run 9 launch sequence:** the operational docs are now SCP-aligned and the SCP-back procedure is documented end-to-end in `scripts/run9_launch.md` and `docs/RUN9_OPERATIONS_PLAYBOOK.md`. The pre-launch checklist's only remaining blocker is the SpliceAI fixture leak (blocker (b)) — 430.7 MB stale cache at `data\raw\cache\spliceai_scores_snv.parquet`. Fix: add a module-scoped autouse `_isolate_spliceai` fixture to `tests/unit/conftest.py`.

## References

- Stage 3 helper: `agent_data/arch_cleanup_stage3_code.py` (21838 bytes; SHA `154884df6e976e1614c43c879e7dd71bbcdb1222ce61f277dd379fdd0b33fc1f`)
- Stage 3 batch: `agent_data/arch_cleanup_stage3_batch.ps1` (8991 bytes; SHA `952daab6457d22c9459c5fe9288030eb9f117c776ba57ac769e8957ecf5c1fae`)
- Stage 3 discovery: `agent_data/arch_cleanup_stage3_discovery.ps1` (5266 bytes)
- Stage 3 discovery output: `agent_data/arch_cleanup_stage3_discovery.out.txt` (36124 bytes; 30 hits across 4 files)
- Stage 4 helper: `agent_data/arch_cleanup_stage4_code.py`
- Stage 4 batch: `agent_data/arch_cleanup_stage4_batch.ps1`
- Incident: `docs/incidents/INCIDENT_2026-04-29_gcp-billing-deletion.md`
- Prior session: `docs/sessions/SESSION_2026-05-09_C5.md`
