# SESSION 2026-05-10 — SpliceAI cache leak fix (path-aware conftest.py)

**Scope:** clear the last Run 9 launch blocker by fixing the pre-Run-9 SpliceAI cache regeneration leak. Four iterations, each caught at a different verification gate (helper post-apply check, pytest verification, or precise diagnostic) before any bad state could reach origin. Final design is **path-aware**: blocks BaseConnector cache I/O only when the cache target resolves under `data/raw/cache/`, leaving `tmp_path`-scoped FetchConfigs untouched.

**Pre-state:** post-arch-cleanup at `34eaf98`, working tree clean, CI green. The 430 MB `data/raw/cache/spliceai_scores_snv.parquet` (44,688,180 rows, `lookup_key` + 5 float32 score columns) had been observed regenerating during prior pytest runs of `TestSpliceAIConnector`.

## Root cause (confirmed via direct source inspection)

`database_connectors.py` revealed the exact mechanism:

- L64: `FetchConfig.cache_dir` defaults to `Path("data/raw/cache")`
- L72: `FetchConfig.__post_init__` runs `cache_dir.mkdir(parents=True, exist_ok=True)`
- L107: `_cache_path(key)` returns `self.config.cache_dir / f"{self.source_name}_{safe_key}.parquet"`
- L116: `BaseConnector._save_cache` writes via `df.to_parquet(self._cache_path(key))`
- L217: `BaseConnector.fetch()` ends with `self._save_cache(cache_key, result)` — **unconditional**

For `SpliceAIConnector` (`source_name = "spliceai"`) with `cache_key = "scores_snv"`, that path resolves to exactly the 430 MB leak file. The pre-existing class-scoped `_isolate_spliceai` fixture in `TestAnnotationPipeline` patched `_load_cache` but NOT `_save_cache`, so the stub-zero result still got written. Additionally, `TestSpliceAIConnector` (73 SpliceAI test hits in test_core.py, sibling class) had no fixture coverage at all.

## Iterations

### Attempt 1 — Move fixture to conftest.py, add `_save_cache` patch (FAILED at helper post-apply check)

Created `tests/unit/conftest.py` with module-scoped autouse `_isolate_spliceai` patching `DEFAULT_SPLICEAI_PATH`, `_load_cache`, and `_save_cache`. Substitution worked correctly — `git diff` confirmed clean fixture removal + new docstring — but the helper's in-line post-apply check used a loose grep:

```python
if "_isolate_spliceai" in final_tc:
    return 1
```

This false-positived on the NEW docstring's legitimate cross-reference (`"SpliceAI isolation is provided by ... _isolate_spliceai in tests/unit/conftest.py"`). Helper exited 1, batch aborted at Stage 2 before commit. Same-pattern-bug-different-location: I had fixed the precise check in the batch verification (`def _isolate_spliceai(`) but forgot to fix the helper's identical internal check.

### Attempt 2 — Fix the helper's loose check (FAILED at pytest verification)

Updated helper's post-apply check to `def _isolate_spliceai(`. Batch progressed cleanly through Stages 1, 2, 3 — substitution applied, conftest.py created, OLD fixture removed. Stage 3b (pytest verification) then aborted with:

```
FAILED tests/unit/test_core.py::TestSpliceAIConnector::test_parquet_cache_used_on_second_call
  Expected: 0.42  Obtained: 0.0
WARNING: SpliceAI VCF not found at 'C:\Users\...\test.vcf.gz'
```

The fixture's unconditional `_save_cache → no-op` blocked the legitimate cache write that `test_parquet_cache_used_on_second_call` depends on. That test uses `FetchConfig(cache_dir=tmp_path / "cache")` — a tmp-scoped cache that does NOT touch the production cache dir. My fixture conflated two distinct cases: writes to production `data/raw/cache/` (the actual leak — must block) and writes to `tmp_path` caches (legitimate test behavior — must work). Treating them identically broke the legitimate one.

Cache mtime UNCHANGED throughout: leak prevention was working; the over-blocking was the only issue.

### Attempt 3 — Path-aware design (SUCCESS)

Updated `CONFTEST_CONTENT` and `NEW_FIXTURE`. Path-aware design:

```python
def _is_prod_cache_path(cache_path: Path) -> bool:
    try:
        cache_path.resolve().relative_to(_PROD_CACHE_DIR.resolve())
        return True
    except ValueError:
        return False

def _safe_load_cache(self, key):
    if _is_prod_cache_path(self._cache_path(key)):
        return None
    return _orig_load_cache(self, key)

def _safe_save_cache(self, key, df):
    if _is_prod_cache_path(self._cache_path(key)):
        return None
    return _orig_save_cache(self, key, df)
```

Production cache reads/writes blocked; tmp_path caches transparent. Batch ran end-to-end:

- Stage 1 (dry-run): preview correct, char delta -695
- Stage 2 (apply): both files written; in-helper post-apply verification passed
- Stage 3 (file verification): all OK; size-delta WARN -3063 (harmless — CRLF→LF normalization + UTF-8 multi-byte chars from existing em-dashes in test_core.py)
- Stage 3b (pytest): **16 passed in 58.90s**; cache mtime + size identical before and after
- Stage 4 (commit + push): created `a01eef3`, pushed cleanly
- CI poll: `a01eef3|completed|success` at 20:31:20 (~4 min CI runtime)

## Final state

- **Commit:** `a01eef3` — `test(spliceai): move _isolate_spliceai fixture to conftest.py and add _save_cache patch to prevent 430 MB cache regeneration`
- **Files changed:** 2 (`tests/unit/conftest.py` created, `tests/unit/test_core.py` class-scoped fixture removed and class docstring replaced)
- **Diff stat:** +93 / -19 lines
- **Cache mtime:** `04/19/2026 13:56:19` (unchanged throughout all four iterations — leak never occurred)
- **Cache size:** 451,626,904 bytes (unchanged)
- **CI:** green on `a01eef3`

## Lessons

### Test fixture design — autouse + unconditional patching is dangerous

An autouse fixture that nulls out shared infrastructure (cache load/save, DB sessions, HTTP mocks) MUST be conditional or path-aware. Unconditional no-ops break tests that legitimately exercise the infrastructure via tmp-scoped resources. Use `Path.resolve().relative_to()` to detect production paths and block only those. The cost of over-blocking is silent test failures that look like real bugs but aren't.

### Verification chains catch errors at the right gate

Each iteration failed at a DIFFERENT gate:
1. Helper post-apply check (false-positive grep) → caught BEFORE commit
2. Pytest verification (test broken) → caught BEFORE commit
3. None — success → committed

The 4-stage batch (dry-run / apply / file-verify / pytest / commit) never allowed bad state to reach origin. Standing rule (#30 in memory) on apply-batch hygiene + the new pytest verification step both contributed.

### Same-pattern-bug-different-location

The Iteration 1 failure (loose grep) was the SAME class of bug I had just fixed in the batch verification minutes earlier. I propagated the fix to one location and missed the identical pattern in another. **Lesson:** when fixing a pattern, search the entire change-set for similar instances. Memory item already captured.

### CRLF/UTF-8 byte-delta surprises

PowerShell's `Get-Item.Length` reports disk byte size (CRLF on this repo). Python's `len(str)` reports character count after universal-newline normalization. The two differ by:
- Number of `\r` chars stripped on read (~3000 for a 2000-line file)
- UTF-8 multi-byte char overhead (~2 extra bytes per em-dash or smart quote)

The `[WARN]` at Stage 3 (size delta -3063 vs expected -500 to -1500) was harmless but should widen the bound in future batches: realistic range is approximately `python_char_delta − num_lines_with_CRLF + 2*multibyte_char_count`.

### PowerShell here-string + Python f-string interaction

Pre-check B used `& python -c @"...f'{\"X\" if ok else \"Y\"}'..."@`. PS here-strings pass content literally; `\"` is not interpreted. Python sees `\"` inside an f-string `{expr}`, which is a syntax error (backslashes forbidden in f-expressions). **Fix:** use single quotes inside double-quoted f-strings: `f"{'X' if ok else 'Y'}"`. Captured as memory hygiene item #9.

## Refs

- Commit: `a01eef3`
- Helper: `agent_data/spliceai_cache_fix_code.py` (path-aware, SHA `3ca0cca1cddaea0b0f46ec56be012482dae3fe8448875ad36cdc8b00b36d5d1e`)
- Batch: `agent_data/spliceai_cache_fix_batch.ps1` (unchanged across iterations, SHA `4d7023a9424f9b54a4e4fce0360bde0fa496736a7da1c1051c5bf6ba80a1491e`)
- Batch output: `agent_data/spliceai_cache_fix_batch.out.txt`
- New conftest: `tests/unit/conftest.py`
- Regression test that exposed the over-blocking: `tests/unit/test_core.py::TestSpliceAIConnector::test_parquet_cache_used_on_second_call`
- Prior session: `docs/sessions/SESSION_2026-05-10_arch-cleanup.md`

## What's next

**Run 9 launch blockers are fully cleared.** Next session is the launch itself:

1. Exhaustive pre-flight: verify outputs/run9_ready/ contents, repo state, `.venv312` packages, Vast.ai SSH key, billing
2. Provision Vast.ai (RTX 4090, 60 GB RAM, 200 GB disk, CUDA 12.x)
3. SCP `outputs/run9_ready/` + repo up (~1.0-1.2 GB)
4. Run training (CatBoost + LightGBM + XGBoost + GNN + meta-learner + 1D-CNN + TabularNN + KAN ensemble)
5. SCP results back
6. **Destroy instance immediately** upon completion

Standing rule (memory #4): no run begins until all pre-flight items confirmed.
