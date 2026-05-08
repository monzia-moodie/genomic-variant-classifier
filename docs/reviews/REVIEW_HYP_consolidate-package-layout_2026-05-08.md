---
id: REVIEW_HYP_consolidate-package-layout_2026-05-08
title: "Pre-commit review: HYP_consolidate-package-layout.md (revision 2)"
status: draft
date: 2026-05-08
---

# Pre-Commit Review: `HYP_consolidate-package-layout.md` (revision 2)

> **Revision history**
>
> - **rev 1 (superseded):** Initial review based on the assertion that `src/__init__.py` was empty. That assertion was incorrect.
> - **rev 2 (this version):** User supplied the actual `src/__init__.py` content (4-line module: docstring, `__version__`, `__author__`, plus `from src.utils.helpers import resolve_data_dir  # noqa: F401`). User also ran a broader empirical grep covering `from src import …`, bare `import src`, attribute access on `src`, and the `import src.X.Y as <alias>` form. Findings:
>   - rev 1 §A1 (claim that `src/__init__.py` is empty) — **retracted**.
>   - rev 1 §A8 (claim that the `git rm src/__init__.py` rationale was wrong) — **retracted**.
>   - rev 1 §F-1 (patch text rewriting the pre-condition row to "empty") — **retracted**.
>   - rev 1's `src/__init__.py` finding is replaced by an empirical observation that the file's re-exports have **zero consumers**, so the C1 drop is safe. The spec's hedge ("likely no longer load-bearing") can be tightened to a definitive statement.
>   - **New high-severity finding** added as rev 2 §A1: the spec's import-rewrite regex misses 14 production lines that use the `import src.X.Y as <alias>` form. These would silently survive C1 and break pytest with `ModuleNotFoundError: No module named 'src'`. Empirical evidence in §A1 below.

**Reviewer:** Research agent (commissioned by Monzia Moodie)
**Date:** 2026-05-08
**Document under review:** `docs/hypotheses/HYP_consolidate-package-layout.md` (draft, not yet committed)
**Repo HEAD at review time:** `b74daf1` on `main`
**Disposition:** **DO NOT COMMIT AS-IS.** Several factual issues, one technically-suspect core mechanism (pickle re-namespacing), and one empirically-confirmed regex sweep gap must be reconciled first. None of the issues are fatal to the migration concept — all are addressable with the patches in §F. Estimated additional spec authoring time: ~45–75 min before the spec is safe to commit and execute.

---

## Executive summary

The spec describes a five-commit migration (C1–C5) that consolidates two parallel top-level Python namespaces (`src.*`, `agent_layer.*`) into a single `genomic_variant_classifier` package rooted under `src/` (PEP 621 src-layout). The architectural target is correct and well-motivated. However:

1. **The spec's regex sweep silently misses 14 production lines.** Empirical grep at HEAD `b74daf1` confirms 14 lines across `tests/unit/test_api.py` and `tests/unit/test_core.py` use the form `import src.X.Y as <alias>`, which the spec's `from src.X.` family of patterns does not match. Without the §F-5 patch, pytest will fail with `ModuleNotFoundError: No module named 'src'` after C1 lands. **This is the single most concrete spec defect.**
2. **The C4 pickle-migration mechanism is under-specified in a way that will silently fail on nested classes** unless every old-namespace submodule is individually aliased in `sys.modules` — the spec only aliases `src` and `agent_layer` at the top level (§D-1).
3. **C4's "skipped (rename-safe)" classification of `experiments/2026-04-04_03-39/ensemble_v1.joblib` is wrong** — it contains `src.models.variant_ensemble` references identical to `models/v1/ensemble_v1.joblib` and either needs migration or explicit historical-artifact treatment (§A2).
4. **The `pyproject.toml` declares no dependencies and no extras**, silently breaking any current/future `pip install -e ".[dev]"` or `pip install -e ".[api]"` workflow that the multi-lockfile structure (`requirements-api.lock`, `requirements-dev.lock`) implies (§A4).
5. **The stale `genomic_variant_classifier.egg-info/` at repo root must be deleted *before* C2's reinstall**, not as a "Phase 4 cheap rider" (§A5).
6. **The commit choreography leaves the working copy unimportable between C1 and C2** — defensible, but should be acknowledged explicitly with a mitigation.

**On `src/__init__.py`:** The spec correctly identifies that line 8 contains `from src.utils.helpers import resolve_data_dir  # noqa: F401`. The empirical grep at HEAD `b74daf1` (broadened to cover `from src import …`, bare `import src`, and `src.resolve_data_dir`/`src.__version__`/`src.__author__` attribute access) confirms **zero consumers** of these re-exports. The C1 instruction "drop this line entirely" can therefore be made definitive rather than hedged — `git rm src/__init__.py` is safe with no ripple. See §F-1 for the spec wording change.

---

## A. Spec error inventory (factual claims that are wrong or unverifiable)

| # | Spec claim | Reality | Severity |
|---|---|---|---|
| **A1 [NEW, rev 2]** | C1 import-rewrite map enumerates only `from src.X.Y` patterns and a `from src.` catch-all. Spec implies the sweep is exhaustive. | Empirical grep at HEAD `b74daf1` returns 14 lines using `import src.X.Y as <alias>`: 13 in `tests/unit/test_api.py` (8× `import src.api.main as api_main` at lines 490/515/610/660/681/696/725/741, 5× `import src.api.auth as auth_module` at lines 775/786/797/808/819) and 1 in `tests/unit/test_core.py` line 878 (`import src.data.dbnsfp as m`). The spec's regex set does not handle the `import X.Y` form. After C1 runs as written, these 14 lines survive untouched, and pytest fails with `ModuleNotFoundError: No module named 'src'`. | **HIGH** — empirically confirmed silent breakage |
| **A2** | C4 table marks `experiments/2026-04-04_03-39/ensemble_v1.joblib` as **"Skipped (rename-safe)"** | Per the user's screen capture, this file is byte-identical (SHA256 match) to `models/v1/ensemble_v1.joblib` (both 1344.55 MB), which the spec itself classifies as containing `src.models.variant_ensemble`. It is **not** rename-safe; "rename-safe" is reserved for pure stdlib/sklearn/numpy pickles. | **HIGH** — irreversible data hazard if user follows spec literally |
| **A3** | Pre-condition #6: "experiments/2026-04-04_03-39/ensemble_v1.joblib is byte-identical to models/v1/ensemble_v1.joblib" — given as a *current* assertion | True at HEAD `b74daf1`, but **becomes false the moment C4 rewrites `models/v1/ensemble_v1.joblib`**. The spec doesn't address what happens to the experiments copy after C4. | MEDIUM — invariant silently broken mid-migration |
| **A4** | Pyproject claim: "dependencies remain in requirements.in / requirements.lock per pip-tools workflow" — implies that's the *only* lockfile family | Repo also has `requirements-api.in/.lock/.txt` and `requirements-dev.in/.lock`. The pyproject lacks `[project.optional-dependencies]` for `api` / `dev` extras; any existing `pip install -e ".[api]"` invocation in deploy scripts or docs will break. | MEDIUM (silent breakage) |
| **A5** | "`genomic_variant_classifier.egg-info/` will be regenerated cleanly by the next `pip install -e .`" — implicitly assumes deleting it as a Phase-4 rider is safe | Per pip issue #6048 and setuptools docs: when multiple `*.egg-info` directories exist, **pip silently picks one in lexical order**, and the existing `top_level.txt` from the prior install state can pollute the import namespace until rebuild completes. The existing egg-info **must be deleted before** C2's `pip install -e .`. | **HIGH** — install ordering hazard |
| **A6** | C5 description says ".pyc 3.14 cleanup" — the project is on **Python 3.12.10** | The ".pyc 3.14" phrasing is wrong; 3.14 broke scipy+torch and is not in use. The cleanup target is presumably stale `.pyc` files cached under `__pycache__` directories with old `src.*` / `agent_layer.*` module hashes. | LOW (cosmetic), but suggests authoring drift |
| **A7** | Smoke test `EXPECTED_IMPORTS` includes `genomic_variant_classifier.reports.report_generator` | The repo has BOTH a top-level `reports/` directory AND `src/reports/` per the spec's own enumeration. Whether `src/reports/report_generator.py` exists has not been verified in the spec; smoke test could fail for legitimate "file does not exist" reasons rather than packaging reasons. | MEDIUM — gates won't differentiate signal from noise |
| **A8** | Smoke test list includes `genomic_variant_classifier.data.cadd`, `…data.finngen`, etc. | Some of these may be aspirational stubs. The spec does not cross-check `EXPECTED_IMPORTS` against `src/` ground truth. | MEDIUM |
| **A9** | "string_graph_700.pkl is rename-safe (NetworkX, no project classes)" | `nx.read_gpickle` was **removed in NetworkX 3.0** (per official migration guide). If the loading site uses `nx.read_gpickle`, it's already broken regardless of this migration; if it uses raw `pickle.load`, the spec's claim is correct. The spec should either (a) verify the load-site code, or (b) note this is out-of-scope for the migration but flag it as a separate hygiene item. | LOW (orthogonal but worth noting) |
| **A10** | Spec implies all 58 tracked files in `src/` migrate cleanly via `git mv` | Per Git semantics, `git mv` of a directory moves all tracked files inside. **Untracked files in `src/*/`** (e.g., locally generated artifacts, ad-hoc notes) will NOT be moved and may end up orphaned at the old path. Spec should add a `git status` cleanliness gate. | MEDIUM |
| **A11** | "Pre-condition: monitoring/ stub deleted in Phase 0" | Confirmed correct by user. No action. | — |
| **A12** | "agent_layer has 29 tracked files" | Confirmed by user. No action. | — |

**Retracted from rev 1:**

- **rev 1 §A1** (claim that `src/__init__.py` is empty) — RETRACTED. The user supplied the actual file contents; the spec's pre-condition that line 8 contains `from src.utils.helpers import resolve_data_dir  # noqa: F401` is correct.
- **rev 1 §A8** (claim that `git rm src/__init__.py` rationale was wrong) — RETRACTED. The action is correct; the rationale ("dead re-export with zero consumers") is now empirically supported (see pre-flight check 4 in §G).

---

## B. Spec gap inventory (things the spec fails to address)

Organized by commit.

### Pre-C1 (preparation)

- **B-pre-1.** No mandatory `git status --porcelain` clean-tree check before starting. Implicit assumption only.
- **B-pre-2.** No baseline capture of `pip list --format=freeze` and `python -c "import sys; print(sys.path)"` for the active `.venv312`, useful for forensic diff if C2 install regresses environment.
- **B-pre-3.** No pre-condition checking which Python version is active (must be 3.12.x; 3.14 is known broken). Add a guard.
- **B-pre-4.** No check that `genomic_variant_classifier.egg-info/` is the **only** stale egg-info at root; if an older `src.egg-info/` or `agent_layer.egg-info/` exists, all must be removed.
- **B-pre-5.** No pre-flight import check that the *current* layout actually imports cleanly under the current `setup.py` before changing anything.
- **B-pre-6 [NEW, rev 2].** No empirical verification step that `src/__init__.py` re-exports have zero consumers. The §G pre-flight script now includes this.

### C1 (filesystem + git mv)

- **B-C1-1.** No handling of edge cases where `src/<subpkg>/` is missing `__init__.py` (PEP 420 namespace package directories).
- **B-C1-2.** The `tests/tests/` apparent layout glitch visible in the screen capture is not addressed at all.
- **B-C1-3.** No update to `tests/conftest.py` (if present) or `tests/__init__.py` for the new namespace.
- **B-C1-4.** No instruction to update `genomic-variant-classifier.code-workspace` (VS Code workspace file).
- **B-C1-5.** No instruction to update `Dockerfile` or `docker-compose.yml`.
- **B-C1-6.** No handling of `.github/workflows/*.yml` beyond `drift_monitor.yml`.
- **B-C1-7.** No handling of `.pre-commit-config.yaml` and the user's `check_run_id_trailer.py` hook.
- **B-C1-8.** No mention of `pytest.ini`, `mypy.ini`, `setup.cfg`, `.flake8`, `tox.ini` if any exist.
- **B-C1-9.** No mention of the 7 stray top-level Python files (`catboost_wrapper.py`, `test_catboost.py`, `test_phylop_block.py`, `variant_ensemble_cff925c.py`, `NOTEBOOK_CELL_FIXES.py`, `patch_finngen_wiring.py`, `download_finngen.py`).
- **B-C1-10.** No mention of the 6 root-level log files.
- **B-C1-11.** No instruction to add `*.egg-info/`, `build/`, `dist/` to `.gitignore`.

### C2 (pyproject.toml + editable reinstall)

- **B-C2-1.** No `[project.optional-dependencies]` block (see A4).
- **B-C2-2.** No `[tool.setuptools.package-data]` directive — non-`.py` files won't ship with the wheel.
- **B-C2-3.** No `[tool.pytest.ini_options]` declaring `testpaths = ["tests"]`.
- **B-C2-4.** No `[tool.mypy]` block.
- **B-C2-5.** No `pip uninstall genomic-variant-classifier` step before `pip install -e .`.
- **B-C2-6.** No verification of `pip show genomic-variant-classifier` post-install.
- **B-C2-7.** No clear statement about whether `setup.py` is removed in C1 or C2.
- **B-C2-8.** No analogous safety measure to C4's `.bak` for venv corruption.

### C3 (find/replace import sweep)

- **B-C3-1.** The PowerShell sweep doesn't include all needed file globs. Per audit, must include at minimum: `*.py`, `*.md`, `*.yml`, `*.yaml`, `*.toml`, `*.cfg`, `*.txt`, `*.ipynb`, `Dockerfile*`, `*.dockerfile`, `*.code-workspace`, `*.json` (e.g., for VS Code launch.json).
- **B-C3-2 [NEW, rev 2, HIGH].** **The regex pattern set is incomplete.** The spec's patterns cover only `from src.X.` and `from agents.`/`from config import`/`from message_bus import`/`from shared_state import` forms. It does NOT cover:
  - `import src.X.Y` (with or without `as <alias>`) — empirically present 14× at HEAD `b74daf1`
  - `import agent_layer.X` (form not present today but should be in the safety net)
  - Module-string forms like `"src.api.main:app"` in YAML/Dockerfile/docs (README line 223 has this exact form)
  - `from agent_layer.X` (vs. the bare `from agents.X` the spec covers — there could be both forms in the codebase)

  See §F-5 for the corrected regex set.
- **B-C3-3.** No special-case for `.ipynb` notebooks — they are JSON-with-embedded-source.
- **B-C3-4.** No verification step that the regex sweep didn't damage string literals, docstrings containing example code, or commented-out code.
- **B-C3-5.** README.md line 223 (`uvicorn src.api.main:app`) is not handled by any pattern in the spec's regex set.

### C4 (pickle migration) — highest-risk commit

- **B-C4-1.** **CRITICAL: insufficient sys.modules aliasing depth.** See §D-1.
- **B-C4-2.** No verification that loaded objects' nested classes' `__module__` attributes are *all* under `genomic_variant_classifier.*`.
- **B-C4-3.** `experiments/2026-04-04_03-39/ensemble_v1.joblib` is incorrectly classified (see A2).
- **B-C4-4.** No disk-space pre-flight (~9 GB pickles + ~9 GB `.bak` originals = ≥18 GB headroom).
- **B-C4-5.** No memory pre-flight.
- **B-C4-6.** No checksumming step.
- **B-C4-7.** No handling of the API service if it's hot-loading these joblibs.
- **B-C4-8.** Time estimate is unrealistic (see §C, risk register).

### C5 (cleanup + final audit)

- **B-C5-1.** `.pyc` cleanup ordering — should happen before C2's editable reinstall, not after C4. Stale bytecode cached under old module names can cause spurious `ModuleNotFoundError` during C2's smoke test that masks real issues.
- **B-C5-2.** Final audit's file-pattern globs need to match B-C3-1.
- **B-C5-3.** No verification that `pip install -e .` is idempotent post-migration.
- **B-C5-4.** No check that `python -m build` produces a valid wheel.
- **B-C5-5.** No update of `ROADMAP.md`, `METHODS.md`, `PHASE_1_ASSESSMENT.md`, `PHASE_2_FEATURES.md` for the namespace change.

### Cross-cutting

- **B-X-1.** No definition of "done." A success-criteria checklist (pytest green, smoke-imports green, all 5 joblibs load+predict-equivalent, API container builds) belongs in the spec body.
- **B-X-2.** No model-prediction equivalence test in C4. **Single most valuable addition.**
- **B-X-3.** No guidance on pre-commit hook bypass strategy.
- **B-X-4.** No mention of LFS / git-lfs for the 2-GB joblibs.

---

## C. Risk register (per commit)

Likelihood × Blast radius graded L/M/H. "Mitigation maturity" = how well the spec already handles it.

### C1 — Filesystem moves + remove `src/__init__.py`

| Failure mode | Likelihood | Blast radius | Mitigation maturity | Notes |
|---|---|---|---|---|
| `git mv` partial failure (untracked files left behind) | M | L (cosmetic) | LOW | Add `git status` clean-tree gate (A10, B-pre-1) |
| Empty namespace-package directory not picked up by find packages | L | M | LOW | Add post-mv `git ls-files src/genomic_variant_classifier/` enumeration |
| `agent_layer/agents/__init__.py` docstring lost | L | L | OK | Spec says preserved; add explicit byte-level diff |
| `tests/tests/` glitch hides test files post-migration | L | M | NONE | Investigate before C1 (B-C1-2) |
| Working tree unimportable between C1 and C2 | H (certain) | L (transient) | OK if accepted | See §F suggested fix |
| Removing `src/__init__.py` breaks any `from src import resolve_data_dir` consumer | L (zero by empirical grep) | H | OK now (verified) | §G check 4 makes this reproducible |
| **Rollback** | — | — | EASY | `git reset --hard HEAD~1` |

### C2 — pyproject.toml + editable reinstall

| Failure mode | Likelihood | Blast radius | Mitigation maturity | Notes |
|---|---|---|---|---|
| Stale `genomic_variant_classifier.egg-info/` chosen by pip with wrong `top_level.txt` | M | M | NONE (A5) | Must `rm -rf *.egg-info/` before C2 |
| pip resolves wrong `*.egg-info` (lexical ordering) | L | M | NONE | Verify only one egg-info post-install |
| Wheel-shipped resources (YAML, CSV) silently dropped | M | M | NONE (B-C2-2) | Inventory non-`.py` files |
| `pip install -e ".[api]"` breaks deploy/CI | M | H (CI red) | NONE (A4, B-C2-1) | **MUST FIX** |
| Duplicate source-of-truth: `setup.py` coexists with `pyproject.toml` | H | L | LOW | Remove `setup.py` in C2 |
| **Rollback** | — | — | EASY-MEDIUM | `git reset --hard HEAD~1`, then `pip install -e .` against old setup.py |

### C3 — Import sweep (regex find/replace)

| Failure mode | Likelihood | Blast radius | Mitigation maturity | Notes |
|---|---|---|---|---|
| **`import src.X.Y as <alias>` form not matched by spec's regex** | **H (certain — 14 lines confirmed)** | **H (pytest fails on every run after C1)** | **NONE (A1)** | **MUST FIX** — see §F-5 |
| Module-string forms like `"src.api.main:app"` not matched | H (README confirmed) | M | NONE (B-C3-5) | **MUST FIX** |
| Regex damages string literals or docstrings | M | L–M | LOW | Anchored to start-of-line mitigates most cases |
| Regex misses notebook (`.ipynb`) imports | M | M | LOW | B-C3-3 |
| BOM round-trip loses BOM that originally existed | L | L | OK on PS 7 | Confirm `$PSVersionTable.PSVersion` ≥ 7 |
| **Rollback** | — | — | MEDIUM | `git checkout HEAD~1 -- <files>` |

### C4 — Pickle migration **(IRREVERSIBLE WITHOUT BACKUP)**

| Failure mode | Likelihood | Blast radius | Mitigation maturity | Notes |
|---|---|---|---|---|
| sys.modules alias is too shallow → ModuleNotFoundError | **HIGH** | HIGH | NONE (B-C4-1, §D-1) | **MUST FIX** |
| Re-dump records old module name on nested classes | M | H | NONE (B-C4-2) | Walk object graph |
| `experiments/.../ensemble_v1.joblib` left referencing dead `src.models.variant_ensemble` | H | M | NONE (A2) | **MUST FIX** |
| Disk full mid-write | L | H (corrupted file, `.bak` survives) | LOW (B-C4-4) | Add disk-space pre-flight |
| OOM during 2 GB load | L | M | LOW (B-C4-5) | Note in spec |
| Compression mismatch breaks downstream tooling | M | L | NONE | Read-back original compress level |
| **Predict equivalence not verified — model artifact silently degraded** | L | **CATASTROPHIC** | NONE (B-X-2) | **MUST ADD** equivalence test |
| **Rollback** | — | — | OK | `.bak` restore works *if* C4 is run sequentially per-file with backup verification |

### C5 — Cleanup + final audit

| Failure mode | Likelihood | Blast radius | Mitigation maturity | Notes |
|---|---|---|---|---|
| `.pyc` cleanup happens too late | M | L | NONE (B-C5-1) | Move earlier |
| Audit globs too narrow | M | M | LOW | Expand globs |
| Phase docs become inconsistent | M | L | OK (4 incident docs covered) | Extend to ROADMAP, METHODS (B-C5-5) |
| **Rollback** | — | — | EASY | Cleanup is largely cosmetic |

---

## D. Technical correctness verification

### D-1. Pickle re-namespacing via `sys.modules` aliases — the spec's strategy is incomplete

**The mechanism, in detail.** When `pickle` (and therefore `joblib`) deserializes an object whose pickled stream encodes class location `src.api.pipeline.Pipeline`:

1. The unpickler reads the module name `"src.api.pipeline"` and class name `"Pipeline"` from the stream.
2. It calls (effectively) `__import__("src.api.pipeline", level=0, fromlist=["Pipeline"])`.
3. Then looks up `sys.modules["src.api.pipeline"]` and accesses attribute `Pipeline`.

If `src/` has been physically removed from disk (post C1) and the only alias is `sys.modules["src"] = genomic_variant_classifier`, step 2 fails because the import machinery has no FileFinder that can locate `src.api.pipeline` — the alias makes `sys.modules["src"]` resolvable as an *already-imported* package, but `__import__("src.api.pipeline")` requires either (a) the submodule to already be in `sys.modules`, or (b) the parent package's `__path__` to point to a filesystem location containing `api/pipeline.py`. Neither holds.

The correct pattern is to **walk the new package and create a sys.modules alias for every submodule**:

```python
import importlib
import pkgutil
import sys

import genomic_variant_classifier as _new_root

# Top-level alias (necessary, not sufficient)
sys.modules["src"] = _new_root
sys.modules["agent_layer"] = _new_root.agent_layer

# Walk every submodule of the new package and alias under both old roots
for finder, qualname, ispkg in pkgutil.walk_packages(
    _new_root.__path__, prefix="genomic_variant_classifier."
):
    mod = importlib.import_module(qualname)
    suffix = qualname[len("genomic_variant_classifier."):]
    sys.modules[f"src.{suffix}"] = mod
    if suffix.startswith("agent_layer."):
        rest = suffix[len("agent_layer."):]
        sys.modules[f"agent_layer.{rest}"] = mod
```

**With this in place, the rest of the spec's mechanism is sound**: when `joblib.load` reconstructs each class, Python resolves `src.api.pipeline.Pipeline` → (via alias) → `genomic_variant_classifier.api.pipeline.Pipeline`. The class object's `__module__` attribute is `"genomic_variant_classifier.api.pipeline"` (set at class-definition time, *not* derived from the alias key). When `joblib.dump` later re-pickles, `pickle.Pickler.save` calls `whichmodule(obj, name)` (CPython `Lib/pickle.py`), which first reads `obj.__module__` — yielding the new namespace path — and only falls back to scanning `sys.modules` if `__module__` is `None` or `"__main__"`. So **all nested classes get rewritten to the new namespace**, *provided every nested class's submodule was successfully aliased*.

**Failure mode if alias is incomplete.** A nested attribute whose class lives in a submodule not in the alias map will fail at load time with `ModuleNotFoundError: No module named 'src.<x>.<y>'`. The `.bak` original is then the only recovery path.

**Verdict.** The conceptual approach is correct. The spec's *implementation* of the alias is incomplete. **Patch text in §F-6.**

### D-2. PowerShell `[regex]::Replace` with `Multiline` and `$1` backreference

Confirmed correct. PowerShell uses the .NET regex engine, where:

- `[System.Text.RegularExpressions.RegexOptions]::Multiline` makes `^` and `$` match line boundaries.
- `$1` (not `\1`) is the .NET substitution syntax for backreference 1.
- When using `[regex]::Replace($text, $pattern, $substitute, [RegexOptions]::Multiline)`, the substitute string is parsed by the .NET regex replace engine, so `$1` is interpreted at substitution time. **No backtick-escaping needed when the substitute is a single-quoted PowerShell string** (single quotes prevent PowerShell variable interpolation from consuming the `$`).

⚠️ **Caveat the spec doesn't mention:** if the substitute is written as a *double-quoted* PowerShell string `"$1..."`, PowerShell will interpolate `$1` (as `$null` typically) **before** handing the string to .NET. The spec must use single-quoted substitute strings throughout.

### D-3. PEP 621 src-layout setuptools config

The spec's

```toml
[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]
include = ["genomic_variant_classifier*"]
```

is **syntactically and semantically correct** per the official setuptools 67.6+ documentation. With these directives:

- `find` looks for packages under `src/`.
- `include` constrains discovery to `genomic_variant_classifier` and any subpackages.
- `package-dir = {"" = "src"}` declares that the empty-string package (root) is rooted at `src/`.

**Recommended addition** (not in spec): `namespaces = false` to explicitly disable PEP 420 implicit namespace package discovery, so missing `__init__.py` files become loud failures rather than silent partial inclusion.

### D-4. Editable install and egg-info regeneration

When `pip install -e .` runs against a project with `pyproject.toml`:

1. The build backend (setuptools) generates a fresh `<dist_name>.egg-info/` directory at repo root.
2. The fresh egg-info contains `top_level.txt` listing only `genomic_variant_classifier`.
3. **However**, if a stale `genomic_variant_classifier.egg-info/` already exists with different `top_level.txt` contents, pip may read the stale one before regenerating it.

**Recommendation:** Delete *all* `*.egg-info/` directories at repo root **as the first action of C2**, before invoking `pip install -e .`.

### D-5. Pickle `whichmodule` semantics

Per CPython `Lib/pickle.py`: `whichmodule` first checks `getattr(obj, '__module__', None)`. If non-None, that's used directly; only falls back to scanning `sys.modules` if `__module__` is `None` or `'__main__'`.

This confirms: **after loading via aliases, the loaded class objects carry `__module__ = 'genomic_variant_classifier.<sub>'` (set at class definition in the new module)**, and re-dumping will faithfully record the new module name. The strategy is sound *iff* the aliases are deep enough to make the load itself succeed (D-1).

---

## E. Recommendations

### MUST FIX before committing the spec

1. **[A1, B-C3-2, F-5]** Extend the C1 (or C3) regex sweep to include the `import src.X.Y` form. **Empirical evidence: 14 lines at HEAD `b74daf1` would silently break pytest after C1 lands.** Also add patterns for module-string forms (e.g., `src.api.main:app` in uvicorn invocations, README, Dockerfile) and `import agent_layer.X`.
2. **[A2, B-C4-3]** Reclassify `experiments/2026-04-04_03-39/ensemble_v1.joblib`. Choose ONE: (a) migrate it; (b) delete it as Phase-4 cheap rider with SHA256 lineage in commit message; (c) keep as frozen historical artifact with README explaining the compatibility shim.
3. **[A4, B-C2-1]** Add `[project.optional-dependencies]` to `pyproject.toml` with at least `dev` and `api` extras, OR explicitly state in the spec that `pip install -e ".[api]"` is no longer supported.
4. **[A5, B-C5-1]** Move `*.egg-info/` and `__pycache__/` cleanup to **before** C2's reinstall.
5. **[B-C4-1, D-1]** Replace the C4 `sys.modules` aliasing block with the deep `pkgutil.walk_packages` form (§F-6). **Single highest-impact fix.**
6. **[B-C4-2, B-X-2]** Add a model-prediction equivalence test to C4: compare pre- and post- migration `predict_proba` on a frozen fixture within `np.testing.assert_allclose(rtol=1e-7)`.
7. **[A3]** Add note in C4: "After this commit, `experiments/.../ensemble_v1.joblib` is no longer byte-identical to `models/v1/ensemble_v1.joblib`. Pre-condition #6 holds only at HEAD `b74daf1`."
8. **[B-C2-7]** State explicitly: `setup.py` is removed in C2 (same commit that introduces `pyproject.toml`). Provide the `git rm setup.py` line.
9. **[F-1, src/__init__.py]** Tighten the spec's hedge ("likely no longer load-bearing") to a definitive statement supported by §G empirical check: drop the line and the file outright; no relocation of the re-export needed.

### SHOULD FIX before committing

10. **[B-pre-1, B-pre-3, B-pre-5, B-C4-4]** Add the pre-flight checklist (§G).
11. **[B-C2-3, B-C2-4]** Add `[tool.pytest.ini_options]` and `[tool.mypy]` blocks to `pyproject.toml`.
12. **[B-C1-4 through B-C1-7]** Extend C1's checklist to call out `Dockerfile`, `docker-compose.yml`, `.code-workspace`, `.pre-commit-config.yaml`, all `.github/workflows/*.yml`.
13. **[B-C1-9]** Add a decision section for the 7 stray top-level Python files.
14. **[B-C4-6]** Record SHA256 of each .joblib pre- and post- C4 in the commit message body.
15. **[A7, A8]** Verify each entry in `EXPECTED_IMPORTS` against the `src/` filesystem before committing the smoke test.
16. **[B-C5-5]** Extend the historical-doc footnote treatment to ROADMAP.md, METHODS.md, PHASE_*.md.

### SHOULD CONSIDER (user-decision tradeoffs)

17. **C1+C2 atomic merge** vs. keeping separate. Recommendation: keep separate but add explicit "TRANSITIONAL: codebase is intentionally unimportable at this commit" note in C1's commit message.
18. **C4 per-file commits** vs. single bulk commit. Recommendation: single commit, after every file's `.bak` and post-migration verification passes.
19. **Rename `src/` to flat layout?** No change recommended; document rationale.
20. **`tests/tests/` glitch.** Investigate before C1.
21. **Pre-commit hook bypass policy** for migration commits — decide in advance.

### CAN DEFER

22. **NetworkX gpickle deprecation** for `string_graph_700.pkl` (A9) — orthogonal hygiene.
23. **Root-level log-file cleanup** (B-C1-10) — Phase 4 cheap rider.
24. **Wheel build smoke test** (B-C5-4) — useful but not blocking.
25. **Production API rollout coordination** (B-C4-7) — orthogonal deployment concern.

---

## F. Suggested patch text

Apply these edits to `HYP_consolidate-package-layout.md` before committing.

### F-1. Spec wording on `src/__init__.py` (corrected from rev 1)

The spec already states (correctly):

> `src\__init__.py` line 8: `from src.utils.helpers import resolve_data_dir` → `from genomic_variant_classifier.utils.helpers import resolve_data_dir`. Drop this line entirely or relocate; it's likely no longer load-bearing in the new layout.

Tighten the hedge with empirical confirmation:

```diff
- `src\__init__.py` line 8: `from src.utils.helpers import resolve_data_dir` → `from genomic_variant_classifier.utils.helpers import resolve_data_dir`. Drop this line entirely or relocate; it's likely no longer load-bearing in the new layout.
+ `src\__init__.py` (4 module-level lines: docstring, `__version__`, `__author__`, plus `from src.utils.helpers import resolve_data_dir  # noqa: F401` on line 8): **drop the file entirely as part of C1's `git rm src/__init__.py`**. Empirical grep at HEAD `b74daf1` confirms zero consumers of `from src import resolve_data_dir`, `from src import __version__`, `from src import __author__`, or bare `import src` across all tracked `*.py` and `*.ipynb` files outside `.venv*`/`.git`/`__pycache__`/`*.egg-info`. The re-export pattern is dead code. No relocation to `genomic_variant_classifier/__init__.py` is needed; if any future code wants `resolve_data_dir`, it can use the canonical `from genomic_variant_classifier.utils.helpers import resolve_data_dir`.
```

### F-2. Pre-condition table (add new rows)

```markdown
| Active Python venv | `.venv312` (Python 3.12.10). Python 3.14 is broken (scipy/torch). C1–C5 all execute under `.venv312`. |
| Git working tree | `git status --porcelain` returns empty (verified at start). |
| Disk free space | ≥ 25 GB free on the volume hosting `models/`, `outputs/`, `experiments/` (for C4 `.bak` originals + new dumps). |
| egg-info | `genomic_variant_classifier.egg-info/` exists at repo root from a prior install attempt. **C2 deletes this BEFORE `pip install -e .`** (see §C2). |
| pre-commit | `.pre-commit-config.yaml` and `check_run_id_trailer.py` are in scope for C3's sweep. |
| Bare imports working today | Bare imports `from agents import X`, `from config import X`, `from message_bus import X`, `from shared_state import X` work today **only because** `setup.py` calls `find_packages()` with no `package_dir`, so `agent_layer/` at repo root is auto-discovered as a top-level package. After C1+C2, these become `from genomic_variant_classifier.agent_layer.agents import X` etc., handled by the C3 regex sweep. |
| `src/__init__.py` consumers | Empirically zero (broader grep returns no `from src import …` or `import src` hits at HEAD `b74daf1`). C1's drop is safe. |
| `import src.X.Y as <alias>` consumers | **14 lines confirmed at HEAD `b74daf1`** — `tests/unit/test_api.py` (13×: `import src.api.main as api_main`, `import src.api.auth as auth_module`) and `tests/unit/test_core.py:878` (`import src.data.dbnsfp as m`). Spec's regex sweep MUST handle this form (see §F-5). |
```

### F-3. C1 narrative (add explicit transitional-state note)

```markdown
> **Note on commit choreography.** This commit (C1) intentionally leaves the working
> tree in a temporarily-unimportable state: the old `src/` and `agent_layer/` paths no
> longer exist, but the new `genomic_variant_classifier` package is not yet installed
> (no `pyproject.toml` and the old `setup.py` no longer matches the on-disk layout).
> **C2 restores installability.** Do not `git bisect` across C1 alone; treat C1+C2 as
> a logical pair.
```

### F-4. C2 — replace setup.py removal + egg-info deletion + pyproject

Add to the very top of C2's command list:

```powershell
# C2 prologue: clear stale install state before reinstalling
git rm setup.py
Remove-Item -Recurse -Force .\genomic_variant_classifier.egg-info -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force .\src.egg-info -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force .\agent_layer.egg-info -ErrorAction SilentlyContinue
Get-ChildItem -Path . -Recurse -Force -Directory -Filter "__pycache__" |
    Remove-Item -Recurse -Force
pip uninstall -y genomic-variant-classifier 2>$null
```

Then the `pyproject.toml` content with the additions:

```toml
[build-system]
requires = ["setuptools>=67.6", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "genomic-variant-classifier"
version = "0.1.0"
description = "Ensemble ML system for pathogenic variant classification"
readme = "README.md"
requires-python = ">=3.10,<3.13"
authors = [{name = "Monzia Moodie", email = "monzia.moodie@gmail.com"}]
license = {file = "LICENSE"}
# Runtime dependencies are managed via pip-tools (requirements.in / requirements.lock).
# Editable installs do NOT pull from pyproject's dependencies array; install lockfile separately:
#   pip install -r requirements.lock
# Extras below allow `pip install -e ".[api]"` and `pip install -e ".[dev]"` to remain
# functional, deferring to lockfiles for full version pinning.
dependencies = []

[project.optional-dependencies]
api = []   # populated from requirements-api.in if/when this is migrated to pyproject-driven extras
dev = []   # populated from requirements-dev.in if/when this is migrated

[project.urls]
Homepage = "https://github.com/monzia-moodie-repo-projects/genomic-variant-classifier"
Repository = "https://github.com/monzia-moodie-repo-projects/genomic-variant-classifier"

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]
include = ["genomic_variant_classifier*"]
namespaces = false

[tool.setuptools.package-data]
"genomic_variant_classifier" = ["py.typed", "**/*.yaml", "**/*.yml", "**/*.json"]
# ^ adjust glob to match actual non-.py resources in src/genomic_variant_classifier/.
# Verify with: Get-ChildItem -Path src/genomic_variant_classifier -Recurse -File |
#   Where-Object { $_.Extension -notin '.py', '.pyc' }

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-ra --strict-markers"

[tool.mypy]
python_version = "3.12"
ignore_missing_imports = true
# (carry over additional settings from any existing mypy.ini)
```

### F-5. C3 — extend regex sweep file globs and patterns (CRITICAL)

This patch closes the empirically-confirmed gap (A1, B-C3-2). 14 production lines at HEAD `b74daf1` use the `import src.X.Y as <alias>` form that the spec's original sweep does not match.

```powershell
$includeGlobs = @(
    "*.py", "*.md", "*.yml", "*.yaml", "*.toml", "*.cfg", "*.txt",
    "*.ipynb", "*.json", "*.code-workspace",
    "Dockerfile", "Dockerfile.*", "*.dockerfile"
)

$patterns = @(
    # ----- from src.X import Y -> from genomic_variant_classifier.X import Y -----
    @{ Find = '(?m)^(\s*)from\s+src(\.[\w.]+)?\s+import\s';
       Replace = '$1from genomic_variant_classifier$2 import ' },

    # ----- import src.X.Y [as Z] -> import genomic_variant_classifier.X.Y [as Z] -----
    # (CRITICAL: empirically 14 hits at HEAD b74daf1)
    @{ Find = '(?m)^(\s*)import\s+src(\.[\w.]+)';
       Replace = '$1import genomic_variant_classifier$2' },

    # ----- from agent_layer.X import Y -> from genomic_variant_classifier.agent_layer.X import Y -----
    @{ Find = '(?m)^(\s*)from\s+agent_layer(\.[\w.]+)?\s+import\s';
       Replace = '$1from genomic_variant_classifier.agent_layer$2 import ' },

    # ----- import agent_layer.X [as Z] -> import genomic_variant_classifier.agent_layer.X [as Z] -----
    @{ Find = '(?m)^(\s*)import\s+agent_layer(\.[\w.]+)';
       Replace = '$1import genomic_variant_classifier.agent_layer$2' },

    # ----- bare imports of agent_layer subpackages -----
    @{ Find = '(?m)^(\s*)from\s+(agents|config|message_bus|shared_state)(\.[\w.]+)?\s+import\s';
       Replace = '$1from genomic_variant_classifier.agent_layer.$2$3 import ' },
    @{ Find = '(?m)^(\s*)import\s+(agents|config|message_bus|shared_state)(\.[\w.]+)?';
       Replace = '$1import genomic_variant_classifier.agent_layer.$2$3' },

    # ----- Module-string forms (uvicorn, celery, FastAPI, docs) -----
    # "src.api.main:app" -> "genomic_variant_classifier.api.main:app"
    @{ Find = '\bsrc\.([\w.]+):';
       Replace = 'genomic_variant_classifier.$1:' },

    # ----- Generic dotted-path mentions in docs -----
    # WARNING: DESTRUCTIVE in prose. Apply ONLY to specific files (Dockerfile,
    #    docker-compose.yml, README.md, METHODS.md, ROADMAP.md, *.code-workspace)
    #    or use a diff review before committing.
    @{ Find = '(?<![\w.])src\.([\w][\w.]*[\w])(?![\w.])';
       Replace = 'genomic_variant_classifier.$1' }
)
```

### F-6. C4 — replace the `sys.modules` aliasing block

```python
# scripts/migrate_pickles.py — replacement for the alias block

import importlib
import pkgutil
import sys

import genomic_variant_classifier as _new_root


def install_compat_aliases() -> None:
    """Make every old src.* and agent_layer.* module name resolve to the new namespace.

    Required before joblib.load() of any pre-migration pickle. Must be deep enough
    to cover every nested class's __module__ — a top-level alias only is NOT
    sufficient because pickle.find_class does direct sys.modules[full.dotted.name]
    lookups for nested classes.
    """
    # Top-level aliases (necessary, not sufficient on their own)
    sys.modules.setdefault("src", _new_root)
    sys.modules.setdefault("agent_layer", _new_root.agent_layer)

    # Walk every submodule and alias under both legacy roots
    for _finder, qualname, _ispkg in pkgutil.walk_packages(
        _new_root.__path__, prefix="genomic_variant_classifier."
    ):
        try:
            mod = importlib.import_module(qualname)
        except Exception as exc:  # pragma: no cover  -- defensive
            print(f"[migrate_pickles] WARNING: cannot import {qualname}: {exc}",
                  file=sys.stderr)
            continue
        suffix = qualname[len("genomic_variant_classifier."):]
        sys.modules[f"src.{suffix}"] = mod
        if suffix.startswith("agent_layer."):
            rest = suffix[len("agent_layer."):]
            sys.modules[f"agent_layer.{rest}"] = mod


def verify_no_legacy_modules(obj) -> list[str]:
    """Walk the object graph and return a list of any class.__module__ values
    still rooted under 'src.' or 'agent_layer.'. Empty list = success.

    Handles sklearn Pipelines, lists/dicts/tuples, and __dict__-bearing objects.
    """
    seen: set[int] = set()
    bad: list[str] = []

    def _walk(x: object) -> None:
        if id(x) in seen:
            return
        seen.add(id(x))
        cls_mod = getattr(type(x), "__module__", "") or ""
        if cls_mod.split(".", 1)[0] in {"src", "agent_layer"}:
            bad.append(f"{cls_mod}.{type(x).__name__}")
        if isinstance(x, dict):
            for k, v in x.items():
                _walk(k); _walk(v)
        elif isinstance(x, (list, tuple, set, frozenset)):
            for v in x:
                _walk(v)
        for attr_name in ("steps", "estimators_", "named_steps", "base_estimator",
                          "estimator", "model", "models", "pipeline"):
            sub = getattr(x, attr_name, None)
            if sub is not None and not callable(sub):
                _walk(sub)
        d = getattr(x, "__dict__", None)
        if isinstance(d, dict):
            for v in d.values():
                _walk(v)

    _walk(obj)
    return sorted(set(bad))
```

In the per-file C4 verification, use the deep walker:

```python
# After joblib.dump(obj, path), verify in a fresh subprocess (no aliases):
subprocess.run(
    [sys.executable, "-c",
     "import joblib, sys, json;"
     f"obj = joblib.load(r'{path}');"
     "from scripts.migrate_pickles import verify_no_legacy_modules;"
     "leftovers = verify_no_legacy_modules(obj);"
     "sys.exit(0 if not leftovers else (print(json.dumps(leftovers)) or 1))"],
    check=True,
)
```

⚠️ The subprocess imports `verify_no_legacy_modules` from `scripts.migrate_pickles`, but the verification subprocess must NOT have aliases installed (otherwise the verification is circular). The function itself only inspects `__module__` strings; it does not need aliases.

### F-7. C4 — explicitly handle `experiments/.../ensemble_v1.joblib`

Choose ONE and write into the spec.

**Option A — migrate it:**

```markdown
| `experiments/2026-04-04_03-39/ensemble_v1.joblib` | 1344.55 MB | `src.models.variant_ensemble` | **Migrate** (same procedure as `models/v1/ensemble_v1.joblib`). Note: post-C4, no longer byte-identical to `models/v1/ensemble_v1.joblib`. |
```

**Option B — delete it:**

```markdown
| `experiments/2026-04-04_03-39/ensemble_v1.joblib` | 1344.55 MB | `src.models.variant_ensemble` | **Delete** as Phase-4 cheap rider. Pre-deletion SHA256 lineage recorded in commit message proves it was byte-identical to the migrated `models/v1/ensemble_v1.joblib`. |
```

**Option C — freeze with shim doc:**

```markdown
| `experiments/2026-04-04_03-39/ensemble_v1.joblib` | 1344.55 MB | `src.models.variant_ensemble` | **Frozen historical artifact.** Add `experiments/2026-04-04_03-39/README.md` explaining that this file is intentionally pinned to the pre-migration namespace and requires `scripts/migrate_pickles.install_compat_aliases()` to load. |
```

### F-8. C4 — add prediction equivalence test (B-X-2)

```python
def equivalence_check(bak_path: str, new_path: str, fixture_X) -> None:
    """Assert that pre- and post- migration models produce identical predictions
    on a frozen fixture. Run BEFORE deleting the .bak."""
    import numpy as np
    install_compat_aliases()
    old = joblib.load(bak_path)
    new = joblib.load(new_path)  # already in new namespace
    if hasattr(old, "predict_proba"):
        np.testing.assert_allclose(old.predict_proba(fixture_X),
                                   new.predict_proba(fixture_X),
                                   rtol=1e-7, atol=0)
    elif hasattr(old, "predict"):
        np.testing.assert_array_equal(old.predict(fixture_X),
                                      new.predict(fixture_X))
    else:
        raise RuntimeError(f"No predict / predict_proba on {type(old).__name__}")
```

Reference a small fixture (5–10 rows) committed under `tests/fixtures/migration_smoke.parquet`.

### F-9. C5 — final stale-reference scan

```powershell
# C5 audit — final stale-reference scan
$globs = @("*.py","*.md","*.yml","*.yaml","*.toml","*.cfg","*.txt",
           "*.ipynb","*.json","*.code-workspace",
           "Dockerfile","Dockerfile.*","*.dockerfile")
$hits = Get-ChildItem -Path . -Recurse -File -Include $globs `
        -Exclude "*.bak" |
    Where-Object { $_.FullName -notmatch '\\(\.git|\.venv|\.venv312|node_modules|.*\.egg-info|__pycache__)\\' } |
    Select-String -Pattern '\b(from|import)\s+(src|agent_layer|agents|config|message_bus|shared_state)(\.|\s)','\bsrc\.[\w.]+:','\bagent_layer\.[\w.]+:'

if ($hits) {
    $hits | Format-Table -AutoSize
    throw "C5 audit: stale references remain. See above."
}
```

---

## G. Pre-flight checklist (run BEFORE C1)

Save the following as `scripts/preflight_consolidate_package_layout.ps1` and run it from repo root.

```powershell
# scripts/preflight_consolidate_package_layout.ps1
# Run BEFORE starting C1 of the consolidate-package-layout migration.
# Each block prints OK / FAIL. Any FAIL -> stop and resolve before proceeding.

$ErrorActionPreference = 'Continue'
$failures = @()

# 1. PowerShell version (must be 7+ for consistent UTF-8 no-BOM behavior)
$psVer = $PSVersionTable.PSVersion.Major
if ($psVer -ge 7) { "OK    PowerShell $psVer" } else {
    "FAIL  PowerShell $psVer (need 7+)"; $failures += "psversion"
}

# 2. Active venv is .venv312 with Python 3.12.x
$pyVer = & python -c "import sys; print('.'.join(map(str, sys.version_info[:3])))"
if ($pyVer -match '^3\.12\.') { "OK    Python $pyVer" } else {
    "FAIL  Python $pyVer (need 3.12.x)"; $failures += "pyver"
}
$venv = & python -c "import sys, os; print(os.path.basename(getattr(sys, 'prefix', '')))"
if ($venv -eq '.venv312') { "OK    venv = $venv" } else {
    "FAIL  venv = $venv (need .venv312)"; $failures += "venv"
}

# 3. Git working tree clean and on commit b74daf1 (or descendant)
$gitStatus = git status --porcelain
if (-not $gitStatus) { "OK    Working tree clean" } else {
    "FAIL  Working tree dirty:"; Write-Host $gitStatus; $failures += "gitclean"
}
$head = git rev-parse --short HEAD
"INFO  HEAD = $head (expected b74daf1 or descendant)"

# 4. src/__init__.py contains expected re-export, AND no consumers exist
if (-not (Test-Path .\src\__init__.py)) {
    "FAIL  src/__init__.py missing"; $failures += "srcinit_missing"
} else {
    $initContent = Get-Content .\src\__init__.py -Raw
    if ($initContent -match 'from\s+src\.utils\.helpers\s+import\s+resolve_data_dir') {
        "OK    src/__init__.py contains expected re-export"
    } else {
        "WARN  src/__init__.py present but does not match expected content"
    }
    # Empirical consumer check
    $consumers = Get-ChildItem -Path . -Recurse -Include "*.py","*.ipynb" `
        -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -notmatch '\\(\.git|\.venv|\.venv312|__pycache__|.*\.egg-info)\\' } |
        Select-String -Pattern '^\s*from\s+src\s+import\b|^\s*import\s+src\b|\bsrc\.(resolve_data_dir|__version__|__author__)\b'
    if ($consumers) {
        "FAIL  src/__init__.py re-exports have consumers:"
        $consumers | Format-Table -AutoSize
        $failures += "srcinit_consumers"
    } else {
        "OK    src/__init__.py re-exports have zero consumers (safe to drop)"
    }
}

# 5. setup.py present and matches expected size (587 bytes)
$setupBytes = (Get-Item .\setup.py).Length
if ($setupBytes -eq 587) { "OK    setup.py is 587 bytes" }
else { "WARN  setup.py is $setupBytes bytes (spec expected 587)" }

# 6. pyproject.toml does NOT yet exist
if (-not (Test-Path .\pyproject.toml)) { "OK    No pre-existing pyproject.toml" }
else { "FAIL  pyproject.toml already exists (spec assumes it doesn't)"; $failures += "pyproject" }

# 7. monitoring/ stub deleted (Phase 0 invariant)
if (-not (Test-Path .\monitoring)) { "OK    monitoring/ absent" }
else { "FAIL  monitoring/ still present"; $failures += "monitoring" }

# 8. egg-info inventory at repo root
$eggInfos = Get-ChildItem -Directory -Filter "*.egg-info" -ErrorAction SilentlyContinue
"INFO  egg-info dirs at root: $(($eggInfos | ForEach-Object { $_.Name }) -join ', ')"
if ($eggInfos.Count -gt 1) {
    "WARN  Multiple egg-info dirs - pip may pick wrong one. C2 must delete all."
}

# 9. Disk space (need >= 25 GB free for C4 .bak originals)
$drive = (Get-Item .).PSDrive
$freeGB = [math]::Round($drive.Free / 1GB, 1)
if ($freeGB -ge 25) { "OK    Free space: ${freeGB} GB" }
else { "FAIL  Free space: ${freeGB} GB (need >= 25 GB)"; $failures += "disk" }

# 10. The 5 (or 6) production joblibs exist at expected paths and sizes
$joblibs = @(
    @{ Path = "models\phase2_pipeline.joblib";          MB = 2029.75 },
    @{ Path = "models\phase4_pipeline.joblib";          MB = 2036.06 },
    @{ Path = "models\phase4_pipeline_calibrated.joblib"; MB = 2036.06 },
    @{ Path = "models\v1\ensemble_v1.joblib";           MB = 1344.55 },
    @{ Path = "outputs\run9_ready\models\ensemble.joblib"; MB = 1478.4  },
    @{ Path = "experiments\2026-04-04_03-39\ensemble_v1.joblib"; MB = 1344.55 }
)
foreach ($j in $joblibs) {
    if (Test-Path $j.Path) {
        $actualMB = [math]::Round((Get-Item $j.Path).Length / 1MB, 2)
        "OK    $($j.Path) - $actualMB MB (expected $($j.MB))"
    } else {
        "FAIL  $($j.Path) missing"; $failures += "joblib:$($j.Path)"
    }
}

# 11. Verify experiments/.../ensemble_v1.joblib SHA256 = models/v1/ensemble_v1.joblib SHA256
$h1 = (Get-FileHash -Algorithm SHA256 .\models\v1\ensemble_v1.joblib).Hash
$h2 = (Get-FileHash -Algorithm SHA256 .\experiments\2026-04-04_03-39\ensemble_v1.joblib).Hash
if ($h1 -eq $h2) { "OK    Both ensemble_v1.joblib copies SHA256-match: $h1" }
else { "FAIL  ensemble_v1.joblib copies differ"; $failures += "sha256_ensemble" }

# 12. Current bare imports work (baseline health check)
& python -c "import agents, config, message_bus, shared_state; print('bare imports OK')" 2>&1 |
    ForEach-Object { if ($_ -match 'OK') { "OK    Bare agent_layer imports work" }
                     else { "FAIL  Bare imports broken: $_"; $script:failures += "bareimports" } }

# 13. src.api.main importable today
& python -c "import src.api.main; print('src.api.main OK')" 2>&1 |
    ForEach-Object { if ($_ -match 'OK') { "OK    src.api.main imports" }
                     else { "FAIL  src.api.main does not import: $_"; $script:failures += "srcmain" } }

# 14. Empirical inventory of `import src.X.Y` lines that the C3 sweep MUST handle (rev 2)
$importSrcHits = Get-ChildItem -Path . -Recurse -Include "*.py","*.ipynb" `
    -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -notmatch '\\(\.git|\.venv|\.venv312|__pycache__|.*\.egg-info)\\' } |
    Select-String -Pattern '^\s*import\s+src\.[\w.]+'
if ($importSrcHits) {
    "INFO  $($importSrcHits.Count) lines of 'import src.X.Y' form found (must be handled by C3 §F-5):"
    $importSrcHits | Select-Object Path, LineNumber, Line | Format-Table -AutoSize -Wrap
} else {
    "OK    No 'import src.X.Y' lines (consistent with empirical findings — but spec sweep should still handle the pattern)"
}

# 15. Inventory non-.py files inside src/
$nonPy = Get-ChildItem -Path .\src -Recurse -File |
    Where-Object { $_.Extension -notin '.py', '.pyc' -and $_.FullName -notmatch '__pycache__' }
"INFO  Non-.py files under src/: $($nonPy.Count)"
if ($nonPy.Count -gt 0) {
    $nonPy | Select-Object -First 20 | ForEach-Object { "      $($_.FullName)" }
    if ($nonPy.Count -gt 20) { "      ... and $($nonPy.Count - 20) more" }
}

# 16. tests/tests/ glitch
if (Test-Path .\tests\tests) {
    "WARN  tests/tests/ exists - investigate before C1"
    Get-ChildItem .\tests\tests -Recurse -File | Select-Object FullName | Format-Table -AutoSize
}

# 17. Pre-commit / hook script existence
if (Test-Path .\.pre-commit-config.yaml) { "OK    .pre-commit-config.yaml exists - must be in C3 sweep" }
if (Test-Path .\scripts\check_run_id_trailer.py) { "OK    check_run_id_trailer.py exists" }

# 18. validate_docs.py exists
if (Test-Path .\validate_docs.py) { "OK    validate_docs.py exists - re-run after C5 doc edits" }

Write-Host ""
if ($failures.Count -eq 0) {
    Write-Host "================================================================" -ForegroundColor Green
    Write-Host "PRE-FLIGHT GREEN - safe to start C1." -ForegroundColor Green
    Write-Host "================================================================" -ForegroundColor Green
    exit 0
} else {
    Write-Host "================================================================" -ForegroundColor Red
    Write-Host "PRE-FLIGHT RED - resolve these before starting C1:" -ForegroundColor Red
    $failures | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
    Write-Host "================================================================" -ForegroundColor Red
    exit 1
}
```

---

## Closing notes

The `HYP_consolidate-package-layout.md` migration is structurally sound and addresses real layout debt that has been blocking modern packaging-tool adoption (uv, ruff src-detection, mypy strict-mode src-resolution). The five-commit choreography is reasonable. But it is **not yet safe to commit as a hypothesis document** because:

- **The regex sweep is empirically incomplete** (A1) — 14 lines at HEAD `b74daf1` would silently survive C1 and break pytest. The §F-5 patch is mandatory.
- **One file is mis-classified** in a way that would produce an irreversibly broken artifact if executed literally (A2).
- **The pickle-rename mechanism is incomplete** in a way that virtually guarantees C4 will fail on first run (D-1, B-C4-1) — though `.bak` makes recovery easy, repeated failures will erode user confidence and waste hours.
- **Egg-info ordering is dangerous** (A5, D-4) and can produce silent install corruption.
- **Optional-dependencies / extras gap** (A4) silently breaks any deploy script using `pip install -e ".[api]"`.

After applying the patches in §F and running the §G pre-flight script, the spec should be safe to commit to `docs/hypotheses/` and execute. Estimated realistic walltime for C4 alone (with the corrected aliasing, equivalence checks, and 5–6 files at ~9 GB total): **2.5–4 hours of wall time**, of which ~30–60 minutes is hands-on and the rest is I/O-bound load+dump+verify cycles. The spec's "4.25–6.75 hours total" estimate is plausible for the migration as a whole only if C4's per-file verification is shallow; with the deep equivalence check recommended here, expect the upper end (6–8 hours total over a working day, with breaks).

Recommend the user incorporate the §F patches, re-run the §G pre-flight, then commit the corrected spec to `docs/hypotheses/HYP_consolidate-package-layout.md` and proceed to execution. This review document should be committed to `docs/reviews/REVIEW_HYP_consolidate-package-layout_2026-05-08.md` as the audit trail.
