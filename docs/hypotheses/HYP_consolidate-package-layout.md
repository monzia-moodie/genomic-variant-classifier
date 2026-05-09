---
id: HYP_consolidate-package-layout
title: "Consolidate src.* and agent_layer.* into single genomic_variant_classifier package (PEP 621 src-layout)"
status: hypothesis
date: 2026-05-08
---

# Consolidate package layout

## Executive summary

The project currently exposes two parallel top-level Python namespaces — `src.*` (58 tracked files) and `agent_layer.*` (29 tracked files) — both made importable today by `setup.py` calling `find_packages()` with no `package_dir` directive. This works but blocks several modern tooling workflows (uv, ruff src-detection, mypy strict-mode src-resolution, clean wheel packaging) and creates two-source-of-truth confusion when an `agent_layer/` module imports from `src/`.

This hypothesis proposes a five-commit migration (C1–C5) that consolidates both namespaces into a single `genomic_variant_classifier` package rooted under `src/` per PEP 621 src-layout convention. After execution:

- `src/genomic_variant_classifier/` is the only Python package
- `pyproject.toml` replaces `setup.py`
- All imports resolve via `genomic_variant_classifier.*`
- Five production joblib pickles (~9 GB total) are migrated to record class paths under the new namespace
- One redundant joblib copy in `experiments/2026-04-04_03-39/` is **deleted** (SHA256 lineage preserved in C4 commit message body)

The spec was reviewed in `docs/reviews/REVIEW_HYP_consolidate-package-layout_2026-05-08.md` (committed in `40fae1f`, rev-2 corrections applied). The §A1, §A8, and §F-1 retractions have been folded in: `src/__init__.py` is **not** empty — it contains a load-bearing re-export — but a broader empirical grep confirmed **zero consumers** of that re-export, so the C1 `git rm` remains a clean no-ripple drop. Pre-flight GREEN was achieved at `12b8315` (log: `docs/sessions/SESSION_2026-05-08_preflight_GREEN.txt`).

## Pre-conditions

All verified empirically at HEAD `12b8315` via `scripts/preflight_consolidate_package_layout.ps1`. Re-run that script before C1 and require a GREEN exit before proceeding.

| Pre-condition | Status |
|---|---|
| Active venv | `.venv312` (Python 3.12.10). Python 3.14 breaks scipy/torch and is not in use. C1–C5 all execute under `.venv312`. |
| Git working tree | `git status --porcelain` returns empty. |
| Git HEAD | `12b8315` on `main`, pushed to origin. |
| Disk free | 168.9 GB on the volume hosting `models/`, `outputs/`, `experiments/` (≫25 GB needed for C4 `.bak` originals). |
| `src/__init__.py` | Contains a load-bearing re-export: `from src.utils.helpers import resolve_data_dir  # noqa: F401`, plus `__version__` and `__author__`. **Empirically verified: zero consumers** (the broader grep for `from src import …` and `src.{resolve_data_dir,__version__,__author__}` returned empty). C1's `git rm src/__init__.py` is therefore a clean no-ripple drop. |
| `setup.py` | 587 bytes; `find_packages()` with no `package_dir`, which is what makes both `src/` and `agent_layer/` discoverable today. |
| `pyproject.toml` | Does NOT exist. C2 introduces it. |
| `monitoring/` stub | Absent (deleted in Phase 0). |
| egg-info | Single `genomic_variant_classifier.egg-info/` at repo root from a prior install attempt. **C2 deletes this BEFORE `pip install -e .`** to prevent stale `top_level.txt` polluting the namespace (per pip issue #6048; setuptools issue #4197). |
| Production joblibs | 5 production files at expected sizes; 1 redundant copy in `experiments/2026-04-04_03-39/` (deletion target). |
| SHA256 lineage | `models/v1/ensemble_v1.joblib` and `experiments/2026-04-04_03-39/ensemble_v1.joblib` are byte-identical: `64FEF61170E98FF722C338734819008A4F8307111D6AB97EC6BE14A29ADCEE23`. **This invariant breaks the moment C4 rewrites the production copy** — recorded in C4 commit message body for forensic audit. |
| Bare imports today | `from agents import X`, `from config import X`, `from message_bus import X`, `from shared_state import X` work today only because `setup.py` calls `find_packages()` with no `package_dir`, so `agent_layer/` at repo root is auto-discovered as a top-level package and its submodules become directly importable. After C1+C2 these become `from genomic_variant_classifier.agent_layer.{agents,config,message_bus,shared_state} import X`, handled by C3. |
| `src.api.main` import | Currently importable. Baseline confirmed before C1. |
| Non-`.py` files under `src/` | **0** (verified). No `[tool.setuptools.package-data]` declaration is needed in `pyproject.toml`. |
| `tests/tests/` glitch | **Not present** (was a Get-ChildItem rendering artifact from prior screen capture). No cleanup needed. |
| Run-ID trailer convention | **Manual discipline only.** No `.pre-commit-config.yaml` exists in the repo; no `check_run_id_trailer.py` script exists; only `*.sample` hooks present in `.git/hooks/`. The `Run-ID: <slug>` trailer is appended manually to every commit message. |
| `validate_docs.py` | Exists at repo root; validates frontmatter on `docs/hypotheses/*.md`. Re-run after C5 doc edits. |

### Empirical inventory of `import src.X.Y as <alias>` lines for C3

The pre-flight discovered 14 lines using the `import X.Y as Z` form (which a naïve `from X import Y` regex would miss):

| File | Line | Statement |
|---|---:|---|
| `tests/unit/test_api.py` | 490 | `import src.api.main as api_main` |
| `tests/unit/test_api.py` | 515 | `import src.api.main as api_main` |
| `tests/unit/test_api.py` | 610 | `import src.api.main as api_main` |
| `tests/unit/test_api.py` | 660 | `import src.api.main as api_main` |
| `tests/unit/test_api.py` | 681 | `import src.api.main as api_main` |
| `tests/unit/test_api.py` | 696 | `import src.api.main as api_main` |
| `tests/unit/test_api.py` | 725 | `import src.api.main as api_main` |
| `tests/unit/test_api.py` | 741 | `import src.api.main as api_main` |
| `tests/unit/test_api.py` | 775 | `import src.api.auth as auth_module` |
| `tests/unit/test_api.py` | 786 | `import src.api.auth as auth_module` |
| `tests/unit/test_api.py` | 797 | `import src.api.auth as auth_module` |
| `tests/unit/test_api.py` | 808 | `import src.api.auth as auth_module` |
| `tests/unit/test_api.py` | 819 | `import src.api.auth as auth_module` |
| `tests/unit/test_core.py` | 878 | `import src.data.dbnsfp as m` |

The C3 sweep MUST include the `import X.Y as Z` pattern. Post-C3 verification: zero hits for any of these.

## Out of scope

This migration does NOT touch:

- Run 9 launch / Vast.ai infrastructure (separate workstream; queued post-migration).
- Drift agent wiring beyond the namespace rewrite.
- Root-level cruft (6 stray top-level Python files: `catboost_wrapper.py`, `test_catboost.py`, `test_phylop_block.py`, `variant_ensemble_cff925c.py`, `NOTEBOOK_CELL_FIXES.py`, `patch_finngen_wiring.py`, `download_finngen.py`; 6 root log files) — Phase-4 cheap rider.
- LOVD silent-zero incident remediation (R10-A queued post-migration).
- ESM-2 HGVSp parser fix (R10 queued post-migration).
- NetworkX `gpickle` deprecation hygiene for `data/raw/cache/string_graph_700.pkl` — orthogonal hypothesis if the load-site code uses `nx.read_gpickle`.
- Production API rollout coordination (this is dev-side migration only).
- Wheel build smoke test (`python -m build`) — useful but not blocking.

## Target layout

```
genomic-variant-classifier/
├── src/
│   └── genomic_variant_classifier/
│       ├── __init__.py            # new: holds the resolve_data_dir re-export iff
│       │                          # the broader grep (re-run pre-C1) finds any
│       │                          # surprise consumer; otherwise empty
│       ├── api/                   # was src/api/
│       ├── data/                  # was src/data/
│       ├── models/                # was src/models/
│       ├── reports/               # was src/reports/
│       ├── utils/                 # was src/utils/ (provides helpers.resolve_data_dir)
│       ├── ...                    # all current src/<subpkg>/ moved here
│       └── agent_layer/
│           ├── __init__.py        # new: empty
│           ├── agents/            # was agent_layer/agents/
│           │   ├── __init__.py    # preserved verbatim (170 B docstring)
│           │   └── ...
│           ├── config/            # was agent_layer/config/
│           ├── message_bus/       # was agent_layer/message_bus/
│           └── shared_state/      # was agent_layer/shared_state/
├── pyproject.toml                 # new (replaces setup.py in C2)
├── tests/                         # unchanged structure; imports rewritten in C3
├── scripts/
│   └── migrate_pickles.py         # populated pre-C4 (currently empty placeholder)
├── tests/smoke_test_imports.py    # populated pre-C2 (currently empty placeholder)
├── README.md                      # 2 hits to fix in C5 (lines 196, 223)
├── ROADMAP.md, METHODS.md, PHASE_*.md  # historical-doc footnotes in C5
├── docs/                          # active docs updated in C5; historical incident
│                                  # docs get [Layout note: ...] footnotes
├── Dockerfile, docker-compose.yml # updated in C3 (module-string forms)
├── .github/workflows/             # drift_monitor.yml:215 + others fixed in C5
└── ...                            # other root files unchanged
```

## `pyproject.toml` (created in C2)

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

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]
include = ["genomic_variant_classifier*"]
namespaces = false  # surface missing __init__.py as loud failure, not silent partial inclusion

# [tool.setuptools.package-data] omitted: pre-flight verified 0 non-.py files under src/.

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-ra --strict-markers"

[tool.mypy]
python_version = "3.12"
ignore_missing_imports = true
# (carry over additional settings here if any mypy.ini surfaces during C2)
```

## Find / replace map

The C3 import sweep applies these regex transformations across an expanded glob set. **All substitute strings are single-quoted** in PowerShell to prevent variable interpolation of `$1`, `$2`, etc.

```powershell
$includeGlobs = @(
    "*.py", "*.md", "*.yml", "*.yaml", "*.toml", "*.cfg", "*.txt",
    "*.ipynb", "*.json", "*.code-workspace",
    "Dockerfile", "Dockerfile.*", "*.dockerfile"
)

$patterns = @(
    # 1. from src.X import Y      -> from genomic_variant_classifier.X import Y
    @{ Find = '(?m)^(\s*)from\s+src(\.[\w.]+)?\s+import\s';
       Replace = '$1from genomic_variant_classifier$2 import ' },
    # 2. import src.X             -> import genomic_variant_classifier.X
    @{ Find = '(?m)^(\s*)import\s+src(\.[\w.]+)';
       Replace = '$1import genomic_variant_classifier$2' },
    # 3. import src.X as Z        -> import genomic_variant_classifier.X as Z
    #    (Pattern 2 already covers this; verified against the 14 known lines.)
    # 4. from agent_layer.X       -> from genomic_variant_classifier.agent_layer.X
    @{ Find = '(?m)^(\s*)from\s+agent_layer(\.[\w.]+)?\s+import\s';
       Replace = '$1from genomic_variant_classifier.agent_layer$2 import ' },
    # 5. import agent_layer.X     -> import genomic_variant_classifier.agent_layer.X
    @{ Find = '(?m)^(\s*)import\s+agent_layer(\.[\w.]+)?';
       Replace = '$1import genomic_variant_classifier.agent_layer$2' },
    # 6. Bare imports of agent_layer subpackages
    @{ Find = '(?m)^(\s*)from\s+(agents|config|message_bus|shared_state)(\.[\w.]+)?\s+import\s';
       Replace = '$1from genomic_variant_classifier.agent_layer.$2$3 import ' },
    @{ Find = '(?m)^(\s*)import\s+(agents|config|message_bus|shared_state)(\.[\w.]+)?';
       Replace = '$1import genomic_variant_classifier.agent_layer.$2$3' },
    # 7. Module-string forms (uvicorn, celery, FastAPI, docs):
    #    "src.api.main:app"  ->  "genomic_variant_classifier.api.main:app"
    @{ Find = '\bsrc\.([\w.]+):';
       Replace = 'genomic_variant_classifier.$1:' },
    # 8. Generic dotted-path mentions in docs (anchor whole-token, run last with preview)
    @{ Find = '(?<![\w.])src\.([\w][\w.]*[\w])(?![\w.])';
       Replace = 'genomic_variant_classifier.$1' }
)
```

⚠️ **Pattern 8 is destructive in prose** — it will rewrite "src.foo" inside any English sentence containing that token. Apply pattern 8 to specific files only (`README.md`, `METHODS.md`, `ROADMAP.md`, `Dockerfile`, `docker-compose.yml`, `.code-workspace`); NEVER apply pattern 8 globally.

Post-C3 verification regex (must yield zero hits, run as audit before commit):

```powershell
'\b(from|import)\s+(src|agent_layer|agents|config|message_bus|shared_state)(\.|\s)'
'\bsrc\.[\w.]+:'
'\bagent_layer\.[\w.]+:'
```

---

# Commit choreography (C1–C5)

## C1 — Filesystem move

**Goal:** physically move `src/<subpkgs>/` → `src/genomic_variant_classifier/<subpkgs>/` and `agent_layer/` → `src/genomic_variant_classifier/agent_layer/`. Remove the now-orphaned `src/__init__.py`. Do not change any imports yet — that's C3.

> **Note on commit choreography.** This commit (C1) intentionally leaves the working tree in a temporarily-unimportable state: the old `src/` and `agent_layer/` paths no longer exist, but the new `genomic_variant_classifier` package is not yet installed (no `pyproject.toml`, and the old `setup.py`'s `find_packages()` call no longer matches the on-disk layout). **C2 restores installability.** Do not `git bisect` across C1 alone; treat C1+C2 as a logical pair.

### Pre-checks

```powershell
cd C:\Projects\genomic-variant-classifier
git rev-parse --short HEAD                                           # Expected: 12b8315 (or descendant)
git status --porcelain                                               # Expected: empty
$env:VIRTUAL_ENV                                                     # Expected: ...\.venv312
pwsh -File .\scripts\preflight_consolidate_package_layout.ps1        # Expected: PRE-FLIGHT GREEN
```

### Action

```powershell
# 1. Remove the load-bearing-but-orphaned src/__init__.py
git rm src/__init__.py

# 2. Create the destination package directory
New-Item -ItemType Directory -Path src/genomic_variant_classifier -Force | Out-Null
New-Item -ItemType File -Path src/genomic_variant_classifier/__init__.py -Force | Out-Null

# 3. Move every subpackage of src/ into src/genomic_variant_classifier/
$srcSubpkgs = Get-ChildItem -Path src -Directory |
              Where-Object { $_.Name -ne 'genomic_variant_classifier' } |
              Select-Object -ExpandProperty Name
foreach ($sp in $srcSubpkgs) {
    git mv "src/$sp" "src/genomic_variant_classifier/$sp"
}

# 4. Move agent_layer/ into src/genomic_variant_classifier/agent_layer/
#    (agent_layer has no __init__.py at root — PEP 420 namespace today; create one in destination)
New-Item -ItemType Directory -Path src/genomic_variant_classifier/agent_layer -Force | Out-Null
New-Item -ItemType File -Path src/genomic_variant_classifier/agent_layer/__init__.py -Force | Out-Null
$alSubpkgs = Get-ChildItem -Path agent_layer -Directory | Select-Object -ExpandProperty Name
foreach ($sp in $alSubpkgs) {
    git mv "agent_layer/$sp" "src/genomic_variant_classifier/agent_layer/$sp"
}
# Move any tracked top-level files in agent_layer/ (if any)
$alFiles = git ls-files agent_layer | Where-Object { $_ -notmatch '/' }
foreach ($f in $alFiles) {
    $name = Split-Path -Leaf $f
    git mv "agent_layer/$name" "src/genomic_variant_classifier/agent_layer/$name"
}
# Verify agent_layer/ is now empty per git's view
$leftover = git ls-files agent_layer
if ($leftover) { throw "agent_layer/ still has tracked files: $leftover" }
# If the directory is fully empty on disk, drop it
if ((Get-ChildItem -Path agent_layer -Force -ErrorAction SilentlyContinue).Count -eq 0) {
    Remove-Item -Path agent_layer -Recurse -Force
}
```

### Post-checks

```powershell
# 1. Verify the new package contains all 58 src files + 29 agent_layer files
$count = (git ls-files src/genomic_variant_classifier | Measure-Object).Count
"Files under new package: $count (expected: 58 + 29 = 87, plus 2 new __init__.py)"

# 2. Verify old paths are empty
$oldSrc = git ls-files src | Where-Object { $_ -notmatch '^src/genomic_variant_classifier/' }
if ($oldSrc) { throw "Stale files at old src/ path:`n$($oldSrc -join "`n")" }
$oldAL = git ls-files agent_layer
if ($oldAL) { throw "Stale files at old agent_layer/ path:`n$($oldAL -join "`n")" }

# 3. agent_layer/agents/__init__.py docstring preserved (170 B)
$agentInit = "src/genomic_variant_classifier/agent_layer/agents/__init__.py"
$bytes = (Get-Item $agentInit).Length
"agents/__init__.py size: $bytes bytes (expected: ~170)"

# 4. The codebase is INTENTIONALLY unimportable here — do not test imports yet
"Working tree is unimportable until C2; this is expected."
```

### Commit

```powershell
git status                                                            # review staged moves
git commit -m @"
chore(migration): C1 layout move - src/* and agent_layer/* into src/genomic_variant_classifier/

Physical filesystem reorganization only - imports are NOT rewritten in this commit
(that is C3). The working tree is intentionally unimportable between C1 and C2;
C2 introduces pyproject.toml and reinstalls the package, restoring importability.

- git mv src/<subpkg> -> src/genomic_variant_classifier/<subpkg> (all subpackages)
- git mv agent_layer/<subpkg> -> src/genomic_variant_classifier/agent_layer/<subpkg>
- git rm src/__init__.py (load-bearing re-export with empirically zero consumers)
- New empty __init__.py files for genomic_variant_classifier/ and agent_layer/

TRANSITIONAL: do not git bisect across C1 alone. Treat C1+C2 as a logical pair.

Run-ID: migration-c1
"@
git push origin main
```

### Rollback

`git reset --hard HEAD~1` — nothing destructive on disk; `git mv` is fully reversible at this stage.

---

## C2 — `pyproject.toml` + editable reinstall

**Goal:** introduce `pyproject.toml`, remove `setup.py`, clear stale install state, and re-install the package editable. After C2 the codebase is importable again under the new namespace name.

### Pre-checks

```powershell
git rev-parse --short HEAD                                           # Expected: C1 commit
git status --porcelain                                               # Expected: empty
$env:VIRTUAL_ENV                                                     # Expected: ...\.venv312

# Verify the C1 layout is intact
Test-Path src/genomic_variant_classifier/__init__.py                  # Expected: True
Test-Path src/genomic_variant_classifier/agent_layer/__init__.py      # Expected: True
Test-Path agent_layer                                                 # Expected: False
```

### Action — prologue: clear stale install state

```powershell
# Remove old packaging
git rm setup.py

# Wipe ALL egg-info (per pip issue #6048: stale top_level.txt can pollute namespace)
Remove-Item -Recurse -Force .\genomic_variant_classifier.egg-info -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force .\src.egg-info -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force .\agent_layer.egg-info -ErrorAction SilentlyContinue
Get-ChildItem -Directory -Filter "*.egg-info" | Remove-Item -Recurse -Force

# Wipe stale __pycache__ (cached bytecode under old namespace would mask the new layout)
Get-ChildItem -Path . -Recurse -Force -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -notmatch '\\(\.venv|\.venv312|node_modules)\\' } |
    Remove-Item -Recurse -Force

# Uninstall any prior editable install of this distribution
pip uninstall -y genomic-variant-classifier 2>$null
```

### Action — create `pyproject.toml`

Write the file content shown in the **`pyproject.toml`** section above (use UTF-8 no-BOM). Then:

### Action — install

```powershell
pip install -e .
```

### Post-checks

```powershell
# 1. pyproject.toml is the only build config
Test-Path pyproject.toml                                              # Expected: True
Test-Path setup.py                                                    # Expected: False

# 2. Editable install resolved correctly
pip show genomic-variant-classifier
# Expected: Location matches repo root; Editable-project-location set; metadata sane

# 3. Single egg-info, listing only the new package
$eggInfos = Get-ChildItem -Directory -Filter "*.egg-info"
$eggInfos.Count                                                       # Expected: 1
Get-Content (Join-Path $eggInfos[0].FullName "top_level.txt")
# Expected (single line): genomic_variant_classifier

# 4. New namespace imports cleanly
python -c "import genomic_variant_classifier; print(genomic_variant_classifier.__file__)"
python -c "import genomic_variant_classifier.api.main; print('api.main OK')"
python -c "import genomic_variant_classifier.agent_layer.agents; print('agent_layer.agents OK')"

# 5. Old namespace is GONE (sanity: this should fail)
python -c "import src.api.main" 2>&1 | Select-String "ModuleNotFoundError"
# Expected: ModuleNotFoundError - confirms the old namespace really doesn't resolve

# 6. Idempotent reinstall
pip install -e . --quiet
"Reinstall completed without changes (no errors above)"
```

### Commit

```powershell
git add pyproject.toml
git status                                                            # review: pyproject.toml added, setup.py removed
git commit -m @"
chore(migration): C2 introduce pyproject.toml + editable reinstall

Replaces setup.py with PEP 621 pyproject.toml. After C2 the codebase
is importable as genomic_variant_classifier.* (and only that namespace).

- New pyproject.toml: PEP 621, src-layout, namespaces=false
- [project.optional-dependencies] with api/dev placeholder extras
- [tool.pytest.ini_options]: testpaths = tests
- [tool.mypy]: python_version 3.12
- Removed setup.py (587 B)
- Cleared all *.egg-info/ and __pycache__/ directories (stale top_level.txt risk)
- pip uninstall -y genomic-variant-classifier && pip install -e .

Restores importability after C1's transitional state.

Run-ID: migration-c2
"@
git push origin main
```

### Rollback

`git reset --hard HEAD~1` then `pip install -e .` against the restored `setup.py`. Confirm with `pip show genomic-variant-classifier` and `python -c "import src.api.main"`.

---

## C3 — Import sweep

**Goal:** rewrite all `from src.X`, `from agent_layer.X`, bare-agent-subpackage imports, and module-string forms (e.g. `"src.api.main:app"`) to use `genomic_variant_classifier.*`. The post-sweep audit must show zero stale references in any tracked file.

### Pre-checks

```powershell
git rev-parse --short HEAD                                           # Expected: C2 commit
git status --porcelain                                               # Expected: empty
$env:VIRTUAL_ENV                                                     # Expected: ...\.venv312

# Confirm the new namespace currently imports cleanly
python -c "import genomic_variant_classifier.api.main; print('OK')"

# Snapshot pre-sweep hit count (will be the verification target post-sweep: 0)
$preCount = (Get-ChildItem -Path . -Recurse -File -Include *.py,*.md,*.yml,*.yaml,*.toml,*.cfg,*.txt,*.ipynb,*.json,*.code-workspace,Dockerfile,Dockerfile.*,*.dockerfile -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -notmatch '\\(\.git|\.venv|\.venv312|node_modules|.*\.egg-info|__pycache__|catboost_info)\\' } |
    Select-String -Pattern '\b(from|import)\s+(src|agent_layer|agents|config|message_bus|shared_state)(\.|\s)','\bsrc\.[\w.]+:','\bagent_layer\.[\w.]+:').Count
"Pre-sweep stale-reference hits: $preCount"
```

### Action — apply patterns 1–7 across full glob set

```powershell
$includeGlobs = @(
    "*.py","*.md","*.yml","*.yaml","*.toml","*.cfg","*.txt",
    "*.ipynb","*.json","*.code-workspace",
    "Dockerfile","Dockerfile.*","*.dockerfile"
)
$patterns = @(
    @{ Find = '(?m)^(\s*)from\s+src(\.[\w.]+)?\s+import\s';
       Replace = '$1from genomic_variant_classifier$2 import ' },
    @{ Find = '(?m)^(\s*)import\s+src(\.[\w.]+)';
       Replace = '$1import genomic_variant_classifier$2' },
    @{ Find = '(?m)^(\s*)from\s+agent_layer(\.[\w.]+)?\s+import\s';
       Replace = '$1from genomic_variant_classifier.agent_layer$2 import ' },
    @{ Find = '(?m)^(\s*)import\s+agent_layer(\.[\w.]+)?';
       Replace = '$1import genomic_variant_classifier.agent_layer$2' },
    @{ Find = '(?m)^(\s*)from\s+(agents|config|message_bus|shared_state)(\.[\w.]+)?\s+import\s';
       Replace = '$1from genomic_variant_classifier.agent_layer.$2$3 import ' },
    @{ Find = '(?m)^(\s*)import\s+(agents|config|message_bus|shared_state)(\.[\w.]+)?';
       Replace = '$1import genomic_variant_classifier.agent_layer.$2$3' },
    @{ Find = '\bsrc\.([\w.]+):';
       Replace = 'genomic_variant_classifier.$1:' }
)
$files = Get-ChildItem -Path . -Recurse -File -Include $includeGlobs -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -notmatch '\\(\.git|\.venv|\.venv312|node_modules|.*\.egg-info|__pycache__|catboost_info)\\' }

$utf8NoBom = New-Object System.Text.UTF8Encoding $false
foreach ($file in $files) {
    $orig = [System.IO.File]::ReadAllText($file.FullName, $utf8NoBom)
    $new = $orig
    foreach ($pat in $patterns) {
        $new = [regex]::Replace($new, $pat.Find, $pat.Replace,
            [System.Text.RegularExpressions.RegexOptions]::Multiline)
    }
    if ($new -ne $orig) {
        [System.IO.File]::WriteAllText($file.FullName, $new, $utf8NoBom)
        Write-Host "PATCHED $($file.FullName)"
    }
}
```

### Action — apply pattern 8 surgically (prose docs only)

```powershell
$proseFiles = @("README.md","METHODS.md","ROADMAP.md","Dockerfile","docker-compose.yml",
                "genomic-variant-classifier.code-workspace") |
              Where-Object { Test-Path $_ }
foreach ($f in $proseFiles) {
    $orig = [System.IO.File]::ReadAllText((Resolve-Path $f), $utf8NoBom)
    $new = [regex]::Replace($orig, '(?<![\w.])src\.([\w][\w.]*[\w])(?![\w.])',
                            'genomic_variant_classifier.$1')
    if ($new -ne $orig) {
        [System.IO.File]::WriteAllText((Resolve-Path $f), $new, $utf8NoBom)
        Write-Host "PROSE-PATCHED $f"
    }
}
```

### Post-checks

```powershell
# 1. Audit: zero stale references remain
$hits = Get-ChildItem -Path . -Recurse -File -Include $includeGlobs -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -notmatch '\\(\.git|\.venv|\.venv312|node_modules|.*\.egg-info|__pycache__|catboost_info)\\' } |
    Select-String -Pattern '\b(from|import)\s+(src|agent_layer|agents|config|message_bus|shared_state)(\.|\s)','\bsrc\.[\w.]+:','\bagent_layer\.[\w.]+:'
if ($hits) {
    $hits | Format-Table -AutoSize
    throw "C3 audit: stale references remain. See above."
}
"AUDIT OK: zero stale references"

# 2. The 14 known import-as-Z lines must all be rewritten
Get-ChildItem -Path tests/unit/test_api.py,tests/unit/test_core.py |
    Select-String -Pattern '^\s*import\s+(src|agent_layer)\.'
# Expected: empty

# 3. Test suite collection still works (catches gross syntax errors from regex sweep)
python -m pytest --collect-only -q 2>&1 | Select-Object -Last 5

# 4. Smoke test (created in scripts/ pre-C2; covered by tests/smoke_test_imports.py)
python tests/smoke_test_imports.py
# Expected: all EXPECTED_IMPORTS resolve under genomic_variant_classifier.*
```

### Commit

```powershell
git status                                                            # review the bulk diff
git diff --stat | Select-Object -Last 5                              # rough magnitude check
git commit -am @"
chore(migration): C3 import sweep - rewrite src.* and agent_layer.* references

Bulk regex sweep across .py, .md, .yml, .yaml, .toml, .cfg, .txt,
.ipynb, .json, .code-workspace, Dockerfile* over the working tree
(excluding .git, .venv*, node_modules, *.egg-info, __pycache__, catboost_info).

8 patterns covering:
- from src.X import Y                  -> from genomic_variant_classifier.X import Y
- import src.X (incl. 'as Z' form)     -> import genomic_variant_classifier.X
- from agent_layer.X import Y          -> from genomic_variant_classifier.agent_layer.X import Y
- import agent_layer.X                 -> import genomic_variant_classifier.agent_layer.X
- bare 'from agents/config/message_bus/shared_state...'
- module strings: 'src.api.main:app'   -> 'genomic_variant_classifier.api.main:app'
- prose dotted-path mentions in README/METHODS/ROADMAP/Dockerfile (surgical, not global)

Verification: post-sweep audit returns zero stale references; pytest --collect-only
exits 0; smoke_test_imports.py passes.

Run-ID: migration-c3
"@
git push origin main
```

### Rollback

`git checkout HEAD~1 -- .` then re-run pre-sweep audit to confirm restoration. Bulk-friendly.

---

## C4 — Pickle migration *(highest-risk, irreversible without backup)*

**Goal:** rewrite the 5 production joblibs to record class paths under `genomic_variant_classifier.*`. Delete the redundant `experiments/` copy. Verify each rewrite via deep object-graph inspection AND prediction equivalence on a frozen fixture.

### Files affected

| Path | Size | Action | Class location |
|---|---|---|---|
| `models/phase2_pipeline.joblib` | 2029.75 MB | Migrate | `src.api.pipeline` → `genomic_variant_classifier.api.pipeline` |
| `models/phase4_pipeline.joblib` | 2036.06 MB | Migrate | `src.api.pipeline` → ditto |
| `models/phase4_pipeline_calibrated.joblib` | 2036.06 MB | Migrate | `src.api.pipeline` → ditto |
| `models/v1/ensemble_v1.joblib` | 1344.55 MB | Migrate | `src.models.variant_ensemble` → `genomic_variant_classifier.models.variant_ensemble` |
| `outputs/run9_ready/models/ensemble.joblib` | 1478.4 MB | Migrate | `src.models.variant_ensemble` → ditto |
| `outputs/run9_ready/scaler.joblib` | 4055 B | Skip | Pure sklearn+numpy; rename-safe |
| **`experiments/2026-04-04_03-39/ensemble_v1.joblib`** | **1344.55 MB** | **Delete** | Byte-identical to `models/v1/ensemble_v1.joblib` (SHA256 `64FEF6…CEE23`); SHA256 lineage recorded in commit message body for forensic audit. Saves 1.34 GB. |

### Pre-checks

```powershell
git rev-parse --short HEAD                                           # Expected: C3 commit
git status --porcelain                                               # Expected: empty

# Confirm scripts/migrate_pickles.py is populated (per the §F-6 deep-aliasing
# implementation) and not the empty placeholder
$migrate = Get-Item .\scripts\migrate_pickles.py
"migrate_pickles.py size: $($migrate.Length) bytes (expected: > 2000)"

# Disk space: need >= 18 GB for .bak originals (existing 9 GB doubles)
$freeGB = [math]::Round(((Get-Item .).PSDrive.Free / 1GB), 1)
"Free space: ${freeGB} GB (need >= 18)"

# Memory: close memory-heavy applications. Each 2 GB joblib expands 2-3x in Python.
# Recommend Task Manager check: <8 GB resident in other processes.

# Smoke test: compat aliases install successfully against current import tree
python -c "from scripts.migrate_pickles import install_compat_aliases; install_compat_aliases(); print('aliases installed OK')"

# Smoke test: verify_no_legacy_modules works on a known-clean object
python -c "from scripts.migrate_pickles import verify_no_legacy_modules; import numpy as np; print(verify_no_legacy_modules(np.array([1,2,3])))"
# Expected: [] (empty list)

# Capture pre-migration SHA256 of every target (for the commit message body)
$targets = @(
    "models\phase2_pipeline.joblib",
    "models\phase4_pipeline.joblib",
    "models\phase4_pipeline_calibrated.joblib",
    "models\v1\ensemble_v1.joblib",
    "outputs\run9_ready\models\ensemble.joblib",
    "experiments\2026-04-04_03-39\ensemble_v1.joblib"
)
foreach ($t in $targets) { "{0,-65}  {1}" -f $t, (Get-FileHash -Algorithm SHA256 $t).Hash }
# Save this output verbatim into the commit message body.

# Frozen fixture for equivalence check exists
Test-Path tests/fixtures/migration_smoke.parquet
# If False: create a 5-10 row sample from any existing input fixture before proceeding.
```

### Action

Run `scripts/migrate_pickles.py` with the deep `pkgutil.walk_packages` aliasing (§F-6) and equivalence checks (§F-8):

```powershell
python -u scripts\migrate_pickles.py
```

The script's main loop, per target file:

1. **Backup**: copy `<target>` to `<target>.bak`. Capture SHA256 of `.bak`.
2. **Install aliases**: `install_compat_aliases()` runs the deep walk over `genomic_variant_classifier.*`, registering every submodule under `src.<sub>` and `agent_layer.<sub>` in `sys.modules`.
3. **Load**: `obj = joblib.load(target)` — succeeds because aliases resolve every submodule reference.
4. **Walk-and-verify (in-process)**: `verify_no_legacy_modules(obj)` should return `[]` immediately after load, since loaded class objects' `__module__` attributes already point to `genomic_variant_classifier.*` (set at class-definition time when the new module was imported).
5. **Re-dump**: `joblib.dump(obj, target, compress=<read-back-from-original>)`. Pickle's `whichmodule(obj, name)` reads `obj.__module__` directly, so re-dump records the new namespace path.
6. **Subprocess verification (no aliases)**: spawn a fresh Python process with no compat aliases installed; load `target`; run `verify_no_legacy_modules(obj)` and assert empty result. This proves the rewrite is genuine, not a runtime artifact of aliasing.
7. **Equivalence check**: run `predict_proba(fixture_X)` on both `<target>.bak` (with aliases) and `<target>` (without); assert `np.testing.assert_allclose(rtol=1e-7, atol=0)`.
8. **Capture post-SHA256**: record into the per-file log.
9. **On all checks passing**: keep `<target>.bak` as a recovery artifact; proceed to next file.

For the deletion target:

10. After all 5 production migrations succeed: capture pre-deletion SHA256 of `experiments/2026-04-04_03-39/ensemble_v1.joblib` (one last verification it still matches the historical `64FEF6…CEE23`); record both SHAs (the historical and the post-migration `models/v1/ensemble_v1.joblib`) in the commit message body; `Remove-Item experiments/2026-04-04_03-39/ensemble_v1.joblib`. Lineage proof preserved in commit message.

### Post-checks

```powershell
# 1. All 5 production targets re-dumped; subprocess verification clean
foreach ($t in $targets[0..4]) {
    $r = python -c "import joblib, sys; sys.path.insert(0,'.'); from scripts.migrate_pickles import verify_no_legacy_modules; o = joblib.load(r'$t'); leftovers = verify_no_legacy_modules(o); print('OK' if not leftovers else 'BAD:' + str(leftovers))"
    "$t : $r"
}
# Expected: all "OK"

# 2. Deletion target removed
Test-Path experiments/2026-04-04_03-39/ensemble_v1.joblib
# Expected: False

# 3. .bak files all present (recovery artifacts)
foreach ($t in $targets[0..4]) {
    $bak = "$t.bak"
    "$bak : $(Test-Path $bak)"
}
# Expected: all True (delete .bak files only after validating C5, never in C4 itself)

# 4. SHA256 audit table for commit message body
"`n=== SHA256 lineage table for commit message ===`n"
foreach ($t in $targets[0..4]) {
    $bak = "$t.bak"
    "{0,-65}  pre={1}" -f $t, (Get-FileHash -Algorithm SHA256 $bak).Hash
    "{0,-65}  post={1}" -f $t, (Get-FileHash -Algorithm SHA256 $t).Hash
}
"experiments\2026-04-04_03-39\ensemble_v1.joblib  pre=64FEF6...CEE23  post=DELETED"
```

### Commit

```powershell
git status                                                            # review: 5 .joblib modified, 1 deleted
git commit -am @"
chore(migration): C4 pickle migration - 5 joblibs rewritten, 1 deleted

Re-pickled with class paths under genomic_variant_classifier.* using
scripts/migrate_pickles.py (deep pkgutil.walk_packages aliasing per
review section F-6 + equivalence check per F-8).

Migrated (5):
  models/phase2_pipeline.joblib                       (src.api.pipeline)
  models/phase4_pipeline.joblib                       (src.api.pipeline)
  models/phase4_pipeline_calibrated.joblib            (src.api.pipeline)
  models/v1/ensemble_v1.joblib                        (src.models.variant_ensemble)
  outputs/run9_ready/models/ensemble.joblib           (src.models.variant_ensemble)

Deleted (1):
  experiments/2026-04-04_03-39/ensemble_v1.joblib     1344.55 MB
  Pre-deletion SHA256: 64FEF61170E98FF722C338734819008A4F8307111D6AB97EC6BE14A29ADCEE23
  Byte-identical to models/v1/ensemble_v1.joblib at HEAD 12b8315 (verified).
  Saves 1.34 GB. SHA256 lineage proves the artifact is recoverable from the migrated copy.

Verification per file:
1. Subprocess load (no aliases) succeeds with verify_no_legacy_modules() == []
2. predict_proba(fixture) matches .bak via np.testing.assert_allclose(rtol=1e-7)
3. .bak originals retained until C5 closes successfully

Pre/post SHA256 lineage:
[paste the post-checks SHA256 audit table here verbatim]

Run-ID: migration-c4
"@
git push origin main
```

### Rollback

Per-file, sequentially: `Remove-Item <target>` then `Move-Item <target>.bak <target>`. Re-run `verify_no_legacy_modules` post-restore. The `experiments/.../ensemble_v1.joblib` deletion can be reversed by copying from `models/v1/ensemble_v1.joblib.bak` (which still has the original `src.models.variant_ensemble` references) — the SHA256 lineage in the commit message proves equivalence.

---

## C5 — Docs + CI fix + final audit

**Goal:** update README, active docs, historical docs, the CI workflow with the one-line `drift_monitor.yml:215` fix, then run a wider audit to prove the migration is complete.

### Pre-checks

```powershell
git rev-parse --short HEAD                                           # Expected: C4 commit
git status --porcelain                                               # Expected: empty
python -c "import genomic_variant_classifier; print('OK')"           # Expected: OK
python -m pytest --collect-only -q 2>&1 | Select-Object -Last 5      # Expected: collected, no errors
```

### Action — README.md surgical edits

```powershell
# Line 196: 'src/' standalone in directory tree -> 'src/genomic_variant_classifier/'
# Line 223: 'uvicorn src.api.main:app' -> 'uvicorn genomic_variant_classifier.api.main:app'
# (verify line numbers haven't shifted from prior edits with: grep -n 'src/' README.md)
# Apply manually with VS Code or:
$readme = [System.IO.File]::ReadAllText("README.md", $utf8NoBom)
$readme = $readme -replace 'uvicorn src\.api\.main:app','uvicorn genomic_variant_classifier.api.main:app'
# directory-tree edit is contextual — review and edit by hand
[System.IO.File]::WriteAllText("README.md", $readme, $utf8NoBom)
```

### Action — `.github/workflows/drift_monitor.yml:215` (the original spec's "C3 = CI fix" item)

```powershell
# Line 215 contains 'src.' or 'src/' in some module-string or python -m form.
# Inspect:
Get-Content .github/workflows/drift_monitor.yml | Select-Object -Index 213,214,215,216 | ForEach-Object {$i++; "$($i+213): $_"}
# Apply the correct namespace fix manually (line is reviewer-context; usually `python -m src.X.Y`)
```

### Action — historical doc footnotes

For each file in `ROADMAP.md`, `METHODS.md`, `PHASE_1_ASSESSMENT.md`, `PHASE_2_FEATURES.md`, and any pre-migration incident/session doc that still references `src.*` paths in narrative form, add a footnote near the top (after frontmatter if present):

> _**[Layout note]** Module paths shown as `src.X` or `agent_layer.X` reflect the pre-2026-05-08 layout. After commit `<C2-commit-hash>` the canonical namespace is `genomic_variant_classifier.X` (and `genomic_variant_classifier.agent_layer.X`). Where a path appears in this historical context, mentally substitute the new namespace._

Skip docs already rewritten by the C3 sweep — those don't need a footnote.

### Action — final wide audit

```powershell
# Same audit pattern as C3 post-checks, run AGAIN over the wider tree
$hits = Get-ChildItem -Path . -Recurse -File -Include $includeGlobs -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -notmatch '\\(\.git|\.venv|\.venv312|node_modules|.*\.egg-info|__pycache__|catboost_info)\\' } |
    Where-Object { $_.Name -notlike '*.bak' } |
    Select-String -Pattern '\b(from|import)\s+(src|agent_layer|agents|config|message_bus|shared_state)(\.|\s)','\bsrc\.[\w.]+:','\bagent_layer\.[\w.]+:'

if ($hits) {
    $hits | Format-Table -AutoSize
    Write-Host "C5 audit: $($hits.Count) stale references remain (see above)" -ForegroundColor Yellow
    Write-Host "Decide per-hit: surgical edit + re-audit, OR add to footnote-acceptable historical context list." -ForegroundColor Yellow
    # If all remaining hits are inside historical incident/session docs that already
    # carry the [Layout note] footnote, that's acceptable. Otherwise: stop and resolve.
} else {
    "FINAL AUDIT OK: zero stale references."
}

# Re-validate all docs/hypotheses/, docs/validated/ frontmatter post-edits
python validate_docs.py
```

### Action — `.bak` cleanup (only after audit OK and equivalence checks confirmed)

```powershell
# Remove the C4 .bak files now that everything is validated
foreach ($t in @(
    "models\phase2_pipeline.joblib",
    "models\phase4_pipeline.joblib",
    "models\phase4_pipeline_calibrated.joblib",
    "models\v1\ensemble_v1.joblib",
    "outputs\run9_ready\models\ensemble.joblib"
)) {
    Remove-Item "$t.bak" -Force
}
```

### Action — `.gitignore` hardening

Add (if not already present): `*.egg-info/`, `build/`, `dist/`, `*.bak`.

### Post-checks

```powershell
# 1. Final audit clean
"(re-run the audit block above; expected: FINAL AUDIT OK)"

# 2. Test suite green
python -m pytest -q 2>&1 | Select-Object -Last 10
# Expected: 0 failures (modulo pre-existing failures unrelated to the migration)

# 3. Production smoke: load each migrated joblib successfully
foreach ($t in $targets[0..4]) {
    python -c "import joblib; o = joblib.load(r'$t'); print(type(o).__module__, type(o).__name__)"
}
# Expected: all start with 'genomic_variant_classifier.'

# 4. Wheel build (optional smoke test; not blocking)
python -m build --wheel 2>&1 | Select-Object -Last 5
# Expected: builds without error; produces dist/genomic_variant_classifier-0.1.0-py3-none-any.whl
```

### Commit

```powershell
git status                                                            # review: README.md, drift_monitor.yml, footnoted historical docs, .gitignore, removed .bak files
git commit -am @"
chore(migration): C5 docs + CI fix + final audit

- README.md lines 196, 223: directory tree + uvicorn module string
- .github/workflows/drift_monitor.yml line 215: namespace fix
- Historical-doc footnotes added to ROADMAP.md, METHODS.md, PHASE_*.md,
  pre-migration incident/session docs
- .gitignore hardened: *.egg-info/, build/, dist/, *.bak
- C4 .bak files removed (audit + equivalence checks both passed)

Final audit: zero stale src.*, agent_layer.*, agents.*, config.*,
message_bus.*, shared_state.* references in tracked .py/.md/.yml/.yaml/
.toml/.cfg/.txt/.ipynb/.json/.code-workspace/Dockerfile* outside
[Layout note]-bearing historical docs.

Run-ID: migration-c5
"@
git push origin main
```

### Rollback

Cosmetic; `git checkout HEAD~1 -- <file>` per-file as needed. The audit re-run from C3 establishes regression freedom.

---

# Smoke test: `tests/smoke_test_imports.py`

Populated pre-C2 (currently empty placeholder). Runs from any cwd to verify every expected module under `genomic_variant_classifier.*` resolves.

```python
"""Smoke test: every expected module imports cleanly under the new namespace.

Runs from any cwd. Used as a post-C2 gate and a C3 verification step.

EXPECTED_IMPORTS is built FROM THE FILESYSTEM (src/genomic_variant_classifier/)
not from a hardcoded list, to avoid the §A9/§A10 review concern about aspirational
entries failing the smoke test for legitimate "file does not exist" reasons.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys

import genomic_variant_classifier as _root


def collect_expected_imports() -> list[str]:
    return sorted(
        qualname
        for _finder, qualname, _ispkg in pkgutil.walk_packages(
            _root.__path__, prefix="genomic_variant_classifier."
        )
    )


def main() -> int:
    expected = collect_expected_imports()
    failures: list[tuple[str, str]] = []
    for qualname in expected:
        try:
            importlib.import_module(qualname)
        except Exception as exc:  # noqa: BLE001 — surface any failure
            failures.append((qualname, f"{type(exc).__name__}: {exc}"))
    if failures:
        print(f"SMOKE TEST FAILED: {len(failures)} import(s) broken:", file=sys.stderr)
        for qualname, msg in failures:
            print(f"  {qualname}\n    {msg}", file=sys.stderr)
        return 1
    print(f"SMOKE TEST OK: {len(expected)} module(s) imported.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

---

# `scripts/migrate_pickles.py`

Populated pre-C4 (currently empty placeholder). Implements the §F-6 deep aliasing, §F-8 equivalence check, and §F-7 = delete handling. Full reference implementation provided in a separate sub-batch (next deliverable after this spec lands).

Public API (must remain stable for the C5 verification subprocess):
- `install_compat_aliases() -> None`
- `verify_no_legacy_modules(obj) -> list[str]`
- `migrate_one(path: Path) -> None`
- `delete_one(path: Path) -> str`  (returns SHA256 of the deleted file for the audit log)
- `equivalence_check(bak_path, new_path, fixture_X) -> None`
- `main() -> int`

`TARGETS` and `DELETIONS` are module-level constants:
- `TARGETS = [Path("models/phase2_pipeline.joblib"), Path("models/phase4_pipeline.joblib"), Path("models/phase4_pipeline_calibrated.joblib"), Path("models/v1/ensemble_v1.joblib"), Path("outputs/run9_ready/models/ensemble.joblib")]`
- `DELETIONS = [Path("experiments/2026-04-04_03-39/ensemble_v1.joblib")]`

---

# Time budget

Total: **6–8 hours** (revised upward from the original spec's 4.25–6.75 estimate per review's closing note, given deep equivalence checks).

| Commit | Hands-on | I/O-bound (waiting) | Total |
|---|---|---|---|
| C1 (move) | 15 min | < 1 min | ~15 min |
| C2 (pyproject + install) | 30 min | 2–5 min | ~35 min |
| C3 (sweep + audit) | 45 min | 5 min | ~50 min |
| C4 (pickles) | 60–90 min | **2.5–4 hr** (load + dump + verify × 5) | **3.5–5.5 hr** |
| C5 (docs + audit) | 45 min | 2 min | ~50 min |

C4 dominates. Run it on a day with no other heavy laptop workload. Keep the laptop plugged in (suspend during a 2 GB joblib load can corrupt the partial dump).

# Success criteria

The migration is complete when ALL of the following hold:

1. `git log --oneline` shows commits for C1, C2, C3, C4, C5 in order, all pushed to `origin/main`.
2. `python -c "import genomic_variant_classifier"` succeeds.
3. `python -c "import src.api.main"` raises `ModuleNotFoundError` (the old namespace is genuinely gone).
4. `python -m pytest` passes the same set of tests it passed at HEAD `12b8315` (modulo pre-existing failures unrelated to the migration; equivalence is established by diffing the pre/post test reports).
5. `python tests/smoke_test_imports.py` exits 0.
6. C5 final audit returns zero stale references (excluding documented `[Layout note]` historical docs).
7. All 5 migrated joblibs load via `joblib.load` in a fresh Python process with NO compat aliases, and each loaded object's `verify_no_legacy_modules` returns `[]`.
8. Equivalence check: for each migrated joblib, `predict_proba(fixture)` matches `.bak` via `np.testing.assert_allclose(rtol=1e-7)`.
9. SHA256 lineage for `experiments/2026-04-04_03-39/ensemble_v1.joblib` recorded in C4 commit body, file deleted.
10. `validate_docs.py` exits 0 on `docs/hypotheses/` and `docs/validated/`.

# Post-migration follow-ups

After the migration lands, promote this hypothesis to a validated rule by moving it to `docs/validated/RULE_consolidate-package-layout.md` (with `status: validated` frontmatter) and document the executed commit hashes for C1–C5.

Defer to separate hypotheses or roadmap items:
- Phase 4 cheap rider: 6 stray top-level `.py` files + 6 root log files cleanup
- NetworkX `gpickle` deprecation hygiene for `string_graph_700.pkl`
- `pip install -e ".[api]"` populating real `requirements-api.in` contents into pyproject's `[project.optional-dependencies]`
- Wheel-build CI check
- Production API rollout coordination if/when the API service is hosted

# Risk register cross-reference

See `docs/reviews/REVIEW_HYP_consolidate-package-layout_2026-05-08.md` §C for full per-commit failure modes, blast radii, and rollback maturity grades. Highest-risk: C4's deep aliasing (§B-C4-1, §D-1) and the equivalence test absence (§B-X-2) — both addressed in this spec via §F-6 and §F-8 mechanisms.

# Run-ID convention

Manual discipline only (no `.pre-commit-config.yaml` or `check_run_id_trailer.py` installed). Append `Run-ID: migration-cN` to each commit message body. The audit trail is the commit history itself.
