# INCIDENT 2026-05-09: C1 and C3 sweep misses surface during C4-prep

## TL;DR

Two regressions introduced by the consolidate-package-layout migration
series (C1, C3) went undetected through C3.x because the test suite never
exercised the affected code paths. They surfaced together when
`scripts/migrate_pickles.py`'s `install_compat_aliases` function was first
invoked with the full alias-walk loop — first as an `AttributeError` at
L122, then as 6 `ModuleNotFoundError` walk_failures.

Both regressions resolved in commit `e0f4c6e` (C3.6 hotfix).

## Regression 1: C1 sweep miss — missing `agent_layer/__init__.py`

### What

`src/genomic_variant_classifier/agent_layer/` had no `__init__.py` file.
This left it as a PEP 420 namespace package.

### Why it matters

`pkgutil.walk_packages` does NOT recurse into namespace packages by
default. The `install_compat_aliases` walk over
`genomic_variant_classifier.__path__` yielded 52 entries, with **zero
under `agent_layer.*`**. Direct walk over `agent_layer.__path__` yielded
all 27 expected children — confirming the namespace-vs-regular distinction
as the cause.

### Why it didn't surface earlier

- C2 spec stated `agent_layer/` should have an empty `__init__.py` at the
  new location, but the implementation step never created it.
- Other code paths (direct imports like `from
  genomic_variant_classifier.agent_layer.message_bus import X`) worked
  because Python resolves namespace packages individually for explicit
  imports. Walk-based discovery is the only thing that fails silently.
- The CI test suite did not invoke `pkgutil.walk_packages` over the
  package tree.

### Fix

Created empty `src/genomic_variant_classifier/agent_layer/__init__.py`
(0 bytes). `importlib.util.find_spec('genomic_variant_classifier.agent_layer')`
now returns `spec.origin` set, confirming REGULAR package status.

## Regression 2: C3 sweep miss — bare imports in 8 files

### What

8 files retained bare top-level imports of `agents` /  `config` /
`message_bus` / `shared_state`:

| File | Lines |
|------|-------|
| `agent_layer/agents/base_agent.py` | 3 |
| `agent_layer/agents/data_freshness_agent.py` | 4 |
| `agent_layer/agents/interpretability_agent.py` | 4 |
| `agent_layer/agents/literature_scout_agent.py` | 4 |
| `agent_layer/agents/training_lifecycle_agent.py` | 4 |
| `agent_layer/orchestrator.py` | 7 |
| `agent_layer/run_agents.py` | 1 |
| `agent_layer/test_message_bus.py` | 17 |
| **Total** | **44** |

C3's regex patterns 6 and 7 should have rewritten these to fully-qualified
form (`from genomic_variant_classifier.agent_layer.<name> import ...`).

### Why it matters

Pre-migration, `find_packages()` at repo root discovered `agent_layer/`
AND its subpackages (`agents/`, `config.py`, `message_bus.py`,
`shared_state.py`) as TOP-LEVEL packages. So bare `from agents import X`
resolved correctly because `agents` was a discoverable top-level package
on `sys.path`.

After C1 nested `agent_layer` under `genomic_variant_classifier/`, those
bare names no longer resolve. The 6 modules with these imports failed
at module-load time with `ModuleNotFoundError` when
`install_compat_aliases` tried to alias them via `walk_packages`.

Critically: pickled classes inside the production joblibs may carry
`__module__` values like `agents.base_agent` or `config.<...>` (depending
on which import form was used at the time of class definition). If the
walk fails to alias `genomic_variant_classifier.agent_layer.agents.base_agent`
under both `src.*` AND `agent_layer.*` prefixes, `joblib.load` would
crash mid-unpickling on at least one of the 5 production joblibs in the
real C4 run.

### Why it didn't surface earlier

- The pre-C4 CI test suite imports modules via the new fully-qualified
  path. The bare-import lines lived in modules that were not directly
  exercised at import time by tests.
- Walk-based discovery (which DOES eagerly evaluate every module) was
  not part of CI either.
- `run_agents.py` and `test_message_bus.py` had bare imports that did
  not even surface in `install_compat_aliases`'s walk_failures, but the
  C3.6 sweep caught them via static regex anyway.

### Fix

Sweep script `agent_data/c4_fix_bare_imports.py` applied C3 spec patterns
6 and 7 with `\b` word-boundary hardening over
`src/genomic_variant_classifier/**/*.py`. 44 lines rewritten in 8 files,
+1716 bytes.

The `\b` hardening prevents accidental over-match against names like
`agents_helper` (no such names exist in current codebase, but the spec
patterns lacked this safeguard).

## Verification (post-fix)

```
src.* aliases registered:        81 (was 75 pre-fix; 53 pre-__init__.py)
agent_layer.* aliases registered: 28 (was 22 pre-fix;  1 pre-__init__.py)
walk_failures:                     0
verify_no_legacy_modules(np.array): []
```

All 27 expected children of `agent_layer` now register, plus the
top-level `setdefault` itself = 28 total. Clean.

End-to-end smoke on `outputs/run9_ready/scaler.joblib` (4 KB):
`TYPE=StandardScaler | MOD=sklearn.preprocessing._data | LEGACY=[]`

## Forensic timeline

| Commit | Note |
|--------|------|
| C1 (parent of `66fdbfe`) | Created `src/` layout; missed `agent_layer/__init__.py` |
| C2 spec | Specified the `__init__.py` as a deliverable; not implemented |
| C3 (`66fdbfe`) | Regex sweep of imports; missed 8 files with bare names |
| C3.1 - C3.5 (`fc7f63a`) | Focused on Path joins, CI install, Dockerfile; no agent_layer touch |
| 2026-05-09 (this session) | Both regressions surfaced during C4-prep |
| C3.6 (`e0f4c6e`) | Both regressions resolved |

## Lessons

1. **Migration sweeps need a full-tree post-condition test.** A simple
   `pkgutil.walk_packages` import-all check would have caught both
   regressions immediately. Add to CI for any future migration.

2. **Removal of namespace ambiguity** (`find_packages()` discovering
   subpackages as top-level) requires sweeping ALL bare imports of
   those former top-level names. C3's regex patterns are correct but
   were applied incompletely; root cause for the miss remains
   undiagnosed (file-glob omission, vs. re-introduction during
   C3.x — neither verified).

3. **Spec items like "create empty `__init__.py`" need explicit
   verification in the implementation step.** The C2 spec called for it;
   the implementation did not. Future spec implementations should have
   an "implementation checklist" with one test per deliverable.

4. **Defense-in-depth in regex sweeps**: `\b` word boundaries are a
   strict improvement over the spec patterns. Adopt for any future
   sweep of similar shape.

## References

- Diagnostic script: `agent_data/c4_diagnose_walk.py` (Stage 2.5c output
  in session log)
- Fix script: `agent_data/c4_fix_bare_imports.py` (Stage 2.5e + retry)
- Spec: `docs/hypotheses/HYP_consolidate-package-layout.md` (C1, C2, C3
  sections)
- Resolution commit: `e0f4c6e`
- Session log: `docs/sessions/SESSION_2026-05-09.md`