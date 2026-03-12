# Phase 1 Code Assessment — Genomic Variant Classifier
**Repository:** `monzia-moodie/genomic-variant-classifier`
**Assessed:** 74 notebook cells · ~6,300 lines of Python

---

## Executive Summary

The Phase 1 notebook contains a conceptually sound, architecturally ambitious
genomic variant classification pipeline. The data model (canonical variant
schema), the 8-model ensemble with stacking meta-learner, the gene-aware
train/test split, and the HTML reporting framework are all well-designed
abstractions that will serve Phase 2 well.

However, **7 critical bugs** prevent the notebook from running at all, and
**19 additional issues** ranging from architectural inconsistencies to logic
errors and code quality problems will cause silent failures or incorrect
results if left unfixed.

All issues have been corrected in the `src/` and `scripts/` files delivered
alongside this assessment.

---

## Bug Severity Reference

| Level | Definition |
|-------|-----------|
| **Critical** | Raises an exception before the pipeline reaches any real work |
| **Design** | Causes incorrect results or makes the system hard to extend |
| **Quality** | No crash, but misleading, duplicated, or fragile code |

---

## Part 1 — Critical Bugs (7)

These bugs cause the notebook to fail completely on first run.

---

### Bug 1 — `userdata` not imported before first use
**Location:** Cell 2
**Exception:** `NameError: name 'userdata' is not defined`

**Original:**
```python
# Cell 2
token = userdata.get('GITHUB_TOKEN')   # ← userdata not yet imported
!git clone https://{token}@...
```
```python
# Cell 3
from google.colab import userdata      # ← import appears here, one cell too late
```

**Fix:** Move `from google.colab import userdata` to the top of Cell 2, before
any call to `userdata.get()`. Cell 3 now imports only the additional secrets
that are needed later (`NCBI_API_KEY`, `OMIM_API_KEY`).

---

### Bug 2 — Python variable not interpolated inside Colab shell command
**Location:** Cell 2
**Symptom:** Git clones with the literal string `{token}` instead of the real token value.

**Original:**
```python
token = userdata.get('GITHUB_TOKEN')
!git clone https://{token}@github.com/monzia-moodie/genomic-variant-classifier.git
```
Colab's `!`-prefix shell commands do **not** support Python f-string syntax.
`{token}` is passed literally to git, which then fails to authenticate.

**Fix:** Build the full URL in Python, then pass the variable to the shell via
`subprocess.run()`:
```python
repo_url = f"https://{token}@github.com/monzia-moodie/genomic-variant-classifier.git"
import subprocess
subprocess.run(["git", "clone", repo_url], check=True)
```

---

### Bug 3 — Six modules defined as string literals or inline code, never written to disk
**Location:** Cells 49, 64, 65, 67, 68, 69
**Exception:** `ModuleNotFoundError` on every subsequent `import` of these modules.

This is the most pervasive bug. Six major pipeline modules were either written
as bare `"""..."""` triple-quoted string expressions (the interpreter evaluates
and discards the string; nothing is written to disk), or executed inline
without a `%%writefile` directive.

| Cell | Module path intended | What actually happened |
|------|---------------------|----------------------|
| 49   | `src/data/spark_etl.py` | Code ran inline; no file created |
| 64   | `src/models/gnn.py` | Code ran inline; no file created |
| 65   | `src/reports/report_generator.py` | Bare string literal; file not created |
| 67   | `src/data/real_data_prep.py` | Bare string literal; file not created |
| 68   | `scripts/train.py` | Bare string literal; file not created |
| 69   | `src/evaluation/evaluator.py` | Bare string literal; file not created |

**Fix:** Each was rewritten as a proper Python source file (delivered in the
corrected `src/` tree). In a notebook context, adding `%%writefile path/to/file.py`
at the top of each cell achieves the same effect.

---

### Bug 4 — Module path inconsistency: `src/data_ingestion/` vs `src/data/`
**Location:** All cells that import database connectors.
**Exception:** `ModuleNotFoundError: No module named 'src.data.database_connectors'`

`database_connectors.py` was written to `src/data_ingestion/` but every
subsequent import used the path `src.data.database_connectors`.

**Fix:** Consolidated all data-layer modules under `src/data/`. The directory
`src/data_ingestion/` is eliminated. The corrected layout is:
```
src/data/
    database_connectors.py
    spark_etl.py
    real_data_prep.py
```

---

### Bug 5 — Wrong class name imported in `scripts/train.py`
**Location:** `scripts/train.py`
**Exception:** `ImportError: cannot import name 'VariantEnsemble' from 'src.models.ensemble'`

```python
# Original (wrong):
from src.models.ensemble import VariantEnsemble, EnsembleConfig
```
`src/models/ensemble.py` exports `EnsembleClassifier`, not `VariantEnsemble`.
`VariantEnsemble` lives in `src/models/variant_ensemble.py`.

**Fix:**
```python
from src.models.variant_ensemble import VariantEnsemble, EnsembleConfig
```

---

### Bug 6 — `nx.read_gpickle` / `nx.write_gpickle` removed in NetworkX 3.3+
**Location:** `src/models/gnn.py`
**Exception:** `AttributeError: module 'networkx' has no attribute 'read_gpickle'`

These helpers were removed from NetworkX in version 3.3 (February 2024). The
`requirements.txt` pins `networkx==3.2.1`, but a Phase 2 dependency update
will immediately break graph caching.

**Fix:** Replace with stdlib `pickle`, which has no external dependencies:
```python
import pickle

def _save_graph(self, G, path):
    with open(path, "wb") as fh:
        pickle.dump(G, fh, protocol=pickle.HIGHEST_PROTOCOL)

def _load_graph(self, path):
    with open(path, "rb") as fh:
        return pickle.load(fh)
```
Cache files renamed from `.gpickle` → `.pkl` to make the format explicit.

---

### Bug 7 — `encode_sequence(seq, k=3)`: `k` accepted but never used
**Location:** `src/models/variant_ensemble.py`

The function accepts a `k` parameter (implying k-mer encoding) but implements
only one-hot encoding, ignoring `k` entirely. This is both a broken API
promise and misleading documentation.

**Fix:**
- Remove the `k` parameter from the signature.
- Correct the docstring to describe one-hot encoding only.
- Add `k` to `PHASE_2_FEATURES` with a `# TODO: Phase 2` comment to preserve
  intent for future implementation.

---

## Part 2 — Architecture Issues (5)

These cause conflicts or make the codebase hard to maintain.

---

### Issue A — Two competing ensemble implementations
**Location:** `src/models/ensemble.py` and `src/models/variant_ensemble.py`

Both files define `EnsembleConfig` with different fields. Callers using the
wrong import get a config that silently ignores parameters.

**Fix:** Consolidated into `src/models/variant_ensemble.py`. `ensemble.py`
removed. `EnsembleClassifier` retained as an alias for backward compatibility.

---

### Issue B — Duplicate Spark ETL pipelines
**Location:** Cells 71 and 72

Both cells define `create_spark_session()`, `normalize_variants()`, and
`run_etl_pipeline()`. Cell 72 supersedes Cell 71 with a more complete
implementation.

**Fix:** Cell 71 removed. Single canonical Spark session factory in
`src/data/spark_etl.py`.

---

### Issue C — Utility functions defined directly in `src/utils/__init__.py`
**Location:** `src/utils/__init__.py`

Defining implementation in `__init__.py` conflates the package namespace with
the module, makes individual functions hard to mock in tests, and violates
the convention that `__init__.py` should only re-export.

**Fix:** Functions moved to `src/utils/helpers.py`. `__init__.py` re-exports
them so all existing `from src.utils import make_variant_id` calls continue
to work without change.

---

### Issue D — `src/reports/` has no `__init__.py`
**Location:** `src/reports/`

Without an `__init__.py`, `from src.reports import ReportGenerator` fails.
Callers must use the full `from src.reports.report_generator import ...` path,
which breaks the clean package API.

**Fix:** `src/reports/__init__.py` added, re-exporting the public API.

---

### Issue E — Identical synthetic data generation in two cells
**Location:** Cells 51 and 59

Both cells define and run the same 80-line synthetic DataFrame generator. Any
schema change must be made twice and the versions will inevitably drift.

**Fix:** Extracted to `tests/fixtures/make_synthetic_data.py` as the single
canonical source. Both the notebook and unit tests import `make_synthetic_variants()`
from there.

---

## Part 3 — Design & Logic Issues (6)

These produce incorrect model behavior or misleading results.

---

### Issue F — `filter_high_quality(min_quality=2)` parameter never used
**Location:** `src/data/database_connectors.py`

```python
def filter_high_quality(self, df, min_quality=2):
    return df[df["pathogenicity"].notna()]   # min_quality ignored
```
The filter always returns everything with a non-null pathogenicity regardless
of `min_quality`, allowing low-confidence VUS and conflicting interpretations
to pollute the training set.

**Fix:** Renamed parameter to `min_review_tier` and implemented actual ClinVar
review status filtering using the `REVIEW_STATUS_TIER` mapping.

---

### Issue G — `_map_pathogenicity` uses exact set membership; misses compound values
**Location:** `src/data/database_connectors.py`

ClinVar commonly uses compound clinical significance strings such as
`"Pathogenic, risk factor"` or `"Pathogenic/Likely pathogenic"`. Exact set
lookup returns `"uncertain"` for all compound values, silently dropping a
significant fraction of clearly pathogenic records.

**Fix:** Switched to substring matching, checking "likely pathogenic" before
"pathogenic" to avoid false positives:
```python
v = str(value).lower()
if "likely pathogenic" in v: return "likely_pathogenic"
if "pathogenic"        in v: return "pathogenic"
if "likely benign"     in v: return "likely_benign"
if "benign"            in v: return "benign"
return "uncertain"
```

---

### Issue H — `VariantEnsemble.fit()` leaves unfitted models in memory
**Location:** `src/models/variant_ensemble.py`

After `fit()`, trained models are stored in `self.trained_models_` but the
original estimators remain in `self.base_estimators`, doubling memory
consumption for large tree ensembles (each RandomForest can exceed 500 MB).

**Fix:** Added `self.base_estimators.clear()` at the end of `fit()`.

---

### Issue I — Gene-aware split raises unhelpful `ValueError`
**Location:** `src/data/real_data_prep.py`

When a split fold contained only one class, the code raised a bare
`ValueError` with no actionable information about what happened or how to fix it.

**Fix:** Pre-split class balance validation with a diagnostic message:
```python
raise ValueError(
    f"Split '{split_name}' missing class(es): {classes}. "
    "Lower min_review_tier or increase dataset size."
)
```

---

### Issue J — `load_all_databases()` always re-downloads ClinVar (~500 MB)
**Location:** `src/data/database_connectors.py`

There was no way to pass a pre-downloaded ClinVar file. Every run re-downloaded
the full ClinVar XML (~500 MB), making iteration expensive.

**Fix:** Added optional `clinvar_path` parameter to `ClinVarConnector.fetch()`
and `load_all_databases()`. If the file exists locally, the download is skipped.

---

### Issue O — `Template.globals["format_int"]` raises `AttributeError`
**Location:** `src/reports/report_generator.py`

```python
# Original (wrong — Template has no .globals):
self.template = Template(HTML_TEMPLATE)
self.template.globals["format_int"] = lambda x: f"{int(x):,}"
```
`.globals` is an attribute of Jinja2 `Environment` objects, not `Template` objects.
This raises `AttributeError` the first time a report is generated.

**Fix:** Use `jinja2.Environment` with a custom filter:
```python
env = Environment(autoescape=False)
env.filters["format_int"] = lambda x: f"{int(x):,}"
template = env.from_string(HTML_TEMPLATE)
```

---

## Part 4 — Code Quality Issues (8)

These do not cause crashes but produce misleading, duplicated, or fragile code.

---

### Issue K — Two conflicting requirements files
`requirements.txt` and `requirements-production.txt` have overlapping but
incompatible version specifications.

**Fix:** Single `requirements.txt` with lower-bound pins (`>=`). A separate
`requirements-lock.txt` with exact frozen versions for reproducible CI.

---

### Issue L — `logging.basicConfig()` called at module level in library files
Multiple library modules called `logging.basicConfig()` at import time. The
first such call wins and silently overrides any configuration the application
set up — a standard Python anti-pattern for library code.

**Fix:** All `logging.basicConfig()` calls removed from library modules.
The only remaining call is at the top of `scripts/train.py`, before any
pipeline imports.

---

### Issue M — `create_spark_session()` defined three times
Defined independently in cells 49, 71, and 72 with slightly different
configuration.

**Fix:** Single canonical implementation in `src/data/spark_etl.py`.

---

### Issue N — Inconsistent `from __future__ import annotations`
Present in some files, missing in others, causing inconsistent behavior with
forward-type references in Python 3.9.

**Fix:** Added to all source files.

---

### Issue P — `codon_position` always 0 in `TABULAR_FEATURES`
`codon_position` was included in `TABULAR_FEATURES` but always computed as 0
(the VEP annotation step that would populate it was not implemented). Including
a constant feature in a tree model wastes every potential split point.

**Fix:** Removed from `TABULAR_FEATURES`. Added to `PHASE_2_FEATURES`:
```python
PHASE_2_FEATURES = [
    "codon_position",       # TODO Phase 2: from VEP HGVSp parsing
    "splice_ai_score",      # TODO Phase 2: SpliceAI predictions
    "alphamissense_score",  # TODO Phase 2: AlphaMissense API
]
```

---

### Issue Q — GitHub token visible in git clone output
The original command printed the full HTTPS URL (including the token) to
notebook output, which could be captured in notebook metadata.

**Fix:** Using `subprocess.run()` with `capture_output=True` prevents the URL
from appearing in rendered output.

---

### Issue R — Hardcoded personal email in `git config`
```python
!git config --global user.email "your-email@gmail.com"
```
This line commits a personal email address to a shared public notebook.

**Fix:** Replaced with a comment instructing the developer to fill in their
email before committing.

---

### Issue S — `GeneErrorAnalysis` serialization: fragile `itertuples()` unpack
**Location:** `src/evaluation/evaluator.py`

```python
# Original (fragile):
return [GeneErrorAnalysis(**row._asdict()) for row in gene_errors.itertuples(index=False)]
```
pandas renames DataFrame columns that conflict with NamedTuple reserved
attributes (e.g., a column named `index`) by prepending an underscore,
causing `TypeError` or `KeyError` on unpack.

**Fix:** Use `to_dict(orient="records")` which returns plain dicts immune to
NamedTuple reserved-name collision:
```python
return [GeneErrorAnalysis(**row) for row in gene_errors.to_dict(orient="records")]
```

---

## Corrected Files Delivered

| File | Bugs / Issues fixed |
|------|-------------------|
| `src/data/database_connectors.py` | Bug 4, Issues F, G, J, L, N |
| `src/data/spark_etl.py` | Bug 3, Issues B, L, M, N |
| `src/data/real_data_prep.py` | Bug 3, Issues F, G, I, L, N |
| `src/models/variant_ensemble.py` | Bugs 5, 7, Issues A, H, L, N, P |
| `src/models/gnn.py` | Bugs 3, 6, Issues L, N |
| `src/reports/__init__.py` | Issue D |
| `src/reports/report_generator.py` | Bug 3, Issues L, N, O |
| `src/evaluation/__init__.py` | new package init |
| `src/evaluation/evaluator.py` | Bug 3, Issues L, N, S |
| `src/utils/__init__.py` | Issue C |
| `src/utils/helpers.py` | Issue C, N |
| `scripts/train.py` | Bugs 3, 5, Issues L, N |
| `tests/unit/test_core.py` | Bugs 4, 5; all import paths updated |
| `tests/fixtures/make_synthetic_data.py` | Issue E |
| `NOTEBOOK_CELL_FIXES.py` | Bugs 1, 2, Issues Q, R |

---

## Phase 2 Readiness Checklist

Before beginning Phase 2, confirm the following:

- [ ] Run `pytest tests/unit/test_core.py -v` — all tests should pass
- [ ] Run `python scripts/train.py --fast --skip-nn` against synthetic data
      and confirm AUROC > 0.75
- [ ] Download ClinVar GRCh38 XML; run `database_connectors.py` to produce
      `data/processed/clinvar_grch38.parquet`
- [ ] (Optional) Download gnomAD v4 exome VCF; run gnomAD connector
- [ ] Implement `codon_position`, `splice_ai_score`, `alphamissense_score`
      from `PHASE_2_FEATURES` using VEP annotations — these are the largest
      available signal improvements for Phase 2
- [ ] Confirm ClinVar-to-gnomAD locus join is enriching AF correctly
- [ ] Evaluate AUROC on real ClinVar data; target ≥ 0.90 before clinical validation
