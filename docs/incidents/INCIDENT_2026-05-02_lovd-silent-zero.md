# INCIDENT 2026-05-02 — LOVD silent-zero in run9_ready cohort

## Status

OPEN — pending R10-A log grep to distinguish two remaining root-cause
candidates. Run 9 launch path unaffected. Run 9 will inherit the same
silent-zero baseline as run9_ready (Test AUROC 0.9814).

## Summary

`lovd_variant_class` is identically `0` for all 1,197,216 rows in
`outputs/run9_ready/splits/X_train.parquet` despite the LOVD parquet
on disk being structurally healthy (18,006 rows, joinable schema) and
the LOVD connector being unconditionally invoked at
`src/data/real_data_prep.py:738`. A diagnostic merge replicating the
connector's exact key-construction logic against the same ClinVar
enriched cache yields 5,553 inner-join matches in isolation. The cause
is at one of the runtime join boundaries inside the ETL pipeline, not
in the connector code or the on-disk data.

## Cross-references

- `docs/sessions/SESSION_2026-04-30.md` Finding #4 listed LOVD as one of
  the 30+ all-zero columns out of 78 in the run9_ready feature matrix.
  This INCIDENT is the root-cause investigation that 4/30 deferred.
- `docs/CHANGELOG.md` 2026-04-17 (afternoon, take 2) Design Notes:
  *"Connector fallbacks with INFO logs are silent. ... Audit other
  connectors (EVE, AlphaMissense, CADD) for the same pattern as a Run 10
  prerequisite."* LOVD wasn't on that list, but the audit pattern is the
  same. This INCIDENT extends the audit recommendation to LOVD.
- `docs/incidents/INCIDENT_2026-04-17_esm2-hgvsp-parser.md` — ESM-2
  silent-zero, identical pattern (connector ran, returned defaults),
  resolved via fail-loud detection in
  `tests/unit/test_esm2_activation.py`. Comparable test exists for
  SpliceAI (`tests/unit/test_spliceai_parquet_default.py`,
  commit 9ba3127). LOVD has no equivalent; R10-B adds one.

## Timeline

- **2026-04-01 morning (UTC).** LOVD admin's logged automated-scraping
  incident on Monzia's IP for `?modified_since=...` API loop. IP
  blocked. `data/external/lovd/raw/*.txt` files at 5:38–5:39 AM Eastern:
  three genuine downloads (TP53, PTEN, RB1), seven 96–98 byte error
  pages (BRCA1, BRCA2, APC, MLH1, MSH2, MSH6, NF1).
- **2026-04-01 evening (UTC).** Admin unblock email. Manual browser
  saves of `?format=application/json` views at 6:56 PM Eastern for the
  seven previously-failed genes — permitted action per admin's second
  email; currently unconsumed by `scripts/build_lovd_index.py`.
- **2026-04-01 → 2026-04-30.** `lovd_all_variants.parquet` (18,006 rows,
  10 genes, `source_format="lovd_tab"`) produced by
  `scripts/build_lovd_index.py` at some point between the 4/1 unblock
  and the 4/30 regen. Exact build timestamp not on the docs trail
  (would require `Get-Item` on the parquet to confirm).
- **2026-04-30.** Splits regen via patched `run_phase2_eval.py`,
  13h 14min CPU. Per `SESSION_2026-04-30.md` Finding #4, LOVD already
  recognized as silent-zero alongside 30+ other columns. No root-cause
  investigation in 4/30 session.
- **2026-05-02.** Root-cause investigation. Three findings established
  with high confidence; cause narrowed to two candidates pending log
  grep.

## Evidence

### Connector wiring is correct

`src/data/real_data_prep.py:730–748` (verified by direct read):

```python
        # 14. Protein structure pipeline (Phase 6.2)
        from src.pipelines.protein_pipeline import ProteinStructurePipeline
        protein = ProteinStructurePipeline(cache_dir=ac.protein_cache_dir)
        df = protein.annotate_dataframe(df)
        ...

        # 15. LOVD: variant classification (ordinal 0-4)
        from src.data.lovd import LOVDConnector
        lovd = LOVDConnector(parquet_path=ac.lovd_path)
        df = lovd.annotate_dataframe(df)
        logger.info(
            "Score annotation 15/16 (LOVD): %d variants with lovd_variant_class > 0.",
            int((df.get("lovd_variant_class", pd.Series([0] * len(df), index=df.index)) > 0).sum()),
        )

        # 16. ESM-2 protein language model delta norm (Phase 3C)
        ...
```

Same `df = X.annotate_dataframe(df)` pattern as the 14 connectors before
it and the 2 after it. No flag, no conditional, no fallthrough.

### Data on disk is joinable

`data/external/lovd/lovd_all_variants.parquet`:

```
rows: 18006
cols: chrom, pos, ref, alt, label, gene_symbol, classification_raw,
      source_format, variant_id
sample: chrom=17, pos=7675234, ref=G, alt=T, gene_symbol=TP53,
        classification_raw="pathogenic", source_format="lovd_tab",
        variant_id="17:7675234:G:T"
chroms: 2, 3, 5, 8, 10, 13, 17
genes:  APC, BRCA1, BRCA2, MLH1, MSH2, MSH6, NF1, PTEN, RB1, TP53
```

`models/v1/clinvar_enriched.parquet`:

```
rows: 1,700,687
pos dtype: int64
chrom dtype: object (string, no "chr" prefix)
sample first 3 chroms: ["7", "7", "11"]
sample first 3 pos values: [4781213, 4787730, 126275389]
rows in LOVD genes: 54,710
```

Diagnostic merge (`diag_lovd_join.py`, replicates connector's exact key
construction):

```
LOVD key dtypes:    _chrom=object, _pos=object, _ref=object, _alt=object
ClinVar key dtypes: _chrom=object, _pos=object, _ref=object, _alt=object
Chroms in common: ['10', '13', '17', '2', '3', '5', '8']  (all 7 LOVD chroms present in ClinVar)
Inner-join matches: 5,553
```

5,553 / 54,710 = ~10.1% of ClinVar rows in LOVD's gene set match LOVD-
curated variants. Plausible coverage for a curated subset.

### Trained matrix has zero matches

`outputs/run9_ready/splits/X_train.parquet`:

```
has_col: True
cols matching lovd: ['lovd_variant_class']
lovd_variant_class
0.0    1197216
```

### Falsified causes

- **Float→str trailing `.0`.** `pos` is int64 in `clinvar_enriched.parquet`,
  `astype(str)` produces clean integer strings. Falsified by direct check.
- **Chromosome `chr` prefix mismatch.** Both sides produce strings
  without `chr` prefix; `lstrip("chr")` is a no-op on both. Falsified
  by direct check.
- **Connector not invoked.** Connector is unconditionally called at
  `real_data_prep.py:738` with return value assigned. Falsified by
  direct read of the call site.
- **`process_lovd.py` schema mismatch causing connector to load wrong
  parquet.** `process_lovd.py` is dead code; live merge is
  `scripts/build_lovd_index.py` → `lovd_all_variants.parquet`. The
  connector's `parquet_path=ac.lovd_path` resolves to the live parquet.
  Falsified by tightened grep on `LOVDConnector\(|from src.data.lovd|...`.

## Two remaining root-cause candidates

### Cause 1 — Downstream column overwrite

The connector reports 5,553 matches at runtime, but a downstream step
overwrites `lovd_variant_class` with zeros before the column reaches
`X_train.parquet`. Plausible suspects in `real_data_prep.py`:

- `_engineer_features` may include a step that reinitializes
  ordinal-class columns to zero based on a list of expected feature
  names that doesn't include `lovd_variant_class`.
- A fillna with `0` over the full feature DataFrame after feature
  engineering could overwrite the connector's ordinal output if NaN
  intermediate states exist.
- Column-selection step in feature engineering may rename or drop
  the connector's output and re-create a zeroed default.

`_scale` (StandardScaler) is unlikely — standardizing a sparse 0–4
ordinal would produce mostly-zero standardized values, not literally
zero. Unlikely but should be confirmed in R10-B if Cause 1 is
indicated.

### Cause 2 — Upstream coordinate transformation

The connector reports 0 matches at runtime because one of the 14 prior
`annotate_dataframe` calls (steps 1–14) has transformed `chrom`, `pos`,
`ref`, or `alt` in `df` before the LOVD step at line 738 sees it. Step
14 is `protein.annotate_dataframe(df)` from `ProteinStructurePipeline` —
unknown what it does to coordinate columns, would need direct read of
`src/pipelines/protein_pipeline.py:annotate_dataframe`.

## Distinguishing diagnostic

`real_data_prep.py:740–748` emits:

```
"Score annotation 15/16 (LOVD): %d variants with lovd_variant_class > 0."
```

Per `SESSION_2026-04-30.md` Files Written list, `regen.log` was written
to `outputs/run9_ready/regen.log` and contains the full 13h trace.
Grep for `"Score annotation 15/16 (LOVD)"`:

- non-zero (~5,500): Cause 1, fix is in `_engineer_features`/`_scale`
- zero: Cause 2, fix is in one of steps 1–14

## Remediation plan

### R10-A (verification, ~2 minutes)

```powershell
Select-String -Path C:\Projects\genomic-variant-classifier\outputs\run9_ready\regen.log `
    -Pattern "Score annotation 15/16 \(LOVD\)" |
    Select-Object LineNumber, Line
```

Outcome determines whether R10-B targets Cause 1 or Cause 2.

### R10-B (patch + test)

Patch the identified cause. Add unit test:

```python
def test_lovd_annotation_reaches_training_matrix():
    """
    Post-condition: after _load_and_label + the full annotate chain
    (or at minimum after LOVDConnector.annotate_dataframe), at least
    one variant in a 5-row ClinVar fixture with one matching LOVD row
    has lovd_variant_class > 0.

    Regression for INCIDENT_2026-05-02_lovd-silent-zero.
    """
```

The test must exercise the pipeline path that produced the
silent-zero, not just the connector in isolation (which we already
know works — the diagnostic proved it). Pattern modeled on
`tests/unit/test_spliceai_parquet_default.py` (commit 9ba3127) and
`tests/unit/test_esm2_activation.py` (2026-04-17 session).

### R10-C (re-regen on Vast.ai)

Per standing rule #19, no local retraining. Re-regen splits on
Vast.ai with LOVD live. Post-condition assertion:

```python
import pandas as pd
X = pd.read_parquet("outputs/run10_xxx/splits/X_train.parquet")
counts = X["lovd_variant_class"].value_counts()
nonzero = (X["lovd_variant_class"] > 0).sum()
assert nonzero >= 4500, f"Expected ~5500 of 5553 inner-join matches in train; got {nonzero}"
```

Target: roughly 4,500–5,500 of the 5,553 inner-join matches end up in
the train set, depending on how the gene-aware split distributes the 10
LOVD genes across train/val/test. If 0 again, fix is incomplete.

### R10-D (originally-requested gene-scope expansion)

After A–C are green: manual browser-download `.txt` files for new genes
per LOVD admin's instructions (Path 1 — see SESSION_2026-05-02 for
discipline rules). Optional: expand `build_lovd_index.py` to consume the
6:56 PM `.json` files for the 7 genes whose 4/1 morning `.txt` saves
were error pages. Re-regen splits a second time.

## Lessons (preliminary; full LESSONS LEARNED in R10-B closure)

- **Connector logs at INFO level get lost in 13h training output.**
  `real_data_prep.py:740` does emit a count of `lovd_variant_class > 0`
  variants. That line presumably said `0` in the 4/30 regen log and
  was not flagged at the time. R10-A is essentially "go read the log
  message we should have read on 4/30."
- **The 4/17 silent-zero audit recommendation was correct and
  insufficient.** It listed EVE, AlphaMissense, CADD as audit
  candidates. LOVD was already on the silent-zero list as of 4/30 but
  not investigated. Lesson: when an audit recommendation lists *some*
  connectors and a later session lists *more* connectors as silent-
  zero, the audit scope should expand, not stay anchored to the
  original list.
- **Tests at the connector boundary are not sufficient.** The
  connector itself works correctly in isolation (diagnostic proves
  this). What's needed is a *post-condition* test on the full ETL
  pipeline that asserts the connector's output reaches the training
  matrix unaltered. R10-B will add this for LOVD; the same pattern
  should be retroactively applied to the other 30+ silent-zero
  columns flagged in 4/30 Finding #4.

## Sign-off

This INCIDENT is OPEN. Will be moved to RESOLVED when R10-A through
R10-C are complete and the post-condition assertion in R10-C passes.
R10-D (originally-requested gene scope expansion) does not gate
RESOLVED status — it can ship as a separate Run 11 if scope warrants.

---

## 2026-05-13 Update — Actual root cause identified; both prior candidates falsified

### Status update

OPEN → **ROOT CAUSE IDENTIFIED**; RESOLUTION SHIPPING IN PHASE 1 PATCH
BUNDLE (`run10_phase1_v2.zip`, patch B1). Pending apply + Run 10 dry-run
confirmation, status moves to RESOLVED.

### Both prior candidates falsified

The original INCIDENT (2026-05-02) narrowed the cause to two candidates
pending log grep (R10-A):

- Cause 1 — Downstream column overwrite in `_engineer_features` or
  similar post-connector step
- Cause 2 — Upstream coordinate transformation by one of the 14 prior
  `annotate_dataframe` calls

Both are wrong. The actual cause is a **third unconsidered candidate**:
the LOVD connector is never given a path to its data file in the
Run 9 launch path.

### Evidence

Two source files were compared on 2026-05-13:

**`scripts/train.py` lines 167–172** (works correctly, used in pre-Run-9
local CPU train.py runs per CHANGELOG 2026-04-16):

```python
annotation_config = AnnotationConfig(
    alphamissense_path=Path(args.alphamissense) if args.alphamissense else None,
    lovd_path=Path(args.lovd_path) if args.lovd_path else None,
    finngen_path=Path(args.finngen_path) if args.finngen_path else None,
    dbnsfp_path=Path(args.dbnsfp_path) if args.dbnsfp_path else None,
)
```

**`scripts/run_phase2_eval.py` lines 119–127** (silent-zero, used for
Run 9 splits regen on 2026-04-30 AND Run 9 training on 2026-05-12):

```python
ann = AnnotationConfig(
    spliceai_path=Path(args.spliceai) if args.spliceai else None,
    alphamissense_path=Path(args.alphamissense) if args.alphamissense else None,
    gtex_genes=args.gtex_genes or [],
    kg_path=Path(args.kg) if args.kg else None,
    gnomad_constraint_path=(
        Path(args.gnomad_constraint) if args.gnomad_constraint else None
    ),
)
```

`AnnotationConfig` HAS the `lovd_path`, `dbnsfp_path`, and `finngen_path`
fields (proven by `train.py` constructing with all three). The Run 9
launch path simply never passes them. The LOVD connector then receives
`parquet_path=None` from the default, takes its "no parquet loaded"
silent branch, and returns 0 for all variants.

### Verification from Run 9 regen.log

The 2026-04-30 splits regen (which Run 9 inherited) is the relevant log
to grep. As of 2026-05-13 the local audit (`scripts/run9_outputs_audit.ps1`)
scanned `outputs/run9_ready/regen.log` for AUROC patterns but did not
capture LOVD-specific lines. Direct verification command:

```powershell
Select-String -Path outputs\run9_ready\regen.log -Pattern "LOVD"
```

Expected output: the silent-branch INFO message from the LOVDConnector
(approximately `"no parquet loaded — all variants will receive
lovd_variant_class=0"` or similar wording from the connector's
default-path branch). If this message is found, it confirms the
launch-script wiring gap as the cause and falsifies Cause 2 (which
hypothesizes the connector successfully loaded but got zero matches
post-merge). If a different LOVD line appears (e.g., "Score annotation
15/16 (LOVD): 5553 variants ..."), Cause 1 (downstream column
overwrite) is back in play and B1 patch is insufficient.

### Same bug class affects 2 other connectors

The same wiring gap applies to DbNSFP and FinnGen. Per `scripts/train.py`
(verified entry-point reference):

- **DbNSFP** SIFT/PolyPhen-2/REVEL/MutationTaster/FATHMM/CADD —
  `train.py` log showed 204,384 real SIFT scores when wired
- **FinnGen** finngen_af_fin/finngen_af_nfsee/finngen_enrichment —
  `train.py` log showed 123,990 matches (7.3%) when wired
- **LOVD** lovd_variant_class — ~5,553 expected matches per the
  diagnostic merge documented in the 2026-05-02 evidence section

Run 9 (which used `scripts/run_phase2_eval.py` for splits regen, not
`train.py`) was unknowingly starved of all three. Verification commands:

```powershell
Select-String -Path outputs\run9_ready\regen.log -Pattern "DbNSFP|dbNSFP"
Select-String -Path outputs\run9_ready\regen.log -Pattern "FinnGen|finngen"
```

Expected: similar "no data file" / "default scores" messages for both.

### Why prior analysis missed this

The 2026-05-02 INCIDENT focused on the **connector code path** (call
site at `real_data_prep.py:738`, diagnostic merge against
`clinvar_enriched.parquet`) and the **on-disk data**
(`lovd_all_variants.parquet` schema and row count). Both checked out.
The investigation did not extend up to the **launch script's**
AnnotationConfig construction because the silent-zero pattern matched
"connector internal" failure modes (SpliceAI in run 8, ESM-2 in runs
6-8). The launch-script wiring level was a blind spot.

A clean diagnostic in retrospect would have been:

```bash
grep -n "AnnotationConfig" scripts/*.py
# Check whether train.py and run_phase2_eval.py construct identically
```

This is now a standing rule for future silent-zero investigations.

### Resolution (Phase 1 patch B1)

`scripts/run_phase2_eval.py` patches:

1. Add `--lovd-path`, `--dbnsfp-path`, `--finngen-path` CLI args
2. Pass them through to `AnnotationConfig` construction (mirroring
   `train.py:167-172`)

Both changes shipped in `run10_phase1_v2.zip` as a single str_replace
patch each.

### R10-* sequence — revised

- ~~**R10-A** (log grep for "Score annotation 15/16 (LOVD)")~~ —
  **SUPERSEDED.** The launch-script wiring gap (run_phase2_eval.py
  never passing lovd_path) is the actual cause. Verification of the
  expected silent-branch INFO message is still useful but no longer
  blocking — see Verification command below.
- **R10-B** (patch + post-condition unit test) — **SHIPPED** in
  Phase 1 bundle. Patch is B1 in
  `apply_run10_phase1_patches.py`. Post-condition test is
  `tests/unit/test_lovd_annotation_reaches_training_matrix.py`.
- **R10-C** (re-regen splits on Vast.ai with LOVD live) — pending Run 10
  launch. Post-condition `nonzero >= 4500` assertion unchanged.
- **R10-D** (gene-scope expansion beyond canonical 10) — unchanged;
  manual browser downloads only per LOVD admin instructions, deferred
  to Run 11+.

### Lessons added

- **Silent-zero investigation must extend up to the launch-script
  configuration boundary.** Connector code + on-disk data + diagnostic
  merge together cover three of the four layers. The fourth layer
  (config construction in the launch script) was missed in the
  2026-05-02 investigation. Standing rule for future audits:
  `grep -rn "AnnotationConfig\\b" scripts/` and compare argument
  counts/names across all entry points.
- **Multiple entry points to the same pipeline must be kept in sync.**
  `train.py` and `run_phase2_eval.py` both construct
  `AnnotationConfig`. They have diverged silently over months. Going
  forward, either (a) consolidate to a single config-construction
  helper that both entry points call, or (b) add a CI check that
  greps both files for `AnnotationConfig(` and reports field-list
  diffs.

### Sign-off (revised)

INCIDENT moves to RESOLVED when:
- Phase 1 patch bundle (`run10_phase1_v2.zip`) is applied
- `pytest tests/unit/test_lovd_annotation_reaches_training_matrix.py` passes
  (or skips with a documented import-path reason)
- Run 10 splits regen on Vast.ai produces an `X_train.parquet` with
  `(X_train["lovd_variant_class"] > 0).sum() >= 4500` per the R10-C
  post-condition.
