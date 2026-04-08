---

## Deferred Item: OpenMed Psychiatric GWAS Dataset Integration

**Logged:** 2026-04-08
**Status:** DEFERRED — do not implement before Phase 3 Polars milestone
**Target milestone:** Phase 3 (Polars evaluation) — see rationale below

---

### Background

OpenMed released a Hugging Face dataset of **1.14 billion rows** of
standardised psychiatric genetics summary statistics, consolidating every
GWAS meta-analysis ever published by the Psychiatric Genomics Consortium
(PGC). Covers 12 conditions across 52 landmark studies:

- ADHD, Depression, Schizophrenia, Bipolar disorder, PTSD, OCD, Autism,
  Anxiety, Tourette syndrome, Eating disorders, and additional sub-phenotypes.

Each row is a single variant-phenotype association test containing:
`rsID`, `CHR/POS`, `A1/A2`, `BETA`/`OR`, `SE`, `P`, `INFO`, `FRQ/MAF`,
`N`, `Nca`, `Nco`.

---

### Clinical Relevance Assessment

**Biological mismatch (key constraint):** The PGC GWAS studies are powered
almost entirely by common variants (MAF > 1–5%) with small effect sizes
(OR 1.05–1.3). The classifier's primary signal comes from rare, large-effect
Mendelian variants (MAF < 0.1%, OR >> 5). Feature signal for the primary
pathogenicity task is expected to be **weak to marginal** overall.

**Legitimate value identified:**

1. **Pleiotropy annotation for psychiatric gene panels.** Genes such as
   *CACNA1C*, *SCN2A*, *NRXN1*, *SHANK3* carry both rare pathogenic alleles
   and common GWAS signals. A locus-level enrichment feature could improve VUS
   interpretation specifically for neuro/psychiatric panel variants.

2. **Functional region flagging.** High-density GWAS signal (many variants
   p < 1×10⁻⁵) is an indirect proxy for regulatory/functional importance.

3. **Cross-disorder pleiotropy score.** The multi-condition architecture of
   this dataset uniquely enables a meta-feature: number of independent
   psychiatric conditions with genome-wide significant signal near a locus.

4. **Gap addressed.** The current 78-feature set has no psychiatric /
   neurodevelopmental phenotypic annotation. For *de novo* neurodevelopmental
   disease genes, this fills a real gap.

**Label leakage status:** No direct leakage path — GWAS summary stats are
population-level aggregates, not clinical assertions. Audit recommended to
ensure no ClinVar P/LP labels for psychiatric variants were derived from GWAS
evidence at ingestion time.

---

### Planned Feature Schema (5 new features)

```python
# connector: src/data/database_connectors.py — PsychGWASConnector
# DEFAULT_SCORE = 0.0 (no signal — correct default for rare pathogenic variants)

gwas_psych_min_pval           # min p-value across all 52 studies
gwas_psych_hit_count          # number of studies with p < 5×10⁻⁸
gwas_psych_disorder_breadth   # number of distinct conditions significant
gwas_psych_max_neg_log10p     # strongest single-study signal
gwas_psych_is_lead_snp        # bool: is this a GWAS lead SNP
```

Follows existing `annotate_dataframe()` connector pattern with stub mode.

---

### Preprocessing Requirements (Critical)

Raw dataset is ~80–150 GB on disk. **Must filter before any downstream use:**

```python
# Step 1: Filter to genome-wide significant hits only
# Reduces 1.14B rows → ~2–5M rows across 52 studies

df_filtered = df.filter(pl.col("P") < 5e-8)   # Polars — see rationale below

# Step 2: Aggregate per rsID into a summary lookup table
# Output: ~200–500 MB Parquet — compatible with existing connector pattern

lookup = (
    df_filtered
    .group_by("rsID")
    .agg([
        pl.col("P").min().alias("gwas_psych_min_pval"),
        pl.col("STUDY_ID").n_unique().alias("gwas_psych_hit_count"),
        pl.col("CONDITION").n_unique().alias("gwas_psych_disorder_breadth"),
        (-pl.col("P").log(base=10)).max().alias("gwas_psych_max_neg_log10p"),
        pl.col("IS_LEAD_SNP").any().alias("gwas_psych_is_lead_snp"),
    ])
)
```

---

### Why This Is the Phase 3 Polars Benchmark Candidate

The preprocessing pipeline above (filter 1.14B rows → aggregate per rsID)
is an ideal real-world benchmark for the **Phase 3 Polars evaluation**:

- Demonstrates Polars throughput advantage vs. pandas at scale
- Solves the storage/preprocessing problem as a by-product of the evaluation
- No model changes required — only a new connector + 5 feature columns
- Feature count assertions will need updating: search
  `Get-ChildItem -Path "src","tests" -Recurse -Filter "*.py" | Select-String -Pattern "== N"`
  and increment by 5 before running the test suite

**Proposed sequence at Phase 3 milestone:**
1. Run Polars bottleneck evaluation on existing pandas operations (primary goal)
2. Use this GWAS preprocessing as the scale benchmark case
3. If Polars is adopted, implement `PsychGWASConnector` using Polars throughout
4. Add 5 features; update all hardcoded feature count assertions
5. Retrain; evaluate AUROC delta — expect marginal improvement for global
   classifier, potentially larger for psychiatric gene panel sub-cohort

---

### Data Source

```
Dataset:    OpenMed Psychiatric Genomics Consortium (standardised)
Location:   Hugging Face — https://huggingface.co/datasets/openmed/pgc-gwas
Conditions: ADHD, Depression, Schizophrenia, Bipolar, PTSD, OCD, Autism,
            Anxiety, Tourette, Eating disorders (12 total)
Studies:    52 PGC meta-analyses
Rows:       ~1.14 billion
Format:     Parquet (recommended download format)
Local path: G:\My Drive\genomic-variant-data\external\pgc_gwas_summary\
```

---

### Do Not Implement Until

- [ ] Run 6 (GCP) training is complete with KAN removed and
      `gnomad_constraint_path` wired
- [ ] Phase 3 Polars bottleneck evaluation is scheduled
- [ ] Feature count assertions audited and documented

---

---

## Run 6 — Completed 2026-04-08

**Holdout AUROC: 0.9862** | 78 features | 345K variants | 4,441 genes

### Activated in Run 6
- gnomAD v4.1 gene constraint: `pLI`, `LOEUF`, `syn_z`, `mis_z` (wired via `--gnomad-constraint`)
- KAN unconditionally removed (C++ OOM at scale — not catchable)
- AlphaMissense active (5.2GB TSV)
- Ensemble resume logic added to `run_phase2_eval.py`

### Deferred to Run 7
- GNN: torch-geometric missing on VM → zero GNN contribution
- SpliceAI: `spliceai_index.parquet` corrupt (29GB raw dump) → omitted
- ESM-2: verify non-stub on VM

### Infrastructure issues resolved
- gcloud working (SDK 564.0, project `genomic-variant-prod`)
- Correct GCS bucket: `gs://genomic-classifier-data/`
- VM startup script attachment bug identified (never ran)

### Run 7 priorities
1. Install torch-geometric on VM → activate GNN
2. Fix startup script attachment in `gcloud compute instances create`
3. Regenerate `spliceai_index.parquet` from raw VCF
4. Evaluate Polars to replace bottleneck pandas ops (Phase 3 milestone)

## Phase 4 Agenda Items

### LiteratureScoutAgent Implementation
Deferred from Phase 3. When implemented, configure the following watch targets:

**pykan memory fix (KAN re-enablement trigger)**
- Watch: https://github.com/KindXiaoming/pykan releases
- Trigger condition: release notes mention memory optimization, large-dataset
  support, or OOM fixes
- Action on trigger: test locally with 150K stratified subsample before
  enabling on GCP; remove unconditional `pop("kan", None)` from
  `run_phase2_eval.py` if test passes
- Fallback path (no pykan fix): train KAN on stratified 50K subset only,
  contributing OOF predictions to meta-learner on that subset

**Database version monitoring (existing scope, document here for completeness)**
- ClinVar: monthly release cadence, watch for schema changes
- gnomAD: v4.2+ release, watch for constraint metrics column changes
- AlphaMissense: watch for hg38 TSV format updates
- STRING DB: v13 release monitoring

### DataFreshnessAgent Extension
Add pykan version check hook alongside existing ClinVar/gnomAD freshness checks:
- Check `pip index versions pykan` against installed version on each run
- Log new versions to agent state; surface in DataFreshnessAgent report
- This covers the gap until LiteratureScoutAgent is implemented

### KAN Re-enablement Checklist (when pykan fix detected)
- [ ] `pip install pykan --upgrade` in local venv
- [ ] Run `python -c "import kan; m = kan.KAN([78,64,1])"` smoke test
- [ ] Train on 150K stratified subsample locally, confirm no OOM
- [ ] Remove `ensemble.base_estimators.pop("kan", None)` from `run_phase2_eval.py`
- [ ] Add `--skip-kan` flag as optional override (do not hardcode removal again)
- [ ] Re-run full training on GCP, compare AUROC delta vs Run 7 baseline
