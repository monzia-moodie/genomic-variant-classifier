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