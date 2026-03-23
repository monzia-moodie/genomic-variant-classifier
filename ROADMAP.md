# Genomic Variant Classifier — Project Roadmap

**Author:** Monzia Moodie  
**Repository:** `monzia-moodie/genomic-variant-classifier`  
**Last updated:** March 2026 — Phase 2 complete

---

## Vision

A production-grade, multi-modal genomic variant pathogenicity classifier that:

1. Achieves clinically actionable AUROC >= 0.90 on held-out ClinVar data
2. Provides calibrated uncertainty estimates for Variants of Uncertain Significance (VUS)
3. Integrates population-level WGS controls alongside disease cohort data
4. Serves predictions via a REST API with Docker deployment
5. Benchmarks multiple ML algorithm families to rigorously compare their
   effectiveness on large-scale genomic data

---

## Current State (March 2026)

| Item | Status |
|------|--------|
| Phase 1 bugs fixed | 235 tests passing, 3 skipped |
| 8-model ensemble + stacking meta-learner | done |
| ClinVar connector + parquet pipeline | done — 4.4M variants |
| gnomAD, UniProt, OMIM connectors | done |
| dbNSFP, SpliceAI, AlphaMissense, GTEx connectors | done |
| TABULAR_FEATURES (27 features) | done |
| Baseline AUROC (ClinVar only, no external scores) | 0.72 |
| **Final AUROC (holdout, tier-2, gnomAD + AlphaMissense)** | **0.9847** |
| **Phase 2 status** | **COMPLETE** |

---

## Phase 2 — Core Classifier Completion

**Goal:** AUROC >= 0.90 on real ClinVar data with external functional scores.

### 2A — External Score Downloads

| File | Size | Source | Status |
|------|------|--------|--------|
| AlphaMissense_hg38.tsv.gz | 0.61 GB | GCS (free) | **Done** |
| spliceai_scores.masked.snv.hg38.vcf.gz | ~2.6 GB | Zenodo (free) | Pending |
| gnomad.exomes.v4.1.sites.chr*.vcf.bgz | ~40 GB (chr1-22+X) | GCS (free) | **Done** |
| dbNSFP4.7a.zip | ~30 GB | Google Drive (registration) | Pending |

### 2B — Pipeline Completion Checklist

- [x] Run full eval with AlphaMissense wired — 206k variants annotated
- [x] Build gnomAD parquet from VCF; run eval with --gnomad — 2.9M loci, 60.2% join rate
- [ ] Register for dbNSFP; run eval with SIFT/REVEL/CADD/GERP populated
- [x] Confirm allele_freq appears in top 3 of feature_importance.csv — af_raw is #2
- [x] AUROC >= 0.90 on full holdout set — **0.9847** on 154k gene-stratified val variants
- [x] Validation split added to DataPrepPipeline (gene-aware, 70/10/20 train/val/test)
- [x] ClinVar alleles patched — 99.5% of variants now have real REF/ALT
- [ ] REST API (src/api/) — FastAPI with /predict, /batch, /health
- [ ] Docker deployment (infrastructure/docker/)
- [ ] codon_position from HGVSc + VEP (final PHASE_2_FEATURES item)

### 2C — AUROC Trajectory (actual)

| Configuration | AUROC | Key driver |
|---------------|-------|------------|
| ClinVar only, no external scores | 0.72 | n_pathogenic_in_gene |
| + real REF/ALT (patch_clinvar_alleles) | 0.98 | consequence_severity, len_diff unlocked |
| + AlphaMissense | 0.9776 | alphamissense_score (#2 feature) |
| + gnomAD AF | 0.9821 | af_raw (#2), more robust to novel genes |
| **Tier-2 labels, full data, gnomAD + AlphaMissense** | **0.9847** | Cleaner labels + all signals |

---

## Phase 3 — Data Expansion

**Goal:** Train and validate on diverse real-world genomic data beyond ClinVar.

### 3A — Population Controls (Healthy, Open Access)

| Source | Data | Access | Integration |
|--------|------|--------|-------------|
| 1000 Genomes Project (IGSR) | WGS 2,504 individuals 26 populations 30x | Open, IGSR portal | allele_freq calibration |
| IGSR expanded panels | ~5,000 genomes, continuously updated | Open | Ancestry-stratified AF |
| Simons Genome Diversity Project | 279 WGS 130 populations 43x+ | Open | Rare variant calibration |

Integration: Extract per-locus AF from VCF using the existing _join_gnomad
locus-key pattern. Add population_1kg_af column. No new connector needed.

### 3B — Disease Cohorts (Controlled Access)

Apply for these in parallel — each takes 2-8 weeks for approval.

| Source | Data | Application | Priority |
|--------|------|-------------|----------|
| dbGaP / NCBI | Gateway to TOPMed (300K WGS), CMG rare disease | eRA Commons + institutional | High |
| EGA | European cancer + rare disease WGS | Data Access Agreement | High |
| CMG (Centers for Mendelian Genomics) | High-quality rare disease trios | Via dbGaP / AnVIL | High — best labels |
| CCDG | Cardiovascular, neuropsychiatric WGS | Via dbGaP / AnVIL | Medium |
| Genomics England 100K Genomes | 100K WGS rare disease + cancer | Research access agreement | High |
| UK Biobank | 470K WES + 200K WGS | Formal application | Medium-high |
| All of Us | 250K+ WGS diverse ancestry | Researcher Workbench (free) | Medium |

### 3C — Cancer Cohorts

| Source | Data | Access |
|--------|------|--------|
| TCGA (GDC) | Tumor/normal WGS + MAF files | Open tier free; controlled via dbGaP |
| PCAWG | 2,658 WGS across 38 tumor types | ICGC portal |

### 3D — Compute Strategy for WGS Scale

Raw WGS is too large to download locally. Use cloud analysis platforms:
- NHGRI AnVIL (Terra / Google Cloud) — hosts 1KGP, GTEx, CMG, CCDG
- GDC API — already stubbed in config.yaml as gdc_base
- All of Us Researcher Workbench — cloud-only, no local download needed

The existing Spark ETL pipeline (src/data/spark_etl.py) accepts a master URL
that can point to a cloud Dataproc cluster.

---

## Phase 4 — Algorithm Expansion and Benchmarking

**Goal:** Rigorous comparison of ML families on genomic variant classification.

### 4A — ESM-2 Sequence Embeddings (Highest ROI, lowest effort)

Frozen inference from Meta's protein language model (ESM-2, pretrained on
UniRef90). Embedding delta (wildtype vs. mutant) is one of the strongest
missense pathogenicity signals available, independent of conservation scores.

Complexity: O(L * s^2 * d_model) inference per sequence. Tractable in frozen
mode — no fine-tuning needed.

Expected AUROC lift: +0.03-0.06 on missense variants.

Integration: src/data/esm2.py connector; add esm2_delta_norm to TABULAR_FEATURES.

### 4B — KAN (Kolmogorov-Arnold Network)

Replaces fixed activation functions with learnable spline functions on edges.
More interpretable than MLPs; learned univariate functions can be visualized
directly, extending the SHAP explainability story.

Complexity: O(n * L * H^2 * G) where G is spline grid size (~5-20).
Constant-factor overhead over equivalent MLP; same asymptotic class.

Integration: Replace TabularNNClassifier in VariantEnsemble with KANClassifier
from pykan. Compare OOF AUROC and feature attribution side-by-side with SHAP.

### 4C — Bayesian Uncertainty Quantification

MC Dropout on neural branches; Deep Ensembles on the stacking meta-learner.
Produces calibrated probability estimates with epistemic and aleatoric
uncertainty decomposition.

Clinically essential: a classifier reporting "pathogenic: 0.87" must be right
~87% of the time. VUS handling requires honest uncertainty.

Metrics to add: Expected Calibration Error (ECE), reliability diagrams,
uncertainty decomposition on VUS subset.

Complexity: O(T * L * H^2) inference for T dropout passes (T ~ 50).
No additional parameters over the base model.

### 4D — GNN over Protein-Protein Interaction Network

Graph Neural Network with STRING DB interaction graph. Variant-level node
features; PPI edges weighted by combined interaction score.

src/models/gnn.py already implements StringDBGraph with GAT convolutions.
Needs integration with the variant feature matrix as node attributes.

Complexity: O(G * (V + E) * d) — linear in graph size, independent of n.
Runs on a single GPU alongside the Spark pipeline.

Integration: STRING DB graph + TABULAR_FEATURES as node attributes
-> GAT (2-3 layers) -> gene-level pathogenicity embedding
-> late fusion with ensemble stacker.

### 4E — Contrastive Learning for Histopathology (TCGA branch)

DINO (self-distillation with no labels) applied to TCGA whole-slide image
patches. Produces tile-level embeddings without per-patch labels.
Current state-of-the-art in computational pathology.

### 4F — Algorithm Comparison Framework

src/evaluation/
    benchmark.py        run all models on same train/test split
    calibration.py      ECE, reliability diagrams
    complexity.py       wall-clock time and memory profiling per algorithm

Algorithms to benchmark on identical splits:

| Family | Algorithms |
|--------|-----------|
| Tabular ensemble | XGBoost, LightGBM, Random Forest, GBM |
| Linear | Logistic Regression, MCP-regularized LR |
| Neural tabular | MLP (TabularNN), KAN |
| Sequence | 1D-CNN, ESM-2 embeddings |
| Graph | GAT (STRING DB) |
| Probabilistic | MC Dropout ensemble, Deep Ensemble |

Evaluation dimensions:
- Predictive: AUROC, AUPRC, Brier score, ECE
- Computational: training time, inference latency, memory footprint
- Interpretability: SHAP consistency, feature attribution stability
- Robustness: performance on VUS, rare vs. common disease, ancestry subsets

---

## Phase 5 — Clinical Validation and Deployment

- [ ] Prospective validation on gene panels (BRCA1/2, TP53, PTEN, ATM)
- [ ] Comparison against ClinVar star-rating on expert-reviewed variants
- [ ] REST API: FastAPI /predict /batch /health /model-card
- [ ] Docker: multi-stage build, API image + training image
- [ ] Model card: training data, known limitations, ancestry coverage
- [ ] Manuscript draft: multi-modal genomic variant classifier with
      algorithm benchmarking study

---

## Complexity Reference

Key Big-O for algorithm selection at genomic scale.
n=variants, d=features, s=sequence length, V/E=graph nodes/edges.

| Algorithm | Training | Genomic scale |
|-----------|----------|--------------|
| GBTs (XGBoost/LightGBM) | O(I*n*d*log n) | Excellent |
| MLP / KAN | O(n*L*H^2[*G]) | Good |
| ESM-2 (frozen inference) | N/A | Excellent |
| GNN (sparse PPI) | O(G*(V+E)*d) | Excellent — independent of n |
| Transformer (fine-tune) | O(n*L*s^2*d) | Sequence-length limited |
| SVM (RBF) | O(n^2) | INFEASIBLE above ~100K samples |
| Exact GP | O(n^3) | Infeasible; use sparse approximation |
| UMAP | O(n*log n) | Excellent |

SVM is excluded from all production runs (n > 100K).

---

## Feature Roadmap

TABULAR_FEATURES (27, live):
  allele_freq, ref_len, alt_len, is_snv, is_indel
  cadd_phred, sift_score, polyphen2_score, revel_score
  phylop_score, splice_ai_score, gerp_score, alphamissense_score
  in_coding_region, in_splice_site, is_missense, is_nonsense
  gene_constraint_oe, num_pathogenic_in_gene
  in_active_site, in_domain
  gtex_max_tpm, gtex_n_tissues_expressed, gtex_tissue_specificity
  gtex_is_eqtl, gtex_min_eqtl_pval, gtex_max_abs_effect

PHASE_2_FEATURES (1, pending VEP):
  codon_position — requires HGVSc + VEP annotation pipeline

PHASE_3_FEATURES (planned):
  population_1kg_af     — 1000 Genomes allele frequency
  esm2_delta_norm       — ESM-2 embedding distance (wt vs. mut)
  ppi_network_score     — GNN-derived neighborhood pathogenicity
  uncertainty_epistemic — MC Dropout epistemic uncertainty
  uncertainty_aleatoric — MC Dropout aleatoric uncertainty

---

*This roadmap is a living document. Update after each phase gate.*
