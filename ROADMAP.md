# Genomic Variant Classifier -- Project Roadmap

**Author:** Monzia Moodie
**Repository:** `monzia-moodie/genomic-variant-classifier`
**Last updated:** March 2026 -- Phase 7/8 complete, Phase 4 in progress

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

| Item                                                                   | Status                    |
| ---------------------------------------------------------------------- | ------------------------- |
| 64-feature tabular ensemble (LightGBM, XGBoost, RF, GBM, LR)          | **Done**                  |
| Holdout AUROC (gene-stratified, 154K variants)                         | **0.9847**                |
| RNA splice pipeline (MaxEntScan; 4 features)                           | **Done**                  |
| Protein structure pipeline (AlphaFold/UniProt; 4 features)             | **Done**                  |
| FastAPI REST service (/predict, /batch, /health, /gene, /rsid, /info)  | **Done**                  |
| X-API-Key auth + slowapi rate limiting                                 | **Done**                  |
| Structured JSON logging + Prometheus /metrics                          | **Done**                  |
| Multi-stage Dockerfile + docker-compose (api / trainer / monitoring)   | **Done**                  |
| GitHub Actions CI (lockfile check, pytest, docker build)               | **Done**                  |
| Docker image pushed to GHCR                                            | **Done -- v2.0.0**        |
| Conformal prediction intervals (scripts/conformal_prediction.py)       | **Done**                  |
| External validation script (scripts/validate_external.py)             | **Done**                  |
| Calibration analysis (scripts/calibration_analysis.py)                | **Done**                  |
| METHODS.md (publication-ready methods section)                         | **Done**                  |
| dbSNP index parquet (2.87M ClinVar-matched rs-IDs)                     | **Done**                  |
| ESM-2 connector (src/genomic_variant_classifier/data/esm2.py)                                     | **Done -- ready for retrain** |
| MC Dropout / Deep Ensemble uncertainty (src/genomic_variant_classifier/models/mc_dropout.py)      | **Done**                  |
| KAN classifier (src/genomic_variant_classifier/models/kan.py)                                     | **Done**                  |
| Algorithm benchmark framework (src/genomic_variant_classifier/evaluation/benchmark.py)            | **Done**                  |
| Model retrain incorporating Phase 4 features                           | Pending data + compute    |

---

## Phase 4 -- Algorithm Expansion and Benchmarking

**Goal:** Rigorous comparison of ML families; add ESM-2 and uncertainty features.

### 4A -- ESM-2 Sequence Embeddings

- [x] `src/genomic_variant_classifier/data/esm2.py` connector -- HuggingFace transformers backend, SQLite cache
- [x] `esm2_delta_norm` added to `PHASE_4_FEATURES` (ready for next retrain)
- [ ] Install `transformers torch` in training environment and run annotation
- [ ] Retrain ensemble with 65-feature set; measure AUROC lift (+0.03-0.06 expected)

Expected AUROC lift: +0.03-0.06 on missense variants.
Install: `pip install transformers torch`

### 4B -- KAN (Kolmogorov-Arnold Network)

- [x] `src/genomic_variant_classifier/models/kan.py` -- pykan / efficient-kan backends; MLP fallback
- [x] sklearn-compatible interface; `plot_edge_functions()` for interpretability
- [ ] Run in benchmark framework; compare OOF AUROC against MLP

Install: `pip install pykan`

### 4C -- Bayesian Uncertainty Quantification

- [x] `src/genomic_variant_classifier/models/mc_dropout.py` -- MCDropoutWrapper + DeepEnsembleWrapper
- [x] Uncertainty decomposition: epistemic (variance) + aleatoric (entropy)
- [x] `annotate_uncertainty()` helper for DataFrame annotation
- [ ] Run DeepEnsembleWrapper(LightGBM, n_members=5) on holdout; measure ECE improvement
- [ ] Annotate VUS subset with uncertainty flags; export for clinical review

### 4D -- GNN over Protein-Protein Interaction Network

- [x] `src/genomic_variant_classifier/models/gnn.py` -- GAT convolutions over STRING DB graph
- [ ] Wire STRING DB edge weights into GNN training (currently uses uniform weights)
- [ ] Late fusion: concat GNN gene embedding with TABULAR_FEATURES before stacking

### 4E -- Algorithm Comparison Framework

- [x] `src/genomic_variant_classifier/evaluation/benchmark.py` -- cross-validated benchmark across all families
- [x] Metrics: AUROC, AUPRC, Brier, ECE, train time, inference latency, memory
- [ ] Run full benchmark on ClinVar holdout
- [ ] Produce comparison table for METHODS.md / manuscript

Run:

```bash
python -m genomic_variant_classifier.evaluation.benchmark \
    --parquet data/processed/clinvar_grch38.parquet \
    --output  outputs/benchmark \
    --n-folds 5
```

---

## Phase 3 -- Data Expansion

### 3A -- Population Controls

| Source                      | Data           | Status                                          |
| --------------------------- | -------------- | ----------------------------------------------- |
| 1000 Genomes Project (IGSR) | 2,504 WGS      | `data/external/1000genomes/` empty -- pending   |
| gnomAD v4.1 exomes          | ~730M variants | Done (filtered parquet)                         |

`population_1kg_af` added to `PHASE_4_FEATURES`; pending data download.

### 3B -- Disease Cohorts (Controlled Access)

Apply for these in parallel -- each takes 2-8 weeks for approval.

| Source                          | Data                                    | Application                      | Priority    |
| ------------------------------- | --------------------------------------- | -------------------------------- | ----------- |
| dbGaP / NCBI                    | TOPMed (300K WGS), CMG rare disease     | eRA Commons + institutional      | High        |
| EGA                             | European cancer + rare disease WGS      | Data Access Agreement            | High        |
| CMG (Centers for Mendelian Genomics) | High-quality rare disease trios    | Via dbGaP / AnVIL                | High        |
| UK Biobank                      | 470K WES + 200K WGS                     | Formal application               | Medium-high |
| All of Us                       | 250K+ WGS diverse ancestry              | Researcher Workbench (free)      | Medium      |

### 3C -- Pending Downloads

| File                    | Size     | Source                              | Status  |
| ----------------------- | -------- | ----------------------------------- | ------- |
| dbNSFP4.7a.zip          | ~30 GB   | Google Drive (registration required) | Pending |
| 1000G VCF chr*.vcf.gz   | ~100 GB  | IGSR portal (free)                  | Pending |
| SpliceAI scored VCF     | 27 GB    | On Drive                            | Done    |

---

## Phase 5 -- Clinical Validation and Deployment

- [ ] Prospective validation on gene panels (BRCA1/2, TP53, PTEN, ATM)
- [ ] Comparison against ClinVar star-rating on expert-reviewed variants
- [ ] Model card: training data, known limitations, ancestry coverage
- [ ] Manuscript draft: multi-modal genomic variant classifier with algorithm benchmarking

---

## Feature Roadmap

**Live (64 features -- current model):**
see `TABULAR_FEATURES` in `src/genomic_variant_classifier/models/variant_ensemble.py`

**PHASE_4_FEATURES (pending retrain):**

```text
esm2_delta_norm       -- ESM-2 embedding L2 distance (wt vs. mut); ~+0.03-0.06 AUROC
population_1kg_af     -- 1000 Genomes allele frequency
uncertainty_epistemic -- Deep Ensemble epistemic uncertainty (inference-time)
uncertainty_aleatoric -- Deep Ensemble aleatoric uncertainty (inference-time)
```

---

## Complexity Reference

| Algorithm                | Training           | Genomic scale                  |
| ------------------------ | ------------------ | ------------------------------ |
| GBTs (XGBoost/LightGBM)  | O(I\*n\*d\*log n)  | Excellent                      |
| MLP / KAN                | O(n\*L\*H^2[\*G])  | Good                           |
| ESM-2 (frozen inference) | N/A                | Excellent                      |
| GNN (sparse PPI)         | O(G\*(V+E)\*d)     | Excellent -- independent of n  |
| Deep Ensemble (M members)| O(M \* base)       | Good                           |
| SVM (RBF)                | O(n^2)             | INFEASIBLE above ~100K samples |

SVM is excluded from all production runs (n > 100K).

---

*This roadmap is a living document. Update after each phase gate.*
