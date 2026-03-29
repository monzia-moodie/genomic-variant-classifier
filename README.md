# Genomic Variant Classifier
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
# Genomic Variant Pathogenicity Classifier

A production-grade, continuously self-updating machine learning system for classifying
genomic variant pathogenicity — designed to remain clinically current as scientific
knowledge, gene annotations, and population databases evolve.

## What this is

A research-to-production ML pipeline that classifies genomic variants (SNVs, indels)
as Pathogenic, Likely Pathogenic, Uncertain Significance, Likely Benign, or Benign
using an ensemble of gradient boosting models, a Graph Attention Network over the
STRING protein-protein interaction graph, ESM-2 protein language model embeddings,
and a continual learning framework that automatically detects and adapts to data drift.

**Holdout AUROC: 0.9847** on 154,404 gene-stratified expert-reviewed ClinVar variants.
Target post-Phase-4 retrain: ≥ 0.990.

## Architecture

```
ClinVar · gnomAD · AlphaMissense · SpliceAI · EVE · OMIM · ClinGen
· dbNSFP · GTEx · UniProt · dbSNP · AlphaFold · STRING · 1000 Genomes
                         │
              14-step Spark ETL annotation pipeline
                         │
            64-feature engineering (real_data_prep.py)
                         │
        ┌────────────────┼────────────────────────┐
        │                │                        │
  5-model ensemble    GNN (GAT)              ESM-2 embeddings
  LightGBM/XGBoost/   STRING DB PPI          protein language
  RF/GBM/LR           graph topology         model (esm2_delta_norm)
        │                │                        │
        └────────────────┴────────────────────────┘
                         │
               Stacking meta-learner (LR)
               + Platt calibration
               + Conformal prediction intervals
               + Epistemic/aleatoric uncertainty
                         │
               ClinicalEvaluator (AUROC, AUPRC, ECE,
               calibration, per-consequence breakdown)
                         │
    ┌────────────────────┴──────────────────────────┐
    │                                               │
 FastAPI REST API                       Continual Learning System
 7 endpoints · auth · rate-limit        PSI / MMD / KS drift detection
 Docker · GHCR · CI/CD                  ClinVar reclassification tracker
 Prometheus · Grafana                   EWC + LSIF adaptive retraining
                                        Versioned model registry
                                        Shadow → production promotion
```

## Key properties

**Clinically robust** — Five-tier ACMGish classification (Pathogenic → Benign) with
empirically calibrated probability thresholds, conformal prediction intervals at
configurable coverage levels, and per-variant uncertainty scores that flag cases
requiring human expert review.

**Temporally aware** — A dedicated continual learning pipeline runs on every ClinVar
monthly release. It detects three types of scientific drift: (1) feature/covariate
drift as gnomAD cohorts expand and functional score models are retrained, (2) label
drift as ClinVar reclassifies variants, and (3) concept drift as new biology (e.g.
SpliceAI revealing previously misclassified splice variants) changes what features
predict pathogenicity. When drift exceeds configurable thresholds, adaptive retraining
is triggered automatically using Elastic Weight Consolidation (EWC) to prevent
catastrophic forgetting of stable biological signal while adapting to new knowledge.

**Scientifically current** — Integrates 15 data sources spanning population genetics
(gnomAD v4.1, 1000 Genomes), evolutionary conservation (PhyloP, GERP, EVE), deep
learning functional predictions (AlphaMissense, SpliceAI, ESM-2), gene-disease
knowledge bases (OMIM, ClinGen), protein structure (AlphaFold pLDDT), tissue
expression (GTEx v10), and protein-protein interaction topology (STRING DB v12).

**Production deployed** — FastAPI service on port 8000, Docker image published to
GHCR (`ghcr.io/monzia-moodie/genomic-variant-api`), CI/CD via GitHub Actions with
automated testing on Python 3.11/3.12, Docker smoke tests, and monthly scheduled
drift monitoring.

## Feature set (64 features)

| Group | Count | Key features |
|-------|-------|-------------|
| Allele frequency | 6 | af_raw, af_log10, af_is_absent, af_is_ultra_rare |
| Variant type | 7 | ref_len, alt_len, len_diff, is_snv, is_insertion |
| Consequence | 6 | consequence_severity, is_loss_of_function, is_missense |
| Functional scores | 9 | CADD, SIFT, PolyPhen-2, REVEL, PhyloP, GERP, AlphaMissense, SpliceAI, EVE |
| Score flags | 5 | cadd_high, sift_deleterious, n_tools_pathogenic |
| Gene-level | 4 | gene_constraint_oe, n_pathogenic_in_gene, gene_has_known_disease |
| Protein (UniProt) | 2 | has_uniprot_annotation, n_known_pathogenic_protein_variants |
| Expression (GTEx) | 6 | gtex_max_tpm, gtex_tissue_specificity, gtex_is_eqtl |
| Gene-disease | 3 | omim_n_diseases, omim_is_autosomal_dominant, clingen_validity_score |
| HGMD | 2 | hgmd_is_disease_mutation, hgmd_n_reports |
| Chromosome | 3 | is_autosome, is_sex_chrom, is_mitochondrial |
| GNN-derived | 1 | gnn_score (GAT over STRING PPI graph) |
| RNA splice | 4 | maxentscan_score, dist_to_splice_site, exon_number |
| Protein structure | 4 | alphafold_plddt, solvent_accessibility, secondary_structure_context |

## Data drift handling

The system implements the following drift detection and adaptation stack:

- **PSI (Population Stability Index)** — per-feature, runs on every data source update
- **Kolmogorov-Smirnov test** — nonparametric, continuous features
- **Maximum Mean Discrepancy (MMD)** — kernel-based joint distribution test
- **ADWIN** — adaptive windowing detector for streaming variant ingestion
- **Székely-Rizzo energy statistic** — sensitive to distribution shape changes
- **ClinVar reclassification tracker** — monitors flip rate in training set monthly
- **EWC (Elastic Weight Consolidation)** — protects important weights during retraining
- **Online EWC** — running Fisher estimate across multiple ClinVar releases
- **LSIF importance weighting** — density ratio estimation for sample re-weighting
- **Temporal sample decay** — exponentially downweights older ClinVar submissions
- **Versioned model registry** — staging → shadow → production lifecycle
- **Shadow deployment** — new models run in parallel before promotion

## REST API

```
GET  /health          Liveness + readiness
GET  /info            Model metadata, 64 features, drift status
GET  /metrics         Prometheus metrics
GET  /gene/{symbol}   Gene-level feature lookup
GET  /rsid/{rs_id}    rs-ID resolution + prediction
POST /predict         Single variant → 5-tier classification + uncertainty
POST /batch           Up to 1,000 variants
```

## Performance

| Metric | Value |
|--------|-------|
| Holdout AUROC | **0.9847** |
| Holdout AUPRC | 0.9442 |
| Holdout F1 | 0.9143 |
| Holdout MCC | 0.8302 |
| Brier score | 0.0377 |
| Evaluation set | 154,404 variants, gene-stratified split |
| Label source | ClinVar tier-2+ (expert-reviewed) |

## Repository structure

```
src/
  data/          — 15 database connectors + Spark ETL + DataPrepPipeline
  models/        — VariantEnsemble, GNN (GAT), KAN, MC-Dropout, EWC
  api/           — FastAPI service (7 endpoints) + InferencePipeline
  evaluation/    — ClinicalEvaluator, benchmark framework, conformal prediction
  monitoring/    — DriftDetector, ClinVarTracker, ModelRegistry
  training/      — ContinualLearner, EWC, OnlineEWC, TreeEWCProxy
  reports/       — HTML report generator
  pipelines/     — RNA splice pipeline, protein structure pipeline
scripts/
  run_phase2_eval.py       — main training entry point
  export_model.py          — InferencePipeline serialisation + smoke test
  run_drift_monitor.py     — monthly drift check CLI (exit 0/1/2/3)
  calibrate_thresholds.py  — empirical ACMG threshold calibration
  validate_external.py     — external cohort validation (LOVD, UK Biobank)
  conformal_prediction.py  — split conformal intervals
  benchmark.py             — algorithm comparison (LightGBM vs KAN vs DeepEnsemble)
models/
  registry.json            — versioned model registry
  phase2_pipeline.joblib   — current production InferencePipeline (2 GB)
  drift_reference.pkl      — DriftDetector reference snapshot
```

## Quickstart

```bash
# Run the API
MODEL_PATH=models/phase2_pipeline.joblib uvicorn src.api.main:app --port 8000

# Classify a variant
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"chrom":"17","pos":43115726,"ref":"AAC","alt":"A",
       "consequence":"frameshift_variant","allele_freq":0.0,
       "alphamissense_score":0.95,"n_pathogenic_in_gene":2800}'

# Run monthly drift check
python scripts/run_drift_monitor.py \
  --reference-splits outputs/phase2_with_gnomad/splits/ \
  --new-clinvar  data/processed/clinvar_grch38_2024_07.parquet \
  --old-clinvar  data/processed/clinvar_grch38_2024_01.parquet \
  --output-dir   outputs/drift_reports/2024_07/ \
  --auto-retrain

# Docker
docker compose up api
```
## Author
**Monzia Moodie** - [@monzia-moodie](https://github.com/monzia-moodie)
## License
MIT License
