# Genomic Variant Pathogenicity Classifier

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Holdout AUROC](https://img.shields.io/badge/Holdout%20AUROC-0.9847-brightgreen.svg)]()
[![Variants](https://img.shields.io/badge/Training%20variants-1.70M-blue.svg)]()

A production-grade, multi-modal machine learning system for the five-tier clinical
classification of human genomic variants -- **Pathogenic, Likely Pathogenic, Uncertain
Significance, Likely Benign, and Benign** -- in accordance with ACMG/AMP guidelines.

The system integrates genomic sequence data, population-stratified allele frequencies
from twelve biological databases, protein structural annotations, tissue-specific gene
expression, variant co-classification evidence, and whole-slide histopathology imaging
from The Cancer Genome Atlas into a unified stacking ensemble architecture, deployed
as a production FastAPI REST service with autonomous agentic monitoring.

**Holdout AUROC: 0.9847** on 154,404 gene-stratified expert-reviewed ClinVar variants.

---

## Architecture

The classifier operates as a three-branch fusion model:

**Tabular Branch** -- A stacking meta-learner trained on out-of-fold predictions from
ten base classifiers: Random Forest, XGBoost, LightGBM, CatBoost, Gradient Boosting,
Logistic Regression, SVM, Kolmogorov-Arnold Networks (KAN), MC Dropout, and Deep
Ensemble Wrapper. Input features span 73 dimensions drawn from twelve biological
databases.

**Sequence Branch** -- A 1D-CNN operating over 101 bp genomic context windows
(one-hot encoded) combined with ESM-2 protein language model embeddings capturing
evolutionary and structural variant context.

**Histopathology Branch** -- A ResNet-50 CNN fine-tuned on TCGA whole-slide image tiles
(224x224 px at 20x magnification) across TCGA-BRCA, TCGA-LUAD, and TCGA-COAD cohorts,
providing phenotypic validation that anchors molecular classifications in observable
tissue-level consequences.

```
ClinVar . gnomAD v4.1 . FinnGen R12 . 1000 Genomes . AlphaMissense . SpliceAI
. EVE . OMIM . ClinGen . dbNSFP . GTEx v10 . UniProt . LOVD . AlphaFold . STRING
                              |
               14-step Spark ETL annotation pipeline
                              |
           73-feature engineering (real_data_prep.py)
                              |
     +-----------+------------+------------+-----------+
     |                        |                        |
10-model stacking        GNN (GAT)              ESM-2 embeddings
RF . XGBoost . LGBM      STRING DB PPI          protein language
CatBoost . GBM . LR      graph topology         model embeddings
SVM . KAN . MCDrop                                     |
. DeepEnsemble                                  1D-CNN sequence
     |                        |                 branch (101 bp)
     +-----------+------------+----------------+-------+
                              |
                    ResNet-50 Histopathology Branch
                    TCGA-BRCA . TCGA-LUAD . TCGA-COAD
                    224x224 tiles . 20x magnification
                              |
                  Stacking meta-learner (LR)
                  + Platt calibration
                  + Conformal prediction intervals
                  + Epistemic/aleatoric uncertainty
                              |
              ClinicalEvaluator (AUROC, AUPRC, ECE,
              calibration, per-consequence breakdown)
                              |
     +------------------------+------------------------+
     |                                                 |
  FastAPI REST API                       Continual Learning System
  7 endpoints . auth . rate-limit        PSI / MMD / KS drift detection
  Docker . GHCR . CI/CD                 ClinVar reclassification tracker
  Prometheus . Grafana                  EWC + LSIF adaptive retraining
                                        Agentic monitoring layer
                                        Versioned model registry
                                        Shadow -> production promotion
```

## Key properties

**Clinically robust** -- Five-tier ACMG/AMP classification (Pathogenic to Benign) with
empirically calibrated probability thresholds, conformal prediction intervals at
configurable coverage levels, and per-variant uncertainty scores that flag cases
requiring human expert review.

**Temporally aware** -- A dedicated continual learning pipeline runs on every ClinVar
monthly release. It detects three types of scientific drift: (1) feature/covariate
drift as gnomAD cohorts expand and functional score models are retrained, (2) label
drift as ClinVar reclassifies variants, and (3) concept drift as new biology changes
what features predict pathogenicity. When drift exceeds configurable thresholds,
adaptive retraining is triggered automatically using Elastic Weight Consolidation (EWC)
to prevent catastrophic forgetting of stable biological signal.

**Scientifically current** -- Integrates 12 biological databases spanning population
genetics (gnomAD v4.1, FinnGen R12 with 500,348 Finnish individuals, 1000 Genomes
Phase 3 across 5 continental strata), evolutionary conservation (PhyloP, GERP, EVE),
deep learning functional predictions (AlphaMissense, SpliceAI, ESM-2), gene-disease
knowledge bases (OMIM, ClinGen, LOVD), protein structure (AlphaFold pLDDT), tissue
expression (GTEx v10), and protein-protein interaction topology (STRING DB v12).

**Phenotypically grounded** -- The TCGA histopathology branch provides an empirical
link between variant pathogenicity classification and observable tumor-tissue
morphology, validated across breast, lung adenocarcinoma, and colorectal cancer cohorts.

**Production deployed** -- FastAPI service on port 8000, Docker image published to
GHCR (ghcr.io/monzia-moodie/genomic-variant-api), CI/CD via GitHub Actions with
automated testing, Docker smoke tests, and monthly scheduled drift monitoring.

**Autonomously maintained** -- An agentic monitoring layer (DataFreshnessAgent,
InterpretabilityAgent, TrainingLifecycleAgent, LiteratureScoutAgent) continuously
monitors upstream databases, detects distribution shift, triggers targeted retraining,
and produces SHAP-based interpretability audits without manual intervention.

## Feature set (73 features)

| Group | Count | Key features |
|-------|-------|-------------|
| Allele frequency | 6 | af_raw, af_log10, af_is_absent, af_is_ultra_rare |
| Variant type | 7 | ref_len, alt_len, len_diff, is_snv, is_insertion, is_deletion |
| Consequence | 6 | consequence_severity, is_loss_of_function, is_missense, is_splice |
| Functional scores | 9 | CADD, SIFT, PolyPhen-2, REVEL, PhyloP, GERP, AlphaMissense, SpliceAI, EVE |
| Score flags | 5 | cadd_high, sift_deleterious, polyphen_probably_damaging, n_tools_pathogenic |
| Gene-level | 4 | gene_constraint_oe, n_pathogenic_in_gene, gene_has_known_disease |
| Protein (UniProt) | 2 | has_uniprot_annotation, n_known_pathogenic_protein_variants |
| Expression (GTEx) | 6 | gtex_max_tpm, gtex_tissue_specificity, gtex_is_eqtl, gtex_max_abs_effect |
| Gene-disease | 3 | omim_n_diseases, omim_is_autosomal_dominant, clingen_validity_score |
| HGMD | 2 | hgmd_is_disease_mutation, hgmd_n_reports |
| LOVD | 1 | lovd_variant_class (ordinal 0-4) |
| Chromosome | 3 | is_autosome, is_sex_chrom, is_mitochondrial |
| GNN-derived | 1 | gnn_score (GAT over STRING PPI graph) |
| RNA splice | 4 | maxentscan_score, dist_to_splice_site, exon_number, is_canonical_splice |
| Protein structure | 4 | alphafold_plddt, solvent_accessibility, secondary_structure_context, dist_to_active_site |
| 1000 Genomes AF | 5 | af_1kg_afr, af_1kg_eur, af_1kg_eas, af_1kg_sas, af_1kg_amr |
| FinnGen R12 AF | 3 | finngen_af_fin, finngen_af_nfsee, finngen_enrichment |

## Data drift handling

- **PSI (Population Stability Index)** -- per-feature, runs on every data source update
- **Kolmogorov-Smirnov test** -- nonparametric, continuous features
- **Maximum Mean Discrepancy (MMD)** -- kernel-based joint distribution test
- **ADWIN** -- adaptive windowing detector for streaming variant ingestion
- **Szekely-Rizzo energy statistic** -- sensitive to distribution shape changes
- **ClinVar reclassification tracker** -- monitors flip rate in training set monthly
- **EWC (Elastic Weight Consolidation)** -- protects important weights during retraining
- **Online EWC** -- running Fisher estimate across multiple ClinVar releases
- **LSIF importance weighting** -- density ratio estimation for sample re-weighting
- **Temporal sample decay** -- exponentially downweights older ClinVar submissions
- **Versioned model registry** -- staging -> shadow -> production lifecycle
- **Shadow deployment** -- new models run in parallel before promotion

## REST API

```
GET  /health          Liveness + readiness
GET  /info            Model metadata, 73 features, drift status
GET  /metrics         Prometheus metrics
GET  /gene/{symbol}   Gene-level feature lookup
GET  /rsid/{rs_id}    rs-ID resolution + prediction
POST /predict         Single variant -> 5-tier classification + uncertainty
POST /batch           Up to 1,000 variants
```

## Performance

Evaluated on **349,067 held-out variants** (gene-stratified; no gene appears in both
train and test). Training cohort: 1,197,216 variants (20.3% pathogenic).

| Metric | Value |
|--------|-------|
| Holdout AUROC | **0.9847** |
| Brier score | 0.0584 |
| Sensitivity @ 90% specificity | 0.900 |
| Specificity @ 90% sensitivity | 0.918 |
| Evaluation set | 349,067 variants, gene-stratified |
| Training set | 1,197,216 variants |
| Label source | ClinVar expert-reviewed (tier 2+) |

### Per-model performance (validation set)

| Model | AUROC | AUPRC | F1 (macro) | MCC | Brier |
|-------|-------|-------|-----------|-----|-------|
| **gradient_boosting** | **0.9756** | 0.9190 | 0.8876 | 0.7758 | 0.0497 |
| lightgbm | 0.9751 | 0.9171 | 0.8651 | 0.7522 | 0.0700 |
| logistic_regression | 0.9747 | 0.9133 | 0.8627 | 0.7499 | 0.0696 |
| catboost | 0.9744 | 0.9153 | 0.8657 | 0.7530 | 0.0708 |
| xgboost | 0.9743 | 0.9124 | 0.8471 | 0.7276 | 0.0930 |
| ENSEMBLE_STACKER | 0.9709 | 0.8572 | 0.8791 | 0.7630 | 0.0584 |
| random_forest | 0.9681 | 0.8892 | 0.8725 | 0.7536 | 0.0663 |

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
