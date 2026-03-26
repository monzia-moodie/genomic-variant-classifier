# Methods

## Genomic Variant Pathogenicity Classifier — Technical Description

**Version:** Phase 2 (v2.0.0)
**Holdout AUROC:** 0.9847 (gene-stratified, 154,404 variants)

---

## 1. Data

### 1.1 Training labels

Variants were obtained from ClinVar (GRCh38, quarterly release) and filtered to
high-confidence clinical classifications using ClinVar review status tier ≤ 3
(criteria provided by at least one submitter, no conflicting interpretations).
Pathogenic and Likely pathogenic variants were assigned label 1; Benign and
Likely benign variants were assigned label 0. Variants of Uncertain Significance
(VUS) and those with conflicting interpretations were excluded from training.

Resulting label distribution: ~15% pathogenic, ~85% benign (~1.2 M variants
after quality filtering).

### 1.2 Feature annotation

Variant annotations were added from the following sources, in pipeline order:

| Step | Source | Features |
|------|--------|----------|
| 1 | dbNSFP v4.6 | SIFT, PolyPhen-2 HDIV, REVEL, CADD raw, PhyloP 100-way, GERP++ |
| 2 | PhyloP v1 | phylop_score (multi-alignment conservation override) |
| 3 | CADD v1.7 | cadd_phred (REST API or pre-scored file; optional) |
| 4 | SpliceAI v1.3 | splice_ai_score (max delta score across 4 splice signals) |
| 5 | AlphaMissense | alphamissense_score (per-amino-acid pathogenicity) |
| 6 | GTEx v8 | max TPM, tissue expression breadth, eQTL flag, effect size |
| 7 | VEP v110 | Consequence, codon position, exon/intron annotation |
| 8 | OMIM | Disease count, inheritance mode |
| 9 | ClinGen | Gene validity score (curated gene–disease relationships) |
| 10 | dbSNP build 156 | Supplemental allele frequency |
| 11 | EVE | Evolutionary model variant effect score |
| 12 | HGMD Professional | Disease mutation flag, report count |
| 13 | MaxEntScan (Phase 6.1) | Splice-site strength score, distance to canonical splice site, exon number, canonical GT-AG flag |
| 14 | AlphaFold / UniProt (Phase 6.2) | Per-residue pLDDT, relative solvent accessibility, secondary structure class, distance to active site |

Allele frequencies were sourced primarily from gnomAD v4.1 exomes; variants
absent from gnomAD were supplemented with 1000 Genomes Phase 3 allele
frequencies.

### 1.3 Train / validation / test split

Variants were split gene-stratified (no gene appears in more than one split)
using GroupShuffleSplit:

- **Train**: 70% of genes
- **Validation**: 10% of genes (used for hyperparameter selection and Platt
  scaling)
- **Test / holdout**: 20% of genes (single evaluation at end of training)

This strategy prevents gene-level label leakage, which inflates AUROC estimates
when the same gene appears in both train and test sets.

---

## 2. Feature Engineering

A total of **64 tabular features** were derived from raw annotations. Features
are grouped into:

| Group | Count | Description |
|-------|-------|-------------|
| Allele frequency | 6 | Raw AF, log₁₀ AF, binary rarity indicators |
| Variant type | 7 | SNV/indel, insertion/deletion, ref/alt length |
| Consequence | 6 | Missense, LoF, splice, coding, severity score |
| Conservation | 9 | PhyloP, GERP, CADD, REVEL, SpliceAI, EVE |
| Protein function | 5 | SIFT, PolyPhen-2, AlphaMissense, codon position, dbSNP AF |
| Gene-level | 4 | n_pathogenic_in_gene, gene_constraint_oe (pLI/LOEUF proxy), UniProt annotation, n_known_pathogenic |
| GTEx expression | 6 | Max TPM, tissue breadth, specificity, eQTL flag, p-value, effect size |
| Gene-disease | 5 | OMIM disease count, dominant inheritance, ClinGen validity, HGMD mutation flag, HGMD report count |
| Chromosome | 2 | One-hot: autosomal, X chromosome |
| Genomic position | 3 | Position log₁₀, GC-content proxy, repeat mask flag (via ref_len) |
| Gene network (GNN) | 1 | Gene-level pathogenicity score from STRING protein interaction graph |
| RNA splice context | 4 | MaxEntScan score, distance to splice site, exon number, canonical GT-AG flag |
| Protein structure | 4 | AlphaFold pLDDT, relative solvent accessibility, secondary structure, distance to active site |

Missing values were imputed with biologically neutral defaults (e.g., AF = 0 for
absent from gnomAD, SIFT = 0.5 for uncovered positions).

---

## 3. Model Architecture

### 3.1 Base estimators

Four tabular base models were trained on the 64-feature matrix:

| Model | Library | Key hyperparameters |
|-------|---------|---------------------|
| LightGBM | lightgbm 4.x | num_leaves=63, learning_rate=0.05, n_estimators=500 |
| XGBoost | xgboost 2.x | max_depth=6, learning_rate=0.05, n_estimators=500 |
| Gradient Boosting | scikit-learn | max_depth=5, learning_rate=0.05, n_estimators=300 |
| Random Forest | scikit-learn | n_estimators=300, max_features=0.4 |

Hyperparameters were optimised using Optuna (TPE sampler, 100 trials) on the
validation split. Final hyperparameters are stored in
`models/best_lgbm_params.json` and `models/best_xgboost_params.json`.

A 1D-CNN sequence model was trained on 101-bp FASTA context windows but is
excluded from the inference pipeline (requires sequence context unavailable
at API inference time).

### 3.2 Stacking meta-learner

Out-of-fold predictions from 5-fold cross-validation were used as inputs to a
Logistic Regression meta-learner (C = 1.0, solver = lbfgs). The stacking
ensemble averages model strengths and reduces variance.

### 3.3 Graph neural network (gene-level prior)

A variant graph was constructed from the STRING protein interaction database
(combined score ≥ 500) with genes as nodes and interaction confidence as edge
weights. A 3-layer Graph Attention Network (GAT) with 64-dimensional hidden
layers was trained to predict gene-level pathogenicity from the mean variant
features per gene. At inference time, gene-level GNN scores are pre-computed
and stored as a lookup table (`GNNScorer`), enabling O(1) retrieval.

---

## 4. Calibration

Probability calibration was performed using Platt scaling (logistic regression
fit on the validation split predictions). Classification thresholds were set by
anchoring to ≥ 90% positive predictive value (PPV) for the Pathogenic tier on
the validation set, then sweeping down through the ACMG five-tier scale.
Calibrated thresholds are stored in `models/classification_thresholds.json`.

Conformal prediction intervals (split conformal, Papadopoulos 2002) provide
guaranteed marginal coverage at α ∈ {0.01, 0.05, 0.10, 0.20} calibrated on
the validation split.

---

## 5. Evaluation

### 5.1 Held-out performance

| Metric | Value |
|--------|-------|
| AUROC (gene-stratified holdout) | 0.9847 |
| AUPRC | 0.8936 |
| ECE (15 bins) | see `outputs/calibration/calibration_metrics.json` |

### 5.2 External validation

The pipeline can be evaluated against external cohorts using
`scripts/validate_external.py`, which produces AUROC, AUPRC, ECE, MCE, and
per-threshold sensitivity/specificity/PPV/NPV breakdowns.

---

## 6. Software and Reproducibility

| Component | Version |
|-----------|---------|
| Python | 3.11–3.12 |
| LightGBM | ≥ 4.3 |
| XGBoost | ≥ 2.0 |
| scikit-learn | ≥ 1.4 |
| FastAPI | ≥ 0.111 |
| Optuna | ≥ 3.5 |

Training is fully reproducible from `scripts/run_phase2_eval.py` given the
same input data and random seed (`--random-state 42`).  All splits and
hyperparameter search results are written to `data/splits/` and
`models/` respectively.

---

## 7. Ethical Considerations

This classifier is intended as a research and clinical decision-support tool.
Outputs should be interpreted by trained clinical geneticists in the context of
the full clinical picture.  The model does not account for compound
heterozygosity, polygenic risk, or phenotype-specific penetrance.  Users are
responsible for compliance with institutional guidelines for variant
interpretation.

---

## References

1. Landrum MJ et al. ClinVar: improving access to variant interpretations and
   supporting evidence. *Nucleic Acids Res.* 2018;46:D1062-D1067.
2. Karczewski KJ et al. The mutational constraint spectrum quantified from
   variation in 141,456 humans. *Nature.* 2020;581:434-443.
3. Cheng J et al. Accurate proteome-wide missense variant effect prediction
   with AlphaMissense. *Science.* 2023;381:eadg7492.
4. Jaganathan K et al. Predicting Splicing from Primary Sequence with Deep
   Learning. *Cell.* 2019;176:535-548.
5. Yeo G, Burge CB. Maximum Entropy Modeling of Short Sequence Motifs with
   Applications to RNA Splicing Signals. *J Comput Biol.* 2004;11:377-394.
6. Jumper J et al. Highly accurate protein structure prediction with
   AlphaFold. *Nature.* 2021;596:583-589.
7. Szklarczyk D et al. The STRING database in 2021. *Nucleic Acids Res.*
   2021;49:D605-D612.
8. Papadopoulos H et al. Inductive confidence machines for regression.
   *ECML.* 2002;345-356.
