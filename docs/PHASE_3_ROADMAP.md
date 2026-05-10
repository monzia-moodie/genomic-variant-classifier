# Phase 3 Roadmap — Genomic Variant Classifier
**Author:** Monzia Moodie  
**Repository:** monzia-moodie/genomic-variant-classifier  
**Last updated:** 2026-04-04  
**Commits this session:** 2389ee2, 54cf817, c401f25

---

## Run 4 Post-Mortem

| Issue | Root Cause | Fix | Commit |
|---|---|---|---|
| XGBoost MCC 0.8565 despite AUROC 0.9940 | score distribution compressed near boundary; scale_pos_weight shifts decision surface but raw probabilities uncorrected | _IsotonicCalibrator on dedicated 15% cal split | 2389ee2 |
| PPV≥80% n_flagged=1 | `_find_high_ppv_point` returned on first threshold (highest score, PPV=1.0, n=1) | Already correct in current evaluator.py — confirmed | n/a |
| KAN OOM 17.9 GB | pykan materialises [n_samples, grid_size, n_features] tensor in one shot | Stratified subsample to max_fit_samples=100,000 | 2389ee2 |
| Ensemble stacker 0.9938 < best base 0.9941 | LR meta-learner fails when base model scores are highly correlated | Nelder-Mead convex blend search over OOF predictions | 54cf817 |
| CI failure: InvalidParameterError cv='prefit' | CalibratedClassifierCV removed cv='prefit' in sklearn on Python 3.11 CI runner | Replaced with _IsotonicCalibrator using IsotonicRegression directly | c401f25 |
| CI failure: catboost missing from trained_models_ | else branch missing after _RECALIBRATE block — non-recalibrated models never stored | Restored else: self.trained_models_[name] = model | c401f25 |

---

## Phase 3A — Immediate Fixes ✅

- `VariantEnsemble.fit()`: 15% calibration split by index (preserves DataFrame for CatBoost); _IsotonicCalibrator for xgboost, lightgbm, random_forest
- `KANClassifier._fit_pykan()`: stratified subsample gate at 100K samples; peak RAM ~0.3 GB vs 17.9 GB
- `evaluator._find_high_ppv_point`: confirmed already correct in current file

## Phase 3B — Calibration and Stacking Overhaul ✅

- Nelder-Mead `_find_blend_weights()`: convex weights over OOF; logs per-model weights and delta vs LR stacker
- `predict_proba()`: uses blend_weights_ when present; falls back to LR for old artifacts on disk
- Temperature scaling in `calibrate_thresholds.py`: Brent method, T in (0.01, 10.0), ECE-selected vs Platt

## Run 5 Targets

| Metric | Run 4 | Target |
|---|---|---|
| Ensemble AUROC | 0.9938 | ≥ 0.9941 |
| XGBoost MCC | 0.8565 | ≥ 0.91 |
| PPV≥80% n_flagged | 1 | ≥ 3,000 |
| ECE | 0.0114 | ≤ 0.006 |
| Brier | 0.0198 | ≤ 0.016 |

## Phase 3C — Next (New Biological Signals)

Pending: ESM-2 embeddings (src/genomic_variant_classifier/data/esm2.py exists — ready to integrate),
gnomAD constraint metrics (pLI, LOEUF), LOVD (blocked — IP ban),
UK Biobank/TOPMed AFs

## Pending Maintenance

- GitHub Actions Node.js 20 deprecation: actions/checkout, actions/setup-python,
  actions/upload-artifact must be updated to Node.js 24 compatible versions
  before June 2, 2026 hard deadline. Update .github/workflows/ at next
  infrastructure session.

## Lessons Learned

- Always read the actual current file before writing edit instructions — stale
  snapshots caused multiple rounds of incorrect find/replace blocks
- PowerShell integrated terminal = same parser as standalone; Python function
  definitions must go in editor pane, never pasted into any terminal
- 73 features confirmed (not 69); KAN OOM scales linearly with feature count
- cv='prefit' removed from CalibratedClassifierCV in Python 3.11 sklearn;
  use IsotonicRegression directly for version-stable calibration
- Missing else branch after conditional assignment is a silent failure —
  models succeed but are never stored; always verify trained_models_ keys
  in debug runs before assuming a model is skipped

  ## Phase 3C — New Biological Signals ✅

| Feature group | Columns | Source | Status |
|---|---|---|---|
| ESM-2 protein LM delta norm | esm2_delta_norm (1) | HuggingFace / fair-esm | Live (stub 0.0 without torch) |
| gnomAD v4.1 gene constraint | pli_score, loeuf, syn_z, mis_z (4) | gnomad.v4.1.constraint_metrics.tsv | Live — 91 MB TSV downloaded |
| LOVD variant classification | lovd_variant_class | LOVD REST | Blocked — IP ban pending lift |
| UK Biobank / TOPMed AFs | af_ukb, af_topmed | Controlled access | Pending application |

Feature count: 73 → 74 (ESM-2) → 78 (gnomAD constraint)

Key finding: BRCA1 pLI=0.000 (LoF-tolerant, second-hit mechanism), mis_z=2.338 
(missense-constrained — correctly reflects BRCA1 biology). TP53 pLI=0.998, 
LOEUF=0.449 — correctly identified as LoF-intolerant.

## Run 6 Targets (after 3C features active)

| Metric | Run 5 target | Run 6 target |
|---|---|---|
| Ensemble AUROC | ≥ 0.9941 | ≥ 0.9955 |
| MCC | ≥ 0.930 | ≥ 0.940 |
| ECE | ≤ 0.006 | ≤ 0.005 |

## Phase 3D — Architecture Upgrades ✅

- STRING DB v12 graph built: 16,201 nodes, 236,930 edges at threshold=700
- Edge attributes upgraded 1-dim → 3-dim: [experimental, database, coexpression]
  Textmining excluded to prevent label leakage from disease papers
- VariantGAT: edge_dim=1 → edge_dim=3 across all 3 GATConv layers
- GNN training wired into run_phase2_eval.py via --string-db flag
- gnn_score injected into X_train/X_val/X_test after GNN training
- Bayesian UQ: predict_proba_with_uncertainty() now called by _predict_df()
  uncertainty_epistemic + uncertainty_aleatoric returned in every /predict response
- BRCA1-ERCC1 edge: database=0.54, coexpression=0.055 (experimental=0.0 expected)

## Phase 3E — Agentic Layer ✅

- DataFreshnessAgent: POST_DOWNLOAD_HOOKS list runs before Spark ETL
  Hook 1: patch_clinvar_alleles.py (mandatory — prevents AUROC collapse)
  Hook 2: MD5 checksum validation of downloaded VCFs
- InterpretabilityAgent: CatBoost SHAP branch added
  Uses model.get_feature_importance(Pool(X_df), type='ShapValues')
  Strips bias column; returns (n, f, 1) array consistent with XGB/LGB shape
  Rankings persisted in shared state for stability comparison
- TrainingLifecycleAgent: deferred to avoid EWC duplication
- LiteratureScoutAgent: deferred to Phase 4

## Run 5 Readiness

All Phase 3 code complete. Ready to train with:
- 78 features (73 original + ESM-2 + pli_score + loeuf + syn_z + mis_z)
- --string-db 700 flag for GNN training
- gnomAD constraint TSV at data/external/gnomad/
- CatBoost isotonic calibration active
- Nelder-Mead blend weights replacing LR stacker
- ESM-2 stub mode (0.0) until transformers+torch installed in training env