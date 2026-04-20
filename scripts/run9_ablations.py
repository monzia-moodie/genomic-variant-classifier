"""
scripts/run9_ablations.py
==========================
Leave-one-class-out (LOCO) ablation harness for Run 9+.

NOT a replacement for scripts/run_phase2_eval.py. Coexists with it. The
mainline script runs the expensive data-prep once and persists scaled
feature frames to <o>/splits/; this harness reads those, applies an
ablation mask by zeroing matching-prefix columns, retrains the ensemble,
and writes the Rule-5 artefact set to its own output directory.

Why zeroing works as an ablation: the splits written by DataPrepPipeline
are already StandardScaler-transformed (mean 0, std 1). Setting a column
to 0 means every row has the training-set mean for that feature, which
is the correct "ignore this feature" semantic — tree models see zero
information gain, linear models see zero contribution.

SCHEMA CALIBRATION — 2026-04-19
--------------------------------
ABLATION_MASKS in this file is calibrated against the 78-column schema
confirmed by direct `_engineer_features` probe on 2026-04-19. Do NOT
assume your on-disk split parquet matches this schema — check it:

    python -c "import pandas as pd; \\
        print(pd.read_parquet('outputs/<run>/splits/X_train.parquet').shape)"

If your persisted parquet has 46 columns, it is stale (pre-4/19) and
needs regeneration via `scripts/run_phase2_eval.py` before any ablation
will produce meaningful results. The harness logs a WARNING when an
ablation matches zero columns, so stale parquet produces loud evidence
of the mismatch rather than silent non-ablation.

GNN CAVEAT (see PATCHES_AND_INTEGRATION.md Patch 6)
---------------------------------------------------
The persisted `gnn_score` column is likely a DEFAULT constant (0.0 for
most rows), not real GNN predictions — because DataPrepPipeline writes
splits BEFORE run_phase2_eval.py trains the GNN. The in-memory
overwrite at run_phase2_eval.py:271 never reaches disk.

Consequences for `no_gnn` ablation:
  - If you run ablations against splits written by unpatched mainline:
    the column is already ≈0, so ablation is a no-op. Harness logs it.
  - If you apply Patch 6a (re-inject real GNN scores in the harness):
    the ablation compares real-GNN vs zeroed-GNN. Correct.
  - If you apply Patch 6b (train GNN inside DataPrepPipeline): cleanest
    — splits contain real GNN scores from the start.

Typical Run 9 workflow
----------------------
# Step 1: produce the full baseline + persist splits (the slow part)
python scripts/run_phase2_eval.py `
    --clinvar data/processed/clinvar_grch38.parquet `
    --gnomad  data/processed/gnomad_v4_exomes.parquet `
    --spliceai      data/external/spliceai/spliceai_index.parquet `
    --alphamissense data/external/alphamissense/AlphaMissense_hg38.tsv.gz `
    --gtex-genes BRCA1 BRCA2 TP53 PTEN ATM `
    --string-db  auto `
    --output outputs/run9/full

# Step 2: run the five LOCO ablations against the same splits
foreach ($A in "no_spliceai","no_gnn","no_alphamissense",
               "no_conservation","no_population_af") {
    python scripts/run9_ablations.py `
        --splits-dir outputs/run9/full/splits `
        --ablation $A `
        --run-id run9 `
        --output-dir outputs/run9/$A
}

# Step 3: aggregate
python scripts/run9_aggregate_ablations.py --run-dir outputs/run9/

Ablation mask invariants
------------------------
ABLATION_MASKS is the single source of truth for feature-class groupings.
Every new feature class added to DataPrepPipeline MUST have an entry here
or Rule 6 of docs/RUN9_SCIENTIFIC_DESIGN.md is violated silently. The
prefix-match logic warns (does not fail) when a mask matches no columns,
so schema drift produces a log line rather than a silent miscalibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("run9_ablations")


# ---------------------------------------------------------------------------
# Ablation masks — calibrated against the 78-column schema
# ---------------------------------------------------------------------------
#
# Columns verified present on 2026-04-19 via DataPrepPipeline._engineer_features
# probe (50-row synthetic input; all 78 columns produced regardless of
# annotation config because _engineer_features uses df.get(col, default)):
#
#   Allele frequency (6):   af_raw, af_log10, af_is_absent,
#                           af_is_ultra_rare, af_is_rare, af_is_common
#   Length/type (7):        ref_len, alt_len, len_diff, is_snv,
#                           is_insertion, is_deletion, is_indel
#   Consequence (6):        consequence_severity, is_loss_of_function,
#                           is_missense, is_synonymous, is_splice, in_coding
#   Functional scores (9):  cadd_phred, sift_score, polyphen2_score,
#                           revel_score, phylop_score, gerp_score,
#                           alphamissense_score, splice_ai_score, eve_score
#   Binary flags (5):       cadd_high, sift_deleterious,
#                           polyphen_probably_damaging, revel_pathogenic,
#                           n_tools_pathogenic
#   Gene-level (4):         gene_constraint_oe, gene_is_constrained,
#                           n_pathogenic_in_gene, gene_has_known_disease
#   UniProt (2):            has_uniprot_annotation,
#                           n_known_pathogenic_protein_variants
#   GTEx (6):               gtex_max_tpm, gtex_n_tissues_expressed,
#                           gtex_tissue_specificity, gtex_is_eqtl,
#                           gtex_min_eqtl_pval, gtex_max_abs_effect
#   Coding position (1):    codon_position
#   dbSNP (1):              dbsnp_af
#   OMIM (2):               omim_n_diseases, omim_is_autosomal_dominant
#   ClinGen (1):            clingen_validity_score
#   HGMD (2):               hgmd_is_disease_mutation, hgmd_n_reports
#   LOVD (1):               lovd_variant_class
#   Chromosome (3):         is_autosome, is_sex_chrom, is_mitochondrial
#   GNN (1):                gnn_score        [SEE GNN CAVEAT ABOVE]
#   Splice mechanics (4):   maxentscan_score, dist_to_splice_site,
#                           exon_number, is_canonical_splice
#   Structure (3):          alphafold_plddt, solvent_accessibility,
#                           secondary_structure_context
#   Active site (1):        dist_to_active_site
#   1KG pop AF (5):         af_1kg_afr, af_1kg_eur, af_1kg_eas,
#                           af_1kg_sas, af_1kg_amr
#   FinnGen (3):            finngen_af_fin, finngen_af_nfsee,
#                           finngen_enrichment
#   ESM-2 (1):              esm2_delta_norm  [stub until HGVSp parser]
#   gnomAD constraint (4):  pli_score, loeuf, syn_z, mis_z
#
# Total: 78 columns.
#
# RUN 9 CORE (6 ablations):
#   full, no_spliceai, no_gnn, no_alphamissense,
#   no_conservation, no_population_af
#
# RUN 10+ EXTENSIONS (8 additional):
#   no_esm2, no_eve, no_alphafold, no_constraint_scores, no_gtex,
#   no_disease_dbs, no_splice_mechanics, no_individual_predictors
#
ABLATION_MASKS: dict[str, list[str]] = {
    # ── Baseline ──────────────────────────────────────────────────────────
    "full": [],
    # ── Run 9 core LOCO (6) ───────────────────────────────────────────────
    "no_spliceai": [
        # SpliceAI delta-score column only. `is_splice` stays live because
        # it's a CONSEQUENCE-based flag (derived from VEP consequence
        # strings), not a SpliceAI score. `maxentscan_score`,
        # `dist_to_splice_site`, etc. also stay live (mechanical splice
        # features, not SpliceAI). To ablate ALL splice signals, use
        # no_spliceai + no_splice_mechanics in sequence.
        "splice_ai_",
    ],
    "no_gnn": [
        # See GNN caveat in module docstring. Without Patch 6, this column
        # is a default constant on disk and the ablation is a no-op.
        "gnn_score",
        "gnn_embed_",
        "gnn_",
    ],
    "no_kan": [
        # KAN is a MODEL-LEVEL ablation, not a feature-level one. KAN uses
        # the same 78 input features as every other base estimator, so there
        # are no feature columns to zero. Instead, this ablation is handled
        # by the CLI flag --skip-kan, which pops KAN from the ensemble's
        # base_estimators before fit(). An empty prefix list here documents
        # that intent and causes apply_ablation() to log "matched zero columns",
        # which is the correct signal: no feature zeroing, model removal only.
        #
        # To run the no_kan ablation:
        #   python scripts/run9_ablations.py --ablation no_kan --skip-kan ...
        #
        # The harness enforces this pairing at runtime (see main()).
    ],
    "no_alphamissense": [
        "alphamissense_",
    ],
    "no_conservation": [
        # Inter-species conservation stack. phylop_score and gerp_score in
        # the current schema; prefix covers future additions
        # (phastcons_*, siphy_*).
        "phylop_",
        "gerp_",
        "phastcons_",
        "siphy_",
    ],
    "no_population_af": [
        # ALL allele-frequency signals: primary AF features (af_*),
        # 1000 Genomes population AFs (af_1kg_*), dbSNP AF, and FinnGen
        # (Finnish isolate) AFs. Tests the "rare = pathogenic" baseline
        # that AUROC is thought to depend on heavily.
        #
        # Note: `af_1kg_*` share the `af_` prefix so a single `af_` entry
        # covers all 6 primary + 5 1KG columns. Explicit `finngen_` and
        # `dbsnp_af` entries cover the rest.
        "af_",
        "finngen_",
        "dbsnp_af",
    ],
    # ── Run 10+ extensions ────────────────────────────────────────────────
    "no_esm2": [
        # Currently stub (returns 0.0 until HGVSp parser lands; see
        # INCIDENT_2026-04-17_esm2-hgvsp-parser.md). Ablation is a no-op
        # until ESM-2 becomes non-stub in Run 10+.
        "esm2_",
    ],
    "no_eve": [
        # Similar plumbing-gap issue to ESM-2. eve_score column exists but
        # may contain defaults unless EVE annotation is plumbed.
        "eve_",
    ],
    "no_alphafold": [
        # Structural context features derived from AlphaFold predictions.
        "alphafold_",
        "solvent_accessibility",
        "secondary_structure_context",
        "dist_to_active_site",
    ],
    "no_constraint_scores": [
        # gnomAD gene-level constraint scores: pLI (probability of LoF
        # intolerance), LOEUF (loss-of-function observed/expected upper
        # fraction), syn_z (synonymous z-score), mis_z (missense z-score).
        # Ablating these tests whether gene-level intolerance is a
        # dominant predictor vs. per-variant features.
        "pli_score",
        "loeuf",
        "syn_z",
        "mis_z",
    ],
    "no_gtex": [
        # Tissue expression and eQTL features from GTEx.
        "gtex_",
    ],
    "no_disease_dbs": [
        # LABEL-LEAKAGE CHECK. HGMD, OMIM, LOVD, ClinGen are curated
        # disease-variant databases. If these features strongly predict
        # ClinVar labels, it may be because some variants appear in both
        # sources — not because of independent biological signal.
        # A large `no_disease_dbs` Δ-AUROC should be investigated for
        # leakage before being claimed as a biological result.
        "omim_",
        "hgmd_",
        "lovd_",
        "clingen_",
    ],
    "no_splice_mechanics": [
        # Mechanical splice features (NOT SpliceAI). Distance to splice
        # sites, exon numbering, canonical vs non-canonical splice sites,
        # MaxEntScan splice-site scores.
        "maxentscan_",
        "dist_to_splice_site",
        "exon_number",
        "is_canonical_splice",
    ],
    "no_individual_predictors": [
        # Raw outputs and derived binary flags from the individual in-silico
        # predictors (CADD, SIFT, PolyPhen-2, REVEL) PLUS the ensemble
        # n_tools_pathogenic flag. Tests whether the modern predictors
        # (AlphaMissense, ESM-2, EVE) capture signal independently of the
        # legacy predictor stack.
        "cadd_phred",
        "cadd_high",
        "sift_score",
        "sift_deleterious",
        "polyphen2_score",
        "polyphen_probably_damaging",
        "revel_score",
        "revel_pathogenic",
        "n_tools_pathogenic",
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _git_sha() -> str:
    try:
        repo = Path(__file__).resolve().parent.parent
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _versions() -> dict[str, str]:
    v: dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for pkg in (
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "xgboost",
        "lightgbm",
        "catboost",
        "torch",
        "torch_geometric",
        "transformers",
        "shap",
    ):
        try:
            mod = __import__(pkg.replace("-", "_"))
            v[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            v[pkg] = "not_installed"
    try:
        import torch

        v["cuda"] = torch.version.cuda or "cpu_only"
        v["cuda_available"] = str(torch.cuda.is_available())
    except ImportError:
        pass
    return v


def _hash_df(df: pd.DataFrame) -> str:
    """Content hash of a dataframe's values (order-sensitive)."""
    h = hashlib.sha256()
    h.update(str(df.shape).encode())
    h.update(",".join(df.columns).encode())
    h.update(pd.util.hash_pandas_object(df, index=False).values.tobytes())
    return h.hexdigest()[:16]


def apply_ablation(
    X: pd.DataFrame,
    ablation: str,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Zero matching-prefix columns. Returns (X_ablated, zeroed_col_names).

    The 'full' ablation returns X unchanged. For every other ablation,
    a COPY of X is returned — the caller's X is never mutated.
    """
    if ablation not in ABLATION_MASKS:
        raise ValueError(
            f"Unknown ablation {ablation!r}. " f"Valid: {sorted(ABLATION_MASKS)}"
        )
    prefixes = ABLATION_MASKS[ablation]
    if not prefixes:
        return X, []

    X_abl = X.copy()
    zeroed: list[str] = []
    for col in X_abl.columns:
        if any(col.startswith(p) or col == p for p in prefixes):
            X_abl[col] = 0.0
            zeroed.append(col)

    if not zeroed:
        logger.warning(
            "Ablation %r matched zero columns. Either the feature class "
            "is not present in this pipeline, or ABLATION_MASKS prefixes "
            "are stale. Columns seen (first 25 of %d): %s",
            ablation,
            X.shape[1],
            list(X.columns)[:25],
        )
    else:
        # Also warn if a named feature is EXPECTED to be present but the
        # resulting column count is surprisingly low (common signal of
        # stale-parquet mismatch).
        logger.info(
            "Ablation %r zeroed %d columns: %s",
            ablation,
            len(zeroed),
            zeroed[:10] + (["...trimmed"] if len(zeroed) > 10 else []),
        )

        # Schema drift check: if the total column count is well below 78,
        # warn that the parquet may be stale.
        if X.shape[1] < 70:
            logger.warning(
                "Input has only %d columns — expected ~78 per the "
                "2026-04-19 schema probe. The split parquet may be "
                "stale (pre-4/19). Regenerate via scripts/run_phase2_eval.py "
                "before drawing conclusions from this ablation.",
                X.shape[1],
            )
    return X_abl, zeroed


# ---------------------------------------------------------------------------
# Splits loader
# ---------------------------------------------------------------------------
def load_splits(
    splits_dir: Path,
) -> dict[str, pd.DataFrame | pd.Series]:
    """
    Load the 8-tuple produced by DataPrepPipeline.run() from a directory
    written by DataPrepPipeline._save_splits (called via the mainline
    run_phase2_eval.py).

    Required files:
        X_train.parquet, X_val.parquet, X_test.parquet
        y_train.parquet, y_val.parquet, y_test.parquet
        meta_val.parquet, meta_test.parquet

    The y_*.parquet files use column name "label" (per DataPrepPipeline
    _save_splits line 1207: `y_train.to_frame("label").to_parquet(...)`).
    """
    required = [
        "X_train.parquet",
        "X_val.parquet",
        "X_test.parquet",
        "y_train.parquet",
        "y_val.parquet",
        "y_test.parquet",
        "meta_val.parquet",
        "meta_test.parquet",
    ]
    missing = [f for f in required if not (splits_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required split files in {splits_dir}: {missing}. "
            f"Run scripts/run_phase2_eval.py first and point --splits-dir "
            f"at its <output>/splits/ directory."
        )

    out: dict[str, pd.DataFrame | pd.Series] = {
        "X_train": pd.read_parquet(splits_dir / "X_train.parquet"),
        "X_val": pd.read_parquet(splits_dir / "X_val.parquet"),
        "X_test": pd.read_parquet(splits_dir / "X_test.parquet"),
        "meta_val": pd.read_parquet(splits_dir / "meta_val.parquet"),
        "meta_test": pd.read_parquet(splits_dir / "meta_test.parquet"),
    }
    for split in ("train", "val", "test"):
        y_df = pd.read_parquet(splits_dir / f"y_{split}.parquet")
        # DataPrepPipeline writes the Series as a one-column frame named
        # "label" (real_data_prep.py line 1207). Handle both that and the
        # generic one-column case for forward-compat.
        col = "label" if "label" in y_df.columns else y_df.columns[0]
        out[f"y_{split}"] = y_df[col].astype(int).reset_index(drop=True)

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(
        description="LOCO ablation harness for Run 9+ (78-column schema)"
    )
    p.add_argument(
        "--splits-dir",
        required=True,
        type=Path,
        help="Directory holding X_{train,val,test}.parquet, y_*.parquet, "
        "meta_{val,test}.parquet from a prior mainline run.",
    )
    p.add_argument(
        "--ablation",
        default="full",
        choices=sorted(ABLATION_MASKS.keys()),
        help="Feature-class ablation. 'full' zeros no columns (baseline).",
    )
    p.add_argument(
        "--run-id",
        required=True,
        help="Logical run identifier (e.g. 'run9'). Written to manifest.",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Artefact output root for THIS ablation.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Use the SAME value across ablations in one run.",
    )
    p.add_argument(
        "--skip-nn",
        action="store_true",
        help="Exclude cnn_1d and tabular_nn (smoke tests only).",
    )
    p.add_argument("--skip-svm", action="store_true")
    p.add_argument(
        "--skip-kan",
        action="store_true",
        help="Exclude KAN base estimator. KANClassifier already caps memory "
        "via a 100K-sample stratified subsample gate; this flag is "
        "provided as an opt-in override.",
    )
    p.add_argument(
        "--skip-mc-dropout",
        action="store_true",
        help="Exclude mc_dropout and deep_ensemble base estimators (the "
        "two neural-ensemble wrappers). Useful for CPU smoke tests "
        "where these dominate wall-clock.",
    )
    p.add_argument("--skip-shap", action="store_true")
    p.add_argument("--skip-permutation", action="store_true")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    t0 = time.perf_counter()
    logger.info("=" * 72)
    logger.info(
        "run9_ablations.py  run_id=%s  ablation=%s  seed=%d",
        args.run_id,
        args.ablation,
        args.seed,
    )
    logger.info("  splits_dir = %s", args.splits_dir)
    logger.info("  output_dir = %s", args.output_dir)
    logger.info("=" * 72)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Delayed heavy imports (keeps --help responsive) ───────────────────
    from src.models.variant_ensemble import (
        EnsembleConfig,
        VariantEnsemble,
    )
    from src.evaluation.evaluator import ClinicalEvaluator
    from src.evaluation.prediction_artifacts import RunArtifactWriter

    # ── Load splits ───────────────────────────────────────────────────────
    splits = load_splits(args.splits_dir)
    X_train, X_val, X_test = splits["X_train"], splits["X_val"], splits["X_test"]
    y_train, y_val, y_test = splits["y_train"], splits["y_val"], splits["y_test"]
    meta_val, meta_test = splits["meta_val"], splits["meta_test"]

    logger.info(
        "Splits loaded: train=%d val=%d test=%d features=%d",
        len(X_train),
        len(X_val),
        len(X_test),
        X_train.shape[1],
    )

    # Hash the splits BEFORE ablation so the manifest records the canonical
    # split that all ablations share (cross-ablation comparability).
    split_hashes = {
        "X_train_full": _hash_df(X_train),
        "X_val_full": _hash_df(X_val),
        "X_test_full": _hash_df(X_test),
    }

    # ── Apply ablation to all three frames ────────────────────────────────
    # Enforce the no_kan ablation pairing: it's a model-level removal
    # (zero columns zeroed), so --skip-kan must be set or the ablation
    # silently no-ops and KAN still participates in the ensemble.
    if args.ablation == "no_kan" and not args.skip_kan:
        logger.error(
            "--ablation no_kan requires --skip-kan. The no_kan ablation "
            "is a model-level removal of KAN from the ensemble, not a "
            "feature-column zero-out. Pass --skip-kan alongside "
            "--ablation no_kan to actually run it."
        )
        return 2

    X_train_abl, zeroed_train = apply_ablation(X_train, args.ablation)
    X_val_abl, zeroed_val = apply_ablation(X_val, args.ablation)
    X_test_abl, zeroed_test = apply_ablation(X_test, args.ablation)

    assert zeroed_train == zeroed_val == zeroed_test, (
        "Inconsistent ablation across splits — column schema differs "
        "between train/val/test. This is a Rule-6 violation; check "
        "DataPrepPipeline for schema-drift across splits."
    )

    # ── Sequence placeholder (mirrors mainline) ───────────────────────────
    poly_a = "A" * 101
    seq_tr = pd.Series([poly_a] * len(y_train))
    seq_va = pd.Series([poly_a] * len(y_val))
    seq_te = pd.Series([poly_a] * len(y_test))

    # ── Ensemble ──────────────────────────────────────────────────────────
    ens_cfg = EnsembleConfig(
        n_folds=args.n_folds,
        random_state=args.seed,
        model_dir=args.output_dir / "models",
        skip_kan=args.skip_kan,
        skip_mc_dropout=args.skip_mc_dropout,
    )
    ensemble = VariantEnsemble(ens_cfg)

    # Mirror mainline skip rules. KAN and mc_dropout/deep_ensemble are
    # controlled via EnsembleConfig above, not popped here.
    if args.skip_nn:
        ensemble.base_estimators.pop("cnn_1d", None)
        ensemble.base_estimators.pop("tabular_nn", None)
    if args.skip_svm or len(y_train) > 100_000:
        ensemble.base_estimators.pop("svm", None)
        logger.info("SVM skipped: O(n²) infeasible at n=%d", len(y_train))

    # ── Fit ───────────────────────────────────────────────────────────────
    fit_t0 = time.perf_counter()
    ensemble.fit(X_train_abl, seq_tr, y_train)
    fit_sec = time.perf_counter() - fit_t0
    logger.info("Fit complete in %.1f min", fit_sec / 60)
    ensemble.save(args.output_dir / "models" / "ensemble.joblib")

    # ── Evaluate ──────────────────────────────────────────────────────────
    test_results = ensemble.evaluate(X_test_abl, seq_te, y_test)
    val_results = ensemble.evaluate(X_val_abl, seq_va, y_val)

    test_proba = ensemble.predict_proba(X_test_abl, seq_te)[:, 1]

    # Per-base-model test probabilities (for test_predictions.parquet)
    base_probs: dict[str, np.ndarray] = {}
    for name, model in ensemble.trained_models_.items():
        X_input = seq_te if name == "cnn_1d" else X_test_abl.values
        try:
            base_probs[name] = model.predict_proba(X_input)[:, 1]
        except Exception as exc:
            logger.warning("base-model %s predict_proba failed: %s", name, exc)

    evaluator = ClinicalEvaluator(n_bootstrap=1000, random_state=args.seed)
    report = evaluator.evaluate(
        y_true=y_test,
        y_proba=test_proba,
        meta=meta_test,
        model_name=f"{args.run_id}_{args.ablation}",
    )

    # ── Rule-5 artefacts ──────────────────────────────────────────────────
    writer = RunArtifactWriter(
        run_id=args.run_id,
        ablation=args.ablation,
        output_dir=args.output_dir,
    )

    writer.save_manifest(
        git_sha=_git_sha(),
        versions=_versions(),
        config={
            "ablation": args.ablation,
            "zeroed_columns": zeroed_train,
            "seed": args.seed,
            "n_folds": args.n_folds,
            "splits_dir": str(args.splits_dir),
            "split_hashes": split_hashes,
            "n_train": int(len(y_train)),
            "n_val": int(len(y_val)),
            "n_test": int(len(y_test)),
            "n_features": int(X_train.shape[1]),
            "feature_columns": list(X_train.columns),
            "ensemble_models": list(ensemble.trained_models_.keys()),
            "fit_seconds": round(fit_sec, 1),
            "run_started_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    writer.save_test_predictions(
        y_test=y_test,
        proba=test_proba,
        base_probs=base_probs,
        meta=meta_test,
    )
    writer.save_eval_report(report)
    writer.save_calibration(report)

    # OOF predictions — requires Patch 1 (VariantEnsemble.oof_predictions_)
    if hasattr(ensemble, "oof_predictions_"):
        oof = pd.DataFrame(
            ensemble.oof_predictions_,
            columns=[
                f"{n}_prob"
                for n in getattr(
                    ensemble,
                    "oof_model_names_",
                    ensemble.trained_models_.keys(),
                )
            ],
        )
        oof["label"] = y_train.values
        oof["fold"] = -1
        if "variant_id" in X_train.columns:
            oof["variant_id"] = X_train["variant_id"].values
        else:
            oof["variant_id"] = [f"train_{i}" for i in range(len(oof))]
        writer.save_oof_predictions(oof)
    else:
        logger.warning(
            "ensemble has no `oof_predictions_` attribute — apply Patch 1 "
            "to src/models/variant_ensemble.py (see "
            "docs/PATCHES_AND_INTEGRATION.md)."
        )

    if not args.skip_shap:
        try:
            writer.save_shap_values(ensemble, base_probs, meta_test, top_k=20)
        except Exception as exc:
            logger.error("SHAP failed: %s", exc)

    if not args.skip_permutation:
        try:
            writer.save_permutation_importance(
                ensemble,
                X_test_abl,
                seq_te,
                y_test,
                n_repeats=5,
                sample_size=min(50_000, len(X_test_abl)),
                seed=args.seed,
            )
        except Exception as exc:
            logger.error("Permutation importance failed: %s", exc)

    # Mainline-compatible artefacts
    (args.output_dir / "per_model_metrics.csv").write_text(test_results.to_csv())
    (args.output_dir / "per_model_metrics_val.csv").write_text(val_results.to_csv())

    # Headline ablation row (for the aggregator)
    aggregator_path = args.output_dir.parent / "ablation_results.parquet"
    ens_row = test_results.loc["ENSEMBLE_STACKER"]
    ens_val = val_results.loc["ENSEMBLE_STACKER"]
    writer.append_ablation_row(
        aggregator_path,
        {
            "ablation": args.ablation,
            "auroc": float(ens_row["auroc"]),
            "auprc": float(ens_row["auprc"]),
            "mcc": float(ens_row["mcc"]),
            "brier": float(ens_row["brier"]),
            "val_auroc": float(ens_val["auroc"]),
            "val_auprc": float(ens_val["auprc"]),
            "val_mcc": float(ens_val["mcc"]),
            "n_zeroed": len(zeroed_train),
            "fit_seconds": round(fit_sec, 1),
        },
    )

    total = time.perf_counter() - t0
    logger.info("=" * 72)
    logger.info(
        "DONE  run_id=%s  ablation=%s  AUROC(test)=%.5f  AUROC(val)=%.5f  %.1f min",
        args.run_id,
        args.ablation,
        float(ens_row["auroc"]),
        float(ens_val["auroc"]),
        total / 60,
    )
    logger.info("  artefacts: %s", args.output_dir)
    logger.info("  aggregator: %s", aggregator_path)
    logger.info("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
