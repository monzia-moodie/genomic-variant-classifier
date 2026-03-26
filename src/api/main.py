"""
src/api/main.py
===============
FastAPI REST API for the Genomic Variant Pathogenicity Classifier.

Endpoints
---------
  GET  /health           Liveness + readiness check
  GET  /info             Model metadata and feature list
  POST /predict          Classify a single variant
  POST /batch            Classify up to 1 000 variants

Usage
-----
  # Development
  uvicorn src.api.main:app --reload --port 8000

  # Production (inside Docker)
  gunicorn src.api.main:app -k uvicorn.workers.UvicornWorker \
      --bind 0.0.0.0:8000 --workers 2

Environment variables
---------------------
  MODEL_PATH          Path to serialised InferencePipeline artifact
                      (default: models/phase2_pipeline.joblib)
  GNOMAD_INDEX_PATH   Optional path to gnomAD parquet for live AF lookup
                      (default: None — callers must supply allele_freq)
  LOG_LEVEL           Python logging level (default: INFO)

Implementation notes
--------------------
* The model is loaded once at startup into a module-level ``_PIPELINE``
  singleton.  Concurrent requests share it read-only (joblib artifacts are
  thread-safe after load).
* Feature engineering is delegated to the InferencePipeline wrapper
  (``src/api/pipeline.py``), which replicates the 46-feature logic from
  ``DataPrepPipeline._engineer_features`` without any I/O side-effects.
* SHAP explanations are computed only when the request payload is small
  (≤ 10 variants) or when explicitly requested, to keep p99 latency low.

PHASE_2_FEATURES remaining (not yet in active feature set):
  - codon_position    (requires VEP annotation)
  - splice_ai_score   (requires SpliceAI or pre-scored TSV)
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    GeneSummaryResponse,
    HealthResponse,
    InfoResponse,
    PredictResponse,
    VariantPrediction,
    VariantRequest,
    score_to_classification,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (from environment)
# ---------------------------------------------------------------------------

MODEL_PATH: Path = Path(os.environ.get("MODEL_PATH", "models/phase2_pipeline.joblib"))
GNOMAD_INDEX_PATH: Optional[Path] = (
    Path(p) if (p := os.environ.get("GNOMAD_INDEX_PATH")) else None
)
GENE_SUMMARY_PATH: Optional[Path] = Path(
    os.environ.get("GENE_SUMMARY_PATH", "data/processed/gene_summary.parquet")
)

# Filled at startup
_PIPELINE: Any = None                         # InferencePipeline instance
_GNOMAD_INDEX: Optional[pd.DataFrame] = None
_GENE_SUMMARY: Optional[pd.DataFrame] = None  # gene_symbol-indexed gene summary table
_START_TIME: float = time.monotonic()

# Model provenance — update after each training run
MODEL_VERSION    = "phase2-v1"
PIPELINE_VERSION = "1.0.0"
TRAINING_AUROC   = 0.9780
TRAINING_AUPRC   = 0.8936
HOLDOUT_AUROC    = 0.9847   # gene-stratified, 154 K variants


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model, optional gnomAD index, and gene summary table at startup."""
    global _PIPELINE, _GNOMAD_INDEX, _GENE_SUMMARY

    # --- Load inference pipeline ---
    if MODEL_PATH.exists():
        try:
            from src.api.pipeline import InferencePipeline
            _PIPELINE = InferencePipeline.load(MODEL_PATH)
            logger.info("Loaded inference pipeline from %s", MODEL_PATH)
        except Exception as exc:
            logger.error("Failed to load pipeline from %s: %s", MODEL_PATH, exc)
            # API starts in degraded mode; /health will report model_loaded=False
    else:
        logger.warning(
            "MODEL_PATH %s does not exist.  "
            "Run scripts/export_model.py to serialise the trained pipeline.",
            MODEL_PATH,
        )

    # --- Load gnomAD AF index (optional) ------------------------------------
    if GNOMAD_INDEX_PATH and GNOMAD_INDEX_PATH.exists():
        try:
            _GNOMAD_INDEX = pd.read_parquet(
                GNOMAD_INDEX_PATH, columns=["variant_id", "allele_freq"]
            )
            _GNOMAD_INDEX = _GNOMAD_INDEX.set_index("variant_id")
            logger.info("Loaded gnomAD AF index: %d loci", len(_GNOMAD_INDEX))
        except Exception as exc:
            logger.warning("Could not load gnomAD index: %s", exc)

    # --- Load gene summary table (strongly recommended) ---------------------
    if GENE_SUMMARY_PATH and GENE_SUMMARY_PATH.exists():
        try:
            _GENE_SUMMARY = pd.read_parquet(GENE_SUMMARY_PATH).set_index("gene_symbol")
            logger.info("Loaded gene summary: %d genes", len(_GENE_SUMMARY))
        except Exception as exc:
            logger.warning("Could not load gene summary: %s", exc)
    else:
        logger.warning(
            "GENE_SUMMARY_PATH %s not found — gene features will use defaults "
            "when callers omit them.  Build with scripts/build_gene_summary.py.",
            GENE_SUMMARY_PATH,
        )

    yield  # app is running

    logger.info("API shutting down.")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Genomic Variant Pathogenicity Classifier",
    version=PIPELINE_VERSION,
    description=(
        "Ensemble classifier for ClinVar/gnomAD-trained variant pathogenicity. "
        f"Holdout AUROC {HOLDOUT_AUROC} on 154,404 gene-stratified expert-reviewed variants."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # tighten in production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lookup_gene_count(gene_symbol: str) -> Optional[int]:
    """Return ClinVar pathogenic variant count for a gene, or None if table not loaded."""
    if _GENE_SUMMARY is None or not gene_symbol:
        return None
    try:
        return int(_GENE_SUMMARY.loc[gene_symbol, "n_pathogenic_in_gene"])
    except KeyError:
        return 0   # gene not in ClinVar — treat as no known pathogenic variants


def _lookup_gnomad_af(variant_id: str) -> Optional[float]:
    """Try to resolve allele frequency from the in-memory gnomAD index."""
    if _GNOMAD_INDEX is None:
        return None
    try:
        return float(_GNOMAD_INDEX.loc[variant_id, "allele_freq"])
    except KeyError:
        return 0.0   # absent from gnomAD → treat as ultra-rare
    except Exception:
        return None


def _variant_to_row(req: VariantRequest) -> dict:
    """Convert a VariantRequest to the flat dict expected by the pipeline."""
    variant_id = f"{req.chrom}:{req.pos}:{req.ref}:{req.alt}"

    af = req.allele_freq
    if af is None:
        af = _lookup_gnomad_af(variant_id)
    if af is None:
        af = 0.0   # conservative default

    return {
        "variant_id":             variant_id,
        "chrom":                  req.chrom,
        "pos":                    req.pos,
        "ref":                    req.ref,
        "alt":                    req.alt,
        "allele_freq":            af,
        "consequence":            req.consequence or "",
        "gene_symbol":            req.gene_symbol or "",
        "cadd_phred":             req.cadd_phred,
        "sift_score":             req.sift_score,
        "polyphen2_score":        req.polyphen2_score,
        "revel_score":            req.revel_score,
        "phylop_score":           req.phylop_score,
        "gerp_score":             req.gerp_score,
        "alphamissense_score":    req.alphamissense_score,
        "gene_constraint_oe":     req.gene_constraint_oe,
        "n_pathogenic_in_gene":   (
            req.n_pathogenic_in_gene
            if req.n_pathogenic_in_gene is not None
            else _lookup_gene_count(req.gene_symbol or "")
        ),
        "has_uniprot_annotation": req.has_uniprot_annotation or 0,
        "n_known_pathogenic_protein_variants": req.n_known_pathogenic_protein_variants or 0,
        # GTEx features omitted — engineer_features() defaults all to 0
    }


def _make_prediction(row: dict) -> VariantPrediction:
    """Run feature engineering + model inference for one variant dict."""
    result = _PIPELINE.predict_single(row)
    score  = float(np.clip(result["pathogenicity_score"], 0.0, 1.0))
    classification, confidence = score_to_classification(score)

    top_features: Optional[dict[str, float]] = None
    if hasattr(_PIPELINE, "feature_importances"):
        try:
            top_features = _PIPELINE.feature_importances(pd.DataFrame([row]), top_n=5)
        except Exception:
            pass

    return VariantPrediction(
        variant_id          = row["variant_id"],
        pathogenicity_score = round(score, 4),
        classification      = classification,
        confidence          = confidence,
        top_features        = top_features,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness and readiness check",
    tags=["ops"],
)
async def health() -> HealthResponse:
    return HealthResponse(
        status              = "ok" if _PIPELINE is not None else "degraded",
        model_loaded        = _PIPELINE is not None,
        gnomad_index_loaded = _GNOMAD_INDEX is not None,
        gene_counts_loaded  = _GENE_SUMMARY is not None,
        uptime_seconds      = round(time.monotonic() - _START_TIME, 1),
    )


@app.get(
    "/info",
    response_model=InfoResponse,
    summary="Model metadata, version, and feature list",
    tags=["ops"],
)
async def info() -> InfoResponse:
    feature_names: list[str] = []
    n_features = 0

    if _PIPELINE is not None and hasattr(_PIPELINE, "metadata"):
        feature_names = list(_PIPELINE.metadata.feature_names)
        n_features    = _PIPELINE.metadata.n_features

    return InfoResponse(
        model_version             = MODEL_VERSION,
        pipeline_version          = PIPELINE_VERSION,
        training_auroc            = TRAINING_AUROC,
        training_auprc            = TRAINING_AUPRC,
        holdout_auroc             = HOLDOUT_AUROC,
        n_features                = n_features,
        feature_names             = feature_names,
        phase2_features_remaining = [
            "codon_position (requires VEP)",
            "splice_ai_score (requires SpliceAI or pre-scored TSV)",
        ],
        description=(
            "LightGBM / XGBoost / GBM / RF / LR ensemble with stacking "
            "meta-learner.  Trained on 1.2 M tier-2 ClinVar variants with "
            "gnomAD v4.1 AF and AlphaMissense scores."
        ),
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Classify a single genomic variant",
    tags=["inference"],
)
async def predict(request: VariantRequest) -> PredictResponse:
    if _PIPELINE is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Model not loaded.  "
                "Check MODEL_PATH and run scripts/export_model.py."
            ),
        )
    try:
        row  = _variant_to_row(request)
        pred = _make_prediction(row)
    except Exception as exc:
        logger.exception("Prediction failed for %s", request)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {exc}",
        ) from exc

    return PredictResponse(
        prediction       = pred,
        model_version    = MODEL_VERSION,
        pipeline_version = PIPELINE_VERSION,
    )


@app.post(
    "/batch",
    response_model=BatchPredictResponse,
    summary="Classify up to 1 000 variants",
    tags=["inference"],
)
async def batch_predict(request: BatchPredictRequest) -> BatchPredictResponse:
    if _PIPELINE is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded.",
        )
    try:
        rows        = [_variant_to_row(v) for v in request.variants]
        raw_results = _PIPELINE.predict_batch(rows)

        predictions: list[VariantPrediction] = []
        for row, result in zip(rows, raw_results):
            score          = float(np.clip(result["pathogenicity_score"], 0.0, 1.0))
            classification, confidence = score_to_classification(score)
            predictions.append(VariantPrediction(
                variant_id          = row["variant_id"],
                pathogenicity_score = round(score, 4),
                classification      = classification,
                confidence          = confidence,
            ))

    except Exception as exc:
        logger.exception("Batch prediction failed (%d variants)", len(request.variants))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction error: {exc}",
        ) from exc

    classes = [p.classification for p in predictions]
    return BatchPredictResponse(
        predictions      = predictions,
        n_pathogenic     = sum(1 for c in classes if "Pathogenic" in c),
        n_benign         = sum(1 for c in classes if "Benign" in c),
        n_uncertain      = sum(1 for c in classes if "Uncertain" in c),
        model_version    = MODEL_VERSION,
        pipeline_version = PIPELINE_VERSION,
    )


@app.get(
    "/gene/{gene_symbol}",
    response_model=GeneSummaryResponse,
    summary="Gene-level features for request enrichment (n_pathogenic_in_gene, gene_constraint_oe, has_uniprot_annotation)",
    tags=["reference"],
)
async def gene_summary(gene_symbol: str) -> GeneSummaryResponse:
    """
    Return the three gene-level features used by the model for a given HGNC symbol.

    Callers can use this to auto-enrich /predict or /batch requests rather than
    looking up counts themselves.  gene_constraint_oe is null when gnomAD
    constraint data has not been loaded; engineer_features() will default to 1.0.
    """
    if _GENE_SUMMARY is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Gene summary table not loaded.  Check GENE_SUMMARY_PATH.",
        )
    try:
        row = _GENE_SUMMARY.loc[gene_symbol]
    except KeyError:
        # Unknown gene — return zeros (conservative defaults)
        return GeneSummaryResponse(
            gene_symbol          = gene_symbol,
            n_pathogenic_in_gene = 0,
            gene_constraint_oe   = None,
            has_uniprot_annotation = 0,
        )

    oe = row["gene_constraint_oe"]
    return GeneSummaryResponse(
        gene_symbol            = gene_symbol,
        n_pathogenic_in_gene   = int(row["n_pathogenic_in_gene"]),
        gene_constraint_oe     = None if pd.isna(oe) else float(oe),
        has_uniprot_annotation = int(row["has_uniprot_annotation"]),
    )


# ---------------------------------------------------------------------------
# Global error handler
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def _global_handler(request, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error."},
    )
