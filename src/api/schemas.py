"""
src/api/schemas.py
==================
Pydantic request / response schemas for the variant pathogenicity API.

Design notes
------------
* All optional fields default to None / sentinel values so that callers
  can submit a minimal {chrom, pos, ref, alt} payload and the inference
  pipeline will impute the rest (using population-mean defaults where safe,
  or conservative "unknown" fills where not).
* ``VariantRequest`` mirrors the raw-input columns accepted by
  ``DataPrepPipeline._engineer_features``; any additional columns in the
  parquet schema are simply ignored by the pipeline.
* ``BatchPredictRequest`` caps at MAX_BATCH_SIZE variants to bound memory.
  Larger jobs should use the offline ``scripts/run_phase2_eval.py`` path.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# Hard cap enforced in /batch endpoint.
MAX_BATCH_SIZE: int = 1_000


# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------

class VariantRequest(BaseModel):
    """A single genomic variant to classify."""

    # --- Required: locus + alleles -------------------------------------------
    chrom: str = Field(
        ...,
        description="Chromosome (e.g. '1', 'X', 'MT').  'chr' prefix accepted.",
        examples=["1", "17", "X"],
    )
    pos: int = Field(..., gt=0, description="1-based genomic position (GRCh38).")
    ref: str = Field(..., min_length=1, description="Reference allele (ACGT).")
    alt: str = Field(..., min_length=1, description="Alternate allele (ACGT).")

    # --- Optional: population allele frequency --------------------------------
    allele_freq: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "gnomAD v4.1 allele frequency.  If absent the pipeline will "
            "attempt a lookup against the in-memory gnomAD index; if the "
            "variant is not found, AF is treated as 0 (absent from gnomAD)."
        ),
    )

    # --- Optional: functional annotation ------------------------------------
    consequence: Optional[str] = Field(
        default=None,
        description=(
            "VEP consequence term or '&'-delimited list "
            "(e.g. 'missense_variant', 'stop_gained&splice_region_variant')."
        ),
    )
    gene_symbol: Optional[str] = Field(
        default=None,
        description="HGNC gene symbol (e.g. 'BRCA1').  Used for gene-level features.",
    )

    # --- Optional: pre-computed tool scores ---------------------------------
    cadd_phred:          Optional[float] = Field(default=None, ge=0.0)
    sift_score:          Optional[float] = Field(default=None, ge=0.0, le=1.0)
    polyphen2_score:     Optional[float] = Field(default=None, ge=0.0, le=1.0)
    revel_score:         Optional[float] = Field(default=None, ge=0.0, le=1.0)
    phylop_score:        Optional[float] = Field(default=None)
    gerp_score:          Optional[float] = Field(default=None)
    alphamissense_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="AlphaMissense pathogenicity score (0 = benign, 1 = pathogenic).",
    )

    # --- Optional: gene-level constraint and ClinVar gene reputation --------
    gene_constraint_oe: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="gnomAD pLoF observed/expected ratio for this gene.",
    )
    n_pathogenic_in_gene: Optional[int] = Field(
        default=None,
        ge=0,
        description=(
            "Number of ClinVar pathogenic variants in this gene. "
            "Top feature by importance (1448 vs next at 417). "
            "Defaults to 0 when absent — conservative but will underestimate "
            "pathogenicity for known disease genes (BRCA1, TP53, LDLR, etc.). "
            "Callers should supply this from a ClinVar gene summary lookup."
        ),
    )

    # --- Optional: UniProt protein features ---------------------------------
    has_uniprot_annotation: Optional[int] = Field(
        default=None,
        ge=0,
        le=1,
        description="1 if the gene has any UniProt functional annotation.",
    )
    n_known_pathogenic_protein_variants: Optional[int] = Field(
        default=None,
        ge=0,
        description="Pathogenic variant count for this gene from UniProt.",
    )

    @field_validator("chrom")
    @classmethod
    def _strip_chr_prefix(cls, v: str) -> str:
        """Accept 'chr1' / 'chrM' as well as '1' / 'MT' for user convenience."""
        v = v.strip()
        if v.lower().startswith("chr"):
            v = v[3:]
        if v == "M":
            v = "MT"
        return v

    @field_validator("ref", "alt")
    @classmethod
    def _allele_uppercase(cls, v: str) -> str:
        return v.upper()

    @model_validator(mode="after")
    def _derive_variant_id(self) -> VariantRequest:
        """Attach a canonical variant_id used by the feature pipeline."""
        object.__setattr__(
            self,
            "_variant_id",
            f"{self.chrom}:{self.pos}:{self.ref}:{self.alt}",
        )
        return self

    model_config = {"populate_by_name": True}


class BatchPredictRequest(BaseModel):
    """Up to MAX_BATCH_SIZE variants in a single request."""

    variants: list[VariantRequest] = Field(
        ...,
        min_length=1,
        max_length=MAX_BATCH_SIZE,
        description=f"List of variants to classify (max {MAX_BATCH_SIZE}).",
    )


# ---------------------------------------------------------------------------
# Output schemas
# ---------------------------------------------------------------------------

class VariantPrediction(BaseModel):
    """Per-variant prediction result."""

    variant_id: str = Field(description="chrom:pos:ref:alt")
    pathogenicity_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Ensemble probability of pathogenicity (0 = benign, 1 = pathogenic).",
    )
    classification: str = Field(
        description=(
            "Categorical call: 'Pathogenic', 'Likely pathogenic', "
            "'Uncertain significance', 'Likely benign', or 'Benign'."
        )
    )
    confidence: str = Field(
        description="'high' | 'medium' | 'low' — based on score distance from thresholds."
    )
    # Feature contributions — top-5 only, to keep payload small.
    top_features: Optional[dict[str, float]] = Field(
        default=None,
        description="SHAP values or model importances for the top-5 features.",
    )


class PredictResponse(BaseModel):
    """Response for /predict (single variant)."""

    prediction: VariantPrediction
    model_version: str
    pipeline_version: str


class BatchPredictResponse(BaseModel):
    """Response for /batch."""

    predictions: list[VariantPrediction]
    n_pathogenic: int
    n_benign: int
    n_uncertain: int
    model_version: str
    pipeline_version: str


class HealthResponse(BaseModel):
    status: str  # "ok" | "degraded"
    model_loaded: bool
    gnomad_index_loaded: bool
    gene_counts_loaded: bool
    uptime_seconds: float


class GeneSummaryResponse(BaseModel):
    gene_symbol: str
    n_pathogenic_in_gene: int
    gene_constraint_oe: Optional[float] = Field(
        default=None,
        description=(
            "gnomAD pLoF observed/expected ratio.  None = not available; "
            "engineer_features() defaults to 1.0 (unconstrained) when absent."
        ),
    )
    has_uniprot_annotation: int = Field(
        default=0,
        description="1 if the gene has any UniProt functional annotation.",
    )
    source: str = "ClinVar (training set)"


class InfoResponse(BaseModel):
    model_version: str
    pipeline_version: str
    training_auroc: float
    training_auprc: float
    holdout_auroc: float  # 0.9847 — gene-stratified, 154 K variants
    n_features: int
    feature_names: list[str]
    phase2_features_remaining: list[str]
    description: str


# ---------------------------------------------------------------------------
# Classification thresholds
# ---------------------------------------------------------------------------

# ACMGish five-tier mapping based on calibrated probability.
# Calibrate empirically against ClinVar gold-standard after deployment.
CLASSIFICATION_THRESHOLDS: dict[str, tuple[float, float]] = {
    "Pathogenic":             (0.90, 1.01),
    "Likely pathogenic":      (0.70, 0.90),
    "Uncertain significance": (0.30, 0.70),
    "Likely benign":          (0.10, 0.30),
    "Benign":                 (-0.01, 0.10),
}


def score_to_classification(score: float) -> tuple[str, str]:
    """Return (classification, confidence) for a raw probability score."""
    for label, (lo, hi) in CLASSIFICATION_THRESHOLDS.items():
        if lo < score <= hi:
            dist = min(score - lo, hi - score)
            if dist >= 0.15:
                confidence = "high"
            elif dist >= 0.05:
                confidence = "medium"
            else:
                confidence = "low"
            return label, confidence
    # Fallback — should not occur for valid scores in [0, 1]
    return "Uncertain significance", "low"
