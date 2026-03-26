# ============================================================================
# Genomic Variant Classifier — multi-stage Dockerfile
# ============================================================================
#
# Stage 1: builder
#   - Installs all dependencies (including heavy training deps)
#   - Produces a virtualenv at /opt/venv
#
# Stage 2: api  (default target)
#   - Copies only the inference virtualenv and src/api + src/models
#   - No PySpark, no TensorFlow unless NNs are included
#   - ~1.2 GB final image
#
# Stage 3: trainer  (optional, build with --target trainer)
#   - Includes PySpark, XGBoost, LightGBM, and all data connectors
#   - Used for periodic re-training jobs; not deployed as a service
#
# Usage
# -----
#   # Build the API image (default)
#   docker build -t genomic-variant-api:phase2-v1 .
#
#   # Build the trainer image
#   docker build --target trainer -t genomic-variant-trainer:phase2-v1 .
#
#   # Run the API locally
#   docker run -p 8000:8000 \
#       -e MODEL_PATH=/app/models/phase2_pipeline.joblib \
#       -v $(pwd)/models:/app/models:ro \
#       genomic-variant-api:phase2-v1
#
# Environment variables (API container)
# --------------------------------------
#   MODEL_PATH           Path to serialised InferencePipeline joblib
#                        (default: /app/models/phase2_pipeline.joblib)
#   GNOMAD_INDEX_PATH    Optional gnomAD parquet for live AF lookup
#   LOG_LEVEL            INFO (default) | DEBUG | WARNING
#   WORKERS              Number of gunicorn workers (default: 2)
# ============================================================================

ARG PYTHON_VERSION=3.11
ARG DEBIAN_FRONTEND=noninteractive


# ----------------------------------------------------------------------------
# Stage 1 — dependency builder
# ----------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder

WORKDIR /build

# System build tools (removed after this stage)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# Create an isolated virtualenv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt requirements-api.txt* ./

# Install API-only runtime dependencies
# requirements-api.txt should be a subset of requirements.txt containing:
#   fastapi, uvicorn[standard], gunicorn, pydantic>=2, joblib,
#   numpy, pandas, scikit-learn, lightgbm, xgboost, pyarrow
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements-api.txt


# ----------------------------------------------------------------------------
# Stage 2 — API runtime image  (default)
# ----------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS api

LABEL maintainer="monzia-moodie" \
      description="Genomic Variant Pathogenicity API — Phase 2 (AUROC 0.9847)" \
      org.opencontainers.image.source="https://github.com/monzia-moodie/genomic-variant-classifier"

# Copy virtualenv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Application source — only what inference needs
COPY src/api/          src/api/
COPY src/models/       src/models/
COPY src/utils/        src/utils/
COPY src/__init__.py   src/__init__.py

# Model artefact placeholder — override at runtime via bind-mount or COPY
RUN mkdir -p models

# Non-root user for security
RUN adduser --disabled-password --gecos "" appuser \
    && chown -R appuser:appuser /app
USER appuser

# Default environment
ENV MODEL_PATH=/app/models/phase2_pipeline.joblib \
    GENE_SUMMARY_PATH=/app/data/processed/gene_summary.parquet \
    LOG_LEVEL=INFO \
    WORKERS=2 \
    PORT=8000

EXPOSE ${PORT}

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c \
        "import urllib.request, sys; \
         r = urllib.request.urlopen('http://localhost:${PORT}/health', timeout=4); \
         sys.exit(0 if r.status == 200 else 1)"

# gunicorn with uvicorn workers for async FastAPI
CMD ["sh", "-c", \
     "gunicorn src.api.main:app \
        -k uvicorn.workers.UvicornWorker \
        --bind 0.0.0.0:${PORT} \
        --workers ${WORKERS} \
        --timeout 120 \
        --log-level ${LOG_LEVEL}"]


# ----------------------------------------------------------------------------
# Stage 3 — trainer  (build with --target trainer)
# ----------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS trainer

LABEL description="Genomic Variant Classifier — training image (Phase 2)"

RUN apt-get update && apt-get install -y --no-install-recommends \
        openjdk-17-jre-headless \
        procps \
    && rm -rf /var/lib/apt/lists/*

# Re-use builder venv but install full training stack on top
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    JAVA_HOME=/usr/lib/jvm/default-java

WORKDIR /app

# Full source tree for training
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# Train with the canonical Phase 2 config
CMD ["python", "scripts/run_phase2_eval.py", \
     "--clinvar",       "data/processed/clinvar_grch38.parquet", \
     "--alphamissense", "data/external/alphamissense/AlphaMissense_hg38.tsv.gz", \
     "--gnomad",        "data/processed/gnomad_v4_exomes.parquet", \
     "--skip-nn", "--skip-svm", \
     "--min-review-tier", "2", \
     "--output",        "outputs/latest"]
