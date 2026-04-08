#!/bin/bash
# =============================================================================
# GCP VM Startup Script — Run 7
# =============================================================================
set -euo pipefail
exec > /var/log/genomic_run7.log 2>&1

echo "=== Startup: $(date) ==="

export HOME=/root
export REPO_DIR=/home/monzi/genomic-variant-classifier
export GCS_DATA=gs://genomic-variant-prod-outputs/run6/data/data
export GCS_MODELS=gs://genomic-variant-prod-outputs/run7/models

cd $REPO_DIR
git pull
source ~/venv/bin/activate

# --- Install torch-geometric (cu128 matching VM image) -----------------------
pip install torch-geometric torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.7.0+cu128.html --quiet
echo "=== torch-geometric installed: $(date) ==="

# --- Verify transformers installed (for ESM-2) -------------------------------
python -c "import transformers; print('transformers ok:', transformers.__version__)"

# --- Fix .gitkeep file-vs-directory collisions -------------------------------
for p in data/raw data/processed data/external; do
    if [ -f "$p" ]; then rm "$p" && mkdir -p "$p"; fi
done

# --- Sync data from GCS ------------------------------------------------------
echo "=== Syncing data: $(date) ==="
mkdir -p data/processed data/external/gnomad data/external/alphamissense \
         data/external/string data/raw/cache

gsutil -m cp "$GCS_DATA/processed/clinvar_grch38.parquet"           data/processed/
gsutil -m cp "$GCS_DATA/processed/gnomad_v4_exomes.parquet"         data/processed/
gsutil -m cp "$GCS_DATA/processed/gene_pathogenic_counts.parquet"   data/processed/
gsutil -m cp "$GCS_DATA/processed/gene_summary.parquet"             data/processed/
gsutil -m cp "$GCS_DATA/processed/dbsnp_index.parquet"              data/processed/
gsutil -m cp "$GCS_DATA/external/gnomad/gnomad.v4.1.constraint_metrics.tsv" \
             data/external/gnomad/
mkdir -p "data/external/alphamissense/AlphaMissense_hg38.tsv"
gsutil -m cp "$GCS_DATA/external/alphamissense/AlphaMissense_hg38.tsv/AlphaMissense_hg38.tsv" \
             "data/external/alphamissense/AlphaMissense_hg38.tsv/"
gsutil -m cp "$GCS_DATA/external/string/9606.protein.links.detailed.v12.0.txt.gz" \
             data/external/string/
gsutil -m cp "$GCS_DATA/external/string/9606.protein.info.v12.0.txt.gz" \
             data/external/string/
echo "=== Data sync complete: $(date) ==="

# --- Launch Run 7 ------------------------------------------------------------
echo "=== Starting Run 7: $(date) ==="
PYTHONPATH=$REPO_DIR nohup python scripts/run_phase2_eval.py \
    --clinvar       data/processed/clinvar_grch38.parquet \
    --gnomad        data/processed/gnomad_v4_exomes.parquet \
    --gnomad-constraint data/external/gnomad/gnomad.v4.1.constraint_metrics.tsv \
    --alphamissense "data/external/alphamissense/AlphaMissense_hg38.tsv/AlphaMissense_hg38.tsv" \
    --string-db     data/external/string/9606.protein.links.detailed.v12.0.txt.gz \
    --min-review-tier 2 \
    --skip-svm \
    --skip-nn \
    --n-folds       5 \
    --output        models/v1 \
    2>&1 | tee logs/training_run7.log

echo "=== Run 7 complete: $(date) ==="

# --- Upload models to GCS BEFORE shutdown ------------------------------------
echo "=== Uploading models: $(date) ==="
gcloud storage ls $GCS_MODELS || true
gsutil -m cp -r models/v1/ $GCS_MODELS/
gsutil -m cp logs/training_run7.log gs://genomic-variant-prod-outputs/run7/logs/
echo "=== Upload complete: $(date) ==="

# --- Shut down VM to stop billing --------------------------------------------
shutdown -h now
