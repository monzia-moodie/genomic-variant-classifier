#!/bin/bash
# =============================================================================
# GCP VM Startup Script — Run 6
# Runs automatically on boot. Clones repo, syncs data, launches training.
# =============================================================================
set -euo pipefail
exec > /var/log/genomic_run6.log 2>&1

echo "=== Startup: $(date) ==="

# --- Environment -------------------------------------------------------------
export HOME=/root
export REPO_DIR=/opt/genomic-variant-classifier
export DATA_DIR=$REPO_DIR/data
export GCS_BUCKET=gs://genomic-classifier-data
export GITHUB_REPO=https://github.com/monzia-moodie-repo-projects/genomic-variant-classifier.git
export EXPERIMENT_OUT=experiments/2026-04-05_run6

# --- System dependencies -----------------------------------------------------
apt-get update -qq
apt-get install -y --no-install-recommends \
    git python3-pip python3-venv build-essential \
    libgomp1 openjdk-17-jre-headless

# --- Clone repo --------------------------------------------------------------
git clone $GITHUB_REPO $REPO_DIR
cd $REPO_DIR

# --- Python environment ------------------------------------------------------
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

# --- Sync data from GCS (fast — same region, no egress cost) -----------------
echo "=== Syncing data from GCS: $(date) ==="
mkdir -p data/processed data/external/alphamissense data/external/spliceai \
         data/external/gnomad data/external/string data/raw/cache

gcloud storage cp $GCS_BUCKET/data/processed/clinvar_grch38.parquet      data/processed/
gcloud storage cp $GCS_BUCKET/data/processed/gnomad_v4_exomes.parquet    data/processed/
gcloud storage cp $GCS_BUCKET/data/external/alphamissense/AlphaMissense_hg38.tsv  data/external/alphamissense/
gcloud storage cp $GCS_BUCKET/data/external/spliceai/spliceai_scores.masked.snv.hg38.vcf.gz  data/external/spliceai/
gcloud storage cp $GCS_BUCKET/data/external/gnomad/gnomad.v4.1.constraint_metrics.tsv  data/external/gnomad/
gcloud storage cp $GCS_BUCKET/data/external/string/9606.protein.links.detailed.v12.0.txt.gz  data/external/string/
gcloud storage cp $GCS_BUCKET/data/external/string/9606.protein.info.v12.0.txt.gz  data/external/string/
gcloud storage cp $GCS_BUCKET/data/raw/cache/alphamissense_scores_hg38.parquet  data/raw/cache/
gcloud storage cp $GCS_BUCKET/data/raw/cache/string_graph_700.pkl         data/raw/cache/
gcloud storage cp $GCS_BUCKET/data/raw/cache/string_links.parquet         data/raw/cache/
gcloud storage cp $GCS_BUCKET/data/raw/cache/string_names.parquet         data/raw/cache/

echo "=== Data sync complete: $(date) ==="

# --- Install torch + torch_geometric for GNN ---------------------------------
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --quiet
pip install torch-geometric --quiet

# --- Launch Run 6 ------------------------------------------------------------
echo "=== Starting Run 6: $(date) ==="
python scripts/run_phase2_eval.py \
    --clinvar       data/processed/clinvar_grch38.parquet \
    --gnomad        data/processed/gnomad_v4_exomes.parquet \
    --spliceai      data/external/spliceai/spliceai_scores.masked.snv.hg38.vcf.gz \
    --alphamissense data/external/alphamissense/AlphaMissense_hg38.tsv \
    --gnomad-constraint data/external/gnomad/gnomad.v4.1.constraint_metrics.tsv \
    --string-db     700 \
    --skip-svm \
    --skip-nn \
    --n-folds       5 \
    --output        $EXPERIMENT_OUT \
    --auroc-target  0.990

echo "=== Run 6 complete: $(date) ==="

# --- Upload results to GCS ---------------------------------------------------
gcloud storage cp -r $EXPERIMENT_OUT gs://genomic-classifier-models/experiments/
echo "=== Results uploaded: $(date) ==="

# --- Shut down VM to stop billing --------------------------------------------
shutdown -h now

# Upload models to GCS on completion
gcloud storage cp -r models/v1/ gs://genomic-variant-prod-outputs/run7/models/
echo "Models uploaded to GCS"
