"""
config.py
=========
Central configuration for the genomic variant classifier agent layer.
All paths, endpoints, and thresholds live here — no magic strings in agents.
"""

import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Root of the project — override via env var for GCP / Colab environments
PROJECT_ROOT = Path(os.getenv("GVC_PROJECT_ROOT", r"C:\Projects\genomic-variant-classifier"))

# Shared state JSON (acts as the integration backbone between agents)
# On Colab/GCP, point this at a Google Drive mount, e.g. /content/drive/MyDrive/gvc/
SHARED_STATE_PATH = Path(os.getenv(
    "GVC_SHARED_STATE_PATH",
    str(PROJECT_ROOT / "agent_layer" / "shared_state.json")
))

# Directory where agent audit logs are written
AUDIT_LOG_DIR = Path(os.getenv(
    "GVC_AUDIT_LOG_DIR",
    str(PROJECT_ROOT / "agent_layer" / "logs")
))

# Directory where raw downloaded source data lands before Spark ingestion
RAW_DATA_DIR = Path(os.getenv(
    "GVC_RAW_DATA_DIR",
    str(PROJECT_ROOT / "data" / "raw")
))

# Local corpus manifest (tracks what's been ingested into the training set)
CORPUS_MANIFEST_PATH = PROJECT_ROOT / "data" / "corpus_manifest.json"


# ---------------------------------------------------------------------------
# Data source endpoints
# ---------------------------------------------------------------------------

CLINVAR_FTP_ROOT       = "ftp.ncbi.nlm.nih.gov"
CLINVAR_FTP_VCF_DIR    = "/pub/clinvar/vcf_GRCh38/"
CLINVAR_RELEASE_FILE   = "ClinVar.vcf.gz"                  # weekly build
CLINVAR_SUMMARY_URL    = (
    "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/"
    "variant_summary.txt.gz"
)

GNOMAD_API_BASE        = "https://gnomad.broadinstitute.org/api"
GNOMAD_DATASET_LATEST  = "gnomad_r4"                       # update when v5 ships

LOVD_API_BASE          = "https://databases.lovd.nl/shared"
LOVD_VARIANTS_ENDPOINT = "/variants"
LOVD_GENES_OF_INTEREST = [
    # All 10 genes confirmed available via LOVD shared database
    "BRCA1", "BRCA2", "MLH1", "MSH2", "MSH6",
    "APC", "NF1", "TP53", "PTEN", "RB1",
]

ALPHAMISSENSE_MANIFEST  = (
    "https://storage.googleapis.com/dm_alphamissense/"
    "AlphaMissense_hg38.tsv.gz"
)


# ---------------------------------------------------------------------------
# GCP / Spark
# ---------------------------------------------------------------------------

GCP_PROJECT_ID         = os.getenv("GCP_PROJECT_ID", "your-gcp-project-id")
GCP_REGION             = os.getenv("GCP_REGION", "us-central1")
DATAPROC_CLUSTER_NAME  = os.getenv("DATAPROC_CLUSTER", "gvc-spark-cluster")
DATAPROC_BUCKET        = os.getenv("DATAPROC_BUCKET", "gs://your-bucket/gvc")
SPARK_INGEST_JOB_PATH  = f"{DATAPROC_BUCKET}/jobs/vcf_ingest.py"


# ---------------------------------------------------------------------------
# Drift detection thresholds
# ---------------------------------------------------------------------------

# Jensen-Shannon divergence threshold above which drift is flagged
DRIFT_JS_THRESHOLD     = 0.05

# Fraction of variants with changed classification to trigger forced retraining
RECLASSIFICATION_RATE_THRESHOLD = 0.01   # 1 %

# Minimum new variants before considering a Spark re-run worthwhile
MIN_NEW_VARIANTS_FOR_INGEST = 500


# ---------------------------------------------------------------------------
# Model paths and checkpointing
# ---------------------------------------------------------------------------

MODELS_DIR             = PROJECT_ROOT / "models"
CHECKPOINT_DIR         = MODELS_DIR / "checkpoints"

# Subdirectory names inside CHECKPOINT_DIR
RESNET_SUBDIR          = "resnet50"
ENSEMBLE_SUBDIR        = "ensemble"

# GCS checkpoint prefix (mirrors local structure in the cloud)
GCS_CHECKPOINT_PREFIX  = os.getenv("GVC_GCS_CHECKPOINTS", "gs://your-bucket/gvc/checkpoints")

# Google Drive mount root (Colab: /content/drive/MyDrive/gvc)
GDRIVE_CHECKPOINT_DIR  = Path(os.getenv(
    "GVC_GDRIVE_CHECKPOINTS",
    "/content/drive/MyDrive/gvc/checkpoints",
))

# Training data (parquet files produced by Spark ingest)
PROCESSED_DATA_DIR     = PROJECT_ROOT / "data" / "processed"
TRAIN_PARQUET          = PROCESSED_DATA_DIR / "train.parquet"
REPLAY_BUFFER_PARQUET  = PROCESSED_DATA_DIR / "replay_buffer.parquet"

# Validation split written by train.py PHASE 6 — used by InterpretabilityAgent.
# Mirrors out_dir from train.py (default: models/v1/).
# Override via env var if you pass a custom --out-dir to train.py.
VAL_PARQUET            = Path(os.getenv(
    "GVC_VAL_PARQUET",
    str(MODELS_DIR / "v1" / "val.parquet")
))


# ---------------------------------------------------------------------------
# EWC (Elastic Weight Consolidation) — ResNet-50 branch
# ---------------------------------------------------------------------------

# JS divergence bands that select the update strategy:
#   score < EWC_DRIFT_LOW             → no update needed
#   EWC_DRIFT_LOW  <= score < EWC_DRIFT_HIGH  → EWC fine-tune
#   score >= EWC_DRIFT_HIGH           → queue full retrain for human review
EWC_DRIFT_LOW          = float(os.getenv("GVC_EWC_DRIFT_LOW",  "0.02"))
EWC_DRIFT_HIGH         = float(os.getenv("GVC_EWC_DRIFT_HIGH", "0.15"))

# λ — weight on the EWC penalty term (higher = stronger protection of old knowledge)
EWC_LAMBDA             = float(os.getenv("GVC_EWC_LAMBDA", "400.0"))

# Number of samples used to estimate the Fisher diagonal
EWC_FISHER_SAMPLES     = int(os.getenv("GVC_EWC_FISHER_SAMPLES", "1000"))

# ResNet fine-tuning hyper-parameters
EWC_LR                 = float(os.getenv("GVC_EWC_LR",     "1e-4"))
EWC_EPOCHS             = int(os.getenv("GVC_EWC_EPOCHS",   "5"))
EWC_BATCH_SIZE         = int(os.getenv("GVC_EWC_BATCH",    "32"))

# Number of ResNet output classes (Pathogenic / LP / VUS / LB / Benign)
RESNET_NUM_CLASSES     = int(os.getenv("GVC_RESNET_CLASSES", "5"))


# ---------------------------------------------------------------------------
# Ensemble continual learning — XGBoost / LightGBM
# ---------------------------------------------------------------------------

# Size of the memory replay buffer (old samples mixed into each update round)
REPLAY_BUFFER_SIZE     = int(os.getenv("GVC_REPLAY_BUFFER", "5000"))

# Fraction of the replay buffer to sample on each ensemble update
REPLAY_SAMPLE_FRAC     = float(os.getenv("GVC_REPLAY_FRAC", "0.3"))

# XGBoost / LightGBM continued-training rounds per update
ENSEMBLE_BOOST_ROUNDS  = int(os.getenv("GVC_BOOST_ROUNDS", "50"))


# ---------------------------------------------------------------------------
# Interpretability (SHAP + GradCAM)
# ---------------------------------------------------------------------------

SHAP_REPORT_DIR        = PROJECT_ROOT / "reports" / "shap"

# Number of validation samples fed to SHAP (TreeExplainer is exact; cap for speed)
SHAP_VAL_SAMPLES       = int(os.getenv("GVC_SHAP_VAL_SAMPLES", "2000"))

# Top-K features shown in the summary report and used for stability tracking
SHAP_TOP_K             = int(os.getenv("GVC_SHAP_TOP_K", "25"))

# Spearman rank-correlation threshold for the top-K importance ranking.
# If the correlation between current and previous run drops below this,
# the agent flags an instability anomaly.
SHAP_STABILITY_THRESHOLD = float(os.getenv("GVC_SHAP_STABILITY", "0.80"))

# Fractional change in mean |SHAP| for a single feature that triggers an alert.
# E.g. 0.5 = flag if a feature's importance changed by >50% since the last run.
SHAP_IMPORTANCE_DELTA  = float(os.getenv("GVC_SHAP_DELTA", "0.50"))

# Features expected to dominate importance for pathogenicity prediction.
# Any feature NOT in this list appearing in the top-K is flagged for review.
# Verified against actual model output — update after each major feature
# engineering change.
EXPECTED_HIGH_IMPORTANCE_FEATURES: list[str] = [
    # --- Tier 1: confirmed top features from production model ---
    # Gene-level clinical prior (was #1 feature; nearly lost to Pydantic silent drop)
    "n_pathogenic_in_gene",
    "n_benign_in_gene",
    "pathogenic_rate_in_gene",

    # AlphaMissense (top functional impact predictor)
    "alphamissense_score",
    "alphamissense_class",

    # gnomAD allele frequencies (population evidence)
    "gnomad_af",
    "gnomad_af_popmax",
    "gnomad_af_nfe",
    "gnomad_af_afr",
    "gnomad_af_sas",
    "gnomad_af_eas",
    "gnomad_nhomalt",

    # SpliceAI delta scores (splice impact — critical for splice-region variants)
    "spliceai_ds_ag",   # delta score acceptor gain
    "spliceai_ds_al",   # delta score acceptor loss
    "spliceai_ds_dg",   # delta score donor gain
    "spliceai_ds_dl",   # delta score donor loss
    "spliceai_max_ds",  # max across all four (engineered feature)

    # PhyloP / conservation
    "phylop100way_vertebrate",
    "phylop17way_primate",
    "phastcons100way_vertebrate",
    "phastcons17way_primate",
    "gerp_rs",

    # --- Tier 2: strong secondary features ---
    # CADD
    "cadd_phred",
    "cadd_raw",

    # REVEL / other meta-scores
    "revel_score",
    "sift_score",
    "polyphen2_hdiv_score",
    "polyphen2_hvar_score",
    "mutationtaster_score",
    "fathmm_score",
    "provean_score",

    # ClinVar evidence strength
    "clinvar_review_status_score",
    "clinvar_gold_stars",
    "clinvar_conflicted",

    # 1KGP population AF (from KGPConnector — 5 new features in latest commit)
    "kgp_af",
    "kgp_af_afr",
    "kgp_af_amr",
    "kgp_af_eas",
    "kgp_af_eur",

    # Variant type / consequence
    "is_lof",
    "is_missense",
    "is_splice_region",
    "is_synonymous",
    "consequence_severity",

    # ESM-2 sequence embedding (transformer-derived)
    "esm2_embedding_norm",
    "esm2_pathogenicity_score",

    # Gene-level annotation
    "gene_constraint_lof_oe",   # gnomAD pLI / LOEUF
    "gene_constraint_mis_oe",

    # LOVD classification (where available: RB1, TP53, PTEN)
    "lovd_variant_class",

    # GTEx expression context
    "gtex_tissue_expression_max",
    "gtex_tissue_expression_mean",
]

# UniProt REST base URL for biological plausibility lookups
UNIPROT_API_BASE       = "https://rest.uniprot.org/uniprotkb"

# GTEx portal API (v8 expression endpoint)
GTEX_API_BASE          = "https://gtexportal.org/api/v2"


# ---------------------------------------------------------------------------
# Human-in-the-loop
# ---------------------------------------------------------------------------

# If True, all state-mutating actions require a human confirmation prompt
REQUIRE_HUMAN_APPROVAL = os.getenv("GVC_REQUIRE_HUMAN_APPROVAL", "true").lower() == "true"

# Email to notify on pending review items (optional — leave None to skip)
REVIEW_NOTIFY_EMAIL    = os.getenv("GVC_REVIEW_EMAIL", None)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_LEVEL = os.getenv("GVC_LOG_LEVEL", "INFO")


# ---------------------------------------------------------------------------
# Literature Scout
# ---------------------------------------------------------------------------

LITERATURE_DIGEST_DIR  = PROJECT_ROOT / "reports" / "literature"

# NCBI E-utilities (no API key = 3 req/s; set NCBI_API_KEY for 10 req/s)
NCBI_EUTILS_BASE       = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NCBI_API_KEY           = os.getenv("NCBI_API_KEY", None)

# bioRxiv RSS feeds (subject areas most relevant to the project)
BIORXIV_RSS_FEEDS: list[str] = [
    "https://connect.biorxiv.org/biorxiv_xml.php?subject=genomics",
    "https://connect.biorxiv.org/biorxiv_xml.php?subject=bioinformatics",
]

# ClinGen Evidence Repository
CLINGEN_API_BASE       = "https://search.clinicalgenome.org/kb/gene-validity"

# PubMed search queries run on each scout cycle.
LITERATURE_PUBMED_QUERIES: list[tuple[str, int]] = [
    ("variant pathogenicity prediction deep learning",                    15),
    ("missense variant functional effect prediction",                     12),
    ("genome variant clinical significance classification machine learning", 10),
    ("protein language model variant effect",                             10),
    ("DNA language model genomic variant",                                 8),
    ("continual learning genomics",                                        6),
    ("federated learning variant classification",                          6),
    ("BRCA1 BRCA2 variant classification pathogenicity 2024",              8),
    ("TP53 MLH1 MSH2 variant interpretation 2024",                         6),
    ("tumor histopathology genomic variant multimodal",                    8),
    ("whole slide image genomic feature prediction",                       6),
]

LITERATURE_MAX_PAPERS_PER_RUN  = int(os.getenv("GVC_LIT_MAX_PAPERS", "80"))
LITERATURE_MIN_RELEVANCE       = float(os.getenv("GVC_LIT_MIN_RELEVANCE", "0.35"))
LITERATURE_CANDIDATE_MIN_SCORE = float(os.getenv("GVC_LIT_CAND_SCORE", "0.55"))

LITERATURE_KNOWN_TOOLS: set[str] = {
    "cadd", "revel", "alphamissense", "sift", "polyphen", "polyphen2",
    "polyphen-2", "mutationtaster", "fathmm", "provean", "vest",
    "clinpred", "primateai", "esm", "esm-2", "esm2", "dnabert", "dnabert-2",
    "gnomad", "clinvar", "lovd", "gtex", "uniprot", "gerp", "phylop",
    "phastcons", "siphy", "shap", "xgboost", "lightgbm", "resnet",
}

LITERATURE_RELEVANCE_KEYWORDS: list[str] = [
    "pathogenicity", "variant classification", "clinical significance",
    "missense", "loss of function", "splice", "frameshift",
    "feature importance", "predictive score", "functional score",
    "deep learning", "transformer", "language model", "embedding",
    "benchmark", "precision", "recall", "auc", "roc",
    "acmg", "variant curation", "evidence weight",
    "tumor", "histopathology", "whole slide", "tcga",
    "continual learning", "catastrophic forgetting", "elastic weight",
    "federated", "multimodal", "multi-modal",
]

LITERATURE_FEATURE_PATTERNS: list[str] = [
    r"we (?:propose|present|introduce|develop) (?P<name>[A-Z][A-Za-z0-9_\-]{2,30})",
    r"(?P<name>[A-Z][A-Za-z0-9_\-]{2,30}) score\b",
    r"(?P<name>[A-Z][A-Za-z0-9_\-]{2,30}) (?:metric|index|model|predictor)\b",
    r"novel (?:feature|score|metric|predictor)[:\s]+(?P<name>[A-Z][A-Za-z0-9_\-]{2,30})",
    r"(?P<name>[A-Z][A-Za-z0-9_\-]{2,30})-score\b",
]