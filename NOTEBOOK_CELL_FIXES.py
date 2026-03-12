# =============================================================================
# NOTEBOOK CELL CORRECTIONS  (Bugs 1, 2, R)
# =============================================================================
# Replace the first three cells of the notebook with the code below.
# These are the only cells that required direct inline edits.
# All other bugs were resolved by fixing the library modules themselves.
# =============================================================================


# ─── CELL 1 — Environment setup  (original had no bugs here) ─────────────────
# !pip install -q \
#     pyspark==3.5.1 \
#     torch==2.2.0 \
#     torch-geometric==2.4.0 \
#     xgboost==2.0.3 \
#     lightgbm==4.3.0 \
#     scikit-learn==1.4.0 \
#     pandas==2.1.4 \
#     pyarrow==15.0.0 \
#     requests==2.31.0 \
#     networkx==3.2.1 \
#     jinja2==3.1.3 \
#     matplotlib==3.8.2 \
#     seaborn==0.13.2 \
#     biopython==1.83 \
#     scipy==1.12.0 \
#     joblib==1.3.2 \
#     optuna==3.5.0


# ─── CELL 2 — Git clone  (Bug 2, R) ──────────────────────────────────────────
#
# ORIGINAL (broken):
#   from google.colab import userdata
#   token = userdata.get('GITHUB_TOKEN')
#   !git clone https://{token}@github.com/monzia-moodie/genomic-variant-classifier.git
#   !git config --global user.email "your-email@gmail.com"   ← hardcoded
#
# PROBLEMS:
#   Bug 2: Python f-string variable is not interpolated inside a Colab !-shell
#          command. The literal string {token} is passed to git, not the token value.
#   Issue R: Personal email address visible in committed history.
#
# FIXED:
#   - Build the URL in Python, then pass the variable to the shell command.
#   - Replace the hardcoded email with a placeholder comment.

from google.colab import userdata          # Bug 1: was missing in Cell 2 originally

token   = userdata.get("GITHUB_TOKEN")
repo    = "monzia-moodie/genomic-variant-classifier"
repo_url = f"https://{token}@github.com/{repo}.git"

import subprocess, os
result = subprocess.run(["git", "clone", repo_url], capture_output=True, text=True)
print(result.stdout or result.stderr)

# Configure git identity — fill in your email before committing
# (never hardcode personal email in a shared notebook)
# !git config --global user.email "YOUR_EMAIL@example.com"
# !git config --global user.name "Your Name"

os.chdir("genomic-variant-classifier")


# ─── CELL 3 — Secrets (Bug 1) ────────────────────────────────────────────────
#
# ORIGINAL (broken):
#   from google.colab import userdata   ← this line was missing from Cell 2
#   token = userdata.get('GITHUB_TOKEN')
#
# The `userdata` import was only in Cell 3 (after it was already needed in
# Cell 2), so running the cells in order raised NameError on Cell 2.
#
# FIX: `from google.colab import userdata` moved to the top of Cell 2 (above).
# Cell 3 now reads additional secrets that are only needed later.

NCBI_API_KEY = userdata.get("NCBI_API_KEY")     # for ClinVar FTP access
OMIM_API_KEY = userdata.get("OMIM_API_KEY")     # for OMIM gene-disease data
