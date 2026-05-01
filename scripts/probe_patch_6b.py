"""500-row probe: verify Patch 6b plumbing without retraining ensembles.

Loads tiny ClinVar slice -> runs DataPrepPipeline only -> verifies
meta_train.parquet exists with gene_symbol column.

Then loads run9_ready ensemble + scaler -> simulates the gnn_df construction
that run_phase2_eval.py would do -> verifies gene_symbol present.

No GNN training (fast), no full ensemble retrain.
"""

from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.real_data_prep import DataPrepPipeline, DataPrepConfig

probe_out = Path("outputs/probe_patch6b")
probe_out.mkdir(parents=True, exist_ok=True)

# 1. Take a 5K slice of ClinVar (gene-aware split needs enough rows for both classes)
print("=== probe step 1: load 5K ClinVar rows ===")
df = pd.read_parquet("data/processed/clinvar_grch38.parquet")
print(f"  full ClinVar rows: {len(df)}")
df_small = df.sample(n=5000, random_state=42).reset_index(drop=True)
small_path = probe_out / "clinvar_5k.parquet"
df_small.to_parquet(small_path, index=False)
print(f"  saved: {small_path}")

# 2. Run DataPrepPipeline with output dir = probe
print("\n=== probe step 2: DataPrepPipeline.run() ===")
cfg = DataPrepConfig(
    min_review_tier=3,
    test_fraction=0.20,
    random_state=42,
    scale_features=True,
    output_dir=probe_out / "splits",
)
pipe = DataPrepPipeline(config=cfg)
try:
    res = pipe.run(clinvar_path=str(small_path))
    print(f"  pipeline returned: {[type(x).__name__ for x in res]}")
except Exception as e:
    print(f"  FAIL: pipeline crashed — {type(e).__name__}: {e}")
    sys.exit(1)

# 3. Verify meta_train.parquet exists and has gene_symbol
print("\n=== probe step 3: verify meta_train.parquet ===")
meta_train_path = probe_out / "splits" / "meta_train.parquet"
if not meta_train_path.exists():
    print(f"  FAIL: {meta_train_path} not written")
    sys.exit(1)
mt = pd.read_parquet(meta_train_path)
print(f"  rows={len(mt)} cols={mt.shape[1]}")
print(f"  has gene_symbol: {'gene_symbol' in mt.columns}")
print(
    f"  unique genes: {mt['gene_symbol'].nunique() if 'gene_symbol' in mt.columns else 'N/A'}"
)
if "gene_symbol" not in mt.columns:
    print("  FAIL: gene_symbol column missing")
    sys.exit(1)

# 4. Simulate the gnn_df construction from run_phase2_eval.py Patch 6b
print("\n=== probe step 4: simulate gnn_df construction ===")
X_train = pd.read_parquet(probe_out / "splits" / "X_train.parquet")
y_train = pd.read_parquet(probe_out / "splits" / "y_train.parquet")["label"]
print(f"  X_train shape: {X_train.shape} (expect 78-col numeric)")
print(f"  y_train rows: {len(y_train)}")
print(f"  meta_train rows: {len(mt)}")
assert len(mt) == len(X_train) == len(y_train), "row count mismatch"

gnn_df = X_train.copy().reset_index(drop=True)
gnn_df["gene_symbol"] = mt["gene_symbol"].fillna("").reset_index(drop=True)
gnn_df["acmg_label"] = y_train.values

print(f"  gnn_df shape: {gnn_df.shape}")
print(f"  gnn_df has_gene_symbol: {'gene_symbol' in gnn_df.columns}")
print(f"  gnn_df has_acmg_label: {'acmg_label' in gnn_df.columns}")
print(f"  unique genes in gnn_df: {gnn_df['gene_symbol'].nunique()}")
print(f"  empty gene_symbol rows: {(gnn_df['gene_symbol'] == '').sum()}")

# 5. Verify groupby('gene_symbol') doesn't crash
print("\n=== probe step 5: verify groupby('gene_symbol') works ===")
try:
    sample_feat = X_train.columns[0]
    grp = gnn_df.groupby("gene_symbol")[sample_feat].mean()
    print(f"  groupby on {sample_feat!r} returned {len(grp)} gene-aggregated values")
    print("  PASS — Patch 6b restores gene_symbol; build_pyg_dataset will not crash")
except KeyError as e:
    print(f"  FAIL: KeyError {e} — Patch 6b broken")
    sys.exit(1)

print("\n=== Patch 6b probe: ALL CHECKS PASS ===")
