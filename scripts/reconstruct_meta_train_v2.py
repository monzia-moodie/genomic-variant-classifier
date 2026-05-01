"""Reconstruct meta_train.parquet by mirroring _gene_aware_split.

Algorithm (matches src/data/real_data_prep.py::_gene_aware_split exactly):
  1. Take labeled cohort (P/LP/B/LB rows)
  2. Compute groups = gene_symbol.fillna("unknown")
  3. Train rows = rows whose gene is NOT in (val_genes ∪ test_genes)

Verifies against meta_val + meta_test by checking gene-set disjointness
and row count consistency.

Cost: ~1 minute wall-clock (reads ~135 MB ClinVar parquet + 2 split parquets).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PATHOGENIC_TERMS = {
    "Pathogenic",
    "Likely pathogenic",
    "Pathogenic/Likely pathogenic",
}
BENIGN_TERMS = {
    "Benign",
    "Likely benign",
    "Benign/Likely benign",
}


def filter_to_labeled_cohort(clinvar: pd.DataFrame) -> pd.DataFrame:
    """Apply DataPrepPipeline._load_and_label filtering."""
    df = clinvar.copy()
    df["clinical_sig"] = df["clinical_sig"].fillna("").astype(str).str.strip()
    
    n_before = len(df)
    
    # Apply label filter (matches lines 287-294 of real_data_prep.py)
    import numpy as np
    df["label"] = np.nan
    df.loc[df["clinical_sig"].isin(PATHOGENIC_TERMS), "label"] = 1
    df.loc[df["clinical_sig"].isin(BENIGN_TERMS), "label"] = 0
    df = df[df["label"].notna()].copy()
    df["label"] = df["label"].astype(int)
    
    print(f"  label filter: {n_before:,} -> {len(df):,} rows ({n_before - len(df):,} VUS/conflicting removed)")
    
    # exclude_conflicting (line 311-314)
    n_before2 = len(df)
    df = df[~df["clinical_sig"].str.contains("onflict", na=False)]
    if len(df) < n_before2:
        print(f"  conflicting filter: removed {n_before2 - len(df):,} rows")
    
    return df.reset_index(drop=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--splits-dir", required=True, type=Path)
    p.add_argument("--clinvar", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--expected-train", type=int, default=1197216)
    p.add_argument("--expected-val", type=int, default=154404)
    p.add_argument("--expected-test", type=int, default=349067)
    args = p.parse_args()

    print(f"=== reconstruct_meta_train v2 (gene-set complement) ===")
    print(f"  splits-dir: {args.splits_dir}")
    print(f"  clinvar:    {args.clinvar}")
    print(f"  output:     {args.output}")
    print()

    # Load splits
    meta_val = pd.read_parquet(args.splits_dir / "meta_val.parquet")
    meta_test = pd.read_parquet(args.splits_dir / "meta_test.parquet")
    print(f"  meta_val:  {len(meta_val):>10,} rows ({meta_val['gene_symbol'].nunique():,} unique gene strings)")
    print(f"  meta_test: {len(meta_test):>10,} rows ({meta_test['gene_symbol'].nunique():,} unique gene strings)")

    if len(meta_val) != args.expected_val:
        print(f"  WARN: meta_val rows {len(meta_val)} != expected {args.expected_val}", file=sys.stderr)
    if len(meta_test) != args.expected_test:
        print(f"  WARN: meta_test rows {len(meta_test)} != expected {args.expected_test}", file=sys.stderr)

    # Build val ∪ test gene set (mirroring groups.fillna("unknown"))
    val_genes = set(meta_val["gene_symbol"].fillna("unknown").astype(str).unique())
    test_genes = set(meta_test["gene_symbol"].fillna("unknown").astype(str).unique())
    
    overlap_genes = val_genes & test_genes
    print(f"  val genes:  {len(val_genes):,}")
    print(f"  test genes: {len(test_genes):,}")
    print(f"  val ∩ test: {len(overlap_genes):,} (should be 0 — gene-aware split)")
    
    if overlap_genes:
        print(f"  WARN: {len(overlap_genes)} genes appear in BOTH val and test:", file=sys.stderr)
        for g in list(overlap_genes)[:5]:
            print(f"    - {g[:80]}{'...' if len(g) > 80 else ''}", file=sys.stderr)

    val_test_genes = val_genes | test_genes
    print(f"  val ∪ test: {len(val_test_genes):,}")
    print()

    # Load + filter ClinVar
    print(f"  loading ClinVar ({args.clinvar.stat().st_size / 1024 / 1024:.1f} MB)...")
    clinvar = pd.read_parquet(args.clinvar)
    print(f"  full ClinVar: {len(clinvar):,} rows")

    labeled = filter_to_labeled_cohort(clinvar)
    expected_total = args.expected_train + args.expected_val + args.expected_test
    print(f"\n  labeled cohort: {len(labeled):,} rows")
    print(f"  expected total: {expected_total:,}")
    print(f"  diff:           {len(labeled) - expected_total:+,}")

    # Reconstruct: train = rows whose gene is NOT in (val ∪ test)
    labeled_genes = labeled["gene_symbol"].fillna("unknown").astype(str)
    is_train = ~labeled_genes.isin(val_test_genes)
    meta_train = labeled[is_train].reset_index(drop=True)
    
    print(f"\n  meta_train: {len(meta_train):>10,} rows reconstructed")
    print(f"             expected: {args.expected_train:>10,}")
    print(f"             diff:     {len(meta_train) - args.expected_train:>+10,}")

    if len(meta_train) != args.expected_train:
        print(f"\n  NOTE: row count mismatch. Investigating...", file=sys.stderr)
        train_genes_count = labeled_genes[is_train].nunique()
        all_labeled_genes = labeled_genes.nunique()
        print(f"  train unique genes: {train_genes_count:,}", file=sys.stderr)
        print(f"  labeled unique genes: {all_labeled_genes:,}", file=sys.stderr)
        print(f"  unaccounted genes: {all_labeled_genes - train_genes_count - len(val_test_genes):,}", file=sys.stderr)
        # Continue anyway — small drift is acceptable, large is alarming
        if abs(len(meta_train) - args.expected_train) > args.expected_train * 0.01:
            print(f"\nFAIL: train row count diff exceeds 1% of expected. Refusing to write.", file=sys.stderr)
            return 2

    # Verify gene_symbol present
    if "gene_symbol" not in meta_train.columns:
        print(f"FAIL: meta_train missing gene_symbol column", file=sys.stderr)
        return 3

    n_genes = meta_train["gene_symbol"].nunique()
    n_missing = meta_train["gene_symbol"].isna().sum()
    print(f"\n  meta_train gene_symbol: {n_genes:,} unique, {n_missing:,} NaN")

    # Verify gene disjointness with val+test
    train_gene_set = set(meta_train["gene_symbol"].fillna("unknown").astype(str).unique())
    leak_train_val = train_gene_set & val_genes
    leak_train_test = train_gene_set & test_genes
    print(f"  train ∩ val:  {len(leak_train_val):,} (should be 0)")
    print(f"  train ∩ test: {len(leak_train_test):,} (should be 0)")
    
    if leak_train_val or leak_train_test:
        print(f"FAIL: gene leakage detected", file=sys.stderr)
        return 4

    # Persist
    args.output.parent.mkdir(parents=True, exist_ok=True)
    meta_train.to_parquet(args.output, index=False)
    size_mb = args.output.stat().st_size / 1024 / 1024
    print(f"\nOK: wrote {args.output} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
