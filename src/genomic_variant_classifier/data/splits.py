"""
src/data/splits.py
===================
Deterministic train/val/test splitters for the variant-pathogenicity pipeline.

Two public functions:

    gene_stratified_split(df, test_frac, val_frac, seed)
        -> (train_idx, val_idx, test_idx)
        GroupShuffleSplit by gene_symbol. No gene appears in more than one
        set. This is the Run 8 split; use for direct-comparison runs.

    unseen_gene_holdout_split(df, holdout_frac, seed)
        -> (train_idx, unseen_gene_test_idx)
        Holds out a deterministic fraction of GENES by hashing the gene
        symbol. The same genes are always in the holdout set for the same
        holdout_frac, enabling longitudinal comparison across runs.

Why hash-stable gene selection (for unseen-gene holdout):
    Using `seed` to sample genes makes the holdout depend on both the data
    rows and the sampling seed. If the data grows between runs (new ClinVar
    release), the same seed gives different genes. Hashing gene_symbol mod
    K buckets means holdout_frac=0.2 always selects the same ~20 % of
    genes regardless of dataset size, so Run 9's unseen-gene result is
    directly comparable to Run 15's.

Invariants enforced by unit tests:
    - No variant appears in more than one split.
    - No gene appears in more than one split.
    - Fractional sizes within 1 % of requested (respecting gene-group sizes).
    - unseen_gene_holdout_split is idempotent: same df -> same indices.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gene-stratified (Run 8 compatible)
# ---------------------------------------------------------------------------
def gene_stratified_split(
    df: pd.DataFrame,
    test_frac: float = 0.2,
    val_frac: float = 0.2,
    seed: int = 42,
    gene_col: str = "gene_symbol",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Three-way split such that no gene appears in more than one set.

    Args:
        df:         variant dataframe, must contain `gene_col`.
        test_frac:  fraction of UNIQUE genes in the test set.
        val_frac:   fraction of UNIQUE genes in the val set. Applied after
                    test split, so effective train fraction is
                    (1 - test_frac) * (1 - val_frac).
        seed:       random seed for gene shuffling.
        gene_col:   column name carrying the gene symbol.

    Returns:
        (train_idx, val_idx, test_idx) as np.ndarrays of integer positions.

    Raises:
        ValueError: if `gene_col` is not present or has all-null values.
    """
    if gene_col not in df.columns:
        raise ValueError(f"{gene_col!r} not in df.columns")
    if not (0.0 < test_frac < 1.0 and 0.0 <= val_frac < 1.0):
        raise ValueError("test_frac in (0,1), val_frac in [0,1)")

    rng = np.random.default_rng(seed)

    # Fill missing gene symbols with a sentinel so they don't all collapse
    # into a single NaN group (which would dump the bulk of unknown-gene
    # variants into one split).
    genes = df[gene_col].fillna("__UNKNOWN__").astype(str).values

    unique_genes = np.array(sorted(set(genes)))
    rng.shuffle(unique_genes)

    n = len(unique_genes)
    n_test = max(1, int(round(n * test_frac)))
    n_val = max(0, int(round(n * val_frac)))
    test_genes = set(unique_genes[:n_test])
    val_genes = set(unique_genes[n_test : n_test + n_val])
    train_genes = set(unique_genes[n_test + n_val :])

    test_idx = np.where(np.isin(genes, list(test_genes)))[0]
    val_idx = np.where(np.isin(genes, list(val_genes)))[0]
    train_idx = np.where(np.isin(genes, list(train_genes)))[0]

    logger.info(
        "gene_stratified_split: %d genes -> train=%d val=%d test=%d "
        "(variants: %d/%d/%d)",
        n,
        len(train_genes),
        len(val_genes),
        len(test_genes),
        len(train_idx),
        len(val_idx),
        len(test_idx),
    )
    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# Unseen-gene holdout (hash-stable)
# ---------------------------------------------------------------------------
def unseen_gene_holdout_split(
    df: pd.DataFrame,
    holdout_frac: float = 0.2,
    seed: int = 42,
    gene_col: str = "gene_symbol",
    n_buckets: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Hash-stable unseen-gene holdout.

    Each gene's hash (SHA-256 of `seed:gene_symbol`) mod n_buckets determines
    its bucket; buckets below `holdout_frac * n_buckets` are held out. The
    same (seed, gene_symbol) → same bucket → same holdout membership across
    runs and dataset versions.

    Args:
        df:            variant dataframe.
        holdout_frac:  target fraction of UNIQUE genes in the holdout set.
        seed:          determines the hash namespace. Change only if you
                       want a new holdout partition.
        gene_col:      column name carrying the gene symbol.
        n_buckets:     granularity. Higher -> closer to target holdout_frac,
                       more expensive hashing (default 100 is plenty).

    Returns:
        (train_idx, holdout_idx). Note: no val set — use
        `gene_stratified_split` on the train portion if val is needed.

    Why SHA-256 not Python's hash(): Python's built-in hash is salted per
    process (PYTHONHASHSEED), so hash("BRCA1") differs between runs. SHA-256
    is deterministic across Python versions, OSes, and invocations.
    """
    if gene_col not in df.columns:
        raise ValueError(f"{gene_col!r} not in df.columns")
    if not (0.0 < holdout_frac < 1.0):
        raise ValueError("holdout_frac in (0,1)")
    if n_buckets < 10:
        raise ValueError("n_buckets must be >= 10 for reasonable granularity")

    threshold = int(round(holdout_frac * n_buckets))
    genes = df[gene_col].fillna("__UNKNOWN__").astype(str).values

    # Cache bucket assignments per unique gene so we hash each gene once
    bucket_cache: dict[str, int] = {}

    def _bucket_of(gene: str) -> int:
        if gene not in bucket_cache:
            key = f"{seed}:{gene}".encode("utf-8")
            digest = hashlib.sha256(key).digest()
            bucket_cache[gene] = int.from_bytes(digest[:4], "big") % n_buckets
        return bucket_cache[gene]

    holdout_mask = np.array([_bucket_of(g) < threshold for g in genes], dtype=bool)
    holdout_idx = np.where(holdout_mask)[0]
    train_idx = np.where(~holdout_mask)[0]

    # Sanity: every unique gene is in exactly one set
    holdout_genes = set(genes[holdout_idx])
    train_genes = set(genes[train_idx])
    overlap = holdout_genes & train_genes
    assert not overlap, f"Gene leak between splits: {sorted(overlap)[:5]}"

    n_unique = len(set(genes))
    achieved_frac = len(holdout_genes) / n_unique if n_unique else 0.0
    logger.info(
        "unseen_gene_holdout_split: holdout=%d genes / %d total (%.3f, "
        "target %.3f), train_variants=%d, holdout_variants=%d",
        len(holdout_genes),
        n_unique,
        achieved_frac,
        holdout_frac,
        len(train_idx),
        len(holdout_idx),
    )
    return train_idx, holdout_idx


# ---------------------------------------------------------------------------
# Diagnostics helper
# ---------------------------------------------------------------------------
def split_summary(
    df: pd.DataFrame,
    splits: dict[str, np.ndarray],
    label_col: str = "acmg_label",
    gene_col: str = "gene_symbol",
) -> pd.DataFrame:
    """
    Summarise a set of splits: count, prevalence, unique genes.
    Handy for manifest: writer.save_manifest(... splits_summary=...).
    """
    rows: list[dict] = []
    for name, idx in splits.items():
        sub = df.iloc[idx]
        rows.append(
            {
                "split": name,
                "n_variants": int(len(sub)),
                "n_genes": (
                    int(sub[gene_col].nunique()) if gene_col in sub.columns else -1
                ),
                "prevalence": (
                    float(sub[label_col].mean())
                    if label_col in sub.columns and len(sub)
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)
