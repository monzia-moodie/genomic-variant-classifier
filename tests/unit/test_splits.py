"""
tests/unit/test_splits.py
==========================
Invariant tests for src/data/splits.py. Run with:

    pytest tests/unit/test_splits.py -v

The unseen-gene stability tests are the most important: they gate Rule 6
(reproducibility of longitudinal comparisons across runs).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.splits import (
    gene_stratified_split,
    split_summary,
    unseen_gene_holdout_split,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def variants_df() -> pd.DataFrame:
    """Synthetic variant-like DataFrame: 50 genes, 20 variants each."""
    rng = np.random.default_rng(0)
    genes = [f"GENE{i:03d}" for i in range(50)]
    rows = []
    for g in genes:
        for k in range(20):
            rows.append(
                {
                    "variant_id": f"{g}_v{k}",
                    "gene_symbol": g,
                    "acmg_label": int(rng.random() > 0.7),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def growing_variants_df(variants_df: pd.DataFrame) -> pd.DataFrame:
    """A superset of variants_df with 50 extra genes added."""
    extra_rows = [
        {
            "variant_id": f"NEW{i:03d}_v{k}",
            "gene_symbol": f"NEW{i:03d}",
            "acmg_label": int((i + k) % 2),
        }
        for i in range(50)
        for k in range(20)
    ]
    return pd.concat([variants_df, pd.DataFrame(extra_rows)], ignore_index=True)


# ---------------------------------------------------------------------------
# gene_stratified_split
# ---------------------------------------------------------------------------
class TestGeneStratifiedSplit:

    def test_no_gene_leak(self, variants_df):
        tr, va, te = gene_stratified_split(variants_df, 0.2, 0.2, seed=42)
        tr_g = set(variants_df.iloc[tr]["gene_symbol"])
        va_g = set(variants_df.iloc[va]["gene_symbol"])
        te_g = set(variants_df.iloc[te]["gene_symbol"])
        assert not (tr_g & va_g), "train and val share genes"
        assert not (tr_g & te_g), "train and test share genes"
        assert not (va_g & te_g), "val and test share genes"

    def test_indices_cover_everything(self, variants_df):
        tr, va, te = gene_stratified_split(variants_df, 0.2, 0.2, seed=42)
        covered = np.concatenate([tr, va, te])
        assert len(np.unique(covered)) == len(variants_df)
        assert sorted(covered) == list(range(len(variants_df)))

    def test_fractions_within_tolerance(self, variants_df):
        tr, va, te = gene_stratified_split(variants_df, 0.2, 0.2, seed=42)
        n_genes_total = variants_df["gene_symbol"].nunique()
        n_test = variants_df.iloc[te]["gene_symbol"].nunique()
        n_val = variants_df.iloc[va]["gene_symbol"].nunique()
        assert abs(n_test / n_genes_total - 0.2) < 0.05
        assert abs(n_val / n_genes_total - 0.2) < 0.05

    def test_reproducible_with_same_seed(self, variants_df):
        tr1, va1, te1 = gene_stratified_split(variants_df, 0.2, 0.2, seed=42)
        tr2, va2, te2 = gene_stratified_split(variants_df, 0.2, 0.2, seed=42)
        np.testing.assert_array_equal(tr1, tr2)
        np.testing.assert_array_equal(va1, va2)
        np.testing.assert_array_equal(te1, te2)

    def test_rejects_missing_gene_column(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(ValueError, match="gene_symbol"):
            gene_stratified_split(df, 0.2, 0.2, seed=42)


# ---------------------------------------------------------------------------
# unseen_gene_holdout_split
# ---------------------------------------------------------------------------
class TestUnseenGeneHoldoutSplit:

    def test_no_gene_leak(self, variants_df):
        tr, ho = unseen_gene_holdout_split(variants_df, holdout_frac=0.2, seed=42)
        tr_g = set(variants_df.iloc[tr]["gene_symbol"])
        ho_g = set(variants_df.iloc[ho]["gene_symbol"])
        assert not (tr_g & ho_g), "train and holdout share genes"

    def test_deterministic_across_calls(self, variants_df):
        """Same DataFrame + same seed → byte-identical indices."""
        tr1, ho1 = unseen_gene_holdout_split(variants_df, 0.2, seed=42)
        tr2, ho2 = unseen_gene_holdout_split(variants_df, 0.2, seed=42)
        np.testing.assert_array_equal(tr1, tr2)
        np.testing.assert_array_equal(ho1, ho2)

    def test_hash_stability_across_data_versions(
        self,
        variants_df,
        growing_variants_df,
    ):
        """
        Core Rule 6 invariant. When the dataset grows, genes that WERE in the
        holdout set must REMAIN in the holdout set, so longitudinal
        comparisons across runs are valid.
        """
        _, ho_small = unseen_gene_holdout_split(variants_df, 0.2, seed=42)
        _, ho_large = unseen_gene_holdout_split(growing_variants_df, 0.2, seed=42)

        small_holdout_genes = set(variants_df.iloc[ho_small]["gene_symbol"])
        large_holdout_genes = set(growing_variants_df.iloc[ho_large]["gene_symbol"])

        # Every gene in the small holdout must still be in the large holdout.
        # (The large holdout is a superset, because new genes can also hash
        # into the holdout bucket range.)
        missing = small_holdout_genes - large_holdout_genes
        assert (
            not missing
        ), f"Genes that were held out in v1 were dropped in v2: {sorted(missing)}"

    def test_frac_within_tolerance(self, variants_df):
        _, ho = unseen_gene_holdout_split(variants_df, 0.2, seed=42)
        n_total = variants_df["gene_symbol"].nunique()
        n_ho = variants_df.iloc[ho]["gene_symbol"].nunique()
        # 50 genes × 20 % = 10 held out; tolerance ±3 genes
        assert (
            abs(n_ho / n_total - 0.2) < 0.06
        ), f"Holdout fraction {n_ho / n_total:.3f} outside tolerance of 0.2"

    def test_rejects_bad_frac(self, variants_df):
        for bad in (0.0, 1.0, -0.1, 1.5):
            with pytest.raises(ValueError, match="holdout_frac"):
                unseen_gene_holdout_split(variants_df, holdout_frac=bad)

    def test_rejects_missing_gene_column(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(ValueError, match="gene_symbol"):
            unseen_gene_holdout_split(df, 0.2, seed=42)


# ---------------------------------------------------------------------------
# split_summary
# ---------------------------------------------------------------------------
def test_split_summary_shape(variants_df):
    tr, va, te = gene_stratified_split(variants_df, 0.2, 0.2, seed=42)
    summary = split_summary(
        variants_df,
        {"train": tr, "val": va, "test": te},
    )
    assert set(summary["split"]) == {"train", "val", "test"}
    assert summary["n_variants"].sum() == len(variants_df)
