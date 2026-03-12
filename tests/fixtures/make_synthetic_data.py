"""
tests/fixtures/make_synthetic_data.py
======================================
Canonical synthetic dataset generator for unit tests and dev-mode pipeline runs.

CHANGES FROM PHASE 1:
  - Identical synthetic data generation appeared in both Cell 51 and Cell 59
    of the notebook, with no shared abstraction. Any future change to the
    data schema had to be updated in two places (Issue E).
  - Extracted here as the single source of truth. Both the notebook and
    unit tests should import from this module.

Usage (in notebook):
    from tests.fixtures.make_synthetic_data import make_synthetic_variants
    df = make_synthetic_variants(n=1000, seed=42)

Usage (in test):
    @pytest.fixture
    def synthetic_df():
        return make_synthetic_variants(n=200, seed=0)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_variants(
    n: int = 1000,
    seed: int = 42,
    pathogenic_fraction: float = 0.15,
) -> pd.DataFrame:
    """
    Generate a synthetic genomic variant DataFrame that mimics the canonical
    schema produced by database_connectors.py.

    The synthetic label is deterministic from the features so models can
    learn a signal — useful for integration tests and quick pipeline smoke
    tests where real ClinVar data is unavailable.

    Args:
        n:                    Number of variants to generate.
        seed:                 NumPy random seed for reproducibility.
        pathogenic_fraction:  Approximate fraction of pathogenic variants
                              (mirrors real ClinVar class imbalance ~15%).

    Returns:
        DataFrame with the canonical schema plus feature columns.
    """
    rng = np.random.default_rng(seed)

    # ── Labels ─────────────────────────────────────────────────────────────
    is_pathogenic = rng.uniform(0, 1, n) < pathogenic_fraction

    # ── Allele frequency ───────────────────────────────────────────────────
    # Pathogenic variants are typically ultra-rare; benign variants span the
    # full frequency spectrum.
    af = np.where(
        is_pathogenic,
        rng.exponential(scale=1e-5, size=n).clip(0, 0.01),
        rng.choice(
            [0.0, 1e-5, 1e-4, 1e-3, 0.01, 0.05, 0.1, 0.2, 0.5],
            n, p=[0.1, 0.05, 0.1, 0.1, 0.15, 0.15, 0.15, 0.1, 0.1],
        ),
    )

    # ── Functional scores ──────────────────────────────────────────────────
    cadd_phred = np.where(
        is_pathogenic,
        rng.normal(loc=28, scale=6, size=n).clip(0, 60),
        rng.normal(loc=12, scale=8, size=n).clip(0, 60),
    )
    sift_score = np.where(
        is_pathogenic,
        rng.beta(a=1, b=5, size=n),     # skewed toward 0 (deleterious)
        rng.beta(a=5, b=1, size=n),     # skewed toward 1 (tolerated)
    )
    polyphen2_score = np.where(
        is_pathogenic,
        rng.beta(a=5, b=1, size=n),     # skewed toward 1 (damaging)
        rng.beta(a=1, b=5, size=n),     # skewed toward 0 (benign)
    )
    revel_score = np.where(
        is_pathogenic,
        rng.beta(a=4, b=2, size=n),
        rng.beta(a=2, b=4, size=n),
    )

    # ── Consequence ────────────────────────────────────────────────────────
    pathogenic_consequences = [
        "stop_gained", "frameshift_variant", "splice_donor_variant",
        "splice_acceptor_variant", "missense_variant",
    ]
    benign_consequences = [
        "synonymous_variant", "3_prime_UTR_variant", "5_prime_UTR_variant",
        "intron_variant", "missense_variant",
    ]
    p_consequence_weights = [0.15, 0.20, 0.10, 0.10, 0.45]
    b_consequence_weights = [0.25, 0.15, 0.15, 0.30, 0.15]
    consequence = np.where(
        is_pathogenic,
        rng.choice(pathogenic_consequences, n, p=p_consequence_weights),
        rng.choice(benign_consequences,     n, p=b_consequence_weights),
    )

    # ── Variant type ───────────────────────────────────────────────────────
    ref_bases = rng.choice(list("ACGT"), n)
    alt_bases  = np.array([
        rng.choice([b for b in "ACGT" if b != r])
        for r in ref_bases
    ])

    # ── Gene / genomic location ────────────────────────────────────────────
    genes = [
        "BRCA1", "BRCA2", "TP53", "PTEN", "MLH1", "MSH2", "APC",
        "AGRN", "TTN", "OBSCN", "SYNE1", "RYR1", "NF1", "NF2", "VHL",
    ]
    gene_weights = [
        0.12, 0.10, 0.10, 0.08, 0.07, 0.07, 0.07,
        0.05, 0.05, 0.04, 0.04, 0.04, 0.06, 0.05, 0.06,
    ]
    gene_symbol = rng.choice(genes, n, p=gene_weights)
    chrom_choices = [str(i) for i in range(1, 23)] + ["X"]
    chrom = rng.choice(chrom_choices, n)
    pos   = rng.integers(100_000, 250_000_000, n)

    # ── Sequences ──────────────────────────────────────────────────────────
    bases = np.array(list("ACGT"))
    fasta_seq = np.array(["".join(rng.choice(bases, 101)) for _ in range(n)])

    # ── Pathogenicity labels ───────────────────────────────────────────────
    pathogenicity = np.where(is_pathogenic, "pathogenic", "benign")
    clinical_sig  = np.where(is_pathogenic, "Pathogenic", "Benign")
    label         = is_pathogenic.astype(int)

    # ── Assemble DataFrame ─────────────────────────────────────────────────
    df = pd.DataFrame({
        # Canonical schema columns
        "variant_id":       [
            f"synthetic:{chrom[i]}:{pos[i]}:{ref_bases[i]}:{alt_bases[i]}"
            for i in range(n)
        ],
        "source_db":        "synthetic",
        "chrom":            chrom,
        "pos":              pos,
        "ref":              ref_bases,
        "alt":              alt_bases,
        "gene_symbol":      gene_symbol,
        "transcript_id":    None,
        "consequence":      consequence,
        "pathogenicity":    pathogenicity,
        "clinical_sig":     clinical_sig,
        "allele_freq":      af.round(8),
        "protein_change":   None,
        "fasta_seq":        fasta_seq,
        "source_id":        [f"syn_{i:06d}" for i in range(n)],
        "metadata":         [{}] * n,

        # Pre-computed functional scores
        "cadd_phred":       cadd_phred.round(2),
        "sift_score":       sift_score.round(4),
        "polyphen2_score":  polyphen2_score.round(4),
        "revel_score":      revel_score.round(4),

        # Ground-truth label for supervised training
        "label":            label,
    })

    return df


def save_synthetic_variants(
    path: str = "data/synthetic/variants_n1000.parquet",
    n: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate and save synthetic variants to parquet.
    Convenience wrapper for notebook use.
    """
    import pathlib
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    df = make_synthetic_variants(n=n, seed=seed)
    df.to_parquet(path, index=False)
    print(f"Saved {len(df):,} synthetic variants to {path}")
    print(f"  Pathogenic: {df['label'].sum():,} ({df['label'].mean()*100:.1f}%)")
    print(f"  Features:   {df.shape[1]} columns")
    return df


if __name__ == "__main__":
    save_synthetic_variants()
