"""
src/data/thousandgenomes.py
============================
1000 Genomes Phase 3 allele-frequency fallback connector — Phase 5.1.

Fills allele_freq for variants that are:
  (a) absent from gnomAD entirely, OR
  (b) present in gnomAD but with a null AF after the gnomAD join.

gnomAD covers whole-genome and exome variants; 1000 Genomes provides a
complementary signal for sites gnomAD did not sequence at sufficient depth.

Expected parquet schema (same format as gnomAD AF parquet):
    variant_id    str    "chrom:pos:ref:alt" key (no prefix)
    allele_freq   float  Global alternate AF across all 1000G super-populations

Building the parquet from 1000G Phase 3 VCF:
    python scripts/build_1kg_parquet.py \
        --vcf-dir data/external/1000g/phase3_vcf \
        --out     data/external/1000g/kg_phase3_af.parquet

Data source:
    http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/
    ALL.chr*.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz

Stub mode:
    When parquet_path is None or absent, the connector returns an empty
    result (no AF filled) and logs a WARNING.  The pipeline continues.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.data.database_connectors import BaseConnector, FetchConfig

logger = logging.getLogger(__name__)


class ThousandGenomesConnector(BaseConnector):
    """
    Fills allele_freq from 1000 Genomes Phase 3 for variants where it is
    still null after the gnomAD join.

    Usage
    -----
        kg = ThousandGenomesConnector("data/external/1000g/kg_phase3_af.parquet")
        df = kg.fill_missing_af(df)   # fills only where df["allele_freq"].isna()

    If parquet_path is None or absent, returns df unchanged (stub mode).
    """

    source_name = "1000genomes"

    def __init__(
        self,
        parquet_path: Optional[str | Path] = None,
        config: Optional[FetchConfig] = None,
    ) -> None:
        super().__init__(config)
        self.parquet_path: Optional[Path] = (
            Path(parquet_path) if parquet_path is not None else None
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fill_missing_af(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill null allele_freq values from 1000G AF.

        Only rows where allele_freq is NaN / None are updated; rows that
        already have an AF (from gnomAD or the caller) are left unchanged.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain chrom, pos, ref, alt and allele_freq columns.

        Returns
        -------
        pd.DataFrame with allele_freq partially or fully filled.
        """
        if df.empty:
            return df.copy()

        missing_mask = df["allele_freq"].isna()
        n_missing = int(missing_mask.sum())
        if n_missing == 0:
            logger.debug("ThousandGenomesConnector: no null AFs — skipping.")
            return df

        if self.parquet_path is None:
            logger.warning(
                "ThousandGenomesConnector: parquet_path not set — %d variants "
                "will keep null AF.  "
                "Build it with scripts/build_1kg_parquet.py.",
                n_missing,
            )
            return df

        lookup = self._get_lookup()
        if lookup.empty:
            return df

        return self._fill(df, lookup)

    def fetch(self, variant_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Wraps fill_missing_af for BaseConnector compatibility."""
        return self.fill_missing_af(variant_df)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_lookup(self) -> pd.DataFrame:
        """Return lookup DataFrame, using parquet cache when available."""
        cache_key = "kg_af_lookup"
        cached = self._load_cache(cache_key)
        if cached is not None and not cached.empty:
            logger.info(
                "ThousandGenomesConnector: loaded %d AFs from cache.", len(cached)
            )
            return cached

        if not self.parquet_path.exists():
            logger.warning(
                "ThousandGenomesConnector: parquet not found at '%s'.",
                self.parquet_path,
            )
            return pd.DataFrame(columns=["variant_id", "allele_freq"])

        logger.info(
            "ThousandGenomesConnector: loading 1000G parquet from %s ...",
            self.parquet_path,
        )
        try:
            lookup = pd.read_parquet(
                self.parquet_path, columns=["variant_id", "allele_freq"]
            )
        except Exception as exc:
            logger.error("ThousandGenomesConnector: failed to read parquet: %s", exc)
            return pd.DataFrame(columns=["variant_id", "allele_freq"])

        lookup = lookup.dropna(subset=["variant_id", "allele_freq"])
        lookup = lookup.drop_duplicates(subset=["variant_id"])
        lookup["allele_freq"] = lookup["allele_freq"].astype(float).clip(lower=0.0)

        self._save_cache(cache_key, lookup)
        logger.info(
            "ThousandGenomesConnector: cached %d variant AFs.", len(lookup)
        )
        return lookup

    def _fill(self, df: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
        """
        Left-join 1000G AFs onto df and fill null allele_freq values only.
        """
        result = df.copy()

        # Build join key from chrom, pos, ref, alt
        chrom = result.get(
            "chrom", pd.Series([""] * len(result), index=result.index)
        ).astype(str).str.replace(r"^chr", "", regex=True)

        result["_kg_key"] = (
            chrom + ":" +
            result["pos"].astype(str) + ":" +
            result["ref"].astype(str) + ":" +
            result["alt"].astype(str)
        )

        kg_af = lookup.rename(
            columns={"variant_id": "_kg_key", "allele_freq": "_kg_af"}
        )
        result = result.merge(kg_af, on="_kg_key", how="left")

        # Fill only where allele_freq is still null
        null_mask = result["allele_freq"].isna()
        result.loc[null_mask, "allele_freq"] = result.loc[null_mask, "_kg_af"]
        result = result.drop(columns=["_kg_key", "_kg_af"])

        n_filled = int(null_mask.sum()) - int(result["allele_freq"].isna().sum())
        n_still_null = int(result["allele_freq"].isna().sum())
        logger.info(
            "ThousandGenomesConnector: filled %d AFs from 1000G "
            "(%d still null after fill).",
            n_filled, n_still_null,
        )
        return result
