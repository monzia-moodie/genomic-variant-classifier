"""
src/data/dbsnp.py
=================
dbSNP allele frequency supplement connector — Phase 4, Connector 4.

Reads a dbSNP parquet file (derived from NCBI FTP build 156) and adds one
variant-level feature:

    dbsnp_af   float   Population allele frequency from dbSNP.
                       Used to supplement gnomAD for variants absent from
                       gnomAD v4.  Default: 0.0 (treat absent as ultra-rare).

Variant-level join: by chrom:pos:ref:alt lookup key.

Expected parquet columns:
    variant_id    str   "chrom:pos:ref:alt" key
    allele_freq   float Population AF

Stub mode:
    When parquet_path is None or the file does not exist, all variants receive
    dbsnp_af = 0.0 and a WARNING is logged.

Data source:
    NCBI FTP: https://ftp.ncbi.nlm.nih.gov/snp/
    Build 156 VCF → convert to parquet with scripts/build_dbsnp_parquet.py
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.data.database_connectors import BaseConnector, FetchConfig

logger = logging.getLogger(__name__)

DEFAULT_AF = 0.0


class DbSNPConnector(BaseConnector):
    """
    Annotates variants with dbSNP population allele frequencies.

    Usage
    -----
        connector = DbSNPConnector(parquet_path="data/external/dbsnp156.parquet")
        annotated_df = connector.annotate_dataframe(variant_df)
        # annotated_df now has a dbsnp_af column

    If parquet_path is None or file is absent, stub mode applies:
    all variants receive dbsnp_af = 0.0.
    """

    source_name = "dbsnp"

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

    def annotate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add dbsnp_af column to df.

        Parameters
        ----------
        df : pd.DataFrame
            Variant DataFrame; must contain chrom, pos, ref, alt for joining.

        Returns
        -------
        pd.DataFrame with dbsnp_af column added.
        """
        if df.empty:
            result = df.copy()
            result["dbsnp_af"] = pd.Series(dtype=float)
            return result

        if self.parquet_path is None:
            logger.warning(
                "DbSNPConnector: parquet_path not set — returning dbsnp_af=0.0.  "
                "Download dbSNP build 156 from https://ftp.ncbi.nlm.nih.gov/snp/."
            )
            result = df.copy()
            result["dbsnp_af"] = DEFAULT_AF
            return result

        lookup = self._get_lookup()
        if lookup.empty:
            result = df.copy()
            result["dbsnp_af"] = DEFAULT_AF
            return result

        return self._annotate(df, lookup)

    def fetch(self, variant_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Wraps annotate_dataframe for BaseConnector compatibility."""
        return self.annotate_dataframe(variant_df)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_lookup(self) -> pd.DataFrame:
        """Return lookup DataFrame, using parquet cache when available."""
        cache_key = "af_lookup"
        cached = self._load_cache(cache_key)
        if cached is not None and not cached.empty:
            logger.info(
                "DbSNPConnector: loaded %d AFs from parquet cache.", len(cached)
            )
            return cached

        if not self.parquet_path.exists():
            logger.warning(
                "DbSNPConnector: parquet not found at '%s' — returning dbsnp_af=0.0.",
                self.parquet_path,
            )
            return pd.DataFrame(columns=["variant_id", "allele_freq"])

        logger.info("DbSNPConnector: loading dbSNP parquet from %s ...", self.parquet_path)
        try:
            lookup = pd.read_parquet(self.parquet_path, columns=["variant_id", "allele_freq"])
        except Exception as exc:
            logger.error("DbSNPConnector: failed to read parquet: %s", exc)
            return pd.DataFrame(columns=["variant_id", "allele_freq"])

        lookup = lookup.dropna(subset=["variant_id", "allele_freq"])
        lookup = lookup.drop_duplicates(subset=["variant_id"])
        lookup["allele_freq"] = lookup["allele_freq"].astype(float).clip(lower=0.0)

        self._save_cache(cache_key, lookup)
        logger.info("DbSNPConnector: cached %d variant AFs.", len(lookup))
        return lookup

    def _annotate(self, variant_df: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
        """Left-join dbSNP AFs onto variant_df by chrom:pos:ref:alt key."""
        result = variant_df.copy()

        # Build lookup key from chrom, pos, ref, alt
        chrom = result.get("chrom", pd.Series([""] * len(result), index=result.index)).astype(str)
        # Strip chr prefix to match dbSNP key format
        chrom = chrom.str.replace(r"^chr", "", regex=True)

        result["_lookup_key"] = (
            chrom + ":" +
            result["pos"].astype(str) + ":" +
            result["ref"].astype(str) + ":" +
            result["alt"].astype(str)
        )

        score_cols = lookup[["variant_id", "allele_freq"]].rename(
            columns={"variant_id": "_lookup_key", "allele_freq": "dbsnp_af"}
        )
        result = result.merge(score_cols, on="_lookup_key", how="left")
        result["dbsnp_af"] = result["dbsnp_af"].fillna(DEFAULT_AF).clip(lower=0.0)
        result = result.drop(columns=["_lookup_key"])

        n_found = (result["dbsnp_af"] > 0).sum()
        logger.info(
            "DbSNPConnector: %d / %d variants have dbsnp_af > 0.",
            n_found, len(result),
        )
        return result
