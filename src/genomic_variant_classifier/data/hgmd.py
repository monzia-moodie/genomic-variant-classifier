"""
src/data/hgmd.py
================
HGMD connector — Phase 4, Connector 6.

Requires an institutional license from QIAGEN (https://www.qiagenbioinformatics.com/products/human-gene-mutation-database/).
Most users will NOT have access, so stub mode is the expected default.

Adds two variant-level features:

    hgmd_is_disease_mutation   int (0/1)
        1 if the variant is classified as DM (disease mutation) or DM?
        (possible disease mutation) in HGMD.  0 otherwise.

    hgmd_n_reports   int
        Number of HGMD records for this variant.  Variants can appear
        multiple times (multiple publications).

Expected input format:
    VCF or tab-separated file with at minimum:
        CHROM, POS, REF, ALT, CLASS   (CLASS = DM, DM?, DP, DFP, FP, R)

Stub mode:
    When hgmd_path is None (the default, and appropriate for users without
    a license), all variants receive hgmd_is_disease_mutation=0,
    hgmd_n_reports=0.  No WARNING is emitted to avoid log spam for the
    common case; a DEBUG message is written instead.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.data.database_connectors import BaseConnector, FetchConfig

logger = logging.getLogger(__name__)

# HGMD variant classes that indicate disease causation
DISEASE_MUTATION_CLASSES = {"DM", "DM?"}

DEFAULT_IS_DM      = 0
DEFAULT_N_REPORTS  = 0


class HGMDConnector(BaseConnector):
    """
    Annotates variants with HGMD disease mutation classification.

    IMPORTANT: HGMD requires an institutional license.  Stub mode
    (hgmd_path=None) is the safe default for users without access.

    Usage (with license)
    --------------------
        connector = HGMDConnector(hgmd_path="data/external/hgmd_pro.vcf")
        annotated_df = connector.annotate_dataframe(variant_df)

    If hgmd_path is None or file is absent, all variants receive
    hgmd_is_disease_mutation=0, hgmd_n_reports=0.
    """

    source_name = "hgmd"

    def __init__(
        self,
        hgmd_path: Optional[str | Path] = None,
        config: Optional[FetchConfig] = None,
    ) -> None:
        super().__init__(config)
        self.hgmd_path: Optional[Path] = (
            Path(hgmd_path) if hgmd_path is not None else None
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def annotate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add hgmd_is_disease_mutation and hgmd_n_reports columns to df.

        Parameters
        ----------
        df : pd.DataFrame
            Variant DataFrame; must contain chrom, pos, ref, alt.

        Returns
        -------
        pd.DataFrame with two new columns added.
        """
        if df.empty:
            result = df.copy()
            result["hgmd_is_disease_mutation"] = pd.Series(dtype=int)
            result["hgmd_n_reports"]           = pd.Series(dtype=int)
            return result

        if self.hgmd_path is None:
            logger.debug(
                "HGMDConnector: hgmd_path not set (expected for users without "
                "HGMD license) — returning hgmd_is_disease_mutation=0, hgmd_n_reports=0."
            )
            result = df.copy()
            result["hgmd_is_disease_mutation"] = DEFAULT_IS_DM
            result["hgmd_n_reports"]           = DEFAULT_N_REPORTS
            return result

        lookup = self._get_lookup()
        if lookup.empty:
            result = df.copy()
            result["hgmd_is_disease_mutation"] = DEFAULT_IS_DM
            result["hgmd_n_reports"]           = DEFAULT_N_REPORTS
            return result

        return self._annotate(df, lookup)

    def fetch(self, variant_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Wraps annotate_dataframe for BaseConnector compatibility."""
        return self.annotate_dataframe(variant_df)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_lookup(self) -> pd.DataFrame:
        """Return HGMD lookup DataFrame, using parquet cache when available."""
        cache_key = "variant_lookup"
        cached = self._load_cache(cache_key)
        if cached is not None and not cached.empty:
            logger.info(
                "HGMDConnector: loaded %d HGMD records from cache.", len(cached)
            )
            return cached

        if not self.hgmd_path.exists():
            logger.warning(
                "HGMDConnector: HGMD file not found at '%s' — returning defaults.",
                self.hgmd_path,
            )
            return pd.DataFrame(
                columns=["lookup_key", "hgmd_is_disease_mutation", "hgmd_n_reports"]
            )

        logger.info(
            "HGMDConnector: parsing HGMD file from %s ...", self.hgmd_path
        )
        lookup = self._parse_hgmd(self.hgmd_path)
        if not lookup.empty:
            self._save_cache(cache_key, lookup)
            logger.info("HGMDConnector: cached %d HGMD variant records.", len(lookup))
        return lookup

    def _parse_hgmd(self, path: Path) -> pd.DataFrame:
        """
        Parse HGMD VCF or tab-separated file.

        Supports:
          - VCF format (INFO field contains CLASS=DM or CLASS=DM?)
          - Tab-separated with columns: CHROM, POS, REF, ALT, CLASS
        """
        rows = []
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                is_vcf = None
                for line in f:
                    if line.startswith("##"):
                        continue
                    if line.startswith("#"):
                        # VCF header line
                        is_vcf = True
                        continue

                    parts = line.rstrip("\n").split("\t")

                    if is_vcf:
                        # VCF format: CHROM POS ID REF ALT QUAL FILTER INFO ...
                        if len(parts) < 8:
                            continue
                        chrom  = parts[0].lstrip("chr")
                        pos    = parts[1]
                        ref    = parts[3]
                        alt    = parts[4]
                        info   = parts[7]
                        # Extract CLASS from INFO field
                        hgmd_class = ""
                        for field in info.split(";"):
                            if field.startswith("CLASS="):
                                hgmd_class = field[6:].strip()
                                break
                    else:
                        # Tab-separated: CHROM POS REF ALT CLASS
                        if len(parts) < 5:
                            continue
                        chrom      = str(parts[0]).lstrip("chr")
                        pos        = parts[1]
                        ref        = parts[2]
                        alt        = parts[3]
                        hgmd_class = parts[4].strip()

                    lookup_key = f"{chrom}:{pos}:{ref}:{alt}"
                    is_dm = 1 if hgmd_class.upper() in DISEASE_MUTATION_CLASSES else 0
                    rows.append({"lookup_key": lookup_key, "hgmd_class": hgmd_class, "is_dm": is_dm})

        except OSError as exc:
            logger.error("HGMDConnector: failed to read %s: %s", path, exc)
            return pd.DataFrame(
                columns=["lookup_key", "hgmd_is_disease_mutation", "hgmd_n_reports"]
            )

        if not rows:
            return pd.DataFrame(
                columns=["lookup_key", "hgmd_is_disease_mutation", "hgmd_n_reports"]
            )

        raw = pd.DataFrame(rows)
        # Aggregate: max(is_dm) and count per locus
        agg = (
            raw.groupby("lookup_key")
            .agg(
                hgmd_is_disease_mutation=("is_dm", "max"),
                hgmd_n_reports=("lookup_key", "count"),
            )
            .reset_index()
        )

        logger.info(
            "HGMDConnector: parsed %d records → %d unique loci (%d DM).",
            len(raw),
            len(agg),
            agg["hgmd_is_disease_mutation"].sum(),
        )
        return agg

    def _annotate(self, variant_df: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
        """Left-join HGMD annotations onto variant_df by chrom:pos:ref:alt key."""
        result = variant_df.copy()

        chrom = result.get(
            "chrom", pd.Series([""] * len(result), index=result.index)
        ).astype(str).str.replace(r"^chr", "", regex=True)

        result["_lookup_key"] = (
            chrom + ":" +
            result["pos"].astype(str) + ":" +
            result["ref"].astype(str) + ":" +
            result["alt"].astype(str)
        )

        result = result.merge(
            lookup[["lookup_key", "hgmd_is_disease_mutation", "hgmd_n_reports"]].rename(
                columns={"lookup_key": "_lookup_key"}
            ),
            on="_lookup_key",
            how="left",
        )
        result["hgmd_is_disease_mutation"] = (
            result["hgmd_is_disease_mutation"].fillna(DEFAULT_IS_DM).astype(int)
        )
        result["hgmd_n_reports"] = (
            result["hgmd_n_reports"].fillna(DEFAULT_N_REPORTS).astype(int)
        )
        result = result.drop(columns=["_lookup_key"])

        n_dm = (result["hgmd_is_disease_mutation"] == 1).sum()
        logger.info(
            "HGMDConnector: %d / %d variants flagged as HGMD disease mutations.",
            n_dm, len(result),
        )
        return result
