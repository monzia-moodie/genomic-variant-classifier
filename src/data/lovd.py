"""
src/data/lovd.py
================
LOVDConnector — annotates variants with LOVD classification scores.

Joins the pre-built LOVD parquet (data/external/lovd/lovd_all_variants.parquet)
to the variant DataFrame on (chrom, pos, ref, alt).

Output feature
--------------
lovd_variant_class : int
    0  — not found in LOVD (default for all variants)
    1  — Benign / Likely benign
    2  — Variant of uncertain significance (VUS)
    3  — Likely pathogenic
    4  — Pathogenic

This ordinal encoding preserves the clinical severity ranking and lets
tree models exploit the natural ordering.

Usage
-----
    from src.data.lovd import LOVDConnector
    lovd = LOVDConnector(parquet_path="data/external/lovd/lovd_all_variants.parquet")
    df = lovd.annotate_dataframe(df)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Classification string → ordinal mapping
# ---------------------------------------------------------------------------
_CLASSIFICATION_MAP: dict[str, int] = {
    # Benign tier
    "benign":                    1,
    "likely benign":             1,
    "benign/likely benign":      1,
    # VUS tier
    "uncertain significance":    2,
    "variant of uncertain significance": 2,
    "vus":                       2,
    "conflicting":               2,
    "conflicting interpretations of pathogenicity": 2,
    # Likely pathogenic tier
    "likely pathogenic":         3,
    "pathogenic/likely pathogenic": 3,
    # Pathogenic tier
    "pathogenic":                4,
}

_DEFAULT_CLASS = 0   # not found in LOVD


class LOVDConnector:
    """
    Annotates a variant DataFrame with LOVD classification scores.

    Parameters
    ----------
    parquet_path : path to lovd_all_variants.parquet produced by the LOVD
                   download script.  If None or file not found, the connector
                   runs in stub mode (all variants receive lovd_variant_class=0)
                   and logs a WARNING.
    """

    def __init__(self, parquet_path: Optional[str | Path] = None) -> None:
        self._parquet_path = Path(parquet_path) if parquet_path else None
        self._lookup: Optional[pd.DataFrame] = None
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def annotate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Join LOVD classifications onto *df* by (chrom, pos, ref, alt).

        Adds column:
            lovd_variant_class : int  (0 = not in LOVD, 1–4 = ordinal class)

        Parameters
        ----------
        df : DataFrame with columns chrom, pos, ref, alt (all as strings/ints).
             pos may be int or str — both are handled.

        Returns
        -------
        df with lovd_variant_class column added (or overwritten if already present).
        """
        if self._lookup is None:
            logger.warning(
                "LOVDConnector: no parquet loaded — "
                "all variants will receive lovd_variant_class=0. "
                "Set parquet_path to activate LOVD annotation."
            )
            df["lovd_variant_class"] = _DEFAULT_CLASS
            return df

        # Build join keys on the input DataFrame
        df = df.copy()
        df["_chrom"] = df["chrom"].astype(str).str.lstrip("chr")
        df["_pos"]   = df["pos"].astype(str)
        df["_ref"]   = df["ref"].astype(str)
        df["_alt"]   = df["alt"].astype(str)

        merged = df.merge(
            self._lookup,
            on=["_chrom", "_pos", "_ref", "_alt"],
            how="left",
        )
        df["lovd_variant_class"] = (
            merged["lovd_variant_class"]
            .fillna(_DEFAULT_CLASS)
            .astype(int)
            .values
        )

        n_annotated = (df["lovd_variant_class"] > 0).sum()
        logger.info(
            "LOVDConnector: %d / %d variants annotated (lovd_variant_class > 0).",
            n_annotated, len(df),
        )

        # Clean up temporary join columns
        df = df.drop(columns=["_chrom", "_pos", "_ref", "_alt"])
        return df

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._parquet_path is None or not self._parquet_path.exists():
            if self._parquet_path is not None:
                logger.warning(
                    "LOVDConnector: file not found at %s — stub mode active.",
                    self._parquet_path,
                )
            return

        try:
            raw = pd.read_parquet(self._parquet_path)
        except Exception as exc:
            logger.error("LOVDConnector: failed to load parquet: %s", exc)
            return

        # Parse variant_id into join-key columns
        # Expected format: "chrom:pos:ref:alt"  (e.g. "17:7675234:G:T")
        if "variant_id" in raw.columns:
            parts = raw["variant_id"].str.split(":", expand=True)
            if parts.shape[1] >= 4:
                raw["_chrom"] = parts[0].str.lstrip("chr")
                raw["_pos"]   = parts[1]
                raw["_ref"]   = parts[2]
                raw["_alt"]   = parts[3]
            else:
                logger.error(
                    "LOVDConnector: variant_id column does not have chrom:pos:ref:alt format."
                )
                return
        else:
            # Fall back to individual columns if variant_id not present
            for col in ("chrom", "pos", "ref", "alt"):
                if col not in raw.columns:
                    logger.error(
                        "LOVDConnector: required column '%s' missing from parquet.", col
                    )
                    return
            raw["_chrom"] = raw["chrom"].astype(str).str.lstrip("chr")
            raw["_pos"]   = raw["pos"].astype(str)
            raw["_ref"]   = raw["ref"].astype(str)
            raw["_alt"]   = raw["alt"].astype(str)

        # Map classification_raw → ordinal score
        raw["lovd_variant_class"] = (
            raw["classification_raw"]
            .str.lower()
            .str.strip()
            .map(_CLASSIFICATION_MAP)
            .fillna(_DEFAULT_CLASS)
            .astype(int)
        )

        # Keep only the join key + class score; deduplicate
        # Where there are conflicting LOVD classifications for the same
        # variant (multiple submitters), take the maximum (most severe)
        self._lookup = (
            raw[["_chrom", "_pos", "_ref", "_alt", "lovd_variant_class"]]
            .groupby(["_chrom", "_pos", "_ref", "_alt"], as_index=False)["lovd_variant_class"]
            .max()
        )

        logger.info(
            "LOVDConnector: loaded %d variants from %s.",
            len(self._lookup), self._parquet_path,
        )