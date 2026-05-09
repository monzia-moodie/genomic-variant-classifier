"""
src/data/clingen.py
===================
ClinGen Gene Validity connector — Phase 4, Connector 3.

Reads the ClinGen Gene Validity CSV (downloadable free from
https://search.clinicalgenome.org/kb/gene-validity) and adds one gene-level
feature:

    clingen_validity_score   int (0-5)
        Definitive                   → 5
        Strong                       → 4
        Moderate                     → 3
        Limited                      → 2
        No Known Disease Relationship→ 1
        No evidence / absent         → 0

When a gene appears in multiple disease-gene pairs, the MAXIMUM score
across all pairs is used.

CSV columns (header row included, may have extra whitespace):
    GENE SYMBOL, DISEASE LABEL, MOI, SOP, CLASSIFICATION,
    ONLINE REPORT, GCEP, UUID

Gene-level join: left-join by gene_symbol (= "GENE SYMBOL" in the CSV).
Missing genes → clingen_validity_score = 0.

Stub mode:
    When csv_path is None or the file does not exist, all variants receive
    clingen_validity_score = 0 and a WARNING is logged.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.data.database_connectors import BaseConnector, FetchConfig

logger = logging.getLogger(__name__)

CLASSIFICATION_SCORE: dict[str, int] = {
    "definitive":                    5,
    "strong":                        4,
    "moderate":                      3,
    "limited":                       2,
    "no known disease relationship": 1,
}
DEFAULT_SCORE = 0


class ClinGenConnector(BaseConnector):
    """
    Annotates variants with the ClinGen Gene Validity classification score.

    Usage
    -----
        connector = ClinGenConnector(
            csv_path="data/external/ClinGen-Gene-Disease-Summary.csv"
        )
        annotated_df = connector.annotate_dataframe(variant_df)

    If csv_path is None or file is absent, stub mode applies:
    all variants receive clingen_validity_score = 0.
    """

    source_name = "clingen"

    def __init__(
        self,
        csv_path: Optional[str | Path] = None,
        config: Optional[FetchConfig] = None,
    ) -> None:
        super().__init__(config)
        self.csv_path: Optional[Path] = (
            Path(csv_path) if csv_path is not None else None
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def annotate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add clingen_validity_score column to df.

        Parameters
        ----------
        df : pd.DataFrame
            Variant DataFrame; must contain 'gene_symbol' for a real join.

        Returns
        -------
        pd.DataFrame with clingen_validity_score column added.
        """
        if df.empty:
            result = df.copy()
            result["clingen_validity_score"] = pd.Series(dtype=int)
            return result

        gene_table = self._get_gene_table()

        result = df.copy()
        if gene_table.empty:
            result["clingen_validity_score"] = DEFAULT_SCORE
            return result

        result = result.merge(
            gene_table,
            left_on="gene_symbol",
            right_on="gene_symbol",
            how="left",
        )
        result["clingen_validity_score"] = (
            result["clingen_validity_score"].fillna(DEFAULT_SCORE).astype(int)
        )

        n_annotated = (result["clingen_validity_score"] > 0).sum()
        logger.debug(
            "ClinGenConnector: %d / %d variants have clingen_validity_score > 0.",
            n_annotated, len(result),
        )
        return result

    def fetch(self, variant_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Wraps annotate_dataframe for BaseConnector compatibility."""
        return self.annotate_dataframe(variant_df)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_gene_table(self) -> pd.DataFrame:
        """Return a gene-level score DataFrame, or empty if unavailable."""
        if self.csv_path is None:
            logger.warning(
                "ClinGenConnector: csv_path not set — returning default values "
                "(clingen_validity_score=0).  "
                "Download from https://search.clinicalgenome.org/kb/gene-validity."
            )
            return pd.DataFrame(columns=["gene_symbol", "clingen_validity_score"])

        cache_key = "gene_scores"
        cached = self._load_cache(cache_key)
        if cached is not None and not cached.empty:
            logger.info(
                "ClinGenConnector: loaded gene scores from cache (%d genes).", len(cached)
            )
            return cached

        if not self.csv_path.exists():
            logger.warning(
                "ClinGenConnector: CSV not found at '%s' — returning default values.",
                self.csv_path,
            )
            return pd.DataFrame(columns=["gene_symbol", "clingen_validity_score"])

        gene_table = self._parse_csv(self.csv_path)
        if not gene_table.empty:
            self._save_cache(cache_key, gene_table)
            logger.info(
                "ClinGenConnector: parsed and cached %d genes.", len(gene_table)
            )
        return gene_table

    def _parse_csv(self, path: Path) -> pd.DataFrame:
        """
        Parse ClinGen Gene Validity CSV into a gene-level score DataFrame.

        Expected header (may include extra whitespace):
            GENE SYMBOL, DISEASE LABEL, MOI, SOP, CLASSIFICATION, ...
        """
        try:
            raw = pd.read_csv(path, comment="#", dtype=str)
        except OSError as exc:
            logger.error("ClinGenConnector: failed to read %s: %s", path, exc)
            return pd.DataFrame(columns=["gene_symbol", "clingen_validity_score"])

        # Normalise column names (strip whitespace)
        raw.columns = [c.strip() for c in raw.columns]

        gene_col  = "GENE SYMBOL"
        class_col = "CLASSIFICATION"

        if gene_col not in raw.columns or class_col not in raw.columns:
            logger.error(
                "ClinGenConnector: expected columns '%s' and '%s' not found in %s. "
                "Found: %s",
                gene_col, class_col, path, list(raw.columns),
            )
            return pd.DataFrame(columns=["gene_symbol", "clingen_validity_score"])

        raw = raw[[gene_col, class_col]].copy()
        raw.columns = ["gene_symbol", "classification"]
        raw["gene_symbol"]    = raw["gene_symbol"].str.strip()
        raw["classification"] = raw["classification"].str.strip().str.lower()

        raw["score"] = raw["classification"].map(
            lambda c: CLASSIFICATION_SCORE.get(c, DEFAULT_SCORE)
        )

        # Take maximum score per gene across all disease-gene pairs
        gene_table = (
            raw.groupby("gene_symbol")["score"]
            .max()
            .rename("clingen_validity_score")
            .reset_index()
        )

        logger.info(
            "ClinGenConnector: parsed %d gene-disease pairs → %d unique genes.",
            len(raw), len(gene_table),
        )
        return gene_table
