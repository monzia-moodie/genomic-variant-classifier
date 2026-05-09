"""
src/data/cadd.py
=================
CADD (Combined Annotation Dependent Depletion) REST API connector.
Phase 2, Pillar 1, Connector 3.
 
Annotates variants with cadd_phred scores using the public CADD REST API
hosted at the University of Washington / BIH Charite.
 
  API base URL:  https://cadd.gs.washington.edu/api/v1.0/
  Version used:  GRCh38-v1.7  (latest as of 2024)
  Endpoint:      GRCh38-v1.7/<chrom>:<pos>_<ref>_<alt>
  Response:      [{"Chrom":"5","Pos":"2003402","Ref":"C","Alt":"A",
                   "RawScore":"-0.251","PHRED":"0.850"}]
 
IMPORTANT RATE LIMIT:
  The CADD API is explicitly "experimental" and not designed for bulk
  queries. The server documentation and community guidance both require
  >= 1.5 seconds between requests. This connector enforces a 1.5-second
  sleep by default. Do NOT reduce it.
  For > 1000 variants, use the pre-computed download files instead:
  https://cadd.gs.washington.edu/download
 
ANNOTATOR pattern (same as SpliceAIConnector):
  annotated_df = connector.fetch(variant_df=canonical_df)
  The returned DataFrame has a cadd_phred column added.
  Variants not found in the API or where the API fails receive the
  population-median default of 15.0.
 
cadd_phred is already in TABULAR_FEATURES in variant_ensemble.py with a
median default fill. Once this connector annotates the variant DataFrame
before engineer_features() is called, real CADD scores are used
automatically -- engineer_features already handles real values via df.get()
which falls back to the default only for NaN entries.
 
PHASE_2_PLACEHOLDER: Bulk annotation via pre-computed download files.
  For pipeline runs over ClinVar (~500k variants), the REST API is too
  slow (~8 days at 1 req/sec). Implement file-based lookup using
  https://cadd.gs.washington.edu/download when bulk annotation is needed.
"""
 
from __future__ import annotations
 
import logging
from pathlib import Path
from typing import Optional
 
import pandas as pd
import requests
 
from src.data.database_connectors import (
    CANONICAL_COLUMNS,
    BaseConnector,
    FetchConfig,
)
 
logger = logging.getLogger(__name__)
 
# Latest GRCh38 CADD version as of 2024
CADD_VERSION      = "GRCh38-v1.7"
CADD_BASE_URL     = "https://cadd.gs.washington.edu/api/v1.0"
CADD_RATE_DELAY   = 1.5   # seconds — enforced minimum; do NOT reduce
CADD_MEDIAN_PHRED = 15.0  # population median used as fallback
 
 
class CADDConnector(BaseConnector):
    """
    Annotates variants with CADD PHRED scores via the UW REST API.
 
    Usage
    -----
        connector    = CADDConnector()
        annotated_df = connector.fetch(variant_df=canonical_df)
        # annotated_df now has a cadd_phred column
 
    Variants with API errors or no annotation receive cadd_phred = 15.0
    (the population median). engineer_features() in variant_ensemble.py
    will use these real values automatically.
 
    For large batches (> ~500 variants), the API will be slow. Consider
    implementing the file-based lookup (PHASE_2_PLACEHOLDER).
    """
 
    source_name = "cadd"
 
    def __init__(self, config: Optional[FetchConfig] = None) -> None:
        super().__init__(config)
        # Override the base class rate_limit_delay to enforce CADD's minimum
        self.config.rate_limit_delay = max(
            self.config.rate_limit_delay, CADD_RATE_DELAY
        )
 
    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
 
    def fetch(self, variant_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Add cadd_phred column to variant_df using the CADD REST API.
 
        Args:
            variant_df: Canonical variant DataFrame with chrom, pos, ref, alt.
 
        Returns:
            variant_df copy with cadd_phred column added.
            Variants not found or where the API errors get cadd_phred=15.0.
        """
        if variant_df.empty:
            df = variant_df.copy()
            df["cadd_phred"] = pd.Series(dtype=float)
            return df
 
        df = variant_df.copy()
 
        # Build lookup key for each variant
        df["_lookup_key"] = (
            df["chrom"].astype(str) + ":" +
            df["pos"].astype(str)   + "_" +
            df["ref"].astype(str)   + "_" +
            df["alt"].astype(str)
        )
 
        scores: dict[str, float] = {}
        unique_keys = df["_lookup_key"].unique()
        n_total     = len(unique_keys)
 
        logger.info(
            "CADD: annotating %d unique variants via REST API "
            "(~%.0f seconds at %.1fs per request)...",
            n_total, n_total * CADD_RATE_DELAY, CADD_RATE_DELAY,
        )
 
        for i, key in enumerate(unique_keys, 1):
            if i % 50 == 0:
                logger.info("CADD: %d / %d variants annotated...", i, n_total)
            scores[key] = self._fetch_one(key)
 
        df["cadd_phred"] = df["_lookup_key"].map(scores).fillna(CADD_MEDIAN_PHRED)
        df = df.drop(columns=["_lookup_key"])
 
        n_real   = (df["cadd_phred"] != CADD_MEDIAN_PHRED).sum()
        n_median = (df["cadd_phred"] == CADD_MEDIAN_PHRED).sum()
        logger.info(
            "CADD: %d / %d variants received real scores "
            "(%d used median fallback %.1f).",
            n_real, len(df), n_median, CADD_MEDIAN_PHRED,
        )
        return df
 
    # ------------------------------------------------------------------
    # Single-variant API call with caching
    # ------------------------------------------------------------------
 
    def _fetch_one(self, lookup_key: str) -> float:
        """
        Fetch CADD PHRED score for a single variant.
 
        lookup_key format: "<chrom>:<pos>_<ref>_<alt>"
        Returns CADD_MEDIAN_PHRED (15.0) if the API fails or returns nothing.
        """
        cache_key = lookup_key.replace(":", "_")
        cached    = self._load_cache(cache_key)
        if cached is not None and not cached.empty:
            return float(cached.iloc[0]["cadd_phred"])
 
        url   = f"{CADD_BASE_URL}/{CADD_VERSION}/{lookup_key}"
        score = self._call_api(url, lookup_key)
 
        # Cache the result (even median fallbacks, to avoid re-querying)
        self._save_cache(
            cache_key,
            pd.DataFrame([{"lookup_key": lookup_key, "cadd_phred": score}]),
        )
        return score
 
    def _call_api(self, url: str, lookup_key: str) -> float:
        """
        Call the CADD API and extract the PHRED score.
        Returns CADD_MEDIAN_PHRED on any error.
        """
        try:
            resp = self._get(url)
            data = resp.json()
        except requests.RequestException as exc:
            logger.warning("CADD API request failed for %s: %s", lookup_key, exc)
            return CADD_MEDIAN_PHRED
        except ValueError as exc:
            logger.warning("CADD API returned invalid JSON for %s: %s", lookup_key, exc)
            return CADD_MEDIAN_PHRED
 
        return self.parse_response(data, lookup_key)
 
    # ------------------------------------------------------------------
    # Public helper — exposed for testing
    # ------------------------------------------------------------------
 
    @staticmethod
    def parse_response(data: list, lookup_key: str = "") -> float:
        """
        Extract the PHRED score from a CADD API JSON response.
 
        The API returns a list of dicts, e.g.:
          [{"Chrom":"5","Pos":"2003402","Ref":"C","Alt":"A",
            "RawScore":"-0.251851","PHRED":"0.850"}]
 
        Returns CADD_MEDIAN_PHRED if the list is empty or PHRED is missing.
        """
        if not isinstance(data, list) or not data:
            if lookup_key:
                logger.debug("CADD: empty response for %s", lookup_key)
            return CADD_MEDIAN_PHRED
 
        # The single-variant endpoint returns one element; take the first
        record = data[0]
        if not isinstance(record, dict):
            return CADD_MEDIAN_PHRED
 
        phred_str = record.get("PHRED")
        if phred_str is None:
            logger.debug("CADD: no PHRED field in response for %s", lookup_key)
            return CADD_MEDIAN_PHRED
 
        try:
            return float(phred_str)
        except (ValueError, TypeError):
            logger.warning(
                "CADD: could not parse PHRED '%s' for %s", phred_str, lookup_key
            )
            return CADD_MEDIAN_PHRED