"""
src/data/eve.py
===============
EVE evolutionary model connector — Phase 4, Connector 5.

Reads EVE per-protein CSV files (or a merged parquet) and adds one
variant-level feature:

    eve_score   float (0-1)
        Higher = more pathogenic (EVE pathogenicity score).
        0.5 = not covered / ambiguous (EVE uses 0.5 as the uncertain midpoint).

EVE lookup:
    Variants are matched by gene_symbol + one-letter amino acid change
    (e.g. protein_change "p.Arg175His" → "R175H").

Constructor:
    eve_path can be:
    - A directory containing per-protein CSV files named <GENE>_HUMAN_*.csv
    - A merged parquet file with all proteins combined
    - None → stub mode (all variants receive eve_score=0.5)

Per-protein CSV columns (EVE format):
    mutations_protein_name   str   Protein name (e.g. "TP53_HUMAN")
    position                 int   Amino acid position (1-indexed)
    wt_aa                    str   Wild-type amino acid (one-letter)
    mt_aa                    str   Mutant amino acid (one-letter)
    EVE_scores_ASM           float EVE pathogenicity score
    EVE_classes_25_pct_retained  str  Classification

Default: eve_score = 0.5 (ambiguous / not covered).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from src.data.database_connectors import BaseConnector, FetchConfig

logger = logging.getLogger(__name__)

DEFAULT_SCORE = 0.5

# Standard one-letter amino acid codes for three-letter → one-letter mapping
_THREE_TO_ONE: dict[str, str] = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
    "Ter": "*", "Stop": "*",
}


def _hgvsp_to_eve_key(protein_change: str) -> Optional[str]:
    """
    Convert HGVSp string to EVE lookup key "<WT><pos><MT>".

    Examples:
        "p.Arg175His" → "R175H"
        "p.Gly12Val"  → "G12V"
        "p.Arg175*"   → None   (stop gained, not a missense)
        ""            → None
    """
    if not protein_change or not isinstance(protein_change, str):
        return None

    # Match pattern like p.Arg175His
    m = re.match(r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2}|\*)", protein_change)
    if m:
        wt3  = m.group(1)
        pos  = m.group(2)
        mt3  = m.group(3)
        wt1  = _THREE_TO_ONE.get(wt3)
        mt1  = _THREE_TO_ONE.get(mt3, mt3 if mt3 == "*" else None)
        if wt1 and mt1 and mt1 != "*":
            return f"{wt1}{pos}{mt1}"
        return None

    # Match single-letter format like p.R175H
    m = re.match(r"p\.([A-Z])(\d+)([A-Z])", protein_change)
    if m:
        return f"{m.group(1)}{m.group(2)}{m.group(3)}"

    return None


class EVEConnector(BaseConnector):
    """
    Annotates variants with EVE evolutionary model pathogenicity scores.

    Usage
    -----
        connector = EVEConnector(eve_path="data/external/eve/")
        annotated_df = connector.annotate_dataframe(variant_df)
        # annotated_df now has an eve_score column (default 0.5)

    eve_path may be a directory of per-protein CSV files or a merged parquet.
    If eve_path is None or absent, stub mode applies: all variants get 0.5.
    """

    source_name = "eve"

    def __init__(
        self,
        eve_path: Optional[str | Path] = None,
        config: Optional[FetchConfig] = None,
    ) -> None:
        super().__init__(config)
        self.eve_path: Optional[Path] = (
            Path(eve_path) if eve_path is not None else None
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def annotate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add eve_score column to df.

        Parameters
        ----------
        df : pd.DataFrame
            Variant DataFrame; must contain 'gene_symbol' and 'protein_change'.

        Returns
        -------
        pd.DataFrame with eve_score column added (default 0.5).
        """
        if df.empty:
            result = df.copy()
            result["eve_score"] = pd.Series(dtype=float)
            return result

        if self.eve_path is None:
            logger.warning(
                "EVEConnector: eve_path not set — returning eve_score=0.5 (not covered).  "
                "Download EVE scores from https://evemodel.org."
            )
            result = df.copy()
            result["eve_score"] = DEFAULT_SCORE
            return result

        lookup = self._get_lookup()
        if lookup.empty:
            result = df.copy()
            result["eve_score"] = DEFAULT_SCORE
            return result

        return self._annotate(df, lookup)

    def fetch(self, variant_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Wraps annotate_dataframe for BaseConnector compatibility."""
        return self.annotate_dataframe(variant_df)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_lookup(self) -> pd.DataFrame:
        """Return EVE lookup DataFrame (gene_symbol + aa_change → eve_score)."""
        cache_key = "eve_lookup"
        cached = self._load_cache(cache_key)
        if cached is not None and not cached.empty:
            logger.info("EVEConnector: loaded %d EVE scores from cache.", len(cached))
            return cached

        if not self.eve_path.exists():
            logger.warning(
                "EVEConnector: eve_path '%s' does not exist — returning eve_score=0.5.",
                self.eve_path,
            )
            return pd.DataFrame(columns=["gene_symbol", "aa_change", "eve_score"])

        # Determine whether eve_path is a directory of CSVs or a merged parquet
        if self.eve_path.is_dir():
            lookup = self._parse_csv_directory(self.eve_path)
        elif self.eve_path.suffix in (".parquet", ".pq"):
            lookup = self._parse_merged_parquet(self.eve_path)
        else:
            logger.warning(
                "EVEConnector: eve_path '%s' is not a directory or parquet — "
                "returning eve_score=0.5.",
                self.eve_path,
            )
            return pd.DataFrame(columns=["gene_symbol", "aa_change", "eve_score"])

        if not lookup.empty:
            self._save_cache(cache_key, lookup)
            logger.info("EVEConnector: cached %d EVE scores.", len(lookup))
        return lookup

    def _parse_csv_directory(self, directory: Path) -> pd.DataFrame:
        """Parse a directory of per-protein EVE CSV files."""
        csv_files = list(directory.glob("*.csv"))
        if not csv_files:
            logger.warning(
                "EVEConnector: no CSV files found in directory '%s'.", directory
            )
            return pd.DataFrame(columns=["gene_symbol", "aa_change", "eve_score"])

        parts = []
        for csv_file in csv_files:
            try:
                part = self._parse_single_csv(csv_file)
                if not part.empty:
                    parts.append(part)
            except Exception as exc:
                logger.warning(
                    "EVEConnector: failed to parse %s: %s", csv_file.name, exc
                )

        if not parts:
            return pd.DataFrame(columns=["gene_symbol", "aa_change", "eve_score"])

        combined = pd.concat(parts, ignore_index=True)
        logger.info(
            "EVEConnector: parsed %d CSVs → %d EVE scores.", len(csv_files), len(combined)
        )
        return combined

    def _parse_single_csv(self, csv_file: Path) -> pd.DataFrame:
        """Parse a single per-protein EVE CSV file."""
        raw = pd.read_csv(csv_file, dtype=str)
        raw.columns = [c.strip() for c in raw.columns]

        required = {"mutations_protein_name", "position", "wt_aa", "mt_aa", "EVE_scores_ASM"}
        if not required.issubset(raw.columns):
            logger.warning(
                "EVEConnector: %s missing required columns %s (found: %s).",
                csv_file.name, required - set(raw.columns), list(raw.columns),
            )
            return pd.DataFrame(columns=["gene_symbol", "aa_change", "eve_score"])

        raw["position"]       = pd.to_numeric(raw["position"], errors="coerce")
        raw["EVE_scores_ASM"] = pd.to_numeric(raw["EVE_scores_ASM"], errors="coerce")
        raw = raw.dropna(subset=["position", "EVE_scores_ASM"])

        # Extract gene symbol from protein name (e.g. "TP53_HUMAN" → "TP53")
        raw["gene_symbol"] = raw["mutations_protein_name"].str.split("_").str[0]

        raw["aa_change"] = (
            raw["wt_aa"].str.strip() +
            raw["position"].astype(int).astype(str) +
            raw["mt_aa"].str.strip()
        )
        raw["eve_score"] = raw["EVE_scores_ASM"].clip(0.0, 1.0)

        return raw[["gene_symbol", "aa_change", "eve_score"]].copy()

    def _parse_merged_parquet(self, parquet_path: Path) -> pd.DataFrame:
        """Parse a merged EVE parquet file."""
        try:
            raw = pd.read_parquet(parquet_path)
        except Exception as exc:
            logger.error("EVEConnector: failed to read parquet %s: %s", parquet_path, exc)
            return pd.DataFrame(columns=["gene_symbol", "aa_change", "eve_score"])

        # Support both per-protein CSV column layout and pre-processed layout
        if "gene_symbol" in raw.columns and "aa_change" in raw.columns and "eve_score" in raw.columns:
            return raw[["gene_symbol", "aa_change", "eve_score"]].dropna().copy()

        if "mutations_protein_name" in raw.columns:
            # Treat as merged per-protein CSV
            tmp_path = parquet_path  # reuse the path reference
            raw["gene_symbol"] = raw["mutations_protein_name"].str.split("_").str[0]
            raw["position"]       = pd.to_numeric(raw.get("position", pd.Series()), errors="coerce")
            raw["EVE_scores_ASM"] = pd.to_numeric(raw.get("EVE_scores_ASM", pd.Series()), errors="coerce")
            raw = raw.dropna(subset=["position", "EVE_scores_ASM"])
            raw["aa_change"] = (
                raw["wt_aa"].str.strip() +
                raw["position"].astype(int).astype(str) +
                raw["mt_aa"].str.strip()
            )
            raw["eve_score"] = raw["EVE_scores_ASM"].clip(0.0, 1.0)
            return raw[["gene_symbol", "aa_change", "eve_score"]].copy()

        logger.error(
            "EVEConnector: parquet %s does not have expected columns.", parquet_path
        )
        return pd.DataFrame(columns=["gene_symbol", "aa_change", "eve_score"])

    def _annotate(self, variant_df: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
        """Left-join EVE scores onto variant_df by gene_symbol + aa_change."""
        result = variant_df.copy()

        # Derive aa_change from protein_change
        protein_change = result.get(
            "protein_change",
            pd.Series([""] * len(result), index=result.index),
        ).fillna("")
        result["_aa_change"] = protein_change.map(_hgvsp_to_eve_key)

        gene_symbol = result.get(
            "gene_symbol",
            pd.Series([""] * len(result), index=result.index),
        ).fillna("")
        result["_gene_symbol"] = gene_symbol

        # Only attempt to join for rows with a valid aa_change
        has_key = result["_aa_change"].notna()

        score_table = lookup.rename(
            columns={"gene_symbol": "_gene_symbol", "aa_change": "_aa_change"}
        )

        result = result.merge(
            score_table,
            on=["_gene_symbol", "_aa_change"],
            how="left",
        )

        # Rows without a valid key get the default
        result["eve_score"] = result["eve_score"].fillna(DEFAULT_SCORE).clip(0.0, 1.0)
        result = result.drop(columns=["_aa_change", "_gene_symbol"])

        n_covered = (result["eve_score"] != DEFAULT_SCORE).sum()
        logger.info(
            "EVEConnector: %d / %d variants covered by EVE (score != default).",
            n_covered, len(result),
        )
        return result
