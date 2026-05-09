"""
src/data/revel.py
=================
REVEL (Rare Exome Variant Ensemble Learner) connector.
 
REVEL is a pre-computed ensemble pathogenicity score in [0, 1] for all
possible human missense SNVs, trained on 13 individual tools (SIFT, PolyPhen,
MutationAssessor, FATHMM, MetaSVM, ...).  Higher = more likely pathogenic.
 
    Score ≥ 0.5   broadly damaging  (ACMG PP3/BP4 supporting)
    Score ≥ 0.75  strong evidence    (commonly used in clinical variant review)
 
Data source — manual download required (~1.3 GB gzip):
    https://sites.google.com/site/revelgenomics/downloads
    Recommended file: revel-v1.3_all_chromosomes.csv.gz
    Alternative:      revel_with_transcript_ids.csv.zip
 
Raw file format (comma-delimited, has header row):
    chr,hg19_pos,grch38_pos,ref,alt,aaref,aaalt,REVEL[,transcript,...]
 
    • Chromosome values have no "chr" prefix (e.g. "1", "X")
    • One row per (chrom, grch38_pos, ref, alt) combination
    • Only missense SNVs are present; indels and synonymous variants get
      DEFAULT_SCORE on lookup
 
Lookup strategy:
    On first use the connector parses the raw file in chunks, extracts the
    four key columns (chr, grch38_pos, ref, alt, REVEL), writes them to a
    compact parquet cache, then builds an in-memory index:
 
        _index: dict[(chrom: str, pos: int, ref: str, alt: str), float]
 
    The parquet cache (~200 MB) is re-used on subsequent runs, making cold
    start ~30 s → warm start ~3 s.
 
Public interface:
    annotate_dataframe(df, coord="grch38", missing_value=DEFAULT_SCORE)
        → pd.DataFrame  (copy of df with "revel_score" column added/replaced)
 
    get_score(chrom, pos, ref, alt) → float
 
Phase 2 feature delivered:
    revel_score   — REVEL pathogenicity score [0, 1]
 
CHANGES:
    Initial implementation for Phase 2, Connector 4.
"""
 
from __future__ import annotations
 
import logging
from pathlib import Path
from typing import Optional
 
import pandas as pd
 
logger = logging.getLogger(__name__)
 
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
 
DEFAULT_SCORE: float = 0.5          # population-median fill for missing variants
CHUNK_SIZE: int = 500_000           # rows per chunk when parsing the raw CSV
 
# Column names expected in the REVEL flat file
_REVEL_FILE_COLS = {
    "chr":         str,
    "grch38_pos":  "Int64",   # nullable int — some rows have blank hg38 pos
    "ref":         str,
    "alt":         str,
    "REVEL":       float,
}
 
 
# ---------------------------------------------------------------------------
# Normalisation helpers (shared with other connectors)
# ---------------------------------------------------------------------------
 
def _normalise_chrom(chrom: str) -> str:
    """Strip 'chr' prefix and upper-case; 'chrM' → 'MT'."""
    c = str(chrom).strip()
    if c.upper().startswith("CHR"):
        c = c[3:]
    if c.upper() == "M":
        c = "MT"
    return c.upper() if c in ("X", "Y", "MT") else c
 
 
# ---------------------------------------------------------------------------
# REVELConnector
# ---------------------------------------------------------------------------
 
class REVELConnector:
    """
    File-based connector for REVEL pre-computed missense scores.
 
    Parameters
    ----------
    revel_file:
        Path to the raw REVEL CSV or gzip/zip file.  If *None*, the connector
        operates in stub mode — every lookup returns DEFAULT_SCORE and a
        warning is emitted.  This allows the rest of the pipeline to run
        without the large data file present during development.
    cache_dir:
        Directory for the parquet index cache.  Defaults to the directory
        containing *revel_file*.  The cache file is named
        ``revel_grch38_index.parquet``.
    """
 
    source_name = "revel"
 
    def __init__(
        self,
        revel_file: Optional[str | Path] = None,
        cache_dir: Optional[str | Path] = None,
    ) -> None:
        self._path: Optional[Path] = Path(revel_file) if revel_file else None
        self._cache_dir: Optional[Path] = (
            Path(cache_dir) if cache_dir
            else (self._path.parent if self._path else None)
        )
        self._index: Optional[dict[tuple[str, int, str, str], float]] = None
 
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
 
    def annotate_dataframe(
        self,
        df: pd.DataFrame,
        missing_value: float = DEFAULT_SCORE,
    ) -> pd.DataFrame:
        """
        Add (or replace) a ``revel_score`` column on a copy of *df*.
 
        The input DataFrame must have columns ``chrom``, ``pos``, ``ref``,
        ``alt`` (canonical schema).  Rows that have no REVEL entry — indels,
        synonymous variants, or variants outside the REVEL file — receive
        *missing_value* (default 0.5).
 
        Parameters
        ----------
        df:
            Canonical-schema DataFrame (from database_connectors.py).
        missing_value:
            Score assigned to variants absent from the REVEL index.
 
        Returns
        -------
        pd.DataFrame
            A copy of *df* with ``revel_score`` appended / replaced.
        """
        out = df.copy()
        index = self._get_index()
 
        scores: list[float] = []
        for _, row in df.iterrows():
            chrom = _normalise_chrom(str(row.get("chrom", "")))
            try:
                pos = int(row.get("pos", -1))
            except (TypeError, ValueError):
                pos = -1
            ref = str(row.get("ref", "")).upper()
            alt = str(row.get("alt", "")).upper()
            scores.append(index.get((chrom, pos, ref, alt), missing_value))
 
        out["revel_score"] = scores
        logger.debug(
            "REVEL: annotated %d variants; %d had real scores.",
            len(df),
            sum(s != missing_value for s in scores),
        )
        return out
 
    def get_score(
        self,
        chrom: str,
        pos: int,
        ref: str,
        alt: str,
        missing_value: float = DEFAULT_SCORE,
    ) -> float:
        """
        Return the REVEL score for a single variant.
 
        Parameters
        ----------
        chrom:
            Chromosome string, with or without the 'chr' prefix.
        pos:
            GRCh38 genomic position (1-based).
        ref:
            Reference allele (upper or lower case accepted).
        alt:
            Alternate allele (upper or lower case accepted).
        missing_value:
            Value returned when the variant is absent from the index.
 
        Returns
        -------
        float
            REVEL score in [0, 1], or *missing_value* if not found.
        """
        index = self._get_index()
        key = (_normalise_chrom(chrom), int(pos), ref.upper(), alt.upper())
        return index.get(key, missing_value)
 
    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------
 
    def _get_index(self) -> dict[tuple[str, int, str, str], float]:
        """Return the in-memory index, building it if necessary."""
        if self._index is None:
            self._index = self._load_index()
        return self._index
 
    def _cache_path(self) -> Optional[Path]:
        if self._cache_dir is None:
            return None
        return self._cache_dir / "revel_grch38_index.parquet"
 
    def _load_index(self) -> dict[tuple[str, int, str, str], float]:
        """
        Build the ``(chrom, pos, ref, alt) → score`` index.
 
        Precedence:
          1. Parquet cache (fast, ~3 s).
          2. Raw REVEL CSV/gz/zip (slow, ~30–90 s; writes cache afterward).
          3. Empty dict (stub mode — file not present).
        """
        # 1. Parquet cache
        cache = self._cache_path()
        if cache is not None and cache.exists():
            logger.info("REVEL: loading index from parquet cache: %s", cache)
            cdf = pd.read_parquet(cache)
            return self._df_to_index(cdf)
 
        # 2. Raw file
        if self._path is None or not self._path.exists():
            logger.warning(
                "REVEL: no data file supplied or file not found (%s). "
                "All variants will receive DEFAULT_SCORE=%.1f. "
                "Download from https://sites.google.com/site/revelgenomics/downloads",
                self._path,
                DEFAULT_SCORE,
            )
            return {}
 
        logger.info("REVEL: parsing raw file %s (this may take ~30–90 s) …", self._path)
        chunks: list[pd.DataFrame] = []
 
        # Handle zip files — pandas read_csv supports zip but we need to
        # specify compression explicitly for .zip archives.
        compression: str | dict = "infer"
        suffix = "".join(self._path.suffixes).lower()
        if suffix.endswith(".zip"):
            compression = "zip"
 
        try:
            reader = pd.read_csv(
                self._path,
                compression=compression,
                usecols=["chr", "grch38_pos", "ref", "alt", "REVEL"],
                dtype={"chr": str, "ref": str, "alt": str, "REVEL": float},
                # grch38_pos may be empty for a tiny fraction of rows
                na_values=[".", "", "NA"],
                chunksize=CHUNK_SIZE,
                low_memory=False,
            )
            for chunk in reader:
                chunk = chunk.dropna(subset=["grch38_pos", "REVEL"])
                chunk["grch38_pos"] = chunk["grch38_pos"].astype("int64")
                chunk["chr"] = chunk["chr"].apply(_normalise_chrom)
                chunk["ref"] = chunk["ref"].str.upper()
                chunk["alt"] = chunk["alt"].str.upper()
                # Keep only SNVs (REVEL only has missense SNVs, but guard anyway)
                snv_mask = (chunk["ref"].str.len() == 1) & (chunk["alt"].str.len() == 1)
                chunks.append(chunk.loc[snv_mask, ["chr", "grch38_pos", "ref", "alt", "REVEL"]])
        except Exception as exc:
            logger.error("REVEL: failed to parse %s: %s", self._path, exc)
            return {}
 
        if not chunks:
            logger.warning("REVEL: no valid rows parsed from %s.", self._path)
            return {}
 
        full = pd.concat(chunks, ignore_index=True)
        full = full.rename(columns={"chr": "chrom", "grch38_pos": "pos", "REVEL": "revel_score"})
        full = full.drop_duplicates(subset=["chrom", "pos", "ref", "alt"])
        logger.info("REVEL: indexed %d variants.", len(full))
 
        # 3. Write parquet cache
        if cache is not None:
            try:
                cache.parent.mkdir(parents=True, exist_ok=True)
                full.to_parquet(cache, index=False)
                logger.info("REVEL: wrote parquet cache → %s", cache)
            except Exception as exc:
                logger.warning("REVEL: could not write cache (%s).", exc)
 
        return self._df_to_index(full)
 
    @staticmethod
    def _df_to_index(
        df: pd.DataFrame,
    ) -> dict[tuple[str, int, str, str], float]:
        """Convert a (chrom, pos, ref, alt, revel_score) DataFrame to a lookup dict."""
        return {
            (_normalise_chrom(row.chrom), int(row.pos), row.ref.upper(), row.alt.upper()): float(row.revel_score)
            for row in df.itertuples(index=False)
        }