"""
src/data/phylop.py
==================
PhyloP evolutionary conservation score connector.
Phase 2, Pillar 1, Connector 5.

PhyloP (Phylogenetic P-values) measures conservation at individual genomic
positions by comparing observed versus expected substitution rates under a
neutral model across 100 vertebrate genomes (phyloP100way, GRCh38).

    Score > 0   conserved      (slower evolution than expected)
    Score = 0   neutral
    Score < 0   accelerated    (faster evolution than expected)
    Range:       approximately −30 to +30

Data source — manual download required (~9 GB BigWig, or pre-extracted TSV):
    BigWig:  https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP100way/
             hg38.phyloP100way.bw
    Pre-extracted TSV (recommended for this pipeline):
             Produce with bigWigToWig or extract via pyBigWig/pybigtools.

Two lookup modes
----------------
BigWig mode (preferred):
    Requires pyBigWig (pip install pyBigWig) or pybigtools.
    Set phylop_file to the .bw path.  Scores are fetched per-position at
    query time with no pre-loading — low memory, slower for large batches.

TSV / Parquet mode (fast bulk):
    Set phylop_file to a tab-delimited file with columns:
        chrom  pos  phylop_score
    (no header or with header — auto-detected by checking first token).
    On first use the connector builds an in-memory dict index and writes a
    parquet cache for fast subsequent loads.

Stub mode:
    phylop_file=None → every lookup returns DEFAULT_SCORE (0.0) without error.
    Useful during development when the large data file is not yet downloaded.

Public interface
----------------
    connector    = PhyloPConnector(phylop_file="path/to/file")
    annotated_df = connector.annotate_dataframe(canonical_df)
    score        = connector.get_score("17", 43071077)

Phase 2 feature delivered:
    phylop_score  — PhyloP100way conservation score (float, −30 … +30)

CHANGES:
    Initial implementation for Phase 2, Connector 5.
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

DEFAULT_SCORE: float = 0.0      # neutral — used for missing positions / stub mode
CHUNK_SIZE: int = 1_000_000     # rows per chunk when parsing flat TSV files

# ---------------------------------------------------------------------------
# Normalisation helper
# ---------------------------------------------------------------------------

def _normalise_chrom(chrom: str) -> str:
    """Strip 'chr' prefix; 'chrM' / 'M' → 'MT'; upper-case sex chromosomes."""
    c = str(chrom).strip()
    if c.upper().startswith("CHR"):
        c = c[3:]
    if c.upper() == "M":
        c = "MT"
    return c.upper() if c in ("X", "Y", "MT") else c


# ---------------------------------------------------------------------------
# PhyloPConnector
# ---------------------------------------------------------------------------

class PhyloPConnector:
    """
    Annotates variants with PhyloP100way conservation scores.

    Parameters
    ----------
    phylop_file:
        Path to the PhyloP data file.  Accepted formats:

        * ``*.bw`` / ``*.bigWig`` — UCSC BigWig (requires pyBigWig).
        * ``*.tsv`` / ``*.tsv.gz`` / ``*.txt`` — flat tab-delimited file with
          columns ``chrom``, ``pos``, ``phylop_score``.
        * ``*.parquet`` — pre-built index parquet (fastest warm start).

        Pass *None* to operate in stub mode (returns DEFAULT_SCORE for every
        variant without raising an error).

    cache_dir:
        Where to write the parquet index cache when parsing a flat file.
        Defaults to the directory of *phylop_file*.
    """

    source_name = "phylop"

    def __init__(
        self,
        phylop_file: Optional[str | Path] = None,
        cache_dir: Optional[str | Path] = None,
    ) -> None:
        self._path: Optional[Path] = Path(phylop_file) if phylop_file else None
        self._cache_dir: Optional[Path] = (
            Path(cache_dir) if cache_dir
            else (self._path.parent if self._path else None)
        )
        self._index: Optional[dict[tuple[str, int], float]] = None
        self._bw = None   # pyBigWig handle, opened lazily

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def annotate_dataframe(
        self,
        df: pd.DataFrame,
        missing_value: float = DEFAULT_SCORE,
    ) -> pd.DataFrame:
        """
        Add (or replace) a ``phylop_score`` column on a copy of *df*.

        The input DataFrame must have columns ``chrom`` and ``pos``
        (canonical schema).

        Parameters
        ----------
        df:
            Canonical-schema DataFrame.
        missing_value:
            Score for positions absent from the index (default 0.0).

        Returns
        -------
        pd.DataFrame
            Copy of *df* with ``phylop_score`` column appended / replaced.
        """
        out = df.copy()
        scores: list[float] = [
            self.get_score(
                str(row.get("chrom", "")),
                int(row.get("pos", 0)),
                missing_value=missing_value,
            )
            for _, row in df.iterrows()
        ]
        out["phylop_score"] = scores
        return out

    def get_score(
        self,
        chrom: str,
        pos: int,
        missing_value: float = DEFAULT_SCORE,
    ) -> float:
        """
        Return the PhyloP score for a single genomic position.

        Parameters
        ----------
        chrom:
            Chromosome (with or without 'chr' prefix).
        pos:
            1-based genomic position (GRCh38).
        missing_value:
            Returned when the position is not in the index.
        """
        chrom_norm = _normalise_chrom(chrom)

        # If index already populated (loaded or injected), use it directly
        if self._index is not None:
            return self._index.get((chrom_norm, int(pos)), missing_value)

        if self._path is None:
            return missing_value

        # BigWig path
        if self._path.suffix.lower() in (".bw", ".bigwig"):
            return self._query_bigwig(chrom_norm, pos, missing_value)

        # Flat-file / parquet path — build and cache index
        index = self._get_index()
        return index.get((chrom_norm, int(pos)), missing_value)

    # ------------------------------------------------------------------
    # BigWig lookup
    # ------------------------------------------------------------------

    def _open_bigwig(self):
        """Open the BigWig file, trying pyBigWig then pybigtools."""
        if self._bw is not None:
            return self._bw
        try:
            import pyBigWig
            self._bw = pyBigWig.open(str(self._path))
            self._bw_type = "pybigwig"
            return self._bw
        except ImportError:
            pass
        try:
            import pybigtools
            self._bw = pybigtools.open(str(self._path))
            self._bw_type = "pybigtools"
            return self._bw
        except ImportError:
            pass
        raise ImportError(
            "A BigWig reader is required for .bw files. "
            "Install one with: pip install pyBigWig"
        )

    def _query_bigwig(self, chrom: str, pos: int, missing_value: float) -> float:
        """Fetch a single position from the BigWig file (1-based pos → 0-based interval)."""
        try:
            bw = self._open_bigwig()
            # pyBigWig / pybigtools use 0-based half-open intervals
            chrom_bw = f"chr{chrom}" if not chrom.startswith("chr") else chrom
            if self._bw_type == "pybigwig":
                vals = bw.values(chrom_bw, pos - 1, pos)
                if vals and vals[0] is not None:
                    return float(vals[0])
            else:
                vals = list(bw.values(chrom_bw, pos - 1, pos))
                if vals and vals[0] is not None:
                    return float(vals[0])
        except Exception as exc:
            logger.debug("PhyloP BigWig query failed for %s:%d — %s", chrom, pos, exc)
        return missing_value

    # ------------------------------------------------------------------
    # Flat-file index (TSV / parquet)
    # ------------------------------------------------------------------

    def _get_index(self) -> dict[tuple[str, int], float]:
        """Return (building if necessary) the in-memory position → score index."""
        if self._index is not None:
            return self._index

        cache_path = self._cache_path()
        if cache_path and cache_path.exists():
            logger.info("PhyloP: loading index from parquet cache %s", cache_path)
            self._index = self._parquet_to_index(cache_path)
            return self._index

        logger.info("PhyloP: building index from %s (this may take a minute)...", self._path)
        self._index = self._build_index()

        if cache_path:
            self._save_cache(cache_path)

        return self._index

    def _build_index(self) -> dict[tuple[str, int], float]:
        """Parse the flat TSV / parquet file and return the index dict."""
        if self._path is None:
            return {}

        suffix = self._path.suffix.lower()
        if suffix == ".parquet":
            return self._parquet_to_index(self._path)

        # TSV / TSV.GZ — chunked read
        index: dict[tuple[str, int], float] = {}
        compression = "gzip" if suffix == ".gz" else "infer"
        first_chunk = True
        for chunk in pd.read_csv(
            self._path,
            sep="\t",
            header=None,
            names=["chrom", "pos", "phylop_score"],
            compression=compression,
            chunksize=CHUNK_SIZE,
            dtype={"chrom": str, "pos": "Int64", "phylop_score": float},
            on_bad_lines="skip",
        ):
            if first_chunk:
                # Drop header row if the file has one
                if str(chunk.iloc[0]["pos"]).lower() in ("pos", "position", "start"):
                    chunk = chunk.iloc[1:]
                first_chunk = False
            chunk = chunk.dropna(subset=["pos", "phylop_score"])
            for row in chunk.itertuples(index=False):
                chrom = _normalise_chrom(str(row.chrom))
                index[(chrom, int(row.pos))] = float(row.phylop_score)

        logger.info("PhyloP: index built with %d positions.", len(index))
        return index

    @staticmethod
    def _parquet_to_index(path: Path) -> dict[tuple[str, int], float]:
        df = pd.read_parquet(path, columns=["chrom", "pos", "phylop_score"])
        df["chrom"] = df["chrom"].apply(_normalise_chrom)
        return {
            (_normalise_chrom(str(r.chrom)), int(r.pos)): float(r.phylop_score)
            for r in df.itertuples(index=False)
        }

    def _cache_path(self) -> Optional[Path]:
        if self._cache_dir is None:
            return None
        return self._cache_dir / "phylop100way_index.parquet"

    def _save_cache(self, cache_path: Path) -> None:
        if not self._index:
            return
        try:
            rows = [
                {"chrom": chrom, "pos": pos, "phylop_score": score}
                for (chrom, pos), score in self._index.items()
            ]
            pd.DataFrame(rows).to_parquet(cache_path, index=False)
            logger.info("PhyloP: parquet cache written to %s", cache_path)
        except Exception as exc:
            logger.warning("PhyloP: could not write parquet cache: %s", exc)
