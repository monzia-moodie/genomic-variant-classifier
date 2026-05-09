"""
src/data/sift_polyphen.py
=========================
Connector 6: SIFT + PolyPhen-2 annotation from dbNSFP.

Both SIFT and PolyPhen-2 (HDIV model) are pre-computed for every possible
missense SNV in the human exome and distributed in the dbNSFP flat file —
the same source used by Connector 7 (full dbNSFP), which will supersede
this connector and additionally deliver REVEL, CADD, phyloP, and GERP in
one pass.

    SIFT score  : 0 → 1; values ≤ 0.05 are classified "deleterious" (PP3)
    PolyPhen-2  : 0 → 1; HDIV ≥ 0.908 = "probably damaging" (PP3)

Data source — manual download required:
    https://sites.google.com/site/jpopgen/dbNSFP
    Recommended: dbNSFP4.x_variant.chr*.gz files, concatenated.
    Academic / non-commercial use only (see licence on the download page).

Raw file format (tab-delimited, has header row):
    #chr  pos(1-based)  ref  alt  ...  SIFT_score  ...  Polyphen2_HDIV_score  ...

    • "#chr" column — no "chr" prefix (e.g. "1", "X")
    • Scores are semicolon-delimited when multiple transcripts are present
      (e.g. "0.04;.;0.12"). Dots represent missing transcript values.
    • Only missense SNVs are present; all other variant types receive
      DEFAULT_SIFT / DEFAULT_PP2 on lookup.

Multi-transcript aggregation:
    SIFT        → min across non-missing values (most deleterious transcript)
    PolyPhen-2  → max across non-missing values (most damaging transcript)

Lookup strategy (identical to REVELConnector):
    First call builds an in-memory dict index and writes a parquet cache.
    Subsequent calls load the parquet cache (~3 s vs ~60–120 s cold start).

Stub mode:
    Pass sift_polyphen_file=None to operate without the data file.
    Every lookup returns (DEFAULT_SIFT, DEFAULT_PP2) and a WARNING is logged.
    The pipeline continues normally; engineer_features() applies its own
    median fill for variants not found in the index.

Public interface:
    connector = SIFTPolyPhenConnector(sift_polyphen_file="path/to/dbNSFP.gz")
    df_out    = connector.annotate_dataframe(canonical_df)
    sift      = connector.get_sift_score("17", 43071077, "G", "T")
    pp2       = connector.get_pp2_score("17", 43071077, "G", "T")

Phase 2 features delivered:
    sift_score      — SIFT score in [0, 1]; already in TABULAR_FEATURES
    polyphen2_score — PolyPhen-2 HDIV score in [0, 1]; already in TABULAR_FEATURES

PHASE_2_PLACEHOLDER:
    PolyPhen-2 REST batch API (http://genetics.bwh.harvard.edu/pph2/bgi.do)
    is not yet implemented. Required for novel missense variants absent from
    dbNSFP (e.g. private germline variants, somatic variants in cancer).
    Add as a fallback path in Connector 7 or as a separate PhyloP-style
    BigWig-equivalent connector.

CHANGES:
    Initial implementation for Phase 2, Connector 6.
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

#: Default SIFT score for variants absent from dbNSFP (non-missense, novel).
#: Must match the fill-in in engineer_features() → score_defaults["sift_score"].
DEFAULT_SIFT: float = 0.5

#: Default PolyPhen-2 score for variants absent from dbNSFP.
#: Must match the fill-in in engineer_features() → score_defaults["polyphen2_score"].
DEFAULT_PP2: float = 0.5

#: Rows per chunk when parsing the raw dbNSFP flat file.
CHUNK_SIZE: int = 500_000

# dbNSFP v4.x column names (verify against file header for other versions)
_COL_CHROM = "#chr"
_COL_POS   = "pos(1-based)"
_COL_REF   = "ref"
_COL_ALT   = "alt"
_COL_SIFT  = "SIFT_score"
_COL_PP2   = "Polyphen2_HDIV_score"

# ACMG PP3/BP4 evidence thresholds (informational; not used in lookup logic)
SIFT_DELETERIOUS_THRESHOLD: float = 0.05
PP2_PROBABLY_DAMAGING_THRESHOLD: float = 0.908
PP2_POSSIBLY_DAMAGING_THRESHOLD: float = 0.446


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _normalise_chrom(chrom: str) -> str:
    """Strip 'chr' prefix; 'chrM'/'M' → 'MT'; upper-case sex chromosomes."""
    c = str(chrom).strip()
    if c.upper().startswith("CHR"):
        c = c[3:]
    if c.upper() == "M":
        c = "MT"
    return c.upper() if c in ("X", "Y", "MT") else c


def _parse_multival(raw: object, agg: str) -> Optional[float]:
    """
    Parse a semicolon-delimited dbNSFP score field and aggregate.

    Parameters
    ----------
    raw:
        Raw field value from the dbNSFP file, e.g. ``"0.04;.;0.12"`` or
        a float already parsed by pandas, or NaN / None.
    agg:
        ``"min"`` or ``"max"`` — aggregation applied across non-missing values.

    Returns
    -------
    float or None
        Aggregated score, or None if all transcript values are missing.
    """
    if raw is None:
        return None
    # pandas may have already parsed a single-value field as float
    try:
        f = float(raw)  # type: ignore[arg-type]
        return f if not pd.isna(f) else None
    except (TypeError, ValueError):
        pass
    # String with potential semicolons
    parts = str(raw).split(";")
    floats: list[float] = []
    for p in parts:
        p = p.strip()
        if p not in (".", "", "NA", "nan"):
            try:
                floats.append(float(p))
            except ValueError:
                continue
    if not floats:
        return None
    return min(floats) if agg == "min" else max(floats)


# ---------------------------------------------------------------------------
# SIFTPolyPhenConnector
# ---------------------------------------------------------------------------

class SIFTPolyPhenConnector:
    """
    File-based connector for SIFT and PolyPhen-2 pre-computed scores.

    Reads from a dbNSFP flat file (tab-delimited, GRCh38 coordinates).
    Behaviour mirrors REVELConnector exactly: stub mode when no file is
    supplied, parquet cache for fast warm starts, in-memory dict index.

    Parameters
    ----------
    sift_polyphen_file:
        Path to the dbNSFP file (plain text, gzip, or zip).  Pass *None*
        to run in stub mode (returns DEFAULT_SIFT / DEFAULT_PP2 for every
        variant without raising an error).
    cache_dir:
        Directory for the parquet index cache.  Defaults to the directory
        containing *sift_polyphen_file*.  Cache file is named
        ``dbnsfp_sift_pp2_index.parquet``.
    """

    source_name = "sift_polyphen"

    def __init__(
        self,
        sift_polyphen_file: Optional[str | Path] = None,
        cache_dir: Optional[str | Path] = None,
    ) -> None:
        self._path: Optional[Path] = (
            Path(sift_polyphen_file) if sift_polyphen_file else None
        )
        self._cache_dir: Optional[Path] = (
            Path(cache_dir) if cache_dir
            else (self._path.parent if self._path else None)
        )
        # Index: (chrom, pos, ref, alt) → (sift_score, pp2_score)
        self._index: Optional[dict[tuple[str, int, str, str], tuple[float, float]]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def annotate_dataframe(
        self,
        df: pd.DataFrame,
        sift_missing: float = DEFAULT_SIFT,
        pp2_missing: float = DEFAULT_PP2,
    ) -> pd.DataFrame:
        """
        Add (or replace) ``sift_score`` and ``polyphen2_score`` columns on a
        copy of *df*.

        The input DataFrame must have columns ``chrom``, ``pos``, ``ref``,
        ``alt`` (canonical schema).  Variants absent from the dbNSFP index —
        indels, synonymous SNVs, or novel variants — receive *sift_missing*
        and *pp2_missing* respectively.

        Parameters
        ----------
        df:
            Canonical-schema DataFrame.
        sift_missing:
            SIFT fill value for variants not in the index (default 0.5).
        pp2_missing:
            PolyPhen-2 fill value for variants not in the index (default 0.5).

        Returns
        -------
        pd.DataFrame
            A copy of *df* with ``sift_score`` and ``polyphen2_score``
            columns appended / replaced.
        """
        out = df.copy()
        index = self._get_index()

        sift_scores: list[float] = []
        pp2_scores: list[float] = []

        for _, row in df.iterrows():
            chrom = _normalise_chrom(str(row.get("chrom", "")))
            try:
                pos = int(row.get("pos", -1))
            except (TypeError, ValueError):
                pos = -1
            ref = str(row.get("ref", "")).upper()
            alt = str(row.get("alt", "")).upper()
            key = (chrom, pos, ref, alt)
            if key in index:
                s, p = index[key]
                sift_scores.append(s)
                pp2_scores.append(p)
            else:
                sift_scores.append(sift_missing)
                pp2_scores.append(pp2_missing)

        out["sift_score"] = sift_scores
        out["polyphen2_score"] = pp2_scores

        n_sift = sum(s != sift_missing for s in sift_scores)
        n_pp2  = sum(p != pp2_missing  for p in pp2_scores)
        logger.debug(
            "SIFTPolyPhen: annotated %d variants; "
            "%d had real SIFT scores, %d had real PolyPhen-2 scores.",
            len(df), n_sift, n_pp2,
        )
        return out

    def get_sift_score(
        self,
        chrom: str,
        pos: int,
        ref: str,
        alt: str,
        missing_value: float = DEFAULT_SIFT,
    ) -> float:
        """
        Return the SIFT score for a single variant.

        Parameters
        ----------
        chrom:
            Chromosome (with or without 'chr' prefix).
        pos:
            GRCh38 position (1-based).
        ref:
            Reference allele (case-insensitive).
        alt:
            Alternate allele (case-insensitive).
        missing_value:
            Returned when the variant is absent from the index.

        Returns
        -------
        float
            SIFT score in [0, 1], or *missing_value*.
        """
        key = (_normalise_chrom(chrom), int(pos), ref.upper(), alt.upper())
        entry = self._get_index().get(key)
        return entry[0] if entry is not None else missing_value

    def get_pp2_score(
        self,
        chrom: str,
        pos: int,
        ref: str,
        alt: str,
        missing_value: float = DEFAULT_PP2,
    ) -> float:
        """
        Return the PolyPhen-2 HDIV score for a single variant.

        Parameters
        ----------
        chrom:
            Chromosome (with or without 'chr' prefix).
        pos:
            GRCh38 position (1-based).
        ref:
            Reference allele (case-insensitive).
        alt:
            Alternate allele (case-insensitive).
        missing_value:
            Returned when the variant is absent from the index.

        Returns
        -------
        float
            PolyPhen-2 score in [0, 1], or *missing_value*.
        """
        key = (_normalise_chrom(chrom), int(pos), ref.upper(), alt.upper())
        entry = self._get_index().get(key)
        return entry[1] if entry is not None else missing_value

    # ------------------------------------------------------------------
    # Index management (mirrors REVELConnector._load_index)
    # ------------------------------------------------------------------

    def _get_index(
        self,
    ) -> dict[tuple[str, int, str, str], tuple[float, float]]:
        """Return the in-memory index, building it if necessary."""
        if self._index is None:
            self._index = self._load_index()
        return self._index

    def _cache_path(self) -> Optional[Path]:
        if self._cache_dir is None:
            return None
        return self._cache_dir / "dbnsfp_sift_pp2_index.parquet"

    def _load_index(
        self,
    ) -> dict[tuple[str, int, str, str], tuple[float, float]]:
        """
        Build the (chrom, pos, ref, alt) → (sift, pp2) index.

        Precedence:
          1. Parquet cache  (~3 s warm start).
          2. Raw dbNSFP flat file (~60–120 s; writes cache afterward).
          3. Empty dict (stub mode).
        """
        # 1. Parquet cache
        cache = self._cache_path()
        if cache is not None and cache.exists():
            logger.info(
                "SIFTPolyPhen: loading index from parquet cache: %s", cache
            )
            cdf = pd.read_parquet(cache)
            return self._df_to_index(cdf)

        # 2. Raw file
        if self._path is None or not self._path.exists():
            logger.warning(
                "SIFTPolyPhen: no data file supplied or file not found (%s). "
                "All variants will receive DEFAULT_SIFT=%.1f, DEFAULT_PP2=%.1f. "
                "Download dbNSFP from https://sites.google.com/site/jpopgen/dbNSFP "
                "and set the path via SIFTPolyPhenConnector(sift_polyphen_file=...).",
                self._path, DEFAULT_SIFT, DEFAULT_PP2,
            )
            return {}

        logger.info(
            "SIFTPolyPhen: parsing raw file %s "
            "(this may take 60–120 s) …", self._path,
        )

        compression: str = "infer"
        suffix = "".join(self._path.suffixes).lower()
        if suffix.endswith(".zip"):
            compression = "zip"
        elif suffix.endswith(".gz"):
            compression = "gzip"

        chunks: list[pd.DataFrame] = []
        try:
            reader = pd.read_csv(
                self._path,
                sep="\t",
                compression=compression,
                usecols=[_COL_CHROM, _COL_POS, _COL_REF, _COL_ALT,
                          _COL_SIFT, _COL_PP2],
                dtype=str,          # read everything as str; we parse floats
                na_values=[".", "", "NA"],
                chunksize=CHUNK_SIZE,
                low_memory=False,
            )
            for chunk in reader:
                # Rename the awkward "#chr" header
                chunk = chunk.rename(columns={_COL_CHROM: "chrom",
                                               _COL_POS:   "pos"})
                chunk = chunk.dropna(subset=["pos"])
                chunk["pos"] = pd.to_numeric(chunk["pos"], errors="coerce")
                chunk = chunk.dropna(subset=["pos"])
                chunk["pos"] = chunk["pos"].astype("int64")

                chunk["chrom"] = chunk["chrom"].apply(_normalise_chrom)
                chunk["ref"]   = chunk["ref"].str.upper()
                chunk["alt"]   = chunk["alt"].str.upper()

                # Only SNVs (dbNSFP should only contain SNVs, but guard anyway)
                snv = (chunk["ref"].str.len() == 1) & (chunk["alt"].str.len() == 1)
                chunk = chunk.loc[snv].copy()

                # Parse multi-transcript score fields
                chunk["sift_score"] = chunk[_COL_SIFT].apply(
                    lambda v: _parse_multival(v, "min")
                )
                chunk["polyphen2_score"] = chunk[_COL_PP2].apply(
                    lambda v: _parse_multival(v, "max")
                )

                # Keep only rows where at least one score is present
                has_score = (
                    chunk["sift_score"].notna() | chunk["polyphen2_score"].notna()
                )
                chunk = chunk.loc[has_score, [
                    "chrom", "pos", "ref", "alt",
                    "sift_score", "polyphen2_score",
                ]].copy()

                # Fill NaN scores with defaults so the index always holds floats
                chunk["sift_score"]      = chunk["sift_score"].fillna(DEFAULT_SIFT)
                chunk["polyphen2_score"] = chunk["polyphen2_score"].fillna(DEFAULT_PP2)

                chunks.append(chunk)

        except Exception as exc:
            logger.error(
                "SIFTPolyPhen: failed to parse %s: %s", self._path, exc
            )
            return {}

        if not chunks:
            logger.warning(
                "SIFTPolyPhen: no valid rows parsed from %s.", self._path
            )
            return {}

        full = pd.concat(chunks, ignore_index=True)
        full = full.drop_duplicates(subset=["chrom", "pos", "ref", "alt"])
        logger.info("SIFTPolyPhen: indexed %d variants.", len(full))

        # Write parquet cache
        if cache is not None:
            try:
                cache.parent.mkdir(parents=True, exist_ok=True)
                full.to_parquet(cache, index=False)
                logger.info("SIFTPolyPhen: wrote parquet cache → %s", cache)
            except Exception as exc:
                logger.warning(
                    "SIFTPolyPhen: could not write cache (%s).", exc
                )

        return self._df_to_index(full)

    @staticmethod
    def _df_to_index(
        df: pd.DataFrame,
    ) -> dict[tuple[str, int, str, str], tuple[float, float]]:
        """
        Convert a (chrom, pos, ref, alt, sift_score, polyphen2_score)
        DataFrame to the lookup dict.

        Keys are normalised: chrom stripped of 'chr' prefix, alleles
        upper-cased.  Values are (sift_score, polyphen2_score) tuples of
        Python floats.
        """
        return {
            (
                _normalise_chrom(row.chrom),
                int(row.pos),
                row.ref.upper(),
                row.alt.upper(),
            ): (float(row.sift_score), float(row.polyphen2_score))
            for row in df.itertuples(index=False)
        }