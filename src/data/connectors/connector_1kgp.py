"""
1000 Genomes Project (1KGP) Population-Stratified Allele Frequency Connector
=============================================================================
Adds five population-specific allele frequencies to the variant feature matrix:

    af_1kg_afr   African
    af_1kg_eur   European
    af_1kg_eas   East Asian
    af_1kg_sas   South Asian
    af_1kg_amr   Admixed American

Source
------
1000 Genomes Project 30x high-coverage phase (GRCh38):
  https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/
  1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV/
  1kGP_high_coverage_Illumina.sites.vcf.gz   (~300 MB gzipped)

Download
--------
    Invoke-WebRequest `
        -Uri "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/\\
              1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV/\\
              1kGP_high_coverage_Illumina.sites.vcf.gz" `
        -OutFile data\\external\\1kgp\\1kGP_high_coverage_Illumina.sites.vcf.gz

Motivation
----------
The phase-4 model shows a 0.166-point AUROC drop on temporal holdout variants
(0.9847 in-distribution → 0.8191 on 2023+ variants).  A significant fraction of
that gap is caused by variants that are rare in EUR gnomAD populations but common
in AFR/SAS/EAS populations — or vice versa.  The current feature matrix has only
one allele frequency column (`allele_freq` from gnomAD total AF).  Adding
population-stratified AFs allows the model to distinguish:

  - Rare globally, common in one population → likely population-specific, not
    inherently pathogenic
  - Rare in all populations → stronger prior towards pathogenicity
  - AF differences across populations → informative of ancestry-specific risk

This directly addresses the ACMG PM2 and BS1 criteria which are population-specific
by definition.

Usage
-----
    from src.data.connectors.connector_1kgp import KGPConnector

    connector = KGPConnector(vcf_path="data/external/1kgp/1kGP_high_coverage_Illumina.sites.vcf.gz")
    df_annotated = connector.annotate(variant_df)
    # Adds columns: af_1kg_afr, af_1kg_eur, af_1kg_eas, af_1kg_sas, af_1kg_amr

Integration with engineer_features()
-------------------------------------
Add to TABULAR_FEATURES in src/api/schemas.py:
    "af_1kg_afr", "af_1kg_eur", "af_1kg_eas", "af_1kg_sas", "af_1kg_amr"

Add to engineer_features() in src/api/pipeline.py:
    for col in KGPConnector.POPULATION_COLS:
        row[col] = float(request_dict.get(col) or 0.0)

Add to VariantRequest in src/api/schemas.py:
    af_1kg_afr: float | None = None
    af_1kg_eur: float | None = None
    af_1kg_eas: float | None = None
    af_1kg_sas: float | None = None
    af_1kg_amr: float | None = None
"""

from __future__ import annotations

import gzip
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Population codes as they appear in the 1KGP VCF INFO field
_VCF_INFO_KEYS: dict[str, str] = {
    "AF_afr": "af_1kg_afr",
    "AF_eur": "af_1kg_eur",
    "AF_eas": "af_1kg_eas",
    "AF_sas": "af_1kg_sas",
    "AF_amr": "af_1kg_amr",
}

# Default path — override at construction time or via AnnotationConfig
DEFAULT_VCF_PATH = Path("data/external/1kgp/1kGP_high_coverage_Illumina.sites.vcf.gz")

# Parquet cache stores the parsed VCF as a flat table for fast reloads
_CACHE_SUFFIX = ".1kgp_index.parquet"

# Fill value for variants absent from 1KGP (treated as globally ultra-rare)
MISSING_AF_DEFAULT: float = 0.0


# ---------------------------------------------------------------------------
# Scores dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KGPScores:
    """Population-stratified allele frequencies for a single variant."""
    af_1kg_afr: float = MISSING_AF_DEFAULT
    af_1kg_eur: float = MISSING_AF_DEFAULT
    af_1kg_eas: float = MISSING_AF_DEFAULT
    af_1kg_sas: float = MISSING_AF_DEFAULT
    af_1kg_amr: float = MISSING_AF_DEFAULT

    def as_dict(self) -> dict[str, float]:
        return {
            "af_1kg_afr": self.af_1kg_afr,
            "af_1kg_eur": self.af_1kg_eur,
            "af_1kg_eas": self.af_1kg_eas,
            "af_1kg_sas": self.af_1kg_sas,
            "af_1kg_amr": self.af_1kg_amr,
        }


# ---------------------------------------------------------------------------
# Connector
# ---------------------------------------------------------------------------

class KGPConnector:
    """
    File-based connector for 1000 Genomes Project population allele frequencies.

    The connector is lazy: the VCF index is built (or loaded from parquet cache)
    on the first call to annotate() or get_scores().  Subsequent calls reuse the
    in-memory index.

    Parameters
    ----------
    vcf_path:
        Path to the gzipped VCF sites file.  Pass None to run in stub mode
        (all scores return MISSING_AF_DEFAULT — useful for testing).
    cache_dir:
        Directory for the parquet index cache.  Defaults to the same directory
        as the VCF file.
    missing_af:
        Value returned for variants not present in 1KGP.
    """

    source_name = "1kgp"

    # Feature column names exposed to the rest of the pipeline
    POPULATION_COLS: list[str] = [
        "af_1kg_afr",
        "af_1kg_eur",
        "af_1kg_eas",
        "af_1kg_sas",
        "af_1kg_amr",
    ]

    def __init__(
        self,
        vcf_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
        missing_af: float = MISSING_AF_DEFAULT,
    ) -> None:
        self._vcf_path: Path | None = Path(vcf_path) if vcf_path else None
        self._missing_af = missing_af
        self._cache_dir: Path | None = (
            Path(cache_dir) if cache_dir
            else (self._vcf_path.parent if self._vcf_path else None)
        )
        # Lazy-loaded index: (chrom, pos, ref, alt) → KGPScores
        self._index: dict[tuple[str, int, str, str], KGPScores] | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_scores(
        self,
        chrom: str,
        pos: int,
        ref: str,
        alt: str,
    ) -> KGPScores:
        """
        Return population AFs for a single variant.

        Parameters
        ----------
        chrom : str
            Chromosome without 'chr' prefix (e.g. "1", "X", "MT").
        pos : int
            GRCh38 position.
        ref : str
            Reference allele (uppercase).
        alt : str
            Alternate allele (uppercase).

        Returns
        -------
        KGPScores
            Population AFs, or MISSING_AF_DEFAULT for each population if
            the variant is absent from 1KGP.
        """
        if self._vcf_path is None:
            return KGPScores()

        if not self._vcf_path.exists():
            logger.warning(
                "1KGP VCF not found at %s — returning default AFs.  "
                "Download from https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/"
                "data_collections/1000G_2504_high_coverage/working/"
                "20220422_3202_phased_SNV_INDEL_SV/"
                "1kGP_high_coverage_Illumina.sites.vcf.gz",
                self._vcf_path,
            )
            return KGPScores()

        self._ensure_index()
        key = (_norm_chrom(chrom), int(pos), ref.upper(), alt.upper())
        return self._index.get(key, KGPScores())  # type: ignore[union-attr]

    def annotate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add five population AF columns to a variant DataFrame.

        The input DataFrame must contain columns: chrom, pos, ref, alt.
        Returns a copy with the five af_1kg_* columns appended (or
        overwritten if already present).  Missing variants receive
        MISSING_AF_DEFAULT.  No NaNs are produced.

        Parameters
        ----------
        df : pd.DataFrame
            Variant DataFrame in canonical schema.

        Returns
        -------
        pd.DataFrame
            Copy of df with af_1kg_afr, af_1kg_eur, af_1kg_eas,
            af_1kg_sas, af_1kg_amr columns added.
        """
        if df.empty:
            out = df.copy()
            for col in self.POPULATION_COLS:
                out[col] = pd.Series(dtype=float)
            return out

        # Stub mode — vcf_path not provided
        if self._vcf_path is None:
            out = df.copy()
            for col in self.POPULATION_COLS:
                out[col] = self._missing_af
            return out

        if not self._vcf_path.exists():
            logger.warning(
                "1KGP VCF not found at %s — all population AFs set to %.1f",
                self._vcf_path, self._missing_af,
            )
            out = df.copy()
            for col in self.POPULATION_COLS:
                out[col] = self._missing_af
            return out

        self._ensure_index()
        out = df.copy()

        # Vectorised lookup via lookup key column
        keys = list(zip(
            out["chrom"].astype(str).apply(_norm_chrom),
            out["pos"].astype(int),
            out["ref"].astype(str).str.upper(),
            out["alt"].astype(str).str.upper(),
        ))

        for col in self.POPULATION_COLS:
            pop = col  # e.g. "af_1kg_afr"
            out[col] = [
                getattr(self._index.get(k, KGPScores()), pop)  # type: ignore[union-attr]
                for k in keys
            ]

        # Guarantee no NaNs
        for col in self.POPULATION_COLS:
            out[col] = out[col].fillna(self._missing_af).clip(lower=0.0, upper=1.0)

        return out

    def fetch(self, **kwargs) -> pd.DataFrame:
        """
        Return the full 1KGP index as a DataFrame (BaseConnector-compatible).

        Useful for exploratory analysis and for joining into the ETL pipeline.
        Triggers index build if not already loaded.
        """
        if self._vcf_path is None or not self._vcf_path.exists():
            logger.warning("1KGP VCF not available — fetch() returns empty DataFrame.")
            return pd.DataFrame(columns=["chrom", "pos", "ref", "alt"] + self.POPULATION_COLS)

        self._ensure_index()
        records = [
            {"chrom": k[0], "pos": k[1], "ref": k[2], "alt": k[3], **v.as_dict()}
            for k, v in self._index.items()  # type: ignore[union-attr]
        ]
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _ensure_index(self) -> None:
        """Build the in-memory lookup index, using parquet cache if available."""
        if self._index is not None:
            return

        cache_path = self._cache_path()
        if cache_path is not None and cache_path.exists():
            logger.info("Loading 1KGP parquet cache from %s", cache_path)
            self._index = _df_to_index(pd.read_parquet(cache_path))
            logger.info("1KGP index loaded: %d variants", len(self._index))
            return

        logger.info("Parsing 1KGP VCF: %s (this takes ~2 minutes)", self._vcf_path)
        df = _parse_vcf(self._vcf_path)  # type: ignore[arg-type]
        logger.info("Parsed %d variants from 1KGP VCF", len(df))

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path, index=False)
            logger.info("1KGP parquet cache written to %s", cache_path)

        self._index = _df_to_index(df)
        logger.info("1KGP index ready: %d variants", len(self._index))

    def _cache_path(self) -> Path | None:
        if self._vcf_path is None or self._cache_dir is None:
            return None
        stem = self._vcf_path.name.replace(".vcf.gz", "").replace(".vcf", "")
        return self._cache_dir / f"{stem}{_CACHE_SUFFIX}"


# ---------------------------------------------------------------------------
# VCF parsing helpers
# ---------------------------------------------------------------------------

def _norm_chrom(chrom: str) -> str:
    """Normalise chromosome string: strip 'chr' prefix, map 'chrM' → 'MT'."""
    c = str(chrom).strip()
    if c.lower().startswith("chr"):
        c = c[3:]
    if c == "M":
        c = "MT"
    return c.upper() if c in ("X", "Y", "MT") else c


def _parse_info(info_str: str) -> dict[str, str]:
    """Parse a VCF INFO field string into a flat key→value dict."""
    result: dict[str, str] = {}
    for token in info_str.split(";"):
        if not token:
            continue
        if "=" in token:
            k, _, v = token.partition("=")
            result[k] = v
        else:
            result[token] = "true"
    return result


def _parse_vcf(vcf_path: Path) -> pd.DataFrame:
    """
    Stream-parse the 1KGP gzipped VCF, extracting population AFs.

    Only data lines (non-header) are processed.  Multi-allelic sites
    (comma-separated ALT) are expanded into one row per alternate allele,
    with each allele's AF extracted from the corresponding index in the
    comma-separated AF_* INFO values.

    Returns a DataFrame with columns:
        chrom, pos, ref, alt, af_1kg_afr, af_1kg_eur,
        af_1kg_eas, af_1kg_sas, af_1kg_amr
    """
    rows: list[dict] = []
    open_fn = gzip.open if str(vcf_path).endswith(".gz") else open

    with open_fn(vcf_path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#"):
                continue

            parts = line.rstrip("\n").split("\t", 8)
            if len(parts) < 8:
                continue

            chrom_raw, pos_raw, _id, ref, alt_field, _qual, _filt, info_str = (
                parts[0], parts[1], parts[2], parts[3],
                parts[4], parts[5], parts[6], parts[7],
            )

            chrom = _norm_chrom(chrom_raw)
            try:
                pos = int(pos_raw)
            except ValueError:
                continue

            alts = alt_field.split(",")
            info = _parse_info(info_str)

            # Extract per-population AF lists (may be comma-separated for multi-alt)
            pop_afs: dict[str, list[float]] = {}
            for vcf_key, col_name in _VCF_INFO_KEYS.items():
                raw = info.get(vcf_key, "")
                if not raw:
                    pop_afs[col_name] = [MISSING_AF_DEFAULT] * len(alts)
                    continue
                try:
                    values = [float(v) if v not in (".", "") else MISSING_AF_DEFAULT
                              for v in raw.split(",")]
                    # Pad or truncate to match number of alts
                    while len(values) < len(alts):
                        values.append(MISSING_AF_DEFAULT)
                    pop_afs[col_name] = values[:len(alts)]
                except ValueError:
                    pop_afs[col_name] = [MISSING_AF_DEFAULT] * len(alts)

            for i, alt in enumerate(alts):
                rows.append({
                    "chrom":       chrom,
                    "pos":         pos,
                    "ref":         ref.upper(),
                    "alt":         alt.upper(),
                    "af_1kg_afr":  pop_afs["af_1kg_afr"][i],
                    "af_1kg_eur":  pop_afs["af_1kg_eur"][i],
                    "af_1kg_eas":  pop_afs["af_1kg_eas"][i],
                    "af_1kg_sas":  pop_afs["af_1kg_sas"][i],
                    "af_1kg_amr":  pop_afs["af_1kg_amr"][i],
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Clamp AFs to [0, 1] — occasional upstream data issues
    for col in KGPConnector.POPULATION_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(MISSING_AF_DEFAULT)
        df[col] = df[col].clip(lower=0.0, upper=1.0)

    # Deduplicate: keep first occurrence of (chrom, pos, ref, alt)
    df = df.drop_duplicates(subset=["chrom", "pos", "ref", "alt"], keep="first")
    return df.reset_index(drop=True)


def _df_to_index(
    df: pd.DataFrame,
) -> dict[tuple[str, int, str, str], KGPScores]:
    """Convert a parsed VCF DataFrame into a fast dict lookup index."""
    index: dict[tuple[str, int, str, str], KGPScores] = {}
    for row in df.itertuples(index=False):
        key = (str(row.chrom), int(row.pos), str(row.ref), str(row.alt))
        index[key] = KGPScores(
            af_1kg_afr=float(row.af_1kg_afr),
            af_1kg_eur=float(row.af_1kg_eur),
            af_1kg_eas=float(row.af_1kg_eas),
            af_1kg_sas=float(row.af_1kg_sas),
            af_1kg_amr=float(row.af_1kg_amr),
        )
    return index


# ---------------------------------------------------------------------------
# engineer_features() integration helper
# ---------------------------------------------------------------------------

def engineer_kgp_features(
    row: dict,
    connector: KGPConnector | None = None,
) -> dict:
    """
    Extract 1KGP population AF features from a single variant request dict.

    Called inside engineer_features() in src/api/pipeline.py.  If the
    request already contains pre-computed af_1kg_* values (e.g. submitted
    by a client that looked them up), those values are passed through
    directly.  Otherwise the connector is queried.

    Parameters
    ----------
    row : dict
        Raw variant request dict (from VariantRequest.model_dump()).
    connector : KGPConnector | None
        Live connector.  If None, all columns default to 0.0.

    Returns
    -------
    dict
        The input dict with af_1kg_* keys added/updated.
    """
    for col in KGPConnector.POPULATION_COLS:
        precomputed = row.get(col)
        if precomputed is not None:
            try:
                row[col] = float(precomputed)
                continue
            except (TypeError, ValueError):
                pass

        if connector is not None:
            scores = connector.get_scores(
                chrom=row.get("chrom", "0"),
                pos=int(row.get("pos", 0)),
                ref=row.get("ref", "N"),
                alt=row.get("alt", "N"),
            )
            row[col] = getattr(scores, col)
        else:
            row[col] = MISSING_AF_DEFAULT

    return row
