"""
src/data/connectors/connector_gnomad_constraint.py
====================================================
gnomAD v4.1 gene constraint connector.

Adds four gene-level constraint metrics to the variant feature matrix:

    pli_score   Probability of loss-of-function intolerance (0–1).
                High pLI (>0.9) = gene cannot tolerate heterozygous LoF.
    loeuf       Loss-of-function observed/expected upper bound fraction.
                Low LOEUF (<0.35) = strongly constrained. Replaces pLI as
                the recommended constraint metric in gnomAD v4.
    syn_z       Synonymous variant Z-score. High = more constrained than
                expected even for silent variants.
    mis_z       Missense variant Z-score. High = missense-constrained gene.

Source
------
gnomAD v4.1 constraint metrics TSV (bgzipped):
  https://gnomad.broadinstitute.org/downloads#v4-constraint
  File: gnomad.v4.1.constraint_metrics.tsv.bgz  (~12 MB)

Download
--------
    Invoke-WebRequest `
        -Uri "https://storage.googleapis.com/gcp-public-data--gnomad/release/4.1/constraint/gnomad.v4.1.constraint_metrics.tsv.bgz" `
        -OutFile data\\external\\gnomad\\gnomad.v4.1.constraint_metrics.tsv.bgz

Stub mode
---------
When tsv_path is None or the file does not exist, all four features return
their population-median safe defaults (see CONSTRAINT_DEFAULTS).  The
pipeline never raises on missing data.

Usage
-----
    from src.data.connectors.connector_gnomad_constraint import GnomADConstraintConnector

    connector = GnomADConstraintConnector(
        tsv_path="data/external/gnomad/gnomad.v4.1.constraint_metrics.tsv.bgz"
    )
    df_annotated = connector.annotate_dataframe(variant_df)
    # Adds columns: pli_score, loeuf, syn_z, mis_z
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

# Column names in the gnomAD v4.1 constraint TSV
_COL_GENE  = "gene"
_COL_PLI   = "lof.pLI"
_COL_LOEUF = "lof.oe_ci.upper"
_COL_SYN_Z = "syn.z_score"
_COL_MIS_Z = "mis.z_score"

# Feature column names exposed to the rest of the pipeline
CONSTRAINT_COLS: list[str] = ["pli_score", "loeuf", "syn_z", "mis_z"]

# Safe defaults — population medians / uninformative values
CONSTRAINT_DEFAULTS: dict[str, float] = {
    "pli_score": 0.0,    # median pLI ~0 for most genes
    "loeuf":     1.0,    # unconstrained = LOEUF ~1.0
    "syn_z":     0.0,    # z-score median = 0
    "mis_z":     0.0,
}

DEFAULT_TSV_PATH = Path(
    "data/external/gnomad/gnomad.v4.1.constraint_metrics.tsv"
)
_CACHE_SUFFIX = ".constraint_index.parquet"


# ---------------------------------------------------------------------------
# Scores dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConstraintScores:
    """gnomAD constraint metrics for a single gene."""
    pli_score: float = CONSTRAINT_DEFAULTS["pli_score"]
    loeuf:     float = CONSTRAINT_DEFAULTS["loeuf"]
    syn_z:     float = CONSTRAINT_DEFAULTS["syn_z"]
    mis_z:     float = CONSTRAINT_DEFAULTS["mis_z"]

    def as_dict(self) -> dict[str, float]:
        return {
            "pli_score": self.pli_score,
            "loeuf":     self.loeuf,
            "syn_z":     self.syn_z,
            "mis_z":     self.mis_z,
        }


# ---------------------------------------------------------------------------
# Connector
# ---------------------------------------------------------------------------

class GnomADConstraintConnector:
    """
    Gene-level gnomAD v4.1 constraint connector.

    Lookup is by HGNC gene symbol.  The index is built once and cached in
    memory; a parquet sidecar is written next to the TSV for fast reloads.

    Parameters
    ----------
    tsv_path : path to the bgzipped gnomAD constraint TSV, or None for stub.
    cache_dir : directory for the parquet cache (defaults to tsv parent dir).
    """

    source_name = "gnomad_constraint"

    def __init__(
        self,
        tsv_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        self._tsv_path: Path | None = Path(tsv_path) if tsv_path else None
        self._cache_dir: Path | None = (
            Path(cache_dir) if cache_dir
            else (self._tsv_path.parent if self._tsv_path else None)
        )
        self._index: dict[str, ConstraintScores] | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_scores(self, gene_symbol: str) -> ConstraintScores:
        """Return constraint scores for a gene, or safe defaults."""
        if self._tsv_path is None:
            return ConstraintScores()
        if not self._tsv_path.exists():
            logger.warning(
                "gnomAD constraint TSV not found at %s — returning defaults. "
                "Download from https://gnomad.broadinstitute.org/downloads#v4-constraint",
                self._tsv_path,
            )
            return ConstraintScores()
        self._ensure_index()
        return self._index.get(str(gene_symbol).strip(), ConstraintScores())  # type: ignore[union-attr]

    def annotate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add pli_score, loeuf, syn_z, mis_z columns to df.

        Requires a gene_symbol column.  Variants with missing or unrecognised
        gene symbols receive CONSTRAINT_DEFAULTS.  Returns a copy.
        """
        df = df.copy()

        # Ensure output columns exist with defaults
        for col, default in CONSTRAINT_DEFAULTS.items():
            if col not in df.columns:
                df[col] = default

        if "gene_symbol" not in df.columns:
            logger.info(
                "gnomAD constraint: gene_symbol column absent — returning defaults."
            )
            return df

        if self._tsv_path is None:
            logger.debug("gnomAD constraint: stub mode — all scores are defaults.")
            return df

        if not self._tsv_path.exists():
            logger.warning(
                "gnomAD constraint TSV not found at %s — all scores are defaults.",
                self._tsv_path,
            )
            return df

        self._ensure_index()

        genes = df["gene_symbol"].astype(str).str.strip()
        for col in CONSTRAINT_COLS:
            df[col] = genes.map(
                lambda g, c=col: getattr(
                    self._index.get(g, ConstraintScores()), c  # type: ignore[union-attr]
                )
            ).fillna(CONSTRAINT_DEFAULTS[col])

        n_hit = int((df["pli_score"] != CONSTRAINT_DEFAULTS["pli_score"]).sum())
        logger.info(
            "gnomAD constraint: %d / %d variants matched a gene entry.",
            n_hit, len(df),
        )
        return df

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _ensure_index(self) -> None:
        if self._index is not None:
            return
        cache = self._cache_path()
        if cache is not None and cache.exists():
            logger.info("Loading gnomAD constraint cache from %s", cache)
            self._index = _df_to_index(pd.read_parquet(cache))
            logger.info("gnomAD constraint index: %d genes", len(self._index))
            return

        logger.info("Parsing gnomAD constraint TSV: %s", self._tsv_path)
        df = _parse_tsv(self._tsv_path)  # type: ignore[arg-type]
        logger.info("Parsed %d genes from gnomAD constraint TSV", len(df))

        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache, index=False)
            logger.info("gnomAD constraint cache written to %s", cache)

        self._index = _df_to_index(df)

    def _cache_path(self) -> Path | None:
        if self._tsv_path is None or self._cache_dir is None:
            return None
        stem = self._tsv_path.name.split(".tsv")[0]
        return self._cache_dir / f"{stem}{_CACHE_SUFFIX}"


# ---------------------------------------------------------------------------
# TSV parsing helpers
# ---------------------------------------------------------------------------

def _safe_float(value: str, default: float) -> float:
    try:
        f = float(value)
        return f if np.isfinite(f) else default
    except (ValueError, TypeError):
        return default


def _parse_tsv(tsv_path: Path) -> pd.DataFrame:
    """
    Parse the gnomAD v4.1 constraint TSV (bgzip or plain gzip or uncompressed).

    Returns a DataFrame with columns: gene, pli_score, loeuf, syn_z, mis_z.
    One row per canonical gene (MANE transcript preferred; deduped by gene).
    """
    open_fn = gzip.open if str(tsv_path).endswith((".gz", ".bgz")) else open
    rows: list[dict] = []
    header: list[str] = []

    with open_fn(tsv_path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not header:
                header = line.split("\t")
                continue
            parts = line.split("\t")
            if len(parts) < len(header):
                continue
            record = dict(zip(header, parts))
            gene = record.get(_COL_GENE, "").strip()
            if not gene:
                continue
            rows.append({
                "gene":      gene,
                "pli_score": _safe_float(record.get(_COL_PLI,   ""), CONSTRAINT_DEFAULTS["pli_score"]),
                "loeuf":     _safe_float(record.get(_COL_LOEUF,  ""), CONSTRAINT_DEFAULTS["loeuf"]),
                "syn_z":     _safe_float(record.get(_COL_SYN_Z,  ""), CONSTRAINT_DEFAULTS["syn_z"]),
                "mis_z":     _safe_float(record.get(_COL_MIS_Z,  ""), CONSTRAINT_DEFAULTS["mis_z"]),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Clamp to valid ranges
    df["pli_score"] = df["pli_score"].clip(0.0, 1.0)
    df["loeuf"]     = df["loeuf"].clip(0.0, 5.0)

    # Keep one row per gene symbol (first = MANE transcript in gnomAD ordering)
    df = df.drop_duplicates(subset=["gene"], keep="first")
    return df.reset_index(drop=True)


def _df_to_index(df: pd.DataFrame) -> dict[str, ConstraintScores]:
    index: dict[str, ConstraintScores] = {}
    for row in df.itertuples(index=False):
        index[str(row.gene)] = ConstraintScores(
            pli_score=float(row.pli_score),
            loeuf=float(row.loeuf),
            syn_z=float(row.syn_z),
            mis_z=float(row.mis_z),
        )
    return index