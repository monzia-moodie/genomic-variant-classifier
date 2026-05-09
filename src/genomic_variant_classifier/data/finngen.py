from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FinnGen R10 population AF connector
# ---------------------------------------------------------------------------
# Data source: https://r10.finngen.fi/
# File: finngen_R10_annotated_variants_v1.gz  (or current release equiv.)
# Columns used: #chrom, pos, ref, alt, af_fin, af_nfsee, rsid
#
# Feature columns produced:
#   finngen_af_fin     — Finnish population allele frequency
#   finngen_af_nfsee   — Non-Finnish SEE allele frequency (comparison anchor)
#   finngen_enrichment — af_fin / (af_nfsee + 1e-9); Finnish enrichment ratio
#
# Used as third-tier AF fallback after gnomAD and 1KGP:
#   gnomAD AF → 1KGP AF → FinnGen AF → 0.0 default
# ---------------------------------------------------------------------------

FINNGEN_COLUMNS = [
    "finngen_af_fin",
    "finngen_af_nfsee",
    "finngen_enrichment",
]

_CHROM_NORMALISE = {str(i): str(i) for i in range(1, 23)}
_CHROM_NORMALISE.update({"X": "X", "Y": "Y", "MT": "MT", "M": "MT"})


def _normalise_chrom(c: str) -> str:
    c = str(c).replace("chr", "").upper()
    return _CHROM_NORMALISE.get(c, c)


class FinnGenConnector:
    """
    Annotates a variant DataFrame with FinnGen R10 population AF columns.

    Parameters
    ----------
    tsv_path:
        Path to the FinnGen R10 annotated variants TSV (gzipped or plain).
        Download from https://r10.finngen.fi/
        Expected columns: #chrom, pos, ref, alt, af_fin, af_nfsee
    chunksize:
        Rows per chunk when reading the large TSV. Default 500_000.
    """

    def __init__(
        self,
        tsv_path: Optional[str | Path] = None,
        chunksize: int = 500_000,
    ) -> None:
        self.tsv_path = Path(tsv_path) if tsv_path else None
        self.chunksize = chunksize
        self._index: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def annotate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add finngen_af_fin, finngen_af_nfsee, finngen_enrichment columns
        to *df* in-place and return it.

        Variants with no FinnGen match receive 0.0 / 0.0 / 1.0 defaults.
        """
        for col in FINNGEN_COLUMNS:
            if col not in df.columns:
                df[col] = 0.0

        if self.tsv_path is None or not Path(self.tsv_path).exists():
            logger.warning(
                "FinnGenConnector: tsv_path not set or file not found (%s). "
                "All variants will receive finngen_af_fin=0.0. "
                "Download from https://r10.finngen.fi/",
                self.tsv_path,
            )
            df["finngen_enrichment"] = 1.0
            return df

        if self._index is None:
            self._index = self._build_index(df)

        if self._index.empty:
            df["finngen_enrichment"] = 1.0
            return df

        # Join on chrom / pos / ref / alt
        query_keys = df[["chrom", "pos", "ref", "alt"]].copy()
        query_keys["chrom"] = query_keys["chrom"].astype(str).map(_normalise_chrom)

        merged = query_keys.merge(
            self._index,
            on=["chrom", "pos", "ref", "alt"],
            how="left",
        )

        df["finngen_af_fin"]   = merged["af_fin"].fillna(0.0).values
        df["finngen_af_nfsee"] = merged["af_nfsee"].fillna(0.0).values
        df["finngen_enrichment"] = (
            df["finngen_af_fin"] / (df["finngen_af_nfsee"] + 1e-9)
        ).clip(upper=1000.0)

        n_annotated = (df["finngen_af_fin"] > 0).sum()
        logger.info(
            "FinnGen annotation: %d / %d variants matched (%.1f%%).",
            n_annotated, len(df), 100 * n_annotated / max(len(df), 1),
        )
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Read only the rows from the FinnGen TSV that overlap with *df*'s
        chrom/pos range. Avoids loading the full ~20M row file into memory.
        """
        logger.info("FinnGen: building in-memory index from %s ...", self.tsv_path)

        # Compute query bounding box for early filtering
        query_chroms = set(
            df["chrom"].astype(str).map(_normalise_chrom).unique()
        )
        pos_min = int(df["pos"].min()) - 1
        pos_max = int(df["pos"].max()) + 1

        chunks = []
        compression = "gzip" if str(self.tsv_path).endswith(".gz") else "infer"

        try:
            reader = pd.read_csv(
                self.tsv_path,
                sep="\t",
                comment=None,
                chunksize=self.chunksize,
                compression=compression,
                usecols=["chr", "pos", "ref", "alt", "GENOME_AF_fin", "GENOME_AF_nfe"],
                dtype={"chr": str, "pos": int, "ref": str, "alt": str,
                       "GENOME_AF_fin": float, "GENOME_AF_nfe": float},
            )
            for chunk in reader:
                chunk.rename(columns={"chr": "chrom", "GENOME_AF_fin": "af_fin", "GENOME_AF_nfe": "af_nfsee"}, inplace=True)
                chunk["chrom"] = chunk["chrom"].map(_normalise_chrom)
                mask = (
                    chunk["chrom"].isin(query_chroms)
                    & chunk["pos"].between(pos_min, pos_max)
                )
                filtered = chunk[mask]
                if not filtered.empty:
                    chunks.append(filtered)

        except Exception as exc:
            logger.error("FinnGen: failed to read TSV: %s", exc)
            return pd.DataFrame(columns=["chrom", "pos", "ref", "alt",
                                         "af_fin", "af_nfsee"])

        if not chunks:
            logger.warning("FinnGen: no variants matched in TSV.")
            return pd.DataFrame(columns=["chrom", "pos", "ref", "alt",
                                         "af_fin", "af_nfsee"])

        index = pd.concat(chunks, ignore_index=True).drop_duplicates(
            subset=["chrom", "pos", "ref", "alt"]
        )
        logger.info("FinnGen index: %d unique variants loaded.", len(index))
        return index
