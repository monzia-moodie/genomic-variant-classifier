"""
src/data/vep.py
===============
VEP codon_position connector — Phase 4, Connector 1.

Derives codon_position (1, 2, or 3) from the protein_change (HGVSp) column
that is already present in the canonical variant DataFrame.  No external file
is required for basic operation.

Codon position formula:
    aa_pos extracted from HGVSp (e.g. p.Arg175His → 175)
    codon_position = (aa_pos - 1) % 3 + 1   → 1-indexed

Non-missense and non-coding variants (no parseable protein_change) get 0.

Optional VEP REST API mode (vep_rest=True):
    Queries https://rest.ensembl.org/vep/human/hgvs/... for live annotation.
    Rate limit: 200 req/s.  Stub only in this implementation — pass
    vep_rest=False (the default) for all production use.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from src.data.database_connectors import BaseConnector, FetchConfig

logger = logging.getLogger(__name__)

VEP_REST_URL = "https://rest.ensembl.org/vep/human/hgvs"
VEP_RATE_DELAY = 0.006   # 200 req/s


def _extract_codon_position(protein_change: str) -> int:
    """Extract codon position (1, 2, 3) from HGVSp string, 0 if not applicable."""
    if not protein_change or not isinstance(protein_change, str):
        return 0
    # Match amino acid position number in HGVSp (e.g. p.Arg175His → 175)
    match = re.search(r'(\d+)', protein_change)
    if match:
        aa_pos = int(match.group(1))
        return (aa_pos - 1) % 3 + 1  # 1-indexed codon position
    return 0


class VEPConnector(BaseConnector):
    """
    Annotates variants with codon_position derived from the protein_change column.

    Usage
    -----
        connector = VEPConnector()
        annotated_df = connector.annotate_dataframe(variant_df)
        # annotated_df now has a codon_position column (0, 1, 2, or 3)

    vep_rest=True stubs VEP REST API mode (not yet implemented; reserved for
    future integration when live VEP annotation is required for novel variants
    lacking an HGVSp string).
    """

    source_name = "vep"

    def __init__(
        self,
        file_path: Optional[str | Path] = None,
        config: Optional[FetchConfig] = None,
        vep_rest: bool = False,
    ) -> None:
        super().__init__(config)
        self.file_path = Path(file_path) if file_path is not None else None
        self.vep_rest  = vep_rest
        if vep_rest:
            logger.warning(
                "VEPConnector: vep_rest=True — REST API mode is not yet fully "
                "implemented.  codon_position will be derived from protein_change "
                "column only.  Live VEP queries are planned for Phase 5."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def annotate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add codon_position column to df.

        Derives codon position from protein_change column (HGVSp).
        Non-coding, non-missense, or absent protein_change → codon_position = 0.

        Parameters
        ----------
        df : pd.DataFrame
            Variant DataFrame; must contain 'protein_change' for non-zero values.

        Returns
        -------
        pd.DataFrame with codon_position column added.
        """
        if df.empty:
            result = df.copy()
            result["codon_position"] = pd.Series(dtype=int)
            return result

        result = df.copy()
        protein_change = result.get(
            "protein_change",
            pd.Series([""] * len(result), index=result.index),
        ).fillna("")

        result["codon_position"] = protein_change.map(_extract_codon_position).astype(int)

        n_nonzero = (result["codon_position"] > 0).sum()
        logger.debug(
            "VEPConnector: %d / %d variants have non-zero codon_position.",
            n_nonzero, len(result),
        )
        return result

    def fetch(self, variant_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Wraps annotate_dataframe for BaseConnector compatibility."""
        return self.annotate_dataframe(variant_df)
