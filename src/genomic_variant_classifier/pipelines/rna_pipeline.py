"""
src/pipelines/rna_pipeline.py
==============================
RNA splice-isoform pipeline — Phase 6.1.

Adds four splice-context features for variants within 50 bp of exon boundaries.
These features are gated by consequence type at feature-engineering time:
only variants where ``is_splice = 1`` OR ``consequence_severity >= 7`` receive
non-default values.

Features added
--------------
  maxentscan_score      float  Maximum entropy splice-site strength score.
                               Positive = stronger site; negative = weaker.
                               Computed from the 9-mer (5′ donor) or 23-mer
                               (3′ acceptor) context window using the
                               Yeo & Burge 2004 MaxEntScan algorithm.
                               Default: 0.0 (non-splice or context unavailable).

  dist_to_splice_site   int    Distance in bp from the variant to the nearest
                               canonical splice donor or acceptor.
                               Default: 50 (at the 50-bp gating boundary).

  exon_number           int    Exon number from VEP exon/intron annotation
                               (e.g. "3/12" → 3).  0 = non-exonic / unknown.

  is_canonical_splice   int    1 if the variant falls in the canonical
                               GT-AG dinucleotide of a donor/acceptor site
                               (+1/+2 donor or -2/-1 acceptor).
                               0 otherwise.

Algorithm notes
---------------
The MaxEntScan implementation follows Yeo & Burge (2004), "Maximum Entropy
Modeling of Short Sequence Motifs with Applications to RNA Splicing Signals",
J Comput Biol 11:377–394.

  * 5′ donor scoring uses a 9-mer spanning positions -3 (exon) to +6 (intron).
  * 3′ acceptor scoring uses a 23-mer spanning positions -20 (intron) to +3 (exon).

The position-specific scoring matrices embedded here are simplified
approximations derived from the consensus frequencies in Table 1 of
Yeo & Burge 2004.  For production-grade scores, supply the original
``me2x5`` and ``me2x3`` scoring files from the MaxEntScan web server:
    http://genes.mit.edu/burgelab/maxent/download/

Stub mode
---------
When called without sequence context (no ``fasta_seq`` column), the pipeline
returns default values and logs a WARNING.  The training pipeline continues.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MaxEntScan scoring matrices
# ---------------------------------------------------------------------------
# These matrices encode log-odds of base frequencies at each position of
# a canonical human splice site, relative to background frequencies.
# Values are scaled so that a perfect consensus site scores ~10–12.
#
# 5′ donor consensus:   [exon(-3)][exon(-2)][exon(-1)] | GT [intron]AAGT
# Positions indexed 0–8 corresponding to -3, -2, -1, +1, +2, +3, +4, +5, +6

_DONOR_POSITION_SCORES: list[dict[str, float]] = [
    # pos -3 (exon): slight preference for A/C
    {"A": 0.29, "C": 0.35, "G": 0.20, "T": 0.16},
    # pos -2 (exon)
    {"A": 0.35, "C": 0.18, "G": 0.25, "T": 0.22},
    # pos -1 (exon): preference for G/A
    {"A": 0.34, "C": 0.14, "G": 0.37, "T": 0.15},
    # pos +1 (intron): near-invariant G  (canonical GT)
    {"A": 0.003, "C": 0.002, "G": 0.990, "T": 0.005},
    # pos +2 (intron): near-invariant T  (canonical GT)
    {"A": 0.003, "C": 0.002, "G": 0.005, "T": 0.990},
    # pos +3 (intron): strong A preference
    {"A": 0.76, "C": 0.07, "G": 0.09, "T": 0.08},
    # pos +4 (intron)
    {"A": 0.54, "C": 0.09, "G": 0.15, "T": 0.22},
    # pos +5 (intron)
    {"A": 0.28, "C": 0.18, "G": 0.27, "T": 0.27},
    # pos +6 (intron)
    {"A": 0.27, "C": 0.19, "G": 0.28, "T": 0.26},
]

# Background nucleotide frequencies (GRCh38 exome)
_BG = {"A": 0.27, "C": 0.23, "G": 0.23, "T": 0.27}

# 3′ acceptor (23-mer) — simplified: use per-position log-odds
# Full matrix is 23 positions × 4 bases.  We use a compressed representation
# that captures the key signals: polypyrimidine tract + AG invariant.
_ACCEPTOR_PYRIMIDINE_POSITIONS = set(range(0, 18))   # poly-Y tract approx
_ACCEPTOR_AG_POS = (20, 21)                           # canonical AG positions


def _score_donor(seq9: str) -> float:
    """
    Score a 9-mer splice donor site using a simplified MaxEntScan log-odds model.

    Parameters
    ----------
    seq9 : str
        9-nucleotide sequence: 3 exonic bases + 6 intronic bases.
        Example: "CAGGTAAGT"

    Returns
    -------
    float : log-odds score (higher = stronger donor site)
    """
    if len(seq9) != 9:
        return 0.0
    seq9 = seq9.upper()
    score = 0.0
    for i, base in enumerate(seq9):
        if base not in "ACGT":
            return 0.0
        freq = _DONOR_POSITION_SCORES[i].get(base, 0.001)
        bg   = _BG.get(base, 0.25)
        score += math.log2(freq / bg) if freq > 0 else -8.0
    return round(score, 4)


def _score_acceptor(seq23: str) -> float:
    """
    Score a 23-mer splice acceptor site using a simplified MaxEntScan model.

    Parameters
    ----------
    seq23 : str
        23-nucleotide sequence: 20 intronic bases + 3 exonic bases.
        The key signals are the polypyrimidine tract and invariant AG at +20/+21.

    Returns
    -------
    float : log-odds score (higher = stronger acceptor site)
    """
    if len(seq23) != 23:
        return 0.0
    seq23 = seq23.upper()
    score = 0.0

    # Score polypyrimidine tract (positions 0-17)
    for i in _ACCEPTOR_PYRIMIDINE_POSITIONS:
        base = seq23[i]
        if base not in "ACGT":
            continue
        if base in ("C", "T"):
            score += 0.30
        else:
            score -= 0.10

    # Strong invariant AG at positions 20-21
    ag_bases = seq23[20:22]
    if ag_bases == "AG":
        score += 5.0
    elif ag_bases[0] == "A":
        score += 0.5
    elif ag_bases[1] == "G":
        score += 0.5
    else:
        score -= 3.0

    return round(score, 4)


# ---------------------------------------------------------------------------
# Canonical splice site detection
# ---------------------------------------------------------------------------
def _is_canonical_position(
    dist_to_donor: Optional[int],
    dist_to_acceptor: Optional[int],
) -> int:
    """
    Return 1 if variant is at a canonical GT-AG dinucleotide position.

    Canonical positions:
      Donor:    +1, +2 (GT)
      Acceptor: -1, -2 (AG)
    """
    if dist_to_donor is not None and abs(dist_to_donor) in (1, 2):
        return 1
    if dist_to_acceptor is not None and abs(dist_to_acceptor) in (1, 2):
        return 1
    return 0


# ---------------------------------------------------------------------------
# Exon number extraction
# ---------------------------------------------------------------------------
def _parse_exon_number(exon_str: Optional[str]) -> int:
    """
    Parse exon number from VEP exon/intron annotation strings like '3/12'.

    Returns 0 if not parseable.
    """
    if not exon_str or not isinstance(exon_str, str):
        return 0
    m = re.match(r"(\d+)/\d+", str(exon_str).strip())
    if m:
        return int(m.group(1))
    try:
        return int(exon_str)
    except (ValueError, TypeError):
        return 0


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------
class RNASpliceIsoformPipeline:
    """
    Annotates variants with splice-context features.

    Gating: only variants where ``is_splice = 1`` or
    ``consequence_severity >= 7`` receive non-default values.
    All other variants receive defaults (0.0, 50, 0, 0).

    Usage
    -----
        pipeline = RNASpliceIsoformPipeline()
        df = pipeline.annotate_dataframe(df)
        # df now has maxentscan_score, dist_to_splice_site,
        #   exon_number, is_canonical_splice columns
    """

    DEFAULT_MAXENTSCAN       = 0.0
    DEFAULT_DIST_TO_SPLICE   = 50
    DEFAULT_EXON_NUMBER      = 0
    DEFAULT_IS_CANONICAL     = 0

    def annotate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add RNA splice features to df in-place.

        Parameters
        ----------
        df : pd.DataFrame
            Variant DataFrame.  Optionally contains:
              - consequence             VEP consequence string
              - consequence_severity    precomputed severity (int)
              - is_splice               precomputed binary flag
              - fasta_seq               101-bp context window
              - dist_to_donor           distance to nearest donor (int)
              - dist_to_acceptor        distance to nearest acceptor (int)
              - exon_number             exon/intron annotation string

        Returns
        -------
        pd.DataFrame with four new columns.
        """
        result = df.copy()
        n = len(result)

        # Determine which rows are splice-relevant
        is_splice      = result.get("is_splice", pd.Series(0, index=result.index))
        consev         = result.get("consequence_severity", pd.Series(0, index=result.index))
        splice_mask    = (is_splice.astype(int) == 1) | (consev.astype(int) >= 7)
        n_splice       = int(splice_mask.sum())

        # Initialise with defaults
        result["maxentscan_score"]    = self.DEFAULT_MAXENTSCAN
        result["dist_to_splice_site"] = self.DEFAULT_DIST_TO_SPLICE
        result["exon_number"]         = self.DEFAULT_EXON_NUMBER
        result["is_canonical_splice"] = self.DEFAULT_IS_CANONICAL

        if n_splice == 0:
            logger.debug("RNASpliceIsoformPipeline: no splice variants — defaults applied.")
            return result

        # --- exon_number from VEP annotation string ---
        if "exon_number" in df.columns or "vep_exon" in df.columns:
            exon_col = "exon_number" if "exon_number" in df.columns else "vep_exon"
            result.loc[splice_mask, "exon_number"] = (
                df.loc[splice_mask, exon_col]
                .map(_parse_exon_number)
                .values
            )
        elif "vep_exon_intron" in df.columns:
            result.loc[splice_mask, "exon_number"] = (
                df.loc[splice_mask, "vep_exon_intron"]
                .map(_parse_exon_number)
                .values
            )

        # --- distance to splice site ---
        dist_donor    = result.get("dist_to_donor",    pd.Series(dtype=float))
        dist_acceptor = result.get("dist_to_acceptor", pd.Series(dtype=float))

        if len(dist_donor) == n and len(dist_acceptor) == n:
            # Use minimum absolute distance to either splice site
            d_don = dist_donor.abs().fillna(self.DEFAULT_DIST_TO_SPLICE)
            d_acc = dist_acceptor.abs().fillna(self.DEFAULT_DIST_TO_SPLICE)
            result.loc[splice_mask, "dist_to_splice_site"] = (
                pd.concat([d_don[splice_mask], d_acc[splice_mask]], axis=1)
                .min(axis=1)
                .astype(int)
                .values
            )

            # --- is_canonical_splice ---
            result.loc[splice_mask, "is_canonical_splice"] = [
                _is_canonical_position(
                    int(dist_donor.iloc[i]) if pd.notna(dist_donor.iloc[i]) else None,
                    int(dist_acceptor.iloc[i]) if pd.notna(dist_acceptor.iloc[i]) else None,
                )
                for i in result.index[splice_mask]
            ]

        # --- MaxEntScan from fasta_seq context ---
        if "fasta_seq" in df.columns:
            fasta_col = df["fasta_seq"].fillna("")
            splice_idx = result.index[splice_mask]

            scores = []
            for i in splice_idx:
                seq = str(fasta_col.iloc[i] if isinstance(i, int) else fasta_col.loc[i])
                center = len(seq) // 2   # variant position in 101-bp window

                # Try donor score (variant at position +1 of GT)
                donor_start  = center - 3
                donor_end    = center + 6
                acceptor_start = center - 20
                acceptor_end   = center + 3

                score = self.DEFAULT_MAXENTSCAN
                if donor_start >= 0 and donor_end <= len(seq):
                    seq9  = seq[donor_start:donor_end]
                    score = _score_donor(seq9)
                elif acceptor_start >= 0 and acceptor_end <= len(seq):
                    seq23 = seq[acceptor_start:acceptor_end]
                    score = _score_acceptor(seq23)
                scores.append(score)

            result.loc[splice_idx, "maxentscan_score"] = scores

        else:
            logger.warning(
                "RNASpliceIsoformPipeline: 'fasta_seq' column absent — "
                "maxentscan_score will use default 0.0 for %d splice variants.",
                n_splice,
            )

        logger.info(
            "RNASpliceIsoformPipeline: annotated %d / %d splice variants "
            "(mean maxentscan=%.2f).",
            n_splice,
            n,
            float(result.loc[splice_mask, "maxentscan_score"].mean()),
        )
        return result
