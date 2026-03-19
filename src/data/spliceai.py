"""
src/data/spliceai.py
====================
SpliceAI pre-computed score connector -- Phase 2, Pillar 1, Connector 2.
 
Annotates variants with SpliceAI splice-disruption delta scores from
Illumina's pre-computed lookup file (spliceai_scores.masked.snv.hg38.vcf.gz).
 
SpliceAI predicts four delta scores per variant per gene:
  DS_AG  delta score for acceptor gain
  DS_AL  delta score for acceptor loss
  DS_DG  delta score for donor gain
  DS_DL  delta score for donor loss
 
The canonical splice_ai_score feature = max(DS_AG, DS_AL, DS_DG, DS_DL)
across all genes overlapping the variant. Higher = more splice-disrupting.
Scores >= 0.2 are considered potentially splice-altering; >= 0.5 high-confidence.
 
Data source (free, no account required):
  https://zenodo.org/records/3665373
  File: spliceai_scores.masked.snv.hg38.vcf.gz  (~2.6 GB compressed)
  Place at: data/external/spliceai_scores.masked.snv.hg38.vcf.gz
 
ANNOTATOR pattern:
  Unlike GTExConnector which generates new variant rows, SpliceAIConnector
  ANNOTATES existing variants by adding splice_ai_score to each row.
  Call: annotated_df = connector.fetch(variant_df=my_canonical_df)
 
First-run behaviour:
  The VCF is parsed line-by-line and a parquet lookup cache is written to
  data/raw/cache/. Parsing takes 10-20 minutes for the full ~70M-line file.
  All subsequent runs load from the parquet cache in ~5 seconds.
 
PHASE_2_PLACEHOLDER: Indel scores.
  The masked SNV file covers SNVs only. Indel support requires the separate
  spliceai_scores.masked.indel.hg38.vcf.gz (~0.5 GB). Add after SNV
  coverage is validated on real ClinVar data.
"""
 
from __future__ import annotations
 
import gzip
import logging
from pathlib import Path
from typing import Optional
 
import pandas as pd
 
from src.data.database_connectors import (
    CANONICAL_COLUMNS,
    BaseConnector,
    FetchConfig,
)
 
logger = logging.getLogger(__name__)
 
SPLICEAI_HIGH_CONFIDENCE = 0.5
SPLICEAI_MODERATE        = 0.2
DEFAULT_VCF_PATH = Path("data/external/spliceai_scores.masked.snv.hg38.vcf.gz")
 
 
class SpliceAIConnector(BaseConnector):
    """
    Annotates variants with SpliceAI splice-disruption scores from
    Illumina's pre-computed lookup file.
 
    Usage
    -----
        connector = SpliceAIConnector(
            vcf_path="data/external/spliceai_scores.masked.snv.hg38.vcf.gz"
        )
        annotated_df = connector.fetch(variant_df=canonical_df)
        # annotated_df now has a splice_ai_score column
 
    If vcf_path is None or the file does not exist, all variants receive
    splice_ai_score = 0.0 and a WARNING is logged.
    """
 
    source_name = "spliceai"
 
    def __init__(
        self,
        vcf_path: Optional[str | Path] = None,
        config: Optional[FetchConfig] = None,
    ) -> None:
        super().__init__(config)
        self.vcf_path: Optional[Path] = (
            Path(vcf_path) if vcf_path is not None else None
        )
 
    def fetch(self, variant_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if variant_df.empty:
            df = variant_df.copy()
            df["splice_ai_score"] = pd.Series(dtype=float)
            return df

        if self.vcf_path is None:
            logger.debug(
                "SpliceAIConnector: no vcf_path provided, returning 0.0 scores."
            )
            df = variant_df.copy()
            df["splice_ai_score"] = 0.0
            return df

        # vcf_path is set — try cache first, then the file
        lookup = self._get_lookup()
        if lookup.empty:
            df = variant_df.copy()
            df["splice_ai_score"] = 0.0
            return df
        return self._annotate(variant_df, lookup)
 
    def _get_lookup(self) -> pd.DataFrame:
        """Return lookup DataFrame, using parquet cache when available."""
        cache_key = "scores_snv"
        cached = self._load_cache(cache_key)
        if cached is not None and not cached.empty:
            logger.info("SpliceAI: loaded %d scores from parquet cache.", len(cached))
            return cached

        # Cache miss — need the actual VCF file
        if not self.vcf_path.exists():
            logger.warning(
                "SpliceAI VCF not found at '%s' -- setting splice_ai_score=0.0 "
                "for all variants. Download from: %s",
                self.vcf_path,
                "https://zenodo.org/records/3665373",
            )
            return pd.DataFrame(
                columns=["lookup_key", "splice_ai_score",
                         "ds_ag", "ds_al", "ds_dg", "ds_dl"]
            )

        logger.info(
            "SpliceAI: building lookup from %s "
            "(first run -- may take 10-20 minutes for the full file)...",
            self.vcf_path,
        )
        lookup = self._parse_vcf(self.vcf_path)
        if not lookup.empty:
            self._save_cache(cache_key, lookup)
            logger.info("SpliceAI: cached %d variant scores.", len(lookup))
        return lookup
 
    def _parse_vcf(self, path: Path) -> pd.DataFrame:
        """
        Parse SpliceAI VCF into a lookup DataFrame.
 
        VCF INFO field format (one prediction per overlapping gene,
        comma-separated):
          SpliceAI=ALLELE|SYMBOL|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL
 
        splice_ai_score = max(DS_AG, DS_AL, DS_DG, DS_DL) across all genes.
        """
        rows: list[dict] = []
        opener = gzip.open if str(path).endswith(".gz") else open
        n_lines = 0
 
        with opener(path, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                n_lines += 1
                if n_lines % 1_000_000 == 0:
                    logger.info(
                        "SpliceAI: parsed %dM lines, %d scores so far...",
                        n_lines // 1_000_000, len(rows),
                    )
                parts = line.strip().split("\t")
                if len(parts) < 8:
                    continue
 
                chrom   = parts[0]
                pos_str = parts[1]
                ref     = parts[3]
                alt     = parts[4]
                info    = parts[7]
 
                if chrom.startswith("chr"):
                    chrom = chrom[3:]
 
                spliceai_str = None
                for field in info.split(";"):
                    if field.startswith("SpliceAI="):
                        spliceai_str = field[9:]
                        break
 
                if spliceai_str is None:
                    continue
 
                scores = self.parse_info_field(spliceai_str)
                rows.append({
                    "lookup_key":      f"{chrom}:{pos_str}:{ref}:{alt}",
                    "splice_ai_score": scores["splice_ai_score"],
                    "ds_ag":           scores.get("ds_ag", 0.0),
                    "ds_al":           scores.get("ds_al", 0.0),
                    "ds_dg":           scores.get("ds_dg", 0.0),
                    "ds_dl":           scores.get("ds_dl", 0.0),
                })
 
        logger.info(
            "SpliceAI: finished parsing -- %d lines, %d variants with scores.",
            n_lines, len(rows),
        )
        if not rows:
            return pd.DataFrame(
                columns=["lookup_key", "splice_ai_score",
                         "ds_ag", "ds_al", "ds_dg", "ds_dl"]
            )
        return pd.DataFrame(rows)
 
    def _annotate(self, variant_df: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
        """Left-join SpliceAI scores onto variant_df by genomic locus."""
        df = variant_df.copy()
 
        df["_lookup_key"] = (
            df["chrom"].astype(str) + ":" +
            df["pos"].astype(str)   + ":" +
            df["ref"].astype(str)   + ":" +
            df["alt"].astype(str)
        )
 
        score_cols = lookup[["lookup_key", "splice_ai_score"]].rename(
            columns={"lookup_key": "_lookup_key"}
        )
 
        df = df.merge(score_cols, on="_lookup_key", how="left")
        df["splice_ai_score"] = df["splice_ai_score"].fillna(0.0)
        df = df.drop(columns=["_lookup_key"])
 
        n_found = (df["splice_ai_score"] > 0).sum()
        n_high  = (df["splice_ai_score"] >= SPLICEAI_HIGH_CONFIDENCE).sum()
        logger.info(
            "SpliceAI: %d / %d variants have score > 0  (%d high-confidence >= %.1f).",
            n_found, len(df), n_high, SPLICEAI_HIGH_CONFIDENCE,
        )
        return df
 
    @staticmethod
    def parse_info_field(info_str: str) -> dict:
        """
        Parse a SpliceAI INFO string and return the highest-scoring prediction.
 
        Accepts the portion of the INFO field after "SpliceAI=", which may
        contain multiple comma-separated predictions (one per gene).
 
        Example input:
            "T|BRCA1|0.01|0.00|0.00|0.00|-22|11|5|-32"
 
        Example output:
            {"splice_ai_score": 0.01, "ds_ag": 0.01, "ds_al": 0.0,
             "ds_dg": 0.0, "ds_dl": 0.0, "symbol": "BRCA1"}
 
        Returns {"splice_ai_score": 0.0} if string is empty or unparseable.
        """
        if not info_str or not info_str.strip():
            return {"splice_ai_score": 0.0}
 
        max_score = 0.0
        best: dict = {
            "splice_ai_score": 0.0,
            "ds_ag": 0.0, "ds_al": 0.0, "ds_dg": 0.0, "ds_dl": 0.0,
            "symbol": "",
        }
 
        for pred in info_str.split(","):
            pred_parts = pred.split("|")
            if len(pred_parts) < 6:
                continue
            try:
                ds_ag = float(pred_parts[2])
                ds_al = float(pred_parts[3])
                ds_dg = float(pred_parts[4])
                ds_dl = float(pred_parts[5])
                score = max(ds_ag, ds_al, ds_dg, ds_dl)
                if score > max_score:
                    max_score = score
                    best = {
                        "splice_ai_score": round(score, 4),
                        "ds_ag":           round(ds_ag,  4),
                        "ds_al":           round(ds_al,  4),
                        "ds_dg":           round(ds_dg,  4),
                        "ds_dl":           round(ds_dl,  4),
                        "symbol":          pred_parts[1] if len(pred_parts) > 1 else "",
                    }
            except (ValueError, IndexError):
                continue
 
        return best