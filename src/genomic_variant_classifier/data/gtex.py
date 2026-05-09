"""
src/data/gtex.py
=================
GTEx (Genotype-Tissue Expression) database connector — Phase 2, Pillar 1.
 
Provides two complementary data layers:
  1. Gene expression  — tissue-specific median TPM per gene
  2. eQTL associations — variant -> expression effect size per tissue
 
Both map to CANONICAL_COLUMNS from database_connectors.py so the rest of
the pipeline is database-agnostic.
 
API: GTEx Portal REST API v2 (public, no auth required)
     Base URL: https://gtexportal.org/api/v2/
 
New feature columns (added to PHASE_2_FEATURES in variant_ensemble.py):
  gtex_max_tpm              max median TPM across tissues (gene level)
  gtex_n_tissues_expressed  tissues with median TPM >= 1.0 (gene level)
  gtex_tissue_specificity   1 - mean_tpm/max_tpm  (gene level)
  gtex_is_eqtl              1 if variant is a significant eQTL in any tissue
  gtex_min_eqtl_pval        max -log10(p) across tissues
  gtex_max_abs_effect       max |beta| across tissues
 
PHASE_2_PLACEHOLDER: Full 49-tissue eQTL sweep.
  Currently queries PRIORITY_TISSUES (7 clinical tissues). Expand after
  benchmarking API latency and memory.
 
PHASE_2_PLACEHOLDER: HGVSp-level eQTL -> variant mapping.
  Splice and UTR eQTL variants lack protein_change. Resolve with VEP.
"""
 
from __future__ import annotations
 
import logging
import time
from typing import Optional
 
import numpy as np
import pandas as pd
import requests
 
from src.data.database_connectors import (
    CANONICAL_COLUMNS,
    BaseConnector,
    FetchConfig,
)
 
logger = logging.getLogger(__name__)
 
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
 
# 7 clinically prioritised tissues.
# PHASE_2_PLACEHOLDER: replace with full 49-tissue list after benchmarking.
PRIORITY_TISSUES: list[str] = [
    "Whole_Blood",
    "Brain_Cortex",
    "Liver",
    "Kidney_Cortex",
    "Heart_Left_Ventricle",
    "Lung",
    "Muscle_Skeletal",
]
 
GTEX_DATASET      = "gtex_v8"
GTEX_GENCODE      = "v26"
GTEX_GENOME       = "GRCh38/hg38"
GTEX_EXPR_MIN_TPM = 1.0   # tissues below this threshold are "not expressed"
 
 
# ---------------------------------------------------------------------------
# Helpers: GTEx <-> canonical variant ID conversion
# ---------------------------------------------------------------------------
 
def _gtex_variant_id(chrom: str, pos: int, ref: str, alt: str) -> str:
    """Canonical allele coords -> GTEx variant ID (e.g. chr1_925952_G_A_b38)."""
    return f"chr{chrom}_{pos}_{ref}_{alt}_b38"
 
 
def _from_gtex_variant_id(gtex_id: str) -> dict | None:
    """
    Parse a GTEx variant ID back into components.
 
    GTEx IDs look like: chr1_925952_G_A_b38
    Returns dict(chrom, pos, ref, alt) or None if format is unexpected.
    """
    parts = gtex_id.split("_")
    if len(parts) != 5 or not parts[0].startswith("chr"):
        logger.debug("Unexpected GTEx variant ID format: %s", gtex_id)
        return None
    chrom = parts[0][3:]   # strip leading "chr"
    try:
        pos = int(parts[1])
    except ValueError:
        return None
    return {"chrom": chrom, "pos": pos, "ref": parts[2], "alt": parts[3]}
 
 
# ---------------------------------------------------------------------------
# GTEx connector
# ---------------------------------------------------------------------------
 
class GTExConnector(BaseConnector):
    """
    Fetches gene expression profiles and eQTL associations from GTEx v8.
 
    Usage
    -----
        connector = GTExConnector()
        eqtl_df   = connector.fetch(gene_symbols=["BRCA1", "TP53"])
 
        # Gene-level expression summary populated as a side-effect of fetch()
        expr = connector.gene_expression_summary
        # -> pd.DataFrame indexed by gene_symbol with columns:
        #      gtex_max_tpm, gtex_n_tissues_expressed, gtex_tissue_specificity
    """
 
    source_name = "gtex"
    BASE_URL    = "https://gtexportal.org/api/v2"
 
    def __init__(self, config: Optional[FetchConfig] = None) -> None:
        super().__init__(config)
        self.gene_expression_summary: pd.DataFrame = pd.DataFrame()
 
    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
 
    def fetch(
        self,
        gene_symbols: list[str],
        tissues: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Fetch GTEx eQTL variants and expression profiles for the given genes.
 
        Args:
            gene_symbols: HGNC gene symbols, e.g. ["BRCA1", "TP53"].
            tissues:      GTEx tissue IDs to query. Defaults to PRIORITY_TISSUES.
 
        Returns:
            Canonical DataFrame of eQTL variant records (one row per eQTL).
            Side-effect: populates self.gene_expression_summary.
        """
        if not gene_symbols:
            return pd.DataFrame(columns=CANONICAL_COLUMNS)
 
        tissues     = tissues or PRIORITY_TISSUES
        gencode_map = self._resolve_gencode_ids(gene_symbols)
 
        expression_rows: list[dict] = []
        eqtl_rows:       list[dict] = []
 
        for symbol, gencode_id in gencode_map.items():
            logger.info("GTEx: processing %s (%s)", symbol, gencode_id)
            expression_rows.append(self._fetch_expression(symbol, gencode_id))
            for tissue in tissues:
                eqtl_rows.extend(self._fetch_eqtls(symbol, gencode_id, tissue))
                time.sleep(self.config.rate_limit_delay)
 
        if expression_rows:
            self.gene_expression_summary = (
                pd.DataFrame(expression_rows).set_index("gene_symbol")
            )
 
        if not eqtl_rows:
            logger.warning(
                "GTEx: no eQTL data returned for %d gene(s).", len(gene_symbols)
            )
            return pd.DataFrame(columns=CANONICAL_COLUMNS)
 
        return self._to_canonical(pd.DataFrame(eqtl_rows))
 
    # ------------------------------------------------------------------
    # Step 1: Resolve gene symbols -> Gencode IDs
    # ------------------------------------------------------------------
 
    def _resolve_gencode_ids(self, gene_symbols: list[str]) -> dict[str, str]:
        """Return {gene_symbol: gencode_id} for each successfully resolved gene."""
        result: dict[str, str] = {}
 
        for symbol in gene_symbols:
            cache_key = f"gene_{symbol}"
            cached    = self._load_cache(cache_key)
 
            if cached is not None and not cached.empty:
                result[symbol] = cached.iloc[0]["gencode_id"]
                continue
 
            try:
                resp = self._get(
                    f"{self.BASE_URL}/reference/gene",
                    params={
                        "geneId":         symbol,
                        "gencodeVersion": GTEX_GENCODE,
                        "genomeBuild":    GTEX_GENOME,
                    },
                )
                genes = resp.json().get("data", [])
            except requests.RequestException as exc:
                logger.warning("GTEx gene lookup failed for %s: %s", symbol, exc)
                continue
 
            if not genes or not genes[0].get("gencodeId"):
                logger.warning("GTEx: no Gencode ID found for gene %s", symbol)
                continue
 
            gencode_id     = genes[0]["gencodeId"]
            result[symbol] = gencode_id
            self._save_cache(
                cache_key,
                pd.DataFrame([{"gene_symbol": symbol, "gencode_id": gencode_id}]),
            )
 
        logger.info(
            "GTEx: resolved %d / %d gene symbols to Gencode IDs.",
            len(result), len(gene_symbols),
        )
        return result
 
    # ------------------------------------------------------------------
    # Step 2: Tissue-level median TPM expression
    # ------------------------------------------------------------------
 
    def _fetch_expression(self, gene_symbol: str, gencode_id: str) -> dict:
        """Fetch median TPM per tissue for one gene; return expression summary dict."""
        cache_key = f"expr_{gencode_id}"
        cached    = self._load_cache(cache_key)
        if cached is not None and not cached.empty:
            return self._summarise_expression(gene_symbol, cached)
 
        try:
            resp = self._get(
                f"{self.BASE_URL}/expression/medianGeneExpression",
                params={
                    "datasetId":    GTEX_DATASET,
                    "gencodeId":    gencode_id,
                    "itemsPerPage": 250,
                },
            )
            records = resp.json().get("data", [])
        except requests.RequestException as exc:
            logger.warning(
                "GTEx expression fetch failed for %s: %s", gene_symbol, exc
            )
            return self._empty_expression_row(gene_symbol)
 
        if not records:
            logger.warning(
                "GTEx: no expression records for %s (%s)", gene_symbol, gencode_id
            )
            return self._empty_expression_row(gene_symbol)
 
        expr_df = pd.DataFrame(records)[["tissueSiteDetailId", "median"]]
        self._save_cache(cache_key, expr_df)
        return self._summarise_expression(gene_symbol, expr_df)
 
    @staticmethod
    def _summarise_expression(gene_symbol: str, expr_df: pd.DataFrame) -> dict:
        """Compute summary statistics from a tissue -> TPM DataFrame."""
        tpm         = expr_df["median"].astype(float)
        max_tpm     = float(tpm.max())  if not tpm.empty else 0.0
        mean_tpm    = float(tpm.mean()) if not tpm.empty else 0.0
        n_expressed = int((tpm >= GTEX_EXPR_MIN_TPM).sum())
        specificity = round(1.0 - mean_tpm / max_tpm, 4) if max_tpm > 0 else 0.0
        tissue_tpm  = (
            expr_df.set_index("tissueSiteDetailId")["median"]
            .astype(float)
            .to_dict()
        )
        return {
            "gene_symbol":              gene_symbol,
            "gtex_max_tpm":             round(max_tpm, 4),
            "gtex_n_tissues_expressed": n_expressed,
            "gtex_tissue_specificity":  specificity,
            "gtex_tissue_tpm":          tissue_tpm,
        }
 
    @staticmethod
    def _empty_expression_row(gene_symbol: str) -> dict:
        """Safe fallback when the API returns no expression data."""
        return {
            "gene_symbol":              gene_symbol,
            "gtex_max_tpm":             0.0,
            "gtex_n_tissues_expressed": 0,
            "gtex_tissue_specificity":  0.0,
            "gtex_tissue_tpm":          {},
        }
 
    # ------------------------------------------------------------------
    # Step 3: Single-tissue eQTL associations
    # ------------------------------------------------------------------
 
    def _fetch_eqtls(
        self,
        gene_symbol: str,
        gencode_id:  str,
        tissue:      str,
    ) -> list[dict]:
        """
        Fetch significant single-tissue eQTL variants for one (gene, tissue) pair.
        Returns a list of pre-canonical row dicts.
        """
        cache_key = f"eqtl_{gencode_id}_{tissue}"
        cached    = self._load_cache(cache_key)
        if cached is not None and not cached.empty:
            return cached.to_dict(orient="records")
 
        try:
            resp = self._get(
                f"{self.BASE_URL}/association/singleTissueEqtl",
                params={
                    "datasetId":          GTEX_DATASET,
                    "gencodeId":          gencode_id,
                    "tissueSiteDetailId": tissue,
                    "itemsPerPage":       250,
                },
            )
            records = resp.json().get("data", [])
        except requests.RequestException as exc:
            logger.warning(
                "GTEx eQTL fetch failed for %s / %s: %s", gene_symbol, tissue, exc
            )
            return []
 
        if not records:
            return []
 
        rows: list[dict] = []
        for r in records:
            gtex_vid = r.get("variantId", "")
            parsed   = _from_gtex_variant_id(gtex_vid)
            if not parsed:
                continue
 
            pval         = r.get("pValue") or 1.0
            neg_log_pval = float(-np.log10(float(pval) + 1e-300))
            beta         = float(r.get("nes") or 0.0)
 
            rows.append({
                "variant_id":     (
                    f"gtex:{parsed['chrom']}:{parsed['pos']}"
                    f":{parsed['ref']}:{parsed['alt']}"
                ),
                "source_db":      self.source_name,
                "chrom":          parsed["chrom"],
                "pos":            parsed["pos"],
                "ref":            parsed["ref"],
                "alt":            parsed["alt"],
                "gene_symbol":    gene_symbol,
                "transcript_id":  None,
                "consequence":    "regulatory_region_variant",
                "pathogenicity":  None,
                "allele_freq":    r.get("maf"),
                "clinical_sig":   None,
                "protein_change": None,
                "fasta_seq":      None,
                "source_id":      gtex_vid,
                "metadata": {
                    "tissue":         tissue,
                    "neg_log10_pval": round(neg_log_pval, 4),
                    "beta":           round(beta, 6),
                    "tss_distance":   r.get("tssDistance"),
                },
            })
 
        if rows:
            self._save_cache(cache_key, pd.DataFrame(rows))
        logger.debug(
            "GTEx: %d eQTLs for %s / %s", len(rows), gene_symbol, tissue
        )
        return rows
 
 
# ---------------------------------------------------------------------------
# Feature join: attach GTEx signals to a canonical variant DataFrame
# ---------------------------------------------------------------------------
 
def build_gtex_feature_df(
    connector:  GTExConnector,
    variant_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-join GTEx expression and eQTL features onto a canonical variant DataFrame.
 
    Call connector.fetch() first, then pass the merged canonical variant_df here.
    Returns variant_df with six new NaN-safe feature columns (filled with 0):
 
        gtex_max_tpm              (gene level — joined on gene_symbol)
        gtex_n_tissues_expressed  (gene level)
        gtex_tissue_specificity   (gene level)
        gtex_is_eqtl              (variant level — 1 if significant eQTL)
        gtex_min_eqtl_pval        (variant level — max -log10(p))
        gtex_max_abs_effect       (variant level — max |beta|)
    """
    df = variant_df.copy()
 
    # Gene-level expression join
    if not connector.gene_expression_summary.empty:
        expr_cols = ["gtex_max_tpm", "gtex_n_tissues_expressed", "gtex_tissue_specificity"]
        expr      = connector.gene_expression_summary[expr_cols].reset_index()
        df        = df.merge(expr, on="gene_symbol", how="left")
    else:
        logger.warning("GTEx: gene_expression_summary empty; skipping expression join.")
        df["gtex_max_tpm"]             = 0.0
        df["gtex_n_tissues_expressed"] = 0
        df["gtex_tissue_specificity"]  = 0.0
 
    # Variant-level eQTL join
    gtex_mask = df.get("source_db", pd.Series("", index=df.index)) == "gtex"
    gtex_rows = df[gtex_mask].copy()
 
    if not gtex_rows.empty and "metadata" in gtex_rows.columns:
        gtex_rows["_neg_log_pval"] = gtex_rows["metadata"].apply(
            lambda m: m.get("neg_log10_pval", 0.0) if isinstance(m, dict) else 0.0
        )
        gtex_rows["_abs_beta"] = gtex_rows["metadata"].apply(
            lambda m: abs(m.get("beta", 0.0)) if isinstance(m, dict) else 0.0
        )
        gtex_rows["_locus"] = gtex_rows["variant_id"].str.split(":").str[1:].str.join(":")
        agg = (
            gtex_rows.groupby("_locus", as_index=False)
            .agg(
                gtex_is_eqtl       =("_neg_log_pval", lambda x: int((x > 0).any())),
                gtex_min_eqtl_pval =("_neg_log_pval", "max"),
                gtex_max_abs_effect=("_abs_beta",      "max"),
            )
        )
        df["_locus"] = df["variant_id"].str.split(":").str[1:].str.join(":")
        df = df.merge(agg, on="_locus", how="left").drop(columns=["_locus"])
    else:
        df["gtex_is_eqtl"]        = 0
        df["gtex_min_eqtl_pval"]  = 0.0
        df["gtex_max_abs_effect"] = 0.0
 
    # NaN-safe fill
    df["gtex_max_tpm"]             = df["gtex_max_tpm"].fillna(0.0)
    df["gtex_n_tissues_expressed"] = df["gtex_n_tissues_expressed"].fillna(0).astype(int)
    df["gtex_tissue_specificity"]  = df["gtex_tissue_specificity"].fillna(0.0)
    df["gtex_is_eqtl"]             = df["gtex_is_eqtl"].fillna(0).astype(int)
    df["gtex_min_eqtl_pval"]       = df["gtex_min_eqtl_pval"].fillna(0.0)
    df["gtex_max_abs_effect"]      = df["gtex_max_abs_effect"].fillna(0.0)
    return df