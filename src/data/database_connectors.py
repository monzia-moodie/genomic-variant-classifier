"""
Biomedical Database Connectors
================================
Live connectors for publicly accessible databases:
  - ClinVar  (NCBI FTP bulk TSV download)
  - gnomAD v4 (GraphQL API)
  - UniProt  (REST API)
  - OMIM     (REST API — free key required)

All connectors expose a unified .fetch() → pd.DataFrame interface
with CANONICAL_COLUMNS so the ETL pipeline is database-agnostic.

CHANGES FROM PHASE 1:
  - Moved from src/data_ingestion/ to src/data/ (fixes import path Bug 4)
  - _map_pathogenicity uses substring matching (fixes Issue G)
  - load_all_databases accepts optional clinvar_path (fixes Issue J)
  - from __future__ import annotations added (Issue N)
  - Module-level logging.basicConfig removed (Issue L)
  - nx.read/write_gpickle replaced with pickle (Bug 6)
"""

from __future__ import annotations

import gzip
import io
import logging
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical schema — every connector maps output to these columns
# ---------------------------------------------------------------------------
CANONICAL_COLUMNS = [
    "variant_id",       # unique string: source:chrom:pos:ref:alt
    "source_db",        # e.g. "clinvar", "gnomad", "uniprot"
    "chrom",            # chromosome (str, e.g. "1", "X")
    "pos",              # GRCh38 position (int)
    "ref",              # reference allele
    "alt",              # alternate allele
    "gene_symbol",      # HGNC gene symbol
    "transcript_id",    # Ensembl transcript (if available)
    "consequence",      # VEP consequence term
    "pathogenicity",    # "pathogenic" | "benign" | "uncertain" | None
    "allele_freq",      # population allele frequency (float or None)
    "clinical_sig",     # raw clinical significance string
    "protein_change",   # e.g. "p.Arg177Gln"
    "fasta_seq",        # standardized FASTA context window (101 bp)
    "source_id",        # accession in source DB (e.g. rs ID, ClinVar ID)
    "metadata",         # dict of extra fields per source
]


@dataclass
class FetchConfig:
    cache_dir: Path = Path("data/raw/cache")
    request_timeout: int = 30
    retry_attempts: int = 3
    retry_backoff: float = 2.0
    rate_limit_delay: float = 0.34   # ~3 req/sec for NCBI compliance

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Base connector
# ---------------------------------------------------------------------------
class BaseConnector:
    source_name: str = "base"

    def __init__(self, config: Optional[FetchConfig] = None) -> None:
        self.config = config or FetchConfig()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "genomic-variant-classifier/1.0"})

    def _get(self, url: str, params: dict | None = None, **kwargs) -> requests.Response:
        for attempt in range(self.config.retry_attempts):
            try:
                resp = self.session.get(
                    url, params=params,
                    timeout=self.config.request_timeout, **kwargs
                )
                resp.raise_for_status()
                time.sleep(self.config.rate_limit_delay)
                return resp
            except requests.RequestException as exc:
                if attempt == self.config.retry_attempts - 1:
                    raise
                wait = self.config.retry_backoff ** attempt
                logger.warning("Request failed (%s), retrying in %.1fs...", exc, wait)
                time.sleep(wait)
        # Unreachable, but satisfies type checkers
        raise RuntimeError("_get: exceeded retry attempts")

    def _cache_path(self, key: str) -> Path:
        safe_key = key.replace("/", "_").replace(":", "_")
        return self.config.cache_dir / f"{self.source_name}_{safe_key}.parquet"

    def _load_cache(self, key: str) -> Optional[pd.DataFrame]:
        path = self._cache_path(key)
        if path.exists():
            logger.info("Loading %s cache: %s", self.source_name, path)
            return pd.read_parquet(path)
        return None

    def _save_cache(self, key: str, df: pd.DataFrame) -> None:
        self._cache_path(key).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self._cache_path(key), index=False)

    def fetch(self, **kwargs) -> pd.DataFrame:
        raise NotImplementedError

    def _to_canonical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all canonical columns exist; fill missing with None."""
        for col in CANONICAL_COLUMNS:
            if col not in df.columns:
                df[col] = None
        return df[CANONICAL_COLUMNS].copy()


# ---------------------------------------------------------------------------
# ClinVar connector
# ---------------------------------------------------------------------------
class ClinVarConnector(BaseConnector):
    """
    Downloads the ClinVar variant summary TSV (GRCh38).
    URL: https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz
    ~500 MB compressed; cached locally as parquet after first download.
    """
    source_name = "clinvar"
    FTP_URL = (
        "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/"
        "variant_summary.txt.gz"
    )

    def fetch(
        self,
        assembly: str = "GRCh38",
        pathogenicity_filter: Optional[list[str]] = None,
        gene_filter: Optional[list[str]] = None,
        force_refresh: bool = False,
        local_path: Optional[str] = None,    # NEW: skip download if file exists
    ) -> pd.DataFrame:
        """
        Args:
            local_path: If provided and exists, read from this file instead of downloading.
                        Avoids the 500 MB download when you already have the file.
        """
        cache_key = f"summary_{assembly}"
        if not force_refresh:
            cached = self._load_cache(cache_key)
            if cached is not None:
                return cached

        # Support pre-downloaded local file (Issue J fix)
        if local_path and Path(local_path).exists():
            logger.info("Reading ClinVar from local file: %s", local_path)
            open_fn = gzip.open if str(local_path).endswith(".gz") else open
            with open_fn(local_path, "rt") as f:
                df = pd.read_csv(f, sep="\t", low_memory=False)
        else:
            logger.info("Downloading ClinVar variant_summary.txt.gz ...")
            resp = self._get(self.FTP_URL, stream=True)
            raw = b"".join(resp.iter_content(chunk_size=1 << 20))
            with gzip.open(io.BytesIO(raw), "rt") as f:
                df = pd.read_csv(f, sep="\t", low_memory=False)

        df = df[df["Assembly"] == assembly].copy()

        df = df.rename(columns={
            "GeneSymbol":           "gene_symbol",
            "ClinicalSignificance": "clinical_sig",
            "Chromosome":           "chrom",
            "Start":                "pos",
            "ReferenceAllele":      "ref",
            "AlternateAllele":      "alt",
            "ProteinChange":        "protein_change",
            "VariationID":          "source_id",
            "RS# (dbSNP)":          "rs_id",
        })
        df["source_db"]     = self.source_name
        df["pathogenicity"] = df["clinical_sig"].apply(self._map_pathogenicity)
        df["allele_freq"]   = None   # joined from gnomAD in ETL
        df["fasta_seq"]     = None   # populated by ETL enrichment
        df["transcript_id"] = None
        df["consequence"]   = None
        df["metadata"]      = df.apply(
            lambda r: {
                "rs_id":         r.get("rs_id"),
                "review_status": r.get("ReviewStatus"),
            },
            axis=1,
        )
        df["variant_id"] = (
            "clinvar:" + df["chrom"].astype(str) + ":" +
            df["pos"].astype(str) + ":" +
            df["ref"].astype(str) + ":" +
            df["alt"].astype(str)
        )

        if pathogenicity_filter:
            df = df[df["pathogenicity"].isin(pathogenicity_filter)]
        if gene_filter:
            df = df[df["gene_symbol"].isin(gene_filter)]

        result = self._to_canonical(df)
        self._save_cache(cache_key, result)
        logger.info("ClinVar: %d variants loaded.", len(result))
        return result

    @staticmethod
    def _map_pathogenicity(sig: str) -> str:
        """
        Map raw ClinVar ClinicalSignificance string to a canonical label.

        CHANGE: Now uses substring matching instead of exact set membership
        so compound terms like 'Pathogenic, risk factor' are handled correctly.
        """
        if not isinstance(sig, str) or not sig.strip():
            return "uncertain"
        s = sig.lower()
        # Check "likely" variants first so they don't match the shorter base terms
        if "likely pathogenic" in s:
            return "likely_pathogenic"
        if "likely benign" in s:
            return "likely_benign"
        if "pathogenic" in s:
            return "pathogenic"
        if "benign" in s:
            return "benign"
        return "uncertain"


# ---------------------------------------------------------------------------
# gnomAD v4 connector
# ---------------------------------------------------------------------------
class GnomADConnector(BaseConnector):
    """
    Queries gnomAD v4 GraphQL API for variant-level allele frequencies.
    """
    source_name = "gnomad"
    GRAPHQL_URL = "https://gnomad.broadinstitute.org/api"

    _VARIANT_QUERY = """
    query GnomADVariants($gene_id: String!, $dataset: DatasetId!) {
      gene(gene_id: $gene_id, reference_genome: GRCh38) {
        variants(dataset: $dataset) {
          variant_id
          chrom
          pos
          ref
          alt
          consequence
          transcript_id
          exome { af ac an }
          genome { af ac an }
          clinvar_variation { clinical_significance }
        }
      }
    }
    """

    def fetch(
        self,
        gene_ids: list[str],
        dataset: str = "gnomad_r4",
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        all_rows: list[pd.DataFrame] = []
        for gene_id in gene_ids:
            cache_key = f"{gene_id}_{dataset}"
            if not force_refresh:
                cached = self._load_cache(cache_key)
                if cached is not None:
                    all_rows.append(cached)
                    continue

            logger.info("Querying gnomAD for gene %s ...", gene_id)
            payload = {
                "query": self._VARIANT_QUERY,
                "variables": {"gene_id": gene_id, "dataset": dataset},
            }
            try:
                resp = self.session.post(
                    self.GRAPHQL_URL,
                    json=payload,
                    timeout=self.config.request_timeout,
                )
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException as exc:
                logger.warning("gnomAD query failed for %s: %s", gene_id, exc)
                continue

            variants = (data.get("data") or {}).get("gene", {}).get("variants", [])
            if not variants:
                logger.warning("No gnomAD variants returned for %s", gene_id)
                continue

            rows = []
            for v in variants:
                exome_af  = (v.get("exome") or {}).get("af")
                genome_af = (v.get("genome") or {}).get("af")
                af = exome_af if exome_af is not None else genome_af

                clinvar_sig = None
                if v.get("clinvar_variation"):
                    clinvar_sig = v["clinvar_variation"].get("clinical_significance")

                rows.append({
                    "variant_id":    f"gnomad:{v['chrom']}:{v['pos']}:{v['ref']}:{v['alt']}",
                    "source_db":     self.source_name,
                    "chrom":         v["chrom"],
                    "pos":           v["pos"],
                    "ref":           v["ref"],
                    "alt":           v["alt"],
                    "gene_symbol":   gene_id,
                    "transcript_id": v.get("transcript_id"),
                    "consequence":   v.get("consequence"),
                    "pathogenicity": None,
                    "allele_freq":   af,
                    "clinical_sig":  clinvar_sig,
                    "protein_change": None,
                    "fasta_seq":     None,
                    "source_id":     v["variant_id"],
                    "metadata": {
                        "exome_ac":  (v.get("exome") or {}).get("ac"),
                        "genome_ac": (v.get("genome") or {}).get("ac"),
                        "dataset":   dataset,
                    },
                })

            gene_df = self._to_canonical(pd.DataFrame(rows))
            self._save_cache(cache_key, gene_df)
            all_rows.append(gene_df)
            time.sleep(self.config.rate_limit_delay)

        if not all_rows:
            return pd.DataFrame(columns=CANONICAL_COLUMNS)
        return self._to_canonical(pd.concat(all_rows, ignore_index=True))


# ---------------------------------------------------------------------------
# UniProt connector
# ---------------------------------------------------------------------------
class UniProtConnector(BaseConnector):
    """
    Queries UniProt REST API for protein-level variant annotations.
    """
    source_name = "uniprot"
    BASE_URL = "https://rest.uniprot.org/uniprotkb"

    def fetch(
        self,
        gene_symbols: list[str],
        organism: str = "9606",   # Homo sapiens
        fields: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        fields = fields or [
            "accession", "gene_names", "sequence",
            "ft_natural_var", "ft_act_site", "cc_disease",
        ]
        all_rows: list[pd.DataFrame] = []
        for gene in gene_symbols:
            cache_key = f"{gene}_{organism}"
            cached = self._load_cache(cache_key)
            if cached is not None:
                all_rows.append(cached)
                continue

            params = {
                "query":  f"gene:{gene} AND organism_id:{organism} AND reviewed:true",
                "format": "json",
                "fields": ",".join(fields),
                "size":   25,
            }
            try:
                resp = self._get(f"{self.BASE_URL}/search", params=params)
            except requests.RequestException as exc:
                logger.warning("UniProt query failed for %s: %s", gene, exc)
                continue

            data = resp.json()
            rows = []
            for entry in data.get("results", []):
                accession = entry.get("primaryAccession")
                seq = entry.get("sequence", {}).get("value", "")
                diseases = [
                    c.get("disease", {}).get("diseaseId")
                    for c in entry.get("comments", [])
                    if c.get("commentType") == "DISEASE"
                ]
                natural_variants = [
                    f for f in entry.get("features", [])
                    if f.get("type") == "Natural variant"
                ]

                for var in natural_variants:
                    pos  = var.get("location", {}).get("start", {}).get("value")
                    desc = var.get("description", "")
                    orig = var.get("alternativeSequence", {}).get("originalSequence", "")
                    alts = var.get("alternativeSequence", {}).get("alternativeSequences", [])
                    alt  = alts[0] if alts else None

                    rows.append({
                        "variant_id":    f"uniprot:{accession}:{pos}:{orig}:{alt}",
                        "source_db":     self.source_name,
                        "chrom":         None,
                        "pos":           pos,
                        "ref":           orig,
                        "alt":           alt,
                        "gene_symbol":   gene,
                        "transcript_id": None,
                        "consequence":   "missense_variant",
                        "pathogenicity": "pathogenic" if "disease" in desc.lower() else None,
                        "allele_freq":   None,
                        "clinical_sig":  desc,
                        "protein_change": f"p.{pos}{orig}>{alt}" if (pos and alt) else None,
                        "fasta_seq":     seq[:101] if seq else None,
                        "source_id":     accession,
                        "metadata":      {"diseases": diseases, "uniprot_accession": accession},
                    })

            gene_df = (
                self._to_canonical(pd.DataFrame(rows))
                if rows else pd.DataFrame(columns=CANONICAL_COLUMNS)
            )
            self._save_cache(cache_key, gene_df)
            all_rows.append(gene_df)

        if not all_rows:
            return pd.DataFrame(columns=CANONICAL_COLUMNS)
        return self._to_canonical(pd.concat(all_rows, ignore_index=True))


# ---------------------------------------------------------------------------
# OMIM connector (requires free API key: https://www.omim.org/api)
# ---------------------------------------------------------------------------
class OMIMConnector(BaseConnector):
    source_name = "omim"
    BASE_URL = "https://api.omim.org/api"

    def __init__(self, api_key: Optional[str] = None, config: Optional[FetchConfig] = None) -> None:
        super().__init__(config)
        self.api_key = api_key or os.environ.get("OMIM_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OMIM_API_KEY not set. Get a free key at https://www.omim.org/api"
            )

    def fetch(self, search_terms: list[str]) -> pd.DataFrame:
        rows = []
        for term in search_terms:
            params = {
                "search":  term,
                "format":  "json",
                "limit":   20,
                "apiKey":  self.api_key,
            }
            try:
                resp = self._get(f"{self.BASE_URL}/entry/search", params=params)
            except requests.RequestException as exc:
                logger.warning("OMIM query failed for '%s': %s", term, exc)
                continue

            for entry in (
                resp.json()
                .get("omim", {})
                .get("searchResponse", {})
                .get("entryList", [])
            ):
                e = entry.get("entry", {})
                rows.append({
                    "variant_id":    f"omim:{e.get('mimNumber')}",
                    "source_db":     self.source_name,
                    "chrom": None, "pos": None, "ref": None, "alt": None,
                    "gene_symbol":   e.get("titles", {}).get("preferredTitle", ""),
                    "transcript_id": None, "consequence": None,
                    "pathogenicity": "pathogenic",
                    "allele_freq":   None,
                    "clinical_sig":  e.get("titles", {}).get("preferredTitle", ""),
                    "protein_change": None, "fasta_seq": None,
                    "source_id":     str(e.get("mimNumber")),
                    "metadata":      {"mim_number": e.get("mimNumber")},
                })

        return (
            self._to_canonical(pd.DataFrame(rows))
            if rows else pd.DataFrame(columns=CANONICAL_COLUMNS)
        )


# ---------------------------------------------------------------------------
# Convenience: load all databases into a merged DataFrame
# ---------------------------------------------------------------------------
def load_all_databases(
    genes: list[str],
    gnomad_gene_ids: list[str],
    config: Optional[FetchConfig] = None,
    omim_api_key: Optional[str] = None,
    clinvar_path: Optional[str] = None,   # NEW: skip download if provided
) -> pd.DataFrame:
    """
    Unified entry point. Loads ClinVar (bulk), gnomAD (per gene),
    and UniProt (per gene); merges on variant_id; deduplicates.
    OMIM is loaded if api_key is provided.

    Args:
        clinvar_path: Optional path to a pre-downloaded ClinVar file (.gz or plain).
                      Pass this to avoid the ~500 MB download on every run.
    """
    parts: list[pd.DataFrame] = []

    clinvar = ClinVarConnector(config)
    parts.append(clinvar.fetch(local_path=clinvar_path))

    gnomad = GnomADConnector(config)
    parts.append(gnomad.fetch(gene_ids=gnomad_gene_ids))

    uniprot = UniProtConnector(config)
    parts.append(uniprot.fetch(gene_symbols=genes))

    if omim_api_key:
        omim = OMIMConnector(api_key=omim_api_key, config=config)
        parts.append(omim.fetch(search_terms=genes[:10]))

    non_empty = [p for p in parts if not p.empty]
    if not non_empty:
        logger.warning("No data fetched from any database.")
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    merged = pd.concat(non_empty, ignore_index=True)
    merged = merged.drop_duplicates(subset=["variant_id"])
    logger.info("Total variants across all databases: %d", len(merged))
    return merged
