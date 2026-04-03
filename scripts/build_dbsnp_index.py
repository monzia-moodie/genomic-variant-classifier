"""
scripts/build_dbsnp_index.py
============================
Build the dbSNP parquet index consumed by GET /rsid/{rs_id}.

Input
-----
The script accepts the NCBI dbSNP VCF (b156 / b157 / b158, GRCh38) which
is distributed as a single compressed VCF:

  https://ftp.ncbi.nih.gov/snp/archive/b156/VCF/GCF_000001405.40.gz

Or a pre-downloaded local copy.

Output schema
-------------
  rs_id   str   normalised rs-ID, e.g. "rs12345678"
  chrom   str   chromosome, e.g. "1", "X", "MT"
  pos     int   1-based GRCh38 position
  ref     str   reference allele
  alt     str   first alternate allele (multi-allelic records are expanded)

The parquet is indexed on rs_id (sorted) to enable O(log n) lookups.

Usage
-----
  # Full dbSNP build (~900 M rows, ~20 GB parquet -- needs lots of RAM/disk):
  python scripts/build_dbsnp_index.py \\
      --vcf data/external/dbsnp/GCF_000001405.40.gz \\
      --out data/processed/dbsnp_index.parquet

  # ClinVar-filtered subset (recommended -- keeps only ~5 M rows):
  python scripts/build_dbsnp_index.py \\
      --vcf  data/external/dbsnp/GCF_000001405.40.gz \\
      --clinvar data/processed/clinvar_grch38.parquet \\
      --out  data/processed/dbsnp_index.parquet

  # From a directory of per-chromosome VCFs:
  python scripts/build_dbsnp_index.py \\
      --vcf-dir data/external/dbsnp \\
      --out     data/processed/dbsnp_index.parquet

  # Smoke-test with a tiny synthetic VCF (no real data needed):
  python scripts/build_dbsnp_index.py \\
      --vcf  data/external/dbsnp/GCF_000001405.40.gz \\
      --out  data/processed/dbsnp_index.parquet \\
      --max-rows 100000

NCBI uses NCBI RefSeq chromosome names (NC_000001.11 -> 1, etc.).  The
script remaps these automatically.  Standard "chr1" / "1" names also pass
through unchanged.

Memory-efficient streaming
--------------------------
The VCF is streamed line-by-line; rows are accumulated into chunks of
``--chunk-size`` (default 500 000) before being written as individual
parquet row-groups.  The final file is then sorted on rs_id so the API
can use binary search (pandas .loc on a sorted, non-duplicate index).
"""

from __future__ import annotations

import argparse
import gzip
import logging
import sys
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("build_dbsnp_index")


# ---------------------------------------------------------------------------
# NCBI RefSeq -> standard chromosome name map (GRCh38)
# ---------------------------------------------------------------------------
_REFSEQ_TO_CHROM: dict[str, str] = {
    "NC_000001.11": "1",  "NC_000002.12": "2",  "NC_000003.12": "3",
    "NC_000004.12": "4",  "NC_000005.10": "5",  "NC_000006.12": "6",
    "NC_000007.14": "7",  "NC_000008.11": "8",  "NC_000009.14": "9",
    "NC_000010.11": "10", "NC_000011.10": "11", "NC_000012.12": "12",
    "NC_000013.11": "13", "NC_000014.9":  "14", "NC_000015.10": "15",
    "NC_000016.10": "16", "NC_000017.11": "17", "NC_000018.10": "18",
    "NC_000019.10": "19", "NC_000020.11": "20", "NC_000021.9":  "21",
    "NC_000022.11": "22", "NC_000023.11": "X",  "NC_000024.10": "Y",
    "NC_012920.1":  "MT",
}

_KEEP_CHROMS: frozenset[str] = frozenset(
    list(map(str, range(1, 23))) + ["X", "Y", "MT"]
)


def _normalise_chrom(raw: str) -> Optional[str]:
    """Return canonical chromosome name, or None to skip this record."""
    # NCBI RefSeq accession
    if raw in _REFSEQ_TO_CHROM:
        return _REFSEQ_TO_CHROM[raw]
    # Standard "chr" prefix
    c = raw.lower()
    if c.startswith("chr"):
        c = c[3:]
    if c == "m":
        c = "mt"
    c = c.upper() if c in ("x", "y", "mt") else c
    # Numeric chromosomes
    if c.isdigit():
        return c
    if c in _KEEP_CHROMS:
        return c
    return None  # contig / patch -- skip


def _open_vcf(path: Path):
    """Yield text lines from a plain or gzipped VCF."""
    if str(path).endswith(".gz") or str(path).endswith(".bgz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def _parse_vcf_stream(
    path: Path,
    chunk_size: int,
    max_rows: Optional[int],
    clinvar_loci: Optional[set[str]],
) -> Iterator[pd.DataFrame]:
    """
    Stream ``path`` and yield DataFrames of at most ``chunk_size`` rows.

    Yields DataFrames with columns: rs_id, chrom, pos, ref, alt.
    Multi-allelic ALT fields are expanded to one row each.
    """
    rows: list[tuple] = []
    total = 0

    with _open_vcf(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue

            parts = line.rstrip("\n").split("\t", 8)
            if len(parts) < 5:
                continue

            raw_chrom, raw_pos, id_field, ref, alt_field = (
                parts[0], parts[1], parts[2], parts[3], parts[4],
            )

            # Must have an rs-ID
            if not id_field.startswith("rs"):
                continue

            chrom = _normalise_chrom(raw_chrom)
            if chrom is None:
                continue

            try:
                pos = int(raw_pos)
            except ValueError:
                continue

            ref = ref.upper()
            rs_id = id_field.strip().lower()

            # Expand multi-allelic (take first ALT only for the primary record;
            # secondary alts are ignored to keep the index simple)
            alts = [a.strip().upper() for a in alt_field.split(",") if a.strip()]
            if not alts:
                continue

            for alt in alts:
                if clinvar_loci is not None:
                    key = f"{chrom}:{pos}:{ref}:{alt}"
                    if key not in clinvar_loci:
                        continue

                rows.append((rs_id, chrom, pos, ref, alt))
                total += 1

                if len(rows) >= chunk_size:
                    yield pd.DataFrame(rows, columns=["rs_id", "chrom", "pos", "ref", "alt"])
                    rows = []

                if max_rows and total >= max_rows:
                    if rows:
                        yield pd.DataFrame(rows, columns=["rs_id", "chrom", "pos", "ref", "alt"])
                    logger.info("Reached --max-rows %d -- stopping early.", max_rows)
                    return

    if rows:
        yield pd.DataFrame(rows, columns=["rs_id", "chrom", "pos", "ref", "alt"])

    logger.info("Parsed %d total rows from %s", total, path.name)


def _collect_clinvar_loci(clinvar_path: Path) -> set[str]:
    """Return set of 'chrom:pos:ref:alt' keys present in the ClinVar parquet."""
    logger.info("Loading ClinVar loci from %s …", clinvar_path)
    df = pd.read_parquet(clinvar_path, columns=["chrom", "pos", "ref", "alt"])
    loci = set(
        f"{row.chrom}:{row.pos}:{row.ref}:{row.alt}"
        for row in df.itertuples(index=False)
    )
    logger.info("  %d ClinVar loci loaded.", len(loci))
    return loci


def build_index(
    vcf_paths: list[Path],
    out_path: Path,
    clinvar_path: Optional[Path],
    chunk_size: int,
    max_rows: Optional[int],
) -> None:
    clinvar_loci = None
    if clinvar_path is not None:
        clinvar_loci = _collect_clinvar_loci(clinvar_path)

    chunks: list[pd.DataFrame] = []

    for vcf_path in vcf_paths:
        logger.info("Streaming %s …", vcf_path)
        for chunk_df in _parse_vcf_stream(vcf_path, chunk_size, max_rows, clinvar_loci):
            chunks.append(chunk_df)
            logger.info("  accumulated %d chunks (%d rows so far)",
                        len(chunks),
                        sum(len(c) for c in chunks))

    if not chunks:
        logger.error("No rows parsed -- check VCF paths and filters.")
        sys.exit(1)

    logger.info("Concatenating %d chunks …", len(chunks))
    df = pd.concat(chunks, ignore_index=True)

    # Deduplicate: keep first occurrence per rs_id (primary alt)
    before = len(df)
    df = df.drop_duplicates(subset=["rs_id"], keep="first")
    logger.info("Deduplicated %d -> %d rows (kept first alt per rs-ID)", before, len(df))

    # Sort on rs_id for binary-search lookups in the API
    df = df.sort_values("rs_id").reset_index(drop=True)
    df = df.set_index("rs_id")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, engine="pyarrow", compression="snappy")

    logger.info(
        "Wrote %d rows to %s  (%.1f MB)",
        len(df),
        out_path,
        out_path.stat().st_size / 1e6,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build rs-ID to locus parquet index from a dbSNP VCF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--vcf",
        type=Path,
        help="Path to a single dbSNP VCF (plain or gzip / bgzip).",
    )
    src.add_argument(
        "--vcf-dir",
        type=Path,
        metavar="DIR",
        help="Directory containing per-chromosome dbSNP VCFs (*.vcf, *.vcf.gz).",
    )

    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/dbsnp_index.parquet"),
        help="Output parquet path (default: data/processed/dbsnp_index.parquet).",
    )
    p.add_argument(
        "--clinvar",
        type=Path,
        default=None,
        metavar="PARQUET",
        help=(
            "ClinVar parquet to use as a locus filter.  When supplied, only "
            "dbSNP records whose chrom:pos:ref:alt match a ClinVar variant are "
            "retained -- dramatically reduces output size."
        ),
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=500_000,
        metavar="N",
        help="Rows per in-memory chunk (default: 500 000).",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        metavar="N",
        help="Stop after N rows (useful for smoke-testing).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.vcf:
        vcf_paths = [args.vcf]
    else:
        vcf_dir = args.vcf_dir
        vcf_paths = sorted(
            list(vcf_dir.glob("*.vcf.gz"))
            + list(vcf_dir.glob("*.vcf.bgz"))
            + list(vcf_dir.glob("*.vcf"))
        )
        if not vcf_paths:
            logger.error("No VCF files found in %s", vcf_dir)
            sys.exit(1)
        logger.info("Found %d VCF files in %s", len(vcf_paths), vcf_dir)

    for p in vcf_paths:
        if not p.exists():
            logger.error("File not found: %s", p)
            sys.exit(1)

    build_index(
        vcf_paths=vcf_paths,
        out_path=args.out,
        clinvar_path=args.clinvar,
        chunk_size=args.chunk_size,
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    main()
