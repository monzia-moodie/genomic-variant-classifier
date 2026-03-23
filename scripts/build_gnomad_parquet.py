"""
scripts/build_gnomad_parquet.py
================================
Convert gnomAD v4 exomes sites VCF(s) to the parquet expected by DataPrepPipeline.

Output schema (only two columns matter to the pipeline):
  variant_id   str   "chrom:pos:ref:alt"   e.g. "1:925952:G:A"
  allele_freq  float exome AF (falls back to genome AF if exome AF absent)

Usage — directory of per-chromosome VCFs, filtered to ClinVar loci only:
  python scripts/build_gnomad_parquet.py \\
      --clinvar data/processed/clinvar_grch38.parquet \\
      --vcf-dir data/external/gnomad \\
      --out     data/processed/gnomad_v4_exomes.parquet

Usage — explicit glob (no ClinVar filter):
  python scripts/build_gnomad_parquet.py \\
      --vcf "data/external/gnomad/gnomad.exomes.v4.1.sites.chr*.vcf.bgz" \\
      --out data/processed/gnomad_v4_exomes.parquet

The script streams the VCF line-by-line so memory stays flat regardless of file size.
When --clinvar is supplied, only loci present in the ClinVar parquet are kept,
which reduces the output from ~500M rows to ~1-4M rows and makes the join fast.
Chromosomes 1-22, X, Y, M are all handled; chrM is normalised to MT.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("build_gnomad_parquet")

# gnomAD v4 INFO field names for exome / genome AF
_EXOME_AF_FIELD  = "AF"          # exomes VCF: AF is exome AF
_GENOME_AF_FIELD = "AF_genome"   # exomes VCF: AF_genome is genome AF (when present)


def _open_vcf(path: str):
    """Return a line iterator for a plain or bgzipped VCF."""
    if path.endswith(".gz") or path.endswith(".bgz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


def _parse_info(info_str: str, keys: tuple[str, ...]) -> dict[str, str | None]:
    """Extract specific key=value pairs from a VCF INFO column string."""
    result: dict[str, str | None] = {k: None for k in keys}
    for field in info_str.split(";"):
        if "=" in field:
            k, _, v = field.partition("=")
            if k in result:
                result[k] = v
    return result


def _strip_chr(chrom: str) -> str:
    """Normalise chromosome name: strip 'chr' prefix, chrM → MT."""
    chrom = chrom.lstrip("chr") if chrom.startswith("chr") else chrom
    return "MT" if chrom == "M" else chrom


def load_clinvar_loci(clinvar_path: str) -> set[str]:
    """
    Build a set of locus strings ("chrom:pos:ref:alt") from the ClinVar parquet.
    The variant_id column has a "clinvar:" prefix that is stripped.
    """
    logger.info("Loading ClinVar loci from %s ...", clinvar_path)
    df = pd.read_parquet(clinvar_path, columns=["variant_id"])
    loci: set[str] = set()
    for vid in df["variant_id"].dropna():
        # Strip source prefix: "clinvar:1:925952:G:A" → "1:925952:G:A"
        parts = str(vid).split(":", 1)
        loci.add(parts[1] if len(parts) > 1 else vid)
    logger.info("  %d unique ClinVar loci loaded (filter target).", len(loci))
    return loci


def parse_vcf(vcf_path: str, loci_filter: set[str] | None = None) -> list[dict]:
    rows: list[dict] = []
    skipped_multi = 0
    skipped_filter = 0

    with _open_vcf(vcf_path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue

            parts = line.rstrip("\n").split("\t", 8)
            if len(parts) < 8:
                continue

            chrom, pos, _id, ref, alt_field, _qual, _filter, info_str = (
                parts[0], parts[1], parts[2], parts[3],
                parts[4], parts[5], parts[6], parts[7],
            )

            # Skip multi-allelic sites
            if "," in alt_field:
                skipped_multi += 1
                continue

            chrom = _strip_chr(chrom)
            locus = f"{chrom}:{pos}:{ref}:{alt_field}"

            if loci_filter is not None and locus not in loci_filter:
                skipped_filter += 1
                continue

            info = _parse_info(info_str, (_EXOME_AF_FIELD, _GENOME_AF_FIELD))

            # Prefer exome AF; fall back to genome AF
            af_str = info[_EXOME_AF_FIELD] or info[_GENOME_AF_FIELD]
            try:
                af = float(af_str) if af_str is not None else None
            except ValueError:
                af = None

            rows.append({
                "variant_id":  locus,
                "allele_freq": af,
            })

    if skipped_multi:
        logger.info("  Skipped %d multi-allelic lines", skipped_multi)
    if skipped_filter:
        logger.info("  Skipped %d lines not in ClinVar loci", skipped_filter)
    return rows


def resolve_vcf_paths(args: argparse.Namespace) -> list[str]:
    """Return sorted list of VCF paths from --vcf glob or --vcf-dir."""
    if args.vcf:
        paths = sorted(glob.glob(args.vcf))
    else:
        vcf_dir = Path(args.vcf_dir)
        paths = sorted(
            str(p) for p in vcf_dir.iterdir()
            if p.suffix in (".gz", ".bgz") or p.name.endswith(".vcf")
        )
    return paths


def main() -> int:
    p = argparse.ArgumentParser(description="gnomAD VCF → parquet for DataPrepPipeline")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--vcf",     help="Path or glob to gnomAD sites VCF(s)")
    src.add_argument("--vcf-dir", help="Directory containing per-chromosome VCF files")
    p.add_argument("--out",      required=True,
                   help="Output parquet path (e.g. data/processed/gnomad_v4_exomes.parquet)")
    p.add_argument("--clinvar",  default=None,
                   help="ClinVar parquet path; when supplied, only matching loci are kept")
    p.add_argument("--drop-null-af", action="store_true",
                   help="Drop rows where allele_freq could not be parsed (default: keep as NaN)")
    args = p.parse_args()

    vcf_paths = resolve_vcf_paths(args)
    if not vcf_paths:
        logger.error("No VCF files found.")
        return 1
    logger.info("Found %d VCF file(s).", len(vcf_paths))

    loci_filter: set[str] | None = None
    if args.clinvar:
        loci_filter = load_clinvar_loci(args.clinvar)

    all_rows: list[dict] = []
    for vcf_path in vcf_paths:
        logger.info("Parsing %s ...", vcf_path)
        rows = parse_vcf(vcf_path, loci_filter)
        logger.info("  → %d variants kept", len(rows))
        all_rows.extend(rows)

    logger.info("Total variants parsed: %d", len(all_rows))

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates("variant_id")
    logger.info("After dedup: %d variants", len(df))

    if args.drop_null_af:
        before = len(df)
        df = df.dropna(subset=["allele_freq"])
        logger.info("Dropped %d rows with null AF; %d remain", before - len(df), len(df))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    logger.info("Written to %s  (%.1f MB)", out, out.stat().st_size / 1e6)

    if len(df) > 0:
        null_af = int(df["allele_freq"].isna().sum())
        logger.info(
            "AF stats: min=%.2e  median=%.2e  max=%.2e  null=%d",
            df["allele_freq"].min(),
            df["allele_freq"].median(),
            df["allele_freq"].max(),
            null_af,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
