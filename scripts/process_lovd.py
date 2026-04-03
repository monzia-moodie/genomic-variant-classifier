"""
scripts/process_lovd.py
Merge raw LOVD per-gene TSVs into a single parquet used by the
LOVDConnector.  Run after manual browser downloads are complete.

Usage:
    python scripts/process_lovd.py
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/external/lovd/raw")
OUT_FILE = Path("data/external/lovd/lovd_variants.parquet")

# LOVD column name aliases (vary slightly across gene databases)
_CHROM_COLS  = {"chromosome", "chrom", "chromosome/position"}
_POS_COLS    = {"position_g_start", "position", "g_start"}
_REF_COLS    = {"ref", "reference", "dna_change_ref"}
_ALT_COLS    = {"alt", "alternate", "dna_change_alt"}
_CLASS_COLS  = {"classification", "variant_class", "pathogenicity"}
_GENE_COLS   = {"gene", "gene_symbol", "gene_id"}


def _pick(cols: set[str], df_cols: list[str]) -> str | None:
    for c in df_cols:
        if c.lower() in cols:
            return c
    return None


def _map_class(val: str | None) -> str | None:
    if not isinstance(val, str):
        return None
    v = val.lower()
    if "pathogenic" in v and "likely" not in v:
        return "pathogenic"
    if "likely pathogenic" in v:
        return "pathogenic"
    if "benign" in v and "likely" not in v:
        return "benign"
    if "likely benign" in v:
        return "benign"
    if "uncertain" in v or "vus" in v:
        return "uncertain"
    return None


def _parse_one(path: Path, gene: str) -> pd.DataFrame:
    # LOVD TSVs have comment lines starting with ##; skip them
    df = pd.read_csv(path, sep="\t", comment="#", low_memory=False)
    df.columns = df.columns.str.strip()

    chrom_col = _pick(_CHROM_COLS, list(df.columns))
    pos_col   = _pick(_POS_COLS,   list(df.columns))
    ref_col   = _pick(_REF_COLS,   list(df.columns))
    alt_col   = _pick(_ALT_COLS,   list(df.columns))
    cls_col   = _pick(_CLASS_COLS, list(df.columns))

    out = pd.DataFrame()
    out["chrom"] = (
        df[chrom_col].astype(str).str.replace(r"^chr", "", regex=True)
        if chrom_col else pd.NA
    )
    out["pos"] = pd.to_numeric(df[pos_col], errors="coerce") if pos_col else pd.NA
    out["ref"] = df[ref_col].astype(str).str.upper() if ref_col else pd.NA
    out["alt"] = df[alt_col].astype(str).str.upper() if alt_col else pd.NA
    out["gene_symbol"]    = gene
    out["pathogenicity"]  = df[cls_col].apply(_map_class) if cls_col else None
    out["source_db"]      = "lovd"
    out["variant_id"]     = (
        "lovd:" + out["chrom"].astype(str) + ":" +
        out["pos"].astype(str) + ":" +
        out["ref"].astype(str) + ":" +
        out["alt"].astype(str)
    )
    # Drop rows with no usable coordinate
    out = out.dropna(subset=["pos"])
    return out


def main() -> None:
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    for tsv in sorted(RAW_DIR.glob("*_variants.tsv")):
        gene = tsv.stem.replace("_variants", "").upper()
        df   = _parse_one(tsv, gene)
        print(f"  {gene:8s}  {len(df):>6,} rows")
        frames.append(df)

    if not frames:
        print("No TSV files found. Download them first.")
        return

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["variant_id"])
    merged.to_parquet(OUT_FILE, index=False)
    print(f"\nWrote {len(merged):,} variants → {OUT_FILE}")
    # Pathogenicity breakdown
    print(merged["pathogenicity"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()