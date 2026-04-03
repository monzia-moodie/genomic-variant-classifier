"""
scripts/parse_lovd_all.py
=========================
Parse all LOVD raw files into a single labelled parquet for external validation.

Handles two source formats:
  1. LOVD3 full tab export (.txt)  — TP53, PTEN, RB1
  2. LOVD3 REST API JSON (.json)   — BRCA1, BRCA2, MLH1, MSH2, MSH6, APC, NF1

Output: data/external/lovd/lovd_all_variants.parquet
Columns: variant_id, chrom, pos, ref, alt, label (1=pathogenic, 0=benign),
         gene_symbol, classification_raw, source_format

Usage:
    python scripts/parse_lovd_all.py
    python scripts/parse_lovd_all.py --raw-dir data/external/lovd/raw --out data/external/lovd/lovd_all_variants.parquet
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/external/lovd/raw")
OUT_FILE = Path("data/external/lovd/lovd_all_variants.parquet")

# ---------------------------------------------------------------------------
# Effect/classification → binary label
# ---------------------------------------------------------------------------

# LOVD3 REST API vocabulary
_JSON_PATH = {"functionprobablyaffected", "functionaffected"}
_JSON_BEN  = {"functionnotaffected"}

# LOVD3 tab export ClinicalClassification vocabulary
_TAB_PATH  = {"pathogenic", "likely pathogenic", "definitely pathogenic",
              "probably pathogenic", "class 5", "class 4"}
_TAB_BEN   = {"benign", "likely benign", "probably not pathogenic",
              "not pathogenic", "class 1", "class 2"}

# effectid integer codes (tab export)
_EFF_PATH  = {"99", "1"}   # definitely/probably pathogenic
_EFF_BEN   = {"11", "10"}  # definitely/probably not pathogenic


def _label_from_effect_terms(terms: list[str]) -> int | None:
    """Map LOVD3 REST API effect vocabulary to 0/1."""
    for t in terms:
        tl = t.lower().strip()
        if tl in _JSON_PATH:
            return 1
        if tl in _JSON_BEN:
            return 0
    return None


def _label_from_classification(val: str) -> int | None:
    """Map LOVD3 tab export ClinicalClassification text to 0/1."""
    if not isinstance(val, str):
        return None
    v = val.lower().strip()
    for p in _TAB_PATH:
        if p in v:
            return 1
    for b in _TAB_BEN:
        if b in v:
            return 0
    return None


def _label_from_effectid(eid: str) -> int | None:
    """Map LOVD3 effectid integer to 0/1."""
    s = str(eid).strip()
    if s in _EFF_PATH:
        return 1
    if s in _EFF_BEN:
        return 0
    return None


# ---------------------------------------------------------------------------
# HGVS genomic → (pos, ref, alt)  — SNV substitutions only
# ---------------------------------------------------------------------------

_SNV_RE = re.compile(r"g\.(\d+)([ACGT]+)>([ACGT]+)$", re.IGNORECASE)
_POS_RE = re.compile(r"g\.(\d+)")


def _hgvs_to_vcf(hgvs: str) -> tuple[int | None, str | None, str | None]:
    """
    Parse a HGVS genomic string and return (pos, ref, alt).
    Only substitutions (g.NNNNA>B) yield ref/alt; other variants return
    (pos, None, None) where pos is the start coordinate.
    """
    hgvs = str(hgvs).strip()
    m = _SNV_RE.search(hgvs)
    if m:
        return int(m.group(1)), m.group(2).upper(), m.group(3).upper()
    # Extract position only for non-SNV variants
    mp = _POS_RE.search(hgvs)
    if mp:
        return int(mp.group(1)), None, None
    return None, None, None


def _parse_chrom(raw: str) -> str:
    """Normalise chromosome: strip 'chr', map M→MT."""
    c = str(raw).strip().replace("chr", "").replace("Chr", "")
    if c == "M":
        c = "MT"
    return c


# ---------------------------------------------------------------------------
# Parser A: LOVD3 full tab export (.txt)
# ---------------------------------------------------------------------------

def _parse_section(lines: list[str], section_header: str) -> pd.DataFrame | None:
    """
    Extract a named section from LOVD3 full-export lines.
    Returns a DataFrame or None if section not found.
    """
    start = None
    for i, line in enumerate(lines):
        if line.startswith(f"## {section_header} ##"):
            start = i
            break
    if start is None:
        return None

    # Find the column header line (starts with "{{id}}")
    col_line = None
    for i in range(start, min(start + 10, len(lines))):
        if lines[i].startswith('"{{'):
            col_line = i
            break
    if col_line is None:
        return None

    # Collect data lines until next section or EOF
    data_lines = []
    for i in range(col_line + 1, len(lines)):
        line = lines[i].strip()
        if not line:
            continue
        if line.startswith("##"):
            break
        if line.startswith('"'):
            data_lines.append(line)

    if not data_lines:
        return None

    # Parse as TSV
    import io
    raw_tsv = "\n".join([lines[col_line]] + data_lines)
    # Strip {{ }} template markers from column names
    df = pd.read_csv(
        io.StringIO(raw_tsv),
        sep="\t",
        quotechar='"',
        low_memory=False,
    )
    df.columns = [c.strip("{}") for c in df.columns]
    return df


def parse_lovd_tab(path: Path, gene: str) -> list[dict]:
    """Parse a LOVD3 full tab export (.txt) for one gene."""
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    vog = _parse_section(lines, "Variants_On_Genome")
    if vog is None or vog.empty:
        print(f"  {gene}: no Variants_On_Genome section found in {path.name}")
        return []

    rows = []
    for _, r in vog.iterrows():
        # --- pathogenicity label ---
        cls_raw = str(r.get("VariantOnGenome/ClinicalClassification", "")).strip()
        label = _label_from_classification(cls_raw)
        if label is None:
            # Fall back to effectid
            label = _label_from_effectid(str(r.get("effectid", "")))
        if label is None:
            continue  # VUS / unclassified

        # --- coordinates (prefer hg38) ---
        chrom_raw = str(r.get("chromosome", "")).strip()
        hgvs_hg38 = str(r.get("VariantOnGenome/DNA/hg38", "")).strip()
        hgvs_hg19 = str(r.get("VariantOnGenome/DNA", "")).strip()

        hgvs = hgvs_hg38 if hgvs_hg38 and hgvs_hg38 != "nan" else hgvs_hg19
        pos, ref, alt = _hgvs_to_vcf(hgvs)

        if pos is None:
            # Fall back to position_g_start
            try:
                pos = int(float(str(r.get("position_g_start", ""))))
            except (ValueError, TypeError):
                pass

        if pos is None:
            continue

        chrom = _parse_chrom(chrom_raw) if chrom_raw and chrom_raw != "nan" else ""
        if not chrom:
            continue

        rows.append({
            "chrom":             chrom,
            "pos":               pos,
            "ref":               ref,
            "alt":               alt,
            "label":             label,
            "gene_symbol":       gene,
            "classification_raw": cls_raw,
            "source_format":     "lovd_tab",
            "variant_id":        f"{chrom}:{pos}:{ref or '?'}:{alt or '?'}",
        })
    return rows


# ---------------------------------------------------------------------------
# Parser B: LOVD3 REST API JSON (.json)
# ---------------------------------------------------------------------------

def parse_lovd_json(path: Path, gene: str) -> list[dict]:
    """Parse a LOVD3 REST API JSON file for one gene."""
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError as e:
        print(f"  {gene}: JSON parse error — {e}")
        return []

    if not isinstance(data, list):
        print(f"  {gene}: unexpected JSON root type {type(data)}")
        return []

    rows = []
    for entry in data:
        if not isinstance(entry, dict):
            continue

        # --- pathogenicity ---
        concluded = entry.get("effect_concluded") or []
        reported  = entry.get("effect_reported")  or []
        if isinstance(concluded, str):
            concluded = [concluded]
        if isinstance(reported, str):
            reported = [reported]

        label = _label_from_effect_terms(concluded)
        if label is None:
            label = _label_from_effect_terms(reported)
        if label is None:
            continue

        # --- coordinates (hg38 preferred) ---
        pos_genomic = entry.get("position_genomic") or {}
        if isinstance(pos_genomic, str):
            # some entries have "chr17:43094027" directly
            pos_genomic = {"hg38": pos_genomic}

        hg38_pos = pos_genomic.get("hg38") or pos_genomic.get("hg19") or ""
        var_genomic = entry.get("variant_genomic") or {}
        if isinstance(var_genomic, str):
            var_genomic = {"hg38": var_genomic}
        hg38_var = var_genomic.get("hg38") or var_genomic.get("hg19") or ""

        # Chromosome from position_genomic.hg38 (e.g. "chr17:43094027_43094031")
        chrom = ""
        if hg38_pos:
            m = re.match(r"(?:chr)?(\w+):", str(hg38_pos))
            if m:
                chrom = _parse_chrom(m.group(1))

        if not chrom:
            # Fall back to symbol-based inference (unreliable — skip)
            continue

        # HGVS from variant_genomic.hg38 (e.g. "chr17:g.43094027G>T")
        hgvs_part = re.sub(r"^(?:chr)?\w+:", "", str(hg38_var)).strip()
        pos, ref, alt = _hgvs_to_vcf(hgvs_part) if hgvs_part else (None, None, None)

        if pos is None:
            # Fall back: extract position from position_genomic.hg38
            m2 = re.search(r":(\d+)", str(hg38_pos))
            if m2:
                pos = int(m2.group(1))

        if pos is None:
            continue

        cls_raw = "; ".join(concluded or reported)
        rows.append({
            "chrom":             chrom,
            "pos":               pos,
            "ref":               ref,
            "alt":               alt,
            "label":             label,
            "gene_symbol":       gene,
            "classification_raw": cls_raw,
            "source_format":     "lovd_json",
            "variant_id":        f"{chrom}:{pos}:{ref or '?'}:{alt or '?'}",
        })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out",     type=Path, default=OUT_FILE)
    args = parser.parse_args()

    raw_dir: Path = args.raw_dir
    all_rows: list[dict] = []

    # ---- tab exports (.txt) ----
    tab_genes = {
        "TP53": raw_dir / "TP53.txt",
        "PTEN": raw_dir / "PTEN.txt",
        "RB1":  raw_dir / "RB1.txt",
    }
    for gene, path in tab_genes.items():
        if not path.exists():
            print(f"  SKIP {gene}: {path} not found")
            continue
        rows = parse_lovd_tab(path, gene)
        print(f"  {gene:8s}  {len(rows):>6,} labelled variants  (tab format)")
        all_rows.extend(rows)

    # ---- REST API JSON (.json) — skip the empty _api.json stubs ----
    json_genes = ["BRCA1", "BRCA2", "MLH1", "MSH2", "MSH6", "APC", "NF1"]
    for gene in json_genes:
        path = raw_dir / f"{gene}.json"
        if not path.exists():
            print(f"  SKIP {gene}: {path} not found")
            continue
        if path.stat().st_size == 0:
            print(f"  SKIP {gene}: empty file")
            continue
        rows = parse_lovd_json(path, gene)
        print(f"  {gene:8s}  {len(rows):>6,} labelled variants  (json format)")
        all_rows.extend(rows)

    if not all_rows:
        print("No variants parsed. Check raw files.")
        return

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset=["variant_id"])

    print(f"\nTotal: {len(df):,} unique labelled variants")
    print(f"  Pathogenic: {(df['label']==1).sum():,}")
    print(f"  Benign:     {(df['label']==0).sum():,}")
    print(f"\nBy gene:")
    for gene, grp in df.groupby("gene_symbol"):
        n_p = (grp["label"]==1).sum()
        n_b = (grp["label"]==0).sum()
        snv = grp["ref"].notna().sum()
        print(f"  {gene:8s}  {len(grp):>5,}  ({n_p} path / {n_b} benign)  "
              f"SNV coords: {snv}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
