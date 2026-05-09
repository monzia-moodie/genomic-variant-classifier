"""
scripts/build_lovd_index.py
============================
Download and parse LOVD (Leiden Open Variation Database) variants for use as
an independent external validation cohort -- Step 7B.

LOVD provides curated variants from European clinical genetics centres.
It overlaps minimally with ClinVar in terms of submitting labs, making it
a genuinely independent validation source.

Access notes (as of March 2026)
---------------------------------
LOVD now enforces a browser-validation anti-bot challenge on ALL endpoints,
including the public REST API.  Automated HTTP clients (requests, curl, wget)
receive a JavaScript challenge and are blocked.  The data is free; access
requires a real browser session.

To use this script you must supply a session cookie obtained from a logged-in
browser session:

  1. Open https://databases.lovd.nl/shared/ in your browser.
  2. Create a free account (or log in) at /register.
  3. Open DevTools → Application → Cookies and copy the PHPSESSID value.
  4. Pass it via  --lovd-cookie "PHPSESSID=<value>"

Without the cookie, the script exits with a clear error message.

Alternative: ClinVar temporal holdout (scripts/validate_clinvar_temporal.py)
provides a rigorous, fully automated external validation using post-cutoff
ClinVar submissions -- no auth required.

Output
------
  data/external/lovd/lovd_variants.parquet
  Columns: variant_id, chrom, pos, ref, alt, label (1=pathogenic, 0=benign),
           gene_symbol, classification_raw, lovd_id

Usage
-----
  # Download + parse (5 genes, REST + NCBI coordinate normalisation)
  python scripts/build_lovd_index.py \\
      --genes BRCA1 BRCA2 TP53 PTEN ATM \\
      --output data/external/lovd/lovd_variants.parquet

  # Validate against model
  python scripts/build_lovd_index.py \\
      --genes BRCA1 BRCA2 TP53 PTEN ATM \\
      --output data/external/lovd/lovd_variants.parquet \\
      --model  models/phase2_pipeline.joblib \\
      --eval-output outputs/lovd_validation

  # If you have LOVD credentials, supply a session cookie for bulk access:
  python scripts/build_lovd_index.py \\
      --genes BRCA1 BRCA2 TP53 PTEN ATM \\
      --lovd-cookie "PHPSESSID=<your_session_id>" \\
      --output data/external/lovd/lovd_variants.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("build_lovd_index")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LOVD_ATOM_URL   = "https://databases.lovd.nl/shared/api/rest.php/variants/{gene}"
_LOVD_TAB_URL    = "https://databases.lovd.nl/shared/variants/{gene}?format=tab"
_NCBI_VAR_URL    = "https://api.ncbi.nlm.nih.gov/variation/v0/hgvs/{hgvs}/contextuals"
_CLINGEN_URL     = "https://reg.clinicalgenome.org/allele?hgvsOrDescriptor={hgvs}"

_ATOM_NS = "{http://www.w3.org/2005/Atom}"

_REQUEST_TIMEOUT = 30
_RATE_DELAY      = 0.3   # seconds between NCBI calls

# LOVD classification strings -> binary label
_PATH_TERMS = {
    "pathogenic", "likely pathogenic", "definitely pathogenic",
    "probably pathogenic", "class 5", "class 4",
}
_BENIGN_TERMS = {
    "benign", "likely benign", "probably not pathogenic",
    "not pathogenic", "class 1", "class 2",
}


def _lovd_label(classification: str) -> Optional[int]:
    c = str(classification).lower().strip()
    if any(t in c for t in _PATH_TERMS):
        return 1
    if any(t in c for t in _BENIGN_TERMS):
        return 0
    return None


# ---------------------------------------------------------------------------
# Bulk tab download (requires LOVD account session cookie)
# ---------------------------------------------------------------------------

def _download_tab_with_cookie(
    gene: str, cookie: str, retries: int = 3
) -> Optional[pd.DataFrame]:
    """
    Download LOVD gene tab export using an authenticated session cookie.
    The cookie string should be the value of the PHPSESSID cookie obtained
    by logging in at https://databases.lovd.nl/shared/ via a browser.
    """
    import io
    session = requests.Session()
    session.headers["Cookie"] = cookie
    url = _LOVD_TAB_URL.format(gene=gene)
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=60)
            if resp.status_code == 402:
                logger.error(
                    "LOVD returned 402 for %s. The provided session cookie may "
                    "be expired or lack sufficient permissions. Create a free "
                    "account at https://databases.lovd.nl/shared/register and "
                    "copy your PHPSESSID cookie after login.", gene
                )
                return None
            resp.raise_for_status()
            lines = [l for l in resp.text.splitlines() if not l.startswith("##")]
            if not lines:
                return None
            df = pd.read_csv(io.StringIO("\n".join(lines)), sep="\t", low_memory=False)
            logger.info("  [tab] %s: %d rows", gene, len(df))
            return df
        except Exception as exc:
            logger.warning("Attempt %d failed for %s: %s", attempt + 1, gene, exc)
            if attempt < retries - 1:
                time.sleep(2)
    return None


# ---------------------------------------------------------------------------
# REST + NCBI coordinate normalisation (no auth required)
# ---------------------------------------------------------------------------

def _fetch_lovd_atom(gene: str, cookie: Optional[str], page_size: int = 100) -> list[dict]:
    """
    Paginate the LOVD Atom feed for a gene and return raw entry dicts.
    Fields: id, hgvs_cdna, position_genomic, classification

    Requires a valid LOVD session cookie; raises RuntimeError if the anti-bot
    challenge is detected or if auth fails.
    """
    if not cookie:
        raise RuntimeError(
            "LOVD requires a browser session cookie. "
            "Pass --lovd-cookie 'PHPSESSID=<value>' obtained from a logged-in "
            "browser at https://databases.lovd.nl/shared/"
        )

    session = requests.Session()
    session.headers.update({
        "Cookie": cookie,
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
    })

    entries = []
    page = 1
    while True:
        url = (
            f"{_LOVD_ATOM_URL.format(gene=gene)}"
            f"?page_size={page_size}&page={page}"
        )
        try:
            resp = session.get(url, timeout=_REQUEST_TIMEOUT)
            resp.raise_for_status()
        except Exception as exc:
            logger.warning("LOVD REST error page %d for %s: %s", page, gene, exc)
            break

        # Detect anti-bot challenge page
        if "Checking your browser" in resp.text or "failed validation" in resp.text:
            raise RuntimeError(
                "LOVD returned a browser-validation challenge. The session cookie "
                "may be expired or invalid. Re-obtain the PHPSESSID from a fresh "
                "browser login at https://databases.lovd.nl/shared/"
            )

        try:
            root = ET.fromstring(resp.text)
        except ET.ParseError as exc:
            logger.warning("XML parse error for %s page %d: %s", gene, page, exc)
            break

        batch = root.findall(f"{_ATOM_NS}entry")
        if not batch:
            break

        for entry in batch:
            content = entry.find(f"{_ATOM_NS}content")
            if content is None or not content.text:
                continue
            fields: dict[str, str] = {}
            for line in content.text.strip().splitlines():
                line = line.strip()
                if ":" in line:
                    k, _, v = line.partition(":")
                    fields[k.strip()] = v.strip()

            lovd_id  = fields.get("id", "")
            hgvs_rna = fields.get("position_mRNA", "")   # e.g. NM_007294.3:c.5266dup
            pos_g    = fields.get("position_genomic", "") # sometimes chr17:43045708
            classif  = fields.get("classification", fields.get("Variant/classification", ""))

            entries.append({
                "lovd_id":  lovd_id,
                "hgvs_cdna": hgvs_rna,
                "position_genomic_raw": pos_g,
                "classification_raw": classif,
                "gene_symbol": gene,
            })

        if len(batch) < page_size:
            break
        page += 1
        time.sleep(_RATE_DELAY)

    return entries


def _ncbi_hgvs_to_vcf(hgvs: str) -> Optional[tuple[str, int, str, str]]:
    """
    Resolve a coding HGVS string to (chrom, pos, ref, alt) using NCBI Variation
    Services.  Returns None if conversion fails or variant is not SNV/indel.
    """
    try:
        url = _NCBI_VAR_URL.format(hgvs=quote(hgvs, safe=""))
        resp = requests.get(url, timeout=_REQUEST_TIMEOUT)
        if resp.status_code != 200:
            return None
        data = resp.json()
        # data["contextuals"] is a list of genomic placements
        for ctx in data.get("contextuals", []):
            assembly = ctx.get("assembly", "")
            if "GRCh38" not in assembly and "hg38" not in assembly.lower():
                continue
            chrom = str(ctx.get("chromosome", "")).replace("chr", "").replace("Chr", "")
            if chrom == "M":
                chrom = "MT"
            pos = ctx.get("start")
            ref = ctx.get("referenceAllele", "")
            alt = ctx.get("alternateAllele", "")
            if pos and ref is not None and alt is not None:
                return chrom, int(pos), str(ref).upper(), str(alt).upper()
    except Exception:
        pass
    return None


def _parse_genomic_raw(pos_raw: str) -> Optional[tuple[str, int]]:
    """
    Parse LOVD's position_genomic field (e.g. 'chr17:43045708') into
    (chrom, pos).  Returns None if unparseable.
    """
    import re
    m = re.match(r"(?:chr)?(\w+):(\d+)", str(pos_raw).strip())
    if m:
        chrom = m.group(1)
        if chrom == "M":
            chrom = "MT"
        return chrom, int(m.group(2))
    return None


def download_gene_list(
    genes: list[str],
    output_path: Path,
    cookie: Optional[str] = None,
    max_per_gene: int = 5_000,
) -> pd.DataFrame:
    """
    Fetch LOVD variants for a list of genes.

    If a session cookie is provided, attempts the tab bulk export first.
    Falls back to REST API + NCBI coordinate normalisation.
    """
    all_rows: list[dict] = []

    for gene in genes:
        gene_rows: list[dict] = []

        # ---- attempt 1: tab export with cookie --------------------------------
        if cookie:
            df_tab = _download_tab_with_cookie(gene, cookie)
            if df_tab is not None:
                gene_rows = _parse_tab_rows(df_tab, gene)

        # ---- attempt 2: REST + NCBI -------------------------------------------
        if not gene_rows:
            logger.info("Fetching LOVD REST feed for %s ...", gene)
            entries = _fetch_lovd_atom(gene, cookie)
            logger.info("  %s: %d atom entries", gene, len(entries))

            resolved = 0
            for e in entries[:max_per_gene]:
                label = _lovd_label(e["classification_raw"])
                if label is None:
                    continue  # skip VUS / unclassified

                chrom, pos, ref, alt = None, None, None, None

                # Try position_genomic_raw first (fast, no API call)
                if e["position_genomic_raw"]:
                    parsed = _parse_genomic_raw(e["position_genomic_raw"])
                    # genomic raw only has chrom:pos, not ref/alt — need NCBI anyway
                    # but we can skip if HGVS is also missing
                    pass

                # Resolve via NCBI HGVS
                if e["hgvs_cdna"]:
                    coords = _ncbi_hgvs_to_vcf(e["hgvs_cdna"])
                    if coords:
                        chrom, pos, ref, alt = coords
                        resolved += 1
                    time.sleep(_RATE_DELAY)

                if not all([chrom, pos, ref is not None, alt is not None]):
                    continue

                gene_rows.append({
                    "variant_id": f"{chrom}:{pos}:{ref}:{alt}",
                    "chrom": chrom,
                    "pos": pos,
                    "ref": ref,
                    "alt": alt,
                    "label": label,
                    "gene_symbol": gene,
                    "classification_raw": e["classification_raw"],
                    "lovd_id": e["lovd_id"],
                })

            logger.info(
                "  %s: %d classified entries, %d coordinate-resolved",
                gene, sum(1 for e in entries if _lovd_label(e["classification_raw"]) is not None),
                resolved,
            )

        all_rows.extend(gene_rows)
        logger.info("  %s: %d usable variants so far (cumulative %d)", gene, len(gene_rows), len(all_rows))
        time.sleep(0.5)

    if not all_rows:
        logger.error("No parseable classified variants found for any gene.")
        raise SystemExit(1)

    df = pd.DataFrame(all_rows).drop_duplicates(subset=["variant_id"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(
        "Saved %d variants (%d pathogenic, %d benign) to %s",
        len(df), int(df["label"].sum()), int((df["label"] == 0).sum()), output_path,
    )
    return df


def _parse_tab_rows(df_raw: pd.DataFrame, gene: str) -> list[dict]:
    """Parse rows from the tab-separated LOVD export."""
    import re
    rows = []
    for _, row in df_raw.iterrows():
        def get(*names):
            for n in names:
                if n in row.index and pd.notna(row[n]) and str(row[n]).strip():
                    return str(row[n]).strip()
            return None

        chrom = get("chromosome", "Chromosome", "chrom")
        pos   = get("position", "Position", "pos")
        ref   = get("ref", "Ref", "ReferenceAllele")
        alt   = get("alt", "Alt", "AlternativeAllele")

        dna_field = get("VariantOnGenome/DNA")
        if (not chrom or not pos or not ref or not alt) and dna_field:
            m = re.match(r"(?:chr)?(\w+):g\.(\d+)([ACGT]+)>([ACGT]+)", str(dna_field))
            if m:
                chrom = chrom or m.group(1)
                pos   = pos   or m.group(2)
                ref   = ref   or m.group(3)
                alt   = alt   or m.group(4)

        if not all([chrom, pos, ref, alt]):
            continue

        chrom = str(chrom).replace("chr", "").replace("Chr", "")
        if chrom == "M":
            chrom = "MT"
        try:
            pos_int = int(float(str(pos)))
        except (ValueError, TypeError):
            continue

        classification = get(
            "pathogenicity", "Pathogenicity", "classification", "Classification",
            "VariantOnGenome/ClinVar", "VariantOnGenome/Pathogenicity",
        ) or ""
        label = _lovd_label(classification)
        if label is None:
            continue

        rows.append({
            "variant_id": f"{chrom}:{pos_int}:{ref.upper()}:{alt.upper()}",
            "chrom": chrom,
            "pos": pos_int,
            "ref": ref.upper(),
            "alt": alt.upper(),
            "label": label,
            "gene_symbol": gene or get("gene", "Gene", "GeneSymbol") or "",
            "classification_raw": classification,
            "lovd_id": get("id", "ID", "VariantOnGenome/DBID") or "",
        })
    return rows


def evaluate_against_model(
    lovd_df: pd.DataFrame,
    model_path: Path,
    output_dir: Path,
) -> None:
    """Score LOVD variants and compute validation metrics."""
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import (
        average_precision_score, brier_score_loss, roc_auc_score,
    )
    from genomic_variant_classifier.api.pipeline import InferencePipeline

    pipeline = InferencePipeline.load(model_path)
    logger.info("Loaded model: val_auroc=%.4f", pipeline.metadata.val_auroc)

    y_true  = lovd_df["label"].values
    y_proba = pipeline.predict_proba(lovd_df)

    auroc = roc_auc_score(y_true, y_proba)
    auprc = average_precision_score(y_true, y_proba)
    brier = brier_score_loss(y_true, y_proba)

    frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="uniform")
    bins   = np.linspace(0, 1, 11)
    counts = np.histogram(y_proba, bins=bins)[0]
    ece    = float(sum((c / len(y_true)) * abs(fp - mp)
                       for fp, mp, c in zip(frac_pos, mean_pred, counts)))

    logger.info(
        "LOVD external validation: AUROC=%.4f  AUPRC=%.4f  Brier=%.4f  ECE=%.4f",
        auroc, auprc, brier, ece,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "validation_type": "lovd_external",
        "n_variants": int(len(lovd_df)),
        "n_pathogenic": int(y_true.sum()),
        "n_benign": int((y_true == 0).sum()),
        "auroc": auroc,
        "auprc": auprc,
        "brier": brier,
        "ece": ece,
        "model_val_auroc": pipeline.metadata.val_auroc,
    }
    (output_dir / "lovd_metrics.json").write_text(json.dumps(metrics, indent=2))

    out_df = lovd_df.copy()
    out_df["predicted_proba"] = y_proba
    out_df.to_parquet(output_dir / "lovd_predictions.parquet", index=False)
    logger.info("LOVD validation results saved to %s", output_dir)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Download and parse LOVD variants for external validation."
    )
    p.add_argument("--output", type=Path,
                   default=Path("data/external/lovd/lovd_variants.parquet"))
    p.add_argument("--genes", nargs="*",
                   default=["BRCA1", "BRCA2", "TP53", "PTEN", "ATM",
                            "MLH1", "MSH2", "MSH6", "APC", "LDLR"],
                   help="Gene list for targeted download.")
    p.add_argument("--lovd-cookie", default=None,
                   help=(
                       "PHPSESSID session cookie from a logged-in LOVD session. "
                       "Enables bulk tab export (faster, more data). "
                       "Obtain by logging into https://databases.lovd.nl/shared/ "
                       "and copying the PHPSESSID cookie value."
                   ))
    p.add_argument("--max-per-gene", type=int, default=5_000,
                   help="Cap on NCBI coordinate-resolution calls per gene.")
    p.add_argument("--model", type=Path, default=None,
                   help="InferencePipeline joblib; if set, runs validation.")
    p.add_argument("--eval-output", type=Path,
                   default=Path("outputs/lovd_validation"))
    args = p.parse_args()

    if args.output.exists():
        logger.info("Loading existing LOVD parquet from %s ...", args.output)
        lovd_df = pd.read_parquet(args.output)
    else:
        lovd_df = download_gene_list(
            genes=args.genes,
            output_path=args.output,
            cookie=args.lovd_cookie,
            max_per_gene=args.max_per_gene,
        )

    logger.info(
        "LOVD dataset: %d variants, %d pathogenic, %d benign",
        len(lovd_df),
        int(lovd_df["label"].sum()),
        int((lovd_df["label"] == 0).sum()),
    )

    if args.model:
        evaluate_against_model(lovd_df, args.model, args.eval_output)


if __name__ == "__main__":
    main()
