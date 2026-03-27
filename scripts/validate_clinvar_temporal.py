"""
scripts/validate_clinvar_temporal.py
=====================================
ClinVar temporal holdout validation -- Step 7A.

Evaluates the InferencePipeline on ClinVar variants last-evaluated *after* the
training data cutoff date.  Uses the ClinVar variant_summary.txt.gz from NCBI
FTP (free, no auth):
  https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz

Limitations vs. full external validation
-----------------------------------------
- Same label source (ClinVar) -- shares submitter and review biases
- Reviews from the same clinical centres may share systematic errors
- Treats this as a weak external validation; LOVD and UK Biobank are stronger

Metrics produced
----------------
  AUROC, AUPRC, Brier score, ECE (15 bins)
  Per-threshold sensitivity, specificity, PPV, NPV, F1
  Label distribution of new variants (new pathogenic / benign / VUS)
  Overlap with training set (fraction of new variants already in training)

Usage
-----
  # With fresh variant_summary.txt.gz (recommended -- has LastEvaluated dates)
  python scripts/validate_clinvar_temporal.py \\
      --clinvar    data/external/clinvar_fresh/variant_summary.txt.gz \\
      --splits-dir outputs/phase2_with_gnomad/splits \\
      --model      models/phase2_pipeline.joblib \\
      --cutoff     2024-01-01 \\
      --output     outputs/temporal_validation

  # Download variant summary on the fly and run
  python scripts/validate_clinvar_temporal.py \\
      --download   \\
      --cutoff     2024-01-01 \\
      --output     outputs/temporal_validation

  # With pre-processed ClinVar parquet (must contain a date column)
  python scripts/validate_clinvar_temporal.py \\
      --clinvar    data/processed/clinvar_grch38.parquet \\
      --splits-dir outputs/phase2_with_gnomad/splits \\
      --cutoff     2024-01-01 \\
      --output     outputs/temporal_validation
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("validate_clinvar_temporal")

_CLINVAR_SUMMARY_URL = (
    "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
)

# ClinSigSimple: 1=pathogenic, 0=benign, -1=other/conflicting
_BINARY_LABELS = {1: 1, 0: 0}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _download_variant_summary(dest: Path) -> None:
    import requests
    logger.info("Downloading ClinVar variant_summary.txt.gz (~435 MB) ...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(_CLINVAR_SUMMARY_URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        written = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                f.write(chunk)
                written += len(chunk)
                if total:
                    sys.stderr.write(
                        f"\r  {written / 1e6:.0f}/{total / 1e6:.0f} MB"
                        f"  ({100 * written / total:.0f}%)"
                    )
    sys.stderr.write("\n")
    logger.info("Downloaded to %s (%d MB)", dest, dest.stat().st_size // 1_000_000)


def _load_variant_summary(path: Path, cutoff: str = "2000-01-01") -> pd.DataFrame:
    """
    Parse ClinVar variant_summary.txt.gz into a DataFrame with columns:
      variant_id, chrom, pos, ref, alt, label, gene_symbol, LastEvaluated,
      clinical_sig
    Filters to GRCh38, biallelic SNV/indel, ClinSigSimple in {0, 1},
    and LastEvaluated > cutoff.  Uses chunked reading to handle multi-GB files.
    """
    logger.info("Parsing %s (chunked) ...", path)
    opener = gzip.open if str(path).endswith(".gz") else open

    # Read header first to get column names and detect schema
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        raw_header = f.readline().rstrip("\n")
    col_names = [c.lstrip("#") for c in raw_header.split("\t")]

    ref_col = "ReferenceAlleleVCF" if "ReferenceAlleleVCF" in col_names else "ReferenceAllele"
    alt_col = "AlternateAlleleVCF" if "AlternateAlleleVCF" in col_names else "AlternateAllele"
    pos_col = "PositionVCF"        if "PositionVCF"        in col_names else "Start"

    keep_raw = list({
        "Assembly", "Chromosome", pos_col, ref_col, alt_col,
        "ClinSigSimple", "LastEvaluated", "GeneSymbol", "ClinicalSignificance",
    } & set(col_names))

    cutoff_dt = pd.Timestamp(cutoff)
    chunks: list[pd.DataFrame] = []

    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        reader = pd.read_csv(
            f,
            sep="\t",
            header=0,
            names=col_names,
            usecols=keep_raw,
            dtype=str,
            chunksize=200_000,
            on_bad_lines="skip",
        )
        for i, chunk in enumerate(reader):
            # GRCh38 only
            chunk = chunk[chunk["Assembly"].str.upper() == "GRCH38"]
            if chunk.empty:
                continue

            # Binary label
            chunk["ClinSigSimple"] = pd.to_numeric(chunk["ClinSigSimple"], errors="coerce")
            chunk = chunk[chunk["ClinSigSimple"].isin([0, 1])]
            if chunk.empty:
                continue

            # Date filter (fast string-based pre-check, then proper parse for kept rows)
            if "LastEvaluated" in chunk.columns:
                chunk["LastEvaluated"] = pd.to_datetime(
                    chunk["LastEvaluated"], errors="coerce", format="mixed"
                )
                chunk = chunk[chunk["LastEvaluated"] > cutoff_dt]
                if chunk.empty:
                    continue

            # Allele filters
            ref_s = chunk[ref_col].astype(str)
            alt_s = chunk[alt_col].astype(str)
            mask = (
                ref_s.notna() & alt_s.notna()
                & ~ref_s.isin(["na", ".", "N", "-", "nan"])
                & ~alt_s.isin(["na", ".", "N", "-", "nan"])
                & (ref_s.str.len() <= 50)
                & (alt_s.str.len() <= 50)
            )
            chunk = chunk[mask]
            if chunk.empty:
                continue

            # Normalise
            chunk["chrom"] = (
                chunk["Chromosome"].astype(str)
                .str.replace("chr", "", regex=False)
                .str.strip()
            )
            chunk.loc[chunk["chrom"] == "M", "chrom"] = "MT"
            chunk["pos"]   = pd.to_numeric(chunk[pos_col], errors="coerce")
            chunk["ref"]   = ref_s.str.upper()
            chunk["alt"]   = alt_s.str.upper()
            chunk["label"] = chunk["ClinSigSimple"].astype(int)
            chunk["gene_symbol"]  = chunk.get("GeneSymbol", pd.Series("", index=chunk.index)).astype(str).str.strip()
            chunk["clinical_sig"] = chunk.get("ClinicalSignificance", pd.Series("", index=chunk.index)).astype(str)

            chunk = chunk.dropna(subset=["pos"])
            chunk["pos"] = chunk["pos"].astype(int)
            chunk["variant_id"] = (
                chunk["chrom"] + ":" + chunk["pos"].astype(str) + ":"
                + chunk["ref"]  + ":" + chunk["alt"]
            )

            keep = ["variant_id", "chrom", "pos", "ref", "alt", "label",
                    "gene_symbol", "LastEvaluated", "clinical_sig"]
            chunks.append(chunk[[c for c in keep if c in chunk.columns]])

            if (i + 1) % 5 == 0:
                n_so_far = sum(len(c) for c in chunks)
                logger.info("  Chunk %d processed — %d variants kept so far", i + 1, n_so_far)

    if not chunks:
        logger.error("No qualifying variants found in %s", path)
        raise SystemExit(1)

    result = pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["variant_id"])
    logger.info(
        "Parsed: %d biallelic SNV/indel GRCh38 variants after cutoff "
        "(%d pathogenic, %d benign)",
        len(result), int(result["label"].sum()), int((result["label"] == 0).sum()),
    )
    return result


def _load_clinvar_parquet(path: Path) -> pd.DataFrame:
    """Load a pre-processed ClinVar parquet (may or may not have date columns)."""
    df = pd.read_parquet(path)
    logger.info("Loaded parquet: %d rows, columns: %s", len(df), list(df.columns)[:12])
    return df


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _ece(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 15) -> float:
    frac_pos, mean_pred = calibration_curve(
        y_true, y_proba, n_bins=n_bins, strategy="uniform"
    )
    bins   = np.linspace(0, 1, n_bins + 1)
    counts = np.histogram(y_proba, bins=bins)[0]
    return float(sum((c / len(y_true)) * abs(fp - mp)
                     for fp, mp, c in zip(frac_pos, mean_pred, counts)))


def _threshold_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    ppv  = tp / max(tp + fp, 1)
    npv  = tn / max(tn + fn, 1)
    f1   = 2 * ppv * sens / max(ppv + sens, 1e-9)
    return dict(threshold=threshold, sensitivity=sens, specificity=spec,
                ppv=ppv, npv=npv, f1=f1, tp=tp, tn=tn, fp=fp, fn=fn)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="ClinVar temporal holdout validation.")
    p.add_argument(
        "--clinvar", type=Path, default=None,
        help=(
            "Path to variant_summary.txt.gz (recommended) or a pre-processed "
            "ClinVar parquet.  If omitted, use --download to fetch it."
        ),
    )
    p.add_argument(
        "--download", action="store_true",
        help=(
            "Download variant_summary.txt.gz from NCBI FTP before running. "
            "Saved to data/external/clinvar_fresh/variant_summary.txt.gz."
        ),
    )
    p.add_argument("--splits-dir", type=Path,
                   default=Path("outputs/phase2_with_gnomad/splits"),
                   help="Directory with X_train.parquet / X_val.parquet for deduplication.")
    p.add_argument("--model", type=Path, default=Path("models/phase2_pipeline.joblib"))
    p.add_argument("--cutoff", type=str, default="2024-01-01",
                   help="ISO date; variants last-evaluated after this date form the test set.")
    p.add_argument("--output", type=Path, default=Path("outputs/temporal_validation"))
    args = p.parse_args()

    fresh_gz = Path("data/external/clinvar_fresh/variant_summary.txt.gz")

    # -----------------------------------------------------------------------
    # Resolve ClinVar source
    # -----------------------------------------------------------------------
    clinvar_path = args.clinvar
    if args.download or (clinvar_path is None and not fresh_gz.exists()):
        _download_variant_summary(fresh_gz)
        clinvar_path = fresh_gz
    elif clinvar_path is None:
        clinvar_path = fresh_gz

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    logger.info("Loading model from %s ...", args.model)
    from src.api.pipeline import InferencePipeline
    pipeline = InferencePipeline.load(args.model)
    logger.info("Model val_auroc=%.4f, n_features=%d",
                pipeline.metadata.val_auroc, pipeline.metadata.n_features)

    # -----------------------------------------------------------------------
    # Load ClinVar data
    # -----------------------------------------------------------------------
    suffix = clinvar_path.suffix.lower()
    if suffix in (".gz", ".tsv", ".txt") or clinvar_path.name.endswith(".txt.gz"):
        # Chunked parser does date filtering internally — pass cutoff to avoid
        # loading all 4M+ rows into memory before the temporal split.
        clinvar = _load_variant_summary(clinvar_path, cutoff=args.cutoff)
        date_col = "LastEvaluated"
        already_filtered = True
    else:
        clinvar = _load_clinvar_parquet(clinvar_path)
        date_col = next(
            (c for c in ("LastEvaluated", "last_evaluated", "SubmissionDate",
                         "date_last_evaluated") if c in clinvar.columns),
            None,
        )
        already_filtered = False

    # -----------------------------------------------------------------------
    # Build training ID set for deduplication
    # -----------------------------------------------------------------------
    train_ids: set[str] = set()
    for split in ("X_train", "X_val"):
        f = args.splits_dir / f"{split}.parquet"
        if f.exists():
            split_df = pd.read_parquet(f)
            id_src = "variant_id" if "variant_id" in split_df.columns else None
            if id_src:
                train_ids.update(split_df[id_src].astype(str))
            else:
                train_ids.update(split_df.index.astype(str))
    logger.info("Training+val set size (for deduplication): %d", len(train_ids))

    # -----------------------------------------------------------------------
    # Temporal filter (only needed for parquet input; gz already filtered)
    # -----------------------------------------------------------------------
    if already_filtered:
        new_variants = clinvar
    elif date_col and date_col in clinvar.columns:
        clinvar[date_col] = pd.to_datetime(clinvar[date_col], errors="coerce")
        new_variants = clinvar[clinvar[date_col] > args.cutoff].copy()
        logger.info("Variants last-evaluated after %s: %d", args.cutoff, len(new_variants))
    else:
        logger.warning(
            "No date column found -- using all %d variants (no temporal split).", len(clinvar)
        )
        new_variants = clinvar.copy()

    # -----------------------------------------------------------------------
    # Deduplicate against training set
    # -----------------------------------------------------------------------
    if "variant_id" in new_variants.columns and train_ids:
        before = len(new_variants)
        new_variants = new_variants[
            ~new_variants["variant_id"].astype(str).isin(train_ids)
        ].copy()
        logger.info(
            "After removing training overlap: %d (removed %d)",
            len(new_variants), before - len(new_variants),
        )

    # -----------------------------------------------------------------------
    # Resolve label column
    # -----------------------------------------------------------------------
    label_col = next(
        (c for c in ("label", "ClinSigSimple", "acmg_label", "pathogenic")
         if c in new_variants.columns),
        None,
    )
    if label_col is None:
        logger.error("No label column found in data.")
        raise SystemExit(1)

    labeled = new_variants[new_variants[label_col].isin([0, 1])].copy()
    if len(labeled) < 50:
        logger.error("Only %d labeled variants after cutoff — too few to evaluate.", len(labeled))
        raise SystemExit(1)

    logger.info(
        "Labeled temporal holdout: %d total, %d pathogenic (%.1f%%), %d benign",
        len(labeled),
        int(labeled[label_col].sum()),
        100 * labeled[label_col].mean(),
        int((labeled[label_col] == 0).sum()),
    )

    # -----------------------------------------------------------------------
    # Score
    # -----------------------------------------------------------------------
    logger.info("Scoring %d variants ...", len(labeled))
    y_proba = pipeline.predict_proba(labeled)
    y_true  = labeled[label_col].values.astype(int)

    # -----------------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------------
    auroc = roc_auc_score(y_true, y_proba)
    auprc = average_precision_score(y_true, y_proba)
    brier = brier_score_loss(y_true, y_proba)
    ece   = _ece(y_true, y_proba)

    threshold_results = [
        _threshold_metrics(y_true, y_proba, t)
        for t in [0.30, 0.50, 0.70, 0.90]
    ]

    fpr, tpr, _ = roc_curve(y_true, y_proba)

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    args.output.mkdir(parents=True, exist_ok=True)

    metrics = {
        "validation_type": "clinvar_temporal_holdout",
        "cutoff_date":     args.cutoff,
        "n_variants":      int(len(labeled)),
        "n_pathogenic":    int(y_true.sum()),
        "n_benign":        int((y_true == 0).sum()),
        "auroc":           auroc,
        "auprc":           auprc,
        "brier":           brier,
        "ece":             ece,
        "model_val_auroc": pipeline.metadata.val_auroc,
        "threshold_metrics": threshold_results,
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_parquet(
        args.output / "roc_curve.parquet", index=False
    )
    labeled_out = labeled.copy()
    labeled_out["predicted_proba"] = y_proba
    labeled_out.to_parquet(args.output / "predictions.parquet", index=False)

    logger.info("Results saved to %s", args.output)
    logger.info(
        "\n=== Temporal Holdout Summary ===\n"
        "  Cutoff:  %s\n"
        "  AUROC:   %.4f  (training val: %.4f)\n"
        "  AUPRC:   %.4f\n"
        "  Brier:   %.4f\n"
        "  ECE:     %.4f\n"
        "  N:       %d  (%d pathogenic, %d benign)",
        args.cutoff,
        auroc, pipeline.metadata.val_auroc,
        auprc, brier, ece,
        len(labeled), int(y_true.sum()), int((y_true == 0).sum()),
    )

    # Threshold table
    header = f"{'Threshold':>10}{'Sens':>8}{'Spec':>8}{'PPV':>8}{'NPV':>8}{'F1':>8}"
    logger.info("\n%s\n%s", header, "-" * len(header))
    for r in threshold_results:
        logger.info(
            "%10.2f%8.4f%8.4f%8.4f%8.4f%8.4f",
            r["threshold"], r["sensitivity"], r["specificity"],
            r["ppv"], r["npv"], r["f1"],
        )


if __name__ == "__main__":
    main()
