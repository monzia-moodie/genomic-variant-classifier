"""
scripts/validate_lovd_enriched.py
==================================
Enrich LOVD variants with gnomAD AF + gene features, then run model validation.

Enrichment steps (in priority order):
  1. gnomAD v4 allele frequency join (variant_id = chrom:pos:ref:alt)
  2. Gene-level features from gene_summary parquet
  3. Consequence derived from variant pattern (SNV / indel / del / dup)

Output: outputs/lovd_validation_full/metrics.json  +  predictions.parquet

Usage:
    python scripts/validate_lovd_enriched.py
    python scripts/validate_lovd_enriched.py --model models/phase4_pipeline_calibrated.joblib
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Consequence inference from variant_id pattern
# ---------------------------------------------------------------------------

def _infer_consequence(variant_id: str) -> str:
    """
    Heuristic consequence from variant_id string 'chrom:pos:ref:alt'.

    LOVD variants in known disease genes are predominantly coding; for SNVs
    we default to missense_variant (the most common pathogenic consequence).
    Indels (ref or alt is '?') are treated as frameshift_variant.
    """
    parts = variant_id.split(":")
    if len(parts) != 4:
        return "missense_variant"
    _, _, ref, alt = parts
    if ref == "?" or alt == "?":
        return "frameshift_variant"
    if len(ref) == 1 and len(alt) == 1:
        return "missense_variant"
    if len(ref) > 1 and len(alt) == 1:
        return "frameshift_variant"   # deletion
    if len(ref) == 1 and len(alt) > 1:
        return "frameshift_variant"   # insertion
    return "missense_variant"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cohort",
        type=Path,
        default=Path("data/external/lovd/lovd_all_variants.parquet"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/phase4_pipeline.joblib"),
    )
    parser.add_argument(
        "--gnomad",
        type=Path,
        default=Path("data/processed/gnomad_v4_exomes.parquet"),
    )
    parser.add_argument(
        "--gene-summary",
        type=Path,
        default=Path("data/processed/gene_summary.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/lovd_validation_full"),
    )
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.api.pipeline import InferencePipeline

    # ---- load cohort ----
    logger.info("Loading LOVD cohort from %s ...", args.cohort)
    df = pd.read_parquet(args.cohort)
    logger.info(
        "Cohort: %d variants  (%d pathogenic, %d benign)",
        len(df), int((df["label"] == 1).sum()), int((df["label"] == 0).sum()),
    )

    # ---- enrich: gnomAD allele frequency ----
    if args.gnomad.exists():
        logger.info("Loading gnomAD AF from %s ...", args.gnomad)
        gnomad = pd.read_parquet(args.gnomad)[["variant_id", "allele_freq"]]
        gnomad = gnomad.rename(columns={"allele_freq": "gnomad_af"})
        df = df.merge(gnomad, on="variant_id", how="left")
        df["allele_freq"] = df["gnomad_af"].fillna(0.0)
        n_matched = df["gnomad_af"].notna().sum()
        logger.info(
            "gnomAD match: %d / %d variants (%.1f%%)",
            n_matched, len(df), 100 * n_matched / len(df),
        )
        df = df.drop(columns=["gnomad_af"])
    else:
        logger.warning("gnomAD parquet not found; allele_freq will default to 0.0")
        df["allele_freq"] = 0.0

    # ---- enrich: gene-level features ----
    if args.gene_summary.exists():
        logger.info("Joining gene summary features ...")
        gs = pd.read_parquet(args.gene_summary)
        df = df.merge(gs, on="gene_symbol", how="left")
        logger.info("Gene summary joined.")
    else:
        logger.warning("gene_summary parquet not found.")

    # ---- enrich: AlphaMissense ----
    am_path = Path("data/external/alphamissense/am_lovd_genes.parquet")
    if am_path.exists():
        logger.info("Joining AlphaMissense scores from %s ...", am_path)
        am = pd.read_parquet(am_path)
        df = df.merge(am, on="variant_id", how="left")
        n_am = df["am_pathogenicity"].notna().sum()
        logger.info(
            "AlphaMissense match: %d / %d SNV variants (%.1f%%)",
            n_am, len(df), 100 * n_am / len(df),
        )
        # engineer_features() reads alphamissense_score; fill missing with 0.5 (not covered)
        df["alphamissense_score"] = df["am_pathogenicity"].fillna(0.5)
        df = df.drop(columns=["am_pathogenicity"])
    else:
        logger.warning("AlphaMissense parquet not found; alphamissense_score defaults to 0.5")

    # ---- enrich: consequence ----
    df["consequence"] = df["variant_id"].apply(_infer_consequence)
    logger.info(
        "Consequence distribution:\n%s",
        df["consequence"].value_counts().to_string(),
    )

    # ---- load model ----
    logger.info("Loading model from %s ...", args.model)
    pipeline = InferencePipeline.load(args.model)
    logger.info(
        "Model loaded: val_auroc=%.4f  features=%d",
        pipeline.metadata.val_auroc,
        len(pipeline.metadata.feature_names) if pipeline.metadata.feature_names else "?",
    )

    # ---- score ----
    logger.info("Scoring %d variants ...", len(df))
    scores_chunks = []
    batch_size = 2048
    for start in range(0, len(df), batch_size):
        chunk = df.iloc[start:start + batch_size]
        scores_chunks.append(pipeline.predict_proba(chunk))
    y_prob = np.concatenate(scores_chunks)
    y_true = df["label"].astype(int).values

    # ---- overall metrics ----
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    logger.info("Overall: AUROC=%.4f  AUPRC=%.4f  Brier=%.4f", auroc, auprc, brier)

    # ---- per-gene metrics ----
    df["_prob"] = y_prob
    gene_metrics: dict[str, dict] = {}
    for gene, grp in df.groupby("gene_symbol"):
        if grp["label"].nunique() < 2:
            continue
        ga = roc_auc_score(grp["label"].values, grp["_prob"].values)
        gene_metrics[gene] = {
            "auroc": round(ga, 4),
            "n": len(grp),
            "n_pathogenic": int((grp["label"] == 1).sum()),
            "n_benign": int((grp["label"] == 0).sum()),
        }
        logger.info(
            "  %-8s  AUROC=%.4f  n=%d  (%d path / %d benign)",
            gene, ga, len(grp),
            gene_metrics[gene]["n_pathogenic"],
            gene_metrics[gene]["n_benign"],
        )

    # ---- SNV-only subset ----
    snv_mask = (df["ref"].notna()) & (df["ref"] != "?") & (df["alt"].notna()) & (df["alt"] != "?")
    if snv_mask.sum() >= 10 and df.loc[snv_mask, "label"].nunique() == 2:
        snv_auroc = roc_auc_score(
            df.loc[snv_mask, "label"].values,
            df.loc[snv_mask, "_prob"].values,
        )
        logger.info(
            "SNV-only subset (n=%d): AUROC=%.4f", snv_mask.sum(), snv_auroc
        )
    else:
        snv_auroc = None

    # ---- save ----
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "validation_type": "lovd_external_enriched",
        "genes": sorted(df["gene_symbol"].unique().tolist()),
        "n_variants": int(len(df)),
        "n_pathogenic": int((y_true == 1).sum()),
        "n_benign": int((y_true == 0).sum()),
        "n_snv": int(snv_mask.sum()),
        "auroc": round(auroc, 4),
        "auprc": round(auprc, 4),
        "brier": round(brier, 4),
        "auroc_snv_only": round(snv_auroc, 4) if snv_auroc is not None else None,
        "per_gene": gene_metrics,
        "model": args.model.name,
        "gnomad_match_rate": round(
            float(df["allele_freq"].gt(0).sum()) / len(df), 4
        ),
    }

    out_metrics = args.output_dir / "metrics.json"
    out_metrics.write_text(json.dumps(metrics, indent=2))
    logger.info("Metrics saved to %s", out_metrics)

    out_preds = args.output_dir / "predictions.parquet"
    df["predicted_proba"] = y_prob
    df.to_parquet(out_preds, index=False)
    logger.info("Predictions saved to %s", out_preds)

    # ---- copy to Google Drive ----
    gdrive_dir = Path("G:/My Drive/genomic-classifier-results/lovd_validation")
    if gdrive_dir.exists():
        import shutil
        shutil.copy2(out_metrics, gdrive_dir / "metrics.json")
        shutil.copy2(out_preds,   gdrive_dir / "predictions.parquet")
        logger.info("Results copied to Google Drive.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
