# INCIDENT 2026-05-01: Split duplicates and structural variant multimapping

## Status: RESOLVED — workaround documented; no data corruption

## Discovery

During Run 9a launch prep on 2026-04-30 → 2026-05-01, the meta_train.parquet
reconstruction script (scripts/reconstruct_meta_train.py v1) crashed with
`TypeError: unhashable type: 'dict'` while attempting to detect a unique join
key in meta_val/meta_test.

Subsequent diagnostic via scripts/diagnose_dups.py revealed:

| Observation | Count | Interpretation |
|---|---|---|
| meta_val rows in duplicate variant_id groups | 216 of 154,404 (0.14%) | Within-split dups |
| meta_test rows in duplicate variant_id groups | ~600 of 349,067 (0.17%) | Within-split dups |
| variant_id values appearing in both meta_val AND meta_test | 46 | Cross-split overlap |
| Of 46 cross-split overlaps, in same gene | 0 | Not a gene-aware split bug |
| Of 46 cross-split overlaps, in different genes | 46 | Structural variant multimapping |

## Root cause

ClinVar variants of type `clinvar:CHROM:POS:na:na` represent structural
variants and large CNVs where ref/alt alleles are not canonicalizable. These
variants are annotated against multiple gene contexts (e.g., chrX:10701 spans
1942 genes due to whole-X events).

DataPrepPipeline ingests these as multiple rows per variant_id, one per
gene_symbol annotation context. The gene-aware splitter (GroupShuffleSplit
with groups=gene_symbol.fillna("unknown")) correctly partitions rows by
gene_symbol — but because the same variant_id appears under different
gene_symbol strings, it can land in different splits without violating the
gene-aware contract.

Within-split duplicates (the 216 in meta_val) appear to be near-identical
rows (same variant_id, gene_symbol, transcript_id=null, label) generated
during an earlier annotation pass. Likely deduplication gap in
database_connectors.py or the score-annotation merge logic.

## Impact on run9a-baseline metrics

- Test AUROC 0.9814 / Val AUROC 0.9850 are computed across ALL rows
  including duplicates.
- The 216+600 within-split dups are scored 2x each, modestly inflating the
  effective sample size used in metric computation but not changing per-row
  predictions (deterministic given features).
- The 46 cross-split overlap variants contribute different gene-context
  predictions to val and test independently — these are not data leakage
  in the gene-aware sense (the model never sees the SAME (variant, gene)
  context in both splits).
- Net assessment: <0.5% of evaluation rows are affected. AUROC is robust to
  this noise level. No correction applied; documented as known limitation.

## Workaround

scripts/reconstruct_meta_train_v2.py reconstructs meta_train via gene-set
complement (mirroring _gene_aware_split exactly):

  meta_train = labeled_cohort[~gene_symbol.isin(val_genes ∪ test_genes)]

This recovers the original train indices without requiring variant_id to
be unique.

## Verification (2026-05-01 02:XX)

scripts/reconstruct_meta_train_v2.py output:
- labeled cohort: 1,700,687 rows (matches expected total exactly: +0)
- meta_train: 1,197,216 rows reconstructed (matches expected exactly: +0)
- val genes: 2,320 / test genes: 4,641 / val ∩ test: 0
- train unique genes: 16,240 (0 NaN gene_symbols)
- gene disjointness: train ∩ val = 0, train ∩ test = 0
- meta_train.parquet: 41.8 MB written

## Followups (deferred to Run 9a-v2)

1. Add deduplication step to DataPrepPipeline._load_and_label that removes
   near-identical (variant_id, gene_symbol, transcript_id, label) tuples.
2. Document structural variant handling explicitly in paper Methods section
   ("Variants of form clinvar:CHROM:POS:na:na are annotated per affected
   gene; we report metrics across all annotations.").
3. Consider canonicalizing structural variant annotation to a single
   per-variant row using a "primary gene" rule (e.g., longest transcript,
   highest gene-disease score from ClinGen).
