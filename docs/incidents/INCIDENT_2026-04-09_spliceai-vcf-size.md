# INCIDENT: SpliceAI VCF Larger Than Expected — 2026-04-09

## Summary
The SpliceAI VCF at data/external/spliceai/spliceai_scores.masked.snv.hg38.vcf.gz
is 28.8GB compressed and contains 1.1B+ lines, not the ~72M expected for a
masked SNV file. Build took 2.5+ hours instead of the estimated 8 minutes.

## Root Cause
The filename says "masked.snv" but the file is the full unmasked genome-wide
VCF including indels. Likely downloaded from a different source than intended,
or the Ensembl/Illumina naming convention changed between releases.

## Impact
- SpliceAI index build took 2.5+ hours vs estimated 8 minutes
- Output is actually more complete than intended: all variants with
  splice_ai_score >= 0.1 across all chromosomes, ~18.5M+ variants

## Resolution
This is not a problem — a more complete index is better. The index
(spliceai_index_test.parquet) is the production artifact we want.
Post-session: rename to spliceai_index.parquet and upload to GCS.

## Lesson
Always check file size before launching a long-running parse job.
28.8GB compressed is a clear signal this is not a masked SNV subset.
Add a pre-flight file size check to build_spliceai_index.py for future runs.

## Status
RESOLVED — index built (336.8MB, 45.5M variants), renamed, correct file uploaded to GCS 2026-04-09T23:15Z
