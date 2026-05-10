---
id: RULE_transformers-5x-esm2-automodel-load-report
title: transformers 5.x AutoModel drops ESM-2 LM head, random-inits pooler
status: validated
date: 2026-05-07
evidence: smoke-2026-05-07-transformers-5x
---

# RULE: transformers 5.x AutoModel correctly drops ESM-2 LM head, random-inits pooler

## Observation

Loading `facebook/esm2_t6_8M_UR50D` via `transformers.AutoModel.from_pretrained`
on transformers 5.8.0 produces an `EsmModel` (base encoder, no MLM head) with
**7,511,801 parameters** total. The new LOAD REPORT diagnostic flags:

- 5 keys UNEXPECTED: `lm_head.{bias, dense.weight, dense.bias, layer_norm.weight, layer_norm.bias}`
  Checkpoint stores MLM head; AutoModel correctly drops it.
- 2 keys MISSING: `pooler.dense.{weight, bias}`
  ESM checkpoints don't include a CLS-pooler; randomly initialized.

Both diagnostics are expected and consistent with transformers 4.x behavior.
The LOAD REPORT format itself is new in transformers 5.x.

## Implication for downstream code

- For per-residue embedding work (HGVSp parser, ESM-2 stub feeding into ensemble
  base models): use `model(input_ids).last_hidden_state`. All weights are
  correctly loaded.
- Do NOT use `model(input_ids).pooler_output` without first training the
  pooler on a downstream task; those weights are random.
- The 7,511,801 figure is the canonical base-encoder param count and should be
  the expected value in `src/genomic_variant_classifier/data/hgvsp_parser.py` smoke tests; NOT the 8M
  figure on the model card (which includes the LM head).

## Verification

```python
from transformers import AutoTokenizer, AutoModel
m = AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
assert sum(p.numel() for p in m.parameters()) == 7_511_801
```

## Promotion criteria met

- Empirically confirmed via smoke test 2026-05-07
- Behavior consistent with transformers 4.x semantics (no regression)
- Implications documented for downstream code