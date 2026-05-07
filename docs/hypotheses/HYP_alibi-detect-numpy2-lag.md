# HYP: alibi-detect dropped from drift stack — numpy 2.x lag

## Observation
alibi-detect 0.13.0 (latest, released 2025-12-11) still pins
`numpy<2.0.0,>=1.16.2` in its wheel METADATA, despite numpy 2.0 having
been GA since June 2024. No newer release exists. The PyPI project
landing page does not surface this runtime constraint; only `pip-compile`
against the wheel reveals it.

## Decision
Drop alibi-detect from `requirements.in` rather than block the project's
numpy 2.x baseline. Replace functionality with:
- scipy.stats (KS, chi-squared)
- nannyml (CBPE, performance estimation)
- river (ADWIN, Page-Hinkley, streaming drift)
- evidently (tabular distribution drift dashboards)
- sklearn (Mahalanobis, IsolationForest)

## Implications for Deliverable 1
- §0.2 install command must drop alibi-detect.
- AdversarialSubmissionAgent rules R1-R9 do not depend on alibi-detect
  (verified — ruleset is statistical/ClinVar-metadata-based, not
  alibi-detect ML-detector-based). No agent re-architecture needed.

## Promotion criteria
Promote to docs/validated/ once: (a) lockfile compiles without
alibi-detect, (b) drift agents implement against the replacement stack,
(c) Run-9 KAN drift comparison shows the replacement stack catches
known synthetic drift cases.