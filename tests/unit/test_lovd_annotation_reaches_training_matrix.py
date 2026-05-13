"""
tests/unit/test_lovd_annotation_reaches_training_matrix.py
===========================================================
Regression test for INCIDENT_2026-05-02_lovd-silent-zero.md.

Background
----------
Run 9's outputs/run9_ready/splits/X_train.parquet had
`lovd_variant_class == 0` for all 1,197,216 rows despite the LOVD parquet
on disk being structurally healthy and the LOVD connector being
unconditionally invoked. INCIDENT's two original candidates (downstream
overwrite, upstream coordinate transform) were both falsified; the
actual root cause was that `scripts/run_phase2_eval.py` constructed
`AnnotationConfig(...)` without passing `lovd_path`, so the connector
took its silent "no parquet loaded" branch.

This test enforces the post-condition that the INCIDENT's R10-B step
requires: after the full ETL pipeline runs against a fixture where the
ClinVar inputs DO have matching LOVD entries, at least one row in the
returned feature matrix has `lovd_variant_class > 0`.

The test exercises the FULL connector + feature-engineering chain (not
just LOVDConnector in isolation — the diagnostic merge already proved
the connector works standalone; the silent-zero bug was at a layer
above the connector). Pattern modeled on
`tests/unit/test_spliceai_parquet_default.py` and
`tests/unit/test_esm2_activation.py`.

Skip behaviour
--------------
If the project's `LOVDConnector` / `AnnotationConfig` / `DataPrepPipeline`
are not importable from `genomic_variant_classifier.data.real_data_prep`
(e.g. the C5 package-consolidation migration is in flight), the test
SKIPS rather than failing — so it can land alongside the patch and only
activate once the consolidation lands. Once the imports are stable, the
skip path should be removed in a follow-up commit.
"""
from __future__ import annotations

from pathlib import Path
import pytest


# Importability guard — skip if pipeline pieces aren't available yet.
_lovd_pieces_available = True
_skip_reason = ""
try:
    from genomic_variant_classifier.data.lovd import LOVDConnector  # noqa: F401
except ImportError as exc:
    _lovd_pieces_available = False
    _skip_reason = f"LOVDConnector not importable: {exc}"

try:
    from genomic_variant_classifier.data.real_data_prep import (  # noqa: F401
        AnnotationConfig,
        DataPrepConfig,
        DataPrepPipeline,
    )
except ImportError as exc:
    _lovd_pieces_available = False
    _skip_reason = f"DataPrepPipeline not importable: {exc}"


pytestmark = pytest.mark.skipif(
    not _lovd_pieces_available,
    reason=_skip_reason or "LOVD pipeline pieces unavailable",
)


@pytest.fixture
def tiny_lovd_parquet(tmp_path):
    """Single-gene LOVD parquet that we know matches the fixture ClinVar row.

    Schema matches `scripts/build_lovd_index.py` output:
    chrom, pos, ref, alt, label, gene_symbol, classification_raw,
    source_format, variant_id.
    """
    import pandas as pd

    df = pd.DataFrame({
        "chrom": ["17"],
        "pos": [7675234],
        "ref": ["G"],
        "alt": ["T"],
        "label": [1],
        "gene_symbol": ["TP53"],
        "classification_raw": ["pathogenic"],
        "source_format": ["lovd_tab"],
        "variant_id": ["17:7675234:G:T"],
    })
    p = tmp_path / "lovd_fixture.parquet"
    df.to_parquet(p, index=False)
    return p


@pytest.fixture
def tiny_clinvar_parquet(tmp_path):
    """5-row ClinVar fixture with one variant intentionally matching the
    LOVD fixture above (chr17:7675234 G>T). Other 4 rows do not.
    """
    import pandas as pd

    df = pd.DataFrame({
        "chrom": ["17", "17", "1", "13", "5"],
        "pos":   [7675234, 7670000, 100000, 32316461, 112116000],
        "ref":   ["G",     "A",     "C",    "T",      "G"],
        "alt":   ["T",     "G",     "T",    "A",      "C"],
        "gene_symbol":   ["TP53", "TP53", "GENE_X", "BRCA2", "APC"],
        "clinical_sig":  ["Pathogenic", "Benign", "Pathogenic",
                          "Benign", "Pathogenic"],
        "review_status": ["criteria provided, multiple submitters, no conflicts"] * 5,
        "consequence":   ["missense_variant"] * 5,
    })
    p = tmp_path / "clinvar_fixture.parquet"
    df.to_parquet(p, index=False)
    return p


def test_lovd_annotation_reaches_training_matrix(
    tmp_path, tiny_lovd_parquet, tiny_clinvar_parquet
):
    """Post-condition: with LOVD wired via AnnotationConfig.lovd_path,
    the chr17:7675234 G>T row carries lovd_variant_class > 0 through
    the full pipeline (connector -> _engineer_features -> output matrix).

    Pre-Run-10 behaviour: this assertion fails because
    scripts/run_phase2_eval.py never passed lovd_path. See INCIDENT
    INCIDENT_2026-05-02_lovd-silent-zero.md.
    """
    ann = AnnotationConfig(lovd_path=Path(tiny_lovd_parquet))
    cfg = DataPrepConfig(
        min_review_tier=0,  # accept the fixture's review_status
        output_dir=tmp_path / "splits",
        scale_features=False,
        test_fraction=0.4,
        random_state=42,
        require_both_classes=False,  # Phase 1.5c: 5-row fixture too small for class-balanced split
    )
    pipeline = DataPrepPipeline(config=cfg, annotation_config=ann)

    # Exact return signature is allowed to vary across HEAD revisions;
    # we destructure carefully and only assert on what matters.
    result = pipeline.run(clinvar_path=str(tiny_clinvar_parquet))
    # Standard tuple ordering:
    # (X_train, X_val, X_test, y_train, y_val, y_test, meta_val, meta_test)
    X_train = result[0]

    assert "lovd_variant_class" in X_train.columns, (
        "lovd_variant_class column missing from training matrix — "
        "feature engineering may have dropped it"
    )

    # The chr17:7675234 G>T row matches the LOVD fixture (classification
    # 'pathogenic' maps to ordinal class 4 per the connector's mapping;
    # accept anything > 0 here to stay robust to mapping refinements).
    n_nonzero = int((X_train["lovd_variant_class"] > 0).sum())
    assert n_nonzero >= 1, (
        f"Expected at least one row with lovd_variant_class > 0 in "
        f"training matrix; got {n_nonzero}. This is the regression "
        f"INCIDENT_2026-05-02_lovd-silent-zero.md guards against. "
        f"value_counts:\n{X_train['lovd_variant_class'].value_counts().to_dict()}"
    )


def test_lovd_annotation_silent_zero_when_path_omitted(
    tmp_path, tiny_clinvar_parquet
):
    """Inverse post-condition: when lovd_path is NOT passed to
    AnnotationConfig, lovd_variant_class is identically 0 across the
    whole training matrix. Documents the failure mode that Run 9 hit.
    """
    ann = AnnotationConfig()  # no lovd_path -> silent zero
    cfg = DataPrepConfig(
        min_review_tier=0,
        output_dir=tmp_path / "splits_no_lovd",
        scale_features=False,
        test_fraction=0.4,
        random_state=42,
        require_both_classes=False,  # Phase 1.5c: 5-row fixture too small for class-balanced split
    )
    pipeline = DataPrepPipeline(config=cfg, annotation_config=ann)
    result = pipeline.run(clinvar_path=str(tiny_clinvar_parquet))
    X_train = result[0]

    assert "lovd_variant_class" in X_train.columns
    assert (X_train["lovd_variant_class"] == 0).all(), (
        "Expected silent-zero behaviour when lovd_path is omitted; "
        "got non-zero values. The connector's default branch may have "
        "changed — update both this test and "
        "INCIDENT_2026-05-02_lovd-silent-zero.md if intentional."
    )
