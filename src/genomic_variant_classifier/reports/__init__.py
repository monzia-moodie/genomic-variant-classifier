"""
src/genomic_variant_classifier/reports
===========
Report generation package for the Genomic Variant Classifier.

CHANGES FROM PHASE 1:
  - This __init__.py did not exist; src/reports/ was not a Python package,
    so `from genomic_variant_classifier.reports.report_generator import ...` raised ModuleNotFoundError
    even after the module was written to disk (Issue D fixed).
"""

from __future__ import annotations

from genomic_variant_classifier.reports.report_generator import (
    ReportGenerator,
    ValidationMetrics,
    bootstrap_metric,
    compute_variant_phenotype_association,
    plot_calibration,
    plot_feature_importance,
    plot_pr_curves,
    plot_roc_curves,
)

__all__ = [
    "ReportGenerator",
    "ValidationMetrics",
    "bootstrap_metric",
    "compute_variant_phenotype_association",
    "plot_calibration",
    "plot_feature_importance",
    "plot_pr_curves",
    "plot_roc_curves",
]
