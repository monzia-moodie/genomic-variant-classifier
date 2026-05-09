"""
src/evaluation
==============
Clinical evaluation package for the Genomic Variant Classifier.
"""

from __future__ import annotations

from genomic_variant_classifier.evaluation.evaluator import (
    ClinicalEvaluator,
    ConsequenceBreakdown,
    EvaluationReport,
    GeneErrorAnalysis,
    OperatingPoint,
    compare_models,
)

__all__ = [
    "ClinicalEvaluator",
    "ConsequenceBreakdown",
    "EvaluationReport",
    "GeneErrorAnalysis",
    "OperatingPoint",
    "compare_models",
]
from genomic_variant_classifier.evaluation.prediction_artifacts import RunArtifactWriter
