"""
src/evaluation
==============
Clinical evaluation package for the Genomic Variant Classifier.
"""

from __future__ import annotations

from src.evaluation.evaluator import (
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
