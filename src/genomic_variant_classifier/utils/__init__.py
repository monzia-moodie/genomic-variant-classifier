"""
src/utils
=========
Shared utility package for the Genomic Variant Classifier.

CHANGES FROM PHASE 1:
  - Utility functions were previously defined directly in this __init__.py.
    They have been moved to src/utils/helpers.py and re-exported here so
    that existing callers (e.g., `from src.utils import make_variant_id`)
    continue to work without modification (Issue C fixed).
"""

from __future__ import annotations

from src.utils.helpers import (
    add_missing_columns,
    ensure_dir,
    file_md5,
    log_dataframe_summary,
    log_step,
    locus_key,
    make_variant_id,
    parse_variant_id,
    proportion_ci,
    retry,
    safe_float,
    safe_log10,
)

__all__ = [
    "add_missing_columns",
    "ensure_dir",
    "file_md5",
    "log_dataframe_summary",
    "log_step",
    "locus_key",
    "make_variant_id",
    "parse_variant_id",
    "proportion_ci",
    "retry",
    "safe_float",
    "safe_log10",
]
