"""
src/utils/helpers.py
=====================
Shared utility functions for the Genomic Variant Classifier pipeline.

CHANGES FROM PHASE 1:
  - In Phase 1, utility functions were written directly into src/utils/__init__.py.
    This conflates the package namespace with implementation, makes testing harder,
    and breaks the convention that __init__.py should only re-export.
    Functions have been moved here; __init__.py re-exports them (Issue C).
  - from __future__ import annotations added (Issue N).
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def load_config() -> dict:
    """Load configs/config.yaml relative to the repo root."""
    import yaml
    config_path = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Variant ID utilities
# ---------------------------------------------------------------------------
def make_variant_id(source: str, chrom: str, pos: int, ref: str, alt: str) -> str:
    """
    Canonical variant identifier: source:chrom:pos:ref:alt.

    This format is used across all database connectors and ensures that
    variants from different sources at the same locus can be matched.

    Examples:
        make_variant_id("clinvar", "17", 43071077, "G", "T")
        → "clinvar:17:43071077:G:T"
    """
    return f"{source}:{chrom}:{pos}:{ref}:{alt}"


def parse_variant_id(variant_id: str) -> dict[str, Any]:
    """
    Parse a canonical variant ID back into its components.

    Returns:
        dict with keys: source, chrom, pos, ref, alt.
        pos is cast to int. Returns empty dict if format is invalid.
    """
    parts = str(variant_id).split(":")
    if len(parts) != 5:
        logger.warning("Invalid variant ID format: %s", variant_id)
        return {}
    source, chrom, pos_str, ref, alt = parts
    try:
        return {"source": source, "chrom": chrom, "pos": int(pos_str), "ref": ref, "alt": alt}
    except ValueError:
        return {}


def locus_key(variant_id: str) -> str:
    """
    Strip the source prefix from a variant ID to get the genomic locus.
    Used for cross-database joins.

    Example:
        locus_key("clinvar:17:43071077:G:T") → "17:43071077:G:T"
    """
    parts = str(variant_id).split(":", 1)
    return parts[1] if len(parts) == 2 else variant_id


# ---------------------------------------------------------------------------
# DataFrame utilities
# ---------------------------------------------------------------------------
def add_missing_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    fill_value: Any = None,
) -> pd.DataFrame:
    """
    Add any missing columns to df with fill_value, preserving existing data.
    Used to enforce the canonical schema across connectors.
    """
    for col in required_columns:
        if col not in df.columns:
            df[col] = fill_value
    return df


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Convert a value to float, returning default on failure.
    Handles None, empty strings, and non-numeric strings gracefully.
    """
    try:
        if value is None or (isinstance(value, str) and value.strip() == ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def log_dataframe_summary(df: pd.DataFrame, label: str = "DataFrame") -> None:
    """
    Log shape, dtypes, null counts, and sample stats for a DataFrame.
    Useful at ETL checkpoints for debugging data quality issues.
    """
    logger.info("─── %s ───", label)
    logger.info("  Shape:    %d rows × %d columns", df.shape[0], df.shape[1])
    null_cols = df.columns[df.isnull().any()].tolist()
    if null_cols:
        for col in null_cols:
            n_null = df[col].isnull().sum()
            logger.info("  %-30s  %d nulls (%.1f%%)", col, n_null, n_null / len(df) * 100)
    else:
        logger.info("  No null values.")


# ---------------------------------------------------------------------------
# Path / file utilities
# ---------------------------------------------------------------------------
def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it does not exist. Returns the Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def file_md5(path: str | Path) -> str:
    """Compute MD5 hex digest of a file. Used for cache validation."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------
def log_step(label: Optional[str] = None):
    """
    Decorator that logs entry/exit and elapsed time for a pipeline step.

    Usage:
        @log_step("ClinVar ingestion")
        def ingest_clinvar(...):
            ...
    """
    def decorator(fn: Callable) -> Callable:
        step_label = label or fn.__name__

        @wraps(fn)
        def wrapper(*args, **kwargs):
            logger.info("── START: %s ──", step_label)
            t0 = time.time()
            result = fn(*args, **kwargs)
            elapsed = time.time() - t0
            logger.info("── DONE:  %s (%.1fs) ──", step_label, elapsed)
            return result

        return wrapper
    return decorator


def retry(max_attempts: int = 3, delay: float = 1.0, exceptions: tuple = (Exception,)):
    """
    Retry decorator for network-bound operations (e.g., API downloads).

    Args:
        max_attempts: Maximum number of attempts.
        delay:        Seconds to wait between attempts.
        exceptions:   Exception types that trigger a retry.
    """
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:
                    if attempt == max_attempts:
                        raise
                    logger.warning(
                        "%s failed (attempt %d/%d): %s — retrying in %.1fs",
                        fn.__name__, attempt, max_attempts, exc, delay,
                    )
                    time.sleep(delay)
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Numeric / array utilities
# ---------------------------------------------------------------------------
def safe_log10(x: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """log10 transform with epsilon floor to avoid -inf for zero allele frequencies."""
    return np.log10(np.clip(x, epsilon, None))


def proportion_ci(
    k: int, n: int, confidence: float = 0.95
) -> tuple[float, float]:
    """
    Wilson score interval for a proportion k/n.
    More accurate than the naive normal approximation for small k or n.
    """
    from scipy import stats as sp_stats
    if n == 0:
        return 0.0, 0.0
    z = sp_stats.norm.ppf((1 + confidence) / 2)
    p_hat = k / n
    denominator = 1 + z ** 2 / n
    centre = (p_hat + z ** 2 / (2 * n)) / denominator
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z ** 2 / (4 * n ** 2)) / denominator
    return float(max(0.0, centre - margin)), float(min(1.0, centre + margin))


# ---------------------------------------------------------------------------
# Environment-aware data directory resolution (Phase 2)
# ---------------------------------------------------------------------------
def resolve_data_dir(config: dict | None = None) -> Path:
    """
    Resolve the root data directory across VS Code, Colab, and CI environments.

    Priority order:
      1. GENOMIC_DATA_DIR environment variable  (set in .env for VS Code)
      2. Google Colab Drive mount               (auto-detected)
      3. Google Drive for Desktop on Windows    (auto-detected, common letters)
      4. Google Drive for Desktop on macOS      (auto-detected)
      5. config["data_dir"] if supplied         (explicit override)
      6. ./data fallback                        (CI / synthetic-only runs)

    Args:
        config: Optional dict loaded from config.yaml. If None, only env-var
                and auto-detection paths are tried before the ./data fallback.

    Returns:
        Path to the resolved data root directory.
    """
    # Load .env file so GENOMIC_DATA_DIR is available in this process
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
    except ImportError:
        pass  # python-dotenv not installed — fall through to auto-detection

    # 1. Explicit environment variable (VS Code .env or shell export)
    if env_val := os.environ.get("GENOMIC_DATA_DIR"):
        p = Path(env_val)
        if p.exists():
            return p
        logger.warning("GENOMIC_DATA_DIR set to %s but path does not exist.", p)

    # 2. Google Colab Drive mount
    colab = Path("/content/drive/MyDrive/genomic-variant-data")
    if colab.exists():
        return colab

    # 3. Google Drive for Desktop — Windows common drive letters
    for letter in "GHIJKL":
        win = Path(f"{letter}:/My Drive/genomic-variant-data")
        if win.exists():
            return win

    # 4. Google Drive for Desktop — macOS
    mac = Path.home() / "Google Drive" / "My Drive" / "genomic-variant-data"
    if mac.exists():
        return mac

    # 5. Explicit config value
    if config and config.get("data_dir"):
        p = Path(config["data_dir"]).expanduser().resolve()
        if p.exists():
            return p
        logger.warning("config data_dir %s does not exist; falling back to ./data", p)

    # 6. Local fallback — works for CI and synthetic-only dev runs
    fallback = Path("data")
    fallback.mkdir(parents=True, exist_ok=True)
    logger.info("resolve_data_dir: using local fallback at %s", fallback.resolve())
    return fallback