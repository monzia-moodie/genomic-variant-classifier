"""
Scalable Data Loader using Dask
"""
import dask.dataframe as dd
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class DaskVariantLoader:
    """Lazy-loading of variant data using Dask."""

    def __init__(self, filepath: str):
        self.filepath = filepath

    def load(self, sep="\t", blocksize="64MB") -> dd.DataFrame:
        """Load data into a Dask DataFrame."""
        logger.info(f"Initializing Dask reader for {self.filepath}")
        # Use string dtype by default to avoid inference errors on genomic columns
        return dd.read_csv(
            self.filepath,
            sep=sep,
            blocksize=blocksize,
            dtype=str,
            on_bad_lines='skip'
        )
