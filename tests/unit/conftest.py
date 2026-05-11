"""Tests-wide pytest conftest for tests/unit/.

Path-aware SpliceAI isolation: blocks BaseConnector cache I/O ONLY when the
cache target resolves under the production data/raw/cache/ directory.
tmp_path-scoped FetchConfigs (the normal test pattern) are unaffected and
exercise the real cache load/save flow.

History:
- 2026-04-XX: class-scoped fixture in TestAnnotationPipeline (test_core.py)
  patched DEFAULT_SPLICEAI_PATH and BaseConnector._load_cache, but NOT
  _save_cache. fetch()'s unconditional _save_cache(cache_key, result) at
  database_connectors.py L217 wrote the stub-zero result to
  data/raw/cache/spliceai_scores_snv.parquet (430 MB). The fixture also
  did not cover TestSpliceAIConnector (a sibling class), leaving 73 of
  the SpliceAI tests with no isolation.
- 2026-05-10 first attempt: module-scoped autouse with UNCONDITIONAL
  _load_cache and _save_cache no-ops. Fixed the leak but broke
  TestSpliceAIConnector::test_parquet_cache_used_on_second_call, which
  uses a tmp_path cache_dir and legitimately exercises cache write->read.
- 2026-05-10 final: path-aware patches. Block load/save only when
  cache_path resolves under data/raw/cache/.

See memory #5 / #9 and INCIDENT_2026-04-29_gcp-billing-deletion.md.
"""

from __future__ import annotations

from pathlib import Path

import pytest


_PROD_CACHE_DIR = Path("data/raw/cache")


def _is_prod_cache_path(cache_path: Path) -> bool:
    """True iff cache_path resolves inside the production data/raw/cache/."""
    try:
        cache_path.resolve().relative_to(_PROD_CACHE_DIR.resolve())
        return True
    except ValueError:
        return False


@pytest.fixture(autouse=True)
def _isolate_spliceai(monkeypatch, tmp_path):
    """Path-aware SpliceAI isolation for every test in tests/unit/.

    Three patches:
      - DEFAULT_SPLICEAI_PATH -> a nonexistent path under tmp_path. Tests
        constructing SpliceAIConnector() with no vcf_path and no explicit
        DEFAULT_SPLICEAI_PATH override see no real index and take the
        stub-zero code path.
      - BaseConnector._load_cache -> return None ONLY when cache_path
        resolves under data/raw/cache/. tmp_path caches load normally.
      - BaseConnector._save_cache -> no-op ONLY when cache_path resolves
        under data/raw/cache/. tmp_path caches save normally. THIS is the
        actual leak fix; the prior class-scoped fixture only nulled
        _load_cache, leaving _save_cache to write zeros to the 430 MB
        prod cache file at the end of fetch().

    Tests that genuinely need to exercise prod-cache I/O can override these
    via in-test monkeypatch.
    """
    from genomic_variant_classifier.data import spliceai as _spliceai_mod
    from genomic_variant_classifier.data.database_connectors import BaseConnector

    monkeypatch.setattr(
        _spliceai_mod, "DEFAULT_SPLICEAI_PATH", tmp_path / "nonexistent.parquet"
    )

    _orig_load_cache = BaseConnector._load_cache
    _orig_save_cache = BaseConnector._save_cache

    def _safe_load_cache(self, key):
        if _is_prod_cache_path(self._cache_path(key)):
            return None
        return _orig_load_cache(self, key)

    def _safe_save_cache(self, key, df):
        if _is_prod_cache_path(self._cache_path(key)):
            return None
        return _orig_save_cache(self, key, df)

    monkeypatch.setattr(BaseConnector, "_load_cache", _safe_load_cache)
    monkeypatch.setattr(BaseConnector, "_save_cache", _safe_save_cache)
