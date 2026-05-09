"""Smoke test: every expected module imports cleanly under the new namespace.

Runs from any cwd. Used as a post-C2 gate and a C3 verification step.

DESIGN: dual-source coverage — SENTINEL_IMPORTS (hardcoded entry points) +
pkgutil.walk_packages (filesystem breadth). The dual approach catches a
blind spot the walk-only approach has:

- pkgutil.walk_packages calls onerror(qualname) when a package's __init__.py
  raises during walk-time enumeration. Default onerror=None silently swallows
  the failure, so a broken module never appears in the EXPECTED_IMPORTS list.
  A broken-but-load-bearing module (e.g., genomic_variant_classifier.api.main
  fails after the C3 sweep) would produce a FALSE-NEGATIVE "SMOKE TEST OK"
  while the API entry point is dead.

Failure modes the dual approach catches that walk-only would miss:
1. A SENTINEL module silently broken at import time (sentinel sweep fails loudly)
2. A package's __init__.py raises during walk enumeration (onerror records,
   re-imported in main() to surface full exception details)
3. Submodule import failures (the walk loop catches these the same as before)

Reference: spec docs/hypotheses/HYP_consolidate-package-layout.md @ 92b52a8.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys

import genomic_variant_classifier as _root


# Load-bearing entry points whose import failure MUST fail the smoke test
# loudly. Add a line here whenever a new public surface area lands.
# Maintenance cost: ~1 line per major addition.
SENTINEL_IMPORTS: tuple[str, ...] = (
    "genomic_variant_classifier.api.main",          # FastAPI entry
    "genomic_variant_classifier.api.auth",          # auth module
    "genomic_variant_classifier.api.pipeline",      # joblib class location for phase2/phase4
    "genomic_variant_classifier.data.dbnsfp",       # data connector
    "genomic_variant_classifier.models.variant_ensemble",  # joblib class location for ensemble_v1
    "genomic_variant_classifier.utils.helpers",     # provides resolve_data_dir
    "genomic_variant_classifier.agent_layer.agents",
    "genomic_variant_classifier.agent_layer.config",
    "genomic_variant_classifier.agent_layer.message_bus",
    "genomic_variant_classifier.agent_layer.shared_state",
)


_walk_errors: list[str] = []


def _walk_onerror(qualname: str) -> None:
    """Record packages that fail during pkgutil.walk_packages enumeration.

    walk_packages calls this with the package name (not an exception object)
    when a package's __init__.py raises during the import-to-enumerate step.
    We collect names here, then re-import in main() to surface the actual
    exception details for the smoke-test failure report.
    """
    _walk_errors.append(qualname)


def collect_expected_imports() -> list[str]:
    return sorted(
        qualname
        for _finder, qualname, _ispkg in pkgutil.walk_packages(
            _root.__path__,
            prefix="genomic_variant_classifier.",
            onerror=_walk_onerror,
        )
    )


def main() -> int:
    failures: list[tuple[str, str]] = []

    # 1. Sentinel sweep — must succeed for every entry.
    for qualname in SENTINEL_IMPORTS:
        try:
            importlib.import_module(qualname)
        except Exception as exc:  # noqa: BLE001 — surface any failure
            failures.append((qualname, f"SENTINEL  {type(exc).__name__}: {exc}"))

    # 2. Filesystem walk — catches breadth.
    expected = collect_expected_imports()

    # 2a. Walk-error recovery: re-import packages that failed during walk
    # to surface the actual exception (walk_onerror only sees the name).
    for qualname in _walk_errors:
        try:
            importlib.import_module(qualname)
        except Exception as exc:  # noqa: BLE001
            failures.append((qualname, f"WALK-ERROR  {type(exc).__name__}: {exc}"))

    # 2b. Successfully-walked modules (skip sentinels — already covered).
    sentinel_set = set(SENTINEL_IMPORTS)
    for qualname in expected:
        if qualname in sentinel_set:
            continue
        try:
            importlib.import_module(qualname)
        except Exception as exc:  # noqa: BLE001
            failures.append((qualname, f"{type(exc).__name__}: {exc}"))

    if failures:
        print(f"SMOKE TEST FAILED: {len(failures)} import(s) broken:", file=sys.stderr)
        for qualname, msg in failures:
            print(f"  {qualname}\n    {msg}", file=sys.stderr)
        return 1

    n_sentinel = len(SENTINEL_IMPORTS)
    n_walked = len(expected)
    n_walk_err = len(_walk_errors)
    print(
        f"SMOKE TEST OK: {n_sentinel} sentinel + {n_walked} walked "
        f"({n_walk_err} walk-errors recovered) module(s) imported."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
