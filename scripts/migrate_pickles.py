# scripts/migrate_pickles.py
"""C4: Re-namespace pre-migration .joblib files to genomic_variant_classifier.*

Run-ID: <run-id-of-C4-commit>
"""

from __future__ import annotations

import importlib
import json
import pkgutil
import shutil
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np

import genomic_variant_classifier as _new_root


# ---- Compat aliasing (the §F-6 form — wrapped, NOT top-level) -------------
def install_compat_aliases() -> None:
    """Make every old src.* and agent_layer.* module name resolve to the new
    namespace. Required immediately before joblib.load() of any pre-migration
    pickle. Idempotent (uses setdefault for top-level, plain assignment for
    submodules)."""
    sys.modules.setdefault("src", _new_root)
    sys.modules.setdefault("agent_layer", _new_root.agent_layer)

    for _finder, qualname, _ispkg in pkgutil.walk_packages(
        _new_root.__path__, prefix="genomic_variant_classifier."
    ):
        try:
            mod = importlib.import_module(qualname)
        except Exception as exc:
            print(
                f"[migrate_pickles] WARNING: cannot import {qualname}: {exc}",
                file=sys.stderr,
            )
            continue
        suffix = qualname[len("genomic_variant_classifier.") :]
        sys.modules[f"src.{suffix}"] = mod
        if suffix.startswith("agent_layer."):
            rest = suffix[len("agent_layer.") :]
            sys.modules[f"agent_layer.{rest}"] = mod


def verify_no_legacy_modules(obj) -> list[str]:
    """Walk an object graph, return any class.__module__ values still rooted
    under 'src.' or 'agent_layer.'. Empty list = success. Does NOT need
    aliases installed — only inspects __module__ strings."""
    seen: set[int] = set()
    bad: list[str] = []

    def _walk(x: object) -> None:
        if id(x) in seen:
            return
        seen.add(id(x))
        cls_mod = getattr(type(x), "__module__", "") or ""
        if cls_mod.split(".", 1)[0] in {"src", "agent_layer"}:
            bad.append(f"{cls_mod}.{type(x).__name__}")
        if isinstance(x, dict):
            for k, v in x.items():
                _walk(k)
                _walk(v)
        elif isinstance(x, (list, tuple, set, frozenset)):
            for v in x:
                _walk(v)
        for attr in (
            "steps",
            "estimators_",
            "named_steps",
            "base_estimator",
            "estimator",
            "model",
            "models",
            "pipeline",
        ):
            sub = getattr(x, attr, None)
            if sub is not None and not callable(sub):
                _walk(sub)
        d = getattr(x, "__dict__", None)
        if isinstance(d, dict):
            for v in d.values():
                _walk(v)

    _walk(obj)
    return sorted(set(bad))


# ---- Migration driver -----------------------------------------------------
TARGETS = [
    Path("models/phase2_pipeline.joblib"),
    Path("models/phase4_pipeline.joblib"),
    Path("models/phase4_pipeline_calibrated.joblib"),
    Path("models/v1/ensemble_v1.joblib"),
    Path("outputs/run9_ready/models/ensemble.joblib"),
    # experiments/2026-04-04_03-39/ensemble_v1.joblib — disposition per §F-7
]


def migrate_one(path: Path) -> None:
    bak = path.with_suffix(path.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(path, bak)

    install_compat_aliases()  # <-- called HERE, just before load
    obj = joblib.load(bak)
    joblib.dump(obj, path, compress=3)  # match original compress level

    # Verify in a clean subprocess (no aliases) — see §F-6 note
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import joblib, sys, json;"
            f"obj = joblib.load(r'{path}');"
            "from scripts.migrate_pickles import verify_no_legacy_modules;"
            "leftovers = verify_no_legacy_modules(obj);"
            "sys.exit(0 if not leftovers else (print(json.dumps(leftovers)) or 1))",
        ],
        check=True,
    )


def main() -> int:
    for p in TARGETS:
        migrate_one(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
