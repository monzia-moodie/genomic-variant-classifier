"""Pickle migration driver for the consolidate-package-layout migration (C4).

Rewrites joblib pickles whose class paths reference the pre-migration
namespace (src.* and agent_layer.*) so they record the new namespace
(genomic_variant_classifier.*).

Per spec docs/hypotheses/HYP_consolidate-package-layout.md (HEAD 92b52a8):

- §F-6 deep pkgutil.walk_packages aliasing in install_compat_aliases.
  Top-level alias only is NOT sufficient: pickle.find_class does direct
  sys.modules[full.dotted.name] lookups for nested classes, so every
  submodule under genomic_variant_classifier.* must be aliased under both
  src.* and agent_layer.* prefixes.

- §F-7 = DELETE for experiments/2026-04-04_03-39/ensemble_v1.joblib.
  SHA256 lineage 64FEF6...CEE23 confirmed at HEAD 12b8315; recorded in
  the C4 commit message body proves recoverability from the migrated
  models/v1/ensemble_v1.joblib if ever needed.

- §F-8 predict_proba equivalence check on a frozen fixture. Run BEFORE
  deleting the .bak (keep .bak through C5 audit closure).

Standalone usage from repo root (post-C2, with genomic_variant_classifier
importable):

    python scripts/migrate_pickles.py

Process per TARGET file (idempotent retry pattern):

    .bak is the IMMUTABLE pre-migration snapshot; path is the MUTABLE target.

    First run:    path exists, .bak doesn't  → snapshot path→.bak, load .bak,
                                                 verify legacy refs present,
                                                 dump to path (with new namespace),
                                                 subprocess-verify, equivalence-check.
    Retry:        .bak exists                → load .bak, run STALE-.BAK GUARD
                                                 (verify legacy refs present),
                                                 dump to path, subprocess-verify,
                                                 equivalence-check.
    Already done: .bak still has legacy refs → re-load .bak, re-dump to path
                                                 (idempotent; identical result).
    Tainted:      .bak has NO legacy refs    → ABORT with stale-.bak error.
                                                 Manual investigation required.

Process for DELETIONS (§F-7):
1. Capture SHA256 (last verification of historical lineage).
2. Remove file. Return SHA256 for the C4 commit message body.

Recovery on failure:
    Fix the underlying issue and re-run the script. .bak files preserved as
    snapshots; idempotent retry pattern guarantees re-running produces correct
    results. The stale-.bak guard in migrate_one catches the rare case where
    .bak is from an unrelated workflow.

Public API (stable; C5 verification subprocess depends on this):
    install_compat_aliases() -> None
    verify_no_legacy_modules(obj) -> list[str]
    migrate_one(path: Path) -> None
    delete_one(path: Path) -> str   # returns pre-deletion SHA256
    equivalence_check(bak_path, new_path, fixture_X) -> None
    main() -> int
    TARGETS, DELETIONS, FIXTURE_PATH (module-level constants)
"""

from __future__ import annotations

import hashlib
import importlib
import pkgutil
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path anchoring: derive everything from the script location, NOT the cwd.
# This keeps the script robust whether invoked from repo root or elsewhere.
# ---------------------------------------------------------------------------
SCRIPT_DIR: Path = Path(__file__).resolve().parent           # <repo>/scripts/
REPO_ROOT: Path = SCRIPT_DIR.parent                          # <repo>/

# Targets requiring rewrite (5 production joblibs).
TARGETS: tuple[Path, ...] = (
    REPO_ROOT / "models" / "phase2_pipeline.joblib",
    REPO_ROOT / "models" / "phase4_pipeline.joblib",
    REPO_ROOT / "models" / "phase4_pipeline_calibrated.joblib",
    REPO_ROOT / "models" / "v1" / "ensemble_v1.joblib",
    REPO_ROOT / "outputs" / "run9_ready" / "models" / "ensemble.joblib",
)

# Deletions per §F-7. Byte-identical to migrated copies; SHA256 lineage
# proves recoverability from models/v1/ensemble_v1.joblib if needed.
DELETIONS: tuple[Path, ...] = (
    REPO_ROOT / "experiments" / "2026-04-04_03-39" / "ensemble_v1.joblib",
)

# Frozen fixture for §F-8 equivalence check. Required for migrate_one to
# fully validate; missing fixture raises in _check_environment.
FIXTURE_PATH: Path = REPO_ROOT / "tests" / "fixtures" / "migration_smoke.parquet"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def install_compat_aliases() -> None:
    """Make every old src.* and agent_layer.* module name resolve to
    genomic_variant_classifier.*.

    Required before joblib.load() of any pre-migration pickle. Must be deep
    enough to cover every nested class's __module__: pickle.find_class does
    direct sys.modules[full.dotted.name] lookups, so a top-level alias on
    'src' alone does NOT make sys.modules['genomic_variant_classifier.api.pipeline'] resolvable.

    Failures during walk are collected and reported but do not abort —
    joblib.load will surface real issues with a precise ModuleNotFoundError
    if any required submodule wasn't aliased.
    """
    import genomic_variant_classifier as _new_root
    import genomic_variant_classifier.agent_layer  # bind subpackage attr on _new_root before .agent_layer access below

    # Top-level aliases (necessary, not sufficient on their own).
    sys.modules.setdefault("src", _new_root)
    sys.modules.setdefault("agent_layer", _new_root.agent_layer)

    walk_failures: list[tuple[str, str]] = []

    def _onerror(qualname: str) -> None:
        # walk_packages calls this with the package name on import failure.
        # Re-import to capture the exception details for the diagnostic.
        try:
            importlib.import_module(qualname)
        except Exception as exc:  # noqa: BLE001
            walk_failures.append(
                (qualname, f"WALK-ERROR  {type(exc).__name__}: {exc}")
            )
        else:  # pragma: no cover — onerror only called on failure
            walk_failures.append((qualname, "WALK-ERROR  (no exception on retry)"))

    for _finder, qualname, _ispkg in pkgutil.walk_packages(
        _new_root.__path__,
        prefix="genomic_variant_classifier.",
        onerror=_onerror,
    ):
        try:
            mod = importlib.import_module(qualname)
        except Exception as exc:  # noqa: BLE001
            walk_failures.append((qualname, f"{type(exc).__name__}: {exc}"))
            continue
        suffix = qualname[len("genomic_variant_classifier."):]
        sys.modules[f"src.{suffix}"] = mod
        if suffix.startswith("agent_layer."):
            rest = suffix[len("agent_layer."):]
            sys.modules[f"agent_layer.{rest}"] = mod

    if walk_failures:
        print(
            "\n[migrate_pickles] WARN: import failures during alias installation:",
            file=sys.stderr,
        )
        for qualname, msg in walk_failures:
            print(f"  {qualname}\n    {msg}", file=sys.stderr)
        print(
            "[migrate_pickles] continuing; joblib.load will surface real problems.",
            file=sys.stderr,
        )


def verify_no_legacy_modules(obj) -> list[str]:
    """Walk the object graph; return any class __module__ values still rooted
    under 'src.' or 'agent_layer.'. Empty list = clean.

    Handles sklearn Pipelines, ColumnTransformers, FeatureUnions, dicts,
    lists/tuples/sets, and generic __dict__-bearing instances. Detects
    cycles via id() to avoid runaway recursion.

    This function is imported by the C5 verification subprocess — it must
    work without compat aliases installed (it only inspects __module__
    strings; it does not need to construct or call any class).
    """
    seen: set[int] = set()
    bad: list[str] = []

    def _walk(x: object) -> None:
        if id(x) in seen:
            return
        seen.add(id(x))
        cls_mod = getattr(type(x), "__module__", "") or ""
        first = cls_mod.split(".", 1)[0]
        if first in {"src", "agent_layer"}:
            bad.append(f"{cls_mod}.{type(x).__name__}")

        # Recurse into common containers.
        if isinstance(x, dict):
            for k, v in x.items():
                _walk(k)
                _walk(v)
            return
        if isinstance(x, (list, tuple, set, frozenset)):
            for v in x:
                _walk(v)
            return

        # Recurse into common sklearn estimator sub-attributes.
        for attr_name in (
            "steps", "estimators_", "named_steps",
            "base_estimator", "estimator", "model", "models",
            "pipeline",
            "transformers", "transformers_", "transformer_list",
            "named_transformers_", "final_estimator_",
        ):
            sub = getattr(x, attr_name, None)
            if sub is not None and not callable(sub):
                _walk(sub)

        # Generic __dict__ traversal (catches transformers_, custom attrs, etc.).
        d = getattr(x, "__dict__", None)
        if isinstance(d, dict):
            for v in d.values():
                _walk(v)

    _walk(obj)
    return sorted(set(bad))


def migrate_one(path: Path) -> None:
    """Idempotent migration: .bak is the immutable pre-migration snapshot;
    path is the mutable target. Safe to re-run after partial failures.

    First run:    path exists, .bak doesn't  → snapshot path→.bak, load .bak,
                                                 dump to path.
    Retry:        path exists, .bak exists   → load .bak, dump to path.
    Already done: path migrated, .bak still
                  has legacy refs            → re-load .bak, re-dump to path
                                                 (idempotent; identical result).

    Stale-.bak guard: if .bak exists but contains NO legacy module refs, the
    file is not a legitimate pre-migration original — abort with a clear
    error rather than silently use it as truth.

    Raises on any verification failure. .bak is preserved for recovery;
    after fixing the underlying issue, just re-run the script.
    """
    import joblib

    if not path.exists():
        raise FileNotFoundError(f"Target missing: {path}")

    bak = path.parent / (path.name + ".bak")

    print(f"\n=== {path.relative_to(REPO_ROOT)} ===")
    install_compat_aliases()  # idempotent; harmless even if path is already migrated

    if bak.exists():
        # Retry / re-run path — .bak should be a pre-migration original.
        print(f"  reusing existing .bak: {bak.name}")
        print(f"  bak-SHA256: {_sha256(bak)}")
        obj = joblib.load(bak)

        # STALE-.BAK GUARD: a legitimate .bak has legacy module refs.
        # If verify returns [], either .bak was already migrated (impossible
        # if the script created it) or .bak came from an unrelated workflow.
        # Either way: abort rather than corrupt the migration.
        bak_legacy = verify_no_legacy_modules(obj)
        if not bak_legacy:
            raise RuntimeError(
                f"Stale-.bak guard tripped: {bak} contains no legacy module refs.\n"
                f"  This .bak is not a legitimate pre-migration original.\n"
                f"  Manual investigation required. Likely scenarios:\n"
                f"    1. .bak is from an unrelated prior workflow (delete it,\n"
                f"       then re-run if path still has legacy refs).\n"
                f"    2. Migration is already complete and .bak/path were\n"
                f"       both replaced with migrated copies (compare SHA256\n"
                f"       of .bak and path; if equal, remove .bak — migration\n"
                f"       is done).\n"
                f"    3. Filesystem-level corruption (rare; check with\n"
                f"       Get-FileHash and reconcile from version control)."
            )
        print(f"  legacy refs in .bak: {len(bak_legacy)} (expected > 0; OK)")
    else:
        # First run — snapshot path then load from the snapshot.
        print(f"  pre-SHA256 (path): {_sha256(path)}")
        print(f"  snapshot -> {bak.name}")
        shutil.copy2(path, bak)
        obj = joblib.load(bak)

        # Sanity: warn if path was already migrated. Non-fatal — the re-dump
        # will produce a structurally-identical file. But surface the unexpected
        # state.
        path_legacy = verify_no_legacy_modules(obj)
        if not path_legacy:
            print(
                f"  NOTE: {path.relative_to(REPO_ROOT)} appears already "
                f"migrated (no legacy refs).",
                file=sys.stderr,
            )
            print(
                "  Continuing for idempotence; re-dump will be structurally "
                "identical.",
                file=sys.stderr,
            )

    # Re-dump to path. With option (b) semantics, this is the only mutation
    # of `path`. .bak is never written to after the first-run snapshot.
    compress = _detect_compress(bak)
    print(f"  re-dump (compress={compress})...")
    joblib.dump(obj, path, compress=compress)
    print(f"  post-SHA256: {_sha256(path)}")

    print("  subprocess verify (no aliases)...")
    _subprocess_verify(path)

    print("  equivalence check (predict_proba on fixture)...")
    fixture_X = _load_fixture()
    equivalence_check(bak, path, fixture_X)


def equivalence_check(bak_path: Path, new_path: Path, fixture_X) -> None:
    """Assert pre/post predict_proba (or predict) match within rtol=1e-7."""
    import joblib
    import numpy as np

    install_compat_aliases()
    old = joblib.load(bak_path)
    new = joblib.load(new_path)

    if hasattr(old, "predict_proba") and hasattr(new, "predict_proba"):
        np.testing.assert_allclose(
            old.predict_proba(fixture_X),
            new.predict_proba(fixture_X),
            rtol=1e-7,
            atol=0,
        )
    elif hasattr(old, "predict") and hasattr(new, "predict"):
        np.testing.assert_array_equal(
            old.predict(fixture_X),
            new.predict(fixture_X),
        )
    else:
        raise RuntimeError(
            f"No matching predict / predict_proba on {type(old).__name__} "
            f"vs {type(new).__name__}"
        )
    print("    equivalence OK")


def delete_one(path: Path) -> str:
    """Capture SHA256 of file, then delete. Returns SHA256 for audit log."""
    if not path.exists():
        raise FileNotFoundError(f"Deletion target missing: {path}")
    sha = _sha256(path)
    print(f"\n=== DELETE {path.relative_to(REPO_ROOT)} ===")
    print(f"  pre-deletion SHA256: {sha}")
    path.unlink()
    print("  deleted.")
    return sha


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    """SHA256 hash of a file, uppercase hex digest."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def _detect_compress(path: Path) -> int:
    """Detect compression of an existing joblib pickle.

    Returns 0 (uncompressed) if the first 2 bytes match a pickle PROTO opcode
    sequence (\\x80 followed by protocol number 2-5). Otherwise returns 3
    (joblib's default compression level), accepting that the on-disk file
    size may shift relative to the original — load equivalence is what the
    equivalence_check verifies, not byte-identity of the container.
    """
    try:
        with path.open("rb") as f:
            head = f.read(2)
    except OSError:
        return 3
    if len(head) >= 2 and head[0] == 0x80 and head[1] in (0x02, 0x03, 0x04, 0x05):
        return 0
    return 3


def _load_fixture():
    """Load the predict-equivalence fixture as a feature DataFrame.

    Drops common label columns (best-effort; the fixture should be feature-only,
    but this guards against accidental inclusion).
    """
    import pandas as pd

    df = pd.read_parquet(FIXTURE_PATH)
    for label_col in ("label", "y", "target", "clinvar_significance"):
        if label_col in df.columns:
            df = df.drop(columns=[label_col])
    return df


def _subprocess_verify(target: Path) -> None:
    """Spawn a fresh Python with NO compat aliases; load target; assert clean.

    Imports verify_no_legacy_modules from this module file. The subprocess
    must NOT have aliases installed — that would defeat the purpose of the
    verification (which proves the rewrite is genuine, not a runtime artifact
    of the alias map).
    """
    code = (
        "import joblib, sys, json;"
        f"sys.path.insert(0, {repr(str(REPO_ROOT))});"
        "from scripts.migrate_pickles import verify_no_legacy_modules;"
        f"obj = joblib.load({repr(str(target))});"
        "leftovers = verify_no_legacy_modules(obj);"
        "sys.exit(0 if not leftovers else (print(json.dumps(leftovers)) or 1))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Subprocess verify failed for {target}:\n"
            f"  stdout: {result.stdout}\n"
            f"  stderr: {result.stderr}"
        )


def _check_environment() -> None:
    """Pre-flight: confirm the script can run in the current environment."""
    # 1. genomic_variant_classifier importable (post-C2 invariant).
    try:
        import genomic_variant_classifier  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "genomic_variant_classifier is not importable.\n"
            "C4 requires C2 (pip install -e .) to have completed first.\n"
            f"Original error: {exc}"
        ) from exc

    # 2. All TARGETS exist.
    missing = [t for t in TARGETS if not t.exists()]
    if missing:
        raise RuntimeError(
            "Missing target file(s):\n  "
            + "\n  ".join(str(p) for p in missing)
        )

    # 3. All DELETIONS exist (so we can capture SHA256 for the audit).
    missing_del = [t for t in DELETIONS if not t.exists()]
    if missing_del:
        raise RuntimeError(
            "Missing deletion target(s):\n  "
            + "\n  ".join(str(p) for p in missing_del)
        )

    # 4. Pre-existing .bak files are LEGITIMATE under the idempotent retry
    # pattern (they're either snapshots from a prior interrupted run or from
    # a successful run not yet cleaned up). Detection of *stale* .bak files
    # (those that contain no legacy module refs) is delegated to migrate_one's
    # stale-.bak guard, which inspects content rather than mere presence.
    existing_baks = [
        t.parent / (t.name + ".bak") for t in TARGETS
        if (t.parent / (t.name + ".bak")).exists()
    ]
    if existing_baks:
        print("  NOTE: pre-existing .bak files found (legitimate under retry):")
        for b in existing_baks:
            print(f"    {b.relative_to(REPO_ROOT)}")
        print("  migrate_one will validate each before use (stale-.bak guard).")

    # 5. Fixture exists (required for §F-8 equivalence check).
    if not FIXTURE_PATH.exists():
        raise RuntimeError(
            f"Equivalence-check fixture missing: {FIXTURE_PATH}\n"
            "Create a 5-10 row sample from the training data before C4. "
            "It must contain feature columns matching what the pipelines "
            "were trained on."
        )


def main() -> int:
    """Driver: pre-flight, migrate all TARGETS, delete all DELETIONS,
    emit SHA256 audit table for the C4 commit message body."""
    print("=" * 64)
    print("scripts/migrate_pickles.py — C4 pickle migration driver")
    print("=" * 64)

    print("\nPre-flight environment check...")
    _check_environment()
    print("  OK: environment ready.")

    deletion_audit: list[tuple[Path, str]] = []

    try:
        # 1. Migrate every target.
        for target in TARGETS:
            migrate_one(target)

        # 2. Delete every deletion target (only after all migrations succeed).
        for target in DELETIONS:
            sha = delete_one(target)
            deletion_audit.append((target, sha))
    except Exception as exc:
        print(
            f"\nMIGRATION FAILED: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        print("\nRECOVERY (idempotent retry pattern):", file=sys.stderr)
        print("  1. Fix the underlying error reported above.", file=sys.stderr)
        print(
            "  2. Re-run: python scripts/migrate_pickles.py",
            file=sys.stderr,
        )
        print(
            "     .bak files preserved; the script will reuse them as the\n"
            "     pre-migration source of truth on each TARGET.",
            file=sys.stderr,
        )
        print(
            "  3. If the error was 'Stale-.bak guard tripped' for any file,\n"
            "     follow the instructions in that error message — the .bak\n"
            "     for that path is not a legitimate pre-migration original.",
            file=sys.stderr,
        )
        return 1

    # 3. Emit SHA256 audit table for the C4 commit message body.
    print("\n" + "=" * 64)
    print("SHA256 lineage audit (paste into C4 commit message body)")
    print("=" * 64)
    for target in TARGETS:
        bak = target.parent / (target.name + ".bak")
        rel = target.relative_to(REPO_ROOT)
        if bak.exists():
            print(f"  {rel}")
            print(f"    pre  (.bak):  {_sha256(bak)}")
            print(f"    post:         {_sha256(target)}")
    for target, sha in deletion_audit:
        rel = target.relative_to(REPO_ROOT)
        print(f"  {rel}")
        print(f"    pre-deletion: {sha}")
        print(f"    post:         DELETED")

    print("\nC4 ready to commit.")
    print(".bak files retained — delete only at C5 close-out.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
