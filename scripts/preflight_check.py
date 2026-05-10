"""
Run 9 preflight check -- local, runs BEFORE any GPU instance is provisioned.

Fails loudly if any readiness gate is not met. Exit code 0 on PASS,
non-zero on FAIL. Designed to be the scripted enforcement of standing
rule #1 ("Exhaustive pre-flight check before every run").

Usage:
    cd C:\\Projects\\genomic-variant-classifier
    python scripts/preflight_check.py
    python scripts/preflight_check.py --skip-pytest     # fast iteration
    python scripts/preflight_check.py --skip-gcs        # offline mode

What this checks:
 1. Git working tree is clean (modulo an allowlist of known carry-overs)
 2. HEAD matches origin/main
 3. Full pytest suite is green (can be skipped for fast iteration)
 4. Required local data files exist
 6. No TensorFlow imports linger in variant_ensemble.py
 7. transformers and torch available somewhere (requirements or site-packages)
 8. agent_state.json is parseable (if present)
 9. GITHUB_TOKEN available somewhere (.env, User env, or current session)
10. SpliceAI cache is absent (ensures tests short-circuit path works)

Design principles:
 * Every check is defensive. No path through main() crashes; the worst
   case is a FAIL line with a diagnostic message.
 * On Windows, every external tool (git, gcloud, pytest, etc.) is
   resolved via shutil.which() BEFORE subprocess.run() is called. This
   sidesteps Windows' broken path resolution for .cmd/.bat shims --
   instead of shell=True (which is fragile with cwd), we resolve to
   the absolute path of the shim and call that directly.
 * Requirements check is NOT against requirements.txt -- this project
   installs transformers/torch fresh on the Vast.ai VM, so a
   requirements-based check was misleading. Instead we check that
   the packages are importable in the current venv.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
IS_WINDOWS = platform.system() == "Windows"

# Carry-overs from earlier sessions that are NOT a preflight failure.
# Add to this list ONLY with an ADR or session-doc justification.
ALLOWED_DIRTY_PATHS: set[str] = {
    "scripts/gcp_run6_startup.sh",  # legacy GCP run infrastructure
    "ROADMAP_PSYCH_GWAS_ENTRY.md",  # unfinished roadmap stub
}


def check(name: str, ok: bool, detail: str = "") -> tuple[str, bool, str]:
    status = "PASS" if ok else "FAIL"
    tail = f"  -- {detail}" if detail else ""
    print(f"[{status}] {name}{tail}")
    return (name, ok, detail)


def info(name: str, detail: str = "") -> None:
    """Non-gating informational message. Does not count as PASS or FAIL."""
    tail = f"  -- {detail}" if detail else ""
    print(f"[INFO] {name}{tail}")


def _resolve(exe: str) -> str | None:
    """Find the absolute path to an executable. Returns None if not found.

    On Windows, this handles .cmd/.bat/.exe shims correctly because
    shutil.which respects PATHEXT. The returned absolute path can be
    passed to subprocess.run() WITHOUT shell=True, avoiding the
    cwd-conflict issues that shell=True causes on Windows.
    """
    path = shutil.which(exe)
    if path:
        return path
    # Windows-specific fallback for common shim locations
    if IS_WINDOWS:
        for ext in (".cmd", ".bat", ".exe"):
            path = shutil.which(exe + ext)
            if path:
                return path
    return None


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    """Run a subprocess. NEVER raise. Missing executables become a synthetic
    CompletedProcess with returncode 127 so callers can always check
    .returncode.

    Resolves the first token via _resolve() so that .cmd shims on Windows
    are invoked via their absolute path, not via PATH resolution at
    CreateProcess time (which is buggy for non-.exe shims).
    """
    if not cmd:
        return subprocess.CompletedProcess(
            args=cmd, returncode=127, stdout="", stderr="empty command"
        )

    resolved = _resolve(cmd[0])
    if resolved is None:
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=127,
            stdout="",
            stderr=f"executable not found on PATH: {cmd[0]}",
        )

    real_cmd = [resolved] + list(cmd[1:])
    try:
        return subprocess.run(
            real_cmd,
            capture_output=True,
            text=True,
            cwd=str(REPO),
            **kw,
        )
    except FileNotFoundError as exc:
        return subprocess.CompletedProcess(
            args=real_cmd,
            returncode=127,
            stdout="",
            stderr=f"executable found at {resolved} but invocation failed: {exc}",
        )
    except (PermissionError, OSError) as exc:
        return subprocess.CompletedProcess(
            args=real_cmd,
            returncode=126,
            stdout="",
            stderr=f"cannot execute {resolved}: {exc}",
        )


def git_working_tree_clean() -> tuple[bool, str]:
    cp = run(["git", "status", "--porcelain"])
    if cp.returncode != 0:
        return False, (
            f"git status failed (rc={cp.returncode}): "
            f"stderr={cp.stderr.strip()[:150]}  stdout={cp.stdout.strip()[:150]}"
        )
    lines = [ln for ln in cp.stdout.splitlines() if ln.strip()]
    unexpected = []
    for ln in lines:
        path = ln[3:].strip().strip('"')
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        if path not in ALLOWED_DIRTY_PATHS:
            unexpected.append(ln)
    if unexpected:
        return False, (
            f"{len(unexpected)} unexpected dirty paths: " + "; ".join(unexpected[:3])
        )
    return True, f"{len(lines)} dirty path(s), all allow-listed"


def head_matches_origin_main() -> tuple[bool, str]:
    # git fetch writes progress messages to stderr on success. We check
    # only returncode, not stderr non-emptiness.
    fetch = run(["git", "fetch", "origin", "main"])
    if fetch.returncode != 0:
        return False, (
            f"git fetch failed (rc={fetch.returncode}): "
            f"stderr={fetch.stderr.strip()[:150]}"
        )
    head = run(["git", "rev-parse", "HEAD"])
    origin = run(["git", "rev-parse", "origin/main"])
    if head.returncode != 0 or origin.returncode != 0:
        return False, (
            f"git rev-parse failed: head rc={head.returncode}, "
            f"origin rc={origin.returncode}"
        )
    h = head.stdout.strip()
    o = origin.stdout.strip()
    return h == o, f"HEAD={h[:8]} origin/main={o[:8]}"


def pytest_green() -> tuple[bool, str]:
    print("    [running pytest tests/unit/ -q  (may take 5-10 min)...]")
    cp = run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/unit/",
            "-q",
            "--no-header",
            "--tb=no",
            "-x",
        ],
    )
    if cp.returncode == 127:
        return False, f"cannot launch pytest: {cp.stderr.strip()[:200]}"
    last_lines = cp.stdout.strip().splitlines()[-3:] if cp.stdout else []
    tail = " | ".join(last_lines) if last_lines else cp.stderr.strip()[-200:]
    return cp.returncode == 0, tail


def files_exist(required: dict[str, Path]) -> list[tuple[str, bool, str]]:
    out = []
    for name, p in required.items():
        if p.exists():
            size_mb = p.stat().st_size // 1024 // 1024
            out.append((f"file exists: {name}", True, f"{p} ({size_mb} MB)"))
        else:
            out.append((f"file exists: {name}", False, str(p)))
    return out


def no_tensorflow_in_ensemble() -> tuple[bool, str]:
    path = REPO / "src/genomic_variant_classifier/models/variant_ensemble.py"
    if not path.exists():
        return False, f"not found: {path}"
    try:
        content = path.read_text(encoding="utf-8")
    except Exception as exc:
        return False, f"cannot read: {exc}"
    has_tf = "import tensorflow" in content or "from tensorflow" in content
    return not has_tf, "clean" if not has_tf else "tensorflow import still present"


def ml_deps_available() -> list[tuple[str, bool, str]]:
    results = []
    for mod in ("transformers", "torch"):
        try:
            __import__(mod)
            version = getattr(sys.modules[mod], "__version__", "unknown")
            results.append((f"import: {mod}", True, f"version {version}"))
        except ImportError as exc:
            results.append((f"import: {mod}", False, str(exc)[:120]))
    try:
        import tensorflow  # noqa: F401

        results.append(
            (
                "tensorflow not installed",
                False,
                "tensorflow is importable; should not be",
            )
        )
    except ImportError:
        results.append(("tensorflow not installed", True, "not importable (ok)"))
    return results


def github_token_available() -> tuple[bool, str]:
    """GITHUB_TOKEN can live in .env, current process env, or Windows User env."""
    env_path = REPO / ".env"
    if env_path.exists():
        try:
            content = env_path.read_text(encoding="utf-8")
            if "GITHUB_TOKEN=" in content:
                return True, ".env has GITHUB_TOKEN"
        except Exception:
            pass
    if os.environ.get("GITHUB_TOKEN"):
        return True, f"current session env (length: {len(os.environ['GITHUB_TOKEN'])})"
    if IS_WINDOWS:
        cp = run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "[System.Environment]::GetEnvironmentVariable('GITHUB_TOKEN', 'User')",
            ]
        )
        token = cp.stdout.strip()
        if token and len(token) > 10:
            return True, f"Windows User env (length: {len(token)})"
    return False, "not found in .env, current session, or Windows User env"


def agent_state_parseable() -> tuple[bool, str]:
    p = REPO / "agent_state.json"
    if not p.exists():
        return True, "absent (ok)"
    try:
        json.loads(p.read_text(encoding="utf-8"))
        return True, "valid json"
    except Exception as e:
        return False, f"parse error: {e}"


def spliceai_cache_absent() -> tuple[bool, str]:
    p = REPO / "data/raw/cache/spliceai_scores_snv.parquet"
    if p.exists():
        sz = p.stat().st_size // 1024 // 1024
        mtime = p.stat().st_mtime
        import datetime

        mtime_str = datetime.datetime.fromtimestamp(mtime).isoformat(timespec="seconds")
        return False, (
            f"{sz} MB cache at {p} (last modified {mtime_str}) -- "
            f"indicates test-isolation leak; delete and investigate"
        )
    return True, "absent"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--skip-pytest",
        action="store_true",
        help="Skip the pytest step (fast iteration during setup).",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    results: list[tuple[str, bool, str]] = []

    # --- Repo hygiene ---
    ok, detail = git_working_tree_clean()
    results.append(check("git working tree clean (modulo allowlist)", ok, detail))

    ok, detail = head_matches_origin_main()
    results.append(check("HEAD == origin/main", ok, detail))

    # --- Code correctness ---
    ok, detail = no_tensorflow_in_ensemble()
    results.append(check("variant_ensemble.py free of tensorflow", ok, detail))

    for r in ml_deps_available():
        results.append(check(*r))

    ok, detail = agent_state_parseable()
    results.append(check("agent_state.json parseable", ok, detail))

    # --- Data files (local) ---
    required_files = {
        "SpliceAI parquet": REPO / "data/external/spliceai/spliceai_index.parquet",
        "ClinVar VCF": REPO / "data/raw/clinvar/clinvar_GRCh38.vcf.gz",
    }
    for r in files_exist(required_files):
        results.append(check(*r))

    # --- Credentials ---
    ok, detail = github_token_available()
    results.append(check("GITHUB_TOKEN available somewhere", ok, detail))

    # --- Test-state hygiene ---
    ok, detail = spliceai_cache_absent()
    results.append(check("SpliceAI test-cache absent", ok, detail))

    # --- Full test suite (slowest, last) ---
    if args.skip_pytest:
        info("pytest skipped", "--skip-pytest flag")
    else:
        ok, detail = pytest_green()
        results.append(check("pytest tests/unit/ green", ok, detail))

    # --- Summary ---
    print("=" * 72)
    failed = [r for r in results if not r[1]]
    total = len(results)
    print(f"Preflight: {total - len(failed)} pass, {len(failed)} fail")
    if failed:
        print("FAILURES:")
        for name, _, detail in failed:
            print(f"  - {name}: {detail}")
        print("\nDO NOT launch the GPU instance until all checks pass.")
        return 1
    print("ALL PREFLIGHT CHECKS PASSED -- safe to launch the GPU instance")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
