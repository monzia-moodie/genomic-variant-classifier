"""
scripts/c3_sweep.py - C3 namespace migration sweep.

Rewrites import statements and string literals in .py files (plus one
GitHub workflow file) to migrate from pre-migration namespaces ('src',
'agent_layer') to the new package ('genomic_variant_classifier').

Modes:
    (default)  dry-run: prints diff per file, no writes
    --apply             writes changes in place; .bak backup per file

Per-file safety:
- Backup to <file>.bak before writing
- For .py files: ast.parse the transformed content; if parse fails,
  restore from .bak and continue with next file
- Walks the project tree (pruning .venv, .venv312, __pycache__, .git,
  node_modules, .pytest_cache); operates on every .py file regardless
  of whether it appeared in the inventory
"""
from __future__ import annotations

import argparse
import ast
import re
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXCLUDE_PARTS = {".venv", ".venv312", "__pycache__", ".git", "node_modules", ".pytest_cache"}

SRC_SUBS = "api|data|evaluation|features|models|monitoring|pipelines|reports|training|utils"
AL_SUBS = "agents|config|message_bus|orchestrator|run_agents|shared_state|test_message_bus"

PATTERNS = [
    # 1. from src.<sub>...
    (re.compile(rf'\bfrom\s+src\.({SRC_SUBS})\b'),
     r'from genomic_variant_classifier.\1'),
    # 2. import src.<sub>...
    (re.compile(rf'\bimport\s+src\.({SRC_SUBS})\b'),
     r'import genomic_variant_classifier.\1'),
    # 3. quoted "src.<sub>..."
    (re.compile(rf'(["\'])src\.({SRC_SUBS})\b'),
     r'\1genomic_variant_classifier.\2'),
    # 4. from agent_layer.<sub>...
    (re.compile(rf'\bfrom\s+agent_layer\.({AL_SUBS})\b'),
     r'from genomic_variant_classifier.agent_layer.\1'),
    # 5. import agent_layer.<sub>...
    (re.compile(rf'\bimport\s+agent_layer\.({AL_SUBS})\b'),
     r'import genomic_variant_classifier.agent_layer.\1'),
    # 6. quoted "agent_layer.<sub>..."
    (re.compile(rf'(["\'])agent_layer\.({AL_SUBS})\b'),
     r'\1genomic_variant_classifier.agent_layer.\2'),
]


def walk_py(d: Path):
    if d.name in EXCLUDE_PARTS:
        return
    try:
        entries = list(d.iterdir())
    except (PermissionError, OSError):
        return
    for entry in entries:
        if entry.is_dir():
            yield from walk_py(entry)
        elif entry.suffix == ".py":
            yield entry


def transform(text: str) -> tuple[str, int]:
    """Apply all patterns. Return (new_text, n_substitutions)."""
    total = 0
    for regex, replacement in PATTERNS:
        text, n = regex.subn(replacement, text)
        total += n
    return text, total


def changed_lines_diff(old: str, new: str, max_lines: int = 8) -> str:
    old_lines = old.splitlines()
    new_lines = new.splitlines()
    out = []
    shown = 0
    for i, (o, n) in enumerate(zip(old_lines, new_lines), start=1):
        if o != n:
            if shown < max_lines:
                out.append(f"  L{i:5d}  - {o.rstrip()}")
                out.append(f"         + {n.rstrip()}")
                shown += 1
    if shown == max_lines and sum(1 for o, n in zip(old_lines, new_lines) if o != n) > max_lines:
        out.append(f"  ... ({sum(1 for o, n in zip(old_lines, new_lines) if o != n) - max_lines} more changed line(s))")
    return "\n".join(out)


def process_file(path: Path, apply: bool, is_python: bool) -> tuple[bool, int]:
    try:
        original = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, PermissionError) as e:
        print(f"SKIP {path.relative_to(PROJECT_ROOT).as_posix()}: {e}", file=sys.stderr)
        return False, 0

    new_text, n_subs = transform(original)
    if n_subs == 0:
        return False, 0

    rel = path.relative_to(PROJECT_ROOT).as_posix()
    print(f"{'CHANGE' if apply else 'WOULD CHANGE'} {rel}  ({n_subs} substitution(s))")
    print(changed_lines_diff(original, new_text))
    print()

    if not apply:
        return True, n_subs

    # Verify Python files still parse
    if is_python:
        try:
            ast.parse(new_text, filename=str(path))
        except SyntaxError as e:
            print(f"  ABORT {rel}: post-transform parse failed: {e}", file=sys.stderr)
            return False, 0

    bak = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, bak)
    path.write_text(new_text, encoding="utf-8")
    return True, n_subs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="Write changes in place. Default is dry-run.")
    args = ap.parse_args()

    print(f"Mode: {'APPLY (writes!)' if args.apply else 'DRY-RUN (no writes)'}")
    print(f"Project root: {PROJECT_ROOT}")
    print("=" * 70)
    print()

    files_changed = 0
    total_subs = 0

    # Python files
    for py in walk_py(PROJECT_ROOT):
        changed, n = process_file(py, apply=args.apply, is_python=True)
        if changed:
            files_changed += 1
            total_subs += n

    # GitHub workflow YAML
    yml = PROJECT_ROOT / ".github" / "workflows" / "drift_monitor.yml"
    if yml.exists():
        changed, n = process_file(yml, apply=args.apply, is_python=False)
        if changed:
            files_changed += 1
            total_subs += n

    print("=" * 70)
    verb = "changed" if args.apply else "would change"
    print(f"Summary: {files_changed} file(s) {verb}, {total_subs} substitution(s) total")
    if not args.apply:
        print()
        print("Re-run with --apply to write changes.")


if __name__ == "__main__":
    main()
