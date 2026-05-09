"""
scripts/c3_inventory.py - C3 scope inventory via AST.

Walks the project tree (pruning .venv312, __pycache__, .git, node_modules,
.pytest_cache at descent time), parses every .py file, and reports import
statements + string-literal module references that touch the pre-migration
namespaces ('src' and 'agent_layer').

Read-only: does not modify any project file.
Outputs: human-readable summary on stdout + full inventory JSON to
agent_data/c3_inventory.json (gitignored).
"""
from __future__ import annotations

import ast
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXCLUDE_PARTS = {".venv312", "__pycache__", ".git", "node_modules", ".pytest_cache"}
OLD_NAMESPACES = {"src", "agent_layer"}

# Known subpackages classify string-literal refs.
# "genomic_variant_classifier.api.main" -> match; "src.txt" / "agent_layer.json" -> reject.
SUBPACKAGES_BY_NS = {
    "src": {"api", "data", "evaluation", "features", "models",
            "monitoring", "pipelines", "reports", "training", "utils"},
    "agent_layer": {"agents", "config", "message_bus", "orchestrator",
                    "run_agents", "shared_state", "test_message_bus"},
}


def walk_py(d: Path):
    """Yield .py files under d, pruning EXCLUDE_PARTS at descent."""
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


def classify_string(s: str):
    """Return the namespace if s looks like a module path under one, else None."""
    for ns, subs in SUBPACKAGES_BY_NS.items():
        prefix = ns + "."
        if not s.startswith(prefix):
            continue
        rest = s[len(prefix):]
        first = rest.split(".")[0]
        if first not in subs:
            continue
        # rest must look like an identifier chain (alphanumeric + . + _)
        if not all(c.isalnum() or c in "._" for c in rest):
            continue
        return ns
    return None


def scan_file(py_path: Path) -> list[dict]:
    try:
        text = py_path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(py_path))
    except (SyntaxError, UnicodeDecodeError) as e:
        print(f"SKIP {py_path}: {type(e).__name__}: {e}", file=sys.stderr)
        return []

    lines = text.splitlines()
    refs = []

    def line_at(lineno):
        return lines[lineno - 1].strip() if 0 < lineno <= len(lines) else ""

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in OLD_NAMESPACES:
                    refs.append({
                        "lineno": node.lineno,
                        "type": "import",
                        "module": alias.name,
                        "alias": alias.asname,
                        "line": line_at(node.lineno),
                    })
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue  # relative import
            if node.module.split(".")[0] in OLD_NAMESPACES:
                refs.append({
                    "lineno": node.lineno,
                    "type": "from",
                    "module": node.module,
                    "names": [a.name + (f" as {a.asname}" if a.asname else "")
                              for a in node.names],
                    "level": node.level,
                    "line": line_at(node.lineno),
                })
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            ns = classify_string(node.value)
            if ns is not None:
                refs.append({
                    "lineno": node.lineno,
                    "type": "string",
                    "module": node.value,
                    "line": line_at(node.lineno),
                })

    return refs


def main():
    all_results = []
    file_count = 0
    for py_file in walk_py(PROJECT_ROOT):
        file_count += 1
        rel = py_file.relative_to(PROJECT_ROOT).as_posix()
        for ref in scan_file(py_file):
            ref["filepath"] = rel
            all_results.append(ref)

    print("=" * 70)
    print(f"C3 SCOPE INVENTORY")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print("=" * 70)
    print(f"Files scanned:               {file_count}")
    print(f"Total references to migrate: {len(all_results)}")

    by_file = defaultdict(list)
    for r in all_results:
        by_file[r["filepath"]].append(r)
    print(f"Files affected:              {len(by_file)}")
    print()

    # By reference form
    by_type = defaultdict(int)
    for r in all_results:
        by_type[r["type"]] += 1
    print("By reference form:")
    for t, c in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  {c:5d}  {t}")
    print()

    # By namespace.subpackage
    by_ns = defaultdict(int)
    for r in all_results:
        parts = r["module"].split(".")
        key = ".".join(parts[:2]) if len(parts) >= 2 else parts[0]
        by_ns[key] += 1
    print("By namespace.subpackage:")
    for ns, c in sorted(by_ns.items(), key=lambda x: -x[1]):
        print(f"  {c:5d}  {ns}")
    print()

    # By top-level dir of affected file
    by_dir = defaultdict(int)
    for filepath, refs in by_file.items():
        parts = filepath.split("/")
        top = parts[0] if len(parts) > 1 else "<root>"
        by_dir[top] += len(refs)
    print("By top-level directory:")
    for d, c in sorted(by_dir.items(), key=lambda x: -x[1]):
        suffix = "/" if d != "<root>" else ""
        print(f"  {c:5d}  {d}{suffix}")
    print()

    # Top 30 files
    print("Top 30 files by reference count:")
    for filepath, refs in sorted(by_file.items(), key=lambda x: -len(x[1]))[:30]:
        print(f"  {len(refs):4d}  {filepath}")
    print()

    out = PROJECT_ROOT / "agent_data" / "c3_inventory.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"Full inventory: {out.relative_to(PROJECT_ROOT).as_posix()}"
          f" ({len(all_results)} entries, {out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
