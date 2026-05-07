"""Validate frontmatter of docs/validated/ and docs/hypotheses/ markdown files."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml  # PyYAML 6.0.3 already in your venv via pre-commit

REQUIRED_FIELDS = {"id", "title", "status", "date"}
VALID_STATUSES = {"validated", "hypothesis", "draft"}


def validate(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        return [f"{path}: missing frontmatter (must start with '---')"]
    end = text.find("\n---\n", 4)
    if end < 0:
        return [f"{path}: frontmatter block not closed"]
    front = yaml.safe_load(text[4:end]) or {}
    errs = []
    missing = REQUIRED_FIELDS - set(front)
    if missing:
        errs.append(f"{path}: missing fields {sorted(missing)}")
    status = front.get("status")
    if status not in VALID_STATUSES:
        errs.append(
            f"{path}: status={status!r}, must be one of {sorted(VALID_STATUSES)}"
        )
    return errs


def main(argv: list[str]) -> int:
    errors = [e for arg in argv[1:] for e in validate(Path(arg))]
    for e in errors:
        print(e, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
