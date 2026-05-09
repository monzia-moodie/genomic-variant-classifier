# agent_layer/utils/performance_tracker.py
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import json
from typing import Iterable


@dataclass(frozen=True)
class GenerationRecord:
    run_id: str
    domain: str
    score: float
    timestamp: str
    metadata: dict[str, object] = field(default_factory=dict)


class PerformanceTracker:
    """Append-only cross-Run performance tracking. No self-modification of this file permitted (Tier 0)."""

    def __init__(self, tracking_file: Path) -> None:
        self.tracking_file = tracking_file
        self.tracking_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.tracking_file.exists():
            self.tracking_file.write_text("")

    def record(self, rec: GenerationRecord) -> None:
        line = json.dumps(rec.__dict__, sort_keys=True) + "\n"
        with self.tracking_file.open("a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
            # Note: Windows fsync via os.fsync(fh.fileno()) per windows-fsync-rwb rule.

    def history(self, domain: str | None = None) -> Iterable[GenerationRecord]:
        for raw in self.tracking_file.read_text(encoding="utf-8").splitlines():
            d = json.loads(raw)
            if domain is None or d.get("domain") == domain:
                yield GenerationRecord(**d)
