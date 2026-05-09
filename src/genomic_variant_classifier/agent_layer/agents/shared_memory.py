# agent_layer/utils/shared_memory.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import hmac
import hashlib
from typing import Literal

EntryKind = Literal["insight", "causal_hypothesis", "forward_plan", "drift_observation"]


@dataclass(frozen=True)
class MemoryEntry:
    key: str
    kind: EntryKind
    value: str
    run_id: str
    timestamp: str
    prev_hmac: str  # rolling HMAC chain


class TamperEvidentMemory:
    """Schema-validated, HMAC-chained, append-only. Tier 0."""

    def __init__(self, path: Path, secret_path: Path) -> None:
        self.path = path
        self._secret = secret_path.read_bytes()  # 32+ bytes, gitignored, escrowed
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("")

    def _compute_hmac(self, prev: str, payload: str) -> str:
        return hmac.new(
            self._secret, (prev + payload).encode("utf-8"), hashlib.sha256
        ).hexdigest()

    def append(self, key: str, kind: EntryKind, value: str, run_id: str) -> MemoryEntry:
        prev = self._last_hmac()
        ts = datetime.now(timezone.utc).isoformat()
        payload = json.dumps(
            {"key": key, "kind": kind, "value": value, "run_id": run_id, "ts": ts},
            sort_keys=True,
        )
        new_hmac = self._compute_hmac(prev, payload)
        entry = MemoryEntry(
            key=key,
            kind=kind,
            value=value,
            run_id=run_id,
            timestamp=ts,
            prev_hmac=new_hmac,
        )
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry.__dict__, sort_keys=True) + "\n")
        return entry

    def verify(self) -> bool:
        prev = ""
        for raw in self.path.read_text(encoding="utf-8").splitlines():
            d = json.loads(raw)
            payload = json.dumps(
                {
                    "key": d["key"],
                    "kind": d["kind"],
                    "value": d["value"],
                    "run_id": d["run_id"],
                    "ts": d["timestamp"],
                },
                sort_keys=True,
            )
            expected = self._compute_hmac(prev, payload)
            if expected != d["prev_hmac"]:
                return False
            prev = d["prev_hmac"]
        return True

    def _last_hmac(self) -> str:
        last = ""
        for raw in self.path.read_text(encoding="utf-8").splitlines():
            last = json.loads(raw)["prev_hmac"]
        return last
