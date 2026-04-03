"""
shared_state.py
===============
Thread-safe, file-backed shared state store.

All agents read and write through this class.  The backing store is a single
JSON file so it works on local disk, a Google Drive mount, or any networked
filesystem without requiring a running database.

Swap the backend for Redis by subclassing SharedState and overriding
_load() / _save() — the agent code stays identical.
"""

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import SHARED_STATE_PATH

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

DEFAULT_STATE: dict[str, Any] = {
    # --- corpus & model provenance ---
    "variant_corpus_version":  None,   # hash of current training corpus
    "corpus_last_updated":     None,   # ISO timestamp
    "model_checkpoint_ref":    None,   # GDrive / GCS path to latest weights
    "model_last_trained":      None,

    # --- data freshness ---
    "clinvar_last_seen_date":  None,   # date of last ingested ClinVar release
    "gnomad_last_seen_version": None,
    "lovd_last_seen_timestamp": None,
    "alphamissense_last_seen": None,

    # --- drift ---
    "drift_score":             0.0,    # running JS divergence
    "drift_history":           [],     # list of {timestamp, score}
    "reclassification_rate":   0.0,    # fraction of variants reclassified

    # --- EWC ---
    "ewc_fisher_ref":          None,   # path to Fisher information matrix
    "ewc_lambda":              400.0,  # EWC regularisation strength

    # --- review queue ---
    "pending_review":          [],     # items awaiting human sign-off

    # --- feature candidates (from Literature Scout) ---
    "feature_candidates":      [],

    # --- SHAP / interpretability ---
    "shap_last_run":           None,
    "shap_report_path":        None,
    "shap_top_features":       {},

    # --- literature scout ---
    "literature_last_run":     None,   # ISO timestamp
    "literature_seen_ids":     [],     # list of paper IDs (PMID:… / DOI:…)
    "literature_digest_path":  None,   # path to latest HTML digest

    # --- agent run history (last 50 entries) ---
    "agent_run_log":           [],
}

_MAX_RUN_LOG = 50
_MAX_DRIFT_HISTORY = 200


class SharedState:
    """
    Thread-safe key-value store backed by a JSON file.

    Usage
    -----
    state = SharedState()
    state.set("drift_score", 0.032)
    score = state.get("drift_score")
    state.append_to("pending_review", {"variant": "rs123", "reason": "..."})
    """

    def __init__(self, path: Path = SHARED_STATE_PATH):
        self._path = Path(path)
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._save(DEFAULT_STATE.copy())
            log.info("Initialised new shared state at %s", self._path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        data = self._load()
        return data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            data = self._load()
            data[key] = value
            self._save(data)

    def get_all(self) -> dict:
        return self._load()

    def append_to(self, key: str, item: Any, max_len: int | None = None) -> None:
        """Append *item* to a list field, optionally capping its length."""
        with self._lock:
            data = self._load()
            lst = data.get(key, [])
            if not isinstance(lst, list):
                raise TypeError(f"State field '{key}' is not a list")
            lst.append(item)
            if max_len:
                lst = lst[-max_len:]
            data[key] = lst
            self._save(data)

    def log_agent_run(self, agent_name: str, action: str, outcome: str,
                      details: dict | None = None) -> None:
        entry = {
            "timestamp": _now_iso(),
            "agent":     agent_name,
            "action":    action,
            "outcome":   outcome,
            "details":   details or {},
        }
        self.append_to("agent_run_log", entry, max_len=_MAX_RUN_LOG)

    def add_pending_review(self, item: dict) -> None:
        """Add an item requiring human sign-off."""
        item.setdefault("added_at", _now_iso())
        item.setdefault("resolved", False)
        self.append_to("pending_review", item)
        log.warning("⚠  Pending review item added: %s", item.get("reason", "—"))

    def resolve_pending_review(self, index: int) -> None:
        with self._lock:
            data = self._load()
            queue = data.get("pending_review", [])
            if 0 <= index < len(queue):
                queue[index]["resolved"] = True
                queue[index]["resolved_at"] = _now_iso()
            data["pending_review"] = queue
            self._save(data)

    def unresolved_review_items(self) -> list[dict]:
        return [i for i in self.get("pending_review", []) if not i.get("resolved")]

    def record_drift(self, score: float) -> None:
        self.set("drift_score", score)
        self.append_to(
            "drift_history",
            {"timestamp": _now_iso(), "score": score},
            max_len=_MAX_DRIFT_HISTORY,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(self) -> dict:
        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, FileNotFoundError) as exc:
            log.error("Could not load shared state: %s — resetting to defaults", exc)
            return DEFAULT_STATE.copy()

    def _save(self, data: dict) -> None:
        tmp = self._path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)
        tmp.replace(self._path)   # atomic on POSIX; near-atomic on Windows


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
