"""
shared_state.py — Persistent Shared State for Agent Layer
==========================================================
Single source of truth for all agent state. Uses atomic write (tmp → rename)
to avoid corruption if a run is interrupted mid-write.

The JSON file is the only coupling between agents — safe to inspect or edit
manually at any time.

Schema (all keys)
-----------------
{
  // --- Existing keys (unchanged) ---
  "data_freshness": {
    "clinvar":       {"last_seen": "<etag-or-date>", "last_checked": "<iso>"},
    "gnomad":        {"last_seen": "<fingerprint>",  "last_checked": "<iso>"},
    "lovd":          {"last_seen": "<etag-or-date>", "last_checked": "<iso>"},
    "alphamissense": {"last_seen": "<etag>",          "last_checked": "<iso>"}
  },
  "training": {
    "last_run":        "<iso> | null",
    "last_checkpoint": "<path> | null",
    "trigger_reason":  "<str> | null"
  },
  "interpretability": {
    "last_run":       "<iso> | null",
    "last_report":    "<path> | null",
    "instability_flags": []
  },
  "literature": {
    "last_run":         "<iso> | null",
    "feature_candidates": []
  },
  "review_items": [
    {
      "index":     0,
      "message":   "<str>",
      "timestamp": "<iso>",
      "resolved":  false
    }
  ],

  // --- New key added for inter-agent messaging ---
  "agent_messages": {
    "<AgentName>": [
      {
        "id":               "<uuid4>",
        "from_agent":       "<str>",
        "to_agent":         "<str>",
        "subject":          "<str>",
        "payload":          {},
        "timestamp":        "<iso>",
        "priority":         "normal",
        "read":             false,
        "requires_approval": true,
        "approved":         null
      }
    ]
  }
}
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# Default state file path — sits alongside this module in agent_layer/
# Override by passing state_file= explicitly to SharedState().
_DEFAULT_STATE_FILE = Path(__file__).parent / "agent_state.json"

logger = logging.getLogger(__name__)
logger.propagate = False
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ",
        )
    )
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Default state template
# ---------------------------------------------------------------------------


def _default_state() -> dict:
    """Return a fully-initialised default state dict."""
    return {
        # Existing sections
        "data_freshness": {
            "clinvar": {"last_seen": None, "last_checked": None},
            "gnomad": {"last_seen": None, "last_checked": None},
            "lovd": {"last_seen": None, "last_checked": None},
            "alphamissense": {"last_seen": None, "last_checked": None},
        },
        "training": {
            "last_run": None,
            "last_checkpoint": None,
            "trigger_reason": None,
        },
        "interpretability": {
            "last_run": None,
            "last_report": None,
            "instability_flags": [],
        },
        "literature": {
            "last_run": None,
            "feature_candidates": [],
        },
        "review_items": [],
        # --- New: inter-agent message bus ---
        "agent_messages": {},
    }


# ---------------------------------------------------------------------------
# SharedState
# ---------------------------------------------------------------------------


class SharedState:
    """
    Atomic-write JSON state store shared across all agents.

    All reads go through load(); all writes go through save().
    Writes use a tmp-file + rename pattern to prevent partial writes.
    """

    def __init__(self, state_file: str | Path | None = None) -> None:
        self._path = Path(state_file or _DEFAULT_STATE_FILE)

    # ------------------------------------------------------------------
    # Core load / save
    # ------------------------------------------------------------------

    def load(self) -> dict:
        """
        Load state from disk.

        If the file does not exist, returns a default state (does NOT write
        it to disk — that happens on the first save()).

        Migrates older state files that are missing the agent_messages key
        so existing deployments upgrade transparently.
        """
        if not self._path.exists():
            return _default_state()

        try:
            with self._path.open("r", encoding="utf-8") as f:
                state = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to load state from %s: %s", self._path, exc)
            logger.warning("Returning default state (existing file untouched).")
            return _default_state()

        return self._migrate(state)

    def save(self, state: dict) -> None:
        """
        Atomically write state to disk (tmp → rename).

        Parameters
        ----------
        state : dict
            The full state dict to persist.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(dir=self._path.parent, suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, default=str)
            os.replace(tmp_path, self._path)
        except Exception:
            # Clean up tmp file on failure; re-raise so caller sees the error.
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # ------------------------------------------------------------------
    # Migration — adds new keys to existing state files transparently
    # ------------------------------------------------------------------

    def _migrate(self, state: dict) -> dict:
        """
        Ensure all expected top-level keys exist, adding defaults for any
        that are missing. This allows zero-downtime upgrades when new keys
        are introduced (such as agent_messages).
        """
        defaults = _default_state()
        changed = False

        for key, default_value in defaults.items():
            if key not in state:
                logger.info("State migration: adding missing key '%s'.", key)
                state[key] = default_value
                changed = True

        # Nested migration: ensure sub-keys exist within existing sections
        for section in ("data_freshness",):
            section_defaults = defaults.get(section, {})
            for sub_key, sub_val in section_defaults.items():
                if sub_key not in state.get(section, {}):
                    state.setdefault(section, {})[sub_key] = sub_val
                    changed = True

        if changed:
            # Persist the migrated state immediately so the file stays current
            try:
                self.save(state)
                logger.info("State migration saved to %s.", self._path)
            except Exception as exc:
                logger.warning("State migration save failed: %s", exc)

        return state

    # ------------------------------------------------------------------
    # Convenience accessors (keep orchestrator / agent code clean)
    # ------------------------------------------------------------------

    def add_review_item(self, message: str) -> int:
        """
        Append a human-review item and return its index.

        These are the same review items surfaced by `run_agents.py --reviews`.
        """
        state = self.load()
        items = state.setdefault("review_items", [])
        idx = len(items)
        items.append(
            {
                "index": idx,
                "message": message,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "resolved": False,
            }
        )
        self.save(state)
        logger.warning("⚠  Pending review item added: %s", message)
        return idx

    def resolve_review_item(self, index: int) -> bool:
        """
        Mark review item at index as resolved.

        Returns True if found and resolved, False if index out of range.
        """
        state = self.load()
        items = state.get("review_items", [])
        for item in items:
            if item.get("index") == index:
                item["resolved"] = True
                self.save(state)
                logger.info("Review item [%d] resolved.", index)
                return True
        return False

    def unresolved_review_items(self) -> list[dict]:
        """Return all unresolved review items."""
        state = self.load()
        return [i for i in state.get("review_items", []) if not i.get("resolved")]

    def get_section(self, section: str) -> dict:
        """Return a top-level section of the state (e.g. 'training')."""
        return self.load().get(section, {})

    def update_section(self, section: str, updates: dict) -> None:
        """
        Merge updates into a top-level section and save.

        Parameters
        ----------
        section : str
            Top-level key in the state dict.
        updates : dict
            Keys/values to merge into state[section].
        """
        state = self.load()
        state.setdefault(section, {}).update(updates)
        self.save(state)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable one-line summary of the current state."""
        state = self.load()

        training_last = state.get("training", {}).get("last_run") or "never"
        interp_last = state.get("interpretability", {}).get("last_run") or "never"
        lit_last = state.get("literature", {}).get("last_run") or "never"
        unresolved = len(self.unresolved_review_items())

        # Message bus summary
        agent_msgs = state.get("agent_messages", {})
        total_unread = sum(
            1 for inbox in agent_msgs.values() for m in inbox if not m.get("read")
        )
        pending_approv = sum(
            1
            for inbox in agent_msgs.values()
            for m in inbox
            if m.get("requires_approval") and m.get("approved") is None
        )

        return (
            f"training={training_last[:10] if training_last != 'never' else 'never'} | "
            f"interpretability={interp_last[:10] if interp_last != 'never' else 'never'} | "
            f"literature={lit_last[:10] if lit_last != 'never' else 'never'} | "
            f"review_items_unresolved={unresolved} | "
            f"messages_unread={total_unread} | "
            f"messages_pending_approval={pending_approv}"
        )
