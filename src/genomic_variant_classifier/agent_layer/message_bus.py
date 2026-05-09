"""
message_bus.py — Agent-to-Agent Message Bus
============================================
Provides typed, persistent, crash-safe inter-agent messaging for the
genomic variant classifier agent layer.

Inspired by OpenClaw's sessions_send / sessions_list / sessions_history model,
adapted to the existing SharedState + JSON architecture.

Message lifecycle
-----------------
  send()        Agent writes a message to another agent's inbox in SharedState.
  get_inbox()   Agent reads all messages addressed to it.
  get_unread()  Convenience: only unread messages.
  mark_read()   Agent marks a message consumed (does not delete — keeps history).
  pending_approval()  Returns messages flagged requires_approval=True that have
                      not yet been approved or rejected.
  approve() / reject()  Human (or orchestrator) approves/rejects a pending message.
  history()     Returns full message history for one or all agents (like
                OpenClaw's sessions_history).

Storage layout inside shared_state.json
----------------------------------------
{
  ...existing keys...,
  "agent_messages": {
    "TrainingLifecycleAgent": [
      {
        "id": "uuid4",
        "from_agent": "DataFreshnessAgent",
        "to_agent": "TrainingLifecycleAgent",
        "subject": "DATA_UPDATED",
        "payload": {...},
        "timestamp": "2026-04-09T12:00:00Z",
        "priority": "normal",
        "read": false,
        "requires_approval": true,
        "approved": null   // null=pending, true=approved, false=rejected
      }
    ],
    "InterpretabilityAgent": [...],
    ...
  }
}

Message subjects (canonical constants)
---------------------------------------
  DATA_UPDATED              DataFreshness  → TrainingLifecycle
  CHECKPOINT_READY          TrainingLifecycle → Interpretability
  FEATURE_INSTABILITY       Interpretability  → TrainingLifecycle
  FEATURE_CANDIDATE_ADDED   LiteratureScout   → TrainingLifecycle

Priority levels
---------------
  "low"     Informational; agent may defer action.
  "normal"  Standard inter-agent notification.
  "high"    Requires immediate attention on next agent run.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

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
# Canonical message subject constants
# ---------------------------------------------------------------------------

DATA_UPDATED = "DATA_UPDATED"
CHECKPOINT_READY = "CHECKPOINT_READY"
FEATURE_INSTABILITY = "FEATURE_INSTABILITY"
FEATURE_CANDIDATE_ADDED = "FEATURE_CANDIDATE_ADDED"

ALL_SUBJECTS = {
    DATA_UPDATED,
    CHECKPOINT_READY,
    FEATURE_INSTABILITY,
    FEATURE_CANDIDATE_ADDED,
}

# Subjects that require human approval before the recipient acts on them.
# These map to consequential downstream actions (retraining, SHAP re-run).
APPROVAL_REQUIRED_SUBJECTS = {
    DATA_UPDATED,  # triggers Spark ingest + retrain
    CHECKPOINT_READY,  # triggers SHAP audit
}

# ---------------------------------------------------------------------------
# Priority constants
# ---------------------------------------------------------------------------

PRIORITY_LOW = "low"
PRIORITY_NORMAL = "normal"
PRIORITY_HIGH = "high"


# ---------------------------------------------------------------------------
# MessageBus
# ---------------------------------------------------------------------------


class MessageBus:
    """
    Thin wrapper around SharedState that provides typed inter-agent messaging.

    All reads and writes go through the SharedState atomic-write mechanism, so
    the bus is safe across concurrent processes and survives crashes mid-write.

    Usage (from inside a BaseAgent subclass via the inherited helpers):

        # Send
        self.send_message(
            to="TrainingLifecycleAgent",
            subject=message_bus.DATA_UPDATED,
            payload={"source": "gnomAD", "release": "v4.2"},
            priority=message_bus.PRIORITY_HIGH,
        )

        # Receive
        for msg in self.read_inbox():
            if msg["subject"] == message_bus.DATA_UPDATED:
                ...

    Or call the bus directly:

        bus = MessageBus(shared_state)
        bus.send("DataFreshnessAgent", "TrainingLifecycleAgent",
                 DATA_UPDATED, {"source": "gnomAD"})
    """

    _MESSAGES_KEY = "agent_messages"

    def __init__(self, shared_state) -> None:
        """
        Parameters
        ----------
        shared_state : SharedState
            The project's SharedState instance. MessageBus reads and writes
            through its load() / save() interface.
        """
        self._state = shared_state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_all(self) -> dict[str, list[dict]]:
        """Return the full agent_messages dict from SharedState."""
        state = self._state.load()
        return state.get(self._MESSAGES_KEY, {})

    def _save_all(self, messages: dict[str, list[dict]]) -> None:
        """Persist the full agent_messages dict back to SharedState."""
        state = self._state.load()
        state[self._MESSAGES_KEY] = messages
        self._state.save(state)

    def _ensure_inbox(
        self, messages: dict[str, list[dict]], agent_name: str
    ) -> dict[str, list[dict]]:
        """Create an inbox for agent_name if it doesn't exist yet."""
        if agent_name not in messages:
            messages[agent_name] = []
        return messages

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def send(
        self,
        from_agent: str,
        to_agent: str,
        subject: str,
        payload: dict[str, Any] | None = None,
        priority: str = PRIORITY_NORMAL,
        requires_approval: bool | None = None,
    ) -> str:
        """
        Send a message from one agent to another.

        Parameters
        ----------
        from_agent : str
            Sending agent class name (e.g. "DataFreshnessAgent").
        to_agent : str
            Receiving agent class name.
        subject : str
            One of the canonical subject constants (DATA_UPDATED, etc.).
        payload : dict, optional
            Arbitrary JSON-serialisable data. Defaults to {}.
        priority : str
            PRIORITY_LOW | PRIORITY_NORMAL | PRIORITY_HIGH.
        requires_approval : bool | None
            If None, defaults to True for subjects in APPROVAL_REQUIRED_SUBJECTS,
            False otherwise.

        Returns
        -------
        str
            The message UUID, for later reference.
        """
        if subject not in ALL_SUBJECTS:
            raise ValueError(
                f"Unknown message subject '{subject}'. "
                f"Use one of: {sorted(ALL_SUBJECTS)}"
            )

        if requires_approval is None:
            requires_approval = subject in APPROVAL_REQUIRED_SUBJECTS

        msg_id = str(uuid.uuid4())
        message: dict[str, Any] = {
            "id": msg_id,
            "from_agent": from_agent,
            "to_agent": to_agent,
            "subject": subject,
            "payload": payload or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "priority": priority,
            "read": False,
            "requires_approval": requires_approval,
            "approved": None,  # None=pending, True=approved, False=rejected
        }

        messages = self._load_all()
        self._ensure_inbox(messages, to_agent)
        messages[to_agent].append(message)
        self._save_all(messages)

        logger.info(
            "📨  %s → %s  [%s]  id=%s  approval_required=%s",
            from_agent,
            to_agent,
            subject,
            msg_id[:8],
            requires_approval,
        )
        return msg_id

    def get_inbox(self, agent_name: str) -> list[dict]:
        """Return all messages (read and unread) addressed to agent_name."""
        messages = self._load_all()
        return list(messages.get(agent_name, []))

    def get_unread(self, agent_name: str) -> list[dict]:
        """Return only unread messages addressed to agent_name."""
        return [m for m in self.get_inbox(agent_name) if not m["read"]]

    def get_actionable(self, agent_name: str) -> list[dict]:
        """
        Return messages the agent can act on right now:
          - unread, AND
          - either does not require approval, OR has been approved.
        """
        return [
            m
            for m in self.get_unread(agent_name)
            if (not m["requires_approval"]) or (m["approved"] is True)
        ]

    def pending_approval(self, agent_name: str | None = None) -> list[dict]:
        """
        Return messages waiting for human approval.

        If agent_name is None, returns pending messages across all agents.
        """
        messages = self._load_all()
        results = []
        targets = [agent_name] if agent_name else list(messages.keys())
        for name in targets:
            for m in messages.get(name, []):
                if m.get("requires_approval") and m.get("approved") is None:
                    results.append(m)
        return results

    def mark_read(self, agent_name: str, msg_id: str) -> bool:
        """
        Mark a specific message as read.

        Returns True if found and updated, False if not found.
        """
        messages = self._load_all()
        inbox = messages.get(agent_name, [])
        for msg in inbox:
            if msg["id"] == msg_id:
                msg["read"] = True
                self._save_all(messages)
                return True
        return False

    def mark_all_read(self, agent_name: str) -> int:
        """Mark all messages for agent_name as read. Returns count marked."""
        messages = self._load_all()
        inbox = messages.get(agent_name, [])
        count = 0
        for msg in inbox:
            if not msg["read"]:
                msg["read"] = True
                count += 1
        if count:
            self._save_all(messages)
        return count

    def approve(self, msg_id: str) -> bool:
        """
        Approve a pending message by ID.

        Returns True if found and approved, False if not found.
        """
        return self._set_approval(msg_id, approved=True)

    def reject(self, msg_id: str) -> bool:
        """
        Reject a pending message by ID.

        Returns True if found and rejected, False if not found.
        """
        return self._set_approval(msg_id, approved=False)

    def _set_approval(self, msg_id: str, approved: bool) -> bool:
        messages = self._load_all()
        for inbox in messages.values():
            for msg in inbox:
                if msg["id"] == msg_id:
                    msg["approved"] = approved
                    self._save_all(messages)
                    verb = "✅ approved" if approved else "❌ rejected"
                    logger.info(
                        "Message %s → %s  [%s]  %s",
                        msg_id[:8],
                        msg["to_agent"],
                        msg["subject"],
                        verb,
                    )
                    return True
        return False

    # ------------------------------------------------------------------
    # Inspection (analogous to OpenClaw's sessions_history)
    # ------------------------------------------------------------------

    def history(
        self,
        agent_name: str | None = None,
        subject_filter: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """
        Return message history, optionally filtered by agent and/or subject.

        Parameters
        ----------
        agent_name : str | None
            If given, return only messages for that agent's inbox.
            If None, return messages across all inboxes.
        subject_filter : str | None
            If given, return only messages with that subject.
        limit : int
            Maximum number of messages to return (most recent first).
        """
        messages = self._load_all()
        all_msgs: list[dict] = []

        targets = [agent_name] if agent_name else list(messages.keys())
        for name in targets:
            all_msgs.extend(messages.get(name, []))

        if subject_filter:
            all_msgs = [m for m in all_msgs if m["subject"] == subject_filter]

        # Sort by timestamp descending (most recent first)
        all_msgs.sort(key=lambda m: m["timestamp"], reverse=True)
        return all_msgs[:limit]

    def agent_list(self) -> list[str]:
        """
        Return a list of all agents that have an inbox.
        Analogous to OpenClaw's sessions_list.
        """
        return list(self._load_all().keys())

    # ------------------------------------------------------------------
    # CLI helpers (used by run_agents.py)
    # ------------------------------------------------------------------

    def print_inbox(self, agent_name: str) -> None:
        """Pretty-print the inbox for the given agent to stdout."""
        msgs = self.get_inbox(agent_name)
        if not msgs:
            print(f"  (no messages for {agent_name})")
            return
        for m in sorted(msgs, key=lambda x: x["timestamp"]):
            status = "✉ unread" if not m["read"] else "✓ read  "
            approval = ""
            if m["requires_approval"]:
                if m["approved"] is None:
                    approval = " [⏳ awaiting approval]"
                elif m["approved"]:
                    approval = " [✅ approved]"
                else:
                    approval = " [❌ rejected]"
            print(
                f"  {status} | {m['timestamp'][:19]} | "
                f"{m['from_agent']:<30} → {m['subject']}{approval}"
            )
            if m["payload"]:
                for k, v in m["payload"].items():
                    print(f"             {k}: {v}")

    def print_all_pending(self) -> None:
        """Pretty-print all messages awaiting approval."""
        pending = self.pending_approval()
        if not pending:
            print("  (no messages pending approval)")
            return
        for m in pending:
            print(
                f"  [{m['id'][:8]}] {m['from_agent']} → {m['to_agent']}  "
                f"[{m['subject']}]  {m['timestamp'][:19]}"
            )
            if m["payload"]:
                for k, v in m["payload"].items():
                    print(f"    {k}: {v}")
