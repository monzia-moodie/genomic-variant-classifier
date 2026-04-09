"""
base_agent.py — Abstract Base Class for All Agents
===================================================
Provides the shared interface every agent inherits:
  - run()             Entry point called by the Orchestrator.
  - send_message()    Send a typed message to another agent via the MessageBus.
  - read_inbox()      Read all messages in this agent's inbox.
  - get_actionable()  Read only messages that are ready to act on (unread +
                      either no approval required, or already approved).
  - _require_approval()  Gate for consequential actions; adds a review item
                         if REQUIRE_HUMAN_APPROVAL is True.
  - _log_*            Structured logging helpers.

MessageBus integration
----------------------
Each agent receives the SharedState instance from the Orchestrator at
construction time. BaseAgent instantiates a MessageBus from it, so all
subclasses get send_message() / read_inbox() for free with no extra wiring.

The agent's canonical name (self.name) is used as both the inbox address
and the from_agent field on outbound messages, so routing is automatic.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from config import REQUIRE_HUMAN_APPROVAL
from message_bus import MessageBus
from shared_state import SharedState


class BaseAgent(ABC):
    """
    Abstract base for all genomic-classifier agents.

    Subclasses must implement:
        run(dry_run: bool) -> dict

    Subclasses should call:
        self.send_message(to, subject, payload, priority)
        self.get_actionable()
        self._require_approval(prompt) -> bool
    """

    def __init__(self, shared_state: SharedState) -> None:
        self._state = shared_state
        self._bus = MessageBus(shared_state)

        # Logger — named after the concrete subclass, not BaseAgent
        self.logger = logging.getLogger(self.name)
        self.logger.propagate = False
        if not self.logger.handlers:
            _h = logging.StreamHandler()
            _h.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
                    datefmt="%Y-%m-%dT%H:%M:%SZ",
                )
            )
            self.logger.addHandler(_h)
            self.logger.setLevel(logging.INFO)

    # ------------------------------------------------------------------
    # Identity — subclasses may override if the class name is not ideal
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Canonical agent name used for logging, inbox addressing, and state keys."""
        return self.__class__.__name__

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def run(self, dry_run: bool = False) -> dict:
        """
        Execute the agent's primary task.

        Parameters
        ----------
        dry_run : bool
            If True the agent must not write any external state or trigger
            any downstream action. Internal SharedState updates are allowed
            so that dry-run output is meaningful.

        Returns
        -------
        dict
            Result summary suitable for Orchestrator logging, e.g.:
            {"action": "poll_and_flag", "changes": 2, "triggered": False}
        """

    # ------------------------------------------------------------------
    # Inter-agent messaging (OpenClaw-inspired)
    # ------------------------------------------------------------------

    def send_message(
        self,
        to: str,
        subject: str,
        payload: dict[str, Any] | None = None,
        priority: str = "normal",
        requires_approval: bool | None = None,
    ) -> str:
        """
        Send a typed message to another agent's inbox.

        Parameters
        ----------
        to : str
            Recipient agent class name (e.g. "TrainingLifecycleAgent").
        subject : str
            One of the canonical subject constants from message_bus
            (DATA_UPDATED, CHECKPOINT_READY, FEATURE_INSTABILITY,
            FEATURE_CANDIDATE_ADDED).
        payload : dict, optional
            Arbitrary JSON-serialisable context. Defaults to {}.
        priority : str
            "low" | "normal" | "high".
        requires_approval : bool | None
            If None, the MessageBus applies the default rule for the subject
            (DATA_UPDATED and CHECKPOINT_READY require approval by default).

        Returns
        -------
        str
            The message UUID.
        """
        self.logger.info("→ Sending [%s] to %s", subject, to)
        return self._bus.send(
            from_agent=self.name,
            to_agent=to,
            subject=subject,
            payload=payload,
            priority=priority,
            requires_approval=requires_approval,
        )

    def read_inbox(self) -> list[dict]:
        """
        Return all messages (read and unread) in this agent's inbox.

        Prefer get_actionable() for messages the agent should act on.
        """
        return self._bus.get_inbox(self.name)

    def get_actionable(self) -> list[dict]:
        """
        Return messages this agent can act on right now:
          - Unread, AND
          - Either requires no approval, OR has been approved by a human.

        Call this at the start of run() to pick up cross-agent triggers.
        """
        msgs = self._bus.get_actionable(self.name)
        if msgs:
            self.logger.info("📬  %d actionable message(s) in inbox.", len(msgs))
        return msgs

    def mark_message_read(self, msg_id: str) -> None:
        """Mark a specific inbox message as read after processing it."""
        self._bus.mark_read(self.name, msg_id)

    # ------------------------------------------------------------------
    # Human-in-the-loop approval gate (unchanged from original)
    # ------------------------------------------------------------------

    def _require_approval(self, prompt: str, dry_run: bool = False) -> bool:
        """
        Gate for consequential actions.

        If REQUIRE_HUMAN_APPROVAL is True (config.py), prompts stdin.
        In dry_run mode, always returns False without prompting.
        In a non-TTY environment (GCP/Colab), queues a review item and
        returns False so the action is blocked but recorded.

        Parameters
        ----------
        prompt : str
            Description of the action requiring approval.
        dry_run : bool
            Pass-through from run(); suppresses the prompt in dry-run mode.

        Returns
        -------
        bool
            True if the action is approved, False otherwise.
        """
        if dry_run:
            self.logger.info("  [dry-run] Skipping approval gate: %s", prompt)
            return False

        if not REQUIRE_HUMAN_APPROVAL:
            return True

        import sys

        if not sys.stdin.isatty():
            review_msg = f"{prompt} — blocked pending human approval"
            self._state.add_review_item(review_msg)
            return False

        print(f"\n⚠  Human approval required\n   {prompt}")
        try:
            answer = input("   Approve? [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"
        approved = answer == "y"
        if not approved:
            review_msg = f"{prompt} — blocked pending human approval"
            self._state.add_review_item(review_msg)
        return approved

    # ------------------------------------------------------------------
    # Structured logging helpers
    # ------------------------------------------------------------------

    def _log_start(self, dry_run: bool) -> None:
        self.logger.info("▶  Starting agent: %s  [dry_run=%s]", self.name, dry_run)

    def _log_finish(self, result: dict) -> None:
        action = result.get("action", "unknown")
        self.logger.info("✓  Agent finished: %s  action=%s", self.name, action)

    def _log_section(self, label: str) -> None:
        self.logger.info("── %s ──", label)

    # ------------------------------------------------------------------
    # SharedState convenience pass-throughs
    # ------------------------------------------------------------------

    def _get_section(self, section: str) -> dict:
        return self._state.get_section(section)

    def _update_section(self, section: str, updates: dict) -> None:
        self._state.update_section(section, updates)

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()
