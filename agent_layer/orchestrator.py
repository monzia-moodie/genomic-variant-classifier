"""
orchestrator.py — Agent Orchestrator
=====================================
Manages agent execution order, pipeline definitions, and the human-review
queue. Now also coordinates inter-agent message delivery between runs and
exposes the MessageBus inspection surface to the CLI.

Changes from existing implementation
--------------------------------------
  1. _deliver_pending_messages()  [NEW]
         Called before each pipeline run. Logs a summary of unread messages
         so you can see what cross-agent signals are waiting before agents
         execute. Messages are not moved or copied — agents read their own
         inboxes via get_actionable() during run(). This method is purely
         informational / audit.

  2. _report_message_status()  [NEW]
         Called after each pipeline run. Prints a concise per-agent inbox
         summary to the log so you always know the post-run message state.

  3. approve_message(msg_id) / reject_message(msg_id)  [NEW]
         Called by run_agents.py --approve-msg / --reject-msg.
         Delegates to MessageBus.approve() / .reject().

  4. print_inbox(agent_name)  [NEW]
         Called by run_agents.py --inbox <AgentName>.
         Delegates to MessageBus.print_inbox().

  5. print_all_pending_messages()  [NEW]
         Called by run_agents.py --pending-msgs.
         Delegates to MessageBus.print_all_pending().

  6. status() / summary  [EXTENDED]
         Now includes unread message count and pending-approval count
         alongside the existing review-item summary.

All existing pipeline definitions, execution logic, dry-run handling,
review-item management, and --status / --reviews / --candidates surfaces
are completely unchanged.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone

from config import REQUIRE_HUMAN_APPROVAL
from message_bus import MessageBus
from shared_state import SharedState

logger = logging.getLogger("Orchestrator")
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

_DIVIDER = "═" * 60


# ---------------------------------------------------------------------------
# Pipeline definitions — unchanged
# ---------------------------------------------------------------------------

PIPELINE_DEFINITIONS: dict[str, list[str]] = {
    "data_freshness": ["DataFreshnessAgent"],
    "training": ["TrainingLifecycleAgent"],
    "interpretability": ["InterpretabilityAgent"],
    "literature": ["LiteratureScoutAgent"],
    "full": [
        "DataFreshnessAgent",
        "TrainingLifecycleAgent",
        "InterpretabilityAgent",
        "LiteratureScoutAgent",
    ],
}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class Orchestrator:
    """
    Runs agent pipelines in sequence and manages the human-review queue
    and inter-agent message bus.
    """

    def __init__(
        self,
        shared_state: SharedState,
        dry_run: bool = False,
    ) -> None:
        self._state = shared_state
        self._dry_run = dry_run
        self._bus = MessageBus(shared_state)

        # Lazy-import agent classes to avoid circular imports at module load
        self._agent_registry: dict[str, type] = {}
        self._register_agents()

        logger.info("Orchestrator initialised  [dry_run=%s]", dry_run)

    # ------------------------------------------------------------------
    # Agent registry — unchanged
    # ------------------------------------------------------------------

    def _register_agents(self) -> None:
        from agents.data_freshness_agent import DataFreshnessAgent
        from agents.training_lifecycle_agent import TrainingLifecycleAgent
        from agents.interpretability_agent import InterpretabilityAgent
        from agents.literature_scout_agent import LiteratureScoutAgent

        self._agent_registry = {
            "DataFreshnessAgent": DataFreshnessAgent,
            "TrainingLifecycleAgent": TrainingLifecycleAgent,
            "InterpretabilityAgent": InterpretabilityAgent,
            "LiteratureScoutAgent": LiteratureScoutAgent,
        }

    # ------------------------------------------------------------------
    # Pipeline execution — extended with message delivery hooks
    # ------------------------------------------------------------------

    def run_pipeline(self, pipeline_name: str) -> dict[str, dict]:
        """
        Execute a named pipeline.

        Parameters
        ----------
        pipeline_name : str
            One of the keys in PIPELINE_DEFINITIONS.

        Returns
        -------
        dict[str, dict]
            Maps agent name → result dict returned by agent.run().
        """
        agent_names = PIPELINE_DEFINITIONS.get(pipeline_name)
        if not agent_names:
            raise ValueError(
                f"Unknown pipeline '{pipeline_name}'. "
                f"Available: {sorted(PIPELINE_DEFINITIONS)}"
            )

        # Check / prompt for unresolved review items (unchanged)
        self._check_review_items()

        logger.info(_DIVIDER)
        logger.info(
            "Starting pipeline: %s  (%d agents)", pipeline_name, len(agent_names)
        )
        logger.info(_DIVIDER)

        # --- NEW: log pre-run message state ---
        self._deliver_pending_messages(pipeline_name, agent_names)

        results: dict[str, dict] = {}

        for agent_name in agent_names:
            agent_cls = self._agent_registry.get(agent_name)
            if not agent_cls:
                logger.error("Unknown agent '%s' — skipping.", agent_name)
                continue

            agent = agent_cls(self._state)
            try:
                result = agent.run(dry_run=self._dry_run)
            except Exception as exc:
                logger.error(
                    "Agent %s raised an unhandled exception: %s",
                    agent_name,
                    exc,
                    exc_info=True,
                )
                result = {"action": "error", "error": str(exc)}

            results[agent_name] = result

        # --- NEW: log post-run message state ---
        self._report_message_status()

        logger.info(_DIVIDER)
        logger.info("Pipeline complete. Agents run: %d", len(results))
        for name, res in results.items():
            action = res.get("action", "unknown")
            logger.info("  ✓ %-30s action=%-25s", name, action)
        logger.info(_DIVIDER)

        return results

    # ------------------------------------------------------------------
    # NEW: pre-run message delivery summary
    # ------------------------------------------------------------------

    def _deliver_pending_messages(
        self, pipeline_name: str, agent_names: list[str]
    ) -> None:
        """
        Log a summary of unread inter-agent messages before the pipeline
        executes. Agents read their own inboxes via get_actionable() during
        run() — this method is purely informational.

        Also warns if any messages are awaiting approval, so you can approve
        them before the pipeline runs (via --approve-msg <id>).
        """
        all_unread = 0
        all_pending_approval = 0

        for name in agent_names:
            inbox = self._bus.get_inbox(name)
            unread = [m for m in inbox if not m["read"]]
            pending = [
                m
                for m in unread
                if m.get("requires_approval") and m.get("approved") is None
            ]
            actionable = [
                m
                for m in unread
                if (not m.get("requires_approval")) or (m.get("approved") is True)
            ]

            if unread:
                logger.info(
                    "📬  %s inbox: %d unread  " "(%d actionable, %d awaiting approval)",
                    name,
                    len(unread),
                    len(actionable),
                    len(pending),
                )
                for m in unread:
                    approval_tag = ""
                    if m.get("requires_approval"):
                        if m.get("approved") is None:
                            approval_tag = " [⏳ awaiting approval — use --approve-msg]"
                        elif m.get("approved"):
                            approval_tag = " [✅ approved]"
                        else:
                            approval_tag = " [❌ rejected]"
                    logger.info(
                        "   ↳ [%s] %s → %s  [%s]%s",
                        m["id"][:8],
                        m["from_agent"],
                        m["to_agent"],
                        m["subject"],
                        approval_tag,
                    )

            all_unread += len(unread)
            all_pending_approval += len(pending)

        if all_pending_approval > 0:
            logger.warning(
                "⚠  %d message(s) awaiting approval will NOT be acted on "
                "until approved.  Run: python run_agents.py --pending-msgs "
                "then: python run_agents.py --approve-msg <id>",
                all_pending_approval,
            )

    # ------------------------------------------------------------------
    # NEW: post-run message state report
    # ------------------------------------------------------------------

    def _report_message_status(self) -> None:
        """
        After the pipeline completes, log a concise per-agent inbox summary
        so you can see what new messages were generated by this run.
        """
        state = self._state.load()
        agent_msgs = state.get("agent_messages", {})

        if not agent_msgs:
            return

        total_unread = sum(
            1 for inbox in agent_msgs.values() for m in inbox if not m.get("read")
        )
        if total_unread == 0:
            return

        logger.info("── Post-run message summary ──")
        for agent_name, inbox in agent_msgs.items():
            unread = [m for m in inbox if not m.get("read")]
            if unread:
                logger.info("  %s: %d unread message(s)", agent_name, len(unread))
                for m in unread:
                    logger.info(
                        "   ↳ [%s] from %-30s  [%s]  priority=%s",
                        m["id"][:8],
                        m["from_agent"],
                        m["subject"],
                        m.get("priority", "normal"),
                    )

    # ------------------------------------------------------------------
    # NEW: MessageBus CLI delegates
    # ------------------------------------------------------------------

    def approve_message(self, msg_id: str) -> None:
        """Approve a pending message by ID. Called by --approve-msg."""
        if self._bus.approve(msg_id):
            logger.info("Message %s approved.", msg_id[:8])
        else:
            logger.warning(
                "Message ID '%s' not found — check --pending-msgs for IDs.",
                msg_id,
            )

    def reject_message(self, msg_id: str) -> None:
        """Reject a pending message by ID. Called by --reject-msg."""
        if self._bus.reject(msg_id):
            logger.info("Message %s rejected.", msg_id[:8])
        else:
            logger.warning(
                "Message ID '%s' not found — check --pending-msgs for IDs.",
                msg_id,
            )

    def print_inbox(self, agent_name: str) -> None:
        """Pretty-print an agent's full inbox. Called by --inbox."""
        known = list(self._agent_registry.keys())
        if agent_name not in known:
            print(f"Unknown agent '{agent_name}'. " f"Known agents: {', '.join(known)}")
            return
        print(f"\n📬  Inbox for {agent_name}:")
        self._bus.print_inbox(agent_name)

    def print_all_pending_messages(self) -> None:
        """Pretty-print all messages awaiting approval. Called by --pending-msgs."""
        print("\n⏳  Messages awaiting approval:")
        self._bus.print_all_pending()

    def print_message_history(
        self,
        agent_name: str | None = None,
        limit: int = 20,
    ) -> None:
        """
        Print message history for one agent or all agents.
        Called by --msg-history [AgentName].
        """
        label = agent_name or "all agents"
        print(f"\n📜  Message history ({label}, last {limit}):")
        msgs = self._bus.history(agent_name=agent_name, limit=limit)
        if not msgs:
            print("  (no messages)")
            return
        for m in msgs:
            read_tag = "✓" if m["read"] else "✉"
            approval = ""
            if m.get("requires_approval"):
                if m.get("approved") is None:
                    approval = " [⏳]"
                elif m.get("approved"):
                    approval = " [✅]"
                else:
                    approval = " [❌]"
            print(
                f"  {read_tag} [{m['id'][:8]}] {m['timestamp'][:19]}  "
                f"{m['from_agent']:<30} → {m['to_agent']:<30}  "
                f"[{m['subject']}]{approval}"
            )

    # ------------------------------------------------------------------
    # Existing: review items — unchanged
    # ------------------------------------------------------------------

    def _check_review_items(self) -> None:
        """Prompt the user if unresolved review items exist."""
        unresolved = self._state.unresolved_review_items()
        if not unresolved:
            return

        logger.warning("⚠  %d unresolved review item(s):", len(unresolved))
        for item in unresolved:
            logger.warning(
                "  [%d] %s — %s",
                item["index"],
                item["message"],
                item["timestamp"],
            )

        if not sys.stdin.isatty():
            logger.warning("Non-TTY environment — proceeding past unresolved items.")
            return

        print(f"\n⚠  {len(unresolved)} unresolved review item(s) exist.")
        try:
            answer = input("   Proceed anyway? [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"

        if answer != "y":
            logger.info("Pipeline aborted by user (unresolved review items).")
            sys.exit(0)

    def resolve_review_item(self, index: int) -> None:
        """Mark a review item as resolved. Called by --resolve."""
        if self._state.resolve_review_item(index):
            logger.info("Review item [%d] marked as resolved.", index)
        else:
            logger.warning("Review item [%d] not found.", index)

    # ------------------------------------------------------------------
    # Existing: status / reviews / candidates — extended for messages
    # ------------------------------------------------------------------

    def print_status(self) -> None:
        """Print a one-line system status summary. Called by --status."""
        print("\n── Agent Layer Status ──")
        print(f"  {self._state.summary()}")

    def print_reviews(self) -> None:
        """Print all unresolved review items. Called by --reviews."""
        items = self._state.unresolved_review_items()
        print("\n── Unresolved Review Items ──")
        if not items:
            print("  (none)")
            return
        for item in items:
            print(
                f"  [{item['index']}] {item['timestamp'][:19]}  " f"{item['message']}"
            )

    def print_candidates(self) -> None:
        """Print pending feature candidates. Called by --candidates."""
        state = self._state.load()
        candidates = state.get("training", {}).get(
            "pending_feature_candidates", []
        ) or state.get("literature", {}).get("feature_candidates", [])
        print("\n── Pending Feature Candidates ──")
        if not candidates:
            print("  (none)")
            return
        for i, c in enumerate(candidates):
            reviewed = "✓" if c.get("reviewed") else "·"
            print(
                f"  [{i}] {reviewed} {c.get('name', '?'):<35} "
                f"source={c.get('literature_source', '?'):<10} "
                f"relevance={c.get('relevance_score', 0):.2f}"
            )

    def mark_candidate_reviewed(self, index: int) -> None:
        """Mark a feature candidate as reviewed. Called by --mark-candidate."""
        state = self._state.load()
        candidates = state.get("training", {}).get("pending_feature_candidates", [])
        if 0 <= index < len(candidates):
            candidates[index]["reviewed"] = True
            self._state.save(state)
            logger.info(
                "Feature candidate [%d] '%s' marked as reviewed.",
                index,
                candidates[index].get("name", "?"),
            )
        else:
            logger.warning("Candidate index [%d] not found.", index)
