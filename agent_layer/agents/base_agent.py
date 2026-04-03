"""
agents/base_agent.py
====================
Abstract base class that every specialist agent inherits from.

Each concrete agent must implement:
  - run() → dict   :  execute the agent's primary task; return a result dict
  - name  (property): human-readable agent name

Optional overrides:
  - pre_run()   : called before run(); good for connectivity checks
  - post_run()  : called after run() regardless of success; good for cleanup
"""

import logging
import sys
import traceback
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
_AL = Path(__file__).resolve().parent.parent
for _p in (str(_AL), str(_AL / "agents")):
    if _p not in sys.path: sys.path.insert(0, _p)
from typing import Any

from config import AUDIT_LOG_DIR, LOG_LEVEL
from shared_state import SharedState


def _configure_logging(agent_name: str) -> logging.Logger:
    AUDIT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = AUDIT_LOG_DIR / f"{agent_name.lower().replace(' ', '_')}.log"

    logger = logging.getLogger(agent_name)
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    logger.propagate = False  # prevent double-logging via root logger

    if not logger.handlers:
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ",
        )
        # File handler
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    return logger


class AgentResult:
    """Structured return value from an agent run."""

    def __init__(
        self,
        success: bool,
        action: str,
        details: dict[str, Any] | None = None,
        errors: list[str] | None = None,
    ):
        self.success   = success
        self.action    = action
        self.details   = details or {}
        self.errors    = errors or []
        self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "success":   self.success,
            "action":    self.action,
            "details":   self.details,
            "errors":    self.errors,
            "timestamp": self.timestamp,
        }

    def __repr__(self) -> str:
        status = "✓" if self.success else "✗"
        return f"AgentResult({status} {self.action})"


class BaseAgent(ABC):
    """
    All specialist agents extend this class.

    Parameters
    ----------
    state : SharedState
        Shared state instance injected by the Orchestrator.
    dry_run : bool
        If True, the agent describes what it *would* do but makes no
        mutations to the corpus, model, or shared state.
    """

    def __init__(self, state: SharedState, dry_run: bool = False):
        self.state   = state
        self.dry_run = dry_run
        self.log     = _configure_logging(self.name)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Short human-readable agent name used in logs and state entries."""

    @abstractmethod
    def run(self) -> AgentResult:
        """
        Execute the agent's primary task.

        Must return an AgentResult.  Never raise — catch exceptions internally
        and return AgentResult(success=False, ...).
        """

    # ------------------------------------------------------------------
    # Lifecycle hooks (optional overrides)
    # ------------------------------------------------------------------

    def pre_run(self) -> None:
        """Called by execute() before run(). Override for health checks."""

    def post_run(self, result: AgentResult) -> None:
        """Called by execute() after run(), even on failure."""

    # ------------------------------------------------------------------
    # Main entry point (called by Orchestrator)
    # ------------------------------------------------------------------

    def execute(self) -> AgentResult:
        """
        Orchestrator-facing entry point.
        Wraps run() with logging, error handling, and state persistence.
        """
        self.log.info("▶  Starting agent: %s  [dry_run=%s]", self.name, self.dry_run)

        try:
            self.pre_run()
        except Exception:
            self.log.warning("pre_run() raised:\n%s", traceback.format_exc())

        try:
            result = self.run()
        except Exception as exc:
            self.log.error("Unhandled exception in run():\n%s", traceback.format_exc())
            result = AgentResult(
                success=False,
                action="run",
                errors=[str(exc)],
            )

        try:
            self.post_run(result)
        except Exception:
            self.log.warning("post_run() raised:\n%s", traceback.format_exc())

        # Persist to shared state audit log
        self.state.log_agent_run(
            agent_name=self.name,
            action=result.action,
            outcome="success" if result.success else "failure",
            details=result.details,
        )

        status_icon = "✓" if result.success else "✗"
        self.log.info("%s  Agent finished: %s", status_icon, self.name)
        return result

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    def require_human_approval(self, prompt: str) -> bool:
        """
        Pause and ask the operator for approval via stdin.
        In non-interactive (CI/GCP) environments, defaults to False so that
        unapproved actions are blocked rather than auto-approved.
        """
        import sys
        if not sys.stdin.isatty():
            self.log.warning(
                "Human approval required but stdin is not a TTY. "
                "Blocking action: %s", prompt
            )
            return False
        print(f"\n⚠  Human approval required\n   {prompt}\n   Approve? [y/N]: ", end="")
        response = input().strip().lower()
        return response == "y"

    def _dry_run_log(self, message: str) -> None:
        self.log.info("[DRY RUN] %s", message)