"""
orchestrator.py
===============
Central Orchestrator — coordinates all specialist agents, owns shared state,
enforces the human-in-the-loop gate, and maintains the audit log.

Usage (interactive / cron)
--------------------------
    from orchestrator import Orchestrator

    orch = Orchestrator()
    orch.run_pipeline("data_freshness")
    orch.run_pipeline("full")

Usage (dry-run, safe to run anywhere)
--------------------------------------
    orch = Orchestrator(dry_run=True)
    orch.run_pipeline("full")
"""

import logging
import sys
from pathlib import Path
from typing import Literal

# --- Path setup ---
# Add agent_layer/ and agent_layer/agents/ so all modules resolve
# regardless of where Python is invoked from.
_AGENT_LAYER = Path(__file__).resolve().parent
_AGENTS_DIR  = _AGENT_LAYER / "agents"
for _p in (_AGENT_LAYER, _AGENTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Plain module imports — no package prefix needed because agents/ is on sys.path
from base_agent import AgentResult
from data_freshness_agent import DataFreshnessAgent
from interpretability_agent import InterpretabilityAgent
from literature_scout_agent import LiteratureScoutAgent
from training_lifecycle_agent import TrainingLifecycleAgent
from config import AUDIT_LOG_DIR, LOG_LEVEL, REQUIRE_HUMAN_APPROVAL
from shared_state import SharedState

log = logging.getLogger("Orchestrator")


# ---------------------------------------------------------------------------
# Run-condition helpers
# ---------------------------------------------------------------------------

def _should_run_training(state: SharedState) -> bool:
    """Run training only if DataFreshnessAgent triggered a Spark ingest."""
    log_entries = state.get("agent_run_log") or []
    recent = [
        e for e in log_entries[-10:]
        if e.get("agent") == "DataFreshnessAgent"
        and e.get("outcome") == "success"
        and e.get("details", {}).get("spark_triggered")
    ]
    return len(recent) > 0


def _should_run_interpretability(state: SharedState) -> bool:
    """Run SHAP audit after any successful training update."""
    log_entries = state.get("agent_run_log") or []
    recent = [
        e for e in log_entries[-10:]
        if e.get("agent") == "TrainingLifecycleAgent"
        and e.get("outcome") == "success"
        and (e.get("details", {}).get("resnet_updated")
             or e.get("details", {}).get("ensemble_updated"))
    ]
    return len(recent) > 0


def _should_run_literature(state: SharedState) -> bool:
    """Run literature scout at most once every 7 days."""
    from datetime import datetime, timezone, timedelta
    last = state.get("literature_last_run")
    if not last:
        return True
    try:
        dt = datetime.fromisoformat(last)
        return (datetime.now(timezone.utc) - dt) >= timedelta(days=7)
    except (ValueError, TypeError):
        return True


# ---------------------------------------------------------------------------
# Pipeline registry
# ---------------------------------------------------------------------------

PIPELINE_REGISTRY: dict[str, list] = {
    "data_freshness": [
        (DataFreshnessAgent,     lambda _: True),
    ],
    "training": [
        (TrainingLifecycleAgent, lambda _: True),
    ],
    "interpretability": [
        (InterpretabilityAgent,  lambda _: True),
    ],
    "literature": [
        (LiteratureScoutAgent,   lambda _: True),
    ],
    "full": [
        (DataFreshnessAgent,     lambda _: True),
        (TrainingLifecycleAgent, _should_run_training),
        (InterpretabilityAgent,  _should_run_interpretability),
        (LiteratureScoutAgent,   _should_run_literature),
    ],
}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    """
    Coordinates agent execution, shared state, logging, and review gates.

    Parameters
    ----------
    dry_run    : propagated to all agents; no mutations made
    state_path : override shared state JSON path (useful for tests)
    """

    def __init__(self, dry_run: bool = False, state_path: Path | None = None):
        self.dry_run = dry_run
        self.state   = SharedState(state_path) if state_path else SharedState()
        _configure_root_logger()
        log.info("Orchestrator initialised  [dry_run=%s]", dry_run)

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    def run_pipeline(
        self,
        pipeline: Literal["data_freshness", "training", "interpretability",
                          "literature", "full"] = "full",
    ) -> dict[str, AgentResult]:
        if pipeline not in PIPELINE_REGISTRY:
            raise ValueError(
                f"Unknown pipeline '{pipeline}'. "
                f"Choose from: {list(PIPELINE_REGISTRY)}"
            )

        self._check_pending_reviews()
        agent_configs = PIPELINE_REGISTRY[pipeline]
        results: dict[str, AgentResult] = {}

        log.info("═" * 60)
        log.info("Starting pipeline: %s  (%d agents)", pipeline, len(agent_configs))
        log.info("═" * 60)

        for AgentClass, condition_fn in agent_configs:
            if not condition_fn(self.state):
                log.info("Skipping %s — run condition not met.", AgentClass.__name__)
                continue

            agent  = AgentClass(state=self.state, dry_run=self.dry_run)
            result = agent.execute()
            results[agent.name] = result

            if not result.success:
                log.error("Agent %s failed: %s — halting pipeline.",
                          agent.name, result.errors)
                break

        log.info("═" * 60)
        log.info("Pipeline complete. Agents run: %d", len(results))
        self._summarise(results)
        log.info("═" * 60)
        return results

    # ------------------------------------------------------------------
    # State inspection
    # ------------------------------------------------------------------

    def status(self) -> dict:
        s = self.state.get_all()
        return {
            "corpus_version":        s.get("variant_corpus_version"),
            "corpus_last_updated":   s.get("corpus_last_updated"),
            "model_checkpoint":      s.get("model_checkpoint_ref"),
            "drift_score":           s.get("drift_score"),
            "reclassification_rate": s.get("reclassification_rate"),
            "pending_reviews":       len(self.state.unresolved_review_items()),
            "last_10_runs":          (s.get("agent_run_log") or [])[-10:],
        }

    def pending_reviews(self) -> list[dict]:
        return self.state.unresolved_review_items()

    def feature_candidates(self, unreviewed_only: bool = True) -> list[dict]:
        candidates = self.state.get("feature_candidates") or []
        if unreviewed_only:
            return [c for c in candidates if not c.get("reviewed")]
        return candidates

    def mark_candidate_reviewed(self, index: int) -> None:
        from datetime import datetime, timezone
        candidates = self.state.get("feature_candidates") or []
        if 0 <= index < len(candidates):
            candidates[index]["reviewed"]    = True
            candidates[index]["reviewed_at"] = datetime.now(timezone.utc).isoformat()
            self.state.set("feature_candidates", candidates)
            log.info("Candidate %d marked reviewed: %s",
                     index, candidates[index].get("feature_name"))

    def resolve_review(self, index: int) -> None:
        self.state.resolve_pending_review(index)
        log.info("Review item %d resolved.", index)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_pending_reviews(self) -> None:
        unresolved = self.state.unresolved_review_items()
        if not unresolved:
            return
        log.warning("⚠  %d unresolved review item(s):", len(unresolved))
        for i, item in enumerate(unresolved):
            log.warning("  [%d] %s — %s", i, item.get("reason"), item.get("added_at"))

        if REQUIRE_HUMAN_APPROVAL and sys.stdin.isatty():
            print(f"\n⚠  {len(unresolved)} unresolved review item(s) exist.\n"
                  "   Proceed anyway? [y/N]: ", end="")
            if input().strip().lower() != "y":
                log.info("Pipeline aborted (unresolved reviews).")
                sys.exit(0)

    def _summarise(self, results: dict[str, AgentResult]) -> None:
        for name, result in results.items():
            icon = "✓" if result.success else "✗"
            log.info("  %s %-30s  action=%-20s", icon, name, result.action)
            if result.errors:
                for err in result.errors:
                    log.warning("      error: %s", err)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _configure_root_logger() -> None:
    AUDIT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    if not root.handlers:
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ",
        )
        fh = logging.FileHandler(AUDIT_LOG_DIR / "orchestrator.log", encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        root.addHandler(ch)
