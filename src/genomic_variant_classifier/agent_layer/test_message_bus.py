#!/usr/bin/env python3
"""
test_message_bus.py — Message Bus Integration Test Suite
=========================================================
Validates the full inter-agent communication layer without requiring
any external services (no ClinVar FTP, no gnomAD, no SHAP, no Spark).

All tests run entirely in memory using a temporary SharedState file.

Run from agent_layer/:
    python test_message_bus.py

Or with verbose output:
    python test_message_bus.py -v

Test groups
-----------
  Group 1  MessageBus core — send, receive, read, approve, reject, history
  Group 2  SharedState migration — agent_messages key added to existing state
  Group 3  BaseAgent helpers — send_message, get_actionable, mark_message_read
  Group 4  DataFreshnessAgent — emits DATA_UPDATED on change detection
  Group 5  TrainingLifecycleAgent — receives DATA_UPDATED, emits CHECKPOINT_READY
  Group 6  InterpretabilityAgent — receives CHECKPOINT_READY, emits FEATURE_INSTABILITY
  Group 7  LiteratureScoutAgent — emits FEATURE_CANDIDATE_ADDED per new candidate
  Group 8  Orchestrator — pre/post-run message logging, approve/reject delegates
  Group 9  Full pipeline message flow — end-to-end signal chain
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Path setup — allow running from project root or agent_layer/
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# ---------------------------------------------------------------------------
# Minimal config stub — prevents ImportError when config.py is absent
# from the test environment
# ---------------------------------------------------------------------------
import types


def _make_config_stub() -> types.ModuleType:
    cfg = types.ModuleType("config")
    cfg.REQUIRE_HUMAN_APPROVAL = False
    cfg.CLINVAR_FTP_HOST = "ftp.ncbi.nlm.nih.gov"
    cfg.CLINVAR_FTP_PATH = "/pub/clinvar/vcf_GRCh38"
    cfg.GNOMAD_FINGERPRINT_URL = "https://example.com/gnomad"
    cfg.LOVD_API_BASE = "https://example.com/lovd"
    cfg.LOVD_GENES_OF_INTEREST = ["BRCA1"]
    cfg.ALPHAMISSENSE_MANIFEST_URL = "https://example.com/alphamissense"
    cfg.SPARK_INGEST_CMD = "echo spark-ingest"
    cfg.CHECKPOINT_DIR = "/tmp/checkpoints"
    cfg.GCS_CHECKPOINT_PREFIX = "gs://bucket/checkpoints"
    cfg.GCP_PROJECT_ID = "test-project"
    cfg.MODEL_RETRAIN_SCRIPT = "echo retrain"
    cfg.SHAP_REPORT_DIR = "/tmp/shap_reports"
    cfg.SHAP_INSTABILITY_THRESHOLD = 0.5
    cfg.EXPECTED_HIGH_IMPORTANCE_FEATURES = ["cadd_score", "af_gnomad"]
    cfg.VALIDATION_PARQUET_PATH = "/tmp/validation.parquet"
    cfg.INTERPRETABILITY_INTERVAL_DAYS = 7
    cfg.LITERATURE_INTERVAL_DAYS = 7
    cfg.LITERATURE_KEYWORDS = ["variant pathogenicity", "BRCA1"]
    cfg.LITERATURE_KNOWN_TOOLS = ["CADD", "SIFT", "PolyPhen"]
    cfg.LITERATURE_MAX_RESULTS = 10
    cfg.LITERATURE_RELEVANCE_THRESHOLD = 0.1
    cfg.NCBI_API_KEY = ""
    cfg.REPORT_DIR = "/tmp/reports"
    return cfg


if "config" not in sys.modules:
    sys.modules["config"] = _make_config_stub()

# Stub heavy optional dependencies so tests don't require them installed
for _mod in ("ewc_utils", "shap", "torch", "feedparser", "requests"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# Now safe to import project modules
from message_bus import (
    MessageBus,
    DATA_UPDATED,
    CHECKPOINT_READY,
    FEATURE_INSTABILITY,
    FEATURE_CANDIDATE_ADDED,
    PRIORITY_HIGH,
    PRIORITY_NORMAL,
    PRIORITY_LOW,
    ALL_SUBJECTS,
    APPROVAL_REQUIRED_SUBJECTS,
)
from shared_state import SharedState


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class _TempState:
    """Context manager: creates a SharedState backed by a temp file."""

    def __enter__(self) -> SharedState:
        self._fd, self._path = tempfile.mkstemp(suffix=".json")
        os.close(self._fd)
        os.unlink(self._path)  # SharedState creates it on first save
        return SharedState(self._path)

    def __exit__(self, *_):
        try:
            os.unlink(self._path)
        except FileNotFoundError:
            pass


def _make_bus(state: SharedState) -> MessageBus:
    return MessageBus(state)


# ---------------------------------------------------------------------------
# Test results tracker
# ---------------------------------------------------------------------------

_PASS = "PASS"
_FAIL = "FAIL"
_ERROR = "ERROR"
_results: list[tuple[str, str, str]] = []  # (group, name, status)


def _run(group: str, name: str, fn):
    try:
        fn()
        _results.append((group, name, _PASS))
        print(f"  ✓ {name}")
    except AssertionError as exc:
        _results.append((group, name, _FAIL))
        print(f"  ✗ {name}  FAIL: {exc}")
    except Exception as exc:
        _results.append((group, name, _ERROR))
        print(f"  ✗ {name}  ERROR: {exc}")
        if "-v" in sys.argv:
            traceback.print_exc()


# ===========================================================================
# Group 1 — MessageBus core
# ===========================================================================


def _test_send_creates_inbox():
    with _TempState() as state:
        bus = _make_bus(state)
        bus.send("AgentA", "AgentB", DATA_UPDATED, {"source": "gnomAD"})
        inbox = bus.get_inbox("AgentB")
        assert len(inbox) == 1
        assert inbox[0]["subject"] == DATA_UPDATED
        assert inbox[0]["from_agent"] == "AgentA"
        assert inbox[0]["payload"]["source"] == "gnomAD"


def _test_send_returns_uuid():
    with _TempState() as state:
        bus = _make_bus(state)
        msg_id = bus.send("A", "B", CHECKPOINT_READY)
        assert isinstance(msg_id, str) and len(msg_id) == 36


def _test_unread_filter():
    with _TempState() as state:
        bus = _make_bus(state)
        id1 = bus.send("A", "B", DATA_UPDATED)
        id2 = bus.send("A", "B", CHECKPOINT_READY)
        bus.mark_read("B", id1)
        unread = bus.get_unread("B")
        assert len(unread) == 1
        assert unread[0]["id"] == id2


def _test_get_actionable_respects_approval():
    with _TempState() as state:
        bus = _make_bus(state)
        # DATA_UPDATED requires approval by default
        id1 = bus.send("A", "B", DATA_UPDATED)
        # FEATURE_CANDIDATE_ADDED does not
        id2 = bus.send("A", "B", FEATURE_CANDIDATE_ADDED, requires_approval=False)
        actionable = bus.get_actionable("B")
        ids = [m["id"] for m in actionable]
        assert id2 in ids, "Non-approval message should be actionable"
        assert id1 not in ids, "Unapproved message should NOT be actionable"


def _test_approve_makes_actionable():
    with _TempState() as state:
        bus = _make_bus(state)
        msg_id = bus.send("A", "B", DATA_UPDATED)
        assert bus.get_actionable("B") == []
        bus.approve(msg_id)
        actionable = bus.get_actionable("B")
        assert len(actionable) == 1
        assert actionable[0]["id"] == msg_id


def _test_reject_stays_non_actionable():
    with _TempState() as state:
        bus = _make_bus(state)
        msg_id = bus.send("A", "B", DATA_UPDATED)
        bus.reject(msg_id)
        assert bus.get_actionable("B") == []
        # Confirm approved=False
        inbox = bus.get_inbox("B")
        assert inbox[0]["approved"] is False


def _test_pending_approval_list():
    with _TempState() as state:
        bus = _make_bus(state)
        id1 = bus.send("A", "B", DATA_UPDATED)
        id2 = bus.send("A", "C", CHECKPOINT_READY)
        bus.approve(id1)
        pending = bus.pending_approval()
        pending_ids = [m["id"] for m in pending]
        assert id1 not in pending_ids
        assert id2 in pending_ids


def _test_history_ordering():
    with _TempState() as state:
        bus = _make_bus(state)
        bus.send("A", "B", DATA_UPDATED)
        bus.send("A", "B", CHECKPOINT_READY)
        bus.send("A", "B", FEATURE_INSTABILITY, requires_approval=False)
        history = bus.history("B", limit=10)
        # Most recent first
        assert history[0]["subject"] == FEATURE_INSTABILITY
        assert history[2]["subject"] == DATA_UPDATED


def _test_agent_list():
    with _TempState() as state:
        bus = _make_bus(state)
        bus.send("A", "AgentX", DATA_UPDATED)
        bus.send("A", "AgentY", CHECKPOINT_READY)
        agents = bus.agent_list()
        assert "AgentX" in agents
        assert "AgentY" in agents


def _test_invalid_subject_raises():
    with _TempState() as state:
        bus = _make_bus(state)
        try:
            bus.send("A", "B", "MADE_UP_SUBJECT")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


def _test_mark_all_read():
    with _TempState() as state:
        bus = _make_bus(state)
        bus.send("A", "B", DATA_UPDATED)
        bus.send("A", "B", CHECKPOINT_READY)
        count = bus.mark_all_read("B")
        assert count == 2
        assert bus.get_unread("B") == []


# ===========================================================================
# Group 2 — SharedState migration
# ===========================================================================


def _test_migration_adds_agent_messages():
    """Existing state file without agent_messages gets it added silently."""
    with _TempState() as state:
        # Write old-style state without agent_messages
        old_state = {
            "data_freshness": {},
            "training": {},
            "interpretability": {},
            "literature": {},
            "review_items": [],
        }
        state.save(old_state)
        # Load triggers migration
        loaded = state.load()
        assert "agent_messages" in loaded, "Migration should add agent_messages"
        assert isinstance(loaded["agent_messages"], dict)


def _test_default_state_has_agent_messages():
    with _TempState() as state:
        # Fresh state (file doesn't exist yet)
        loaded = state.load()
        assert "agent_messages" in loaded
        assert loaded["agent_messages"] == {}


def _test_save_is_atomic(tmp_path=None):
    """Saving should not corrupt existing data."""
    with _TempState() as state:
        original = state.load()
        original["training"]["last_run"] = "2026-01-01T00:00:00+00:00"
        state.save(original)
        reloaded = state.load()
        assert reloaded["training"]["last_run"] == "2026-01-01T00:00:00+00:00"
        assert "agent_messages" in reloaded


def _test_summary_includes_message_counts():
    with _TempState() as state:
        bus = _make_bus(state)
        bus.send("A", "B", DATA_UPDATED)  # requires approval → pending
        bus.send(
            "A", "C", FEATURE_CANDIDATE_ADDED, requires_approval=False
        )  # unread but no approval needed
        summary = state.summary()
        assert "messages_unread=2" in summary
        assert "messages_pending_approval=1" in summary


# ===========================================================================
# Group 3 — BaseAgent helpers
# ===========================================================================


def _make_test_agent(state: SharedState):
    """Create a minimal concrete BaseAgent for testing."""
    from agents.base_agent import BaseAgent

    class _TestAgent(BaseAgent):
        def run(self, dry_run=False):
            return {"action": "test"}

    return _TestAgent(state)


def _test_agent_send_message():
    with _TempState() as state:
        agent = _make_test_agent(state)
        msg_id = agent.send_message(
            to="TrainingLifecycleAgent",
            subject=DATA_UPDATED,
            payload={"source": "ClinVar"},
        )
        assert isinstance(msg_id, str) and len(msg_id) == 36
        # Check it landed in the right inbox
        bus = _make_bus(state)
        inbox = bus.get_inbox("TrainingLifecycleAgent")
        assert len(inbox) == 1
        assert inbox[0]["from_agent"] == "_TestAgent"


def _test_agent_get_actionable():
    with _TempState() as state:
        bus = _make_bus(state)
        # Send a non-approval message to our test agent
        bus.send(
            "Other", "_TestAgent", FEATURE_CANDIDATE_ADDED, requires_approval=False
        )
        agent = _make_test_agent(state)
        actionable = agent.get_actionable()
        assert len(actionable) == 1
        assert actionable[0]["subject"] == FEATURE_CANDIDATE_ADDED


def _test_agent_mark_message_read():
    with _TempState() as state:
        bus = _make_bus(state)
        msg_id = bus.send(
            "Other", "_TestAgent", FEATURE_CANDIDATE_ADDED, requires_approval=False
        )
        agent = _make_test_agent(state)
        agent.mark_message_read(msg_id)
        assert bus.get_unread("_TestAgent") == []


# ===========================================================================
# Group 4 — DataFreshnessAgent emits DATA_UPDATED
# ===========================================================================


def _test_data_freshness_emits_data_updated():
    """
    Simulate a gnomAD fingerprint change and verify DATA_UPDATED is sent
    to TrainingLifecycleAgent.
    """
    with _TempState() as state:
        # Pre-seed a known fingerprint so a change is detected
        state.save(
            {
                **state.load(),
                "data_freshness": {
                    "gnomad": {"last_seen": "old-etag", "last_checked": None},
                    "clinvar": {"last_seen": None, "last_checked": None},
                    "lovd": {"last_seen": None, "last_checked": None},
                    "alphamissense": {"last_seen": None, "last_checked": None},
                },
            }
        )

        from agents.data_freshness_agent import DataFreshnessAgent

        agent = DataFreshnessAgent(state)

        # Patch all network calls; gnomAD returns a different ETag
        with patch("agents.data_freshness_agent.requests") as mock_req, patch(
            "agents.data_freshness_agent.ftplib"
        ) as mock_ftp:

            # gnomAD HEAD → new ETag
            gnomad_resp = MagicMock()
            gnomad_resp.headers = {"ETag": "new-etag"}
            # LOVD → 402 (skip)
            lovd_resp = MagicMock()
            lovd_resp.status_code = 402
            # AlphaMissense HEAD → same ETag (no change)
            am_resp = MagicMock()
            am_resp.headers = {}

            mock_req.head.side_effect = [gnomad_resp, am_resp]
            mock_req.get.return_value = lovd_resp

            # ClinVar FTP → no files
            mock_ftp_inst = MagicMock()
            mock_ftp_inst.__enter__ = lambda s: mock_ftp_inst
            mock_ftp_inst.__exit__ = MagicMock(return_value=False)
            mock_ftp_inst.nlst.return_value = []
            mock_ftp.FTP.return_value = mock_ftp_inst

            result = agent.run(dry_run=False)

        # Verify result
        assert result["changes_detected"] >= 1

        # Verify DATA_UPDATED landed in TrainingLifecycleAgent's inbox
        bus = _make_bus(state)
        inbox = bus.get_inbox("TrainingLifecycleAgent")
        data_updated = [m for m in inbox if m["subject"] == DATA_UPDATED]
        assert len(data_updated) >= 1
        payload = data_updated[0]["payload"]
        assert payload["source"] == "gnomAD"
        assert "ingest_approved" in payload


def _test_data_freshness_dry_run_no_message():
    """dry_run=True must not send any real messages."""
    with _TempState() as state:
        from agents.data_freshness_agent import DataFreshnessAgent

        agent = DataFreshnessAgent(state)

        with patch("agents.data_freshness_agent.requests") as mock_req, patch(
            "agents.data_freshness_agent.ftplib"
        ) as mock_ftp:

            resp = MagicMock()
            resp.headers = {"ETag": "changed-etag"}
            mock_req.head.return_value = resp
            lovd_resp = MagicMock()
            lovd_resp.status_code = 402
            mock_req.get.return_value = lovd_resp

            mock_ftp_inst = MagicMock()
            mock_ftp_inst.__enter__ = lambda s: mock_ftp_inst
            mock_ftp_inst.__exit__ = MagicMock(return_value=False)
            mock_ftp_inst.nlst.return_value = []
            mock_ftp.FTP.return_value = mock_ftp_inst

            agent.run(dry_run=True)

        bus = _make_bus(state)
        inbox = bus.get_inbox("TrainingLifecycleAgent")
        assert inbox == [], "dry_run must not send messages"


# ===========================================================================
# Group 5 — TrainingLifecycleAgent receives DATA_UPDATED, emits CHECKPOINT_READY
# ===========================================================================


def _test_training_processes_data_updated():
    """
    Place an approved DATA_UPDATED in inbox; verify retrain is flagged
    and CHECKPOINT_READY is sent to InterpretabilityAgent.
    """
    with _TempState() as state:
        bus = _make_bus(state)
        msg_id = bus.send(
            "DataFreshnessAgent",
            "TrainingLifecycleAgent",
            DATA_UPDATED,
            {
                "source": "gnomAD",
                "ingest_approved": True,
                "change_type": "fingerprint_changed",
            },
            requires_approval=True,
        )
        bus.approve(msg_id)

        from agents.training_lifecycle_agent import TrainingLifecycleAgent

        agent = TrainingLifecycleAgent(state)

        with patch.object(agent, "_check_drift", return_value=False), patch.object(
            agent, "_run_training", return_value="/tmp/checkpoints/model.pt"
        ), patch.object(agent, "_require_approval", return_value=True):

            result = agent.run(dry_run=False)

        assert result["retrain_triggered"] is True
        assert result["inbox_processed"] == 1

        # CHECKPOINT_READY should now be in InterpretabilityAgent's inbox
        inbox = bus.get_inbox("InterpretabilityAgent")
        cp_msgs = [m for m in inbox if m["subject"] == CHECKPOINT_READY]
        assert len(cp_msgs) == 1
        assert cp_msgs[0]["payload"]["checkpoint_path"] == "/tmp/checkpoints/model.pt"
        assert cp_msgs[0]["payload"]["data_sources"] == ["gnomAD"]


def _test_training_defers_on_unapproved_data_updated():
    """Unapproved DATA_UPDATED must not trigger retraining."""
    with _TempState() as state:
        bus = _make_bus(state)
        # Send but do NOT approve
        bus.send(
            "DataFreshnessAgent",
            "TrainingLifecycleAgent",
            DATA_UPDATED,
            {
                "source": "ClinVar",
                "ingest_approved": False,
                "change_type": "new_release",
            },
        )

        from agents.training_lifecycle_agent import TrainingLifecycleAgent

        agent = TrainingLifecycleAgent(state)

        with patch.object(agent, "_check_drift", return_value=False):
            result = agent.run(dry_run=False)

        assert result["retrain_triggered"] is False
        # No CHECKPOINT_READY should exist
        inbox = bus.get_inbox("InterpretabilityAgent")
        assert inbox == []


def _test_training_stores_instability_flags():
    """FEATURE_INSTABILITY messages are persisted into SharedState."""
    with _TempState() as state:
        bus = _make_bus(state)
        bus.send(
            "InterpretabilityAgent",
            "TrainingLifecycleAgent",
            FEATURE_INSTABILITY,
            {
                "flagged_features": ["weird_score", "bad_feature"],
                "severity": "medium",
                "reason": "High CV on weird_score",
            },
            requires_approval=False,
        )

        from agents.training_lifecycle_agent import TrainingLifecycleAgent

        agent = TrainingLifecycleAgent(state)

        with patch.object(agent, "_check_drift", return_value=False):
            result = agent.run(dry_run=False)

        loaded = state.load()
        flags = loaded.get("training", {}).get("instability_flags", [])
        flag_names = [f["feature"] for f in flags]
        assert "weird_score" in flag_names
        assert "bad_feature" in flag_names


def _test_training_stores_feature_candidates():
    """FEATURE_CANDIDATE_ADDED messages are persisted into SharedState."""
    with _TempState() as state:
        bus = _make_bus(state)
        bus.send(
            "LiteratureScoutAgent",
            "TrainingLifecycleAgent",
            FEATURE_CANDIDATE_ADDED,
            {
                "candidate_name": "SplicePAS score",
                "literature_source": "PubMed",
                "pmid_or_doi": "12345678",
                "paper_title": "Novel splice site predictor",
                "relevance_score": 0.82,
            },
            requires_approval=False,
        )

        from agents.training_lifecycle_agent import TrainingLifecycleAgent

        agent = TrainingLifecycleAgent(state)

        with patch.object(agent, "_check_drift", return_value=False):
            agent.run(dry_run=False)

        loaded = state.load()
        candidates = loaded.get("training", {}).get("pending_feature_candidates", [])
        names = [c["name"] for c in candidates]
        assert "SplicePAS score" in names


# ===========================================================================
# Group 6 — InterpretabilityAgent receives CHECKPOINT_READY, emits FEATURE_INSTABILITY
# ===========================================================================


def _test_interpretability_processes_checkpoint_ready():
    """
    Place an approved CHECKPOINT_READY in inbox; verify SHAP runs against
    that checkpoint and FEATURE_INSTABILITY is sent back on flagged features.
    """
    with _TempState() as state:
        bus = _make_bus(state)
        msg_id = bus.send(
            "TrainingLifecycleAgent",
            "InterpretabilityAgent",
            CHECKPOINT_READY,
            {
                "checkpoint_path": "/tmp/checkpoints/model_v2.pt",
                "trigger_reason": "data_updated",
                "trained_at": "2026-04-09T12:00:00+00:00",
                "ewc_applied": True,
                "data_sources": ["gnomAD"],
            },
        )
        bus.approve(msg_id)

        from agents.interpretability_agent import InterpretabilityAgent

        agent = InterpretabilityAgent(state)

        # Patch the audit to return two flagged features
        with patch.object(
            agent,
            "_run_shap_audit",
            return_value=(
                ["weird_score", "bad_feature"],
                "/tmp/shap_reports/report.html",
                "high",
            ),
        ):
            result = agent.run(dry_run=False)

        assert result["flagged_count"] == 2
        assert result["severity"] == "high"
        assert result["instability_sent"] is True

        # FEATURE_INSTABILITY should be in TrainingLifecycleAgent's inbox
        inbox = bus.get_inbox("TrainingLifecycleAgent")
        instability = [m for m in inbox if m["subject"] == FEATURE_INSTABILITY]
        assert len(instability) == 1
        payload = instability[0]["payload"]
        assert "weird_score" in payload["flagged_features"]
        assert payload["severity"] == "high"
        assert instability[0].get("requires_approval") is False


def _test_interpretability_skips_duplicate_checkpoint():
    """If the checkpoint was already audited, skip — don't emit duplicate."""
    with _TempState() as state:
        # Mark checkpoint as already audited
        state.update_section(
            "interpretability",
            {"last_checkpoint_audited": "/tmp/checkpoints/model_v2.pt"},
        )

        bus = _make_bus(state)
        msg_id = bus.send(
            "TrainingLifecycleAgent",
            "InterpretabilityAgent",
            CHECKPOINT_READY,
            {
                "checkpoint_path": "/tmp/checkpoints/model_v2.pt",
                "trigger_reason": "data_updated",
                "trained_at": "2026-04-09T12:00:00+00:00",
                "ewc_applied": True,
                "data_sources": [],
            },
        )
        bus.approve(msg_id)

        from agents.interpretability_agent import InterpretabilityAgent

        agent = InterpretabilityAgent(state)
        result = agent.run(dry_run=False)

        assert result["action"] == "skipped"
        assert bus.get_inbox("TrainingLifecycleAgent") == []


def _test_interpretability_no_flag_no_message():
    """Clean SHAP audit must not emit FEATURE_INSTABILITY."""
    with _TempState() as state:
        bus = _make_bus(state)
        msg_id = bus.send(
            "TrainingLifecycleAgent",
            "InterpretabilityAgent",
            CHECKPOINT_READY,
            {
                "checkpoint_path": "/tmp/checkpoints/clean.pt",
                "trigger_reason": "scheduled",
                "trained_at": "2026-04-09T00:00:00+00:00",
                "ewc_applied": False,
                "data_sources": [],
            },
        )
        bus.approve(msg_id)

        from agents.interpretability_agent import InterpretabilityAgent

        agent = InterpretabilityAgent(state)

        with patch.object(agent, "_run_shap_audit", return_value=([], None, "low")):
            result = agent.run(dry_run=False)

        assert result["flagged_count"] == 0
        assert result["instability_sent"] is False
        assert bus.get_inbox("TrainingLifecycleAgent") == []


# ===========================================================================
# Group 7 — LiteratureScoutAgent emits FEATURE_CANDIDATE_ADDED
# ===========================================================================


def _test_literature_emits_candidate_per_new_entry():
    with _TempState() as state:
        from agents.literature_scout_agent import LiteratureScoutAgent

        agent = LiteratureScoutAgent(state)

        fake_papers = [
            {
                "source": "PubMed",
                "pmid": "99999",
                "title": "Novel SplicePAS score predicts pathogenicity",
                "abstract": "We propose a novel score called SplicePAS score for variant pathogenicity",
                "url": "https://pubmed.ncbi.nlm.nih.gov/99999/",
            }
        ]

        with patch.object(
            agent, "_fetch_pubmed", return_value=fake_papers
        ), patch.object(agent, "_fetch_biorxiv", return_value=[]), patch.object(
            agent, "_fetch_clingen", return_value=[]
        ), patch.object(
            agent, "_render_digest", return_value=None
        ), patch.object(
            agent, "_should_run_literature", return_value=True
        ):

            result = agent.run(dry_run=False)

        assert result["new_candidates"] >= 1
        assert result["messages_sent"] >= 1

        bus = _make_bus(state)
        inbox = bus.get_inbox("TrainingLifecycleAgent")
        cand_msgs = [m for m in inbox if m["subject"] == FEATURE_CANDIDATE_ADDED]
        assert len(cand_msgs) >= 1
        payload = cand_msgs[0]["payload"]
        assert "candidate_name" in payload
        assert payload["literature_source"] == "PubMed"
        assert (
            payload.get("requires_approval") is None
            or cand_msgs[0].get("requires_approval") is False
        )


def _test_literature_dry_run_no_message():
    with _TempState() as state:
        from agents.literature_scout_agent import LiteratureScoutAgent

        agent = LiteratureScoutAgent(state)

        fake_papers = [
            {
                "source": "PubMed",
                "pmid": "77777",
                "title": "Novel ConScore score predicts pathogenicity",
                "abstract": "We propose a novel score called ConScore for variant pathogenicity",
                "url": "https://pubmed.ncbi.nlm.nih.gov/77777/",
            }
        ]

        with patch.object(
            agent, "_fetch_pubmed", return_value=fake_papers
        ), patch.object(agent, "_fetch_biorxiv", return_value=[]), patch.object(
            agent, "_fetch_clingen", return_value=[]
        ), patch.object(
            agent, "_render_digest", return_value=None
        ), patch.object(
            agent, "_should_run_literature", return_value=True
        ):

            result = agent.run(dry_run=True)

        assert result["messages_sent"] == 0
        bus = _make_bus(state)
        assert bus.get_inbox("TrainingLifecycleAgent") == []


# ===========================================================================
# Group 8 — Orchestrator delegates
# ===========================================================================


def _test_orchestrator_approve_message():
    with _TempState() as state:
        bus = _make_bus(state)
        msg_id = bus.send("A", "B", DATA_UPDATED)

        from orchestrator import Orchestrator

        orch = Orchestrator(state, dry_run=True)
        orch.approve_message(msg_id)

        inbox = bus.get_inbox("B")
        assert inbox[0]["approved"] is True


def _test_orchestrator_reject_message():
    with _TempState() as state:
        bus = _make_bus(state)
        msg_id = bus.send("A", "B", CHECKPOINT_READY)

        from orchestrator import Orchestrator

        orch = Orchestrator(state, dry_run=True)
        orch.reject_message(msg_id)

        inbox = bus.get_inbox("B")
        assert inbox[0]["approved"] is False


def _test_orchestrator_print_inbox_runs(capsys=None):
    """Smoke test: print_inbox must not raise for a known agent name."""
    with _TempState() as state:
        bus = _make_bus(state)
        bus.send(
            "X", "DataFreshnessAgent", FEATURE_CANDIDATE_ADDED, requires_approval=False
        )

        from orchestrator import Orchestrator

        orch = Orchestrator(state, dry_run=True)
        try:
            orch.print_inbox("DataFreshnessAgent")
        except Exception as exc:
            assert False, f"print_inbox raised: {exc}"


def _test_orchestrator_status_includes_messages():
    with _TempState() as state:
        bus = _make_bus(state)
        bus.send("A", "B", DATA_UPDATED)  # 1 unread, 1 pending approval

        from orchestrator import Orchestrator

        orch = Orchestrator(state, dry_run=True)
        summary = state.summary()
        assert "messages_unread=1" in summary
        assert "messages_pending_approval=1" in summary


# ===========================================================================
# Group 9 — Full pipeline message flow (end-to-end chain)
# ===========================================================================


def _test_full_signal_chain():
    """
    Simulate the complete message flow:
      DataFreshness → DATA_UPDATED → TrainingLifecycle
        → CHECKPOINT_READY → Interpretability
          → FEATURE_INSTABILITY → TrainingLifecycle
      Literature → FEATURE_CANDIDATE_ADDED → TrainingLifecycle
    """
    with _TempState() as state:
        bus = _make_bus(state)

        # Step 1: DataFreshness detects a change and sends DATA_UPDATED
        bus.send(
            "DataFreshnessAgent",
            "TrainingLifecycleAgent",
            DATA_UPDATED,
            {
                "source": "gnomAD",
                "ingest_approved": True,
                "change_type": "fingerprint_changed",
            },
        )
        assert len(bus.get_inbox("TrainingLifecycleAgent")) == 1

        # Step 2: Human approves DATA_UPDATED
        data_msg_id = bus.get_inbox("TrainingLifecycleAgent")[0]["id"]
        bus.approve(data_msg_id)

        # Step 3: TrainingLifecycle processes it → emits CHECKPOINT_READY
        from agents.training_lifecycle_agent import TrainingLifecycleAgent

        train_agent = TrainingLifecycleAgent(state)

        with patch.object(
            train_agent, "_check_drift", return_value=False
        ), patch.object(
            train_agent, "_run_training", return_value="/tmp/checkpoints/chain_test.pt"
        ), patch.object(
            train_agent, "_require_approval", return_value=True
        ):
            train_agent.run(dry_run=False)

        interp_inbox = bus.get_inbox("InterpretabilityAgent")
        assert len(interp_inbox) == 1
        assert interp_inbox[0]["subject"] == CHECKPOINT_READY

        # Step 4: Human approves CHECKPOINT_READY
        cp_msg_id = interp_inbox[0]["id"]
        bus.approve(cp_msg_id)

        # Step 5: Interpretability runs → finds instability → sends FEATURE_INSTABILITY
        from agents.interpretability_agent import InterpretabilityAgent

        interp_agent = InterpretabilityAgent(state)

        with patch.object(
            interp_agent,
            "_run_shap_audit",
            return_value=(["unstable_feature"], "/tmp/report.html", "high"),
        ):
            interp_agent.run(dry_run=False)

        # TrainingLifecycle inbox should now have FEATURE_INSTABILITY
        train_inbox_after = bus.get_inbox("TrainingLifecycleAgent")
        subjects = [m["subject"] for m in train_inbox_after]
        assert FEATURE_INSTABILITY in subjects

        # Step 6: LiteratureScout finds a candidate → sends FEATURE_CANDIDATE_ADDED
        from agents.literature_scout_agent import LiteratureScoutAgent

        lit_agent = LiteratureScoutAgent(state)

        fake_papers = [
            {
                "source": "PubMed",
                "pmid": "11223344",
                "title": "Novel ChainTestScore predicts variant pathogenicity",
                "abstract": "We propose a novel score called ChainTestScore "
                "for variant pathogenicity prediction.",
                "url": "https://pubmed.ncbi.nlm.nih.gov/11223344/",
            }
        ]

        with patch.object(
            lit_agent, "_fetch_pubmed", return_value=fake_papers
        ), patch.object(lit_agent, "_fetch_biorxiv", return_value=[]), patch.object(
            lit_agent, "_fetch_clingen", return_value=[]
        ), patch.object(
            lit_agent, "_render_digest", return_value=None
        ), patch.object(
            lit_agent, "_should_run_literature", return_value=True
        ):
            lit_agent.run(dry_run=False)

        final_train_inbox = bus.get_inbox("TrainingLifecycleAgent")
        final_subjects = [m["subject"] for m in final_train_inbox]
        assert FEATURE_CANDIDATE_ADDED in final_subjects

        # Final state: TrainingLifecycle inbox has both FEATURE_INSTABILITY
        # and FEATURE_CANDIDATE_ADDED waiting for the next pipeline run
        assert FEATURE_INSTABILITY in final_subjects
        assert FEATURE_CANDIDATE_ADDED in final_subjects


# ===========================================================================
# Runner
# ===========================================================================

TESTS = [
    # Group 1 — MessageBus core
    ("1 MessageBus core", "send creates inbox", _test_send_creates_inbox),
    ("1 MessageBus core", "send returns UUID", _test_send_returns_uuid),
    ("1 MessageBus core", "unread filter", _test_unread_filter),
    (
        "1 MessageBus core",
        "get_actionable respects approval",
        _test_get_actionable_respects_approval,
    ),
    ("1 MessageBus core", "approve makes actionable", _test_approve_makes_actionable),
    (
        "1 MessageBus core",
        "reject stays non-actionable",
        _test_reject_stays_non_actionable,
    ),
    ("1 MessageBus core", "pending_approval list", _test_pending_approval_list),
    ("1 MessageBus core", "history ordering", _test_history_ordering),
    ("1 MessageBus core", "agent_list", _test_agent_list),
    ("1 MessageBus core", "invalid subject raises", _test_invalid_subject_raises),
    ("1 MessageBus core", "mark_all_read", _test_mark_all_read),
    # Group 2 — SharedState migration
    (
        "2 SharedState",
        "migration adds agent_messages",
        _test_migration_adds_agent_messages,
    ),
    (
        "2 SharedState",
        "default state has agent_messages",
        _test_default_state_has_agent_messages,
    ),
    ("2 SharedState", "save is atomic", _test_save_is_atomic),
    (
        "2 SharedState",
        "summary includes message counts",
        _test_summary_includes_message_counts,
    ),
    # Group 3 — BaseAgent helpers
    ("3 BaseAgent", "send_message", _test_agent_send_message),
    ("3 BaseAgent", "get_actionable", _test_agent_get_actionable),
    ("3 BaseAgent", "mark_message_read", _test_agent_mark_message_read),
    # Group 4 — DataFreshnessAgent
    (
        "4 DataFreshness",
        "emits DATA_UPDATED on change",
        _test_data_freshness_emits_data_updated,
    ),
    (
        "4 DataFreshness",
        "dry_run sends no message",
        _test_data_freshness_dry_run_no_message,
    ),
    # Group 5 — TrainingLifecycleAgent
    (
        "5 Training",
        "processes DATA_UPDATED approved",
        _test_training_processes_data_updated,
    ),
    (
        "5 Training",
        "defers on unapproved DATA_UPDATED",
        _test_training_defers_on_unapproved_data_updated,
    ),
    ("5 Training", "stores instability flags", _test_training_stores_instability_flags),
    (
        "5 Training",
        "stores feature candidates",
        _test_training_stores_feature_candidates,
    ),
    # Group 6 — InterpretabilityAgent
    (
        "6 Interpretability",
        "processes CHECKPOINT_READY",
        _test_interpretability_processes_checkpoint_ready,
    ),
    (
        "6 Interpretability",
        "skips duplicate checkpoint",
        _test_interpretability_skips_duplicate_checkpoint,
    ),
    (
        "6 Interpretability",
        "no flag → no message",
        _test_interpretability_no_flag_no_message,
    ),
    # Group 7 — LiteratureScoutAgent
    (
        "7 Literature",
        "emits FEATURE_CANDIDATE_ADDED",
        _test_literature_emits_candidate_per_new_entry,
    ),
    ("7 Literature", "dry_run sends no message", _test_literature_dry_run_no_message),
    # Group 8 — Orchestrator
    ("8 Orchestrator", "approve_message delegate", _test_orchestrator_approve_message),
    ("8 Orchestrator", "reject_message delegate", _test_orchestrator_reject_message),
    ("8 Orchestrator", "print_inbox smoke test", _test_orchestrator_print_inbox_runs),
    (
        "8 Orchestrator",
        "status includes message counts",
        _test_orchestrator_status_includes_messages,
    ),
    # Group 9 — Full chain
    ("9 Full chain", "end-to-end signal flow", _test_full_signal_chain),
]


def main():
    verbose = "-v" in sys.argv
    current_group = None

    for group, name, fn in TESTS:
        if group != current_group:
            current_group = group
            print(f"\n── {group} ──")
        _run(group, name, fn)

    # Summary
    total = len(_results)
    passed = sum(1 for _, _, s in _results if s == _PASS)
    failed = sum(1 for _, _, s in _results if s == _FAIL)
    errors = sum(1 for _, _, s in _results if s == _ERROR)

    print(f"\n{'═'*55}")
    print(f"  Results: {passed}/{total} passed  " f"({failed} failed, {errors} errors)")
    print(f"{'═'*55}")

    if failed > 0 or errors > 0:
        print("\nFailed / errored tests:")
        for group, name, status in _results:
            if status != _PASS:
                print(f"  [{status}]  {group} — {name}")
        sys.exit(1)
    else:
        print("\n  All tests passed. ✓")
        sys.exit(0)


if __name__ == "__main__":
    main()
