"""
training_lifecycle_agent.py — Model Training Lifecycle Manager
==============================================================
Manages the EWC continual learning loop: decides when to retrain, applies
EWC penalties, checkpoints the model, and coordinates with peer agents via
the MessageBus.

Messages consumed (inbox)
--------------------------
  DATA_UPDATED (from DataFreshnessAgent)
      New genomic data source release detected. If ingest_approved=True the
      data is already in the pipeline and retraining should be considered.
      If ingest_approved=False the ingest is still pending — this agent logs
      the signal and defers the retrain decision.

  FEATURE_INSTABILITY (from InterpretabilityAgent)
      SHAP audit found unstable or counterintuitive feature importances.
      Payload includes the flagged features. This agent logs them into the
      training section of SharedState and factors them into the next EWC run.

  FEATURE_CANDIDATE_ADDED (from LiteratureScoutAgent)
      A new feature candidate has been surfaced from literature and added to
      the queue. This agent logs it for inclusion in the next feature
      engineering review.

Message emitted (outbox)
------------------------
  CHECKPOINT_READY (to InterpretabilityAgent)
      Emitted after a successful training run and checkpoint save.
      Payload: {
          "checkpoint_path": "<path>",
          "trigger_reason":  "<str>",
          "trained_at":      "<iso>",
          "ewc_applied":     true | false,
          "data_sources":    ["gnomAD", ...]   // from triggering DATA_UPDATED
      }
      Priority          : HIGH
      Requires approval : True (triggers a full SHAP audit)

Processing order inside run()
------------------------------
  1. Collect actionable inbox messages.
  2. Process DATA_UPDATED → may set retrain flag + record data sources.
  3. Process FEATURE_INSTABILITY → log flagged features into SharedState.
  4. Process FEATURE_CANDIDATE_ADDED → log candidates into SharedState.
  5. Mark all processed messages as read.
  6. Run existing EWC / drift-detection logic.
  7. If retrain triggered and approved → train, checkpoint, emit CHECKPOINT_READY.
"""

from __future__ import annotations

import logging
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from agents.base_agent import BaseAgent
from config import (
    CHECKPOINT_DIR,
    DATAPROC_BUCKET,
    DATAPROC_CLUSTER_NAME,
    GCP_PROJECT_ID,
    GCP_REGION,
    GCS_CHECKPOINT_PREFIX,
    REQUIRE_HUMAN_APPROVAL,
)
from message_bus import (
    CHECKPOINT_READY,
    DATA_UPDATED,
    FEATURE_CANDIDATE_ADDED,
    FEATURE_INSTABILITY,
    PRIORITY_HIGH,
    PRIORITY_NORMAL,
)
from shared_state import SharedState

# EWC retraining job — assembled from config dataproc components.
_MODEL_RETRAIN_SCRIPT = (
    f"gcloud dataproc jobs submit pyspark {DATAPROC_BUCKET}/jobs/train_ewc.py "
    f"--cluster={DATAPROC_CLUSTER_NAME} "
    f"--region={GCP_REGION} "
    f"--project={GCP_PROJECT_ID}"
)

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

_INTERPRETABILITY_AGENT = "InterpretabilityAgent"


class TrainingLifecycleAgent(BaseAgent):
    """
    Manages EWC continual learning and coordinates retraining across the
    agent layer.
    """

    def __init__(self, shared_state: SharedState) -> None:
        super().__init__(shared_state)
        self._retrain_flag = False  # set by inbox processing
        self._data_sources: list[str] = []  # which sources triggered retrain
        self._trigger_reason = "scheduled"

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self, dry_run: bool = False) -> dict:
        self._log_start(dry_run)
        self._retrain_flag = False
        self._data_sources = []
        self._trigger_reason = "scheduled"

        # ----------------------------------------------------------
        # Step 1: Process inbox messages from peer agents
        # ----------------------------------------------------------
        processed_ids = self._process_inbox(dry_run)

        # ----------------------------------------------------------
        # Step 2: Drift detection (existing EWC logic)
        # ----------------------------------------------------------
        self._log_section("Drift Detection")
        drift_detected = self._check_drift(dry_run)
        if drift_detected and not self._retrain_flag:
            self._retrain_flag = True
            self._trigger_reason = "drift_detected"

        # ----------------------------------------------------------
        # Step 3: Decide whether to retrain
        # ----------------------------------------------------------
        retrained = False
        checkpoint_path = None

        if self._retrain_flag:
            prompt = (
                f"Trigger EWC retraining? "
                f"(reason: {self._trigger_reason}"
                + (
                    f", sources: {', '.join(self._data_sources)}"
                    if self._data_sources
                    else ""
                )
                + ")"
            )
            approved = self._require_approval(prompt, dry_run=dry_run)

            if approved:
                checkpoint_path = self._run_training(dry_run)
                retrained = checkpoint_path is not None

                # --------------------------------------------------
                # Step 4 [NEW]: Emit CHECKPOINT_READY to InterpretabilityAgent
                # --------------------------------------------------
                if retrained and not dry_run:
                    self._emit_checkpoint_ready(checkpoint_path, dry_run)
                elif dry_run:
                    self.logger.info(
                        "  [dry-run] Would emit CHECKPOINT_READY → %s",
                        _INTERPRETABILITY_AGENT,
                    )
            else:
                self.logger.info("Retraining deferred (approval not granted).")
        else:
            self.logger.info("No retrain trigger — skipping training run.")

        # Mark inbox messages as read now that we've acted (or decided not to)
        for msg_id in processed_ids:
            self.mark_message_read(msg_id)

        result = {
            "action": "ewc_lifecycle",
            "drift_detected": drift_detected,
            "retrain_triggered": self._retrain_flag,
            "retrained": retrained,
            "checkpoint": checkpoint_path,
            "trigger_reason": self._trigger_reason,
            "inbox_processed": len(processed_ids),
        }
        self._log_finish(result)
        return result

    # ------------------------------------------------------------------
    # Inbox processing — NEW
    # ------------------------------------------------------------------

    def _process_inbox(self, dry_run: bool) -> list[str]:
        """
        Read actionable inbox messages and update internal state accordingly.

        Returns a list of message IDs that were processed, so run() can
        mark them as read after the training decision is made.
        """
        messages = self.get_actionable()
        processed_ids: list[str] = []

        for msg in messages:
            subject = msg["subject"]
            payload = msg.get("payload", {})
            sender = msg["from_agent"]
            msg_id = msg["id"]

            # ── DATA_UPDATED ───────────────────────────────────────
            if subject == DATA_UPDATED:
                source = payload.get("source", "unknown")
                ingest_approved = payload.get("ingest_approved", False)
                change_type = payload.get("change_type", "unknown")

                self.logger.info(
                    "📨  DATA_UPDATED from %s — source=%s  ingest_approved=%s",
                    sender,
                    source,
                    ingest_approved,
                )

                if ingest_approved:
                    # Data is in the pipeline — flag for retrain
                    self._retrain_flag = True
                    self._data_sources.append(source)
                    self._trigger_reason = "data_updated"
                    self.logger.info(
                        "  ↳ Ingest approved for %s — retrain flagged.", source
                    )
                else:
                    # Ingest not yet approved — log signal but defer retrain
                    self.logger.info(
                        "  ↳ Ingest NOT yet approved for %s — "
                        "retrain deferred until ingest completes.",
                        source,
                    )
                    self._update_section(
                        "training",
                        {"pending_data_source": source},
                    )

            # ── FEATURE_INSTABILITY ────────────────────────────────
            elif subject == FEATURE_INSTABILITY:
                flagged = payload.get("flagged_features", [])
                reason = payload.get("reason", "SHAP instability detected")
                self.logger.info(
                    "📨  FEATURE_INSTABILITY from %s — %d feature(s) flagged.",
                    sender,
                    len(flagged),
                )
                for f in flagged:
                    self.logger.info("  ↳ Flagged feature: %s", f)

                # Persist instability flags into SharedState for next EWC run
                state = self._state.load()
                existing = state.get("training", {}).get("instability_flags", [])
                new_flags = [
                    {
                        "feature": f,
                        "reason": reason,
                        "flagged_at": datetime.now(timezone.utc).isoformat(),
                        "resolved": False,
                    }
                    for f in flagged
                    if f not in [x.get("feature") for x in existing]
                ]
                if new_flags:
                    existing.extend(new_flags)
                    self._update_section("training", {"instability_flags": existing})
                    self.logger.info(
                        "  ↳ %d new instability flag(s) written to SharedState.",
                        len(new_flags),
                    )

                # If SHAP found serious instability, flag for retrain
                if payload.get("severity") == "high":
                    self._retrain_flag = True
                    self._trigger_reason = "feature_instability"
                    self.logger.info("  ↳ High-severity instability — retrain flagged.")

            # ── FEATURE_CANDIDATE_ADDED ────────────────────────────
            elif subject == FEATURE_CANDIDATE_ADDED:
                candidate = payload.get("candidate_name", "unknown")
                source = payload.get("literature_source", "unknown")
                self.logger.info(
                    "📨  FEATURE_CANDIDATE_ADDED from %s — candidate=%s  source=%s",
                    sender,
                    candidate,
                    source,
                )
                # Log to SharedState for the next feature-engineering review
                state = self._state.load()
                candidates = state.get("training", {}).get(
                    "pending_feature_candidates", []
                )
                candidates.append(
                    {
                        "name": candidate,
                        "source": source,
                        "added_at": datetime.now(timezone.utc).isoformat(),
                        "reviewed": False,
                    }
                )
                self._update_section(
                    "training", {"pending_feature_candidates": candidates}
                )
                self.logger.info(
                    "  ↳ Feature candidate '%s' queued for review.", candidate
                )

            else:
                self.logger.warning(
                    "Unrecognised message subject '%s' from %s — skipping.",
                    subject,
                    sender,
                )

            processed_ids.append(msg_id)

        return processed_ids

    # ------------------------------------------------------------------
    # Emit CHECKPOINT_READY — NEW
    # ------------------------------------------------------------------

    def _emit_checkpoint_ready(self, checkpoint_path: str, dry_run: bool) -> None:
        """
        Notify InterpretabilityAgent that a new checkpoint is ready for
        SHAP analysis.
        """
        payload = {
            "checkpoint_path": checkpoint_path,
            "trigger_reason": self._trigger_reason,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "ewc_applied": True,
            "data_sources": self._data_sources,
        }
        self.send_message(
            to=_INTERPRETABILITY_AGENT,
            subject=CHECKPOINT_READY,
            payload=payload,
            priority=PRIORITY_HIGH,
        )
        self.logger.info(
            "→ CHECKPOINT_READY sent to %s  [checkpoint=%s]",
            _INTERPRETABILITY_AGENT,
            checkpoint_path,
        )

    # ------------------------------------------------------------------
    # EWC training logic — unchanged from existing implementation
    # ------------------------------------------------------------------

    def _check_drift(self, dry_run: bool) -> bool:
        """
        Run drift detection against the most recent variant batch.
        Returns True if drift is detected above threshold.
        """
        self.logger.info("Running drift detection …")
        try:
            from ewc_utils import detect_drift

            drift = detect_drift(self._get_section("training"))
            if drift:
                self.logger.info("Drift detected above threshold — retrain warranted.")
            else:
                self.logger.info("Drift within acceptable bounds.")
            return drift
        except Exception as exc:
            self.logger.warning(
                "Drift detection failed: %s — treating as no drift.", exc
            )
            return False

    def _run_training(self, dry_run: bool) -> str | None:
        """
        Execute the EWC retraining script and return the checkpoint path,
        or None if training failed.
        """
        if dry_run:
            self.logger.info("  [dry-run] Would run: %s", _MODEL_RETRAIN_SCRIPT)
            return None

        self._log_section("EWC Training")
        self.logger.info("Running: %s", _MODEL_RETRAIN_SCRIPT)

        # Record the start of this training run
        self._update_section(
            "training",
            {
                "last_run": datetime.now(timezone.utc).isoformat(),
                "trigger_reason": self._trigger_reason,
            },
        )

        try:
            result = subprocess.run(
                _MODEL_RETRAIN_SCRIPT,
                shell=True,
                capture_output=True,
                text=True,
                timeout=7200,
            )
            if result.returncode != 0:
                self.logger.error(
                    "Training script failed (exit %d):\n%s",
                    result.returncode,
                    result.stderr[:1000],
                )
                return None

            self.logger.info("Training completed successfully.")
            checkpoint_path = self._locate_latest_checkpoint()
            if checkpoint_path:
                self._update_section("training", {"last_checkpoint": checkpoint_path})
                self.logger.info("Checkpoint saved: %s", checkpoint_path)
            return checkpoint_path

        except subprocess.TimeoutExpired:
            self.logger.error("Training timed out after 7200s.")
            return None
        except Exception as exc:
            self.logger.error("Training error: %s", exc)
            return None

    def _locate_latest_checkpoint(self) -> str | None:
        """
        Find the most recently modified checkpoint file in CHECKPOINT_DIR.
        Returns the path string, or None if no checkpoints found.
        """
        checkpoint_dir = Path(CHECKPOINT_DIR)
        if not checkpoint_dir.exists():
            self.logger.warning("Checkpoint directory not found: %s", checkpoint_dir)
            return None

        candidates = sorted(
            checkpoint_dir.glob("*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            self.logger.warning("No .pt checkpoints found in %s", checkpoint_dir)
            return None

        return str(candidates[0])
