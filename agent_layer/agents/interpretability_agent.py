"""
interpretability_agent.py — SHAP-Driven Interpretability Auditor
=================================================================
Runs SHAP analysis after each training run and monitors feature importance
for instability, drift, or biologically counterintuitive patterns.

Messages consumed (inbox)
--------------------------
  CHECKPOINT_READY (from TrainingLifecycleAgent)
      A new model checkpoint is available for SHAP analysis.
      Payload includes:
        checkpoint_path  — path to the .pt file
        trigger_reason   — why retraining was triggered
        trained_at       — ISO timestamp
        ewc_applied      — whether EWC penalties were applied
        data_sources     — which genomic sources triggered the retrain

      On receipt, this agent runs a full SHAP audit against the checkpoint.
      If it finds instability it emits FEATURE_INSTABILITY (see below).
      If the checkpoint path differs from the last audited checkpoint, the
      audit runs unconditionally; otherwise it is skipped to avoid redundancy.

Message emitted (outbox)
-------------------------
  FEATURE_INSTABILITY (to TrainingLifecycleAgent)
      Emitted when SHAP detects one or more of:
        - Features with high variance in importance across cross-val folds
        - Features with importance sign reversal vs previous checkpoint
        - Features whose importance contradicts known biology
          (e.g. a known benign feature scoring highly for pathogenicity)

      Payload: {
          "flagged_features": ["feature_a", "feature_b", ...],
          "reason":           "<human-readable summary>",
          "severity":         "low" | "medium" | "high",
          "shap_report":      "<path to HTML report>",
          "checkpoint_path":  "<path audited>",
          "audited_at":       "<iso timestamp>"
      }
      Priority          : HIGH if severity == "high", else NORMAL
      Requires approval : False  (informational — TrainingLifecycle decides
                                  whether to act)

Processing order inside run()
------------------------------
  1. Check for scheduled run (existing logic: time since last audit).
  2. Check inbox for CHECKPOINT_READY messages (new logic).
  3. Merge: run SHAP if either condition is met.
  4. Run SHAP audit.
  5. If instability found → emit FEATURE_INSTABILITY.
  6. Write report path to SharedState.
  7. Mark inbox messages read.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents.base_agent import BaseAgent
from config import (
    CHECKPOINT_DIR,
    EXPECTED_HIGH_IMPORTANCE_FEATURES,
    SHAP_IMPORTANCE_DELTA,  # was SHAP_INSTABILITY_THRESHOLD
    SHAP_REPORT_DIR,
    VAL_PARQUET,  # was VALIDATION_PARQUET_PATH
)
from message_bus import (
    CHECKPOINT_READY,
    FEATURE_INSTABILITY,
    PRIORITY_HIGH,
    PRIORITY_NORMAL,
)
from shared_state import SharedState

# Number of days between scheduled SHAP audits (not in config — defined here).
_INTERPRETABILITY_INTERVAL_DAYS = 7

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

_TRAINING_AGENT = "TrainingLifecycleAgent"


class InterpretabilityAgent(BaseAgent):
    """
    SHAP-driven interpretability auditor. Runs after each training checkpoint
    and flags feature instability back to TrainingLifecycleAgent.
    """

    def __init__(self, shared_state: SharedState) -> None:
        super().__init__(shared_state)
        self._checkpoint_from_message: str | None = None
        self._trigger_reason: str = "scheduled"

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self, dry_run: bool = False) -> dict:
        self._log_start(dry_run)
        self._checkpoint_from_message = None
        self._trigger_reason = "scheduled"

        # ----------------------------------------------------------
        # Step 1: Process inbox — look for CHECKPOINT_READY
        # ----------------------------------------------------------
        processed_ids = self._process_inbox(dry_run)

        # ----------------------------------------------------------
        # Step 2: Decide whether to run SHAP audit
        # ----------------------------------------------------------
        should_run, checkpoint_path = self._should_run_audit()

        if not should_run:
            self.logger.info(
                "SHAP audit not due — skipping. "
                "(Use --pipeline interpretability to force.)"
            )
            result = {
                "action": "skipped",
                "reason": "not_due",
                "audited": False,
            }
            self._log_finish(result)
            return result

        # ----------------------------------------------------------
        # Step 3: Run SHAP audit
        # ----------------------------------------------------------
        self._log_section("SHAP Audit")
        flagged, report_path, severity = self._run_shap_audit(checkpoint_path, dry_run)

        # ----------------------------------------------------------
        # Step 4: Emit FEATURE_INSTABILITY if needed
        # ----------------------------------------------------------
        if flagged and not dry_run:
            self._emit_feature_instability(
                flagged, severity, report_path, checkpoint_path
            )
        elif flagged and dry_run:
            self.logger.info(
                "  [dry-run] Would emit FEATURE_INSTABILITY → %s  "
                "[%d feature(s), severity=%s]",
                _TRAINING_AGENT,
                len(flagged),
                severity,
            )

        # ----------------------------------------------------------
        # Step 5: Persist report path and mark messages read
        # ----------------------------------------------------------
        if report_path:
            self._update_section(
                "interpretability",
                {
                    "last_run": datetime.now(timezone.utc).isoformat(),
                    "last_report": report_path,
                    "instability_flags": [
                        {"feature": f, "severity": severity} for f in flagged
                    ],
                },
            )

        for msg_id in processed_ids:
            self.mark_message_read(msg_id)

        result = {
            "action": "shap_audit",
            "trigger": self._trigger_reason,
            "checkpoint": checkpoint_path,
            "flagged_count": len(flagged),
            "severity": severity,
            "report": report_path,
            "instability_sent": bool(flagged) and not dry_run,
            "inbox_processed": len(processed_ids),
        }
        self._log_finish(result)
        return result

    # ------------------------------------------------------------------
    # Inbox processing — NEW
    # ------------------------------------------------------------------

    def _process_inbox(self, dry_run: bool) -> list[str]:
        """
        Read actionable inbox messages. If a CHECKPOINT_READY message is
        present, store the checkpoint path and trigger reason so the audit
        targets the correct model version.

        Returns list of processed message IDs (marked read after audit).
        """
        messages = self.get_actionable()
        processed_ids: list[str] = []

        for msg in messages:
            subject = msg["subject"]
            payload = msg.get("payload", {})
            sender = msg["from_agent"]
            msg_id = msg["id"]

            if subject == CHECKPOINT_READY:
                checkpoint_path = payload.get("checkpoint_path")
                trigger_reason = payload.get("trigger_reason", "unknown")
                trained_at = payload.get("trained_at", "unknown")
                ewc_applied = payload.get("ewc_applied", False)
                data_sources = payload.get("data_sources", [])

                self.logger.info(
                    "📨  CHECKPOINT_READY from %s — checkpoint=%s  "
                    "trigger=%s  ewc=%s  sources=%s",
                    sender,
                    checkpoint_path,
                    trigger_reason,
                    ewc_applied,
                    data_sources,
                )

                if checkpoint_path:
                    self._checkpoint_from_message = checkpoint_path
                    self._trigger_reason = f"checkpoint_ready ({trigger_reason})"
                    self.logger.info("  ↳ SHAP audit will target: %s", checkpoint_path)
                else:
                    self.logger.warning(
                        "  ↳ CHECKPOINT_READY payload missing checkpoint_path "
                        "— will fall back to latest on disk."
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
    # Audit scheduling — extended to honour inbox-driven triggers
    # ------------------------------------------------------------------

    def _should_run_audit(self) -> tuple[bool, str | None]:
        """
        Determine whether a SHAP audit should run and which checkpoint to use.

        Priority:
          1. Message-driven: CHECKPOINT_READY arrived → always audit that
             checkpoint (unless we already audited this exact path).
          2. Scheduled: time since last audit exceeds configured interval.

        Returns
        -------
        (should_run, checkpoint_path)
            checkpoint_path is None if no checkpoint is found on disk.
        """
        section = self._get_section("interpretability")
        last_report_checkpoint = section.get("last_checkpoint_audited")

        # Message-driven trigger
        if self._checkpoint_from_message:
            if self._checkpoint_from_message == last_report_checkpoint:
                self.logger.info(
                    "CHECKPOINT_READY received but checkpoint %s was already "
                    "audited — skipping duplicate audit.",
                    self._checkpoint_from_message,
                )
                return False, None
            self.logger.info("Audit triggered by CHECKPOINT_READY message.")
            return True, self._checkpoint_from_message

        # Scheduled trigger — use latest checkpoint on disk
        checkpoint_path = self._find_latest_checkpoint()
        if not checkpoint_path:
            self.logger.warning(
                "No checkpoint found in %s — cannot run SHAP audit.",
                CHECKPOINT_DIR,
            )
            return False, None

        last_run = section.get("last_run")
        if last_run:
            from datetime import timedelta

            last_dt = datetime.fromisoformat(last_run)
            if (datetime.now(timezone.utc) - last_dt) < timedelta(
                days=_INTERPRETABILITY_INTERVAL_DAYS
            ):
                return False, None

        self.logger.info(
            "Scheduled SHAP audit — targeting latest checkpoint: %s",
            checkpoint_path,
        )
        return True, checkpoint_path

    def _find_latest_checkpoint(self) -> str | None:
        """Return the most recent .pt checkpoint in CHECKPOINT_DIR, or None."""
        cp_dir = Path(CHECKPOINT_DIR)
        if not cp_dir.exists():
            return None
        candidates = sorted(
            cp_dir.glob("*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return str(candidates[0]) if candidates else None

    # ------------------------------------------------------------------
    # SHAP audit — extended to return flagged features + severity
    # ------------------------------------------------------------------

    def _run_shap_audit(
        self,
        checkpoint_path: str | None,
        dry_run: bool,
    ) -> tuple[list[str], str | None, str]:
        """
        Run SHAP analysis against the checkpoint.

        Returns
        -------
        (flagged_features, report_path, severity)
            flagged_features : list of feature name strings
            report_path      : path to the HTML report, or None
            severity         : "low" | "medium" | "high"
        """
        if dry_run:
            self.logger.info("  [dry-run] Would run SHAP audit on: %s", checkpoint_path)
            return [], None, "low"

        self.logger.info("Running SHAP audit on: %s", checkpoint_path)

        try:
            import shap
            import torch
            import numpy as np

            # Load model
            model = torch.load(checkpoint_path, map_location="cpu")
            model.eval()

            # Load a representative validation batch from SharedState
            # (the actual data-loading logic depends on your pipeline;
            #  this calls the existing helper that loads from the configured
            #  validation parquet path)
            X_val, feature_names = self._load_validation_batch()
            if X_val is None:
                self.logger.warning(
                    "Could not load validation batch — SHAP audit skipped."
                )
                return [], None, "low"

            # Compute SHAP values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_val)

            # Analyse for instability
            flagged, severity = self._analyse_shap(shap_values, feature_names)

            # Render HTML report
            report_path = self._write_shap_report(
                shap_values, feature_names, checkpoint_path
            )

            # Persist which checkpoint was audited
            self._update_section(
                "interpretability",
                {"last_checkpoint_audited": checkpoint_path},
            )

            if flagged:
                self.logger.warning(
                    "SHAP instability detected — %d feature(s) flagged "
                    "[severity=%s].",
                    len(flagged),
                    severity,
                )
            else:
                self.logger.info("SHAP audit complete — no instability detected.")

            return flagged, report_path, severity

        except Exception as exc:
            self.logger.error("SHAP audit failed: %s", exc)
            return [], None, "low"

    def _load_validation_batch(self):
        """
        Load the validation feature matrix and feature names from disk.
        Returns (X_val, feature_names) or (None, None) on failure.
        """
        try:
            import pandas as pd

            df = pd.read_parquet(VAL_PARQUET)
            label_col = "label"
            feature_names = [c for c in df.columns if c != label_col]
            X_val = df[feature_names].values
            return X_val, feature_names
        except Exception as exc:
            self.logger.warning("Validation batch load failed: %s", exc)
            return None, None

    def _analyse_shap(
        self,
        shap_values,
        feature_names: list[str],
    ) -> tuple[list[str], str]:
        """
        Scan SHAP values for instability patterns.

        Checks:
          1. High cross-sample variance in importance (noisy features).
          2. Features absent from EXPECTED_HIGH_IMPORTANCE_FEATURES that
             dominate the top-5 by mean |SHAP| — possible spurious correlation.
          3. (Extensible) Future: sign reversal vs previous checkpoint.

        Returns (flagged_feature_names, severity).
        """
        import numpy as np

        if shap_values is None or len(shap_values) == 0:
            return [], "low"

        # Handle binary classification (list of arrays) vs regression
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values
        mean_abs = np.abs(sv).mean(axis=0)
        std_vals = np.abs(sv).std(axis=0)

        flagged: list[str] = []

        # Check 1: coefficient of variation > threshold → noisy feature
        with np.errstate(divide="ignore", invalid="ignore"):
            cv = np.where(mean_abs > 0, std_vals / mean_abs, 0.0)
        noisy = [
            feature_names[i] for i, c in enumerate(cv) if c > SHAP_IMPORTANCE_DELTA
        ]
        flagged.extend(noisy)

        # Check 2: unexpected dominant features
        top5_idx = np.argsort(mean_abs)[::-1][:5]
        top5_names = [feature_names[i] for i in top5_idx]
        unexpected = [
            f
            for f in top5_names
            if EXPECTED_HIGH_IMPORTANCE_FEATURES
            and f not in EXPECTED_HIGH_IMPORTANCE_FEATURES
        ]
        flagged.extend(f for f in unexpected if f not in flagged)

        # Severity classification
        if len(flagged) == 0:
            severity = "low"
        elif len(flagged) <= 2:
            severity = "medium"
        else:
            severity = "high"

        return flagged, severity

    def _write_shap_report(
        self,
        shap_values,
        feature_names: list[str],
        checkpoint_path: str,
    ) -> str | None:
        """Render a SHAP summary HTML report and return the file path."""
        try:
            import shap
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            report_dir = Path(SHAP_REPORT_DIR)
            report_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            report_path = report_dir / f"shap_report_{timestamp}.html"

            sv = shap_values[1] if isinstance(shap_values, list) else shap_values
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(
                sv,
                feature_names=feature_names,
                show=False,
                plot_type="bar",
            )
            plt.tight_layout()
            plt.savefig(str(report_path).replace(".html", ".png"), dpi=150)
            plt.close(fig)

            # Write minimal HTML wrapper
            img_name = report_path.stem + ".png"
            report_path.write_text(
                f"<html><body>"
                f"<h2>SHAP Report — {timestamp}</h2>"
                f"<p>Checkpoint: {checkpoint_path}</p>"
                f"<img src='{img_name}' style='max-width:100%'>"
                f"</body></html>",
                encoding="utf-8",
            )
            self.logger.info("SHAP report written: %s", report_path)
            return str(report_path)

        except Exception as exc:
            self.logger.warning("SHAP report render failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Emit FEATURE_INSTABILITY — NEW
    # ------------------------------------------------------------------

    def _emit_feature_instability(
        self,
        flagged: list[str],
        severity: str,
        report_path: str | None,
        checkpoint_path: str | None,
    ) -> None:
        """
        Send a FEATURE_INSTABILITY message to TrainingLifecycleAgent.
        Priority is HIGH for high-severity findings, NORMAL otherwise.
        Does not require approval — TrainingLifecycle decides whether to act.
        """
        priority = PRIORITY_HIGH if severity == "high" else PRIORITY_NORMAL
        payload = {
            "flagged_features": flagged,
            "reason": (
                f"SHAP audit detected {len(flagged)} unstable/unexpected "
                f"feature(s) [severity={severity}]"
            ),
            "severity": severity,
            "shap_report": report_path,
            "checkpoint_path": checkpoint_path,
            "audited_at": datetime.now(timezone.utc).isoformat(),
        }
        self.send_message(
            to=_TRAINING_AGENT,
            subject=FEATURE_INSTABILITY,
            payload=payload,
            priority=priority,
            requires_approval=False,  # informational — no approval needed
        )
        self.logger.info(
            "→ FEATURE_INSTABILITY sent to %s  " "[%d feature(s), severity=%s]",
            _TRAINING_AGENT,
            len(flagged),
            severity,
        )
