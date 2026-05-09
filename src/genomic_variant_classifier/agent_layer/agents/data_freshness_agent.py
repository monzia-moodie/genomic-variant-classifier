"""
data_freshness_agent.py — Data Freshness Monitor
=================================================
Polls upstream genomic data sources (ClinVar, gnomAD, LOVD, AlphaMissense)
for new releases. When a change is detected:
  1. Flags the change in SharedState (existing behaviour).
  2. Optionally triggers a Spark ingest (existing behaviour, requires approval).
  3. [NEW] Sends a DATA_UPDATED message to TrainingLifecycleAgent via the
     MessageBus so the training agent is aware a retrain may be warranted,
     even if the Spark ingest is deferred or rejected.

The message is sent regardless of whether the Spark ingest is approved —
it carries the source metadata so TrainingLifecycleAgent can decide
independently whether to act on it.

Message emitted
---------------
  Subject : DATA_UPDATED
  To      : TrainingLifecycleAgent
  Payload : {
      "source":        "gnomAD" | "ClinVar" | "AlphaMissense" | "LOVD",
      "change_type":   "fingerprint_changed" | "etag_changed" | "new_release",
      "previous":      "<old fingerprint/etag>",
      "current":       "<new fingerprint/etag>",
      "detected_at":   "<iso timestamp>",
      "ingest_approved": true | false
  }
  Priority          : HIGH
  Requires approval : True (per APPROVAL_REQUIRED_SUBJECTS default)
"""

from __future__ import annotations

import ftplib
import gzip
import hashlib
import io
import logging
import urllib.request
from datetime import datetime, timezone

# requests is imported lazily inside each poll method so the module
# loads cleanly even before the package is confirmed installed.

from agents.base_agent import BaseAgent
from config import (
    ALPHAMISSENSE_MANIFEST,  # was ALPHAMISSENSE_MANIFEST_URL
    CLINVAR_FTP_ROOT,  # was CLINVAR_FTP_HOST
    CLINVAR_FTP_VCF_DIR,  # was CLINVAR_FTP_PATH
    DATAPROC_BUCKET,
    DATAPROC_CLUSTER_NAME,
    GCP_PROJECT_ID,
    GCP_REGION,
    LOVD_API_BASE,
    LOVD_GENES_OF_INTEREST,
    LOVD_VARIANTS_ENDPOINT,
    REQUIRE_HUMAN_APPROVAL,
    SPARK_INGEST_JOB_PATH,
)
from message_bus import DATA_UPDATED, PRIORITY_HIGH
from shared_state import SharedState

# gnomAD: use a HEAD check against a known stable GCS index file.
# The ETag/Last-Modified changes on every gnomAD release.
_GNOMAD_FINGERPRINT_URL = (
    "https://storage.googleapis.com/gcp-public-data--gnomad/"
    "release/4.0/vcf/exomes/gnomad.exomes.v4.0.sites.chr1.vcf.bgz.tbi"
)

# Spark ingest command — assembled from config components.
_SPARK_INGEST_CMD = (
    f"gcloud dataproc jobs submit pyspark {SPARK_INGEST_JOB_PATH} "
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

_RECIPIENT = "TrainingLifecycleAgent"


class DataFreshnessAgent(BaseAgent):
    """
    Monitors upstream genomic data sources and signals downstream agents
    when new data is available.
    """

    def __init__(self, shared_state: SharedState) -> None:
        super().__init__(shared_state)
        self._changes: list[dict] = []  # accumulated per-run change records

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self, dry_run: bool = False) -> dict:
        self._log_start(dry_run)
        self._changes = []

        prev = self._get_section("data_freshness")
        self.logger.info("Polling upstream data sources …")

        clinvar_new = self._poll_clinvar(prev.get("clinvar", {}))
        gnomad_new = self._poll_gnomad(prev.get("gnomad", {}))
        lovd_new = self._poll_lovd(prev.get("lovd", {}))
        alphamissense_new = self._poll_alphamissense(prev.get("alphamissense", {}))

        any_change = any([clinvar_new, gnomad_new, lovd_new, alphamissense_new])
        ingest_approved = False

        if any_change:
            sources = [c["source"] for c in self._changes]
            self.logger.info(
                "Ingest trigger: new release(s) detected — %s.",
                ", ".join(sources),
            )

            if not dry_run:
                prompt = (
                    f"Trigger Spark ingest? "
                    f"({len(self._changes)} source change(s): "
                    f"{', '.join(sources)})"
                )
                ingest_approved = self._require_approval(prompt, dry_run=False)
                if ingest_approved:
                    self._trigger_spark_ingest()
            else:
                self.logger.info(
                    "  [dry-run] Would trigger Spark ingest for: %s",
                    ", ".join(sources),
                )

            # --- NEW: notify TrainingLifecycleAgent for each changed source ---
            if not dry_run:
                for change in self._changes:
                    self._emit_data_updated(change, ingest_approved)
            else:
                for change in self._changes:
                    self.logger.info(
                        "  [dry-run] Would send DATA_UPDATED → %s  [source=%s]",
                        _RECIPIENT,
                        change["source"],
                    )
        else:
            self.logger.info("No upstream changes detected.")

        result = {
            "action": "poll_and_flag",
            "changes_detected": len(self._changes),
            "ingest_approved": ingest_approved,
            "triggered": ingest_approved,
        }
        self._log_finish(result)
        return result

    # ------------------------------------------------------------------
    # NEW: emit DATA_UPDATED message
    # ------------------------------------------------------------------

    def _emit_data_updated(self, change: dict, ingest_approved: bool) -> None:
        """
        Send a DATA_UPDATED message to TrainingLifecycleAgent carrying
        metadata about what changed and whether ingest was approved.
        """
        payload = {
            "source": change["source"],
            "change_type": change["change_type"],
            "previous": change.get("previous"),
            "current": change.get("current"),
            "detected_at": change.get("detected_at"),
            "ingest_approved": ingest_approved,
        }
        self.send_message(
            to=_RECIPIENT,
            subject=DATA_UPDATED,
            payload=payload,
            priority=PRIORITY_HIGH,
        )

    # ------------------------------------------------------------------
    # Poll helpers — unchanged from existing implementation
    # ------------------------------------------------------------------

    def _record_change(
        self,
        source: str,
        change_type: str,
        previous: str | None,
        current: str | None,
    ) -> None:
        """Record a detected change for this run and update SharedState."""
        now = datetime.now(timezone.utc).isoformat()
        self._changes.append(
            {
                "source": source,
                "change_type": change_type,
                "previous": previous,
                "current": current,
                "detected_at": now,
            }
        )
        # Persist the new fingerprint/etag immediately
        self._update_section(
            "data_freshness",
            {
                source.lower(): {
                    "last_seen": current,
                    "last_checked": now,
                }
            },
        )

    def _touch_checked(self, source_key: str) -> None:
        """Update last_checked timestamp even when no change is found."""
        state = self._state.load()
        state.setdefault("data_freshness", {}).setdefault(source_key, {})[
            "last_checked"
        ] = datetime.now(timezone.utc).isoformat()
        self._state.save(state)

    # ------------------------------------------------------------------
    # ClinVar
    # ------------------------------------------------------------------

    def _poll_clinvar(self, prev: dict) -> bool:
        self.logger.info("Polling ClinVar FTP …")
        last_seen = prev.get("last_seen")
        try:
            with ftplib.FTP(CLINVAR_FTP_ROOT, timeout=30) as ftp:
                ftp.login()
                files = ftp.nlst(CLINVAR_FTP_VCF_DIR)
                vcf_files = sorted(f for f in files if f.endswith(".vcf.gz"))
                latest = vcf_files[-1] if vcf_files else None

            if latest and latest != last_seen:
                self.logger.info("ClinVar: new VCF detected — %s.", latest)
                self._record_change("ClinVar", "new_release", last_seen, latest)
                return True

            self._touch_checked("clinvar")
            return False

        except Exception as exc:
            self.logger.warning("ClinVar FTP poll failed: %s", exc)
            self._touch_checked("clinvar")
            return False

    # ------------------------------------------------------------------
    # gnomAD
    # ------------------------------------------------------------------

    def _poll_gnomad(self, prev: dict) -> bool:
        self.logger.info("Polling gnomAD …")
        last_seen = prev.get("last_seen")
        try:
            import requests

            resp = requests.head(_GNOMAD_FINGERPRINT_URL, timeout=20)
            fingerprint = (
                resp.headers.get("ETag")
                or resp.headers.get("Last-Modified")
                or str(resp.status_code)
            )
            if fingerprint != last_seen:
                self.logger.info("gnomAD: change detected (fingerprint changed).")
                self._record_change(
                    "gnomAD", "fingerprint_changed", last_seen, fingerprint
                )
                return True

            self._touch_checked("gnomad")
            return False

        except Exception as exc:
            self.logger.warning("gnomAD poll failed: %s", exc)
            self._touch_checked("gnomad")
            return False

    # ------------------------------------------------------------------
    # LOVD
    # ------------------------------------------------------------------

    def _poll_lovd(self, prev: dict) -> bool:
        self.logger.info("Polling LOVD …")
        last_seen = prev.get("last_seen")
        if not LOVD_GENES_OF_INTEREST:
            self._touch_checked("lovd")
            return False

        probe_gene = LOVD_GENES_OF_INTEREST[0]
        url = f"{LOVD_API_BASE}{LOVD_VARIANTS_ENDPOINT}/{probe_gene}?format=application/json"
        try:
            import requests

            resp = requests.get(url, timeout=20)
            if resp.status_code in (401, 403):
                self.logger.warning(
                    "LOVD REST API requires authentication (HTTP %d). "
                    "Skipping LOVD poll. Set LOVD_API_KEY in config.py to enable.",
                    resp.status_code,
                )
                self._touch_checked("lovd")
                return False
            if resp.status_code == 402:
                self.logger.warning(
                    "LOVD REST API returned HTTP 402 (unsupported). "
                    "Skipping LOVD poll."
                )
                self._touch_checked("lovd")
                return False

            resp.raise_for_status()
            fingerprint = hashlib.md5(resp.content).hexdigest()
            if fingerprint != last_seen:
                self.logger.info("LOVD: change detected.")
                self._record_change(
                    "LOVD", "fingerprint_changed", last_seen, fingerprint
                )
                return True

            self._touch_checked("lovd")
            return False

        except requests.RequestException as exc:
            self.logger.warning("LOVD poll failed: %s", exc)
            self._touch_checked("lovd")
            return False

    # ------------------------------------------------------------------
    # AlphaMissense
    # ------------------------------------------------------------------

    def _poll_alphamissense(self, prev: dict) -> bool:
        self.logger.info("Checking AlphaMissense manifest …")
        last_seen = prev.get("last_seen")
        try:
            import requests

            resp = requests.head(ALPHAMISSENSE_MANIFEST, timeout=20)
            etag = resp.headers.get("ETag") or resp.headers.get("Last-Modified")
            if etag and etag != last_seen:
                self.logger.info("AlphaMissense: new release (ETag changed).")
                self._record_change("AlphaMissense", "etag_changed", last_seen, etag)
                return True

            self._touch_checked("alphamissense")
            return False

        except Exception as exc:
            self.logger.warning("AlphaMissense poll failed: %s", exc)
            self._touch_checked("alphamissense")
            return False

    # ------------------------------------------------------------------
    # Spark ingest trigger — unchanged
    # ------------------------------------------------------------------

    def _trigger_spark_ingest(self) -> None:
        import subprocess

        self.logger.info("Triggering Spark ingest: %s", _SPARK_INGEST_CMD)
        try:
            result = subprocess.run(
                _SPARK_INGEST_CMD,
                shell=True,
                capture_output=True,
                text=True,
                timeout=3600,
            )
            if result.returncode == 0:
                self.logger.info("Spark ingest completed successfully.")
            else:
                self.logger.error(
                    "Spark ingest failed (exit %d): %s",
                    result.returncode,
                    result.stderr[:500],
                )
        except subprocess.TimeoutExpired:
            self.logger.error("Spark ingest timed out after 3600s.")
        except Exception as exc:
            self.logger.error("Spark ingest error: %s", exc)
