"""
agents/data_freshness_agent.py
==============================
Data Freshness Agent — polls upstream genomic databases for new releases,
detects schema/classification drift, and triggers Spark re-ingestion when
the data has materially changed.

Pipeline
--------
1. poll_clinvar()          — check FTP for a new weekly VCF release
2. poll_gnomad()           — check gnomAD API for a new dataset version
3. poll_lovd()             — query LOVD REST API for recently updated entries
4. check_alphamissense()   — compare AlphaMissense manifest hash
5. compute_drift()         — Jensen-Shannon divergence on pathogenicity dist.
6. trigger_spark_ingest()  — submit Dataproc job if thresholds exceeded
7. flag_reclassifications() — surface variants whose ClinVar sig changed
"""

import ftplib
import gzip
import hashlib
import io
import json
import logging
import re
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import requests

# Add parent dir to path when running agents/ directly
import sys

_AL = Path(__file__).resolve().parent.parent
for _p in (str(_AL), str(_AL / "agents")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from base_agent import AgentResult, BaseAgent
from config import (
    ALPHAMISSENSE_MANIFEST,
    CLINVAR_FTP_ROOT,
    CLINVAR_FTP_VCF_DIR,
    CLINVAR_SUMMARY_URL,
    CORPUS_MANIFEST_PATH,
    DATAPROC_BUCKET,
    DATAPROC_CLUSTER_NAME,
    DRIFT_JS_THRESHOLD,
    GCP_PROJECT_ID,
    GCP_REGION,
    GNOMAD_API_BASE,
    GNOMAD_DATASET_LATEST,
    LOVD_API_BASE,
    LOVD_GENES_OF_INTEREST,
    MIN_NEW_VARIANTS_FOR_INGEST,
    RAW_DATA_DIR,
    RECLASSIFICATION_RATE_THRESHOLD,
    REQUIRE_HUMAN_APPROVAL,
)
from shared_state import SharedState

log = logging.getLogger("DataFreshnessAgent")


# ---------------------------------------------------------------------------
# ClinVar pathogenicity categories we track for drift
# ---------------------------------------------------------------------------
PATHOGENICITY_CATS = [
    "Pathogenic",
    "Likely pathogenic",
    "Uncertain significance",
    "Likely benign",
    "Benign",
]


class DataFreshnessAgent(BaseAgent):
    """
    Monitors ClinVar, gnomAD, LOVD, and AlphaMissense for new data.
    Computes distribution drift and triggers Spark re-ingestion as needed.
    """

    def __init__(self, state: SharedState, dry_run: bool = False):
        super().__init__(state, dry_run)
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        return "DataFreshnessAgent"

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(self) -> AgentResult:
        self.log.info("Polling upstream data sources …")

        findings: dict[str, Any] = {
            "clinvar": None,
            "gnomad": None,
            "lovd": None,
            "alphamissense": None,
            "drift_score": None,
            "new_variants": 0,
            "reclassified": 0,
            "spark_triggered": False,
            "errors": [],
        }

        # ---- 1. Poll each source ----------------------------------------
        try:
            findings["clinvar"] = self.poll_clinvar()
        except Exception as exc:
            findings["errors"].append(f"ClinVar poll failed: {exc}")
            self.log.error("ClinVar: %s", exc)

        try:
            findings["gnomad"] = self.poll_gnomad()
        except Exception as exc:
            findings["errors"].append(f"gnomAD poll failed: {exc}")
            self.log.error("gnomAD: %s", exc)

        try:
            findings["lovd"] = self.poll_lovd()
        except Exception as exc:
            findings["errors"].append(f"LOVD poll failed: {exc}")
            self.log.error("LOVD: %s", exc)

        try:
            findings["alphamissense"] = self.check_alphamissense()
        except Exception as exc:
            findings["errors"].append(f"AlphaMissense check failed: {exc}")
            self.log.error("AlphaMissense: %s", exc)

        # ---- 2. Aggregate new variant count ------------------------------
        for source_key in ("clinvar", "gnomad", "lovd"):
            src = findings[source_key]
            if src and src.get("new_variants"):
                findings["new_variants"] += src["new_variants"]
            if src and src.get("reclassified"):
                findings["reclassified"] += src["reclassified"]

        # ---- 3. Drift computation ----------------------------------------
        clinvar_data = findings.get("clinvar") or {}
        if clinvar_data.get("current_dist") and clinvar_data.get("previous_dist"):
            findings["drift_score"] = self._js_divergence(
                clinvar_data["previous_dist"],
                clinvar_data["current_dist"],
            )
            self.state.record_drift(findings["drift_score"])
            self.log.info(
                "JS divergence = %.4f  (threshold %.4f)",
                findings["drift_score"],
                DRIFT_JS_THRESHOLD,
            )

        # ---- 4. Decide whether to trigger Spark --------------------------
        should_ingest = self._should_trigger_ingest(findings)

        if should_ingest:
            if REQUIRE_HUMAN_APPROVAL:
                approved = self.require_human_approval(
                    f"Trigger Spark ingest? "
                    f"({findings['new_variants']} new variants, "
                    f"drift={findings.get('drift_score', 'n/a'):.4f})"
                    if findings.get("drift_score")
                    else f"Trigger Spark ingest? ({findings['new_variants']} new variants)"
                )
                if not approved:
                    self.state.add_pending_review(
                        {
                            "reason": "Spark ingest blocked pending human approval",
                            "details": findings,
                            "agent": self.name,
                        }
                    )
                    return AgentResult(
                        success=True,
                        action="poll_and_flag",
                        details={
                            **findings,
                            "spark_triggered": False,
                            "queued_for_review": True,
                        },
                    )

            spark_result = self.trigger_spark_ingest(findings)
            findings["spark_triggered"] = spark_result.get("submitted", False)

        # ---- 5. Flag high reclassification rate --------------------------
        if findings["new_variants"] > 0:
            reclass_rate = findings["reclassified"] / findings["new_variants"]
            self.state.set("reclassification_rate", reclass_rate)
            if reclass_rate >= RECLASSIFICATION_RATE_THRESHOLD:
                self._flag_reclassifications(findings)

        # ---- 6. Persist source timestamps --------------------------------
        self._update_last_seen(findings)

        success = len(findings["errors"]) == 0 or findings["spark_triggered"]
        return AgentResult(
            success=success,
            action="poll_and_ingest" if findings["spark_triggered"] else "poll",
            details=findings,
            errors=findings["errors"],
        )

    # ------------------------------------------------------------------
    # Source pollers
    # ------------------------------------------------------------------

    def poll_clinvar(self) -> dict:
        """
        Check NCBI FTP for a new ClinVar weekly VCF release.
        Returns metadata + distribution over pathogenicity categories.
        """
        self.log.info("Polling ClinVar FTP …")

        result: dict = {
            "new_release": False,
            "release_date": None,
            "new_variants": 0,
            "reclassified": 0,
            "current_dist": None,
            "previous_dist": None,
        }

        # Connect to FTP and find the latest dated VCF file
        try:
            ftp = ftplib.FTP(CLINVAR_FTP_ROOT, timeout=30)
            ftp.login()
            ftp.cwd(CLINVAR_FTP_VCF_DIR)

            # NLST returns plain filenames — simpler and more reliable than MLSD
            entries = ftp.nlst()
            ftp.quit()

            vcf_files = [
                e
                for e in entries
                if re.search(r"ClinVar.*\.vcf\.gz$", e, re.IGNORECASE)
                and "weekly" not in e.lower()
            ]
            if not vcf_files:
                self.log.warning("No ClinVar VCF files found on FTP.")
                return result

            # Parse the most recent file name for its date stamp
            latest = sorted(vcf_files)[-1]
            date_match = re.search(r"(\d{8})", latest)
            if not date_match:
                return result

            release_date_str = date_match.group(1)  # e.g. "20250115"
            result["release_date"] = release_date_str

            last_seen = self.state.get("clinvar_last_seen_date")
            if last_seen and last_seen >= release_date_str:
                self.log.info(
                    "ClinVar: no new release (last=%s, latest=%s)",
                    last_seen,
                    release_date_str,
                )
                return result

            result["new_release"] = True
            self.log.info("ClinVar: new release detected → %s", release_date_str)

        except ftplib.all_errors as exc:
            self.log.warning(
                "FTP connection failed (%s); falling back to HTTP summary.", exc
            )
            result["new_release"] = True  # assume new if we can't check

        # Download variant summary (tab-delimited, gzipped) for distribution
        previous_dist = self._load_previous_clinvar_dist()
        current_dist = self._download_clinvar_summary_dist()

        result["current_dist"] = current_dist
        result["previous_dist"] = previous_dist

        if current_dist and previous_dist:
            new_total = sum(current_dist.values())
            prev_total = sum(previous_dist.values())
            result["new_variants"] = max(0, new_total - prev_total)

        return result

    def _download_clinvar_summary_dist(self) -> dict | None:
        """
        Stream-parse ClinVar variant_summary.txt.gz and return a Counter
        over the 5 pathogenicity categories.  Only reads the ClinicalSignificance
        column (col 6) to keep memory usage low on large files.
        """
        self.log.info("Downloading ClinVar summary for distribution …")
        try:
            dest = RAW_DATA_DIR / "clinvar_variant_summary.txt.gz"
            urllib.request.urlretrieve(CLINVAR_SUMMARY_URL, dest)

            counts: Counter = Counter()
            with gzip.open(dest, "rt", encoding="utf-8", errors="replace") as fh:
                for i, line in enumerate(fh):
                    if i == 0:  # skip header
                        continue
                    cols = line.split("\t")
                    if len(cols) < 7:
                        continue
                    sig = cols[6].strip()
                    for cat in PATHOGENICITY_CATS:
                        if cat.lower() in sig.lower():
                            counts[cat] += 1
                            break
                    else:
                        counts["Other"] += 1

            dist = dict(counts)
            # Persist for next run's drift comparison
            cache = RAW_DATA_DIR / "clinvar_dist_cache.json"
            with open(cache, "w") as fh:
                json.dump(dist, fh)
            self.log.info("ClinVar distribution: %s", dist)
            return dist

        except Exception as exc:
            self.log.error("Failed to download/parse ClinVar summary: %s", exc)
            return None

    def _load_previous_clinvar_dist(self) -> dict | None:
        cache = RAW_DATA_DIR / "clinvar_dist_cache.json"
        if cache.exists():
            with open(cache) as fh:
                return json.load(fh)
        return None

    def poll_gnomad(self) -> dict:
        """
        Detect a new gnomAD release by checking the ETag of a known stable
        release index file on Google Cloud Storage.  This avoids the gnomAD
        GraphQL API which changes schema between major versions.
        """
        self.log.info("Polling gnomAD …")
        result = {"new_release": False, "version": None, "new_variants": 0}

        # Stable GCS URL for the gnomAD v4 sites VCF index — changes when a
        # new release is published.
        gnomad_index_url = (
            "https://storage.googleapis.com/gcp-public-data--gnomad/"
            "release/4.1/vcf/genomes/gnomad.genomes.v4.1.sites.chr1.vcf.bgz.tbi"
        )
        try:
            resp = requests.head(gnomad_index_url, timeout=15, allow_redirects=True)
            etag = resp.headers.get("ETag") or resp.headers.get("x-goog-hash", "")
            last_modified = resp.headers.get("Last-Modified", "")
            # Use ETag + Last-Modified as a composite version fingerprint
            fingerprint = f"{etag}|{last_modified}"
            result["version"] = fingerprint[:80]

            last_seen = self.state.get("gnomad_last_seen_version")
            if fingerprint and fingerprint != last_seen:
                result["new_release"] = True
                self.log.info("gnomAD: change detected (fingerprint changed).")
            else:
                self.log.info("gnomAD: no change detected.")
        except requests.RequestException as exc:
            self.log.warning("gnomAD check failed: %s", exc)

        return result

    def poll_lovd(self) -> dict:
        """
        Query the LOVD REST API for variants updated since the last run.

        NOTE: As of 2025, the LOVD shared database REST API (databases.lovd.nl)
        returns HTTP 402 Payment Required for programmatic access.  This method
        returns an empty result and logs a single informational warning rather
        than hammering all configured genes.

        To re-enable: obtain LOVD API credentials, add them to config.py as
        LOVD_API_KEY, and pass an Authorization header in the request below.
        """
        self.log.info("Polling LOVD …")
        result = {"new_variants": 0, "reclassified": 0, "genes_updated": []}

        # Quick probe on a single gene to check current access status
        probe_url = (
            f"{LOVD_API_BASE}/variants/{LOVD_GENES_OF_INTEREST[0]}"
            f"?format=application/json&limit=1"
        )
        try:
            resp = requests.get(probe_url, timeout=10)
            if resp.status_code == 402:
                self.log.warning(
                    "LOVD REST API requires authentication (HTTP 402). "
                    "Skipping LOVD poll. Set LOVD_API_KEY in config.py to enable."
                )
                return result
            if resp.status_code == 404:
                return result
            resp.raise_for_status()
        except requests.RequestException as exc:
            self.log.warning("LOVD probe failed: %s", exc)
            return result

        # If probe succeeded, poll all genes
        last_ts = self.state.get("lovd_last_seen_timestamp") or "2020-01-01T00:00:00"
        for gene in LOVD_GENES_OF_INTEREST:
            try:
                url = (
                    f"{LOVD_API_BASE}/variants/{gene}"
                    f"?format=application/json&modified_since={last_ts}"
                )
                r = requests.get(url, timeout=20)
                if r.status_code in (402, 404):
                    continue
                r.raise_for_status()
                variants = r.json()
                n = len(variants) if isinstance(variants, list) else 0
                if n:
                    result["new_variants"] += n
                    result["genes_updated"].append(gene)
                    self.log.info("  LOVD %s: %d updated variants", gene, n)
            except requests.RequestException as exc:
                self.log.warning("LOVD %s: %s", gene, exc)

        return result

    def check_alphamissense(self) -> dict:
        """
        Compare the ETag / Content-MD5 of the AlphaMissense manifest
        to detect a new release without downloading the full file.
        """
        self.log.info("Checking AlphaMissense manifest …")
        result = {"new_release": False, "etag": None}

        try:
            resp = requests.head(
                ALPHAMISSENSE_MANIFEST, timeout=15, allow_redirects=True
            )
            etag = resp.headers.get("ETag") or resp.headers.get("x-goog-hash")
            result["etag"] = etag
            last_seen = self.state.get("alphamissense_last_seen")
            if etag and etag != last_seen:
                result["new_release"] = True
                self.log.info("AlphaMissense: new release (ETag changed).")
            else:
                self.log.info("AlphaMissense: no change.")
        except requests.RequestException as exc:
            self.log.warning("AlphaMissense HEAD request failed: %s", exc)

        return result

    # ------------------------------------------------------------------
    # Drift computation
    # ------------------------------------------------------------------

    @staticmethod
    def _js_divergence(p_dict: dict, q_dict: dict) -> float:
        """
        Jensen-Shannon divergence between two category distributions.
        Returns a value in [0, 1]; higher = more drift.
        """
        cats = list(set(p_dict) | set(q_dict))
        p = np.array([p_dict.get(c, 0) for c in cats], dtype=float)
        q = np.array([q_dict.get(c, 0) for c in cats], dtype=float)

        # Normalise to probability distributions
        p = p / p.sum() if p.sum() > 0 else p
        q = q / q.sum() if q.sum() > 0 else q

        m = 0.5 * (p + q)

        # KL divergence with numerical stability
        def kl(a, b):
            mask = (a > 0) & (b > 0)
            return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

        return 0.5 * kl(p, m) + 0.5 * kl(q, m)

    # ------------------------------------------------------------------
    # Spark trigger
    # ------------------------------------------------------------------

    def _should_trigger_ingest(self, findings: dict) -> bool:
        new_v = findings.get("new_variants", 0)
        drift = findings.get("drift_score") or 0.0
        am_new = (findings.get("alphamissense") or {}).get("new_release", False)

        if new_v >= MIN_NEW_VARIANTS_FOR_INGEST:
            self.log.info(
                "Ingest trigger: new_variants=%d ≥ threshold=%d",
                new_v,
                MIN_NEW_VARIANTS_FOR_INGEST,
            )
            return True
        if drift >= DRIFT_JS_THRESHOLD:
            self.log.info(
                "Ingest trigger: drift=%.4f ≥ threshold=%.4f", drift, DRIFT_JS_THRESHOLD
            )
            return True
        if am_new:
            self.log.info("Ingest trigger: new AlphaMissense release.")
            return True
        return False

    def trigger_spark_ingest(self, findings: dict) -> dict:
        """
        Submit a Dataproc job to re-run the VCF ingest pipeline.
        Falls back to a local Spark submit if GCP credentials are absent.
        """
        if self.dry_run:
            self._dry_run_log("Would submit Dataproc VCF ingest job.")
            return {"submitted": False, "dry_run": True}

        self.log.info("Submitting Spark VCF ingest job …")

        # Try Google Cloud Dataproc first
        try:
            from google.cloud import dataproc_v1  # type: ignore

            job_client = dataproc_v1.JobControllerClient(
                client_options={
                    "api_endpoint": f"{GCP_REGION}-dataproc.googleapis.com:443"
                }
            )
            job = {
                "placement": {"cluster_name": DATAPROC_CLUSTER_NAME},
                "pyspark_job": {
                    "main_python_file_uri": f"{DATAPROC_BUCKET}/jobs/vcf_ingest.py",
                    "args": [
                        "--sources",
                        ",".join(
                            s
                            for s, v in {
                                "clinvar": findings.get("clinvar", {}).get(
                                    "new_release"
                                ),
                                "gnomad": findings.get("gnomad", {}).get("new_release"),
                                "lovd": bool(
                                    findings.get("lovd", {}).get("new_variants")
                                ),
                            }.items()
                            if v
                        ),
                    ],
                },
            }
            response = job_client.submit_job(
                request={"project_id": GCP_PROJECT_ID, "region": GCP_REGION, "job": job}
            )
            job_id = response.reference.job_id
            self.log.info("Dataproc job submitted: %s", job_id)
            return {"submitted": True, "job_id": job_id, "backend": "dataproc"}

        except ImportError:
            self.log.warning(
                "google-cloud-dataproc not installed; falling back to local spark-submit."
            )
        except Exception as exc:
            self.log.warning(
                "Dataproc submission failed (%s); falling back to local.", exc
            )

        # POST_DOWNLOAD_HOOKS — must run before Spark ETL reads the data.
        # Hook 1 (mandatory): patch ClinVar alleles.
        #   variant_summary.txt has ref="na"/alt="na" for nearly all rows.
        #   Real alleles must be joined from clinvar.vcf.gz before ETL or
        #   consequence-based features are silently zeroed and AUROC collapses.
        POST_DOWNLOAD_HOOKS = [
            self._hook_patch_clinvar_alleles,
            self._hook_validate_vcf_checksums,
        ]
        for hook in POST_DOWNLOAD_HOOKS:
            try:
                hook(findings)
            except Exception as exc:
                self.log.warning("Post-download hook %s failed: %s", hook.__name__, exc)

        # Local fallback: spark-submit
        import subprocess

        spark_script = Path(__file__).parents[2] / "pipelines" / "vcf_ingest.py"
        if not spark_script.exists():
            self.log.error("Spark script not found at %s", spark_script)
            return {"submitted": False, "error": "script not found"}

        cmd = ["spark-submit", str(spark_script), "--local"]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.log.info("Local spark-submit launched (PID %d).", proc.pid)
            return {"submitted": True, "pid": proc.pid, "backend": "local"}
        except FileNotFoundError:
            self.log.error("spark-submit not found on PATH.")
            return {"submitted": False, "error": "spark not on PATH"}

    def _flag_reclassifications(self, findings: dict) -> None:
        """
        Add a pending review item when the reclassification rate is high.
        The Orchestrator surfaces this to the operator.
        """
        reclass_n = findings.get("reclassified", 0)
        new_total = findings.get("new_variants", 1)
        reclass_rate = reclass_n / new_total

        self.log.warning(
            "High reclassification rate: %.1f%% (%d / %d variants)",
            reclass_rate * 100,
            reclass_n,
            new_total,
        )
        self.state.add_pending_review(
            {
                "reason": "High ClinVar reclassification rate",
                "reclassification_rate": reclass_rate,
                "reclassified_n": reclass_n,
                "new_variants_n": new_total,
                "agent": self.name,
                "action_required": (
                    "Review reclassified variants before next training run. "
                    "Check whether label noise exceeds EWC tolerance."
                ),
            }
        )

    # ------------------------------------------------------------------
    # Post-download hooks (run before Spark ETL)
    # ------------------------------------------------------------------

    def _hook_patch_clinvar_alleles(self, findings: dict) -> None:
        """
        Hook 1 — MANDATORY: join real ref/alt alleles from clinvar.vcf.gz
        into the variant_summary.txt before Spark ETL reads it.

        Without this patch, variant_summary.txt has ref='na'/alt='na' for
        nearly all rows. Consequence-based features (is_missense, is_splice,
        consequence_severity, etc.) will be silently zeroed and AUROC collapses
        from ~0.98 to ~0.70.

        Calls scripts/patch_clinvar_alleles.py which must be on PATH or
        reachable relative to the repo root.
        """
        import subprocess

        patch_script = (
            Path(__file__).parents[2] / "scripts" / "patch_clinvar_alleles.py"
        )
        if not patch_script.exists():
            self.log.warning(
                "patch_clinvar_alleles.py not found at %s — "
                "allele data will be missing from next ETL run.",
                patch_script,
            )
            return

        clinvar_summary = RAW_DATA_DIR / "clinvar_variant_summary.txt.gz"
        clinvar_vcf = RAW_DATA_DIR / "clinvar.vcf.gz"

        if not clinvar_summary.exists():
            self.log.info("ClinVar summary not downloaded yet — skipping allele patch.")
            return
        if not clinvar_vcf.exists():
            self.log.warning(
                "clinvar.vcf.gz not found at %s — "
                "allele patch cannot run. Download from NCBI FTP.",
                clinvar_vcf,
            )
            return

        self.log.info("Running ClinVar allele patch (Hook 1) …")
        cmd = [
            "python",
            str(patch_script),
            "--summary",
            str(clinvar_summary),
            "--vcf",
            str(clinvar_vcf),
            "--output",
            str(clinvar_summary),  # patch in-place
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            self.log.info("ClinVar allele patch complete.")
        else:
            self.log.error(
                "ClinVar allele patch failed (exit %d):\n%s",
                result.returncode,
                result.stderr[-500:],
            )
            raise RuntimeError(
                f"patch_clinvar_alleles.py failed: {result.stderr[-200:]}"
            )

    def _hook_validate_vcf_checksums(self, findings: dict) -> None:
        """
        Hook 2 — validate MD5/SHA256 checksums of downloaded VCF files.
        Logs a warning if checksums are unavailable; never blocks ingest.
        """
        self.log.info("Validating VCF checksums (Hook 2) …")
        vcf_files = list(RAW_DATA_DIR.glob("*.vcf.gz"))
        for vcf in vcf_files:
            md5_file = vcf.with_suffix(".gz.md5")
            if not md5_file.exists():
                self.log.debug("No checksum file for %s — skipping.", vcf.name)
                continue
            expected = md5_file.read_text().strip().split()[0]
            h = hashlib.md5()
            with open(vcf, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            actual = h.hexdigest()
            if actual == expected:
                self.log.info("Checksum OK: %s", vcf.name)
            else:
                self.log.error(
                    "CHECKSUM MISMATCH for %s: expected=%s actual=%s — "
                    "file may be corrupted. Re-download before ETL.",
                    vcf.name,
                    expected,
                    actual,
                )

    # ------------------------------------------------------------------
    # State updates
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # State updates
    # ------------------------------------------------------------------

    def _update_last_seen(self, findings: dict) -> None:
        now = datetime.now(timezone.utc).isoformat()

        clinvar = findings.get("clinvar") or {}
        if clinvar.get("release_date"):
            self.state.set("clinvar_last_seen_date", clinvar["release_date"])

        gnomad = findings.get("gnomad") or {}
        if gnomad.get("version"):
            self.state.set("gnomad_last_seen_version", gnomad["version"])

        lovd = findings.get("lovd") or {}
        if lovd.get("new_variants", 0) > 0:
            self.state.set("lovd_last_seen_timestamp", now)

        am = findings.get("alphamissense") or {}
        if am.get("etag"):
            self.state.set("alphamissense_last_seen", am["etag"])
