"""
literature_scout_agent.py — Literature & Feature Research Scout
===============================================================
Monitors PubMed, bioRxiv, and ClinGen for new publications on variant
pathogenicity, extracts proposed features or scoring methods, and surfaces
candidates for feature engineering review.

Messages emitted (outbox)
--------------------------
  FEATURE_CANDIDATE_ADDED (to TrainingLifecycleAgent)
      Emitted once per newly extracted feature candidate, immediately after
      it is written to SharedState. This gives TrainingLifecycleAgent real-
      time awareness of new candidates without waiting for its next scheduled
      run to poll the queue.

      Payload: {
          "candidate_name":    "<feature or score name>",
          "literature_source": "PubMed" | "bioRxiv" | "ClinGen",
          "pmid_or_doi":       "<identifier>",
          "paper_title":       "<str>",
          "relevance_score":   <float 0.0–1.0>,
          "extracted_at":      "<iso timestamp>"
      }
      Priority          : NORMAL
      Requires approval : False  (informational — TrainingLifecycle
                                  queues it for human review, does not
                                  auto-incorporate it)

Processing order inside run()
------------------------------
  1. Check if 7-day minimum interval has elapsed (existing logic).
  2. Fetch papers from PubMed, bioRxiv, ClinGen (existing logic).
  3. Score and filter candidates (existing logic).
  4. For each NEW candidate (not already in SharedState queue):
       a. Write to SharedState["literature"]["feature_candidates"] (existing).
       b. [NEW] Emit FEATURE_CANDIDATE_ADDED to TrainingLifecycleAgent.
  5. Render HTML digest (existing logic).
"""

from __future__ import annotations

import hashlib
import html
import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# feedparser and requests are imported lazily inside their respective fetch
# methods so that the agent module loads cleanly even if they are not yet
# installed in the environment.

from agents.base_agent import BaseAgent
from config import (
    BIORXIV_RSS_FEEDS,
    CLINGEN_API_BASE,
    LITERATURE_CANDIDATE_MIN_SCORE,
    LITERATURE_DIGEST_DIR,  # was REPORT_DIR
    LITERATURE_FEATURE_PATTERNS,
    LITERATURE_KNOWN_TOOLS,
    LITERATURE_MAX_PAPERS_PER_RUN,  # was LITERATURE_MAX_RESULTS
    LITERATURE_MIN_RELEVANCE,  # was LITERATURE_RELEVANCE_THRESHOLD
    LITERATURE_PUBMED_QUERIES,
    LITERATURE_RELEVANCE_KEYWORDS,  # was LITERATURE_KEYWORDS
    NCBI_API_KEY,
    NCBI_EUTILS_BASE,
)
from message_bus import FEATURE_CANDIDATE_ADDED, PRIORITY_NORMAL
from shared_state import SharedState

# Minimum days between scout runs (not in config — defined here).
_LITERATURE_INTERVAL_DAYS = 7

# Compile feature extraction patterns from config.
# Config patterns use named group (?P<name>...).
_TRAINING_AGENT = "TrainingLifecycleAgent"

# Compile feature extraction patterns from config.
# Config patterns use named group (?P<n>...).
_COMPILED_PATTERNS = [re.compile(p, re.I) for p in LITERATURE_FEATURE_PATTERNS]


class LiteratureScoutAgent(BaseAgent):
    """
    Monitors genomic literature and surfaces new feature candidates,
    notifying TrainingLifecycleAgent in real time via the MessageBus.
    """

    def __init__(self, shared_state: SharedState) -> None:
        super().__init__(shared_state)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self, dry_run: bool = False) -> dict:
        self._log_start(dry_run)

        # ----------------------------------------------------------
        # Step 1: Check run interval (existing logic)
        # ----------------------------------------------------------
        if not self._should_run_literature():
            self.logger.info(
                "Literature scout not due (< %d days since last run). "
                "Use --pipeline literature to force.",
                _LITERATURE_INTERVAL_DAYS,
            )
            result = {"action": "skipped", "reason": "interval_not_elapsed"}
            self._log_finish(result)
            return result

        # ----------------------------------------------------------
        # Step 2: Fetch papers from all sources (existing logic)
        # ----------------------------------------------------------
        self._log_section("Fetching papers")
        pubmed_papers = self._fetch_pubmed()
        biorxiv_papers = self._fetch_biorxiv()
        clingen_papers = self._fetch_clingen()
        all_papers = pubmed_papers + biorxiv_papers + clingen_papers
        self.logger.info("Total papers fetched: %d", len(all_papers))

        # ----------------------------------------------------------
        # Step 3: Score, filter, extract candidates (existing logic)
        # ----------------------------------------------------------
        self._log_section("Extracting feature candidates")
        section = self._get_section("literature")
        existing_ids = {
            c.get("pmid_or_doi") for c in section.get("feature_candidates", [])
        }
        existing_names = {
            c.get("name", "").lower() for c in section.get("feature_candidates", [])
        }

        new_candidates: list[dict] = []
        papers_processed = 0

        for paper in all_papers:
            score = self._relevance_score(paper)
            if score < LITERATURE_MIN_RELEVANCE:
                continue
            papers_processed += 1

            paper_id = paper.get("pmid") or paper.get("doi") or paper.get("url", "")
            if paper_id in existing_ids:
                continue  # already processed this paper

            candidates = self._extract_candidates(paper)
            for candidate_name in candidates:
                if candidate_name.lower() in existing_names:
                    continue
                if candidate_name.lower() in {
                    t.lower() for t in LITERATURE_KNOWN_TOOLS
                }:
                    continue

                now = datetime.now(timezone.utc).isoformat()
                candidate = {
                    "name": candidate_name,
                    "pmid_or_doi": paper_id,
                    "paper_title": paper.get("title", ""),
                    "literature_source": paper.get("source", "unknown"),
                    "relevance_score": round(score, 3),
                    "extracted_at": now,
                    "reviewed": False,
                    "incorporated": False,
                }
                new_candidates.append(candidate)
                existing_names.add(candidate_name.lower())

        # ----------------------------------------------------------
        # Step 4a: Write new candidates to SharedState (existing logic)
        # ----------------------------------------------------------
        if new_candidates:
            state = self._state.load()
            lit = state.setdefault("literature", {})
            queue = lit.setdefault("feature_candidates", [])
            queue.extend(new_candidates)
            lit["last_run"] = datetime.now(timezone.utc).isoformat()
            self._state.save(state)
            self.logger.info(
                "%d new feature candidate(s) added to queue.", len(new_candidates)
            )

            # ----------------------------------------------------------
            # Step 4b [NEW]: Emit FEATURE_CANDIDATE_ADDED per new candidate
            # ----------------------------------------------------------
            if not dry_run:
                for candidate in new_candidates:
                    self._emit_candidate(candidate)
            else:
                for candidate in new_candidates:
                    self.logger.info(
                        "  [dry-run] Would send FEATURE_CANDIDATE_ADDED → %s  "
                        "[candidate=%s  source=%s]",
                        _TRAINING_AGENT,
                        candidate["name"],
                        candidate["literature_source"],
                    )
        else:
            self.logger.info("No new feature candidates found.")
            # Still update last_run so the interval resets
            self._update_section(
                "literature",
                {"last_run": datetime.now(timezone.utc).isoformat()},
            )

        # ----------------------------------------------------------
        # Step 5: Render HTML digest (existing logic)
        # ----------------------------------------------------------
        digest_path = None
        if new_candidates and not dry_run:
            digest_path = self._render_digest(new_candidates)

        result = {
            "action": "literature_scout",
            "papers_fetched": len(all_papers),
            "papers_relevant": papers_processed,
            "new_candidates": len(new_candidates),
            "digest": digest_path,
            "messages_sent": len(new_candidates) if not dry_run else 0,
        }
        self._log_finish(result)
        return result

    # ------------------------------------------------------------------
    # NEW: emit FEATURE_CANDIDATE_ADDED
    # ------------------------------------------------------------------

    def _emit_candidate(self, candidate: dict) -> None:
        """
        Send a FEATURE_CANDIDATE_ADDED message to TrainingLifecycleAgent.

        does not require approval — TrainingLifecycle stores it for human
        review without acting on it automatically.
        """
        payload = {
            "candidate_name": candidate["name"],
            "literature_source": candidate["literature_source"],
            "pmid_or_doi": candidate.get("pmid_or_doi"),
            "paper_title": candidate.get("paper_title", ""),
            "relevance_score": candidate.get("relevance_score", 0.0),
            "extracted_at": candidate.get("extracted_at"),
        }
        self.send_message(
            to=_TRAINING_AGENT,
            subject=FEATURE_CANDIDATE_ADDED,
            payload=payload,
            priority=PRIORITY_NORMAL,
            requires_approval=False,
        )
        self.logger.info(
            "→ FEATURE_CANDIDATE_ADDED sent to %s  [candidate=%s]",
            _TRAINING_AGENT,
            candidate["name"],
        )

    # ------------------------------------------------------------------
    # Run interval check — unchanged
    # ------------------------------------------------------------------

    def _should_run_literature(self) -> bool:
        from datetime import timedelta

        section = self._get_section("literature")
        last_run = section.get("last_run")
        if not last_run:
            return True
        try:
            last_dt = datetime.fromisoformat(last_run)
            return (datetime.now(timezone.utc) - last_dt) >= timedelta(
                days=_LITERATURE_INTERVAL_DAYS
            )
        except ValueError:
            return True

    # ------------------------------------------------------------------
    # PubMed fetch — unchanged
    # ------------------------------------------------------------------

    def _fetch_pubmed(self) -> list[dict]:
        self.logger.info("Fetching PubMed papers …")
        papers: list[dict] = []
        try:
            import requests

            for query, max_results in LITERATURE_PUBMED_QUERIES:
                params: dict[str, Any] = {
                    "db": "pubmed",
                    "term": query,
                    "retmax": max_results,
                    "sort": "date",
                    "retmode": "json",
                }
                if NCBI_API_KEY:
                    params["api_key"] = NCBI_API_KEY

                resp = requests.get(
                    f"{NCBI_EUTILS_BASE}/esearch.fcgi",
                    params=params,
                    timeout=20,
                )
                resp.raise_for_status()
                ids = resp.json().get("esearchresult", {}).get("idlist", [])
                if not ids:
                    continue

                fetch_params: dict[str, Any] = {
                    "db": "pubmed",
                    "id": ",".join(ids),
                    "retmode": "xml",
                }
                if NCBI_API_KEY:
                    fetch_params["api_key"] = NCBI_API_KEY

                fetch_resp = requests.get(
                    f"{NCBI_EUTILS_BASE}/efetch.fcgi",
                    params=fetch_params,
                    timeout=30,
                )
                fetch_resp.raise_for_status()
                root = ET.fromstring(fetch_resp.content)

                for article in root.findall(".//PubmedArticle"):
                    pmid_el = article.find(".//PMID")
                    title_el = article.find(".//ArticleTitle")
                    abstract_el = article.find(".//AbstractText")
                    pmid = pmid_el.text if pmid_el is not None else ""
                    title = title_el.text if title_el is not None else ""
                    abstract = abstract_el.text if abstract_el is not None else ""
                    papers.append(
                        {
                            "source": "PubMed",
                            "pmid": pmid,
                            "title": title,
                            "abstract": abstract,
                            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                        }
                    )

            self.logger.info("PubMed: %d paper(s) fetched.", len(papers))
        except Exception as exc:
            self.logger.warning("PubMed fetch failed: %s", exc)
        return papers

    # ------------------------------------------------------------------
    # bioRxiv fetch — unchanged
    # ------------------------------------------------------------------

    def _fetch_biorxiv(self) -> list[dict]:
        self.logger.info("Fetching bioRxiv papers …")
        papers: list[dict] = []
        try:
            import feedparser

            for feed_url in BIORXIV_RSS_FEEDS:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:LITERATURE_MAX_PAPERS_PER_RUN]:
                    papers.append(
                        {
                            "source": "bioRxiv",
                            "doi": getattr(entry, "id", ""),
                            "title": getattr(entry, "title", ""),
                            "abstract": getattr(entry, "summary", ""),
                            "url": getattr(entry, "link", ""),
                        }
                    )
            self.logger.info("bioRxiv: %d paper(s) fetched.", len(papers))
        except Exception as exc:
            self.logger.warning("bioRxiv fetch failed: %s", exc)
        return papers

    # ------------------------------------------------------------------
    # ClinGen fetch — unchanged
    # ------------------------------------------------------------------

    def _fetch_clingen(self) -> list[dict]:
        self.logger.info("Fetching ClinGen gene validity data …")
        papers: list[dict] = []
        try:
            import requests

            resp = requests.get(
                f"{CLINGEN_API_BASE}.json" "?limit=20&sort=scoreDate&direction=DESC",
                timeout=20,
            )
            resp.raise_for_status()
            for record in resp.json().get("gene_validity_list", []):
                papers.append(
                    {
                        "source": "ClinGen",
                        "doi": record.get("uuid", ""),
                        "title": (
                            f"{record.get('gene', '')} — "
                            f"{record.get('disease', '')} "
                            f"({record.get('classification', '')})"
                        ),
                        "abstract": record.get("notes", ""),
                        "url": record.get("url", ""),
                    }
                )
            self.logger.info("ClinGen: %d record(s) fetched.", len(papers))
        except Exception as exc:
            self.logger.warning("ClinGen fetch failed: %s", exc)
        return papers

    # ------------------------------------------------------------------
    # Relevance scoring — unchanged
    # ------------------------------------------------------------------

    def _relevance_score(self, paper: dict) -> float:
        text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
        score = 0.0
        for kw in LITERATURE_RELEVANCE_KEYWORDS:
            kw_lower = kw.lower()
            score += text.count(kw_lower) * (
                0.3 if kw_lower in paper.get("title", "").lower() else 0.1
            )
        return min(score, 1.0)

    # ------------------------------------------------------------------
    # Feature candidate extraction — unchanged
    # ------------------------------------------------------------------

    def _extract_candidates(self, paper: dict) -> list[str]:
        text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
        candidates: list[str] = []
        for pattern in _COMPILED_PATTERNS:
            for match in pattern.finditer(text):
                try:
                    name = match.group("name").strip().rstrip(".,;:")
                except IndexError:
                    continue
                if 3 <= len(name) <= 60 and name not in candidates:
                    candidates.append(name)
        return candidates

    # ------------------------------------------------------------------
    # HTML digest rendering — unchanged
    # ------------------------------------------------------------------

    def _render_digest(self, candidates: list[dict]) -> str | None:
        try:
            report_dir = Path(LITERATURE_DIGEST_DIR)
            report_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            report_path = report_dir / f"literature_digest_{timestamp}.html"

            rows = "".join(
                f"<tr>"
                f"<td>{html.escape(c['name'])}</td>"
                f"<td>{html.escape(c.get('literature_source', ''))}</td>"
                f"<td><a href='{c.get('pmid_or_doi', '')}' target='_blank'>"
                f"{html.escape(c.get('paper_title', ''))[:80]}…</a></td>"
                f"<td>{c.get('relevance_score', 0):.2f}</td>"
                f"</tr>"
                for c in candidates
            )
            report_path.write_text(
                f"""<!DOCTYPE html><html><head>
<meta charset='utf-8'>
<title>Literature Digest {timestamp}</title>
<style>
  body {{font-family:sans-serif;padding:1rem}}
  table {{border-collapse:collapse;width:100%}}
  th,td {{border:1px solid #ccc;padding:6px 10px;text-align:left}}
  th {{background:#f0f0f0}}
</style></head><body>
<h2>Literature Digest — {timestamp}</h2>
<p>{len(candidates)} new feature candidate(s) surfaced.</p>
<table>
<tr><th>Candidate</th><th>Source</th><th>Paper</th><th>Relevance</th></tr>
{rows}
</table></body></html>""",
                encoding="utf-8",
            )
            self.logger.info("Literature digest written: %s", report_path)
            return str(report_path)

        except Exception as exc:
            self.logger.warning("Digest render failed: %s", exc)
            return None
