"""
agents/literature_scout_agent.py
==================================
Literature Scout Agent — monitors PubMed, bioRxiv, and ClinGen for new
research relevant to the genomic variant classifier, extracts feature
candidates, and produces a browsable HTML digest.

Run schedule
------------
Designed to run weekly (not after every training update).
The Orchestrator uses a time-based condition: run if it's been ≥ 7 days
since the last scout run.

Pipeline
--------
1. Determine lookback window (days since last run, minimum 7)
2. Search PubMed — run all LITERATURE_PUBMED_QUERIES
3. Parse bioRxiv RSS feeds
4. Fetch ClinGen gene validity updates
5. Deduplicate against already-seen paper IDs (stored in SharedState)
6. Score relevance of each new paper
7. Filter to papers above LITERATURE_MIN_RELEVANCE
8. Extract feature/tool candidates from high-scoring abstracts
9. Deduplicate candidates against LITERATURE_KNOWN_TOOLS and existing queue
10. Write HTML digest to LITERATURE_DIGEST_DIR
11. Update SharedState:
      literature_last_run, literature_seen_ids (append),
      feature_candidates (extend), literature_digest_path

SharedState keys written
------------------------
  literature_last_run      ISO timestamp
  literature_seen_ids      set of paper IDs (stored as list for JSON compat)
  literature_digest_path   path to latest HTML digest
  feature_candidates       extended with new candidates
"""

from __future__ import annotations

import json
import logging
import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

_AL = Path(__file__).resolve().parent.parent
for _p in (str(_AL), str(_AL / "agents")):
    if _p not in sys.path: sys.path.insert(0, _p)

from base_agent import AgentResult, BaseAgent
from literature_utils import (
    Paper,
    extract_feature_candidates,
    fetch_biorxiv_rss,
    fetch_clingen_updates,
    fetch_pubmed_details,
    generate_digest_html,
    score_relevance,
    search_pubmed,
)
from config import (
    BIORXIV_RSS_FEEDS,
    CLINGEN_API_BASE,
    GDRIVE_CHECKPOINT_DIR,
    LITERATURE_CANDIDATE_MIN_SCORE,
    LITERATURE_DIGEST_DIR,
    LITERATURE_FEATURE_PATTERNS,
    LITERATURE_KNOWN_TOOLS,
    LITERATURE_MAX_PAPERS_PER_RUN,
    LITERATURE_MIN_RELEVANCE,
    LITERATURE_PUBMED_QUERIES,
    LITERATURE_RELEVANCE_KEYWORDS,
    LOVD_GENES_OF_INTEREST,
    NCBI_API_KEY,
)
from shared_state import SharedState

log = logging.getLogger("LiteratureScoutAgent")

# Minimum days between full scout runs (avoid redundant weekly re-runs)
_MIN_RUN_INTERVAL_DAYS = 6


class LiteratureScoutAgent(BaseAgent):
    """
    Monitors genomics literature for new features, scoring methods,
    and variant reclassifications relevant to the classifier.
    """

    def __init__(self, state: SharedState, dry_run: bool = False):
        super().__init__(state, dry_run)
        LITERATURE_DIGEST_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        return "LiteratureScoutAgent"

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(self) -> AgentResult:
        results: dict[str, Any] = {
            "papers_found":       0,
            "papers_new":         0,
            "papers_relevant":    0,
            "candidates_added":   0,
            "digest_path":        None,
            "clingen_updates":    0,
            "errors":             [],
        }

        # ---- 1. Determine lookback window -------------------------------
        since_days = self._days_since_last_run()
        self.log.info("Scout lookback window: %d days", since_days)

        if since_days < _MIN_RUN_INTERVAL_DAYS:
            self.log.info(
                "Last run was %d days ago (< %d minimum) — skipping.",
                since_days, _MIN_RUN_INTERVAL_DAYS,
            )
            return AgentResult(
                success=True, action="skip_too_recent",
                details={**results, "since_days": since_days},
            )

        # ---- 2. PubMed --------------------------------------------------
        pubmed_papers = self._search_pubmed_all(since_days)
        results["papers_found"] += len(pubmed_papers)

        # ---- 3. bioRxiv -------------------------------------------------
        biorxiv_papers = self._fetch_biorxiv(since_days)
        results["papers_found"] += len(biorxiv_papers)

        # ---- 4. ClinGen -------------------------------------------------
        clingen_papers: list[Paper] = []
        try:
            clingen_papers = fetch_clingen_updates(
                LOVD_GENES_OF_INTEREST, since_days, CLINGEN_API_BASE
            )
            results["clingen_updates"] = len(clingen_papers)
        except Exception as exc:
            msg = f"ClinGen fetch failed: {exc}"
            self.log.warning(msg)
            results["errors"].append(msg)

        # ---- 5. Deduplicate against seen IDs ----------------------------
        seen_ids: set[str] = set(self.state.get("literature_seen_ids") or [])
        all_candidate_papers = pubmed_papers + biorxiv_papers

        new_papers = [p for p in all_candidate_papers
                      if p.paper_id not in seen_ids]
        results["papers_new"] = len(new_papers)
        self.log.info(
            "Papers: found=%d  new=%d  (dedup'd %d seen)",
            results["papers_found"], len(new_papers),
            len(all_candidate_papers) - len(new_papers),
        )

        # Cap total processing load
        new_papers = new_papers[:LITERATURE_MAX_PAPERS_PER_RUN]

        # ---- 6. Score relevance -----------------------------------------
        for paper in new_papers:
            paper.relevance_score = score_relevance(
                paper, LITERATURE_RELEVANCE_KEYWORDS
            )

        # ---- 7. Filter to relevant papers --------------------------------
        relevant_papers = [
            p for p in new_papers
            if p.relevance_score >= LITERATURE_MIN_RELEVANCE
        ]
        results["papers_relevant"] = len(relevant_papers)
        self.log.info("Relevant papers (score ≥ %.2f): %d",
                      LITERATURE_MIN_RELEVANCE, len(relevant_papers))

        # ---- 8. Extract feature candidates ------------------------------
        existing_queue = {
            c.get("feature_name", "").lower()
            for c in (self.state.get("feature_candidates") or [])
        }
        all_candidates: list[dict] = []

        for paper in relevant_papers:
            candidates = extract_feature_candidates(
                paper,
                patterns=LITERATURE_FEATURE_PATTERNS,
                known_tools=LITERATURE_KNOWN_TOOLS | existing_queue,
                min_score=LITERATURE_CANDIDATE_MIN_SCORE,
            )
            paper.feature_candidates = candidates
            for c in candidates:
                if c["feature_name"].lower() not in existing_queue:
                    all_candidates.append(c)
                    existing_queue.add(c["feature_name"].lower())

        results["candidates_added"] = len(all_candidates)
        self.log.info("Feature candidates extracted: %d", len(all_candidates))

        # ---- 9. Generate digest -----------------------------------------
        if self.dry_run:
            self._dry_run_log(
                f"Would write digest with {len(relevant_papers)} papers "
                f"and {len(all_candidates)} candidates."
            )
        else:
            try:
                run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                digest_path = LITERATURE_DIGEST_DIR / f"digest_{run_date}.html"
                generate_digest_html(
                    new_papers=relevant_papers,
                    all_candidates=all_candidates,
                    clingen_papers=clingen_papers,
                    run_metadata={
                        "run_date":   run_date,
                        "since_days": since_days,
                    },
                    output_path=digest_path,
                )
                results["digest_path"] = str(digest_path)
                self._mirror_digest(digest_path)
            except Exception as exc:
                msg = f"Digest generation failed: {exc}"
                self.log.error("%s\n%s", msg, traceback.format_exc())
                results["errors"].append(msg)

        # ---- 10. Update SharedState -------------------------------------
        self._update_state(
            new_papers=new_papers,
            new_candidates=all_candidates,
            digest_path=results.get("digest_path"),
        )

        # Notify if high-value candidates found
        if all_candidates:
            self._flag_candidates(all_candidates)

        success = len(results["errors"]) == 0
        return AgentResult(
            success=success,
            action="scout",
            details=results,
            errors=results["errors"],
        )

    # ------------------------------------------------------------------
    # Source fetchers
    # ------------------------------------------------------------------

    def _search_pubmed_all(self, since_days: int) -> list[Paper]:
        """Run all configured PubMed queries, collect unique PMIDs, fetch details."""
        all_pmids: set[str] = set()
        for query, max_n in LITERATURE_PUBMED_QUERIES:
            pmids = search_pubmed(
                query=query,
                max_results=max_n,
                since_days=since_days,
                api_key=NCBI_API_KEY,
            )
            all_pmids.update(pmids)
            if len(all_pmids) >= LITERATURE_MAX_PAPERS_PER_RUN:
                break

        # Fetch details in batches of 20 (NCBI recommendation)
        pmid_list = list(all_pmids)[:LITERATURE_MAX_PAPERS_PER_RUN]
        papers: list[Paper] = []
        for i in range(0, len(pmid_list), 20):
            batch = pmid_list[i:i + 20]
            papers.extend(
                fetch_pubmed_details(batch, api_key=NCBI_API_KEY)
            )

        self.log.info("PubMed: %d unique papers fetched.", len(papers))
        return papers

    def _fetch_biorxiv(self, since_days: int) -> list[Paper]:
        """Fetch and merge papers from all configured bioRxiv RSS feeds."""
        papers: list[Paper] = []
        seen: set[str] = set()
        for url in BIORXIV_RSS_FEEDS:
            for p in fetch_biorxiv_rss(url, since_days):
                if p.paper_id not in seen:
                    papers.append(p)
                    seen.add(p.paper_id)
        return papers

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _days_since_last_run(self) -> int:
        last = self.state.get("literature_last_run")
        if not last:
            return 365   # never run — use a full year lookback
        try:
            dt = datetime.fromisoformat(last)
            return max(0, (datetime.now(timezone.utc) - dt).days)
        except (ValueError, TypeError):
            return 30

    def _update_state(
        self,
        new_papers:     list[Paper],
        new_candidates: list[dict],
        digest_path:    str | None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.state.set("literature_last_run", now)

        if digest_path:
            self.state.set("literature_digest_path", digest_path)

        # Extend seen IDs (cap at 5000 to keep state file manageable)
        seen_ids: list[str] = self.state.get("literature_seen_ids") or []
        seen_ids.extend(p.paper_id for p in new_papers)
        self.state.set("literature_seen_ids", seen_ids[-5000:])

        # Extend feature candidate queue
        if new_candidates:
            existing = self.state.get("feature_candidates") or []
            combined = existing + new_candidates
            # Cap at 200 unreviewed candidates
            unreviewed = [c for c in combined if not c.get("reviewed")]
            reviewed   = [c for c in combined if c.get("reviewed")]
            self.state.set(
                "feature_candidates",
                reviewed + unreviewed[-200:],
            )

        self.log.info(
            "State updated: last_run=%s  new_ids=%d  new_candidates=%d",
            now[:10], len(new_papers), len(new_candidates),
        )

    def _flag_candidates(self, candidates: list[dict]) -> None:
        """
        Add a single aggregated pending-review item for the batch of
        new candidates, rather than one item per candidate (avoids flooding
        the review queue).
        """
        names = [c["feature_name"] for c in candidates[:10]]
        self.state.add_pending_review({
            "reason":          "New feature/tool candidates from literature",
            "n_candidates":    len(candidates),
            "top_candidates":  names,
            "agent":           self.name,
            "action_required": (
                f"{len(candidates)} potential new features/tools were identified "
                "in recent literature. Review the digest and decide which (if any) "
                "should be incorporated into the feature engineering pipeline. "
                "Run `python run_agents.py --reviews` to see the full list."
            ),
        })

    # ------------------------------------------------------------------
    # GDrive mirror
    # ------------------------------------------------------------------

    def _mirror_digest(self, digest_path: Path) -> None:
        try:
            gdrive_lit = GDRIVE_CHECKPOINT_DIR.parent / "reports" / "literature"
            if gdrive_lit.parent.parent.exists():
                import shutil
                gdrive_lit.mkdir(parents=True, exist_ok=True)
                shutil.copy2(digest_path, gdrive_lit / digest_path.name)
                self.log.info("Digest mirrored to GDrive → %s", gdrive_lit)
        except Exception as exc:
            self.log.warning("GDrive mirror failed: %s", exc)
