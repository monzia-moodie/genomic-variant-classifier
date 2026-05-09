"""
agents/literature_utils.py
===========================
Utility functions for the LiteratureScoutAgent.

Covers
------
PubMed        — NCBI E-utilities (esearch + efetch XML)
bioRxiv       — RSS feed parsing
ClinGen       — Gene validity API
Relevance     — keyword-weighted scoring of abstracts
Feature extraction — regex + heuristic mining for novel tools/features
HTML digest   — self-contained report in the project dark theme
"""

from __future__ import annotations

import logging
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests

log = logging.getLogger(__name__)

# Throttle: seconds between NCBI requests when no API key is set
_NCBI_THROTTLE = 0.34   # ≈ 3 req/s
_last_ncbi_call: float = 0.0


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class Paper:
    """Lightweight container for a retrieved paper."""

    __slots__ = (
        "source", "paper_id", "title", "abstract",
        "authors", "journal", "pub_date", "url",
        "relevance_score", "feature_candidates",
    )

    def __init__(
        self,
        source:     str,
        paper_id:   str,
        title:      str,
        abstract:   str = "",
        authors:    list[str] | None = None,
        journal:    str = "",
        pub_date:   str = "",
        url:        str = "",
    ):
        self.source           = source
        self.paper_id         = paper_id
        self.title            = title
        self.abstract         = abstract
        self.authors          = authors or []
        self.journal          = journal
        self.pub_date         = pub_date
        self.url              = url
        self.relevance_score: float      = 0.0
        self.feature_candidates: list[dict] = []

    def to_dict(self) -> dict:
        return {
            "source":            self.source,
            "paper_id":          self.paper_id,
            "title":             self.title,
            "abstract":          self.abstract[:500],
            "authors":           self.authors[:3],
            "journal":           self.journal,
            "pub_date":          self.pub_date,
            "url":               self.url,
            "relevance_score":   self.relevance_score,
            "feature_candidates": self.feature_candidates,
        }


# ---------------------------------------------------------------------------
# PubMed
# ---------------------------------------------------------------------------

def search_pubmed(
    query:       str,
    max_results: int,
    since_days:  int,
    api_key:     str | None = None,
) -> list[str]:
    """
    Run an esearch query and return a list of PMIDs.
    Applies a date filter of the last `since_days` days.
    """
    _throttle_ncbi(api_key)
    since_date = (datetime.now(timezone.utc) - timedelta(days=since_days)
                  ).strftime("%Y/%m/%d")

    params: dict[str, Any] = {
        "db":       "pubmed",
        "term":     f"{query} AND {since_date}[PDAT]:3000/01/01[PDAT]",
        "retmax":   max_results,
        "retmode":  "json",
        "sort":     "relevance",
    }
    if api_key:
        params["api_key"] = api_key

    try:
        resp = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params=params, timeout=20,
        )
        resp.raise_for_status()
        ids = resp.json().get("esearchresult", {}).get("idlist", [])
        log.info("PubMed esearch '%s…': %d results", query[:50], len(ids))
        return ids
    except requests.RequestException as exc:
        log.warning("PubMed esearch failed: %s", exc)
        return []


def fetch_pubmed_details(pmids: list[str], api_key: str | None = None) -> list[Paper]:
    """
    Fetch title, abstract, authors, journal, and date for a list of PMIDs
    via efetch (XML format).  Returns a list of Paper objects.
    """
    if not pmids:
        return []

    _throttle_ncbi(api_key)
    params: dict[str, Any] = {
        "db":       "pubmed",
        "id":       ",".join(pmids),
        "retmode":  "xml",
        "rettype":  "abstract",
    }
    if api_key:
        params["api_key"] = api_key

    try:
        resp = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params=params, timeout=30,
        )
        resp.raise_for_status()
        return _parse_pubmed_xml(resp.text)
    except requests.RequestException as exc:
        log.warning("PubMed efetch failed: %s", exc)
        return []


def _parse_pubmed_xml(xml_text: str) -> list[Paper]:
    papers = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        log.warning("PubMed XML parse error: %s", exc)
        return []

    for article in root.findall(".//PubmedArticle"):
        try:
            pmid = _xml_text(article, ".//PMID") or ""
            title = _xml_text(article, ".//ArticleTitle") or ""
            abstract_parts = [
                node.text or ""
                for node in article.findall(".//AbstractText")
            ]
            abstract = " ".join(abstract_parts).strip()

            authors = []
            for author in article.findall(".//Author")[:5]:
                last  = _xml_text(author, "LastName") or ""
                first = _xml_text(author, "ForeName") or ""
                if last:
                    authors.append(f"{last} {first}".strip())

            journal = _xml_text(article, ".//Journal/Title") or \
                      _xml_text(article, ".//ISOAbbreviation") or ""

            pub_date = _extract_pubmed_date(article)

            papers.append(Paper(
                source    = "pubmed",
                paper_id  = f"PMID:{pmid}",
                title     = title,
                abstract  = abstract,
                authors   = authors,
                journal   = journal,
                pub_date  = pub_date,
                url       = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            ))
        except Exception as exc:
            log.debug("Failed to parse PubMed article: %s", exc)

    return papers


def _extract_pubmed_date(article) -> str:
    for tag in ("PubDate", "ArticleDate", "DateCompleted"):
        node = article.find(f".//{tag}")
        if node is not None:
            year  = _xml_text(node, "Year") or ""
            month = _xml_text(node, "Month") or "01"
            day   = _xml_text(node, "Day") or "01"
            if year:
                return f"{year}-{month[:3]}-{day.zfill(2)}"
    return ""


def _xml_text(node, path: str) -> str | None:
    el = node.find(path)
    return el.text.strip() if el is not None and el.text else None


def _throttle_ncbi(api_key: str | None) -> None:
    global _last_ncbi_call
    gap = 0.10 if api_key else _NCBI_THROTTLE
    elapsed = time.time() - _last_ncbi_call
    if elapsed < gap:
        time.sleep(gap - elapsed)
    _last_ncbi_call = time.time()


# ---------------------------------------------------------------------------
# bioRxiv RSS
# ---------------------------------------------------------------------------

def fetch_biorxiv_rss(feed_url: str, since_days: int) -> list[Paper]:
    """
    Parse a bioRxiv RSS feed, filtering to entries posted within `since_days`.
    """
    try:
        resp = requests.get(feed_url, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as exc:
        log.warning("bioRxiv RSS fetch failed (%s): %s", feed_url, exc)
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(days=since_days)
    papers = []

    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError as exc:
        log.warning("bioRxiv RSS parse error: %s", exc)
        return []

    # Handle both RSS 2.0 (<item>) and Atom (<entry>) layouts
    items = root.findall(".//item") or root.findall(
        ".//{http://www.w3.org/2005/Atom}entry"
    )

    for item in items:
        try:
            title    = (_xml_text(item, "title") or
                        _xml_text(item, "{http://www.w3.org/2005/Atom}title") or "")
            link     = (_xml_text(item, "link") or
                        _xml_text(item, "{http://www.w3.org/2005/Atom}id") or "")
            abstract = (_xml_text(item, "description") or
                        _xml_text(item, "{http://www.w3.org/2005/Atom}summary") or "")
            pub_date = (_xml_text(item, "pubDate") or
                        _xml_text(item, "{http://www.w3.org/2005/Atom}published") or "")

            # Strip HTML tags from abstract
            abstract = re.sub(r"<[^>]+>", " ", abstract).strip()

            # Best-effort date parse for cutoff filtering
            if pub_date:
                try:
                    # RFC 822 format typical in RSS
                    dt = datetime.strptime(pub_date[:25], "%a, %d %b %Y %H:%M:%S")
                    dt = dt.replace(tzinfo=timezone.utc)
                    if dt < cutoff:
                        continue
                except ValueError:
                    pass   # can't parse date — include anyway

            doi = link.split("abs/")[-1] if "abs/" in link else link

            papers.append(Paper(
                source   = "biorxiv",
                paper_id = f"DOI:{doi}",
                title    = title,
                abstract = abstract,
                pub_date = pub_date,
                url      = link,
            ))
        except Exception as exc:
            log.debug("bioRxiv item parse error: %s", exc)

    log.info("bioRxiv RSS '%s…': %d entries (within %d days)",
             feed_url[:60], len(papers), since_days)
    return papers


# ---------------------------------------------------------------------------
# ClinGen
# ---------------------------------------------------------------------------

def fetch_clingen_updates(
    genes:       list[str],
    since_days:  int,
    api_base:    str,
) -> list[Paper]:
    """
    Retrieve recently updated gene validity records from ClinGen.
    Returns Paper objects with the curation summary as the abstract.
    """
    since_date = (datetime.now(timezone.utc) - timedelta(days=since_days)
                  ).strftime("%Y-%m-%d")
    papers = []

    for gene in genes:
        try:
            resp = requests.get(
                api_base,
                params={"search": gene, "limit": 5},
                timeout=15,
                headers={"Accept": "application/json"},
            )
            if resp.status_code == 404:
                continue
            resp.raise_for_status()

            data = resp.json()
            entries = data if isinstance(data, list) else data.get("rows", [])

            for entry in entries:
                updated = entry.get("date_last_evaluated") or ""
                # Simple string comparison on ISO dates is correct here
                if updated and updated < since_date:
                    continue

                title = (
                    f"ClinGen: {gene} — "
                    f"{entry.get('disease_label','unknown disease')} "
                    f"({entry.get('classification','unknown classification')})"
                )
                abstract = (
                    f"Gene: {entry.get('gene_symbol', gene)}. "
                    f"Disease: {entry.get('disease_label', '')}. "
                    f"Classification: {entry.get('classification', '')}. "
                    f"Evaluated: {updated}. "
                    f"Evidence: {entry.get('evidence_summary', '')[:300]}"
                )
                papers.append(Paper(
                    source   = "clingen",
                    paper_id = f"CLINGEN:{entry.get('uuid', gene)}",
                    title    = title,
                    abstract = abstract,
                    pub_date = updated,
                    url      = f"https://search.clinicalgenome.org/kb/gene-validity/{entry.get('uuid','')}",
                ))

        except requests.RequestException as exc:
            log.warning("ClinGen fetch failed for %s: %s", gene, exc)

    log.info("ClinGen: %d updates for %d genes", len(papers), len(genes))
    return papers


# ---------------------------------------------------------------------------
# Relevance scoring
# ---------------------------------------------------------------------------

def score_relevance(
    paper:    Paper,
    keywords: list[str],
) -> float:
    """
    Score a paper's relevance on [0, 1] using keyword frequency in
    title (3×) + abstract (1×).  Normalised by keyword count.
    """
    if not keywords:
        return 0.0

    text = (paper.title + " ").lower() * 3 + paper.abstract.lower()
    matches = sum(1 for kw in keywords if kw.lower() in text)
    score = min(1.0, matches / max(len(keywords) * 0.25, 1))

    # Boost for very recent papers (within 14 days)
    try:
        pub = datetime.fromisoformat(paper.pub_date[:10])
        age = (datetime.now() - pub).days
        if age <= 14:
            score = min(1.0, score + 0.10)
        elif age <= 30:
            score = min(1.0, score + 0.05)
    except (ValueError, TypeError):
        pass

    return round(score, 4)


# ---------------------------------------------------------------------------
# Feature candidate extraction
# ---------------------------------------------------------------------------

def extract_feature_candidates(
    paper:          Paper,
    patterns:       list[str],
    known_tools:    set[str],
    min_score:      float,
) -> list[dict]:
    """
    Mine a paper's title + abstract for novel feature/tool names using
    the configured regex patterns, then filter out already-known tools.

    Each returned candidate dict:
        feature_name, description, paper_id, paper_title, source,
        relevance_score, added_at
    """
    if paper.relevance_score < min_score:
        return []

    text      = paper.title + ". " + paper.abstract
    found:    set[str] = set()
    compiled  = [re.compile(p, re.IGNORECASE) for p in patterns]

    for pat in compiled:
        for m in pat.finditer(text):
            name = m.group("n") if "n" in pat.groupindex else ""
            if name:
                found.add(name)

    candidates = []
    for name in found:
        # Skip if it matches a known tool (case-insensitive)
        if name.lower() in known_tools:
            continue
        # Skip very short or all-digit names
        if len(name) < 3 or name.isdigit():
            continue
        # Skip common English words that happen to match
        if name.lower() in {
            "the", "and", "for", "with", "from", "using", "based",
            "this", "that", "these", "our", "their", "here", "also",
            "both", "each", "such", "thus",
        }:
            continue

        # Extract a short context sentence around the match
        idx = text.lower().find(name.lower())
        start = max(0, idx - 60)
        end   = min(len(text), idx + len(name) + 80)
        context = text[start:end].strip()

        candidates.append({
            "feature_name":    name,
            "description":     context,
            "paper_id":        paper.paper_id,
            "paper_title":     paper.title,
            "paper_url":       paper.url,
            "source":          paper.source,
            "relevance_score": paper.relevance_score,
            "added_at":        datetime.now(timezone.utc).isoformat(),
            "reviewed":        False,
        })

    return candidates


# ---------------------------------------------------------------------------
# HTML digest generation
# ---------------------------------------------------------------------------

def generate_digest_html(
    new_papers:     list[Paper],
    all_candidates: list[dict],
    clingen_papers: list[Paper],
    run_metadata:   dict,
    output_path:    Path,
) -> Path:
    """
    Write a self-contained HTML digest report.
    """
    # Section: feature candidates
    cand_rows = "".join(
        f"""<tr>
          <td><code>{c['feature_name']}</code></td>
          <td style="font-size:10px;color:#8b949e;">{c['description'][:120]}…</td>
          <td style="font-size:10px;">{c['paper_title'][:80]}</td>
          <td><a href="{c['paper_url']}" style="color:#58a6ff;">link</a></td>
          <td style="color:{'#2ea043' if c['relevance_score']>0.6 else '#d29922'}">
            {c['relevance_score']:.2f}</td>
        </tr>"""
        for c in all_candidates[:40]
    )

    # Section: top relevant papers
    paper_rows = "".join(
        f"""<tr>
          <td style="font-size:10px;">{p.pub_date[:10]}</td>
          <td><a href="{p.url}" style="color:#58a6ff;font-size:11px;">{p.title[:100]}</a></td>
          <td style="font-size:10px;color:#8b949e;">{p.journal[:40] or p.source}</td>
          <td style="font-size:10px;color:#8b949e;">{', '.join(p.authors[:2])}</td>
          <td style="color:{'#2ea043' if p.relevance_score>0.6 else '#d29922'}">
            {p.relevance_score:.2f}</td>
        </tr>"""
        for p in sorted(new_papers, key=lambda x: x.relevance_score, reverse=True)[:30]
    )

    # Section: ClinGen updates
    clingen_rows = "".join(
        f"""<tr>
          <td style="font-size:11px;">{p.pub_date[:10]}</td>
          <td style="font-size:11px;">{p.title}</td>
          <td><a href="{p.url}" style="color:#58a6ff;font-size:10px;">view</a></td>
        </tr>"""
        for p in clingen_papers
    )

    src_breakdown = {
        "PubMed":   sum(1 for p in new_papers if p.source == "pubmed"),
        "bioRxiv":  sum(1 for p in new_papers if p.source == "biorxiv"),
        "ClinGen":  len(clingen_papers),
    }
    breakdown_html = "".join(
        f'<div class="meta-item"><div class="label">{k}</div>'
        f'<div class="value">{v}</div></div>'
        for k, v in src_breakdown.items()
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Literature Scout Digest — {run_metadata.get('run_date','')}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'IBM Plex Mono','Courier New',monospace;
    background: #0d1117; color: #c9d1d9;
    padding: 32px 40px; max-width: 1100px; margin: 0 auto;
  }}
  h1 {{ font-size: 22px; color: #e6edf3; font-weight: 400;
        border-bottom: 1px solid #21262d; padding-bottom: 12px; margin-bottom: 24px; }}
  h2 {{ font-size: 12px; color: #e6edf3; font-weight: 600; letter-spacing: 2px;
        text-transform: uppercase; margin: 32px 0 12px; }}
  .meta {{ display: flex; gap: 24px; flex-wrap: wrap; margin-bottom: 28px; }}
  .meta-item {{ background: #161b22; border: 1px solid #21262d;
                border-radius: 6px; padding: 10px 16px; }}
  .meta-item .label {{ color: #8b949e; font-size: 9px; letter-spacing: 1px;
                       text-transform: uppercase; margin-bottom: 4px; }}
  .meta-item .value {{ color: #e6edf3; font-size: 14px; font-weight: 600; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 11px; margin-top: 8px; }}
  th {{ background: #161b22; color: #8b949e; text-align: left;
        padding: 8px 12px; border-bottom: 1px solid #21262d;
        font-size: 9px; letter-spacing: 1px; text-transform: uppercase; }}
  td {{ padding: 7px 12px; border-bottom: 1px solid #161b22; vertical-align: top; }}
  tr:hover td {{ background: #161b22; }}
  .empty {{ color: #3d444d; font-size: 12px; padding: 16px 0; }}
  footer {{ margin-top: 48px; font-size: 10px; color: #3d444d;
            border-top: 1px solid #21262d; padding-top: 16px; }}
  a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
<h1>Literature Scout Digest</h1>
<div class="meta">
  <div class="meta-item">
    <div class="label">Run date</div>
    <div class="value">{run_metadata.get('run_date','—')}</div>
  </div>
  <div class="meta-item">
    <div class="label">New papers</div>
    <div class="value">{len(new_papers)}</div>
  </div>
  <div class="meta-item">
    <div class="label">Feature candidates</div>
    <div class="value" style="color:{'#58a6ff' if all_candidates else '#3d444d'}">
      {len(all_candidates)}</div>
  </div>
  <div class="meta-item">
    <div class="label">Lookback window</div>
    <div class="value">{run_metadata.get('since_days','—')}d</div>
  </div>
  {breakdown_html}
</div>

<h2>Feature &amp; Tool Candidates</h2>
{'<table><thead><tr><th>Name</th><th>Context</th><th>Paper</th><th>URL</th><th>Score</th></tr></thead><tbody>' + cand_rows + '</tbody></table>' if all_candidates else '<p class="empty">No novel candidates extracted this cycle.</p>'}

<h2>Top Relevant Papers</h2>
{'<table><thead><tr><th>Date</th><th>Title</th><th>Journal / Source</th><th>Authors</th><th>Score</th></tr></thead><tbody>' + paper_rows + '</tbody></table>' if new_papers else '<p class="empty">No new papers found.</p>'}

<h2>ClinGen Gene Validity Updates</h2>
{'<table><thead><tr><th>Date</th><th>Summary</th><th>Link</th></tr></thead><tbody>' + clingen_rows + '</tbody></table>' if clingen_papers else '<p class="empty">No ClinGen updates this cycle.</p>'}

<footer>Genomic Variant Classifier · Literature Scout Agent · auto-generated</footer>
</body></html>""", encoding="utf-8")

    log.info("Digest written → %s", output_path)
    return output_path
