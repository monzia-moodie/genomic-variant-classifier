"""
run_agents.py
=============
Command-line entry point for the genomic variant classifier agent layer.

Examples
--------
# Dry-run the full pipeline (no mutations)
python run_agents.py --pipeline full --dry-run

# Run just the data freshness check
python run_agents.py --pipeline data_freshness

# Show current state snapshot
python run_agents.py --status

# List pending review items
python run_agents.py --reviews

# Resolve review item #0
python run_agents.py --resolve 0

# List unreviewed feature candidates
python run_agents.py --candidates

# Mark candidate #0 reviewed
python run_agents.py --mark-candidate 0
"""

import argparse
import json
import sys
from pathlib import Path

# --- Path setup: must happen before any project imports ---
_AGENT_LAYER = Path(__file__).resolve().parent
_AGENTS_DIR  = _AGENT_LAYER / "agents"
for _p in (_AGENT_LAYER, _AGENTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from orchestrator import Orchestrator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genomic Variant Classifier — Agent Layer CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--pipeline",
        choices=["data_freshness", "training", "interpretability", "literature", "full"],
        default=None,
        help="Run a named pipeline.",
    )
    parser.add_argument(
        "--candidates",
        action="store_true",
        help="List unreviewed feature candidates from the literature queue.",
    )
    parser.add_argument(
        "--mark-candidate",
        type=int,
        metavar="INDEX",
        default=None,
        help="Mark feature candidate INDEX as reviewed.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Describe actions without making any mutations.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print shared state snapshot and exit.",
    )
    parser.add_argument(
        "--reviews",
        action="store_true",
        help="List unresolved review items and exit.",
    )
    parser.add_argument(
        "--resolve",
        type=int,
        metavar="INDEX",
        default=None,
        help="Mark review item INDEX as resolved.",
    )

    args = parser.parse_args()
    orch = Orchestrator(dry_run=args.dry_run)

    if args.status:
        print(json.dumps(orch.status(), indent=2, default=str))
        return

    if args.candidates:
        items = orch.feature_candidates()
        if not items:
            print("No unreviewed feature candidates.")
        for i, c in enumerate(items):
            score = c.get("relevance_score", 0)
            print(f"\n[{i}]  {c.get('feature_name')}  (score={score:.2f})")
            print(f"     source : {c.get('source')} — {c.get('paper_title','')[:70]}")
            print(f"     context: {c.get('description','')[:100]}")
            print(f"     url    : {c.get('paper_url','')}")
        return

    if args.mark_candidate is not None:
        orch.mark_candidate_reviewed(args.mark_candidate)
        print(f"Candidate {args.mark_candidate} marked as reviewed.")
        return

    if args.reviews:
        items = orch.pending_reviews()
        if not items:
            print("No unresolved review items.")
        for i, item in enumerate(items):
            print(f"\n[{i}]  {item.get('reason')}")
            print(f"     added : {item.get('added_at')}")
            print(f"     agent : {item.get('agent')}")
            if "action_required" in item:
                print(f"     action: {item['action_required']}")
        return

    if args.resolve is not None:
        orch.resolve_review(args.resolve)
        print(f"Review item {args.resolve} resolved.")
        return

    if args.pipeline:
        results = orch.run_pipeline(args.pipeline)
        failures = [n for n, r in results.items() if not r.success]
        sys.exit(1 if failures else 0)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
