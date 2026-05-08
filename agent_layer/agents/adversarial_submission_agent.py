from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from logging import Logger
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class AdversarialFinding:
    rule_id: str
    submitter_id: str
    detail: str
    severity: str


@dataclass(frozen=True)
class AdversarialResult:
    timestamp: str
    findings: tuple[AdversarialFinding, ...]
    quarantine_submitter_ids: tuple[str, ...]
    severity: str


@dataclass
class AdversarialSubmissionAgent:
    """Detects ClinVar submission poisoning patterns.

    Threat model and statistics drawn from:
      - Yang et al., Genet Med 2017, 19:1118 — submitter outlier rates.
      - Deans et al., F1000Research 2018 — Clinotator submitter weighting.
      - Sahu et al., "Poisoning the Genome", arXiv:2603.27465 (2026).
    """

    output_dir: Path
    bulk_multiplier: float = 10.0
    bulk_absolute_floor: int = 500
    outlier_rate_amber: float = 0.036  # Yang et al. literature-only baseline
    flip_rate_red: float = 0.05
    flip_absolute_floor: int = 10
    coordination_jaccard: float = 0.7
    coordination_min_shared: int = 50
    new_submitter_age_days: int = 90
    new_submitter_plp_red: int = 100
    logger: Optional[Logger] = field(default=None, repr=False)

    def detect(
        self,
        weekly_submissions: pd.DataFrame,
        submitter_baseline: pd.DataFrame,
        aggregate_classifications: pd.DataFrame,
        submitter_metadata: pd.DataFrame,
    ) -> AdversarialResult:
        """Run all 7 rules on a single weekly ClinVar release."""
        findings: list[AdversarialFinding] = []
        quarantine: set[str] = set()

        # R1 bulk submission spike
        counts = weekly_submissions.groupby("submitter_id", sort=False).size()
        for submitter, count in counts.items():
            base = float(
                submitter_baseline.set_index("submitter_id")["median_24h"].get(
                    submitter, 0.0
                )
            )
            if count > self.bulk_absolute_floor and (
                base == 0 or count > self.bulk_multiplier * base
            ):
                findings.append(
                    AdversarialFinding(
                        "R1", str(submitter), f"count={count}, base={base}", "red"
                    )
                )
                quarantine.add(str(submitter))

        # R2 anomalous outlier rate
        merged = weekly_submissions.merge(
            aggregate_classifications,
            on="variant_id",
            how="left",
            suffixes=("", "_agg"),
        )
        merged["is_outlier"] = merged["classification"] != merged["classification_agg"]
        outlier_rate = merged.groupby("submitter_id", sort=False)["is_outlier"].mean()
        for submitter, rate in outlier_rate.items():
            if rate > self.outlier_rate_amber:
                findings.append(
                    AdversarialFinding(
                        "R2", str(submitter), f"outlier_rate={rate:.3f}", "amber"
                    )
                )

        # R4 pathogenic↔benign flip detection
        merged["is_flip"] = (
            (merged["classification"].isin(("Pathogenic", "Likely_pathogenic")))
            & (merged["classification_agg"].isin(("Benign", "Likely_benign")))
        ) | (
            (merged["classification"].isin(("Benign", "Likely_benign")))
            & (merged["classification_agg"].isin(("Pathogenic", "Likely_pathogenic")))
        )
        flips = merged.groupby("submitter_id", sort=False)["is_flip"].agg(
            ["mean", "sum"]
        )
        for submitter, row in flips.iterrows():
            if (
                row["sum"] >= self.flip_absolute_floor
                and row["mean"] >= self.flip_rate_red
            ):
                findings.append(
                    AdversarialFinding(
                        "R4",
                        str(submitter),
                        f"flip_rate={row['mean']:.3f}, n={int(row['sum'])}",
                        "red",
                    )
                )
                quarantine.add(str(submitter))

        # R5 coordinated IDs (Jaccard on variant sets)
        variant_sets = {
            sid: set(g["variant_id"])
            for sid, g in weekly_submissions.groupby("submitter_id", sort=False)
        }
        for a, b in combinations(sorted(variant_sets), 2):
            inter = variant_sets[a] & variant_sets[b]
            if len(inter) < self.coordination_min_shared:
                continue
            jac = len(inter) / max(len(variant_sets[a] | variant_sets[b]), 1)
            if jac >= self.coordination_jaccard:
                findings.append(
                    AdversarialFinding(
                        "R5",
                        f"{a}+{b}",
                        f"jaccard={jac:.3f}, shared={len(inter)}",
                        "red",
                    )
                )
                quarantine.update({str(a), str(b)})

        # R6 cold-start new submitter
        meta = submitter_metadata.set_index("submitter_id")
        plp_counts = (
            weekly_submissions[
                weekly_submissions["classification"].isin(
                    ("Pathogenic", "Likely_pathogenic")
                )
            ]
            .groupby("submitter_id", sort=False)
            .size()
        )
        for submitter, plp_n in plp_counts.items():
            age = meta["age_days"].get(submitter, 999_999)
            if (
                age < self.new_submitter_age_days
                and plp_n >= self.new_submitter_plp_red
            ):
                findings.append(
                    AdversarialFinding(
                        "R6", str(submitter), f"age={age}d, plp={int(plp_n)}", "amber"
                    )
                )

        if any(f.severity == "red" for f in findings):
            severity = "red"
        elif findings:
            severity = "amber"
        else:
            severity = "green"
        return AdversarialResult(
            timestamp=datetime.now(timezone.utc).isoformat(),
            findings=tuple(findings),
            quarantine_submitter_ids=tuple(sorted(quarantine)),
            severity=severity,
        )
