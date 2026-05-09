from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from logging import Logger
from pathlib import Path
from typing import Optional

import pandas as pd
from river import drift as river_drift


@dataclass(frozen=True)
class AnnotationDriftResult:
    timestamp: str
    new_svi_publications: tuple[str, ...]
    pct_variants_with_review_status_change: float
    submitters_with_outlier_alarm: tuple[str, ...]
    severity: str
    requires_variant_scientist_review: bool


@dataclass
class AnnotationPolicyAgent:
    """Detects ACMG/AMP labeling-policy drift and ClinVar review-status flux.

    References:
      - Harrison et al., Genet Med 2017, 19:1096 — discordance resolution.
      - Yang et al., Genet Med 2017, 19:1118 — outlier rates by submitter type.
      - SVI Working Group: Biesecker & Harrison 2018 (PP5/BP6); Abou Tayoun
        et al., Hum Mutat 2018 (PVS1 refinement); Pejaver et al., AJHG 2022
        (PP3/BP4 calibration); Walker et al., AJHG 2023 (splicing).
    """

    output_dir: Path
    review_status_amber: float = 0.005  # 0.5% of weekly inferences
    review_status_red: float = 0.01
    submitter_outlier_threshold: float = 0.036  # Yang et al. literature-only baseline
    page_hinkley_delta: float = 0.005
    page_hinkley_lambda: float = 50.0
    logger: Optional[Logger] = field(default=None, repr=False)

    def _scan_submitter_rates(self, submitter_history: pd.DataFrame) -> tuple[str, ...]:
        """Run Page-Hinkley per submitter on rolling outlier rate."""
        flagged: list[str] = []
        for submitter, group in submitter_history.groupby("submitter_id", sort=False):
            ph = river_drift.PageHinkley(
                delta=self.page_hinkley_delta, threshold=self.page_hinkley_lambda
            )
            for rate in group.sort_values("date")["outlier_rate"].to_list():
                ph.update(float(rate))
                if ph.drift_detected:
                    flagged.append(str(submitter))
                    break
        return tuple(sorted(set(flagged)))

    def detect(
        self,
        new_svi_pubs: list[str],
        clinvar_status_changes: pd.DataFrame,
        submitter_history: pd.DataFrame,
        n_inference_variants: int,
    ) -> AnnotationDriftResult:
        pct_changed = float(len(clinvar_status_changes)) / max(n_inference_variants, 1)
        submitter_flags = self._scan_submitter_rates(submitter_history)
        if new_svi_pubs or pct_changed >= self.review_status_red:
            severity = "red"
        elif pct_changed >= self.review_status_amber or submitter_flags:
            severity = "amber"
        else:
            severity = "green"
        return AnnotationDriftResult(
            timestamp=datetime.now(timezone.utc).isoformat(),
            new_svi_publications=tuple(new_svi_pubs),
            pct_variants_with_review_status_change=pct_changed,
            submitters_with_outlier_alarm=submitter_flags,
            severity=severity,
            requires_variant_scientist_review=bool(new_svi_pubs) or severity == "red",
        )
