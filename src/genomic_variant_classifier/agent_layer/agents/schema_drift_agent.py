from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from logging import Logger
from pathlib import Path
from typing import Optional

import pandas as pd
import pandera.pandas as pa


@dataclass(frozen=True)
class SchemaDriftResult:
    timestamp: str
    expected_schema_hash: str
    observed_schema_hash: str
    columns_added: tuple[str, ...]
    columns_removed: tuple[str, ...]
    columns_dtype_changed: tuple[tuple[str, str, str], ...]  # (col, expected, observed)
    pandera_violations: tuple[str, ...]
    severity: str  # green | red (no amber — schema drift is binary)


@dataclass
class SchemaDriftAgent:
    """Pandera-based schema-contract enforcement on the feature matrix.

    Acts as a hard gate at the Spark ETL boundary. Schema drift is treated
    as red severity: any change halts ETL and emits a hypothesis stub.
    """

    schema: pa.DataFrameSchema
    expected_dtypes: dict[str, str]
    expected_schema_hash: str
    output_dir: Path
    logger: Optional[Logger] = field(default=None, repr=False)

    @staticmethod
    def hash_schema(dtypes: dict[str, str]) -> str:
        canonical = json.dumps(sorted(dtypes.items()), separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def detect(self, df: pd.DataFrame) -> SchemaDriftResult:
        observed_dtypes = {c: str(df[c].dtype) for c in df.columns}
        observed_hash = self.hash_schema(observed_dtypes)
        expected_cols = set(self.expected_dtypes)
        observed_cols = set(observed_dtypes)
        added = tuple(sorted(observed_cols - expected_cols))
        removed = tuple(sorted(expected_cols - observed_cols))
        changed: list[tuple[str, str, str]] = []
        for col in expected_cols & observed_cols:
            if observed_dtypes[col] != self.expected_dtypes[col]:
                changed.append((col, self.expected_dtypes[col], observed_dtypes[col]))
        violations: list[str] = []
        try:
            self.schema.validate(df, lazy=True)
        except pa.errors.SchemaErrors as exc:  # pragma: no cover - exercised in tests
            violations = [str(e) for e in exc.failure_cases.itertuples(index=False)]
        clean = (
            observed_hash == self.expected_schema_hash
            and not added
            and not removed
            and not changed
            and not violations
        )
        return SchemaDriftResult(
            timestamp=datetime.now(timezone.utc).isoformat(),
            expected_schema_hash=self.expected_schema_hash,
            observed_schema_hash=observed_hash,
            columns_added=added,
            columns_removed=removed,
            columns_dtype_changed=tuple(changed),
            pandera_violations=tuple(violations),
            severity="green" if clean else "red",
        )

    def persist(self, result: SchemaDriftResult, run_id: str) -> Path:
        out = self.output_dir / "schema" / f"{run_id}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(result.__dict__, default=str, indent=2), encoding="utf-8"
        )
        return out
