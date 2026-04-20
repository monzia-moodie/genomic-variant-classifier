"""
tests/unit/test_prediction_artifacts.py
========================================
Tests for src/evaluation/prediction_artifacts.py.

Focuses on the atomic-write, manifest-schema, and idempotency guarantees
that Rule 5 depends on. No GCS upload is exercised here; that's an
integration test.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.prediction_artifacts import RunArtifactWriter


@pytest.fixture
def writer(tmp_path: Path) -> RunArtifactWriter:
    return RunArtifactWriter(
        run_id="run_test",
        ablation="full",
        output_dir=tmp_path / "run_test" / "full",
    )


@pytest.fixture
def dummy_meta() -> pd.DataFrame:
    n = 25
    return pd.DataFrame(
        {
            "variant_id": [f"v{i}" for i in range(n)],
            "gene_symbol": [f"G{i%5}" for i in range(n)],
            "consequence": np.random.choice(
                ["missense_variant", "stop_gained", "splice_donor_variant"],
                n,
            ),
            "chrom": ["1"] * n,
            "pos": list(range(n)),
            "ref": ["A"] * n,
            "alt": ["T"] * n,
        }
    )


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
class TestManifest:

    def test_writes_and_parses(self, writer, tmp_path):
        writer.save_manifest(
            git_sha="abc123",
            versions={"python": "3.12.0", "torch": "2.11.0"},
            config={"seed": 42, "ablation": "full"},
        )
        path = writer.output_dir / "manifest.json"
        assert path.exists()
        payload = json.loads(path.read_text())
        assert payload["run_id"] == "run_test"
        assert payload["ablation"] == "full"
        assert payload["git_sha"] == "abc123"
        assert payload["versions"]["python"] == "3.12.0"

    def test_atomic_write_no_tmp_leftover(self, writer):
        writer.save_manifest("sha", {}, {})
        # No .tmp files should remain
        leftover = list(writer.output_dir.glob(".*.tmp"))
        assert not leftover, f"Stale tmp files: {leftover}"

    def test_second_write_overwrites_cleanly(self, writer):
        writer.save_manifest("sha1", {"python": "3.12"}, {})
        writer.save_manifest("sha2", {"python": "3.13"}, {})
        payload = json.loads((writer.output_dir / "manifest.json").read_text())
        assert payload["git_sha"] == "sha2"
        assert payload["versions"]["python"] == "3.13"


# ---------------------------------------------------------------------------
# Test predictions
# ---------------------------------------------------------------------------
class TestTestPredictions:

    def test_writes_parquet_with_correct_columns(self, writer, dummy_meta):
        n = len(dummy_meta)
        y = np.random.randint(0, 2, n)
        proba = np.random.random(n)
        base_probs = {
            "xgboost": np.random.random(n),
            "lightgbm": np.random.random(n),
        }
        writer.save_test_predictions(y, proba, base_probs, dummy_meta)
        df = pd.read_parquet(writer.output_dir / "test_predictions.parquet")
        assert "label" in df.columns
        assert "ensemble_prob" in df.columns
        assert "xgboost_prob" in df.columns
        assert "lightgbm_prob" in df.columns
        assert "variant_id" in df.columns
        assert len(df) == n

    def test_length_mismatch_raises(self, writer, dummy_meta):
        with pytest.raises(ValueError, match="len"):
            writer.save_test_predictions(
                y_test=np.zeros(5),
                proba=np.zeros(10),
                base_probs={},
                meta=dummy_meta,
            )


# ---------------------------------------------------------------------------
# OOF predictions
# ---------------------------------------------------------------------------
class TestOOFPredictions:

    def test_requires_variant_id_fold_label(self, writer):
        df_missing_fold = pd.DataFrame({"variant_id": ["v1"], "label": [1]})
        with pytest.raises(ValueError, match="fold"):
            writer.save_oof_predictions(df_missing_fold)

    def test_writes_with_required_cols(self, writer):
        df = pd.DataFrame(
            {
                "variant_id": ["v1", "v2"],
                "fold": [0, 1],
                "label": [1, 0],
                "xgboost_prob": [0.8, 0.2],
            }
        )
        writer.save_oof_predictions(df)
        loaded = pd.read_parquet(writer.output_dir / "oof_predictions.parquet")
        assert len(loaded) == 2
        assert set(loaded.columns) >= {"variant_id", "fold", "label", "xgboost_prob"}


# ---------------------------------------------------------------------------
# Graph stats
# ---------------------------------------------------------------------------
class TestGraphStats:

    def test_requires_node_and_edge_count(self, writer):
        with pytest.raises(ValueError, match="missing"):
            writer.save_graph_stats({"node_count": 100})

    def test_writes_valid_json(self, writer):
        writer.save_graph_stats(
            {
                "node_count": 19000,
                "edge_count": 850000,
                "median_degree": 42,
            }
        )
        payload = json.loads((writer.output_dir / "graph_stats.json").read_text())
        assert payload["edge_count"] == 850000


# ---------------------------------------------------------------------------
# Ablation aggregator
# ---------------------------------------------------------------------------
class TestAblationAggregator:

    def test_idempotent_append(self, writer, tmp_path):
        master = tmp_path / "ablation_results.parquet"
        writer.save_manifest("sha", {}, {})  # so writer has state

        writer.append_ablation_row(
            master,
            {
                "ablation": "full",
                "auroc": 0.986,
                "auprc": 0.94,
            },
        )
        writer.append_ablation_row(
            master,
            {
                "ablation": "full",
                "auroc": 0.987,
                "auprc": 0.95,  # replaces
            },
        )

        df = pd.read_parquet(master)
        rows_for_full = df[df["ablation"] == "full"]
        assert (
            len(rows_for_full) == 1
        ), "Re-run of same ablation should replace, not duplicate"
        assert rows_for_full.iloc[0]["auroc"] == pytest.approx(0.987)


# ---------------------------------------------------------------------------
# Artefact tracking
# ---------------------------------------------------------------------------
def test_artefact_list_tracks_writes(writer, dummy_meta):
    assert writer.artefacts == []
    writer.save_manifest("sha", {}, {})
    writer.save_graph_stats({"node_count": 1, "edge_count": 1})
    assert "manifest.json" in writer.artefacts
    assert "graph_stats.json" in writer.artefacts
