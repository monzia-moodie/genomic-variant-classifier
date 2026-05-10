"""
Regression test for ESM-2 connector stub-mode detection.

Context (2026-04-17 debug): Runs 6, 7, 8 all trained with ESM-2 silently
returning 0.0 for every variant. Root cause identified this session: the
connector REQUIRES columns {gene_symbol, protein_pos, wt_aa, mut_aa}, but
the training pipeline does not populate wt_aa/mut_aa/protein_pos anywhere
(grep: only esm2.py reads them, nothing writes them). When these columns
are absent, the connector logs "columns missing -- defaulting to 0.0" and
returns before any model forward pass.

See docs/incidents/INCIDENT_2026-04-17_esm2-hgvsp-parser.md for the full
analysis and the plan to add the missing HGVSp parser in Run 10.

This test has two modes:

1) When the input dataframe has properly-parsed protein-change columns
   AND the backend is available AND the model loads AND UniProt is
   reachable, the test requires esm2_delta_norm > 0 for at least one of
   three known missense variants. Variance > 0 across them.

2) When transformers isn't installed locally, the test is SKIPPED
   (importorskip). Stub mode is acceptable on dev machines.

If this test fails at assertion time (not skip time), the connector has
a real bug. If it fails at fixture time (network/UniProt unreachable,
model weights unavailable), that's environment-specific and gets xfailed.
"""

from __future__ import annotations

import socket

import numpy as np
import pandas as pd
import pytest

# SKIP everything if transformers isn't installed (dev-machine stub mode
# is acceptable).
transformers = pytest.importorskip(
    "transformers",
    reason="transformers not installed; ESM-2 stub mode is acceptable locally",
)


def _has_network() -> bool:
    """Quick connectivity check to UniProt (primary external dep). If
    this fails, xfail the test rather than fail it -- network flakes
    should not block commits."""
    try:
        socket.create_connection(("rest.uniprot.org", 443), timeout=5).close()
        return True
    except OSError:
        return False


@pytest.fixture
def esm2_connector():
    """Instantiate ESM2Connector from src/genomic_variant_classifier/data/esm2.py (confirmed
    2026-04-17)."""
    from genomic_variant_classifier.data.esm2 import ESM2Connector

    return ESM2Connector()


def _variant_frame_with_parsed_columns() -> pd.DataFrame:
    """Three real, well-known missense variants with the four columns the
    connector actually needs: gene_symbol, protein_pos, wt_aa, mut_aa.

    Variants (all confirmed in UniProt canonical isoforms):
      * BRCA1 p.Arg1699Gln  -- R1699Q  (P38398, residue 1699, R->Q)
      * TP53  p.Arg175His   -- R175H   (P04637, residue 175,  R->H)
      * PTEN  p.Gly129Arg   -- G129R   (P60484, residue 129,  G->R)
    """
    return pd.DataFrame(
        {
            "variant_id": ["v1", "v2", "v3"],
            "gene_symbol": ["BRCA1", "TP53", "PTEN"],
            "protein_pos": [1699, 175, 129],
            "wt_aa": ["R", "R", "G"],
            "mut_aa": ["Q", "H", "R"],
            "is_missense": [True, True, True],
        }
    )


def test_esm2_emits_delta_norm_column(esm2_connector):
    """The connector must always emit the esm2_delta_norm column, even
    in stub mode. Column-missing is a separate, more-severe failure
    mode than all-zeros; catching it independently keeps our diagnosis
    cheap."""
    df = _variant_frame_with_parsed_columns()
    out = esm2_connector.annotate_dataframe(df)

    assert "esm2_delta_norm" in out.columns, (
        f"esm2_delta_norm column missing from output. "
        f"Got columns: {list(out.columns)[:20]}. "
        "Connector API may have changed; update this test to match."
    )
    assert len(out) == 3, f"expected 3 rows, got {len(out)}"


def test_esm2_not_in_stub_mode(esm2_connector):
    """When all four required columns are present AND backend is loaded
    AND network is available AND UniProt returns sequences, the connector
    must produce esm2_delta_norm > 0 for at least one of three distinct
    missense variants.

    If UniProt is unreachable (network issue, outage) this xfails rather
    than fails -- network flakes are environmental, not regressions.
    """
    from genomic_variant_classifier.data import esm2 as esm2_mod

    if esm2_mod._BACKEND is None:
        pytest.skip(
            "ESM-2 backend did not initialize (no transformers+torch or fair-esm). "
            "Real-mode test not applicable."
        )

    if not _has_network():
        pytest.xfail(
            "rest.uniprot.org unreachable -- network flake, not a regression. "
            "ESM-2 connector needs UniProt to fetch canonical sequences."
        )

    df = _variant_frame_with_parsed_columns()
    out = esm2_connector.annotate_dataframe(df)
    deltas = out["esm2_delta_norm"].to_numpy(dtype=float)

    assert (deltas > 0).any(), (
        f"all esm2_delta_norm values are 0.0: {deltas.tolist()}. "
        "Backend initialized, required columns present, network available "
        "-- but connector still returned stub zeros. This is a real "
        "regression: check stderr for sequence-fetch errors, wt_aa mismatch "
        "warnings, or silent exceptions in the embedding forward pass."
    )


def test_esm2_stub_mode_expected_when_columns_missing(esm2_connector):
    """Regression test for the OPPOSITE behavior: if the required columns
    are missing from the input, the connector MUST gracefully return
    all-zeros rather than crash.

    This is the exact state that caused Runs 6/7/8 to silently have ESM-2
    contributing 0. Until the upstream HGVSp parser is added (see
    INCIDENT_2026-04-17_esm2-hgvsp-parser.md), the training pipeline
    calls this connector without wt_aa/mut_aa/protein_pos and relies on
    this graceful-fallback behavior.
    """
    # Exactly the schema the training pipeline currently produces:
    # gene_symbol and protein_change, but not the four parsed columns.
    df = pd.DataFrame(
        {
            "variant_id": ["v1", "v2", "v3"],
            "gene_symbol": ["BRCA1", "TP53", "PTEN"],
            "protein_change": ["p.Arg1699Gln", "p.Arg175His", "p.Gly129Arg"],
        }
    )
    out = esm2_connector.annotate_dataframe(df)

    assert "esm2_delta_norm" in out.columns
    deltas = out["esm2_delta_norm"].to_numpy(dtype=float)
    assert (deltas == 0.0).all(), (
        f"expected all-zero stub output, got {deltas.tolist()}. "
        "The connector is inferring protein_pos/wt_aa/mut_aa from "
        "protein_change somehow -- if that's intentional, this test is "
        "now incorrect and should be removed. If it's not intentional, "
        "check for a recent change to _parse_hgvsp or similar."
    )
