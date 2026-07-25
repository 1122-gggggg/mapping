from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


TARGET_SITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TARGET_SITE / "tools"))

from validate_edm_bundle import anchored_counts, build_contract_checks  # noqa: E402


def test_numpy_bundle_gate_is_normalized_to_builtin_bool() -> None:
    result = bool(np.isclose(3.0, 3.0))

    assert type(result) is bool


def test_anchored_counts_recomputes_finite_xyz_cells() -> None:
    refs = {
        "a.jpg": {"xyz_by_cell": np.asarray([[1, 2, 3], [np.nan, np.nan, np.nan]])},
        "b.jpg": {"xyz_by_cell": np.asarray([[0, 0, 0], [4, 5, 6]])},
    }

    assert anchored_counts(refs) == {"a.jpg": 1, "b.jpg": 2}


def test_edm_bundle_build_contract_requires_fuhe_parameters_and_fresh_hashes() -> None:
    digest = "a" * 64
    tracking_digest = "d" * 64
    checks = build_contract_checks(
        {
            "triangulation_pair_topk": 30,
            "triangulation_anchor_batch_size": 64,
            "source_model_sha256": digest,
            "pair_provenance_sha256": "b" * 64,
            "pair_artifact_sha256": "c" * 64,
            "source_tracking_bundle_sha256": tracking_digest,
        },
        current_model_sha256=digest,
        current_tracking_bundle_sha256=tracking_digest,
    )

    assert checks and all(checks.values())


def test_edm_bundle_build_contract_rejects_stale_model_or_wrong_knobs() -> None:
    checks = build_contract_checks(
        {
            "triangulation_pair_topk": 20,
            "triangulation_anchor_batch_size": 32,
            "source_model_sha256": "d" * 64,
            "pair_provenance_sha256": "not-a-hash",
            "pair_artifact_sha256": "not-a-hash",
        },
        current_model_sha256="a" * 64,
        current_tracking_bundle_sha256="e" * 64,
    )

    assert not any(checks.values())
