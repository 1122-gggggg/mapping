from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from stability_scores import build_ref_stability, rerank_indices_by_stability


def test_recent_observation_raises_decayed_ref_score():
    names = ["base/a.jpg", "base/b.jpg"]
    obs = {
        "sessions": {
            "P200": {
                "top_base_ref_hits": [
                    {"key": "base/b.jpg", "count": 50},
                ]
            }
        }
    }

    scores, meta = build_ref_stability(
        names,
        prior_names=names,
        prior_scores=np.ones(2, dtype=np.float32),
        observation_stats=obs,
        report_rows=[],
        half_life_sessions=4.0,
    )

    assert meta["half_life_sessions"] == 4.0
    assert scores[1] > scores[0]
    assert scores[0] < 1.0


def test_new_submap_with_warning_gets_conservative_score():
    names = ["P1240124/frame_000001.jpg", "P1250125/frame_000001.jpg"]
    report = [
        {
            "seq": "P1240124",
            "route": "submap",
            "status": "ok",
            "bridges": 60,
            "bridge_geometry": 19,
            "bridge_median_inlier_ratio": 0.37,
        },
        {
            "seq": "P1250125",
            "route": "submap",
            "status": "retrieval_high_but_inliers_low",
            "bridges": 18,
            "bridge_geometry": 2,
            "bridge_median_inlier_ratio": 0.65,
        },
    ]

    scores, _meta = build_ref_stability(
        names,
        prior_names=[],
        prior_scores=[],
        observation_stats={},
        report_rows=report,
    )

    assert scores[0] > 1.0
    assert scores[1] < scores[0]
    assert scores[1] < 1.0


def test_rerank_uses_stability_as_mild_tie_breaker():
    sims = np.array([0.900, 0.899, 0.700], dtype=np.float32)
    stability = np.array([0.2, 2.0, 10.0], dtype=np.float32)

    ranked = rerank_indices_by_stability(
        sims,
        stability,
        topk=2,
        candidate_multiplier=2,
        stability_weight=0.05,
    )

    assert ranked.tolist() == [1, 0]
