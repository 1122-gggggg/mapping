from pathlib import Path

import numpy as np

from sfm_diagnosis.diagnose import DiagnosisCode, diagnose_pose
from sfm_diagnosis.matchability import (
    MatchabilityConfig,
    build_landmark_matchability,
    load_landmark_matchability,
    query_matchability,
    save_landmark_matchability,
)
from sfm_diagnosis.models import Pose
from test_diagnose import healthy_map


def _dead_events(map_data, n_obs: int = 20, n_inlier: int = 0) -> list[dict]:
    events = []
    for point_id in map_data.point_ids.tolist():
        for k in range(n_obs):
            events.append(
                {
                    "query_id": f"q{k}",
                    "point_id": int(point_id),
                    "observed": True,
                    "inlier": k < n_inlier,
                }
            )
    return events


def test_zero_inliers_are_beta_smoothed_and_not_exactly_zero():
    m = healthy_map()
    table = build_landmark_matchability(m, _dead_events(m))
    assert len(table.point_ids) == m.num_points
    assert np.all(table.p > 0.0)
    assert np.all(table.p < 0.1)
    expected = (0 + 1.0) / (20 + 1.0 + 1.0)
    assert np.allclose(table.p, expected)


def test_matchability_does_not_change_track_ba_fim():
    m = healthy_map()
    pose = Pose(np.zeros(3), np.eye(3))
    healthy = diagnose_pose(m, pose)
    table = build_landmark_matchability(m, _dead_events(m))
    weak = diagnose_pose(m, pose, matchability=table)
    assert healthy.primary == DiagnosisCode.HEALTHY
    assert weak.primary == DiagnosisCode.LANDMARK_MATCHABILITY_WEAK
    assert DiagnosisCode.GEOMETRY_WEAK not in weak.codes
    assert healthy.fim.lambda_min == weak.fim.lambda_min
    assert weak.matchability_fim is None


def test_optional_match_fim_is_a_second_matrix():
    m = healthy_map()
    pose = Pose(np.zeros(3), np.eye(3))
    table = build_landmark_matchability(m, _dead_events(m))
    d = diagnose_pose(
        m,
        pose,
        matchability=table,
        matchability_config=MatchabilityConfig(reweight_fim=True),
    )
    assert d.matchability_fim is not None
    assert d.matchability_fim.lambda_min < d.fim.lambda_min


def test_unevidenced_landmarks_do_not_pull_mean_down():
    m = healthy_map()
    live = int(m.point_ids[0])
    events = [
        {"query_id": "q0", "point_id": live, "observed": True, "inlier": True}
        for _ in range(10)
    ]
    events.append({"query_id": "q0", "point_id": 999999, "observed": True, "inlier": False})
    table = build_landmark_matchability(m, events)
    assert 999999 not in set(table.point_ids.tolist())
    metrics = query_matchability(
        table, Pose(np.zeros(3), np.eye(3)), np.arange(m.num_points), m
    )
    assert metrics.mean_matchability is not None
    assert metrics.mean_matchability > 0.8
    assert metrics.evidenced_visible_fraction < 0.1


def test_half_life_prefers_recent_inliers():
    m = healthy_map()
    point_id = int(m.point_ids[0])
    events = [
        {"query_id": "old", "point_id": point_id, "observed": True, "inlier": False, "timestamp": 0.0}
        for _ in range(20)
    ] + [
        {"query_id": "new", "point_id": point_id, "observed": True, "inlier": True, "timestamp": 10.0}
        for _ in range(20)
    ]
    plain = build_landmark_matchability(m, events)
    decayed = build_landmark_matchability(
        m, events, config=MatchabilityConfig(half_life=2.0)
    )
    assert float(decayed.p[0]) > float(plain.p[0])


def test_matchability_csv_roundtrip(tmp_path: Path):
    m = healthy_map()
    table = build_landmark_matchability(m, _dead_events(m, n_obs=8, n_inlier=2))
    path = save_landmark_matchability(tmp_path, table)
    loaded = load_landmark_matchability(path)
    assert np.array_equal(loaded.point_ids, table.point_ids)
    assert np.allclose(loaded.p, table.p)
    d = diagnose_pose(
        m, Pose(np.zeros(3), np.eye(3)), matchability=loaded
    )
    assert d.matchability.mean_matchability == np.mean(table.p)
