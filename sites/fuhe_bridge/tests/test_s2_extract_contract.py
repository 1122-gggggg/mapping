from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import s2_extract as s2  # noqa: E402
import s3_pairs as s3  # noqa: E402
from ts_common import BUILD  # noqa: E402


def test_hover_cap_backfills_structural_frames_without_losing_slots() -> None:
    seq = BUILD[0].seq
    target = s2.EXPECTED_FRAME_COUNTS[seq]
    planned = [(float(i), "hover" if i < 10 else "parallax") for i in range(target)]
    records = [
        {"t": i + 0.25, "motion_class": "low_parallax"}
        for i in range(target + 20)
    ]

    capped = s2.cap_hover_and_backfill(seq, planned, records)

    assert len(capped) == target
    assert len({round(t, 6) for t, _ in capped}) == target
    assert sum(cls == "hover" for _, cls in capped) / target <= s2.MAX_HOVER_RATIO
    assert any(cls == "low_parallax" for _, cls in capped)


def test_wrong_current_role_mapping_is_rejected() -> None:
    frames = [
        {"name": "a.jpg", "motion_class": "unproven", "role": "triangulation"},
        {"name": "b.jpg", "motion_class": "parallax", "role": "triangulation"},
    ]

    ok, roles, bad = s2.role_partition(frames)

    assert not ok
    assert roles == {"triangulation": 2}
    assert bad == ["a.jpg"]


def test_fuhe_probe_budget_and_natural_first_pair_contract_are_config_derived() -> None:
    assert s2.EXPECTED_FRAME_COUNTS == {video.seq: 48 for video in BUILD}
    assert sum(s2.EXPECTED_FRAME_COUNTS.values()) == 240

    direction = {video.seq: video.direction for video in BUILD}
    frame_records = {
        seq: [
            {"name": f"{seq}/{index + 1:06d}.jpg", "t": float(index)}
            for index in range(s2.EXPECTED_FRAME_COUNTS[seq])
        ]
        for seq in s2.EXPECTED_FRAME_COUNTS
    }
    fwd = sorted(seq for seq, value in direction.items() if value == "fwd")
    rev = sorted(seq for seq, value in direction.items() if value == "rev")
    policy = s3.decide_gap_bridge_policy(fwd, rev, None)
    pairs = s3.deterministic_gap_pairs(
        frame_records, policy["missing_sequence_pairs"], max_pairs=12_000
    )

    assert len(policy["expected_sequence_pairs"]) == 6
    assert policy["pending_sequence_pairs"] == policy["expected_sequence_pairs"]
    assert pairs == set()
