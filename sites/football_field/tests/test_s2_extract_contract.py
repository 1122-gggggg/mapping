from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import s2_extract as s2  # noqa: E402
import s3_pairs as s3  # noqa: E402
from ts_common import BUILD, read_json  # noqa: E402


def test_hover_cap_backfills_structural_frames_without_losing_slots() -> None:
    seq = "S02_BA"
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


def test_current_production_manifest_replay_preserves_1414_and_6000_contract() -> None:
    run_dir = ROOT / "runs" / "target_site_v1"
    motion = read_json(run_dir / "motion_manifest.json")
    planned = {video.seq: s2.plan_frames(video.seq, motion) for video in BUILD}

    assert {seq: len(frames) for seq, frames in planned.items()} == s2.EXPECTED_FRAME_COUNTS
    assert sum(map(len, planned.values())) == 1414

    direction = {video.seq: video.direction for video in BUILD}
    direction.update(
        {seq: record["direction"] for seq, record in motion["directions"].items()}
    )
    frame_records = {
        seq: [
            {"name": f"{seq}/{index + 1:06d}.jpg", "t": timestamp}
            for index, (timestamp, _motion_class) in enumerate(frames)
        ]
        for seq, frames in planned.items()
    }
    pairs: set[tuple[str, str]] = set()
    for seq_a in sorted(seq for seq, value in direction.items() if value == "fwd"):
        sampled_a = s3.sample(
            frame_records[seq_a], s3.FWD_STRIDE, s3.ENDPOINT_STRIDE, s3.ENDPOINT_FRAC
        )
        for seq_b in sorted(seq for seq, value in direction.items() if value == "rev"):
            sampled_b = s3.sample(
                frame_records[seq_b], s3.REV_STRIDE, s3.ENDPOINT_STRIDE, s3.ENDPOINT_FRAC
            )
            pairs.update(
                tuple(sorted((frame_a["name"], frame_b["name"])))
                for frame_a in sampled_a
                for frame_b in sampled_b
            )

    assert len(pairs) == 6000
