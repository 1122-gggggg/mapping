from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path


TARGET_SITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TARGET_SITE / "tools"))

from audit_dg_graph import (  # noqa: E402
    LOFTR_SEQUENCE_ALLOWLIST,
    loftr_trigger_contract,
)
from audit_hotspot_repair_reachability import build_parser as reach_parser  # noqa: E402
from audit_map_geometry import (  # noqa: E402
    REQUIRED_GHOST_SEQUENCE_PAIRS,
    apply_official_hotspot_defaults,
    official_g61_hotspot_defaults,
)
from probe_hotspot_loftr import build_parser as loftr_parser  # noqa: E402
from repair_hotspot_tracks import repair_hotspot_tracks  # noqa: E402


def _authorized_trigger() -> dict:
    return loftr_trigger_contract(
        ("P1100110_005", "P1110111"),
        {"G5.1": True, "G5.7": True, "G6.1": False, "G6.3": True},
        ghost_check_id="G6.1",
        blocking_edges={
            ("P1090109_002", "P1110111"),
            ("P1100110_005", "P1110111"),
        },
    )


def _source_db(tmp_path: Path) -> Path:
    path = tmp_path / "source.db"
    sqlite3.connect(path).close()
    return path


def _reachable() -> dict:
    return {"status": "REACHABLE_FOR_EXACT_ROI_PROBE"}


def test_official_pair_sets_are_aligned() -> None:
    official = frozenset(tuple(sorted(edge)) for edge in REQUIRED_GHOST_SEQUENCE_PAIRS)
    allowlist = frozenset(tuple(sorted(edge)) for edge in LOFTR_SEQUENCE_ALLOWLIST)
    assert allowlist == official
    assert allowlist == REQUIRED_GHOST_SEQUENCE_PAIRS

    source, targets = official_g61_hotspot_defaults()
    derived = frozenset(tuple(sorted((source, target))) for target in targets)
    assert derived == official

    loftr = apply_official_hotspot_defaults(
        loftr_parser().parse_args(
            ["--model", "m", "--image-root", "i", "--weights", "w", "--out", "o"]
        )
    )
    reach = apply_official_hotspot_defaults(
        reach_parser().parse_args(
            ["--model", "m", "--geometry-gate", "g", "--out", "o"]
        )
    )
    assert loftr.source_sequence == source
    assert set(loftr.target_sequence) == set(targets)
    assert reach.source_sequence == source
    assert set(reach.target_sequence) == set(targets)


def test_missing_reachability_does_not_write(tmp_path: Path) -> None:
    dest = tmp_path / "dest"
    source = _source_db(tmp_path)

    report = repair_hotspot_tracks(
        source_db=source,
        dest_dir=dest,
        reachability=None,
        trigger=_authorized_trigger(),
        geometry=None,
        matches=[{"edge": ["P1100110_005", "P1110111"], "pairs": [["a", "b"]]}],
    )

    assert report["preflight"]["ok"] is False
    assert report["database_modified"] is False
    assert report["promotion_allowed"] is False
    assert not dest.exists()
    assert source.is_file()


def test_unreachable_reachability_does_not_write(tmp_path: Path) -> None:
    dest = tmp_path / "dest"
    source = _source_db(tmp_path)

    report = repair_hotspot_tracks(
        source_db=source,
        dest_dir=dest,
        reachability={"status": "UNREACHABLE"},
        trigger=_authorized_trigger(),
        geometry=None,
        matches=[{"edge": ["P1100110_005", "P1110111"], "pairs": [["a", "b"]]}],
    )

    assert report["preflight"]["ok"] is False
    assert report["database_modified"] is False
    assert report["promotion_allowed"] is False
    assert not dest.exists()


def test_promotion_stays_false_without_pass_geometry_gate(tmp_path: Path) -> None:
    dest = tmp_path / "dest"
    source = _source_db(tmp_path)

    report = repair_hotspot_tracks(
        source_db=source,
        dest_dir=dest,
        reachability=_reachable(),
        trigger=_authorized_trigger(),
        geometry=None,
        matches=[{"edge": ["P1100110_005", "P1110111"], "pairs": [["a", "b"]]}],
    )

    assert report["preflight"]["ok"] is True
    assert report["database_modified"] is True
    assert report["promotion_allowed"] is False
    assert (dest / source.name).is_file()
    assert source.stat().st_mtime_ns == (tmp_path / "source.db").stat().st_mtime_ns


def test_promotion_allowed_only_with_pass_geometry_gate(tmp_path: Path) -> None:
    dest = tmp_path / "dest"
    source = _source_db(tmp_path)
    gate = dest / "gates" / "S5_7_S6_geometry.json"
    gate.parent.mkdir(parents=True)
    gate.write_text(
        json.dumps(
            {
                "schema_version": "sfm-gate-v2",
                "stage": "S5_7_S6_geometry",
                "status": "PASS",
                "ok": True,
                "checks": [{"id": "G6.1", "state": "PASS", "ok": True}],
            }
        ),
        encoding="utf-8",
    )

    report = repair_hotspot_tracks(
        source_db=source,
        dest_dir=dest,
        reachability=_reachable(),
        trigger=_authorized_trigger(),
        geometry=None,
        matches=[{"edge": ["P1090109_002", "P1110111"], "pairs": []}],
    )

    assert report["database_modified"] is True
    assert report["promotion_allowed"] is True


def test_p112_matches_are_not_injected(tmp_path: Path) -> None:
    dest = tmp_path / "dest"
    source = _source_db(tmp_path)

    report = repair_hotspot_tracks(
        source_db=source,
        dest_dir=dest,
        reachability=_reachable(),
        trigger=_authorized_trigger(),
        geometry=None,
        matches=[
            {"edge": ["P1120112", "P1140114"], "pairs": [["x", "y"]]},
            {"edge": ["P1100110_005", "P1110111"], "pairs": [["a", "b"]]},
        ],
    )

    assert "P1120112|P1140114" not in report["injected_edges"]
    assert report["injected_edges"] == ["P1100110_005|P1110111"]
