from __future__ import annotations

import json
from pathlib import Path

import diagnose_update


def test_register_pnp_only_is_advisory_ok() -> None:
    report = diagnose_update.diagnose_rows(
        [{"seq": "P1", "route": "register", "status": "ok", "points_added": 0}]
    )
    assert report["ok"] is True
    assert report["G-U1"] == "NOT_APPLICABLE"
    assert report["rows"][0]["labels"] == ["REGISTER_PNP_ONLY"]
    assert report["labels"] == ["REGISTER_PNP_ONLY"]


def test_register_points_contract_broken() -> None:
    report = diagnose_update.diagnose_rows(
        [{"seq": "P1", "route": "register", "points_added": 12}]
    )
    assert report["ok"] is False
    assert report["rows"][0]["labels"] == [
        "REGISTER_PNP_ONLY",
        "REGISTER_POINTS_CONTRACT_BROKEN",
    ]


def test_unimplemented_tile_routes() -> None:
    for route in ("changed-region", "changed", "tile_replace"):
        report = diagnose_update.diagnose_rows([{"seq": "P1", "route": route}])
        assert report["ok"] is False, route
        assert report["rows"][0]["labels"] == ["UNIMPLEMENTED_TILE"], route


def test_needs_tile_replace_status_unimplements_tile() -> None:
    report = diagnose_update.diagnose_rows(
        [{"seq": "P1", "route": "observation-only", "status": "needs_tile_replace"}]
    )
    assert report["ok"] is False
    assert report["rows"][0]["labels"] == ["UNIMPLEMENTED_TILE"]


def test_register_plus_needs_tile_replace_keeps_both_labels() -> None:
    report = diagnose_update.diagnose_rows(
        [
            {
                "seq": "P1",
                "route": "register",
                "status": "needs_tile_replace",
                "points_added": 0,
            }
        ]
    )
    assert report["ok"] is False
    assert report["rows"][0]["labels"] == ["REGISTER_PNP_ONLY", "UNIMPLEMENTED_TILE"]


def test_submap_not_incremental_does_not_fail() -> None:
    report = diagnose_update.diagnose_rows(
        [{"seq": "P1", "route": "submap", "points_added": 40}]
    )
    assert report["ok"] is True
    assert report["rows"][0]["labels"] == ["SUBMAP_NOT_INCREMENTAL"]


def test_after_model_requires_g_u1() -> None:
    empty = diagnose_update.diagnose_rows([])
    assert empty["G-U1"] == "NOT_APPLICABLE"
    assert empty["ok"] is True
    report = diagnose_update.diagnose_rows([], after_model=Path("after"))
    assert report["G-U1"] == "REQUIRES_G_U1"
    assert report["ok"] is True


def test_cli_ok_writes_json_and_exits_zero(tmp_path: Path) -> None:
    summary = tmp_path / "map_update_summary.json"
    summary.write_text(
        json.dumps({"rows": [{"seq": "P1", "route": "register", "points_added": 0}]}),
        encoding="utf-8",
    )
    output = tmp_path / "honesty.json"
    rc = diagnose_update.main(["--summary", str(summary), "--output", str(output)])
    assert rc == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["G-U1"] == "NOT_APPLICABLE"
    assert payload["rows"][0]["labels"] == ["REGISTER_PNP_ONLY"]


def test_cli_unimplemented_tile_exits_one(tmp_path: Path) -> None:
    summary = tmp_path / "map_update_summary.json"
    summary.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "seq": "P2000200",
                        "route": "changed-region",
                        "status": "needs_tile_replace",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "nested" / "honesty.json"
    rc = diagnose_update.main(
        [
            "--summary",
            str(summary),
            "--output",
            str(output),
            "--after-model",
            str(tmp_path / "after"),
        ]
    )
    assert rc == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["G-U1"] == "REQUIRES_G_U1"
    assert payload["labels"] == ["UNIMPLEMENTED_TILE"]


def test_cli_register_points_broken_exits_one(tmp_path: Path) -> None:
    summary = tmp_path / "map_update_summary.json"
    summary.write_text(
        json.dumps({"rows": [{"seq": "P1", "route": "register", "points_added": 3}]}),
        encoding="utf-8",
    )
    output = tmp_path / "honesty.json"
    rc = diagnose_update.main(["--summary", str(summary), "--output", str(output)])
    assert rc == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert "REGISTER_POINTS_CONTRACT_BROKEN" in payload["labels"]
