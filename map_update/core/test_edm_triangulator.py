from __future__ import annotations

from pathlib import Path

import pytest

from edm_triangulator import (
    frozen_pose_triangulate_argv,
    require_edm_triangulator,
)


def test_require_edm_triangulator_fails_closed_when_missing(tmp_path: Path):
    missing = tmp_path / "build_reloc_map_edm.py"
    with pytest.raises(SystemExit, match=str(missing)):
        require_edm_triangulator(missing)


def test_require_edm_triangulator_returns_existing_script(tmp_path: Path):
    script = tmp_path / "build_reloc_map_edm.py"
    script.write_text("# placeholder\n", encoding="utf-8")
    assert require_edm_triangulator(script) == script.resolve()


def test_frozen_pose_argv_invokes_stages_via_upstream_script(tmp_path: Path):
    script = tmp_path / "build_reloc_map_edm.py"
    script.write_text("# placeholder\n", encoding="utf-8")
    cmd = frozen_pose_triangulate_argv(
        tmp_path / "model",
        tmp_path / "images",
        tmp_path / "seed.pt",
        tmp_path / "work",
        tmp_path / "out.pt",
        triangulator=script,
        python="/usr/bin/python3",
    )
    assert cmd[:2] == ["/usr/bin/python3", str(script.resolve())]
    assert cmd[cmd.index("--model") + 1] == str(tmp_path / "model")
    assert cmd[cmd.index("--image-root") + 1] == str(tmp_path / "images")
    assert cmd[cmd.index("--in-bundle") + 1] == str(tmp_path / "seed.pt")
