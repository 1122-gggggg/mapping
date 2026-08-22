from __future__ import annotations

from pathlib import Path

import pytest

from update_pipeline import (
    CORE_DIR,
    DEFAULT_BASE,
    DEFAULT_BASE_CACHE,
    DEFAULT_IMAGES,
    DEFAULT_MODEL,
    EVAL_CORE,
    LOC_ROOT,
    PREPARE,
    SPARSIFY,
    UPDATE_TOOL,
    preflight_required_scripts,
    require_explicit_if_default_missing,
)


def test_inrepo_core_scripts_exist_after_core_dir_fix():
    assert CORE_DIR == Path(__file__).resolve().parent / "core"
    assert PREPARE.is_file()
    assert UPDATE_TOOL.is_file()
    assert SPARSIFY.is_file()
    preflight_required_scripts()


def test_snapshot_defaults_are_machine_local_under_loc_root():
    for path in (DEFAULT_BASE, DEFAULT_MODEL, DEFAULT_IMAGES, DEFAULT_BASE_CACHE, EVAL_CORE):
        assert path == LOC_ROOT / path.relative_to(LOC_ROOT)
        assert "定位" in path.parts


def test_missing_machine_local_default_requires_explicit_flag(tmp_path: Path):
    missing = tmp_path / "absent_base.pt"
    with pytest.raises(SystemExit, match="--base-bundle default is missing"):
        require_explicit_if_default_missing("--base-bundle", None, missing)
    explicit = str(tmp_path / "operator.pt")
    assert require_explicit_if_default_missing("--base-bundle", explicit, missing) == explicit


def test_preflight_fails_closed_when_required_script_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr("update_pipeline.PREPARE", tmp_path / "missing_prepare.py")
    with pytest.raises(SystemExit, match="missing in-repo scripts"):
        preflight_required_scripts()
