from __future__ import annotations

from pathlib import Path

import pytest

from mapdoctor.recapture.profiles import CaptureGeometry, PlannerThresholds, load_config


ROOT = Path(__file__).resolve().parents[2]


def test_recapture_config_rejects_unknown_threshold_key(tmp_path) -> None:
    path = tmp_path / "typo.json"
    path.write_text('{"thresholds": {"min_existing_data_repairabilty": 0.99}}', encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown"):
        load_config(path)


def test_recapture_config_rejects_non_object_thresholds(tmp_path) -> None:
    path = tmp_path / "list.json"
    path.write_text('{"thresholds": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="thresholds"):
        load_config(path)


def test_recapture_defaults_still_load() -> None:
    thresholds, capture = load_config(ROOT / "config" / "recapture_defaults.json")
    assert isinstance(thresholds, PlannerThresholds)
    assert isinstance(capture, CaptureGeometry)
