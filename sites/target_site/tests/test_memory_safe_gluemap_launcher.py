from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


TARGET_SITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TARGET_SITE / "tools"))

import run_gluemap_memory_safe as launcher  # noqa: E402


def test_sift_cap_is_applied_to_live_pycolmap_options() -> None:
    seen: dict[str, object] = {}

    def extract_features(*args, **kwargs):
        seen["args"] = args
        seen["options"] = kwargs["extraction_options"]
        return "result"

    fake_pycolmap = SimpleNamespace(extract_features=extract_features)
    original = launcher.install_sift_feature_cap(fake_pycolmap, 2048)
    options = SimpleNamespace(sift=SimpleNamespace(max_num_features=8192))

    result = fake_pycolmap.extract_features(
        "database.db", "images", extraction_options=options
    )

    assert result == "result"
    assert original is extract_features
    assert seen["options"].sift.max_num_features == 2048


def test_sift_cap_rejects_nonpositive_values() -> None:
    fake_pycolmap = SimpleNamespace(extract_features=lambda: None)

    try:
        launcher.install_sift_feature_cap(fake_pycolmap, 0)
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("zero SIFT cap was accepted")


def test_ba_limits_override_gluemap_hardcoded_values() -> None:
    seen: dict[str, object] = {}

    def iterative_ba_options(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return "options"

    module = SimpleNamespace(IterativeBAOptions=iterative_ba_options)
    original = launcher.install_ba_limits(
        module, max_num_iterations=50, max_filter_iterations=2
    )

    result = module.IterativeBAOptions(
        max_ba_iterations=200,
        max_filter_iterations=3,
        normalized_reproj_threshold=0.01,
    )

    assert result == "options"
    assert original is iterative_ba_options
    assert seen["kwargs"] == {
        "max_ba_iterations": 50,
        "max_filter_iterations": 2,
        "normalized_reproj_threshold": 0.01,
    }


def test_optional_ba_limits_are_read_from_config(tmp_path: Path) -> None:
    config = tmp_path / "recovery.yaml"
    config.write_text(
        "sift_max_num_features: 2048\n"
        "ba_max_num_iterations: 50\n"
        "ba_max_filter_iterations: 2\n",
        encoding="utf-8",
    )

    assert launcher.ba_limits_from_config(["--config", str(config)]) == (50, 2)
