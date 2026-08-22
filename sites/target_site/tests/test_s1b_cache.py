from __future__ import annotations

import importlib.util
import pickle
import sys
import types
from pathlib import Path

SITE = Path(__file__).resolve().parents[1]
SCRIPT = SITE / "tools" / "s1b_bridge_feasibility.py"


def _load_s1b():
    fake_s1 = types.ModuleType("s1_motion_scan")

    def _blocked(*_args, **_kwargs):
        raise AssertionError("s1.megaloc_descriptors must not run")

    fake_s1.megaloc_descriptors = _blocked
    sys.modules["s1_motion_scan"] = fake_s1
    name = "s1b_bridge_feasibility_target_site"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


s1b = _load_s1b()


def test_cache_paths_are_under_run_dir(tmp_path: Path) -> None:
    cache, metadata = s1b.cache_paths(tmp_path)
    run_dir = tmp_path.resolve()
    assert cache == run_dir / "cache" / "s1b" / "megaloc_descriptors.pkl"
    assert metadata == run_dir / "cache" / "s1b" / "megaloc_descriptors.meta.json"


def test_cache_is_fresh_false_when_missing(tmp_path: Path) -> None:
    _, metadata = s1b.cache_paths(tmp_path)
    material = tmp_path / "motion_manifest.json"
    material.write_text("{}", encoding="utf-8")
    assert s1b.cache_is_fresh(metadata, [material], {"topk": s1b.TOPK}) is False


def test_matching_pickle_and_meta_are_fresh_until_config_changes(tmp_path: Path) -> None:
    cache, metadata = s1b.cache_paths(tmp_path)
    material = tmp_path / "motion_manifest.json"
    material.write_text('{"ok": true}', encoding="utf-8")
    cache.parent.mkdir(parents=True)
    cache.write_bytes(pickle.dumps({"S01": None}))
    config = {"topk": s1b.TOPK, "sequences": ["S01"]}
    s1b.write_cache_metadata(metadata, cache, [material], config)

    assert s1b.cache_is_fresh(metadata, [material], config) is True
    assert s1b.cache_is_fresh(metadata, [material], {**config, "topk": s1b.TOPK + 1}) is False


def test_retrieval_verdict_never_calls_low_cross_frac_natural() -> None:
    assert s1b.retrieval_verdict(0.09, 12).startswith("FORCE REQUIRED")
    assert s1b.retrieval_verdict(0.02, 4).startswith("FORCE REQUIRED")
    assert "NATURAL" not in s1b.retrieval_verdict(0.099, 1)
    assert s1b.retrieval_verdict(0.25, 8).startswith("NATURAL RETRIEVAL MAY SUFFICE")


def test_retrieval_verdict_zero_mutual_is_force() -> None:
    assert s1b.retrieval_verdict(0.99, 0).startswith("FORCE REQUIRED")
    assert "zero mutual" in s1b.retrieval_verdict(0.99, 0)

