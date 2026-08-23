from pathlib import Path

from update_map.synthetic import run_synthetic_demo


def test_synthetic_protocol_passes_core_invariants(tmp_path: Path) -> None:
    report = run_synthetic_demo(tmp_path)
    assert report["direct_status"] == "DIRECT_STRONG"
    assert report["bridge_validation"]["passed"]
    assert report["regression"]["passed"]
    assert report["core_immutable"]["ok"]
    assert (tmp_path / "sidecar" / "manifest.json").exists()
