from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "run_pixsfm_refinement.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_pixsfm_refinement", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_reconstruction(camera_params=(100.0, 101.0, 50.0, 40.0)):
    camera = SimpleNamespace(
        camera_id=7,
        model_name="PINHOLE",
        width=100,
        height=80,
        params=np.asarray(camera_params, dtype=float),
    )
    images = {
        1: SimpleNamespace(image_id=1, name="S01/a.jpg", camera_id=7, registered=True),
        2: SimpleNamespace(image_id=2, name="S02/b.jpg", camera_id=7, registered=True),
    }
    return SimpleNamespace(cameras={7: camera}, images=images)


def test_pixsfm_config_freezes_every_intrinsic_parameter_but_refines_extrinsics() -> (
    None
):
    module = load_module()

    config = module.fixed_intrinsics_config(max_edge=1280, max_iterations=12)
    optimizer = config["mapping"]["BA"]["optimizer"]

    assert optimizer["refine_focal_length"] is False
    assert optimizer["refine_principal_point"] is False
    assert optimizer["refine_extra_params"] is False
    assert optimizer["refine_extrinsics"] is True
    assert optimizer["solver"]["max_num_iterations"] == 12
    assert config["mapping"]["dense_features"]["max_edge"] == 1280
    # HDF5 2.1 cannot bulk-load PixSfM's FeatureSet_f16 through HighFive.
    # FP32 also avoids the unstable half-precision BA path while preserving
    # the fixed-intrinsics candidate contract.
    assert config["mapping"]["dense_features"]["dtype"] == "float"
    assert config["mapping"]["KA"]["apply"] is False


def test_intrinsics_gate_detects_parameter_or_shared_camera_assignment_drift() -> None:
    module = load_module()
    before = module.intrinsics_signature(fake_reconstruction())

    module.assert_intrinsics_unchanged(
        before, module.intrinsics_signature(fake_reconstruction())
    )

    with pytest.raises(RuntimeError, match="camera parameters changed"):
        module.assert_intrinsics_unchanged(
            before,
            module.intrinsics_signature(
                fake_reconstruction((100.1, 101.0, 50.0, 40.0))
            ),
        )

    reassigned = fake_reconstruction()
    reassigned.images[2].camera_id = 8
    with pytest.raises(RuntimeError, match="camera assignments changed"):
        module.assert_intrinsics_unchanged(
            before, module.intrinsics_signature(reassigned)
        )


def test_output_path_must_be_isolated_from_source_and_production_artifacts(
    tmp_path: Path,
) -> None:
    module = load_module()
    source = tmp_path / "run" / "final_model"
    protected = [source, tmp_path / "run" / "edm", tmp_path / "transfer" / "release"]
    source.mkdir(parents=True)

    with pytest.raises(ValueError, match="protected"):
        module.validate_isolated_output(source, source, protected)
    with pytest.raises(ValueError, match="protected"):
        module.validate_isolated_output(
            source, tmp_path / "transfer" / "release" / "candidate", protected
        )

    allowed = tmp_path / "experiments" / "pixsfm" / "model"
    assert (
        module.validate_isolated_output(source, allowed, protected) == allowed.resolve()
    )


def test_command_runs_only_pixsfm_bundle_adjustment_on_existing_model(
    tmp_path: Path,
) -> None:
    module = load_module()
    command = module.build_pixsfm_command(
        python=Path("/opt/pixsfm/bin/python"),
        input_model=tmp_path / "input",
        output_model=tmp_path / "output",
        image_root=tmp_path / "images",
        config_path=tmp_path / "fixed.yaml",
        cache_path=tmp_path / "features.h5",
    )

    assert command[:4] == [
        "/opt/pixsfm/bin/python",
        "-m",
        "pixsfm.refine_colmap",
        "ba",
    ]
    assert command[command.index("--input_path") + 1] == str(
        (tmp_path / "input").resolve()
    )
    assert command[command.index("--output_path") + 1] == str(
        (tmp_path / "output").resolve()
    )
    assert not any(
        "match" in item.lower() or "triang" in item.lower() for item in command
    )


def test_reused_feature_cache_must_exist_and_is_never_copied(tmp_path: Path) -> None:
    module = load_module()
    output_root = tmp_path / "candidate"
    cache = tmp_path / "previous" / "features.h5"

    with pytest.raises(FileNotFoundError, match="feature cache"):
        module.select_feature_cache(output_root, cache)

    cache.parent.mkdir()
    cache.write_bytes(b"sealed-cache")
    selected, reused = module.select_feature_cache(output_root, cache)

    assert selected == cache.resolve()
    assert reused is True
    assert not output_root.exists()

    selected, reused = module.select_feature_cache(output_root, None)
    assert selected == (output_root / "s2dnet_featuremaps_sparse.h5").resolve()
    assert reused is False


def _write_fake_model(path: Path, marker: bytes) -> None:
    path.mkdir(parents=True)
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        (path / name).write_bytes(marker + b":" + name.encode())


def _gate_reconstruction(mean_reprojection_error: float):
    reconstruction = fake_reconstruction()
    reconstruction.num_reg_images = 2
    reconstruction.num_points3D = 3
    reconstruction.compute_mean_reprojection_error = lambda: mean_reprojection_error
    return reconstruction


def _run_completed_candidate(
    module,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    structurally_eligible: bool,
) -> tuple[Path, Path, dict]:
    source_model = tmp_path / "source_model"
    image_root = tmp_path / "images"
    output_root = tmp_path / "candidate"
    pixsfm_python = tmp_path / "pixsfm-python"
    _write_fake_model(source_model, b"source")
    image_root.mkdir()
    pixsfm_python.write_text("", encoding="utf-8")
    source = _gate_reconstruction(1.0)
    # The stored error looks better, reproducing the old false-PASS path.  The
    # comparator result represents recomputed geometry and is authoritative.
    candidate = _gate_reconstruction(0.5)
    reconstructions = {
        source_model.resolve(): source,
        (output_root / "model.partial").resolve(): candidate,
    }

    fake_pycolmap = SimpleNamespace(
        Reconstruction=lambda path: reconstructions[Path(path).resolve()]
    )
    monkeypatch.setitem(sys.modules, "pycolmap", fake_pycolmap)
    monkeypatch.setattr(
        module,
        "memory_snapshot",
        lambda: {"available_gib": 32.0, "swap_free_gib": 8.0},
    )
    monkeypatch.setattr(
        module,
        "_preflight_pixsfm",
        lambda _python: {
            "pixsfm": "test",
            "pycolmap": "test",
            "module": "test",
        },
    )
    comparison = {
        "schema": "colmap-refinement-comparison/v1",
        "gates": {
            "topology_exact": True,
            "mean_reprojection_nonworse": structurally_eligible,
        },
        "structurally_eligible": structurally_eligible,
        "reprojection_error_px": {
            "source": {"mean_px": 1.0},
            "candidate": {"mean_px": 1.25},
        },
    }
    compared = []

    def compare(source_arg, candidate_arg):
        compared.append((source_arg, candidate_arg))
        return comparison

    monkeypatch.setattr(module, "compare_reconstructions", compare, raising=False)

    def completed_process(command, **_kwargs):
        partial = Path(command[command.index("--output_path") + 1])
        _write_fake_model(partial, b"candidate")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", completed_process)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--input-model",
            str(source_model),
            "--image-root",
            str(image_root),
            "--output-root",
            str(output_root),
            "--pixsfm-python",
            str(pixsfm_python),
        ],
    )

    if structurally_eligible:
        module.main()
    else:
        with pytest.raises(SystemExit, match="rejected"):
            module.main()

    assert compared == [(source, candidate)]
    report = json.loads((output_root / "candidate_report.json").read_text())
    return source_model, output_root, report


def test_recomputed_geometry_rejects_candidate_even_when_stored_error_looks_better(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_model, output_root, report = _run_completed_candidate(
        load_module(), tmp_path, monkeypatch, structurally_eligible=False
    )

    assert report["status"] == "rejected"
    assert report["comparison"]["structurally_eligible"] is False
    assert report["reject_reasons"] == [
        "comparison gate failed: mean_reprojection_nonworse"
    ]
    assert (output_root / "model.partial").is_dir()
    assert not (output_root / "model").exists()
    assert (source_model / "points3D.bin").read_bytes().startswith(b"source:")


def test_only_recomputed_structurally_eligible_candidate_is_promoted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, output_root, report = _run_completed_candidate(
        load_module(), tmp_path, monkeypatch, structurally_eligible=True
    )

    assert report["status"] == "passed"
    assert report["comparison"]["structurally_eligible"] is True
    assert report["reject_reasons"] == []
    assert (output_root / "model").is_dir()
    assert not (output_root / "model.partial").exists()


def test_dry_run_does_not_import_pycolmap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_module()
    source_model = tmp_path / "source_model"
    image_root = tmp_path / "images"
    pixsfm_python = tmp_path / "pixsfm-python"
    output_root = tmp_path / "candidate"
    _write_fake_model(source_model, b"source")
    image_root.mkdir()
    pixsfm_python.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "memory_snapshot",
        lambda: {"available_gib": 32.0, "swap_free_gib": 8.0},
    )
    monkeypatch.setattr(
        module,
        "_preflight_pixsfm",
        lambda _python: {
            "pixsfm": "test",
            "pycolmap": "test",
            "module": "test",
        },
    )
    monkeypatch.delitem(sys.modules, "pycolmap", raising=False)
    original_import = __import__

    def reject_pycolmap_import(name, *args, **kwargs):
        if name == "pycolmap":
            raise AssertionError("dry-run imported pycolmap")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", reject_pycolmap_import)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--input-model",
            str(source_model),
            "--image-root",
            str(image_root),
            "--output-root",
            str(output_root),
            "--pixsfm-python",
            str(pixsfm_python),
            "--dry-run",
        ],
    )

    module.main()

    report = json.loads((output_root / "candidate_report.json").read_text())
    assert report["status"] == "planned"
    assert report["geometry_preflight"] == "deferred_until_execution"
