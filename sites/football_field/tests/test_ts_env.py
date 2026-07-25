from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


TARGET_SITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TARGET_SITE / "tools"))

from ts_common import GLUEMAP_DEMO, GLUEMAP_ENV, GLUEMAP_PY  # noqa: E402
from ts_env import (  # noqa: E402
    PINNED_GLUEMAP_ENV,
    PycolmapRuntimeError,
    verify_pycolmap_runtime,
)


EXPECTED_ENV = Path(
    "/home/cihcilab/micromamba/envs/target-site-gluemap-run"
)


class FakeDistribution:
    def __init__(self, name: str, version: str) -> None:
        self.metadata = {"Name": name}
        self.version = version


def fake_module(prefix: Path, *, version: str = "4.0.4", missing: str | None = None):
    class CameraModelId:
        PINHOLE = 1

    class BundleAdjustmentConfig:
        def set_constant_cam_intrinsics(self) -> None:
            return None

    values = {
        "__version__": version,
        "__file__": str(prefix / "lib/python3.11/site-packages/pycolmap/__init__.py"),
        "CameraModelId": CameraModelId,
        "BundleAdjustmentOptions": type("BundleAdjustmentOptions", (), {}),
        "BundleAdjustmentConfig": BundleAdjustmentConfig,
        "create_default_ceres_bundle_adjuster": lambda: None,
    }
    if missing is not None:
        values.pop(missing)
    return SimpleNamespace(**values)


def verify(
    tmp_path: Path,
    *,
    version: str = "4.0.4",
    providers: list[tuple[str, str]] | None = None,
    module_prefix: Path | None = None,
    missing: str | None = None,
):
    prefix = tmp_path / "env"
    module = fake_module(module_prefix or prefix, version=version, missing=missing)
    distributions = [
        FakeDistribution(name, provider_version)
        for name, provider_version in (
            providers if providers is not None else [("pycolmap-cuda12", "4.0.4")]
        )
    ]
    return verify_pycolmap_runtime(
        module=module,
        distributions=distributions,
        prefix=prefix,
        python_executable=prefix / "bin/python",
    )


def test_exact_pycolmap_runtime_passes(tmp_path: Path) -> None:
    fingerprint = verify(tmp_path)

    assert fingerprint["version"] == "4.0.4"
    assert fingerprint["providers"] == [{"name": "pycolmap-cuda12", "version": "4.0.4"}]
    assert all(fingerprint["required_apis"].values())


def test_source_selected_runtime_is_the_reproducible_run_prefix() -> None:
    assert GLUEMAP_ENV == EXPECTED_ENV
    assert PINNED_GLUEMAP_ENV == EXPECTED_ENV
    assert GLUEMAP_PY == EXPECTED_ENV / "bin" / "python"
    assert GLUEMAP_DEMO == EXPECTED_ENV / "bin" / "gluemap-demo"
    assert GLUEMAP_PY.resolve().is_file()
    assert GLUEMAP_DEMO.resolve().is_file()


def test_wrong_imported_version_fails(tmp_path: Path) -> None:
    with pytest.raises(PycolmapRuntimeError, match="imported version"):
        verify(tmp_path, version="3.10.0")


def test_wrong_provider_fails(tmp_path: Path) -> None:
    with pytest.raises(PycolmapRuntimeError, match="provider set"):
        verify(tmp_path, providers=[("pycolmap", "4.0.4")])


def test_duplicate_providers_fail(tmp_path: Path) -> None:
    with pytest.raises(PycolmapRuntimeError, match="provider set"):
        verify(
            tmp_path,
            providers=[("pycolmap-cuda12", "4.0.4"), ("pycolmap", "0.6.1")],
        )


def test_module_outside_runtime_prefix_fails(tmp_path: Path) -> None:
    with pytest.raises(PycolmapRuntimeError, match="outside runtime prefix"):
        verify(tmp_path, module_prefix=tmp_path / "other_env")


def test_missing_required_api_fails(tmp_path: Path) -> None:
    with pytest.raises(PycolmapRuntimeError, match="missing required APIs"):
        verify(tmp_path, missing="create_default_ceres_bundle_adjuster")


def test_deprecated_reprojection_error_type_fails(tmp_path: Path) -> None:
    prefix = tmp_path / "env"
    module = fake_module(prefix)
    module.ReprojectionErrorType = object()

    with pytest.raises(PycolmapRuntimeError, match="forbidden API"):
        verify_pycolmap_runtime(
            module=module,
            distributions=[FakeDistribution("pycolmap-cuda12", "4.0.4")],
            prefix=prefix,
            python_executable=prefix / "bin/python",
        )
