#!/usr/bin/env python3
"""Strict pycolmap runtime preflight for target_site stages.

Importing this module is cheap and does not import pycolmap. Stages call
``verify_pycolmap_runtime()`` before doing work and record its returned,
JSON-serializable fingerprint in their gate evidence.
"""

from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path
from typing import Any, Iterable


EXPECTED_PYCOLMAP_VERSION = "4.0.4"
EXPECTED_PYCOLMAP_PROVIDER = "pycolmap-cuda12"
KNOWN_PYCOLMAP_PROVIDERS = frozenset({"pycolmap", "pycolmap-cuda12"})
PINNED_GLUEMAP_ENV = Path(
    "/home/cihcilab/micromamba/envs/target-site-gluemap-run"
)


class PycolmapRuntimeError(RuntimeError):
    """The imported pycolmap runtime does not satisfy the frozen contract."""


def _canonical_distribution_name(name: str) -> str:
    return name.strip().lower().replace("_", "-").replace(".", "-")


def _provider_fingerprint(distributions: Iterable[Any]) -> list[dict[str, str]]:
    providers: list[dict[str, str]] = []
    for distribution in distributions:
        raw_name = distribution.metadata.get("Name", "")
        name = _canonical_distribution_name(str(raw_name))
        if name in KNOWN_PYCOLMAP_PROVIDERS:
            providers.append({"name": name, "version": str(distribution.version)})
    return sorted(providers, key=lambda item: (item["name"], item["version"]))


def verify_pycolmap_runtime(
    *,
    module: Any | None = None,
    distributions: Iterable[Any] | None = None,
    prefix: Path | str | None = None,
    python_executable: Path | str | None = None,
) -> dict[str, Any]:
    """Verify and fingerprint the one accepted pycolmap 4.0.4 provider.

    Optional arguments are dependency-injection seams for unit tests. Production
    callers pass no arguments.
    """
    if module is None:
        try:
            import pycolmap as module
        except (ImportError, OSError) as exc:
            raise PycolmapRuntimeError(f"cannot import pycolmap: {exc}") from exc
    if distributions is None:
        distributions = importlib.metadata.distributions()

    runtime_prefix = Path(prefix or sys.prefix).expanduser().resolve()
    executable = Path(python_executable or sys.executable).expanduser().resolve()
    module_file_raw = getattr(module, "__file__", None)
    module_path = (
        Path(module_file_raw).expanduser().resolve()
        if isinstance(module_file_raw, (str, Path))
        else None
    )
    version = str(getattr(module, "__version__", ""))
    providers = _provider_fingerprint(distributions)

    errors: list[str] = []
    if version != EXPECTED_PYCOLMAP_VERSION:
        errors.append(
            f"imported version is {version!r}; expected {EXPECTED_PYCOLMAP_VERSION!r}"
        )
    expected_providers = [
        {
            "name": EXPECTED_PYCOLMAP_PROVIDER,
            "version": EXPECTED_PYCOLMAP_VERSION,
        }
    ]
    if providers != expected_providers:
        errors.append(
            f"provider set is {providers!r}; expected exactly {expected_providers!r}"
        )
    if module_path is None:
        errors.append("pycolmap.__file__ is absent")
    elif not module_path.is_relative_to(runtime_prefix):
        errors.append(
            f"module path {module_path} is outside runtime prefix {runtime_prefix}"
        )

    camera_model = getattr(module, "CameraModelId", None)
    config_type = getattr(module, "BundleAdjustmentConfig", None)
    required_apis = {
        "CameraModelId.PINHOLE": bool(
            camera_model is not None and hasattr(camera_model, "PINHOLE")
        ),
        "BundleAdjustmentOptions": hasattr(module, "BundleAdjustmentOptions"),
        "BundleAdjustmentConfig.set_constant_cam_intrinsics": False,
        "create_default_ceres_bundle_adjuster": hasattr(
            module, "create_default_ceres_bundle_adjuster"
        ),
    }
    if config_type is not None:
        try:
            config = config_type()
        except Exception as exc:  # pybind constructor failures are contract failures
            errors.append(f"BundleAdjustmentConfig construction failed: {exc}")
        else:
            required_apis["BundleAdjustmentConfig.set_constant_cam_intrinsics"] = (
                hasattr(config, "set_constant_cam_intrinsics")
            )
    missing_apis = [name for name, present in required_apis.items() if not present]
    if missing_apis:
        errors.append(f"missing required APIs: {missing_apis}")

    forbidden_apis = {"ReprojectionErrorType": hasattr(module, "ReprojectionErrorType")}
    present_forbidden = [name for name, present in forbidden_apis.items() if present]
    if present_forbidden:
        errors.append(f"forbidden API present: {present_forbidden}")

    if errors:
        raise PycolmapRuntimeError(
            "pycolmap runtime preflight failed: " + "; ".join(errors)
        )

    return {
        "version": version,
        "module_path": str(module_path),
        "providers": providers,
        "required_apis": required_apis,
        "forbidden_apis": forbidden_apis,
        "python_executable": str(executable),
        "sys_prefix": str(runtime_prefix),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(verify_pycolmap_runtime(), indent=2, sort_keys=True))
