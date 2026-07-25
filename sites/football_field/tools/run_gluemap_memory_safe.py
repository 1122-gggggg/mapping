#!/usr/bin/env python3
"""Launch GlueMap with a fail-closed per-image SIFT feature ceiling."""

from __future__ import annotations

import functools
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import yaml


def _config_path(argv: Sequence[str]) -> Path:
    for index, argument in enumerate(argv):
        if argument == "--config" and index + 1 < len(argv):
            return Path(argv[index + 1]).expanduser().resolve()
        if argument.startswith("--config="):
            return Path(argument.split("=", 1)[1]).expanduser().resolve()
    raise ValueError("the memory-safe launcher requires --config")


def _config_payload(argv: Sequence[str]) -> tuple[Path, dict[str, Any]]:
    path = _config_path(argv)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: config must be a YAML mapping")
    return path, payload


def sift_cap_from_config(argv: Sequence[str]) -> int:
    path, payload = _config_payload(argv)
    value = payload.get("sift_max_num_features")
    if type(value) is not int or value <= 0:
        raise ValueError(
            f"{path}: sift_max_num_features must be a positive integer"
        )
    return value


def ba_limits_from_config(
    argv: Sequence[str],
) -> tuple[int | None, int | None]:
    """Read optional recovery-only BA limits from the target-site config."""
    path, payload = _config_payload(argv)
    values: list[int | None] = []
    for key in ("ba_max_num_iterations", "ba_max_filter_iterations"):
        value = payload.get(key)
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError(f"{path}: {key} must be a positive integer")
        values.append(value)
    return values[0], values[1]


def install_sift_feature_cap(
    pycolmap_module: Any, max_num_features: int
) -> Callable[..., Any]:
    """Intercept feature extraction and enforce the configured SIFT ceiling."""
    if type(max_num_features) is not int or max_num_features <= 0:
        raise ValueError("SIFT max_num_features must be a positive integer")

    original = pycolmap_module.extract_features

    @functools.wraps(original)
    def capped_extract_features(*args: Any, **kwargs: Any) -> Any:
        positional = list(args)
        options = kwargs.get("extraction_options")
        if options is None and len(positional) >= 6:
            options = positional[5]
        if options is None:
            options = pycolmap_module.FeatureExtractionOptions()
            kwargs["extraction_options"] = options
        if not hasattr(options, "sift"):
            raise RuntimeError("pycolmap feature options expose no SIFT settings")
        options.sift.max_num_features = max_num_features
        return original(*positional, **kwargs)

    pycolmap_module.extract_features = capped_extract_features
    return original


def install_ba_limits(
    global_refinement_module: Any,
    *,
    max_num_iterations: int | None,
    max_filter_iterations: int | None,
) -> Callable[..., Any]:
    """Override GlueMap's hardcoded BA limits without editing its repository."""
    for label, value in (
        ("max_num_iterations", max_num_iterations),
        ("max_filter_iterations", max_filter_iterations),
    ):
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError(f"{label} must be a positive integer")

    original = global_refinement_module.IterativeBAOptions

    @functools.wraps(original)
    def limited_options(*args: Any, **kwargs: Any) -> Any:
        positional = list(args)
        if max_num_iterations is not None:
            if positional:
                positional[0] = max_num_iterations
            else:
                kwargs["max_ba_iterations"] = max_num_iterations
        if max_filter_iterations is not None:
            if len(positional) >= 2:
                positional[1] = max_filter_iterations
            else:
                kwargs["max_filter_iterations"] = max_filter_iterations
        return original(*positional, **kwargs)

    global_refinement_module.IterativeBAOptions = limited_options
    return original


def main(argv: Sequence[str] | None = None) -> Any:
    arguments = list(sys.argv[1:] if argv is None else argv)
    cap = sift_cap_from_config(arguments)
    ba_max_iterations, ba_max_filter_iterations = ba_limits_from_config(arguments)

    import pycolmap

    install_sift_feature_cap(pycolmap, cap)
    print(f"[memory-safe-launcher] SIFT max_num_features={cap}", flush=True)

    if ba_max_iterations is not None or ba_max_filter_iterations is not None:
        from gluemap.controllers import global_refinement

        install_ba_limits(
            global_refinement,
            max_num_iterations=ba_max_iterations,
            max_filter_iterations=ba_max_filter_iterations,
        )
        print(
            "[memory-safe-launcher] BA limits: "
            f"iterations={ba_max_iterations or 'GlueMap default'}, "
            f"filter_rounds={ba_max_filter_iterations or 'GlueMap default'}",
            flush=True,
        )

    from gluemap.cli import demo_main

    if argv is not None:
        sys.argv = [sys.argv[0], *arguments]
    return demo_main()


if __name__ == "__main__":
    raise SystemExit(main())
