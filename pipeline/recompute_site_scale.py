#!/usr/bin/env python3
"""G-U3: recompute the map-unit tracker profile after any change to the map.

THE ONE THING TO UNDERSTAND
  The robust camera span is

      S = 2 * p95( || center_i - componentwise_median(all centers) || )

  and every deployed tracker threshold is a fixed dimensionless ratio times S.
  S is a p95 over the WHOLE camera set. So it moves when the camera set moves --
  which happens on a pure append, even when the gauge is bit-identical and every
  old pose is untouched.

  "Old poses did not move, so the scale parameters do not change" is FALSE, and
  it is the easiest thing in this whole pipeline to get wrong. Run this tool
  after EVERY update, including ones that pass verify_gauge_invariance.py.

  Use --before to see exactly how far S moved and which thresholds shifted.

WHY p95 AND NOT max
  max is hostage to a single stray camera. Measured on the accepted maps:

    target_site     p95 2.5032 -> S 5.0065   max 3.1189 -> S would be 6.2378 (+25%)
    football_field  p95 0.9164 -> S 1.8328   max 1.0964 -> S would be 2.1929 (+20%)

  One bad pose would inflate every threshold by a fifth. p95 drops the outer 5%
  so a handful of outliers cannot move it. Same reason the origin is the
  componentwise median rather than the mean.

WHAT THIS DOES NOT DO
  It does not metricize anything. S is in map units and the map is scale-free.
  A larger S does not mean a physically larger site.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SCHEMA = "site_scale/1"

# Dimensionless, shared across every site. Source of truth:
# EDM定位測試/build/make_transfer_package.py -> span_normalized_tracker
SPAN_NORMALIZED_TRACKER = {
    "radius": 0.16,
    "max_jump": 0.40,
    "adaptive_jump_floor": 0.0006,
    "adaptive_jump_bootstrap": 0.004,
    "adaptive_jump_ceiling": 0.0016,
}

SPAN_DEFINITION = "2*p95_distance_from_componentwise_median"

# Relative move in S beyond which the deployed profile must be reissued.
DEF_MAX_SCALE_DRIFT = 0.01


def camera_centers(model_dir: Path) -> tuple[np.ndarray, list[str]]:
    try:
        import pycolmap
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "pycolmap is required. Use /usr/bin/python3.12 or the "
            "target-site-gluemap-run env (both carry pycolmap 4.0.4)."
        ) from exc

    rec = pycolmap.Reconstruction(str(model_dir))
    centers, names = [], []
    for image in rec.images.values():
        center = image.projection_center
        center = center() if callable(center) else center
        centers.append(np.asarray(center, dtype=np.float64))
        names.append(image.name)
    return np.array(centers), names


def robust_camera_span(centers: np.ndarray) -> float:
    """Mirror of derive_site_profile()'s robust_camera_span. Keep them identical."""
    values = np.asarray(centers, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 2:
        raise ValueError("camera centers must have shape (N, 3), N >= 2")
    if not np.isfinite(values).all():
        raise ValueError("camera centers must be finite")
    origin = np.median(values, axis=0)
    span = 2.0 * float(np.percentile(np.linalg.norm(values - origin, axis=1), 95))
    if not np.isfinite(span) or span <= 1e-12:
        raise ValueError("camera centers have degenerate robust span")
    return span


def tracker_params(span: float) -> dict[str, float]:
    return {name: float(ratio) * span for name, ratio in SPAN_NORMALIZED_TRACKER.items()}


def build(model_dir: Path, before_dir: Path | None, args: argparse.Namespace) -> dict:
    centers, names = camera_centers(model_dir)
    span = robust_camera_span(centers)

    origin = np.median(centers, axis=0)
    distances = np.linalg.norm(centers - origin, axis=1)
    span_if_max = 2.0 * float(distances.max())

    result = {
        "schema": SCHEMA,
        "model": str(model_dir.resolve()),
        "span_definition": SPAN_DEFINITION,
        "metric_scale": False,
        "num_images": int(len(centers)),
        "robust_span": span,
        "ratios": dict(SPAN_NORMALIZED_TRACKER),
        "tracker": tracker_params(span),
        "outlier_sensitivity": {
            "p95_distance": float(np.percentile(distances, 95)),
            "max_distance": float(distances.max()),
            "span_if_max_were_used": span_if_max,
            "inflation_if_max_were_used": float(span_if_max / span - 1.0),
        },
    }

    if before_dir is not None:
        before_centers, before_names = camera_centers(before_dir)
        before_span = robust_camera_span(before_centers)
        drift = abs(span / before_span - 1.0)
        added = sorted(set(names) - set(before_names))

        # The point of the whole tool: recompute S over ONLY the images that
        # existed before. If that subset span is unchanged but the full span is
        # not, the gauge was preserved and S still moved -- purely from adding
        # cameras.
        shared = [n for n in names if n in set(before_names)]
        index = {n: i for i, n in enumerate(names)}
        shared_span = (
            robust_camera_span(np.array([centers[index[n]] for n in shared]))
            if len(shared) >= 2
            else float("nan")
        )

        result["comparison"] = {
            "before_model": str(before_dir.resolve()),
            "before_span": before_span,
            "after_span": span,
            "relative_drift": drift,
            "added_images": len(added),
            "span_over_shared_images_only": shared_span,
            "gauge_hint": (
                "shared-image span matches before-span, so the gauge held and S moved "
                "purely because cameras were added"
                if np.isfinite(shared_span) and abs(shared_span / before_span - 1.0) < 1e-9
                else "shared-image span ALSO moved -- the old poses were disturbed; "
                "run verify_gauge_invariance.py"
            ),
            "before_tracker": tracker_params(before_span),
            "tracker_delta": {
                key: tracker_params(span)[key] - tracker_params(before_span)[key]
                for key in SPAN_NORMALIZED_TRACKER
            },
        }
        result["gates"] = {
            "G-U3_profile_reissue_required": {
                "value": drift,
                "threshold": args.max_scale_drift,
                "ok": True,  # never a failure; it is a REQUIREMENT flag
                "reissue_required": drift > args.max_scale_drift,
                "meaning": "S moved by more than the tolerance, so the deployed site "
                "profile must be reissued. This is not a build failure, it is a "
                "delivery obligation.",
            }
        }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, type=Path, help="model AFTER the update")
    parser.add_argument("--before", type=Path, help="model BEFORE the update, to show the drift")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--max-scale-drift", type=float, default=DEF_MAX_SCALE_DRIFT)
    args = parser.parse_args(argv)

    if not args.model.is_dir():
        raise SystemExit(f"model dir not found: {args.model}")
    if args.before is not None and not args.before.is_dir():
        raise SystemExit(f"model dir not found: {args.before}")

    result = build(args.model, args.before, args)

    print(f"images {result['num_images']}   S = {result['robust_span']:.6f} (map units)")
    for name, value in result["tracker"].items():
        print(f"  {name:26s} = {value:.6f}")
    sensitivity = result["outlier_sensitivity"]
    print(
        f"  [p95 vs max: S would be {sensitivity['span_if_max_were_used']:.4f} "
        f"(+{sensitivity['inflation_if_max_were_used'] * 100:.0f}%) if max were used]"
    )

    if "comparison" in result:
        comparison = result["comparison"]
        print(
            f"\nbefore S = {comparison['before_span']:.6f} -> after S = "
            f"{comparison['after_span']:.6f}  "
            f"(drift {comparison['relative_drift'] * 100:.3f}%, "
            f"+{comparison['added_images']} images)"
        )
        print(f"  {comparison['gauge_hint']}")
        for name, delta in comparison["tracker_delta"].items():
            print(f"    {name:26s} {delta:+.6f}")
        if result["gates"]["G-U3_profile_reissue_required"]["reissue_required"]:
            print("\n  ==> REISSUE the deployed site profile. Old thresholds are wrong.")
        else:
            print("\n  ==> drift within tolerance; profile may be carried")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
