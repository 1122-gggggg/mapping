#!/usr/bin/env python3
"""G-U1 / G-U2: prove a map update did not move the gauge.

WHY THIS EXISTS
  `colmap mapper --Mapper.fix_existing_frames 1` is a REQUEST, not evidence.
  If bundle adjustment nudges even one old camera, the whole coordinate frame
  rotates slightly and T_align_gravity silently goes stale -- and that failure
  is invisible downstream: localization success rate stays at 99%, the aircraft
  just thinks "up" is somewhere it is not.

  So after any update that claims to be gauge-preserving, diff the old model
  against the new one element by element and refuse to ship if anything moved.

WHAT IS COMPARED
  G-U1  every image present in BOTH models: rotation matrix and translation,
        element by element, plus the resulting camera-centre displacement
  G-U2  every point3D present in BOTH models, by point ID: xyz

TWO TRAPS THIS TOOL IS BUILT AROUND

  1. Point IDs are not stable across a rebuild. COLMAP renumbers them freely.
     If the old point IDs did NOT survive, comparing "point 417" in each model
     compares two unrelated landmarks and the deltas are meaningless noise.
     So ID survival is checked FIRST (G-U2a) and a poor survival rate fails
     the run rather than being silently averaged into a delta.

  2. A tiny per-element delta can still be a large gauge rotation once it is
     common-mode across every camera. Max |delta| alone under-reports that, so
     the residual rigid rotation between the two camera-centre clouds is
     estimated separately (G-U1c) -- that is the number T_align_gravity cares
     about.

USE
  # after a gauge-preserving (R1) update
  verify_gauge_invariance.py --before <old_model> --after <new_model> --out gates/G_U1.json

  # allow the update to ADD images/points but not move existing ones
  verify_gauge_invariance.py --before ... --after ... --expect-appended-images

Pair this with `build_gravity_alignment.py --verify`, which re-derives gravity
independently. This tool checks the poses; that one checks the consequence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SCHEMA = "gauge_invariance/1"

DEF_MAX_POSE_DELTA = 1e-9      # element-wise on R and t
DEF_MAX_POINT_DELTA = 1e-9     # element-wise on xyz
DEF_MAX_ROTATION_DEG = 1e-6    # residual common-mode rotation
DEF_MIN_ID_SURVIVAL = 0.99     # fraction of old point IDs still present


def load_model(model_dir: Path):
    try:
        import pycolmap
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "pycolmap is required. Use /usr/bin/python3.12 or the "
            "target-site-gluemap-run env (both carry pycolmap 4.0.4)."
        ) from exc

    rec = pycolmap.Reconstruction(str(model_dir))
    poses, centers = {}, {}
    for image in rec.images.values():
        rigid = image.cam_from_world
        rigid = rigid() if callable(rigid) else rigid
        center = image.projection_center
        center = center() if callable(center) else center
        poses[image.name] = (
            np.asarray(rigid.rotation.matrix(), dtype=np.float64),
            np.asarray(rigid.translation, dtype=np.float64),
        )
        centers[image.name] = np.asarray(center, dtype=np.float64)
    points = {int(pid): np.asarray(p.xyz, dtype=np.float64) for pid, p in rec.points3D.items()}
    return poses, centers, points


def residual_rotation_deg(before: np.ndarray, after: np.ndarray) -> float:
    """Common-mode rotation between two camera-centre clouds (Kabsch, no scale).

    This is what actually invalidates T_align_gravity. A per-element delta that
    looks negligible can still be a coherent rotation of the whole map.
    """
    if len(before) < 3:
        return 0.0
    a = before - before.mean(axis=0)
    b = after - after.mean(axis=0)
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    rotation = u @ np.diag([1.0, 1.0, d]) @ vt
    cos = (np.trace(rotation) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def compare(before_dir: Path, after_dir: Path, args: argparse.Namespace) -> dict:
    pose_b, center_b, point_b = load_model(before_dir)
    pose_a, center_a, point_a = load_model(after_dir)

    shared_images = sorted(set(pose_b) & set(pose_a))
    dropped_images = sorted(set(pose_b) - set(pose_a))
    added_images = sorted(set(pose_a) - set(pose_b))
    if not shared_images:
        raise SystemExit("no image names in common -- these are not before/after of one map")

    rot_deltas, trans_deltas, center_shifts = [], [], []
    for name in shared_images:
        rb, tb = pose_b[name]
        ra, ta = pose_a[name]
        rot_deltas.append(np.abs(ra - rb).max())
        trans_deltas.append(np.abs(ta - tb).max())
        center_shifts.append(np.linalg.norm(center_a[name] - center_b[name]))
    rot_deltas = np.array(rot_deltas)
    trans_deltas = np.array(trans_deltas)
    center_shifts = np.array(center_shifts)

    rotation_deg = residual_rotation_deg(
        np.array([center_b[n] for n in shared_images]),
        np.array([center_a[n] for n in shared_images]),
    )

    shared_points = sorted(set(point_b) & set(point_a))
    id_survival = len(shared_points) / len(point_b) if point_b else 0.0
    if shared_points:
        point_deltas = np.array(
            [np.abs(point_a[pid] - point_b[pid]).max() for pid in shared_points]
        )
    else:
        point_deltas = np.array([np.inf])

    pose_ok = float(rot_deltas.max()) <= args.max_pose_delta and float(
        trans_deltas.max()
    ) <= args.max_pose_delta

    gates = {
        "G-U1a_pose_elementwise": {
            "value": float(max(rot_deltas.max(), trans_deltas.max())),
            "threshold": args.max_pose_delta,
            "ok": pose_ok,
            "meaning": "every shared image's R and t must be bit-stable; this is the "
            "evidence that fix_existing_frames actually held",
        },
        "G-U1b_no_dropped_images": {
            "value": float(len(dropped_images)),
            "threshold": 0.0,
            "ok": len(dropped_images) == 0,
            "meaning": "a gauge-preserving update may ADD images but must not lose them",
        },
        "G-U1c_residual_rotation_deg": {
            "value": rotation_deg,
            "threshold": args.max_rotation_deg,
            "ok": rotation_deg <= args.max_rotation_deg,
            "meaning": "common-mode rotation of the camera cloud. THIS is what makes "
            "T_align_gravity stale, and a small per-element delta can still hide a "
            "large coherent rotation",
        },
        "G-U2a_point_id_survival": {
            "value": id_survival,
            "threshold": args.min_id_survival,
            "ok": id_survival >= args.min_id_survival,
            "meaning": "old point3D IDs must survive, otherwise G-U2b is comparing "
            "unrelated landmarks and its delta means nothing",
        },
        "G-U2b_point_elementwise": {
            "value": float(point_deltas.max()),
            "threshold": args.max_point_delta,
            "ok": float(point_deltas.max()) <= args.max_point_delta,
            "meaning": "shared old 3D points must not move",
        },
    }
    if not args.expect_appended_images and added_images:
        gates["G-U1d_no_added_images"] = {
            "value": float(len(added_images)),
            "threshold": 0.0,
            "ok": False,
            "meaning": "images were added but --expect-appended-images was not passed; "
            "pass it if this really is an append",
        }

    ok = all(gate["ok"] for gate in gates.values())
    return {
        "schema": SCHEMA,
        "ok": ok,
        "gauge_preserved": ok,
        "before": str(before_dir.resolve()),
        "after": str(after_dir.resolve()),
        "counts": {
            "shared_images": len(shared_images),
            "added_images": len(added_images),
            "dropped_images": len(dropped_images),
            "before_points3D": len(point_b),
            "after_points3D": len(point_a),
            "shared_points3D": len(shared_points),
        },
        "pose_delta": {
            "rotation_elementwise_max": float(rot_deltas.max()),
            "translation_elementwise_max": float(trans_deltas.max()),
            "center_shift_max": float(center_shifts.max()),
            "center_shift_p95": float(np.percentile(center_shifts, 95)),
            "residual_rotation_deg": rotation_deg,
        },
        "point_delta": {
            "elementwise_max": float(point_deltas.max()),
            "elementwise_p95": float(np.percentile(point_deltas, 95)),
            "id_survival": id_survival,
        },
        "added_image_names": added_images[:200],
        "dropped_image_names": dropped_images[:200],
        "gates": gates,
        "downstream": {
            "T_align_gravity": "carry" if ok else "STALE -- re-derive",
            "edm_bundle": "append new keyframes only" if ok else "rebuild in full",
            "tracker_scale_params": "RECOMPUTE ALWAYS -- S depends on the camera set, "
            "not on the gauge. Run recompute_site_scale.py even when this passes.",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--before", required=True, type=Path)
    parser.add_argument("--after", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--max-pose-delta", type=float, default=DEF_MAX_POSE_DELTA)
    parser.add_argument("--max-point-delta", type=float, default=DEF_MAX_POINT_DELTA)
    parser.add_argument("--max-rotation-deg", type=float, default=DEF_MAX_ROTATION_DEG)
    parser.add_argument("--min-id-survival", type=float, default=DEF_MIN_ID_SURVIVAL)
    parser.add_argument(
        "--expect-appended-images",
        action="store_true",
        help="the update is allowed to add new images (the normal R1 case)",
    )
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args(argv)

    for path in (args.before, args.after):
        if not path.is_dir():
            raise SystemExit(f"model dir not found: {path}")

    result = compare(args.before, args.after, args)

    counts = result["counts"]
    print(
        f"shared {counts['shared_images']} images, "
        f"+{counts['added_images']} added, -{counts['dropped_images']} dropped"
    )
    for name, gate in result["gates"].items():
        print(f"[{'PASS' if gate['ok'] else 'FAIL'}] {name}: "
              f"{gate['value']:.3e} (thr {gate['threshold']:.3e})")
    print(f"T_align_gravity -> {result['downstream']['T_align_gravity']}")
    print(f"tracker scale    -> {result['downstream']['tracker_scale_params']}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.out}")

    if not result["ok"] and not args.allow_fail:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
