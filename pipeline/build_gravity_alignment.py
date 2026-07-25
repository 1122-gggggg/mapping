#!/usr/bin/env python3
"""Derive T_align_gravity.json from a COLMAP/GlueMap model.

WHY THIS EXISTS
  A monocular SfM map has an arbitrary gauge: arbitrary rotation, translation and
  scale. Nothing in the reconstruction knows which way is up. The deployment side
  needs that, so every accepted map (and every map update that changes the gauge)
  must ship a T_align_gravity.json alongside the EDM bundle and the map-unit
  tracker profile.

  This does NOT metricize the map. Scale stays arbitrary; the output is a pure
  rotation. Do not read a "height" out of the aligned frame.

HOW GRAVITY IS RECOVERED (primary: camera-x-axis / roll==0)
  A gimballed drone camera holds roll near zero. So the camera's x-axis (image
  right) stays horizontal, i.e. perpendicular to gravity, on every frame:

      g . (row 0 of R_i)  ==  0        for all images i,   X_cam = R X_world + t

  Stack those rows into A and gravity is the direction most orthogonal to all of
  them -- the smallest right singular vector of A. This is well conditioned as
  soon as the flight yaws (a route always does); it degenerates only for a
  perfectly straight single-heading flight, which G-GRAV-1 catches.

  Real footage has a few genuinely rolled or mis-posed frames (target_site has
  frames out to ~10 deg), so the solve is IRLS-reweighted rather than plain SVD.

CROSS-CHECK (secondary: camera-centre plane, CONDITIONAL)
  If the drone flew at roughly constant altitude, the camera centres lie in a
  plane whose normal is gravity. That is an independent estimate -- but it is
  only meaningful when the centres are ACTUALLY planar.

  Measured on this project:
    football_field  centres planar (s2/s3 = 7.0)   -> agrees with primary, 3.3 deg
    target_site     centres NOT planar (s2/s3 = 1.7) -> disagrees by 83.7 deg

  target_site's 83.7 deg is not a gravity failure, it is the cross-check being
  inapplicable. So the planarity of the centres is gated FIRST, and a degenerate
  cross-check is reported as SKIPPED rather than dragging the result down. Never
  compare the two numbers without checking that gate.

SIGN
  Gravity points down. Disambiguated by requiring it to point from the cameras
  toward the scene (drone above what it films). The margin is reported and gated:
  a map of a vertical structure shot side-on has a weak margin and must not have
  its sign guessed silently.

MEASURED ON THE TWO ACCEPTED MAPS (2026-07-26)
    football_field  cond 54.6  roll p95 1.14 deg  sign 0.363  crosscheck  3.33 deg PASS
    target_site     cond 30.8  roll p95 1.38 deg  sign 0.264  crosscheck 83.65 deg SKIP

  Because target_site gets no cross-check, the result was validated out-of-band
  against a third, fully independent signal: inter-frame drone motion should be
  mostly horizontal. Median elevation of the motion direction away from the plane
  perpendicular to the estimate:

                    estimate    -Y      -Z
    target_site       19.5     30.6    39.4   deg
    football_field     7.3      7.9    27.4   deg

  The estimate beats every axis-aligned alternative on both maps. That signal is
  NOT a gate here: its p90 runs past 70 deg on both maps because drones really do
  climb and descend, so it discriminates in the median but is far too noisy to
  threshold. Use it the same way if you need to sanity-check a new site.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

SCHEMA = "T_align_gravity/1"

# Defaults are deliberately loose enough to pass the two accepted maps on this
# project and tight enough to reject a straight-line or side-on capture.
DEF_MIN_CONDITIONING = 5.0    # s2/s3 of the camera-x-axis matrix
# s2/s1 of the same matrix. Guards the case s2/s3 CANNOT guard: a single-heading
# straight flight makes every x-axis parallel, so s2 and s3 are BOTH ~0 and their
# ratio is numerical noise that reads as perfectly conditioned. Measured: 0.40
# (target_site), 0.83 (football_field), 1.1e-16 (synthetic single heading).
DEF_MIN_YAW_DIVERSITY = 0.10
DEF_MAX_ROLL_P95_DEG = 3.0    # residual |angle(camera x-axis, horizontal)|
DEF_MIN_PLANARITY = 3.0       # s2/s3 of centred camera centres, to TRUST the cross-check
DEF_MAX_CROSSCHECK_DEG = 10.0
DEF_MIN_SIGN_MARGIN = 0.15    # |cos| between gravity and (scene mean - camera mean)
DEF_IRLS_ITERS = 12
DEF_IRLS_SOFT_DEG = 2.0       # Huber-ish knee on the roll residual


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def model_provenance(model_dir: Path) -> dict:
    files = {}
    for name in sorted(p.name for p in model_dir.iterdir() if p.is_file()):
        files[name] = sha256_file(model_dir / name)
    return {"path": str(model_dir.resolve()), "files_sha256": files}


def load_model(model_dir: Path):
    """Return (camera_x_axes Nx3, camera_centers Nx3, point_xyz Mx3, names)."""
    try:
        import pycolmap
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "pycolmap is required. Use /usr/bin/python3.12 or the "
            "target-site-gluemap-run env (both carry pycolmap 4.0.4)."
        ) from exc

    rec = pycolmap.Reconstruction(str(model_dir))
    x_axes, centers, names = [], [], []
    for image in rec.images.values():
        rigid = image.cam_from_world
        rigid = rigid() if callable(rigid) else rigid
        rotation = np.asarray(rigid.rotation.matrix(), dtype=np.float64)
        center = image.projection_center
        center = center() if callable(center) else center
        # X_cam = R X_world + t, so the world direction imaged as +x is row 0 of R.
        x_axes.append(rotation[0, :])
        centers.append(np.asarray(center, dtype=np.float64))
        names.append(image.name)
    points = np.array([p.xyz for p in rec.points3D.values()], dtype=np.float64)
    if len(x_axes) < 8:
        raise SystemExit(f"need >= 8 registered images, got {len(x_axes)}")
    if len(points) == 0:
        raise SystemExit("model has no 3D points; cannot disambiguate gravity sign")
    return np.array(x_axes), np.array(centers), points, names


def smallest_singular_direction(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    return vt[-1], singular


def solve_gravity_irls(
    x_axes: np.ndarray, iters: int, soft_deg: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Robustly solve for the direction orthogonal to every camera x-axis."""
    soft = np.sin(np.radians(soft_deg))
    weights = np.ones(len(x_axes))
    gravity, singular = smallest_singular_direction(x_axes)
    for _ in range(iters):
        gravity, singular = smallest_singular_direction(x_axes * weights[:, None])
        residual = np.abs(x_axes @ gravity)
        # Huber weight: unit weight inside the knee, 1/r beyond it.
        weights = np.where(residual <= soft, 1.0, soft / np.maximum(residual, 1e-12))
    return gravity, singular, weights


def rotation_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Minimal-angle rotation taking unit `source` onto unit `target`."""
    axis = np.cross(source, target)
    axis_norm = float(np.linalg.norm(axis))
    cos = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if axis_norm < 1e-12:
        if cos > 0:
            return np.eye(3)
        # Antiparallel: rotate pi about any axis orthogonal to source.
        helper = np.array([1.0, 0.0, 0.0])
        if abs(source[0]) > 0.9:
            helper = np.array([0.0, 1.0, 0.0])
        axis = np.cross(source, helper)
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    axis = axis / axis_norm
    angle = float(np.arctan2(axis_norm, cos))
    skew = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def angle_between_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(abs(float(a @ b)), -1.0, 1.0))))


def build(model_dir: Path, args: argparse.Namespace) -> dict:
    x_axes, centers, points, _ = load_model(model_dir)

    gravity, singular, weights = solve_gravity_irls(
        x_axes, args.irls_iters, args.irls_soft_deg
    )

    # Sign: gravity points from the cameras toward what they filmed.
    toward_scene = points.mean(axis=0) - centers.mean(axis=0)
    toward_norm = float(np.linalg.norm(toward_scene))
    sign_margin = (
        abs(float(gravity @ (toward_scene / toward_norm))) if toward_norm > 1e-12 else 0.0
    )
    if toward_norm > 1e-12 and float(gravity @ toward_scene) < 0:
        gravity = -gravity

    roll_resid = np.degrees(np.arcsin(np.clip(np.abs(x_axes @ gravity), -1.0, 1.0)))
    conditioning = float(singular[1] / singular[2]) if singular[2] > 1e-12 else float("inf")
    yaw_diversity = float(singular[1] / singular[0]) if singular[0] > 1e-12 else 0.0

    # Conditional cross-check: camera-centre plane fit.
    centered = centers - centers.mean(axis=0)
    plane_normal, plane_sv = smallest_singular_direction(centered)
    planarity = float(plane_sv[1] / plane_sv[2]) if plane_sv[2] > 1e-12 else float("inf")
    crosscheck_deg = angle_between_deg(gravity, plane_normal)
    crosscheck_applicable = planarity >= args.min_planarity

    gates = {
        "G-GRAV-1a_yaw_diversity": {
            "value": yaw_diversity,
            "threshold": args.min_yaw_diversity,
            "ok": yaw_diversity >= args.min_yaw_diversity,
            "meaning": "s2/s1: the camera x-axes must actually span two dimensions. "
            "A single-heading straight flight leaves s2 and s3 BOTH near zero, so "
            "G-GRAV-1b alone reads as perfectly conditioned. This is the real guard.",
        },
        "G-GRAV-1b_conditioning": {
            "value": conditioning,
            "threshold": args.min_conditioning,
            "ok": conditioning >= args.min_conditioning,
            "meaning": "s2/s3: given a 2D row space, the null direction must be "
            "well separated from it",
        },
        "G-GRAV-2_roll_residual_p95_deg": {
            "value": float(np.percentile(roll_resid, 95)),
            "threshold": args.max_roll_p95_deg,
            "ok": float(np.percentile(roll_resid, 95)) <= args.max_roll_p95_deg,
            "meaning": "the roll==0 assumption must actually hold for the fleet of frames",
        },
        "G-GRAV-3_sign_margin": {
            "value": sign_margin,
            "threshold": args.min_sign_margin,
            "ok": sign_margin >= args.min_sign_margin,
            "meaning": "cameras must sit clearly above what they filmed; a side-on "
            "capture of a vertical structure cannot have its sign guessed",
        },
        "G-GRAV-4_crosscheck_deg": {
            "value": crosscheck_deg,
            "threshold": args.max_crosscheck_deg,
            "planarity": planarity,
            "planarity_threshold": args.min_planarity,
            "applicable": crosscheck_applicable,
            "ok": (crosscheck_deg <= args.max_crosscheck_deg) if crosscheck_applicable else True,
            "meaning": "independent camera-centre plane fit; SKIPPED when the flight "
            "is not planar, because then the plane normal is not gravity",
        },
    }
    ok = all(gate["ok"] for gate in gates.values())

    rotation = rotation_between(gravity, np.array([0.0, 0.0, -1.0]))
    transform = np.eye(4)
    transform[:3, :3] = rotation

    return {
        "schema": SCHEMA,
        "ok": ok,
        "convention": {
            "up_axis": "+Z",
            "gravity_axis": "-Z",
            "applies_as": "X_aligned = R_align @ X_map",
            "handedness": "right",
            "scale": 1.0,
            "metric": False,
            "note": "rotation only; the map stays scale-free. Do not read heights "
            "out of the aligned frame.",
        },
        "gravity_in_map": [float(v) for v in gravity],
        "R_align": [[float(v) for v in row] for row in rotation],
        "T_align_gravity": [[float(v) for v in row] for row in transform],
        "method": {
            "primary": "camera_x_axis_orthogonality_irls",
            "irls_iters": args.irls_iters,
            "irls_soft_deg": args.irls_soft_deg,
            "effective_weight_sum": float(weights.sum()),
            "singular_values": [float(v) for v in singular],
            "yaw_diversity_s2_over_s1": yaw_diversity,
            "roll_residual_deg": {
                "p50": float(np.percentile(roll_resid, 50)),
                "p95": float(np.percentile(roll_resid, 95)),
                "max": float(roll_resid.max()),
            },
            "crosscheck": {
                "name": "camera_center_plane_fit",
                "normal_in_map": [float(v) for v in plane_normal],
                "singular_values": [float(v) for v in plane_sv],
                "planarity_s2_over_s3": planarity,
                "angle_to_primary_deg": crosscheck_deg,
                "applicable": crosscheck_applicable,
            },
        },
        "gates": gates,
        "model": {
            **model_provenance(model_dir),
            "num_images": int(len(x_axes)),
            "num_points3D": int(len(points)),
        },
    }


def verify(existing: dict, fresh: dict, tol_deg: float) -> tuple[bool, str]:
    if existing.get("schema") != SCHEMA:
        return False, f"schema mismatch: {existing.get('schema')!r} != {SCHEMA!r}"
    old = np.asarray(existing["gravity_in_map"], dtype=np.float64)
    new = np.asarray(fresh["gravity_in_map"], dtype=np.float64)
    drift = angle_between_deg(old, new)
    if drift > tol_deg:
        return False, f"gravity drifted {drift:.4f} deg > {tol_deg} deg -- gauge changed, re-derive"
    return True, f"gravity stable within {drift:.4f} deg"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, type=Path, help="COLMAP model dir")
    parser.add_argument("--out", type=Path, help="write T_align_gravity.json here")
    parser.add_argument(
        "--verify",
        type=Path,
        help="re-derive and check an existing T_align_gravity.json still holds "
        "(use after a map update to prove the gauge did not move)",
    )
    parser.add_argument("--verify-tol-deg", type=float, default=0.05)
    parser.add_argument("--min-conditioning", type=float, default=DEF_MIN_CONDITIONING)
    parser.add_argument("--min-yaw-diversity", type=float, default=DEF_MIN_YAW_DIVERSITY)
    parser.add_argument("--max-roll-p95-deg", type=float, default=DEF_MAX_ROLL_P95_DEG)
    parser.add_argument("--min-planarity", type=float, default=DEF_MIN_PLANARITY)
    parser.add_argument("--max-crosscheck-deg", type=float, default=DEF_MAX_CROSSCHECK_DEG)
    parser.add_argument("--min-sign-margin", type=float, default=DEF_MIN_SIGN_MARGIN)
    parser.add_argument("--irls-iters", type=int, default=DEF_IRLS_ITERS)
    parser.add_argument("--irls-soft-deg", type=float, default=DEF_IRLS_SOFT_DEG)
    parser.add_argument(
        "--allow-fail", action="store_true", help="write the JSON even when gates fail"
    )
    args = parser.parse_args(argv)

    if not args.model.is_dir():
        raise SystemExit(f"model dir not found: {args.model}")

    result = build(args.model, args)

    for name, gate in result["gates"].items():
        if name == "G-GRAV-4_crosscheck_deg" and not gate["applicable"]:
            status = "SKIP"
        else:
            status = "PASS" if gate["ok"] else "FAIL"
        print(f"[{status}] {name}: {gate['value']:.4f} (thr {gate['threshold']})")
    print(f"gravity_in_map = {np.round(result['gravity_in_map'], 6).tolist()}")

    exit_code = 0
    if args.verify:
        existing = json.loads(args.verify.read_text(encoding="utf-8"))
        ok, message = verify(existing, result, args.verify_tol_deg)
        print(f"[{'PASS' if ok else 'FAIL'}] verify: {message}")
        if not ok:
            exit_code = 1

    if not result["ok"]:
        print("gates FAILED -- this map cannot ship a gravity alignment", file=sys.stderr)
        if not args.allow_fail:
            return 1
        exit_code = exit_code or 1

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.out}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
