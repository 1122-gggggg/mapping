#!/usr/bin/env python3
"""Camera models for the target_site build, and the COLMAP intrinsics seed.

THREE SOURCES DISAGREE ABOUT THIS DRONE'S FOCAL LENGTH BY UP TO 7.5%:

    source                              fx/W        implied HFOV
    official69 (the original spec)      0.727505    69.0 deg
    Charuco (原始估計內參.json, 720p)     0.750379    67.3 deg
    river_site map, BA-converged        0.781880    65.2 deg

The river number is not even comparable: those frames were cv2.undistort-ed with
the Charuco K+dist, so it is the focal of a RECTIFIED image, not of a raw one.

The Charuco calibration is weak -- 25 frames, a small 5x7 board, RMS 0.83px, cx
off-centre by +30.8px, and k2 = +0.256. Yet official69 PINHOLE with NO undistortion
empirically works on this drone (LoMa forward map: 267/267 localization). Imagery
with that much barrel distortion could not do that, so the distortion coefficients
are almost certainly overfit noise -- and because focal and distortion are strongly
correlated in a Charuco solve, the FOCAL is suspect for the same reason.

So neither is trusted. We bake off all three and let held-out localization decide.

MODEL: PINHOLE, always. It carries fx != fy and an off-centre principal point;
SIMPLE_PINHOLE carries neither, and candidate B needs both.

NO UNDISTORTION, EVER. The Anafi already delivers a near-rectilinear image, which
is exactly why a pure pinhole model works on it. Distortion coefficients are
therefore DISCARDED, not applied -- never "scaled" (they do not scale with
resolution; only focal and principal point do).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from ts_common import Video, log, resolution_groups, write_json

# --- candidate A: official69 -------------------------------------------------
OFFICIAL_HFOV_DEG = 69.0
# 0.5 / tan(34.5deg) = 0.7275050. (An earlier note said 0.727531 -- that was a
# rounding slip corresponding to HFOV 68.998deg.)
OFFICIAL_FX_OVER_W = 0.5 / math.tan(math.radians(OFFICIAL_HFOV_DEG / 2))

# --- candidate B: Charuco, measured at 1280x720 ------------------------------
# configs/mapping/原始估計內參.json
CHARUCO_BASE_W, CHARUCO_BASE_H = 1280, 720
CHARUCO_FX, CHARUCO_FY = 960.4853099760471, 958.1961747147875
CHARUCO_CX, CHARUCO_CY = 670.8167651412149, 358.7191813450141
# dist = [-0.01636, 0.25634, -0.00610, 0.01951, -0.11986] -- DELIBERATELY UNUSED.


@dataclass(frozen=True)
class Camera:
    """A PINHOLE camera for one resolution group."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def params(self) -> tuple[float, float, float, float]:
        return (self.fx, self.fy, self.cx, self.cy)

    @property
    def fx_over_w(self) -> float:
        return self.fx / self.width

    @property
    def hfov_deg(self) -> float:
        return 2 * math.degrees(math.atan(self.width / (2 * self.fx)))

    def colmap_line(self, camera_id: int) -> str:
        return (
            f"{camera_id} PINHOLE {self.width} {self.height} "
            f"{self.fx:.12f} {self.fy:.12f} {self.cx:.12f} {self.cy:.12f}"
        )


def official69(width: int, height: int) -> Camera:
    fx = OFFICIAL_FX_OVER_W * width
    return Camera(width, height, fx, fx, width / 2, height / 2)


def charuco_scaled(width: int, height: int) -> Camera:
    """Scale the 720p Charuco solve to `width`.

    Focal AND principal point scale linearly with resolution. Distortion does
    NOT -- and here it is discarded outright rather than scaled, because the
    build feeds raw (never undistorted) frames to a PINHOLE model.

    Every source is 16:9, so one isotropic factor is exact on both axes.
    """
    sx, sy = width / CHARUCO_BASE_W, height / CHARUCO_BASE_H
    if abs(sx - sy) > 1e-9:
        raise ValueError(
            f"{width}x{height} is not 16:9 -- an anisotropic rescale of the "
            "Charuco solve is not a valid pinhole camera"
        )
    return Camera(
        width, height,
        CHARUCO_FX * sx, CHARUCO_FY * sy,
        CHARUCO_CX * sx, CHARUCO_CY * sy,
    )


def from_fx_over_w(width: int, height: int, fx_over_w: float) -> Camera:
    """Candidate C: a focal MEASURED by a --refine_intrinsics BA pass."""
    fx = fx_over_w * width
    return Camera(width, height, fx, fx, width / 2, height / 2)


CANDIDATES = {"official69": official69, "charuco": charuco_scaled}


def cameras_for(candidate: str, videos: list[Video] | None = None) -> dict[tuple[int, int], Camera]:
    """One PINHOLE camera per resolution group."""
    if candidate not in CANDIDATES:
        raise ValueError(f"unknown candidate {candidate!r}; have {sorted(CANDIDATES)}")
    make = CANDIDATES[candidate]
    return {shape: make(*shape) for shape in resolution_groups(videos)}


# ---------------------------------------------------------------- COLMAP seed
def write_intrinsics_seed(
    seed_dir: Path,
    image_names: list[str],
    shape_of: dict[str, tuple[int, int]],
    cameras: dict[tuple[int, int], Camera],
) -> dict:
    """Write a COLMAP text model that seeds one camera PER RESOLUTION GROUP.

    gluemap reads this via ``extract_gt_intrinsics``, which resolves each image
    BY NAME and then fills each of its own shape-buckets from the first image it
    matches. So the only thing that must hold is: every image of a given shape is
    assigned the same camera. Writing one camera per shape guarantees it, and
    gluemap logs ``GT intrinsics loaded: N/M cameras matched`` -- gate on N == M.

    Note the old single-camera seed hardcoded ``camera_id=1`` for every image
    (run_gluemap_site_build.py:219), which silently gives 4K frames the intrinsics
    of a 1080p camera.
    """
    seed_dir.mkdir(parents=True, exist_ok=True)
    cam_id_of = {shape: i + 1 for i, shape in enumerate(sorted(cameras))}

    lines = [
        "# Camera list with one line of data per camera:",
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
    ]
    for shape in sorted(cameras):
        lines.append(cameras[shape].colmap_line(cam_id_of[shape]))
    (seed_dir / "cameras.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    img_lines = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
    ]
    per_cam: dict[int, int] = {}
    for idx, name in enumerate(image_names, 1):
        shape = shape_of[name]
        if shape not in cam_id_of:
            raise KeyError(f"{name}: shape {shape} has no seeded camera")
        cid = cam_id_of[shape]
        per_cam[cid] = per_cam.get(cid, 0) + 1
        # Identity pose: only the camera assignment is read from this seed.
        img_lines.append(f"{idx} 1 0 0 0 0 0 0 {cid} {name}")
        img_lines.append("")
    (seed_dir / "images.txt").write_text("\n".join(img_lines) + "\n", encoding="utf-8")
    (seed_dir / "points3D.txt").write_text("# 3D point list\n", encoding="utf-8")

    summary = {
        "seed_dir": str(seed_dir),
        "camera_model": "PINHOLE",
        "undistortion": "NONE -- raw resized frames",
        "n_cameras": len(cameras),
        "n_images": len(image_names),
        "cameras": [
            {
                "camera_id": cam_id_of[shape],
                "width": c.width,
                "height": c.height,
                "fx": c.fx, "fy": c.fy, "cx": c.cx, "cy": c.cy,
                "fx_over_w": c.fx_over_w,
                "hfov_deg": c.hfov_deg,
                "n_images": per_cam.get(cam_id_of[shape], 0),
            }
            for shape, c in sorted(cameras.items())
        ],
    }
    write_json(seed_dir.parent / "intrinsics_seed.json", summary)
    return summary


if __name__ == "__main__":
    print(f"official69: fx/W = {OFFICIAL_FX_OVER_W:.7f}  (HFOV {OFFICIAL_HFOV_DEG} deg)")
    ch = charuco_scaled(CHARUCO_BASE_W, CHARUCO_BASE_H)
    print(f"charuco   : fx/W = {ch.fx_over_w:.7f}  (HFOV {ch.hfov_deg:.2f} deg)")
    print(f"disagreement: {abs(ch.fx_over_w - OFFICIAL_FX_OVER_W) / OFFICIAL_FX_OVER_W * 100:.2f}%\n")
    for cand in CANDIDATES:
        print(f"--- {cand}")
        for shape, c in cameras_for(cand).items():
            print(
                f"  {shape[0]:>4d}x{shape[1]:<4d}  fx={c.fx:9.3f} fy={c.fy:9.3f} "
                f"cx={c.cx:9.3f} cy={c.cy:8.3f}  HFOV={c.hfov_deg:.2f}"
            )
