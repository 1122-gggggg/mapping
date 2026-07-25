#!/usr/bin/env python3
"""Camera model and COLMAP intrinsics seed for the Fuhe Bridge v2 adapter.

P0 locks a single 1920x1080 PINHOLE camera at
``fx=fy=1396.8086675255472, cx=960, cy=540``. Frames are resized to that exact
canvas with INTER_AREA and are never undistorted. Historical camera constructors
remain importable only to inspect v1 evidence; ``cameras_for`` exposes only the
fixed v2 candidate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from ts_common import Video, WORKING_HEIGHT, WORKING_WIDTH, write_json


FUHE_FX = 1396.8086675255472
FUHE_FY = 1396.8086675255472
FUHE_CX = 960.0
FUHE_CY = 540.0

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


def fuhe_v2_fixed(width: int, height: int) -> Camera:
    """Return the only P0 camera; scaling it would violate the fixed adapter."""
    if (width, height) != (WORKING_WIDTH, WORKING_HEIGHT):
        raise ValueError(
            f"Fuhe v2 intrinsics are locked to {WORKING_WIDTH}x{WORKING_HEIGHT}, "
            f"not {width}x{height}"
        )
    return Camera(width, height, FUHE_FX, FUHE_FY, FUHE_CX, FUHE_CY)


CANDIDATES = {"fuhe_v2_fixed": fuhe_v2_fixed}


def cameras_for(candidate: str, videos: list[Video] | None = None) -> dict[tuple[int, int], Camera]:
    """Return the single fixed working-resolution PINHOLE camera."""
    if candidate not in CANDIDATES:
        raise ValueError(f"unknown candidate {candidate!r}; have {sorted(CANDIDATES)}")
    make = CANDIDATES[candidate]
    return {
        (WORKING_WIDTH, WORKING_HEIGHT): make(WORKING_WIDTH, WORKING_HEIGHT)
    }


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
