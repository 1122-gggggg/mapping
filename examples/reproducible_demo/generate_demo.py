#!/usr/bin/env python3
"""Generate a small, deterministic MapDoctor demonstration dataset.

The generated sparse reconstruction is synthetic and intentionally producer-neutral:
the same COLMAP-format model can be loaded through the COLMAP, GLOMAP, and GLUEMAP
interfaces. Two frozen-query benchmark CSVs demonstrate a clean base map and a
candidate with two deliberate localization regressions.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


NUM_IMAGES = 8
NUM_POINTS = 80
NUM_QUERIES = 20


def _point_xy(index: int) -> tuple[float, float]:
    cols = 10
    row, col = divmod(index, cols)
    x = 50.0 + col * 58.0
    y = 45.0 + row * 52.0
    return x, y


def write_sparse_model(root: Path) -> Path:
    model = root / "sparse" / "0"
    model.mkdir(parents=True, exist_ok=True)

    (model / "cameras.txt").write_text(
        "# CAMERA_ID MODEL WIDTH HEIGHT PARAMS\n"
        "1 PINHOLE 640 480 500 500 320 240\n",
        encoding="utf-8",
    )

    image_lines = ["# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n"]
    for image_id in range(1, NUM_IMAGES + 1):
        center_x = (image_id - 1) * 0.8
        tx = -center_x
        image_lines.append(
            f"{image_id} 1 0 0 0 {tx:.6f} 0 0 1 ref_{image_id:02d}.jpg\n"
        )
        observations = []
        for point_id in range(1, NUM_POINTS + 1):
            x, y = _point_xy(point_id - 1)
            # Small deterministic image-to-image shift prevents identical image-space support.
            x += 1.5 * (image_id - 1)
            observations.extend((f"{x:.3f}", f"{y:.3f}", str(point_id)))
        image_lines.append(" ".join(observations) + "\n")
    (model / "images.txt").write_text("".join(image_lines), encoding="utf-8")

    point_lines = ["# POINT3D_ID X Y Z R G B ERROR TRACK[]\n"]
    for point_id in range(1, NUM_POINTS + 1):
        x2d, y2d = _point_xy(point_id - 1)
        x3d = (x2d - 320.0) / 100.0
        y3d = (y2d - 240.0) / 100.0
        z3d = 5.0 + 0.2 * math.sin(point_id)
        track = []
        point2d_idx = point_id - 1
        for image_id in range(1, NUM_IMAGES + 1):
            track.extend((str(image_id), str(point2d_idx)))
        point_lines.append(
            f"{point_id} {x3d:.6f} {y3d:.6f} {z3d:.6f} 180 180 180 0.6 "
            + " ".join(track)
            + "\n"
        )
    (model / "points3D.txt").write_text("".join(point_lines), encoding="utf-8")
    return model


def _benchmark_row(index: int, *, regressed: bool) -> dict[str, object]:
    query = f"query_{index:03d}.jpg"
    if regressed:
        return {
            "query": query,
            "success": False,
            "inliers": 8,
            "inlier_ratio": 0.08,
            "reproj_p90_px": "",
            "hull_coverage": 0.06,
            "grid4_occupancy": 3,
            "positive_depth_ratio": 0.75,
            "pose_consensus": 0.25,
            "x": float(index),
            "y": 0.0,
            "z": 1.5,
        }
    return {
        "query": query,
        "success": True,
        "inliers": 55 + (index % 11),
        "inlier_ratio": 0.45 + 0.01 * (index % 6),
        "reproj_p90_px": 1.1 + 0.04 * (index % 8),
        "hull_coverage": 0.28 + 0.01 * (index % 5),
        "grid4_occupancy": 9 + (index % 3),
        "positive_depth_ratio": 1.0,
        "pose_consensus": 0.85 + 0.01 * (index % 5),
        "x": float(index),
        "y": 0.0,
        "z": 1.5,
    }


def write_benchmarks(root: Path) -> tuple[Path, Path]:
    fields = [
        "query",
        "success",
        "inliers",
        "inlier_ratio",
        "reproj_p90_px",
        "hull_coverage",
        "grid4_occupancy",
        "positive_depth_ratio",
        "pose_consensus",
        "x",
        "y",
        "z",
    ]
    base = root / "base.csv"
    candidate = root / "candidate.csv"
    for path, bad_indices in ((base, set()), (candidate, {7, 15})):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for index in range(NUM_QUERIES):
                writer.writerow(_benchmark_row(index, regressed=index in bad_indices))
    return base, candidate


def generate(output: Path) -> dict[str, Path]:
    output.mkdir(parents=True, exist_ok=True)
    model = write_sparse_model(output)
    base, candidate = write_benchmarks(output)
    return {"model": model, "base": base, "candidate": candidate}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("mapdoctor-demo-data"))
    args = parser.parse_args()
    artifacts = generate(args.output.resolve())
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
