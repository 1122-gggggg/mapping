#!/usr/bin/env python3
"""Colorize a COLMAP sparse model and write an RGB PLY.

Works directly through pycolmap, so it does not need the COLMAP CLI. That
matters here: pycolmap 4.0.x writes rigs.bin/frames.bin, which the system
COLMAP 3.9.1 binary cannot parse (color_extractor/model_converter crash).

Colour per 3D point = per-channel MEDIAN over its track observations, which
is robust to a few bad samples (occlusion, moving objects, JPEG ringing).
"""
import argparse, collections, os
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path
import numpy as np, cv2, pycolmap


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--image-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-dist-cam-diagonals", type=float, default=10.0,
                    help=(
                        "drop points further than N x the camera-bbox diagonal from the "
                        "median camera centre. Scale-relative on purpose: monocular SfM "
                        "maps have arbitrary scale, so an absolute cut tuned on one site "
                        "does not transfer."
                    ))
    ap.add_argument("--outlier-out", type=Path, default=None)
    args = ap.parse_args()

    rec = pycolmap.Reconstruction(str(args.model))
    print(f"model: {rec.num_reg_images()} images, {rec.num_points3D()} points")

    samples = collections.defaultdict(list)
    for n, im in enumerate(rec.images.values(), 1):
        if not im.has_pose:
            continue
        path = args.image_root / im.name
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise SystemExit(f"unreadable image: {path}")
        h, w = bgr.shape[:2]
        for p2 in im.points2D:
            if not p2.has_point3D():
                continue
            x, y = int(round(p2.xy[0])), int(round(p2.xy[1]))
            if 0 <= x < w and 0 <= y < h:
                b, g, r = bgr[y, x]
                samples[p2.point3D_id].append((r, g, b))
        if n % 100 == 0:
            print(f"  sampled {n} images", flush=True)

    ids, xyz, rgb = [], [], []
    for pid, pt in rec.points3D.items():
        s = samples.get(pid)
        if not s:
            continue
        ids.append(pid)
        xyz.append(pt.xyz)
        rgb.append(np.median(np.asarray(s, dtype=np.float32), axis=0))
    xyz = np.asarray(xyz, dtype=np.float64)
    rgb = np.clip(np.asarray(rgb), 0, 255).astype(np.uint8)
    print(f"coloured {len(xyz)}/{rec.num_points3D()} points")

    centres = np.asarray([-im.cam_from_world().rotation.matrix().T
                          @ im.cam_from_world().translation
                          for im in rec.images.values() if im.has_pose])
    med = np.median(centres, axis=0)
    cam_diag = float(np.linalg.norm(centres.max(0) - centres.min(0)))
    limit = args.max_dist_cam_diagonals * cam_diag
    keep = np.linalg.norm(xyz - med, axis=1) <= limit
    print(f"camera bbox diagonal = {cam_diag:.3f}; "
          f"cut = {args.max_dist_cam_diagonals}x = {limit:.3f} map units")
    print(f"outliers beyond cut: {int((~keep).sum())} "
          f"(kept {int(keep.sum())}, {keep.mean():.4%})")

    def write(path: Path, P: np.ndarray, C: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(P)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for (x, y, z), (r, g, b) in zip(P, C):
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
        print(f"wrote {path}: {len(P)} points")

    write(args.out, xyz[keep], rgb[keep])
    if args.outlier_out is not None and (~keep).any():
        write(args.outlier_out, xyz[~keep], rgb[~keep])


if __name__ == "__main__":
    main()
