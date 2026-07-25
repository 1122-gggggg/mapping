#!/usr/bin/env python3
"""export_map_views.py

Export a COLMAP sparse model to PLY artifacts for human visual inspection
(trajectories + camera frusta) plus a JSON stats summary.

This is a *read-only inspection* tool: it opens a COLMAP model with
pycolmap and writes PLY/JSON files. It does NOT touch the point cloud
(use `colmap color_extractor` + `colmap model_converter` for that) and it
does NOT run any dense/patch_match/GPU work.

Usage:
    export_map_views.py --model <colmap_model_dir> --out-dir <dir> \
        [--max-center-dist 100.0] [--frustum-scale AUTO|<float>]

Outputs (written into --out-dir):
    trajectories.ply  - one polyline per sequence (PLY `edge` elements)
    frusta.ply        - one camera-frustum square pyramid per registered
                         image (PLY `face` elements)
    stats.json        - per-sequence + global stats, including outlier info

Design notes (see also the accompanying report for the full rationale):
  * pycolmap 4.0.4: `Image.cam_from_world()` is a METHOD (not a property)
    returning a `Rigid3d` with `.rotation` (a `Rotation3d`, `.matrix()` ->
    3x3 ndarray, world-from-... actually cam-from-world rotation R) and
    `.translation` (3-vector t). Camera centre C = -R^T @ t. This was
    cross-checked against `Image.projection_center()` on the target_site
    fixture (7 sequences, 1390 images) and matches exactly
    (np.allclose == True) on every sampled image.
  * Camera axes in world frame: R_wc = R.T (columns = right, down,
    forward of the camera, COLMAP convention: +x right, +y down, +z
    forward/into the scene). `forward = R_wc[:, 2]` was cross-checked
    against `Image.viewing_direction()` and matches.
"""
from __future__ import annotations

import os

# Keep numpy/BLAS single-threaded-ish: a gluemap solve may be running
# concurrently on this machine and this tool must be CPU-light. Must be
# set before numpy is imported.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import argparse
import json
import posixpath
import re
import sys
from dataclasses import dataclass, field

import numpy as np
import pycolmap

# ---------------------------------------------------------------------------
# Fixed, visually-distinct colour palette (Sasha-Trubetskoy-style 10-colour
# set). Assigned to sequences in sorted-name order so runs are deterministic
# and reproducible regardless of image_id ordering.
# ---------------------------------------------------------------------------
PALETTE = [
    (230, 25, 75),   # red
    (60, 180, 75),   # green
    (0, 130, 200),   # blue
    (245, 130, 48),  # orange
    (145, 30, 180),  # purple
    (70, 240, 240),  # cyan
    (240, 50, 230),  # magenta
    (210, 245, 60),  # lime
    (250, 190, 212),  # pink
    (0, 128, 128),   # teal
]


def log(msg: str) -> None:
    print(f"[export_map_views] {msg}", file=sys.stderr)


def die(msg: str) -> None:
    log(f"FATAL: {msg}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Natural sort of filenames by their numeric part(s).
# ---------------------------------------------------------------------------
_NUM_RE = re.compile(r"(\d+)")


def natural_sort_key(basename: str):
    """Split into alternating text/int chunks so '2' < '10' correctly."""
    parts = _NUM_RE.split(basename)
    key = []
    for i, p in enumerate(parts):
        if i % 2 == 1:  # numeric chunk
            key.append((0, int(p)))
        elif p:  # non-empty text chunk
            key.append((1, p))
    return key


def sequence_of(image_name: str) -> str:
    """Sequence = directory part of image.name, e.g.
    'P1270127/000012.jpg' -> 'P1270127'. Images with no directory
    component fall into a single 'ROOT' bucket.
    """
    d = posixpath.dirname(image_name.replace("\\", "/"))
    return d if d else "ROOT"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class CamRecord:
    image_id: int
    name: str
    basename: str
    center: np.ndarray  # (3,) world-space camera centre C
    R_wc: np.ndarray    # (3,3) world-from-camera rotation (columns: right, down, forward)
    width: int
    height: int


@dataclass
class SeqData:
    name: str
    color: tuple
    n_registered: int = 0
    n_unregistered: int = 0
    records: list = field(default_factory=list)  # CamRecord, natural-sorted
    dropped: list = field(default_factory=list)  # CamRecord dropped as outliers


# ---------------------------------------------------------------------------
# PLY writers (ASCII, CloudCompare-compatible header ordering: each
# `element` block is immediately followed by its `property` lines, and the
# element ordering in the header matches the order data blocks appear in
# the body).
# ---------------------------------------------------------------------------
def write_trajectories_ply(path: str, vertices: list, edges: list) -> None:
    """vertices: list of (x,y,z). edges: list of (v1, v2, r, g, b) with
    v1/v2 0-based indices into `vertices`."""
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment export_map_views.py trajectories (one polyline per sequence)\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element edge {len(edges)}\n")
        f.write("property int vertex1\n")
        f.write("property int vertex2\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for x, y, z in vertices:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
        for v1, v2, r, g, b in edges:
            f.write(f"{v1} {v2} {r} {g} {b}\n")


def write_frusta_ply(path: str, vertices: list, faces: list) -> None:
    """vertices: list of (x,y,z). faces: list of (i0,i1,i2,r,g,b) triangles,
    0-based indices into `vertices`."""
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment export_map_views.py camera frusta (one square pyramid per registered image)\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for x, y, z in vertices:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
        for i0, i1, i2, r, g, b in faces:
            f.write(f"3 {i0} {i1} {i2} {r} {g} {b}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Path to a COLMAP sparse model directory")
    ap.add_argument("--out-dir", required=True, help="Directory to write trajectories.ply / frusta.ply / stats.json")
    ap.add_argument("--max-center-dist", type=float, default=100.0,
                     help="Drop cameras farther than this from the median camera centre (map units). Default 100.0")
    ap.add_argument("--frustum-scale", type=str, default="AUTO",
                     help="Frustum pyramid depth. 'AUTO' = 2%% of camera-centre bbox diagonal, or a float.")
    args = ap.parse_args()

    if not os.path.isdir(args.model):
        die(f"--model dir does not exist: {args.model}")

    os.makedirs(args.out_dir, exist_ok=True)

    log(f"Loading COLMAP model: {args.model}")
    try:
        rec = pycolmap.Reconstruction(args.model)
    except Exception as exc:  # noqa: BLE001
        die(f"failed to load COLMAP model at {args.model}: {exc}")

    all_images = rec.images  # ImageMap: image_id -> Image (registered + any without pose)
    log(f"Reconstruction loaded: {len(all_images)} images total, {rec.num_reg_images()} registered, "
        f"{rec.num_cameras()} camera(s), {rec.num_points3D()} points3D")

    registered = [img for img in all_images.values() if img.has_pose]
    unregistered = [img for img in all_images.values() if not img.has_pose]

    if len(registered) == 0:
        die("zero registered images in model — nothing to export")

    # --- Build per-sequence grouping (registered + unregistered counts) ---
    seq_names = sorted({sequence_of(img.name) for img in all_images.values()})
    sequences: dict[str, SeqData] = {
        name: SeqData(name=name, color=PALETTE[i % len(PALETTE)])
        for i, name in enumerate(seq_names)
    }

    for img in unregistered:
        sequences[sequence_of(img.name)].n_unregistered += 1

    # --- Extract pose for every registered image ---
    for img in registered:
        seq = sequences[sequence_of(img.name)]
        seq.n_registered += 1

        cfw = img.cam_from_world()  # method in pycolmap 4.0.4, NOT a property
        R = cfw.rotation.matrix()   # 3x3, world-from... cam_from_world rotation
        t = cfw.translation         # (3,)
        C = -R.T @ t                # camera centre in world coords
        # Cross-checked equal to img.projection_center() on the test fixture.
        R_wc = R.T                  # columns: camera right/down/forward axes in world frame

        cam = rec.camera(img.camera_id)
        basename = posixpath.basename(img.name)
        rec_obj = CamRecord(
            image_id=img.image_id,
            name=img.name,
            basename=basename,
            center=C,
            R_wc=R_wc,
            width=cam.width,
            height=cam.height,
        )
        seq.records.append(rec_obj)

    for seq in sequences.values():
        seq.records.sort(key=lambda r: natural_sort_key(r.basename))
        if seq.n_registered == 1:
            log(f"WARNING: sequence '{seq.name}' has only 1 registered image — "
                f"vertex will be emitted, no edges.")
        elif seq.n_registered == 0:
            log(f"WARNING: sequence '{seq.name}' has 0 registered images "
                f"({seq.n_unregistered} unregistered).")

    # --- Outlier detection: drop cameras far from the global MEDIAN centre ---
    all_centers = np.array([r.center for seq in sequences.values() for r in seq.records])
    median_center = np.median(all_centers, axis=0)
    dropped_info = []
    for seq in sequences.values():
        kept = []
        for r in seq.records:
            dist = float(np.linalg.norm(r.center - median_center))
            if dist > args.max_center_dist:
                seq.dropped.append(r)
                dropped_info.append({
                    "name": r.name,
                    "sequence": seq.name,
                    "center": r.center.tolist(),
                    "dist_from_median": dist,
                })
            else:
                kept.append(r)
        seq.records = kept

    if dropped_info:
        log(f"Dropped {len(dropped_info)} outlier camera(s) (> {args.max_center_dist} from median centre):")
        for d in dropped_info:
            log(f"  - {d['name']} (seq={d['sequence']}, dist={d['dist_from_median']:.3f})")
    else:
        log(f"No outliers beyond --max-center-dist={args.max_center_dist}")

    used_centers = np.array([r.center for seq in sequences.values() for r in seq.records])
    if used_centers.shape[0] == 0:
        die("all registered cameras were dropped as outliers — nothing left to export "
            "(try a larger --max-center-dist)")

    # --- Frustum scale ---
    bbox_min = used_centers.min(axis=0)
    bbox_max = used_centers.max(axis=0)
    bbox_diag = float(np.linalg.norm(bbox_max - bbox_min))
    if args.frustum_scale.strip().upper() == "AUTO":
        frustum_scale = bbox_diag * 0.02
        if frustum_scale <= 0:
            frustum_scale = 1.0
            log("WARNING: camera-centre bbox diagonal is 0 (single point?) — "
                "falling back to frustum-scale=1.0")
        log(f"AUTO frustum-scale: bbox diagonal={bbox_diag:.4f} -> scale={frustum_scale:.4f}")
    else:
        try:
            frustum_scale = float(args.frustum_scale)
        except ValueError:
            die(f"--frustum-scale must be 'AUTO' or a float, got: {args.frustum_scale!r}")
        if frustum_scale <= 0:
            die(f"--frustum-scale must be > 0, got: {frustum_scale}")

    # --- trajectories.ply ---
    traj_vertices = []
    traj_edges = []
    for seq in sequences.values():
        r_col, g_col, b_col = seq.color
        start_idx = len(traj_vertices)
        for r in seq.records:
            traj_vertices.append(tuple(r.center.tolist()))
        n = len(seq.records)
        for i in range(n - 1):
            traj_edges.append((start_idx + i, start_idx + i + 1, r_col, g_col, b_col))

    traj_path = os.path.join(args.out_dir, "trajectories.ply")
    write_trajectories_ply(traj_path, traj_vertices, traj_edges)
    log(f"Wrote {traj_path}: {len(traj_vertices)} vertices, {len(traj_edges)} edges")

    # --- frusta.ply ---
    frusta_vertices = []
    frusta_faces = []
    for seq in sequences.values():
        r_col, g_col, b_col = seq.color
        for r in seq.records:
            aspect = r.width / r.height if r.height else 4.0 / 3.0
            half_w = frustum_scale * 0.5
            half_h = half_w / aspect

            right = r.R_wc[:, 0]
            down = r.R_wc[:, 1]
            up = -down
            forward = r.R_wc[:, 2]

            apex = r.center
            base_center = r.center + forward * frustum_scale
            tl = base_center - right * half_w + up * half_h
            tr = base_center + right * half_w + up * half_h
            br = base_center + right * half_w - up * half_h
            bl = base_center - right * half_w - up * half_h

            base_idx = len(frusta_vertices)
            frusta_vertices.extend([
                tuple(apex.tolist()),
                tuple(tl.tolist()),
                tuple(tr.tolist()),
                tuple(br.tolist()),
                tuple(bl.tolist()),
            ])
            a, itl, itr, ibr, ibl = base_idx, base_idx + 1, base_idx + 2, base_idx + 3, base_idx + 4
            # 4 side faces + 2 base faces (closed square pyramid)
            frusta_faces.append((a, itl, itr, r_col, g_col, b_col))
            frusta_faces.append((a, itr, ibr, r_col, g_col, b_col))
            frusta_faces.append((a, ibr, ibl, r_col, g_col, b_col))
            frusta_faces.append((a, ibl, itl, r_col, g_col, b_col))
            frusta_faces.append((itl, itr, ibr, r_col, g_col, b_col))
            frusta_faces.append((itl, ibr, ibl, r_col, g_col, b_col))

    frusta_path = os.path.join(args.out_dir, "frusta.ply")
    write_frusta_ply(frusta_path, frusta_vertices, frusta_faces)
    log(f"Wrote {frusta_path}: {len(frusta_vertices)} vertices, {len(frusta_faces)} faces")

    # --- stats.json ---
    seq_stats = {}
    for seq in sequences.values():
        centers = np.array([r.center for r in seq.records]) if seq.records else np.zeros((0, 3))
        if centers.shape[0] > 0:
            c_bbox = {"min": centers.min(axis=0).tolist(), "max": centers.max(axis=0).tolist()}
        else:
            c_bbox = {"min": None, "max": None}

        path_length = 0.0
        for i in range(len(seq.records) - 1):
            path_length += float(np.linalg.norm(seq.records[i + 1].center - seq.records[i].center))

        straightness = None
        if len(seq.records) >= 2:
            net_disp = float(np.linalg.norm(seq.records[-1].center - seq.records[0].center))
            straightness = (net_disp / path_length) if path_length > 0 else None

        seq_stats[seq.name] = {
            "n_registered": seq.n_registered,
            "n_unregistered": seq.n_unregistered,
            "n_used_in_viz": len(seq.records),
            "n_outliers_dropped": len(seq.dropped),
            "center_bbox": c_bbox,
            "path_length": path_length,
            "straightness": straightness,
            "color_rgb": list(seq.color),
        }

    stats = {
        "model_dir": os.path.abspath(args.model),
        "out_dir": os.path.abspath(args.out_dir),
        "max_center_dist": args.max_center_dist,
        "frustum_scale_used": frustum_scale,
        "global": {
            "n_images_total": len(all_images),
            "n_registered_total": len(registered),
            "n_unregistered_total": len(unregistered),
            "n_sequences": len(sequences),
            "median_camera_center": median_center.tolist(),
            "bbox_used_cameras": {"min": bbox_min.tolist(), "max": bbox_max.tolist(), "diagonal": bbox_diag},
            "outliers": {
                "count": len(dropped_info),
                "threshold": args.max_center_dist,
                "dropped": dropped_info,
            },
        },
        "sequences": seq_stats,
    }

    stats_path = os.path.join(args.out_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    log(f"Wrote {stats_path}")

    log("Done.")


if __name__ == "__main__":
    main()
