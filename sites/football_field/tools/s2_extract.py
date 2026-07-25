#!/usr/bin/env python3
"""S2 -- motion-adaptive frame extraction. NO undistortion, NO forced resize.

The old extractor (run_gluemap_site_build.py:140) did this to every frame:

    resized    = cv2.resize(frame, (1280, 720))
    undistorted = cv2.undistort(resized, k, dist, None, k)

Both are gone. Frames are written at their NATIVE resolution, exactly as decoded.

WHY NO UNDISTORTION
  The Anafi already delivers a near-rectilinear image -- which is why a pure
  PINHOLE model works on it at all (LoMa forward map: 267/267 localization). The
  Charuco distortion coefficients (k2 = +0.256) are almost certainly overfit noise;
  "correcting" with them would BEND a straight image. Feed the raw pixels.

WHY NATIVE RESOLUTION
  cameras.bin width MUST equal the on-disk image width, because EDM derives its
  keypoint scale from it (`scale_of = camera.width / EDM_W`,
  build_reloc_map_edm.py:386). Resize the frames without resizing the camera and
  every EDM correspondence lands in the wrong place.

  Keeping the three groups native also keeps them three DISTINCT cameras, which is
  the whole point: we do not yet know whether they share a focal length. Once S2b
  measures that, a common canvas may become available -- but not before.

SAMPLING (driven by S1's motion manifest)
  parallax / low_parallax  full rate         -- the structure comes from here
  hover                    heavily decimated -- a static camera adds no baseline,
                                               only redundant views and BA weight
  pure_rotation / unproven sparse, BRIDGE_ONLY -- current metadata only. Future S5
                                               must enforce observation-level
                                               exclusion and prove it separately.
  fast_motion              DROPPED           -- motion blur poisons tracks
"""
from __future__ import annotations

import argparse
import bisect
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ts_common import (  # noqa: E402
    BUILD, BUILD_BY_SEQ, RUNS, Gate, log, make_lineage_record, read_json,
    required_check_ids, sha256, stage_material_artifacts, write_json,
)
from ts_intrinsics import (  # noqa: E402
    CHARUCO_BASE_W, CHARUCO_CX, CHARUCO_CY, CHARUCO_FX, CHARUCO_FY,
    cameras_for, write_intrinsics_seed,
)
import s1_motion_scan as s1  # noqa: E402


TS_COMMON = Path(__file__).with_name("ts_common.py")
TS_INTRINSICS = Path(__file__).with_name("ts_intrinsics.py")
S1_SOURCE = Path(s1.__file__).resolve()

# Sampling rate per motion class, in frames per second of source video.
RATE = {
    "parallax": 1.0,
    "low_parallax": 1.0,
    "hover": 0.2,          # 1 frame per 5 s: enough to stay registered, no more
    "pure_rotation": 0.5,
    "unproven": 0.5,
    "fast_motion": 0.0,    # dropped
}
BRIDGE_ONLY_CLASSES = {"pure_rotation", "unproven"}
MAX_HOVER_RATIO = 0.02
EXPECTED_FRAME_COUNTS = {
    "S01_ABrot": 392,
    "S02_BA": 104,
    "S03_BA2": 110,
    "S04_ab": 252,
    "S05_P1220122": 173,
    "S06_P1240124": 245,
    "S07_P1250125": 138,
}

# The Charuco distortion, used ONLY to prove we did not apply it (G2.1).
CHARUCO_DIST = np.array([-0.016359355216362784, 0.256336300878371,
                         -0.006099082030819077, 0.019509803298460405,
                         -0.1198628127364991])


def class_at(records: list[dict], times: list[float], t: float) -> str:
    """Motion class of the probe interval containing `t`."""
    i = bisect.bisect_right(times, t) - 1
    return records[max(i, 0)]["motion_class"] if records else "parallax"


def plan_frames(seq: str, mm: dict) -> list[tuple[float, str]]:
    """(timestamp, motion_class) for every frame we intend to write."""
    s = mm["sequences"][seq]
    recs = s["records"]
    times = [r["t"] for r in recs]
    dur = s["duration"]

    out, t = [], 0.0
    while t < dur:
        c = class_at(recs, times, t)
        rate = RATE[c]
        if rate <= 0:                       # fast_motion: skip this stretch
            t += 1.0 / RATE["parallax"]
            continue
        out.append((t, c))
        t += 1.0 / rate
    return cap_hover_and_backfill(seq, out, recs)


def cap_hover_and_backfill(
    seq: str,
    planned: list[tuple[float, str]],
    records: list[dict],
) -> list[tuple[float, str]]:
    """Cap hover per sequence without changing the audited frame-count vector."""
    target_n = EXPECTED_FRAME_COUNTS[seq]
    if len(planned) != target_n:
        raise ValueError(f"{seq}: pre-cap count {len(planned)} != {target_n}")

    hover = [item for item in planned if item[1] == "hover"]
    non_hover = [item for item in planned if item[1] != "hover"]
    keep_n = min(len(hover), math.floor(MAX_HOVER_RATIO * target_n))
    keep_idx = (
        {(i * len(hover)) // keep_n for i in range(keep_n)} if keep_n else set()
    )
    selected = non_hover + [item for i, item in enumerate(hover) if i in keep_idx]
    selected_t = {round(t, 6) for t, _ in selected}
    candidates = [
        (float(record["t"]), record["motion_class"])
        for record in records
        if record["motion_class"] in {"parallax", "low_parallax"}
        and round(float(record["t"]), 6) not in selected_t
    ]

    while len(selected) < target_n:
        if not candidates:
            raise ValueError(f"{seq}: insufficient structural backfills")
        t, cls = max(
            candidates,
            key=lambda item: (
                min(abs(item[0] - chosen_t) for chosen_t, _ in selected),
                -item[0],
            ),
        )
        selected.append((t, cls))
        candidates.remove((t, cls))

    selected.sort(key=lambda item: item[0])
    if len(selected) != target_n:
        raise AssertionError(f"{seq}: post-cap count changed")
    if len({round(t, 6) for t, _ in selected}) != target_n:
        raise AssertionError(f"{seq}: duplicate timestamps after hover backfill")
    hover_ratio = sum(cls == "hover" for _, cls in selected) / target_n
    if hover_ratio > MAX_HOVER_RATIO:
        raise AssertionError(f"{seq}: hover ratio {hover_ratio:.4f} exceeds cap")
    return selected


def extract(
    seq: str,
    plan: list[tuple[float, str]],
    images_dir: Path,
    jpeg_quality: int,
    source_lineage: dict[str, str],
) -> list[dict]:
    v = BUILD_BY_SEQ[seq]
    out_dir = images_dir / seq
    out_dir.mkdir(parents=True, exist_ok=True)

    want = np.array([t for t, _ in plan])
    cls = [c for _, c in plan]
    frames = []
    n = 0
    # width=v.width means decode_at does NOT resize: it only resizes when the
    # decoded width differs from the requested one.
    for i, (t, img) in enumerate(s1.decode_at(v, want, gray=False, width=v.width)):
        h, w = img.shape[:2]
        if (w, h) != (v.width, v.height):
            raise SystemExit(
                f"{seq}: decoded {w}x{h}, expected {v.width}x{v.height} -- a "
                "resize slipped in, and cameras.bin would no longer match the "
                "images EDM reads"
            )
        n += 1
        name = f"{seq}/{n:06d}.jpg"
        image_path = images_dir / name
        if not cv2.imwrite(
            str(image_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        ):
            raise OSError(f"failed to write {image_path}")
        frames.append({
            "name": name, "seq": seq, "t": float(t),
            **source_lineage,
            "image_sha256": sha256(image_path),
            "width": w, "height": h,
            "motion_class": cls[i],
            "role": "bridge_only" if cls[i] in BRIDGE_ONLY_CLASSES else "triangulation",
        })
    return frames


def locked_source_lineage(corpus: dict) -> dict[str, dict[str, str]]:
    """Revalidate S0's seven locked source hashes once, then reuse per frame."""
    locked = {record["seq"]: record for record in corpus.get("build", [])}
    if set(locked) != set(BUILD_BY_SEQ):
        raise ValueError(
            f"corpus build sequences {sorted(locked)} != {sorted(BUILD_BY_SEQ)}"
        )
    out: dict[str, dict[str, str]] = {}
    for video in BUILD:
        record = locked[video.seq]
        lineage = make_lineage_record(video.path, source_rel=video.rel)
        locked_hash = record.get("source_sha256", record.get("sha256"))
        if locked_hash != lineage["source_sha256"]:
            raise ValueError(f"{video.seq}: source SHA-256 differs from S0 lock")
        if record.get("source_rel", record.get("rel")) != lineage["source_rel"]:
            raise ValueError(f"{video.seq}: source path differs from S0 lock")
        out[video.seq] = lineage
    return out


def inspect_written_frames(
    images_dir: Path, frames: list[dict]
) -> tuple[dict[str, tuple[int, int]], list[str], list[str]]:
    """Read every persisted JPEG back and verify its recorded digest."""
    actual_shapes: dict[str, tuple[int, int]] = {}
    unreadable: list[str] = []
    hash_mismatches: list[str] = []
    for frame in frames:
        path = images_dir / frame["name"]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            unreadable.append(frame["name"])
            continue
        h, w = image.shape[:2]
        actual_shapes[frame["name"]] = (w, h)
        if sha256(path) != frame.get("image_sha256"):
            hash_mismatches.append(frame["name"])
    return actual_shapes, unreadable, hash_mismatches


def read_seed_evidence(seed_dir: Path) -> tuple[dict[int, tuple[int, int]], set[str]]:
    """Parse the persisted COLMAP text seed instead of trusting write-time data."""
    cameras: dict[int, tuple[int, int]] = {}
    for line in (seed_dir / "cameras.txt").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 8 or fields[1] != "PINHOLE":
            raise ValueError(f"invalid PINHOLE camera line: {line}")
        cameras[int(fields[0])] = (int(fields[2]), int(fields[3]))

    image_names: set[str] = set()
    for line in (seed_dir / "images.txt").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 10:
            raise ValueError(f"invalid image line: {line}")
        image_names.add(fields[9])
    return cameras, image_names


def role_partition(all_frames: list[dict]) -> tuple[bool, dict[str, int], list[str]]:
    """Evaluate only the current role labels; make no future S5 assertion."""
    roles: dict[str, int] = {}
    bad: list[str] = []
    for frame in all_frames:
        role = frame["role"]
        roles[role] = roles.get(role, 0) + 1
        expected = (
            "bridge_only"
            if frame["motion_class"] in BRIDGE_ONLY_CLASSES
            else "triangulation"
        )
        if role != expected:
            bad.append(frame["name"])
    ok = (
        sum(roles.values()) == len(all_frames)
        and set(roles) <= {"triangulation", "bridge_only"}
        and not bad
    )
    return ok, roles, bad


def prove_no_undistortion(gate: Gate, images_dir: Path, frames: list[dict],
                          n_samples: int = 6) -> None:
    """G2.1 -- a DISCRIMINATIVE test, not a code inspection.

    Re-decode the source frame, and compare the written JPEG against two
    hypotheses: the raw frame, and the same frame put through cv2.undistort with
    the Charuco K+dist. If we had (or ever accidentally) undistorted, the written
    image would sit closer to the second. Asserting only "we didn't call
    cv2.undistort" would prove nothing about what actually landed on disk.
    """
    rng = np.random.default_rng(0)
    picks = [frames[i] for i in rng.choice(len(frames), size=min(n_samples, len(frames)),
                                           replace=False)]
    rows = []
    for f in picks:
        v = BUILD_BY_SEQ[f["seq"]]
        raw = next(img for _t, img in s1.decode_at(
            v, np.array([f["t"]]), gray=False, width=v.width))
        written = cv2.imread(str(images_dir / f["name"]), cv2.IMREAD_COLOR)

        s = v.width / CHARUCO_BASE_W
        K = np.array([[CHARUCO_FX * s, 0, CHARUCO_CX * s],
                      [0, CHARUCO_FY * s, CHARUCO_CY * s], [0, 0, 1]])
        undist = cv2.undistort(raw, K, CHARUCO_DIST, None, K)

        d_raw = float(np.abs(written.astype(np.int16) - raw.astype(np.int16)).mean())
        d_und = float(np.abs(written.astype(np.int16) - undist.astype(np.int16)).mean())
        rows.append({"name": f["name"], "mad_vs_raw": d_raw, "mad_vs_undistorted": d_und})

    worst = max(r["mad_vs_raw"] / max(r["mad_vs_undistorted"], 1e-9) for r in rows)
    ok = all(r["mad_vs_raw"] < 0.35 * r["mad_vs_undistorted"] for r in rows)
    gate.check(
        "G2.1", ok,
        (f"written frames match the RAW decode, not an undistorted one "
         f"(worst raw/undistorted MAD ratio {worst:.3f}; JPEG noise only)")
        if ok else f"frames look UNDISTORTED: {rows}",
        samples=rows,
    )


def stage_gate(run_dir: Path) -> Gate:
    gate = Gate(
        "S2_extract",
        required_check_ids("S2_extract"),
        script_path=Path(__file__),
        input_artifacts=stage_material_artifacts("S2_extract", run_dir),
        source_files=[TS_COMMON, TS_INTRINSICS, S1_SOURCE],
    )
    gate.record_predecessor_gate(
        "S1_motion", run_dir / "gates" / "S1_motion.json", expected_stage="S1_motion"
    )
    return gate


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-name", default="football_field_v1")
    ap.add_argument("--jpeg-quality", type=int, default=95)
    ap.add_argument("--candidate", default="official69",
                    help="which intrinsics candidate to seed (official69 | charuco)")
    args = ap.parse_args()

    run_dir = RUNS / args.run_name
    images_dir = run_dir / "images"
    if images_dir.exists() and any(images_dir.iterdir()):
        raise SystemExit(
            f"refusing to extract into non-empty image root: {images_dir}; "
            "S2 requires a fresh run root"
        )
    motion_path = run_dir / "motion_manifest.json"
    corpus_path = run_dir / "corpus_manifest.json"
    mm = read_json(motion_path)
    corpus = read_json(corpus_path)
    lineage_by_seq = locked_source_lineage(corpus)
    log(f"S2 extract -> {images_dir}")

    all_frames: list[dict] = []
    for v in BUILD:
        plan = plan_frames(v.seq, mm)
        t0 = time.perf_counter()
        fs = extract(
            v.seq, plan, images_dir, args.jpeg_quality, lineage_by_seq[v.seq]
        )
        all_frames += fs
        counts: dict[str, int] = {}
        for f in fs:
            counts[f["motion_class"]] = counts.get(f["motion_class"], 0) + 1
        log(f"  {v.seq}: {len(fs):4d} frames  {v.width}x{v.height}  "
            + " ".join(f"{k}={n}" for k, n in sorted(counts.items()))
            + f"   ({time.perf_counter() - t0:.0f}s)")

    # --- multi-camera intrinsics seed ---------------------------------------
    shape_of = {f["name"]: (f["width"], f["height"]) for f in all_frames}
    names = sorted(shape_of)
    cams = cameras_for(args.candidate)
    seed_dir = run_dir / "intrinsics_seed"
    summary = write_intrinsics_seed(seed_dir, names, shape_of, cams)
    summary["model_files_sha256"] = {
        name: sha256(seed_dir / name)
        for name in ("cameras.txt", "images.txt", "points3D.txt")
    }
    write_json(run_dir / "intrinsics_seed.json", summary)
    log(f"\nintrinsics seed ({args.candidate}), {summary['n_cameras']} PINHOLE cameras:")
    for c in summary["cameras"]:
        log(f"  cam{c['camera_id']}  {c['width']}x{c['height']}  "
            f"fx={c['fx']:.2f} fy={c['fy']:.2f} cx={c['cx']:.2f} cy={c['cy']:.2f}  "
            f"HFOV={c['hfov_deg']:.2f}  n={c['n_images']}")

    ok_roles, roles, bad_role_records = role_partition(all_frames)
    manifest_path = run_dir / "frame_manifest.json"
    write_json(manifest_path, {
        "run_name": args.run_name,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "undistortion": "NONE",
        "resize": "NONE -- native resolution per group",
        "rates_fps": RATE,
        "candidate_seeded": args.candidate,
        "n_frames": len(all_frames),
        "shapes": [f"{w}x{h}" for w, h in sorted(set(shape_of.values()))],
        "roles": roles,
        "source_lineage": [lineage_by_seq[v.seq] for v in BUILD],
        "groups": {v.seq: sum(1 for f in all_frames if f["seq"] == v.seq) for v in BUILD},
        "frames": all_frames,
    })

    g = stage_gate(run_dir)

    expected_lineage = {
        (lineage_by_seq[v.seq]["source_rel"], lineage_by_seq[v.seq]["source_sha256"])
        for v in BUILD
    }
    actual_lineage = {
        (f.get("source_rel"), f.get("source_sha256")) for f in all_frames
    }
    forbidden = [
        f["name"] for f in all_frames
        if any(token in str(f.get("source_rel", "")) for token in ("P0710071", "P1230123", "P1260126"))
    ]
    manifest_names = {f["name"] for f in all_frames}
    disk_names = {
        path.relative_to(images_dir).as_posix()
        for path in images_dir.rglob("*.jpg") if path.is_file()
    }
    g.check(
        "G0.2",
        actual_lineage == expected_lineage and not forbidden and disk_names == manifest_names,
        "exact seven-source locked lineage and exact manifest/disk JPEG set; no held-out source"
        if actual_lineage == expected_lineage and not forbidden and disk_names == manifest_names
        else "source lineage or persisted JPEG set is not exact",
        expected_lineage=sorted(expected_lineage), actual_lineage=sorted(actual_lineage),
        forbidden_lineage=forbidden,
        missing_on_disk=sorted(manifest_names - disk_names),
        unmanifested_on_disk=sorted(disk_names - manifest_names),
    )
    for v in BUILD:
        seq_frames = [f for f in all_frames if f["seq"] == v.seq]
        n_hover = sum(f["motion_class"] == "hover" for f in seq_frames)
        ratio = n_hover / len(seq_frames) if seq_frames else float("inf")
        g.check(
            f"G1.3/{v.seq}",
            len(seq_frames) == EXPECTED_FRAME_COUNTS[v.seq] and ratio <= MAX_HOVER_RATIO,
            f"{len(seq_frames)} written frames; hover {n_hover}/{len(seq_frames)} ({ratio:.2%})",
            n_frames=len(seq_frames), expected_frames=EXPECTED_FRAME_COUNTS[v.seq],
            n_hover=n_hover, hover_ratio=ratio, max_hover_ratio=MAX_HOVER_RATIO,
        )
    prove_no_undistortion(g, images_dir, all_frames)

    actual_shapes, unreadable, hash_mismatches = inspect_written_frames(
        images_dir, all_frames
    )

    # G2.2 -- every written frame is 16:9 and at its group's native size
    bad = [
        f["name"] for f in all_frames
        if actual_shapes.get(f["name"]) != (f["width"], f["height"])
        or abs(f["width"] / f["height"] - 16 / 9) > 1e-9
    ]
    g.check("G2.2", not bad and not unreadable and not hash_mismatches,
            f"all {len(all_frames)} extracted frames are exactly 16:9"
            if not bad and not unreadable and not hash_mismatches
            else "persisted frame dimensions, readability, or hashes differ",
            n_frames=len(all_frames), bad_dimensions=bad,
            unreadable=unreadable, hash_mismatches=hash_mismatches)

    shapes = sorted(set(actual_shapes.values()))
    g.check("G2.3a", len(shapes) == 3,
            f"{len(shapes)} distinct shapes on disk -> {len(shapes)} cameras: {shapes}",
            shapes=[f"{w}x{h}" for w, h in shapes])

    # G2.4 -- frame budget. gluemap costs ~O(N x num_neighbors); the river baseline
    # was 454 frames -> 35 min, and BA runs on CPU ceres here.
    n = len(all_frames)
    est_h = 35 / 60 * (n / 454) * 1.76   # 1.76x for the force_square path
    g.check("G2.4", 700 <= n <= 2200,
            f"{n} frames selected (~{est_h:.1f} h estimated gluemap, force_square path)",
            n_frames=n, est_hours=round(est_h, 2))

    g.check(
        "G2.5", ok_roles,
        f"current roles partition all {n} frames; no S5 behavior is asserted",
        roles=roles, bad_role_records=bad_role_records,
    )

    # G2.6 -- the seed's cameras must cover exactly the shapes on disk. gluemap
    # buckets by shape and fills each bucket from the first image it matches BY
    # NAME, so a missing shape silently gives those frames another group's focal.
    seed_cameras, seed_names = read_seed_evidence(seed_dir)
    seeded = set(seed_cameras.values())
    g.check("G2.6", seeded == set(shapes) and seed_names == manifest_names,
            f"seed covers exactly the on-disk shapes ({len(seeded)} cameras)"
            if seeded == set(shapes) and seed_names == manifest_names
            else "persisted seed camera shapes or image-name coverage differs",
            seeded=[f"{w}x{h}" for w, h in sorted(seeded)],
            missing_seed_names=sorted(manifest_names - seed_names),
            extra_seed_names=sorted(seed_names - manifest_names),
            seed_model_sha256=summary["model_files_sha256"],)
    g.write(run_dir)
    log(f"S2 PASS -- {n} frames, {len(shapes)} cameras, no undistortion")


if __name__ == "__main__":
    main()
