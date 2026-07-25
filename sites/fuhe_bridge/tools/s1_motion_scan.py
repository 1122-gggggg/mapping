#!/usr/bin/env python3
"""S1 -- classify motion BEFORE extracting frames, and resolve flight direction.

Existing motion code (build_localizable_map.py:5608) runs AFTER ffmpeg, over
already-extracted jpgs. That is backwards: you cannot let motion drive the
sampling rate if you have already sampled. This decodes the video directly.

CLASSES (per probe interval)
  hover          camera centre static     -> usable, but decimate hard
  pure_rotation  slew with NO baseline    -> BRIDGE ONLY, never triangulated
  low_parallax   flying, distant structure-> triangulable, weak (diagnostic)
  fast_motion    blur / a cut             -> dropped
  parallax       the good stuff

WHAT THIS STAGE LEARNED (and it contradicted the plan)
  A slew is not the same thing as a degenerate rotation. Every sustained
  >5 deg/s manoeuvres in the calibration corpus can carry substantial parallax
  and a look-left -- carries 0.8-4.3 deg of parallax. They are GIMBAL MOVES DURING
  FORWARD FLIGHT: the drone keeps flying, the camera centre keeps moving, the
  baseline survives, and the structure is real. Excluding them from triangulation
  would have thrown away good geometry for nothing.

  Their actual hazard is different, and S1 is the wrong place to fix it: they look
  OFF the corridor (at the ground, at the left bank), so they overlap the main
  route poorly. That is a connectivity problem for S4, not a degeneracy problem.

  So `pure_rotation` comes out near-empty on this footage (0.2-1.0%). That is the
  honest answer, not a broken detector -- see the G1.5 positive control, which
  tests that the slews ARE found, and deliberately does NOT assume they are
  degenerate.

WHY THE OBVIOUS TESTS DON'T WORK HERE
  h_inliers/f_inliers >= 0.8, the classic rotation test and the one already in
  this codebase, labelled 46% of an ordinary reverse flight "pure rotation": for
  aerial video the scene is far and near-planar, so a homography explains the flow
  almost always, translating or not.

  Residual parallax alone is no better: over 1 s, half of all frames sit below
  0.35 deg simply because the scene is 50-100 m away. But SfM never triangulates
  off a 1-second baseline -- gluemap pairs each frame with up to 100 retrieved
  neighbours spanning many seconds -- so a low one-second parallax says nothing
  about whether a frame is triangulable.

  What actually kills structure is a STATIONARY CAMERA CENTRE. Hence the test used
  here: the camera must be slewing AND no baseline may appear while it does.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ts_common import (  # noqa: E402
    BUILD,
    RUNS,
    RUN_ID,
    TEST,
    WORKING_WIDTH,
    Gate,
    Video,
    log,
    read_json,
    required_check_ids,
    stage_material_artifacts,
    write_json,
)
from ts_intrinsics import FUHE_CX, FUHE_CY, FUHE_FX, FUHE_FY, Camera  # noqa: E402

TS_COMMON = Path(__file__).resolve().with_name("ts_common.py")
TS_INTRINSICS = Path(__file__).resolve().with_name("ts_intrinsics.py")

# --- probe / classifier constants -------------------------------------------
PROBE_HZ = 4.0          # motion metrics between consecutive probes (0.25 s apart)
VPR_HZ = 0.5            # descriptor probes for direction detection
WORK_W = 960            # motion analysis is done at this width

GEOM_STRIDE = 4         # geometry is measured across 4 probes = 1.0 s of motion.
                        # At 0.25 s the inter-frame baseline is tiny next to the
                        # scene depth, so EVERY pair looks degenerate. 1 s is also
                        # the baseline the extractor will actually hand to SfM.

# --- thresholds, all set FROM the measured distributions, not from taste -------
#
# Two earlier classifiers were wrong, and the way they were wrong is worth keeping:
#
#   1. h_inliers/f_inliers >= 0.8 (the classic test, and the one already in this
#      codebase) called 46% of an ordinary reverse flight "pure rotation". For
#      aerial video the scene is far and near-planar, so a homography explains the
#      flow almost always -- translating or not. It is not a rotation test here.
#
#   2. Residual parallax alone was no better: measured over 1 s, HALF of all
#      frames sit below 0.35 deg simply because the scene is 50-100 m away. But
#      SfM does not triangulate off 1-second baselines -- gluemap pairs each frame
#      with up to 100 retrieved neighbours spanning many seconds. Low parallax
#      across one second says nothing about whether a frame is triangulable.
#
# What actually makes a frame useless for structure is a STATIONARY CAMERA CENTRE.
# So: measure the angular rate (Kabsch rotation between bearing vectors) and the
# parallax left after that rotation is removed. Degenerate means BOTH -- the camera
# slews AND no baseline appears.
#
# MEASURED on the original calibration corpus:
#     rot_rate deg/s : p10 0.19  med 0.64  p90 4.08  p99 11.09
#     parallax deg   : p10 0.07  med 0.25  p90 0.96
# and every sustained rot > 5 deg/s run came with parallax 0.8-4.3 deg.
#
# So the manoeuvres in this footage are GIMBAL MOVES DURING FORWARD FLIGHT. The
# drone keeps flying; the camera centre keeps moving; the baseline is intact. They
# are NOT degenerate rotations, and excluding them from triangulation would throw
# away good structure for nothing. Their real hazard is different -- they look off
# the corridor (at the ground, at the left bank), so they overlap the main route
# poorly. That is a CONNECTIVITY problem, and it is S4's to solve, not S1's.
#
# The pure_rotation class is kept because a zero-baseline slew is fatal when it
# does occur. It is simply expected to be near-empty here, and that is the honest
# answer rather than a forced one.
MIN_TRACKS = 20            # below this there is nothing to measure
MIN_FLOW_PX = 2.0          # median flow over 1 s (at WORK_W) -> camera is static
FAST_FLOW_PX = 140.0       # over 1 s -> motion blur or a cut
FLOW_JUMP_X = 12.0         # ... or a spike this many times the running median

ROT_RATE_DEG_S = 5.0       # a deliberate slew, ~8x the 0.64 deg/s cruise median
ROT_MAX_PARALLAX_DEG = 0.35  # ... and if NO baseline appears during it -> degenerate
WEAK_PARALLAX_DEG = 0.15   # flying, but seeing only distant structure (diagnostic)
MIN_INLIERS = 15           # a verdict on fewer correspondences is noise

MERGE_GAP = 5              # probes; islands smaller than this get absorbed

TRIANGULATION = "triangulation"
BRIDGE_ONLY = "bridge_only"
DROP = "drop"

ROLE_OF = {
    "parallax": TRIANGULATION,
    "low_parallax": TRIANGULATION,
    "hover": TRIANGULATION,       # usable, just not densely
    "pure_rotation": BRIDGE_ONLY,  # registered + bridges, NEVER triangulated
    "unproven": BRIDGE_ONLY,       # insufficient finite geometry for a verdict
    "fast_motion": DROP,
}


def is_pure_rotation(m: dict) -> bool:
    """Return the sole predicate permitted to assign ``pure_rotation``."""
    rot = m.get("rot_deg_s")
    par = m.get("parallax_deg")
    try:
        return bool(
            np.isfinite(rot)
            and np.isfinite(par)
            and rot >= ROT_RATE_DEG_S
            and par < ROT_MAX_PARALLAX_DEG
        )
    except (TypeError, ValueError):
        return False


def probe_times(duration: float, hz: float) -> np.ndarray:
    return np.arange(0.0, max(duration - 1e-6, 0.0), 1.0 / hz)


def decode_at(video: Video, times: np.ndarray, *, gray: bool, width: int):
    """Sequentially decode, yielding (time, frame) at the requested times.

    Sequential reads beat seeking by a wide margin on long h264 files.
    """
    cap = cv2.VideoCapture(str(video.path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video.path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or video.fps
    want = list(times)
    wi, idx = 0, 0
    try:
        while wi < len(want):
            ok, frame = cap.read()
            if not ok:
                break
            t = idx / fps
            if t + 1e-9 >= want[wi]:
                img = frame
                h, w = img.shape[:2]
                if w != width:
                    img = cv2.resize(img, (width, int(round(h * width / w))),
                                     interpolation=cv2.INTER_AREA)
                if gray:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                yield want[wi], img
                wi += 1
                while wi < len(want) and want[wi] <= t:
                    wi += 1
            idx += 1
    finally:
        cap.release()


def rotation_and_parallax(p0: np.ndarray, p1: np.ndarray,
                          Kinv: np.ndarray) -> tuple[float, float]:
    """(rotation angle, residual parallax), both in degrees.

    Lift both point sets to unit bearing vectors and solve for the rotation that
    best maps one onto the other (Kabsch / orthogonal Procrustes on the sphere).

    Rotation displaces every bearing identically regardless of depth, so it
    carries no structure. Applying it and measuring what is LEFT isolates exactly
    the depth-dependent, translation-induced component -- the parallax.

      camera slewing in place -> big rotation, residual is sensor noise
      camera flying           -> residual grows with baseline/depth

    Caveat worth knowing: for a narrow FOV on a distant scene, a SIDEWAYS
    translation induces almost the same flow as a small rotation, so Kabsch will
    absorb some of it. Forward flight (a radial expansion field) is not
    rotation-like and is not absorbed, and drones mostly fly forwards -- so this
    is tolerable, but it is why a rotation verdict also requires a HIGH angular
    rate rather than resting on the residual alone.
    """
    b0 = np.hstack([p0, np.ones((len(p0), 1))]) @ Kinv.T
    b1 = np.hstack([p1, np.ones((len(p1), 1))]) @ Kinv.T
    b0 /= np.linalg.norm(b0, axis=1, keepdims=True)
    b1 /= np.linalg.norm(b1, axis=1, keepdims=True)

    U, _S, Vt = np.linalg.svd(b0.T @ b1)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T   # maps b0 -> b1

    rot = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))
    cos = np.clip(np.einsum("ij,ij->i", b0 @ R.T, b1), -1.0, 1.0)
    par = float(np.degrees(np.median(np.arccos(cos))))
    return rot, par


def two_view_metrics(prev: np.ndarray, cur: np.ndarray, K: np.ndarray,
                     Kinv: np.ndarray) -> dict:
    base = {"tracks": 0, "median_flow_px": 0.0, "inliers": 0,
            "rot_deg_s": float("nan"), "parallax_deg": float("nan")}
    pts0 = cv2.goodFeaturesToTrack(prev, maxCorners=1000, qualityLevel=0.01,
                                   minDistance=8)
    if pts0 is None:
        return base
    pts1, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, pts0, None)
    if pts1 is None or st is None:
        return base
    ok = st.reshape(-1).astype(bool)
    p0, p1 = pts0.reshape(-1, 2)[ok], pts1.reshape(-1, 2)[ok]
    if len(p0) < 8:
        return {**base, "tracks": int(len(p0))}

    flow = np.linalg.norm(p1 - p0, axis=1)
    m = {**base, "tracks": int(len(p0)), "median_flow_px": float(np.median(flow))}

    # RANSAC on E first, so moving cars, water glint and LK drift do not drag the
    # rotation fit. Its inlier set is what Kabsch runs on.
    try:
        _E, em = cv2.findEssentialMat(p0, p1, K, cv2.RANSAC, 0.999, 1.5)
        inl = em.reshape(-1).astype(bool) if em is not None else np.ones(len(p0), bool)
    except cv2.error:
        inl = np.ones(len(p0), bool)
    m["inliers"] = int(inl.sum())
    if m["inliers"] < 8:
        return m

    rot, par = rotation_and_parallax(p0[inl], p1[inl], Kinv)
    m["rot_deg_s"] = rot * PROBE_HZ / GEOM_STRIDE   # the window is GEOM_STRIDE probes
    m["parallax_deg"] = par
    return m


def classify(m: dict, flow_median: float) -> tuple[str, str]:
    tracks, flow = m["tracks"], m["median_flow_px"]
    rot, par = m["rot_deg_s"], m["parallax_deg"]

    if tracks < MIN_TRACKS or flow < MIN_FLOW_PX:
        return "hover", f"tracks={tracks} flow={flow:.1f}px -- camera centre is static"

    if flow > FAST_FLOW_PX or (flow_median > 0 and flow > FLOW_JUMP_X * flow_median):
        return "fast_motion", f"flow={flow:.0f}px vs median {flow_median:.0f}px"

    if m["inliers"] < MIN_INLIERS or not np.isfinite(rot) or not np.isfinite(par):
        # Not enough evidence to certify either rotation or baseline. Keep the
        # frame available as a bridge, but reserve pure_rotation for the exact
        # finite two-threshold conjunction below.
        return "unproven", (
            f"inliers={m['inliers']} rot={rot} parallax={par} -- geometry unproven"
        )

    # Degenerate ONLY when the camera slews AND no baseline appears while it does.
    # A slew with parallax is a banking turn or a gimbal move during forward
    # flight: the centre keeps moving, the structure is real, keep it.
    if is_pure_rotation(m):
        return "pure_rotation", (
            f"slewing at {rot:.1f} deg/s with only {par:.2f} deg parallax "
            "-- camera centre is not moving"
        )

    if par < WEAK_PARALLAX_DEG:
        return "low_parallax", f"parallax {par:.2f} deg -- distant structure only"
    return "parallax", f"parallax {par:.2f} deg, rot {rot:.1f} deg/s"


def smooth(classes: list[str]) -> list[str]:
    """Absorb islands shorter than MERGE_GAP into their surroundings.

    A single frame flipping to pure_rotation mid-flight is RANSAC noise, not a
    manoeuvre. A real rotation lasts seconds.
    """
    if not classes:
        return classes
    out = list(classes)
    i = 0
    while i < len(out):
        j = i
        while j < len(out) and out[j] == out[i]:
            j += 1
        run = j - i
        if (
            run < MERGE_GAP
            and i > 0
            and j < len(out)
            and out[i - 1] == out[j]
            and out[i - 1] != "pure_rotation"
        ):
            for k in range(i, j):
                out[k] = out[i - 1]
        i = j
    return out


def segments(times: np.ndarray, classes: list[str]) -> list[dict]:
    segs, i = [], 0
    while i < len(classes):
        j = i
        while j < len(classes) and classes[j] == classes[i]:
            j += 1
        segs.append({
            "motion_class": classes[i],
            "role": ROLE_OF[classes[i]],
            "t_start": float(times[i]),
            "t_end": float(times[min(j, len(times) - 1)]),
            "n_probes": j - i,
        })
        i = j
    return segs


def scan_video(v: Video) -> dict:
    from ts_common import ffprobe
    meta = ffprobe(v.path)
    dur = meta["duration"]
    times = probe_times(dur, PROBE_HZ)
    log(f"{v.seq}: {dur:.0f}s -> {len(times)} probes @ {PROBE_HZ} Hz")

    # Scale the exact fixed 1920x1080 P0 camera to the lightweight motion canvas.
    work_h = int(round(WORK_W * v.height / v.width))
    scale = WORK_W / WORKING_WIDTH
    cam = Camera(
        WORK_W,
        work_h,
        FUHE_FX * scale,
        FUHE_FY * scale,
        FUHE_CX * scale,
        FUHE_CY * scale,
    )
    K = np.array([[cam.fx, 0, cam.cx], [0, cam.fy, cam.cy], [0, 0, 1]], np.float64)
    Kinv = np.linalg.inv(K)

    frames = [(t, img) for t, img in decode_at(v, times, gray=True, width=WORK_W)]
    if len(frames) <= GEOM_STRIDE:
        raise SystemExit(f"{v.seq}: too short to probe")

    # Geometry across GEOM_STRIDE probes (1.0 s), scored at the LATER probe, so
    # every probe still gets its own class and the temporal resolution is kept.
    mets, ts = [], []
    for i in range(GEOM_STRIDE, len(frames)):
        t, img = frames[i]
        mets.append(two_view_metrics(frames[i - GEOM_STRIDE][1], img, K, Kinv))
        ts.append(t)

    flows = np.array([m["median_flow_px"] for m in mets])
    fmed = float(np.median(flows[flows > 0])) if (flows > 0).any() else 0.0

    raw = [classify(m, fmed) for m in mets]
    raw_classes = [c for c, _ in raw]
    classes = smooth(raw_classes)
    reasons = [r for _, r in raw]
    ts = np.asarray(ts)

    counts = {c: int(sum(1 for x in classes if x == c)) for c in set(classes)}
    n = len(classes)
    return {
        "seq": v.seq,
        "rel": v.rel,
        "duration": dur,
        "probe_hz": PROBE_HZ,
        "n_probes": n,
        "flow_median_px": fmed,
        "class_counts": counts,
        "class_ratios": {c: k / n for c, k in counts.items()},
        "segments": segments(ts, classes),
        "records": [
            {
                "t": float(t),
                "raw_motion_class": raw_c,
                "motion_class": c,
                "reason": r,
                **m,
            }
            for t, raw_c, c, r, m in zip(ts, raw_classes, classes, reasons, mets)
        ],
    }


def emit_pure_rotation_replay_gate(g: Gate, manifest: dict) -> bool:
    """Emit G1.6 after replaying the contract over every persisted-shape record."""
    records = [
        (seq, record)
        for seq, scan in manifest.get("sequences", {}).items()
        for record in scan.get("records", [])
    ]
    violations = [
        {
            "seq": seq,
            "t": record.get("t"),
            "rot_deg_s": record.get("rot_deg_s"),
            "parallax_deg": record.get("parallax_deg"),
        }
        for seq, record in records
        if record.get("motion_class") == "pure_rotation"
        and not is_pure_rotation(record)
    ]
    n_pure = sum(
        record.get("motion_class") == "pure_rotation" for _, record in records
    )
    return g.check(
        "G1.6",
        not violations,
        f"replayed {len(records)} final records; "
        + (
            "every pure_rotation record obeys both finite thresholds"
            if not violations
            else f"{len(violations)} pure_rotation record(s) violate the conjunction"
        ),
        n_records=len(records),
        n_pure=n_pure,
        n_violations=len(violations),
        violations=violations,
    )


# ---------------------------------------------------------------- direction
def megaloc_descriptors(v: Video, device: str = "cuda") -> tuple[np.ndarray, np.ndarray]:
    import torch
    import torchvision.transforms as T
    from PIL import Image

    global _MEGALOC
    try:
        _MEGALOC
    except NameError:
        _MEGALOC = torch.hub.load("gmberton/MegaLoc", "get_trained_model",
                                  trust_repo=True).eval().to(device)
    # Same preprocessing as megaloc_lib (square 322 BICUBIC + ImageNet norm).
    tf = T.Compose([
        T.Resize((322, 322), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    from ts_common import ffprobe
    times = probe_times(ffprobe(v.path)["duration"], VPR_HZ)
    buf, keep = [], []
    for t, img in decode_at(v, times, gray=False, width=640):
        buf.append(tf(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))))
        keep.append(t)
    with torch.no_grad():
        d = _MEGALOC(torch.stack(buf).to(device)).float().cpu().numpy()
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
    return np.asarray(keep), d


def _align(d: np.ndarray, ref_d: np.ndarray) -> tuple[float, float]:
    """(spearman rho, median best-match cosine) of this video against a reference."""
    from scipy.stats import spearmanr
    sim = d @ ref_d.T
    best = sim.argmax(axis=1)
    rho, _p = spearmanr(np.arange(len(best)), best)
    return (float(rho) if np.isfinite(rho) else 0.0,
            float(np.median(sim.max(axis=1))))


def resolve_direction(v: Video, refs: dict[str, np.ndarray]) -> dict:
    """Which reference does this flight align with -- the forward one or the reverse one?

    The first attempt compared only against a FORWARD reference and read the sign
    of Spearman's rho: flying the route forwards makes the best-matching reference
    index climb with time, flying it backwards makes it fall.

    Measured, the sign was right and the strength was not: forward flights scored
    rho +0.85, reverse flights only -0.15. That five-fold asymmetry is not a bug in
    the test -- it IS the finding. MegaLoc cannot retrieve across a direction
    reversal, because a reverse traversal simply does not look like the forward
    one, so the nearest-neighbour index degenerates into noise and the ordering
    signal washes out. (This is exactly why S3 has to force VPR-blind bridge pairs:
    global descriptors will never propose a forward<->reverse pair on their own.)

    So compare against BOTH a forward and a reverse anchor and ask which one the
    flight actually aligns with. A reverse flight matched to the reverse anchor
    traverses the route in the same order and looks the same doing it, so it scores
    strongly positive there -- while scoring weakly negative against the forward
    anchor. The verdict comes from the winning anchor, not from a sign.
    """
    ts, d = megaloc_descriptors(v)
    scores = {}
    for name, ref_d in refs.items():
        if ref_d is None:
            continue
        rho, sim = _align(d, ref_d)
        scores[name] = {"rho": rho, "median_sim": sim}

    # The anchor a flight genuinely shares a direction with is the one it both
    # orders WITH (rho > 0) and looks LIKE (high cosine).
    best = max(scores.items(), key=lambda kv: kv[1]["rho"])
    direction = best[0] if best[1]["rho"] > 0.4 else "ambiguous"
    return {
        "seq": v.seq, "direction": direction, "declared": v.direction,
        "aligned_with": best[0], "rho": best[1]["rho"],
        "scores": scores, "n_probes": len(ts),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-name", default=RUN_ID)
    ap.add_argument("--skip-direction", action="store_true")
    args = ap.parse_args()

    run_dir = RUNS / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log(f"S1 motion scan -> {run_dir}")

    corpus_manifest_path = run_dir / "corpus_manifest.json"
    predecessor_gate_path = run_dir / "gates" / "S0_corpus.json"
    g = Gate(
        "S1_motion",
        required_check_ids("S1_motion"),
        script_path=Path(__file__),
        input_artifacts=stage_material_artifacts("S1_motion", run_dir),
        source_files=[TS_COMMON],
    )
    g.record_predecessor_gate(
        "S0_corpus",
        predecessor_gate_path,
        expected_stage="S0_corpus",
    )
    g.record_source_hash(TS_INTRINSICS)

    unavailable = []
    corpus_manifest = None
    if not corpus_manifest_path.is_file():
        unavailable.append(f"missing corpus manifest: {corpus_manifest_path}")
    else:
        try:
            corpus_manifest = read_json(corpus_manifest_path)
        except (OSError, TypeError, ValueError) as exc:
            unavailable.append(f"unreadable corpus manifest: {exc}")
    if not predecessor_gate_path.is_file():
        unavailable.append(f"missing predecessor gate: {predecessor_gate_path}")
    else:
        try:
            predecessor = read_json(predecessor_gate_path)
        except (OSError, TypeError, ValueError) as exc:
            unavailable.append(f"unreadable predecessor gate: {exc}")
            predecessor = {}
        if (
            predecessor.get("stage") != "S0_corpus"
            or predecessor.get("status") != "PASS"
            or predecessor.get("ok") is not True
        ):
            unavailable.append(
                "S0 predecessor is not a PASS S0_corpus gate: "
                f"stage={predecessor.get('stage')!r} status={predecessor.get('status')!r}"
            )
    if unavailable:
        reason = "; ".join(unavailable)
        for gid in sorted(g.required_ids):
            g.not_run(gid, reason)
        g.write(run_dir)

    assert isinstance(corpus_manifest, dict)
    expected_sources = [(v.seq, v.rel) for v in BUILD]
    expected_test_sources = [(v.seq, v.rel) for v in TEST]
    observed_sources = [
        (entry.get("seq"), entry.get("rel"))
        for entry in corpus_manifest.get("build", [])
    ]
    observed_test_sources = [
        (entry.get("seq"), entry.get("rel"))
        for entry in corpus_manifest.get("test", [])
    ]
    locked_hashes = [entry.get("sha256") for entry in corpus_manifest.get("build", [])]
    test_hashes = [entry.get("sha256") for entry in corpus_manifest.get("test", [])]
    hashes_complete = len(locked_hashes) == len(expected_sources) and all(
        isinstance(digest, str) and len(digest) == 64 for digest in locked_hashes
    )
    source_overlap = sorted(
        {rel for _, rel in observed_sources}
        & {rel for _, rel in observed_test_sources}
    )
    hash_overlap = sorted(set(locked_hashes) & set(test_hashes))
    lineage_ok = (
        observed_sources == expected_sources
        and observed_test_sources == expected_test_sources
        and hashes_complete
        and not source_overlap
        and not hash_overlap
    )
    g.check(
        "G0.2",
        lineage_ok,
        f"corpus manifest locks exactly the {len(BUILD)} declared BUILD sources with hashes "
        "and no held-out overlap"
        if lineage_ok
        else "corpus manifest BUILD lineage differs from the declared source allowlist",
        expected_sources=expected_sources,
        observed_sources=observed_sources,
        expected_test_sources=expected_test_sources,
        observed_test_sources=observed_test_sources,
        hashes_complete=hashes_complete,
        source_overlap=source_overlap,
        hash_overlap=hash_overlap,
    )

    scans = {}
    for v in BUILD:
        t0 = time.perf_counter()
        scans[v.seq] = scan_video(v)
        r = scans[v.seq]["class_ratios"]
        log(f"  {v.seq}: " + "  ".join(f"{c}={r.get(c, 0) * 100:.1f}%" for c in
                                       ("parallax", "low_parallax", "pure_rotation",
                                        "hover", "fast_motion"))
            + f"   ({time.perf_counter() - t0:.0f}s)")

    # P0 directions are corpus declarations. A site-specific anchor heuristic
    # would add a GPU/network dependency and would not be config-derived.
    directions = {
        video.seq: {
            "seq": video.seq,
            "direction": video.direction,
            "declared": video.direction,
            "method": "p0_corpus_declaration",
        }
        for video in BUILD
    }

    for seq, s in scans.items():
        expected = max(
            len(probe_times(s["duration"], s["probe_hz"])) - GEOM_STRIDE,
            1,
        )
        actual = len(s["records"])
        coverage = actual / expected
        g.check(
            f"G1.1/{seq}",
            coverage >= 0.99,
            f"decoded {actual}/{expected} geometry probes ({coverage * 100:.2f}%; need >=99%)",
            expected=expected,
            actual=actual,
            coverage=coverage,
        )

        r = s["class_ratios"]
        good = r.get("parallax", 0) + r.get("low_parallax", 0)
        g.check(f"G1.2/{seq}", good >= 0.65,
                f"triangulable ratio {good * 100:.1f}% (need >=65%)",
                parallax=r.get("parallax", 0), low_parallax=r.get("low_parallax", 0))

    finite_by_sequence = {
        seq: sum(
            np.isfinite(record.get("rot_deg_s", np.nan))
            and np.isfinite(record.get("parallax_deg", np.nan))
            for record in scan["records"]
        )
        for seq, scan in scans.items()
    }
    g.check(
        "G1.5",
        set(finite_by_sequence) == {video.seq for video in BUILD}
        and all(count > 0 for count in finite_by_sequence.values()),
        "every declared BUILD sequence produced finite two-view geometry",
        finite_records=finite_by_sequence,
    )

    degen = sum(s["class_ratios"].get("pure_rotation", 0.0) for s in scans.values()) / len(scans)
    log(f"  NOTE: mean pure_rotation (zero-baseline) ratio = {degen * 100:.2f}% -- "
        "near-zero is the expected, honest answer for this footage: the slews are "
        "gimbal moves during forward flight, not stationary spins")

    if not args.skip_direction:
        resolved = {seq: record["direction"] for seq, record in directions.items()}
        unresolved = sorted(
            seq for seq, direction in resolved.items() if direction not in {"fwd", "rev"}
        )
        g.check("G1.4a", not unresolved,
                f"all {len(BUILD)} BUILD sequences resolve to fwd/rev" if not unresolved
                else f"unresolved direction: {unresolved}",
                resolved=resolved, unresolved=unresolved)
        conflict = [f"{d['seq']}: declared={d['declared']} resolved={d['direction']}"
                    for d in directions.values()
                    if d["declared"] != "unknown" and d["declared"] != d["direction"]]
        g.check("G1.4b", not conflict,
                "resolved directions agree with the P0 corpus declaration" if not conflict
                else f"CONFLICT: {conflict}", conflicts=conflict)
        n_fwd = sum(1 for x in resolved.values() if x == "fwd")
        n_rev = sum(1 for x in resolved.values() if x == "rev")
        g.check("G1.4c", n_fwd >= 1 and n_rev >= 1,
                f"{n_fwd} forward, {n_rev} reverse -- both directions present",
                resolved=resolved)
    else:
        g.incomplete("G1.4a", "--skip-direction omitted required direction resolution")
        g.incomplete("G1.4b", "--skip-direction omitted filename agreement evidence")
        g.incomplete("G1.4c", "--skip-direction omitted bidirectional coverage evidence")

    motion_manifest = {
        "run_name": args.run_name,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "probe_hz": PROBE_HZ,
        "thresholds": {
            "MIN_TRACKS": MIN_TRACKS, "MIN_FLOW_PX": MIN_FLOW_PX,
            "FAST_FLOW_PX": FAST_FLOW_PX, "FLOW_JUMP_X": FLOW_JUMP_X,
            "ROT_RATE_DEG_S": ROT_RATE_DEG_S,
            "ROT_MAX_PARALLAX_DEG": ROT_MAX_PARALLAX_DEG,
            "WEAK_PARALLAX_DEG": WEAK_PARALLAX_DEG,
            "MIN_INLIERS": MIN_INLIERS, "GEOM_STRIDE": GEOM_STRIDE,
        },
        "role_of": ROLE_OF,
        "epoch_gate": {
            "applicable": False,
            "reason": "all BUILD videos share one capture epoch",
        },
        "directions": directions,
        "sequences": scans,
    }
    emit_pure_rotation_replay_gate(g, motion_manifest)
    write_json(run_dir / "motion_manifest.json", motion_manifest)
    g.write(run_dir)
    log("S1 PASS -- motion_manifest.json written")


if __name__ == "__main__":
    main()
