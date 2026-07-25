#!/usr/bin/env python3
"""Prepare video frames for map_update_tool.py.

The preferred path is manifest replay: if a previous build recorded source_idx
for a sequence, use those exact frame indices. For new sites without a manifest,
fall back to fps sampling, then gate candidates by motion or parallax. The
parallax mode is preferred for map/submap construction because optical-flow
magnitude alone cannot distinguish translation baseline from pure rotation.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import stat
from pathlib import Path

import cv2
import numpy as np


def find_system_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if p.name == "sfm_system":
            return p
    return Path("/media/cihcilab/新增磁碟區/sfm_system")


SYSTEM_ROOT = find_system_root(Path(__file__).resolve())
DEFAULT_UPDATE_INPUT = SYSTEM_ROOT / "更新地圖" / "inputs" / "補拍影片" / "build"

DEFAULT_VIDEO_MAP = {
    "P1210121": str(DEFAULT_UPDATE_INPUT / "a-b.MP4"),
    "P1220122": str(DEFAULT_UPDATE_INPUT / "P1220122.MP4"),
    "P1240124": str(DEFAULT_UPDATE_INPUT / "P1240124.MP4"),
    "P1250125": str(DEFAULT_UPDATE_INPUT / "P1250125.MP4"),
}

DIRECTORY_OPEN_FLAGS = (
    os.O_RDONLY
    | os.O_DIRECTORY
    | os.O_CLOEXEC
    | os.O_NOFOLLOW
)
FILE_OPEN_FLAGS = (
    os.O_WRONLY
    | os.O_CREAT
    | os.O_CLOEXEC
    | os.O_NONBLOCK
    | os.O_NOFOLLOW
)


def validate_sequence_name(seq: str) -> str:
    if (
        not seq
        or seq in {".", ".."}
        or "\x00" in seq
        or "/" in seq
        or "\\" in seq
        or Path(seq).is_absolute()
    ):
        raise ValueError(
            f"invalid sequence {seq!r}: expected one non-special path component"
        )
    return seq


def open_or_create_directory_at(parent_fd: int, name: str) -> int:
    validate_sequence_name(name)
    try:
        os.mkdir(name, mode=0o755, dir_fd=parent_fd)
    except FileExistsError:
        pass
    except OSError as exc:
        raise ValueError(f"cannot create output directory {name!r}: {exc}") from None
    try:
        return os.open(name, DIRECTORY_OPEN_FLAGS, dir_fd=parent_fd)
    except OSError as exc:
        raise ValueError(f"unsafe output directory {name!r}: {exc}") from None


def _clear_directory_fd(directory_fd: int) -> None:
    for name in os.listdir(directory_fd):
        try:
            info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            continue
        if stat.S_ISDIR(info.st_mode):
            try:
                child_fd = os.open(name, DIRECTORY_OPEN_FLAGS, dir_fd=directory_fd)
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise ValueError(f"unsafe directory entry {name!r}: {exc}") from None
            try:
                _clear_directory_fd(child_fd)
            finally:
                os.close(child_fd)
            try:
                os.rmdir(name, dir_fd=directory_fd)
            except FileNotFoundError:
                continue
        else:
            try:
                os.unlink(name, dir_fd=directory_fd)
            except FileNotFoundError:
                continue


def secure_rmtree_at(parent_fd: int, name: str) -> bool:
    validate_sequence_name(name)
    try:
        directory_fd = os.open(name, DIRECTORY_OPEN_FLAGS, dir_fd=parent_fd)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise ValueError(f"unsafe sequence output {name!r}: {exc}") from None
    try:
        _clear_directory_fd(directory_fd)
    finally:
        os.close(directory_fd)
    try:
        os.rmdir(name, dir_fd=parent_fd)
    except OSError as exc:
        raise ValueError(f"cannot remove sequence output {name!r}: {exc}") from None
    return True


def open_sequence_directory_at(parent_fd: int, seq: str, overwrite: bool) -> int:
    seq = validate_sequence_name(seq)
    if overwrite:
        secure_rmtree_at(parent_fd, seq)
    return open_or_create_directory_at(parent_fd, seq)


def directory_fd_path(directory_fd: int) -> Path:
    return Path(f"/proc/self/fd/{directory_fd}")


def _validate_report_destination_at(directory_fd: int, name: str) -> None:
    try:
        info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError(f"unsafe report path {name!r}: expected one regular file link")


def write_json_report_at(directory_fd: int, name: str, report: list[dict]) -> None:
    payload = json.dumps(report, indent=2)
    _validate_report_destination_at(directory_fd, name)
    temp_name = None
    temp_fd: int | None = None
    for _ in range(10):
        candidate = f".{name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
        try:
            temp_fd = os.open(
                candidate,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | os.O_CLOEXEC
                | os.O_NOFOLLOW,
                0o644,
                dir_fd=directory_fd,
            )
        except FileExistsError:
            continue
        except OSError as exc:
            raise ValueError(f"cannot create report temporary file: {exc}") from None
        temp_name = candidate
        break
    if temp_fd is None or temp_name is None:
        raise ValueError("cannot allocate a unique report temporary file")
    try:
        stream = os.fdopen(temp_fd, "w", encoding="utf-8")
        temp_fd = None
        with stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _validate_report_destination_at(directory_fd, name)
        os.replace(
            temp_name,
            name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        temp_name = None
        os.fsync(directory_fd)
    except OSError as exc:
        raise ValueError(f"cannot publish report {name!r}: {exc}") from None
    finally:
        if temp_fd is not None:
            os.close(temp_fd)
        if temp_name is not None:
            try:
                os.unlink(temp_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass


def write_json_report(path: Path, report: list[dict]) -> None:
    parent = path.parent.resolve(strict=True)
    try:
        directory_fd = os.open(parent, DIRECTORY_OPEN_FLAGS)
    except OSError as exc:
        raise ValueError(f"unsafe report directory {parent}: {exc}") from None
    try:
        write_json_report_at(directory_fd, path.name, report)
    finally:
        os.close(directory_fd)


def parse_video(items: list[str] | None) -> dict[str, Path]:
    if not items:
        return {validate_sequence_name(k): Path(v) for k, v in DEFAULT_VIDEO_MAP.items()}
    out = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--video must be SEQ=/path/video.mp4, got {item}")
        seq, path = item.split("=", 1)
        try:
            seq = validate_sequence_name(seq)
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
        out[seq] = Path(path)
    return out


def flow_score(prev_gray: np.ndarray, gray: np.ndarray) -> tuple[float, int]:
    pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=800, qualityLevel=0.01, minDistance=8)
    if pts is None:
        return 0.0, 0
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None)
    if nxt is None or st is None:
        return 0.0, 0
    ok = st.reshape(-1).astype(bool)
    if int(ok.sum()) < 20:
        return 0.0, int(ok.sum())
    d = np.linalg.norm(nxt[ok] - pts[ok], axis=2).reshape(-1)
    return float(np.median(d)), int(ok.sum())


def resize_gray(frame: np.ndarray, width: int = 480) -> np.ndarray:
    h, w = frame.shape[:2]
    if w > width:
        frame = cv2.resize(frame, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def write_jpg(path: Path, frame: np.ndarray, quality: int) -> None:
    try:
        ok, encoded = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        )
    except cv2.error as exc:
        raise SystemExit(f"failed to encode {path}: {exc}") from None
    if not ok:
        raise SystemExit(f"failed to encode {path}")
    try:
        fd: int | None = os.open(path, FILE_OPEN_FLAGS, 0o644)
    except OSError as exc:
        raise SystemExit(f"failed to open {path}: {exc}") from None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise SystemExit(f"refusing unsafe image output {path}")
        os.ftruncate(fd, 0)
        stream = os.fdopen(fd, "wb")
        fd = None
        with stream:
            stream.write(encoded.tobytes())
    except OSError as exc:
        raise SystemExit(f"failed to write {path}: {exc}") from None
    finally:
        if fd is not None:
            os.close(fd)


def source_frame_name(frame_idx: int) -> str:
    return f"frame_{frame_idx:06d}.jpg"


def extract_manifest(seq: str, video: Path, indices: list[int], out_dir: Path, quality: int) -> dict:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    want = set(map(int, indices))
    out_pos = {int(idx): i + 1 for i, idx in enumerate(indices)}
    saved = 0
    frame_idx = 0
    max_idx = max(want) if want else -1
    while frame_idx <= max_idx:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx in want:
            write_jpg(out_dir / f"{out_pos[frame_idx]:06d}.jpg", frame, quality)
            saved += 1
        frame_idx += 1
    cap.release()
    if saved != len(indices):
        raise SystemExit(f"{seq}: saved {saved}, expected {len(indices)}")
    return {"seq": seq, "mode": "manifest", "saved": saved, "source_indices": len(indices)}


def geometry_score(prev_gray: np.ndarray, gray: np.ndarray) -> dict:
    orb = cv2.ORB_create(nfeatures=1800, fastThreshold=12)
    k0, d0 = orb.detectAndCompute(prev_gray, None)
    k1, d1 = orb.detectAndCompute(gray, None)
    if d0 is None or d1 is None or len(k0) < 30 or len(k1) < 30:
        return {"matches": 0, "f_inliers": 0, "h_inliers": 0, "h_over_f": 1.0,
                "median_flow_px": 0.0, "parallax_ok": False, "rotation_like": False}
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw = bf.knnMatch(d0, d1, k=2)
    good = [m for m, n in raw if n is not None and m.distance < 0.75 * n.distance]
    if len(good) < 30:
        return {"matches": len(good), "f_inliers": 0, "h_inliers": 0, "h_over_f": 1.0,
                "median_flow_px": 0.0, "parallax_ok": False, "rotation_like": False}
    p0 = np.float32([k0[m.queryIdx].pt for m in good])
    p1 = np.float32([k1[m.trainIdx].pt for m in good])
    disp = np.linalg.norm(p1 - p0, axis=1)
    try:
        F, fm = cv2.findFundamentalMat(p0, p1, cv2.FM_RANSAC, 1.5, 0.999)
        H, hm = cv2.findHomography(p0, p1, cv2.RANSAC, 3.0)
    except cv2.error:
        return {"matches": len(good), "f_inliers": 0, "h_inliers": 0, "h_over_f": 1.0,
                "median_flow_px": float(np.median(disp)),
                "parallax_ok": False, "rotation_like": False}
    fi = int(fm.sum()) if fm is not None else 0
    hi = int(hm.sum()) if hm is not None else 0
    h_over_f = hi / max(fi, 1)
    rotation_like = hi >= 30 and fi >= 30 and h_over_f >= 0.85
    return {"matches": len(good), "f_inliers": fi, "h_inliers": hi,
            "h_over_f": float(h_over_f), "median_flow_px": float(np.median(disp)),
            "parallax_ok": fi >= 45 and not rotation_like,
            "rotation_like": rotation_like}


def extract_fps_flow(seq: str, video: Path, fps: float, out_dir: Path, quality: int,
                     min_flow_px: float, max_flow_px: float, motion_filter: str,
                     keep_rotation_every: int, split_classes: bool = False,
                     connector_dir: Path | None = None) -> dict:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(native_fps / fps)))
    saved = 0
    saved_geometry = 0
    saved_connector = 0
    seed_geometry = 0
    tested = 0
    rejected_low = 0
    rejected_high = 0
    rejected_rotation = 0
    rejected_weak = 0
    rotation_seen = 0
    prev_gray = None
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step:
            frame_idx += 1
            continue
        tested += 1
        gray = resize_gray(frame)
        keep = prev_gray is None
        keep_geometry = prev_gray is None
        keep_connector = False
        score, tracks = 0.0, 0
        if prev_gray is not None:
            if motion_filter == "parallax":
                gs = geometry_score(prev_gray, gray)
                score = gs["median_flow_px"]
                tracks = gs["matches"]
                if score < min_flow_px:
                    rejected_low += 1
                    keep = keep_geometry = False
                elif score > max_flow_px:
                    rejected_high += 1
                    keep = keep_geometry = False
                elif gs["rotation_like"]:
                    rotation_seen += 1
                    keep_connector = bool(keep_rotation_every and rotation_seen % keep_rotation_every == 0)
                    keep = keep_connector
                    keep_geometry = False
                    rejected_rotation += 0 if keep_connector else 1
                elif not gs["parallax_ok"]:
                    rejected_weak += 1
                    keep = keep_geometry = False
                else:
                    keep = keep_geometry = True
            else:
                score, tracks = flow_score(prev_gray, gray)
                if score < min_flow_px:
                    rejected_low += 1
                    keep = keep_geometry = False
                elif score > max_flow_px:
                    rejected_high += 1
                    keep = keep_geometry = False
                else:
                    keep = keep_geometry = True
        if split_classes:
            fname = source_frame_name(frame_idx)
            if keep_geometry:
                saved_geometry += 1
                if prev_gray is None:
                    seed_geometry += 1
                write_jpg(out_dir / fname, frame, quality)
            if keep_connector:
                if connector_dir is None:
                    raise SystemExit("connector_dir is required when split_classes=True")
                saved_connector += 1
                write_jpg(connector_dir / fname, frame, quality)
            saved = saved_geometry + saved_connector
        elif keep:
            saved += 1
            write_jpg(out_dir / f"{saved:06d}.jpg", frame, quality)
        prev_gray = gray
        frame_idx += 1
    cap.release()
    out = {"seq": seq, "mode": f"fps_{motion_filter}", "saved": saved, "tested": tested,
            "fps": fps, "min_flow_px": min_flow_px, "max_flow_px": max_flow_px,
            "rejected_low_motion": rejected_low, "rejected_high_motion": rejected_high,
            "rejected_rotation_like": rejected_rotation, "rejected_weak_geometry": rejected_weak,
            "keep_rotation_every": keep_rotation_every}
    if split_classes:
        out.update({"split_classes": True, "saved_geometry": saved_geometry,
                    "saved_connector": saved_connector, "seed_geometry": seed_geometry})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--manifest")
    ap.add_argument("--video", action="append", help="SEQ=/path/video.mp4")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--motion-filter", choices=["none", "flow", "parallax"], default="none")
    ap.add_argument("--min-flow-px", type=float, default=3.0)
    ap.add_argument("--max-flow-px", type=float, default=80.0)
    ap.add_argument("--keep-rotation-every", type=int, default=0,
                    help="In parallax mode, keep every Nth H-dominant rotation-like candidate; 0 drops them.")
    ap.add_argument("--split-classes", action="store_true",
                    help="Write geometry frames to OUT/geometry/SEQ and connector rotation frames to OUT/connector/SEQ.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    videos = parse_video(args.video)
    out_root.mkdir(parents=True, exist_ok=True)
    resolved_out_root = out_root.resolve(strict=True)
    manifest = json.loads(Path(args.manifest).read_text()) if args.manifest else {}
    report = []
    try:
        anchor_fd = os.open(resolved_out_root, DIRECTORY_OPEN_FLAGS)
    except OSError as exc:
        raise SystemExit(f"cannot open output root {resolved_out_root}: {exc}") from None
    data_fd: int | None = None
    connector_root_fd: int | None = None
    try:
        try:
            if args.split_classes:
                data_fd = open_or_create_directory_at(anchor_fd, "geometry")
                connector_root_fd = open_or_create_directory_at(anchor_fd, "connector")
            else:
                data_fd = os.dup(anchor_fd)
        except (OSError, ValueError) as exc:
            raise SystemExit(str(exc)) from None

        for seq, video in videos.items():
            out_fd: int | None = None
            connector_fd: int | None = None
            try:
                try:
                    out_fd = open_sequence_directory_at(data_fd, seq, args.overwrite)
                    if connector_root_fd is not None:
                        connector_fd = open_sequence_directory_at(
                            connector_root_fd, seq, args.overwrite
                        )
                except (OSError, ValueError) as exc:
                    raise SystemExit(str(exc)) from None
                out_dir = directory_fd_path(out_fd)
                connector_dir = (
                    directory_fd_path(connector_fd) if connector_fd is not None else None
                )
                existing = [name for name in os.listdir(out_fd) if name.endswith(".jpg")]
                if existing:
                    report.append(
                        {"seq": seq, "mode": "skip_existing", "saved": len(existing)}
                    )
                    continue
                if seq in manifest and "source_idx" in manifest[seq]:
                    report.append(
                        extract_manifest(
                            seq,
                            video,
                            manifest[seq]["source_idx"],
                            out_dir,
                            args.quality,
                        )
                    )
                elif args.motion_filter in {"flow", "parallax"}:
                    report.append(
                        extract_fps_flow(
                            seq,
                            video,
                            args.fps,
                            out_dir,
                            args.quality,
                            args.min_flow_px,
                            args.max_flow_px,
                            args.motion_filter,
                            args.keep_rotation_every,
                            args.split_classes,
                            connector_dir,
                        )
                    )
                else:
                    cap = cv2.VideoCapture(str(video))
                    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                    cap.release()
                    step = max(1, int(round(native_fps / args.fps)))
                    indices = list(range(0, total, step))
                    report.append(
                        extract_manifest(seq, video, indices, out_dir, args.quality)
                    )
            finally:
                if connector_fd is not None:
                    os.close(connector_fd)
                if out_fd is not None:
                    os.close(out_fd)
        try:
            write_json_report_at(anchor_fd, "frame_selection_report.json", report)
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
    finally:
        if connector_root_fd is not None:
            os.close(connector_root_fd)
        if data_fd is not None:
            os.close(data_fd)
        os.close(anchor_fd)
    for item in report:
        print(item)


if __name__ == "__main__":
    main()
