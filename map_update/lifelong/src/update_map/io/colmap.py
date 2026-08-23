from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import BinaryIO, Iterable

import numpy as np

from ..models import BaseMap, Camera, Landmark, MapImage, Pose
from ..states import GeometryProvenance

CAMERA_MODELS: dict[int, tuple[str, int]] = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
    11: ("RAD_TAN_THIN_PRISM_FISHEYE", 16),
}
CAMERA_MODELS_BY_NAME = {name: (model_id, count) for model_id, (name, count) in CAMERA_MODELS.items()}


def _read_bytes(handle: BinaryIO, count: int, fmt: str, endian: str = "<") -> tuple:
    data = handle.read(count)
    if len(data) != count:
        raise EOFError("Unexpected end of COLMAP binary file")
    return struct.unpack(endian + fmt, data)


def _read_c_string(handle: BinaryIO) -> str:
    result = bytearray()
    while True:
        char = handle.read(1)
        if not char:
            raise EOFError("Unexpected end while reading null-terminated string")
        if char == b"\x00":
            return result.decode("utf-8")
        result.extend(char)


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    q = np.asarray(qvec, dtype=np.float64).reshape(4)
    q /= max(np.linalg.norm(q), 1e-15)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def _non_comment_lines(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if value and not value.startswith("#"):
                yield value


def read_cameras_text(path: Path) -> dict[int, Camera]:
    cameras: dict[int, Camera] = {}
    for line in _non_comment_lines(path):
        tokens = line.split()
        camera_id = int(tokens[0])
        model = tokens[1]
        cameras[camera_id] = Camera(
            camera_id=camera_id,
            model=model,
            width=int(tokens[2]),
            height=int(tokens[3]),
            params=np.asarray([float(item) for item in tokens[4:]], dtype=np.float64),
        )
    return cameras


def read_images_text(path: Path) -> dict[int, MapImage]:
    lines = list(_non_comment_lines(path))
    images: dict[int, MapImage] = {}
    if len(lines) % 2 != 0:
        raise ValueError("COLMAP images.txt must contain two non-comment lines per image")
    for index in range(0, len(lines), 2):
        header = lines[index].split()
        image_id = int(header[0])
        qvec = np.asarray([float(value) for value in header[1:5]], dtype=np.float64)
        tvec = np.asarray([float(value) for value in header[5:8]], dtype=np.float64)
        camera_id = int(header[8])
        name = " ".join(header[9:])
        point_tokens = lines[index + 1].split()
        if point_tokens:
            if len(point_tokens) % 3 != 0:
                raise ValueError(f"Invalid point triplets for image {image_id}")
            values = np.asarray(point_tokens, dtype=object).reshape(-1, 3)
            xys = values[:, :2].astype(np.float64)
            point3d_ids = values[:, 2].astype(np.int64)
        else:
            xys = np.empty((0, 2), dtype=np.float64)
            point3d_ids = np.empty(0, dtype=np.int64)
        images[image_id] = MapImage(
            image_id=image_id,
            name=name,
            camera_id=camera_id,
            pose=Pose(qvec_to_rotmat(qvec), tvec),
            xys=xys,
            point3d_ids=point3d_ids,
        )
    return images


def read_points3d_text(path: Path) -> dict[int, Landmark]:
    points: dict[int, Landmark] = {}
    for line in _non_comment_lines(path):
        tokens = line.split()
        point_id = int(tokens[0])
        track_tokens = tokens[8:]
        if len(track_tokens) % 2 != 0:
            raise ValueError(f"Invalid track for point {point_id}")
        track = [
            (int(track_tokens[i]), int(track_tokens[i + 1]))
            for i in range(0, len(track_tokens), 2)
        ]
        points[point_id] = Landmark(
            point3d_id=point_id,
            xyz=np.asarray([float(value) for value in tokens[1:4]], dtype=np.float64),
            rgb=np.asarray([int(value) for value in tokens[4:7]], dtype=np.uint8),
            error=float(tokens[7]),
            track=track,
        )
    return points


def read_cameras_binary(path: Path) -> dict[int, Camera]:
    cameras: dict[int, Camera] = {}
    with path.open("rb") as handle:
        num_cameras = _read_bytes(handle, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_id, model_id, width, height = _read_bytes(handle, 24, "IiQQ")
            if model_id not in CAMERA_MODELS:
                raise ValueError(f"Unsupported COLMAP camera model id: {model_id}")
            model_name, num_params = CAMERA_MODELS[model_id]
            params = np.asarray(_read_bytes(handle, 8 * num_params, "d" * num_params), dtype=np.float64)
            cameras[camera_id] = Camera(camera_id, model_name, width, height, params)
    return cameras


def read_images_binary(path: Path) -> dict[int, MapImage]:
    images: dict[int, MapImage] = {}
    with path.open("rb") as handle:
        num_images = _read_bytes(handle, 8, "Q")[0]
        for _ in range(num_images):
            properties = _read_bytes(handle, 64, "I" + "d" * 7 + "I")
            image_id = int(properties[0])
            qvec = np.asarray(properties[1:5], dtype=np.float64)
            tvec = np.asarray(properties[5:8], dtype=np.float64)
            camera_id = int(properties[8])
            name = _read_c_string(handle)
            num_points2d = _read_bytes(handle, 8, "Q")[0]
            blob = handle.read(24 * num_points2d)
            if len(blob) != 24 * num_points2d:
                raise ValueError(f"Truncated 2D points for image {image_id}")
            packed = np.frombuffer(
                blob, dtype=np.dtype([("x", "<f8"), ("y", "<f8"), ("id", "<i8")])
            )
            xys = np.column_stack((packed["x"], packed["y"]))
            point3d_ids = packed["id"].astype(np.int64, copy=False)
            images[image_id] = MapImage(
                image_id=image_id,
                name=name,
                camera_id=camera_id,
                pose=Pose(qvec_to_rotmat(qvec), tvec),
                xys=xys,
                point3d_ids=point3d_ids,
            )
    return images


def read_points3d_binary(path: Path) -> dict[int, Landmark]:
    points: dict[int, Landmark] = {}
    with path.open("rb") as handle:
        num_points = _read_bytes(handle, 8, "Q")[0]
        for _ in range(num_points):
            point_id = _read_bytes(handle, 8, "Q")[0]
            xyz = np.asarray(_read_bytes(handle, 24, "ddd"), dtype=np.float64)
            rgb = np.asarray(_read_bytes(handle, 3, "BBB"), dtype=np.uint8)
            error = float(_read_bytes(handle, 8, "d")[0])
            track_length = _read_bytes(handle, 8, "Q")[0]
            track = [tuple(map(int, _read_bytes(handle, 8, "II"))) for _ in range(track_length)]
            points[int(point_id)] = Landmark(int(point_id), xyz, rgb, error, track)
    return points


def _load_provenance(root: Path, points: dict[int, Landmark]) -> None:
    sidecar = root / "point_provenance.json"
    virtual_list = root / "virtual_point_ids.txt"
    mapping: dict[str, str] = {}
    if sidecar.exists():
        mapping = json.loads(sidecar.read_text(encoding="utf-8"))
    virtual_ids: set[int] = set()
    if virtual_list.exists():
        virtual_ids = {
            int(line.strip())
            for line in virtual_list.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
    for point_id, point in points.items():
        raw = mapping.get(str(point_id))
        if point_id in virtual_ids:
            point.provenance = GeometryProvenance.VIRTUAL_BA_ONLY
        elif raw is not None:
            point.provenance = GeometryProvenance(raw)


def load_colmap_reconstruction(root: str | Path) -> BaseMap:
    root_path = Path(root)
    candidates = [root_path, root_path / "sparse", root_path / "sparse" / "0"]
    model_root = next(
        (
            candidate
            for candidate in candidates
            if (candidate / "cameras.bin").exists() or (candidate / "cameras.txt").exists()
        ),
        None,
    )
    if model_root is None:
        raise FileNotFoundError(
            f"No COLMAP model found under {root_path}; expected cameras/images/points3D .bin or .txt"
        )
    if (model_root / "cameras.bin").exists():
        cameras = read_cameras_binary(model_root / "cameras.bin")
        images = read_images_binary(model_root / "images.bin")
        points = read_points3d_binary(model_root / "points3D.bin")
        source_format = "binary"
    else:
        cameras = read_cameras_text(model_root / "cameras.txt")
        images = read_images_text(model_root / "images.txt")
        points = read_points3d_text(model_root / "points3D.txt")
        source_format = "text"
    _load_provenance(model_root, points)
    return BaseMap(cameras, images, points, root=model_root.resolve(), source_format=source_format)
