#!/usr/bin/env python3
"""Backend-neutral matcher layer for the map update pipeline.

WHY
  The update tool was written against XFeat + LighterGlue. The deployed
  localizer is now EDM. Running an update through the old frontend produces a
  reloc_map_xfeat_*.pt -- a bundle format that is no longer flown. Worse, the
  update would be measuring "can XFeat find itself here" when the question that
  matters is "can the thing that will actually fly find itself here".

  So the frontend moves behind this interface and EDM becomes the default.

THE ONE THING THAT MAKES THIS NON-TRIVIAL
  XFeat is a detector: it emits a fixed per-image keypoint table, reusable
  across every pair, so a match is a pair of indices into two stable tables and
  "query keypoint 417" means something on its own.

  EDM is detector-free: it emits different keypoints for every pair, so there
  is no per-image table and no index to dedup on. The update loop needs one,
  because it keeps only the single best 3D anchor per query observation.

  EDM's own structure supplies it. Every match keeps one side on the 1/8 coarse
  grid with the fine offset clamped to +/-4 px, so

      cell = round(kpt / 8)

  is an image-intrinsic identity that is stable across pairs. That is the same
  trick build_reloc_map_edm.py uses to build a keypoint table for a
  detector-free matcher, and it is verified exact round-trip on this project.

  So each backend supplies `query_keys` alongside the correspondences, and the
  dedup logic downstream stays identical:

      XFeat  query_keys = keypoint indices
      EDM    query_keys = query cell ids

  Quantizing the query side is safe even when the query is the REFINED side of
  the pair rather than the grid side: the measured sub-pixel spread within a
  cell is ~2.07 px median, comfortably inside the 8 px cell, so two matches of
  the same physical observation land on the same key.

WHAT IS AND IS NOT VERIFIED HERE
  The dedup/aggregation logic is unit tested with a fake backend. The two real
  backends import from the localization deploy tree (`--repo` / `--edm-deploy`),
  which is not vendored into this repository, needs a GPU and the EDM weights,
  and therefore is NOT exercised by these tests. Treat the first real EDM update
  run as an integration test and check `matcher_backend` in the report.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from edm_cells import (  # noqa: F401
    EDM_CELL,
    EDM_H,
    EDM_W,
    GRID_H,
    GRID_W,
    N_CELLS,
    cell_keys,
)


@dataclass
class QueryRecord:
    """One frame, prepared for whichever backend is active."""

    width: int
    height: int
    state: Any = None
    #: XFeat fills this at prepare time. EDM cannot -- it has no per-image
    #: keypoints until a pair is matched -- so it is filled in from the union of
    #: matched query cells once correspondences() has run.
    keypoints: np.ndarray | None = None

    def keypoints_or_empty(self) -> np.ndarray:
        if self.keypoints is None:
            return np.zeros((0, 2), dtype=np.float64)
        return np.asarray(self.keypoints, dtype=np.float64)


@dataclass
class RefMatch:
    """Correspondences between one query frame and one reference."""

    ref_name: str
    ref_index: int
    query_xy: np.ndarray      # (N, 2) query pixel coordinates
    xyz: np.ndarray           # (N, 3) map-frame 3D anchors
    query_keys: np.ndarray    # (N,) stable per-image identity, for dedup
    ref_keys: np.ndarray | None = None


@dataclass
class AnchoredResult:
    p2: np.ndarray
    p3: np.ndarray
    meta: list[dict] = field(default_factory=list)

    def as_tuple(self):
        if len(self.meta) == 0:
            return None, None, []
        return self.p2, self.p3, self.meta


def dedup_anchored(rows: Sequence[RefMatch]) -> AnchoredResult:
    """Keep one 3D anchor per query observation.

    Guarantees inliers <= number of distinct query observations, which is what
    makes the downstream inlier counts comparable between backends. First
    reference to claim a key wins, and references arrive in retrieval-rank
    order, so the best-retrieved reference wins ties.
    """
    best: dict[Any, dict] = {}
    for row in rows:
        query_xy = np.asarray(row.query_xy, dtype=np.float64)
        xyz = np.asarray(row.xyz, dtype=np.float64)
        keys = np.asarray(row.query_keys)
        if len(query_xy) == 0:
            continue
        if not (len(query_xy) == len(xyz) == len(keys)):
            raise ValueError(
                f"ref {row.ref_name}: ragged correspondence arrays "
                f"({len(query_xy)}, {len(xyz)}, {len(keys)})"
            )
        for i in range(len(keys)):
            point = xyz[i]
            if not np.isfinite(point).all():
                continue
            key = keys[i]
            key = tuple(key.tolist()) if isinstance(key, np.ndarray) else key.item() if hasattr(key, "item") else key
            if key in best:
                continue
            best[key] = {
                "qidx": key,
                "p2": query_xy[i],
                "xyz": point,
                "ref_name": row.ref_name,
                "ref_index": int(row.ref_index),
                "ref_kp": int(row.ref_keys[i]) if row.ref_keys is not None else -1,
            }
    if not best:
        return AnchoredResult(np.zeros((0, 2)), np.zeros((0, 3)), [])
    values = list(best.values())
    return AnchoredResult(
        np.array([v["p2"] for v in values]),
        np.array([v["xyz"] for v in values]),
        values,
    )




def _add_to_path(directory: Path | None) -> None:
    if directory is None:
        return
    resolved = str(Path(directory).resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


class EDMUpdateMatcher:
    """Default backend. Routes update-time matching through the deployed EDM stack.

    Wraps EDMLocalizer.correspondences_by_ref(), which already returns per-ref
    (query_xy, xyz, count) via the matcher-neutral CorrespondenceProvider. The
    only thing added here is the query-cell identity used for dedup.
    """

    name = "edm"

    def __init__(self, localizer, cell: int = EDM_CELL):
        self.localizer = localizer
        self.cell = cell

    @classmethod
    def from_deploy(cls, deploy_dir: Path, bundle_path: Path, camera, *, topk: int,
                    min_conf: float, pnp_max_error: float, min_inliers: int, device: str = "cuda"):
        """Build from the localization deploy tree. Not exercised by unit tests."""
        _add_to_path(deploy_dir)
        from edm_matcher import EDMMatcher  # type: ignore
        from reloc_localizer_edm import EDMLocalizer, EDMRelocMap, MegaLocQuery  # type: ignore

        reloc_map = EDMRelocMap.load(str(bundle_path))
        return cls(
            EDMLocalizer(
                reloc_map=reloc_map,
                camera=camera,
                matcher=EDMMatcher(device=device),
                megaloc=MegaLocQuery(device=device, input_size=322),
                topk=topk,
                min_conf=min_conf,
                pnp_max_error=pnp_max_error,
                min_inliers=min_inliers,
            )
        )

    def prepare_query(self, rgb: np.ndarray) -> QueryRecord:
        height, width = rgb.shape[:2]
        # EDM has no per-image keypoints until a pair is matched, so `keypoints`
        # stays None here and is filled by correspondences().
        return QueryRecord(width=width, height=height,
                           state=self.localizer.prepare_query(rgb), keypoints=None)

    @property
    def scale(self) -> float:
        """EDM canvas px -> camera px. correspondences_by_ref returns CAMERA px."""
        return float(getattr(self.localizer, "scale", 1.0))

    def canvas_xy(self, camera_xy: np.ndarray) -> np.ndarray:
        return np.asarray(camera_xy, dtype=np.float64) / self.scale

    def correspondences(self, query: QueryRecord, ref_names: Sequence[str],
                        ref_indices: Sequence[int], min_conf: float | None = None
                        ) -> list[RefMatch]:
        rows = self.localizer.correspondences_by_ref(query.state, list(ref_names))
        out: list[RefMatch] = []
        seen: dict[tuple[int, int], np.ndarray] = {}
        for name, index, row in zip(ref_names, ref_indices, rows):
            query_xy = np.asarray(row[0], dtype=np.float64).reshape(-1, 2)
            xyz = np.asarray(row[1], dtype=np.float64).reshape(-1, 3)
            # Quantise in EDM canvas coordinates, not camera coordinates, so the
            # key really is EDM's own cell id and can index xyz_by_cell directly.
            keys = cell_keys(self.canvas_xy(query_xy), self.cell)
            for k, xy in zip(keys, query_xy):
                seen.setdefault((int(k[0]), int(k[1])), xy)
            out.append(RefMatch(name, int(index), query_xy, xyz, keys))
        # Synthesise the per-image keypoint table the update loop needs for tile
        # statistics and connector inheritance: the union of matched query cells.
        query.keypoints = (
            np.array(list(seen.values()), dtype=np.float64)
            if seen else np.zeros((0, 2), dtype=np.float64)
        )
        return out

    def bundle_keyframe(self, query: QueryRecord, meta: Sequence[dict], rgb: np.ndarray,
                        name: str) -> dict:
        """Build an EDM reference entry: cell->xyz LUT plus the grayscale JPEG.

        Schema must match EDMRelocMap.load(): xyz_by_cell is (N_CELLS, 3) float32
        with NaN meaning unanchored, and image_jpg is an encoded uint8 buffer of
        the EDM_W x EDM_H grayscale canvas. The bundle carries the image because
        a detector-free matcher has to SEE the reference at flight time.
        """
        import cv2  # local: only the EDM path needs it

        xyz_by_cell = np.full((N_CELLS, 3), np.nan, dtype=np.float32)
        for entry in meta:
            key = entry["qidx"]
            if not isinstance(key, tuple) or len(key) != 2:
                raise ValueError(
                    f"EDM keyframe expects (cx, cy) cell keys, got {key!r}. "
                    "This means correspondences() and the bundle writer disagree."
                )
            cx = int(np.clip(key[0], 0, GRID_W - 1))
            cy = int(np.clip(key[1], 0, GRID_H - 1))
            xyz_by_cell[cy * GRID_W + cx] = entry["xyz"]

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
        if gray.shape[:2] != (EDM_H, EDM_W):
            gray = cv2.resize(gray, (EDM_W, EDM_H), interpolation=cv2.INTER_AREA)
        ok, jpg = cv2.imencode(".jpg", gray, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            raise RuntimeError(f"failed to encode reference image for {name!r}")
        return {
            "xyz_by_cell": xyz_by_cell,
            "image_jpg": np.frombuffer(jpg.tobytes(), np.uint8),
        }

    @staticmethod
    def bundle_meta() -> dict:
        return {"feature": "edm", "bundle_vpr": "megaloc",
                "grid_width": GRID_W, "grid_height": GRID_H}


class XFeatUpdateMatcher:
    """Legacy backend, kept only so an EDM-vs-XFeat A/B can be run on one map.

    Do not ship an update built with this: the bundle it feeds is a format that
    is no longer deployed.
    """

    name = "xfeat"

    def __init__(self, xfeat, refs: dict, topk_default: int, min_conf_default: float):
        self.xfeat = xfeat
        self.refs = refs
        self.topk_default = topk_default
        self.min_conf_default = min_conf_default
        self.device = "cuda"

    @classmethod
    def from_deploy(cls, deploy_dir: Path, scripts_dir: Path | None, refs: dict,
                    *, qk: int, topk: int, min_conf: float):
        _add_to_path(deploy_dir)
        _add_to_path(scripts_dir)
        from reloc_localizer_xfeat import load_xfeat  # type: ignore

        return cls(load_xfeat(qk), refs, topk, min_conf)

    def prepare_query(self, rgb: np.ndarray, qk: int | None = None) -> QueryRecord:
        from reloc_localizer_xfeat import extract_xfeat  # type: ignore

        height, width = rgb.shape[:2]
        feats = extract_xfeat(self.xfeat, rgb, qk) if qk else extract_xfeat(self.xfeat, rgb)
        keypoints = np.asarray(feats["keypoints"].detach().cpu())
        return QueryRecord(width=width, height=height, state=feats, keypoints=keypoints)

    def correspondences(self, query: QueryRecord, ref_names: Sequence[str],
                        ref_indices: Sequence[int], min_conf: float | None = None
                        ) -> list[RefMatch]:
        conf = self.min_conf_default if min_conf is None else min_conf
        keypoints = query.keypoints_or_empty()
        out: list[RefMatch] = []
        for name, index in zip(ref_names, ref_indices):
            ref = self.refs[name]
            feats = {k: (v.to(self.device) if hasattr(v, "to") else v)
                     for k, v in ref["feats"].items()}
            try:
                _, _, matched = self.xfeat.match_lighterglue(query.state, feats, min_conf=conf)
            except Exception:
                continue
            if matched is None or len(matched) == 0:
                continue
            matched = np.asarray(
                matched.detach().cpu() if hasattr(matched, "detach") else matched
            )
            ref_xyz = np.asarray(ref["xyz"])
            query_idx = matched[:, 0].astype(int)
            ref_idx = matched[:, 1].astype(int)
            out.append(
                RefMatch(
                    ref_name=name,
                    ref_index=int(index),
                    query_xy=keypoints[query_idx],
                    xyz=ref_xyz[ref_idx],
                    query_keys=query_idx,   # XFeat's own stable identity
                    ref_keys=ref_idx,
                )
            )
        return out

    def bundle_keyframe(self, query: QueryRecord, meta: Sequence[dict], rgb: np.ndarray,
                        name: str) -> dict:
        """Legacy XFeat reference entry: per-keypoint descriptors + per-keypoint xyz."""
        keypoints = query.keypoints_or_empty()
        inherited = np.full((len(keypoints), 3), np.nan, np.float32)
        for entry in meta:
            index = entry["qidx"]
            if not isinstance(index, (int, np.integer)):
                raise ValueError(
                    f"XFeat keyframe expects integer keypoint indices, got {index!r}"
                )
            if 0 <= int(index) < len(inherited):
                inherited[int(index)] = entry["xyz"]
        feats = query.state
        height, width = query.height, query.width
        return {
            "feats": {
                "keypoints": feats["keypoints"].detach().cpu(),
                "descriptors": feats["descriptors"].detach().cpu(),
                "scores": feats["scores"].detach().cpu() if "scores" in feats else None,
                "image_size": (width, height),
            },
            "xyz": inherited,
        }

    @staticmethod
    def bundle_meta() -> dict:
        return {"feature": "xfeat", "bundle_vpr": "megaloc"}


def build_matcher(kind: str, **kwargs):
    """Factory. `kind` comes straight from --matcher."""
    if kind == "edm":
        return EDMUpdateMatcher.from_deploy(**kwargs)
    if kind == "xfeat":
        return XFeatUpdateMatcher.from_deploy(**kwargs)
    raise SystemExit(f"unknown matcher backend {kind!r}; expected 'edm' or 'xfeat'")
