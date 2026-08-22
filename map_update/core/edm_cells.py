"""EDM coarse-cell identity. Single grid definition for update and S8."""
from __future__ import annotations

import numpy as np

EDM_CELL = 8
EDM_W, EDM_H = 1024, 576
GRID_W, GRID_H = EDM_W // EDM_CELL, EDM_H // EDM_CELL  # 128 x 72
N_CELLS = GRID_W * GRID_H  # 9216


def cell_keys(points: np.ndarray, cell: int = EDM_CELL) -> np.ndarray:
    """round(kpt / cell) -- EDM's image-intrinsic keypoint identity."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    return np.rint(pts / float(cell)).astype(np.int64)


def cell_identity_ok(refs: dict) -> bool:
    """True when every finite xyz_by_cell row uses the EDM (cx, cy) packing."""
    if not isinstance(refs, dict) or not refs:
        return False
    for payload in refs.values():
        if not isinstance(payload, dict):
            return False
        xyz = np.asarray(payload.get("xyz_by_cell"))
        if xyz.ndim != 2 or xyz.shape != (N_CELLS, 3):
            return False
        finite = np.isfinite(xyz[:, 0])
        for index in np.flatnonzero(finite):
            cell_i = int(index)
            cx = cell_i % GRID_W
            cy = cell_i // GRID_W
            mapped = cell_keys(np.asarray([[cx * EDM_CELL, cy * EDM_CELL]]))
            if mapped.shape != (1, 2):
                return False
            if int(mapped[0, 0]) != cx or int(mapped[0, 1]) != cy:
                return False
    return True
