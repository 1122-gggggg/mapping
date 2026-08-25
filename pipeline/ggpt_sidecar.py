"""Pose-locked GGPT dense-refinement admission.

Chen et al., "GGPT: Geometry-Grounded Point Transformer", CVPR 2026, refine a
dense feed-forward point map with sparse SfM geometry. The paper's own SfM
front-end is a *sparse-view* (4–16 image) recipe. That is not a replacement
for this repo's S5 global mapper or S8 EDM cell-anchor map.

The only production-safe use is a visualization/QA sidecar after poses and
intrinsics are locked, on tiles that already have real co-visibility. A failed
overlap gate means "do not run GGPT", not "the transformer will invent the
missing bridge".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class GgptTile:
    image_names: tuple[str, ...]
    n_views: int
    min_pair_overlap: int


@dataclass(frozen=True)
class GgptPlan:
    accepted: bool
    role: str
    reasons: tuple[str, ...]
    tiles: tuple[GgptTile, ...]


def plan_ggpt_sidecar(
    *,
    image_names: Sequence[str],
    shared_points: Mapping[tuple[str, str], int],
    poses_locked: bool,
    intrinsics_delta: float = 0.0,
    tile_size: int = 8,
    max_views_per_tile: int = 16,
    min_views_per_tile: int = 4,
    min_pair_overlap: int = 50,
    max_intrinsics_delta: float = 1e-6,
) -> GgptPlan:
    """Admit GGPT tiles for visualization only, or reject with reasons."""
    names = list(dict.fromkeys(image_names))
    reasons: list[str] = []
    if not poses_locked:
        reasons.append("poses are not locked; GGPT must not re-estimate the S5 gauge")
    if abs(float(intrinsics_delta)) > float(max_intrinsics_delta):
        reasons.append(
            "intrinsics moved relative to the seed; GGPT triangulation is not allowed "
            "to refine camera parameters"
        )
    if tile_size < min_views_per_tile or tile_size > max_views_per_tile:
        reasons.append(
            f"tile_size={tile_size} is outside the paper's sparse-view range "
            f"[{min_views_per_tile}, {max_views_per_tile}]"
        )
    if reasons:
        return GgptPlan(False, "rejected", tuple(reasons), ())

    overlap = _undirected_overlap(shared_points)
    tiles = _grow_tiles(
        names,
        overlap,
        tile_size=tile_size,
        min_views=min_views_per_tile,
        min_pair_overlap=min_pair_overlap,
    )
    if not tiles:
        return GgptPlan(
            False,
            "rejected",
            (
                "overlap gate failed: no tile has enough shared sparse points; "
                "GGPT cannot hallucinate missing co-visibility",
            ),
            (),
        )
    return GgptPlan(True, "visualization_only", (), tuple(tiles))


def _undirected_overlap(
    shared_points: Mapping[tuple[str, str], int],
) -> dict[tuple[str, str], int]:
    overlap: dict[tuple[str, str], int] = {}
    for (a, b), count in shared_points.items():
        if a == b or int(count) <= 0:
            continue
        key = (a, b) if a < b else (b, a)
        overlap[key] = max(int(count), overlap.get(key, 0))
    return overlap


def _pair_overlap(overlap: dict[tuple[str, str], int], a: str, b: str) -> int:
    key = (a, b) if a < b else (b, a)
    return int(overlap.get(key, 0))


def _grow_tiles(
    names: Sequence[str],
    overlap: dict[tuple[str, str], int],
    *,
    tile_size: int,
    min_views: int,
    min_pair_overlap: int,
) -> list[GgptTile]:
    remaining = list(names)
    tiles: list[GgptTile] = []
    while remaining:
        seed = remaining.pop(0)
        members = [seed]
        while len(members) < tile_size and remaining:
            best_index = -1
            best_score = -1
            for index, candidate in enumerate(remaining):
                score = max(_pair_overlap(overlap, candidate, member) for member in members)
                if score > best_score:
                    best_score = score
                    best_index = index
            if best_index < 0 or best_score < min_pair_overlap:
                break
            candidate = remaining[best_index]
            positive = [
                _pair_overlap(overlap, candidate, member)
                for member in members
                if _pair_overlap(overlap, candidate, member) > 0
            ]
            if not positive or min(positive) < min_pair_overlap:
                break
            members.append(remaining.pop(best_index))
        if len(members) < min_views:
            remaining[0:0] = members[1:]
            continue
        pair_values = [
            _pair_overlap(overlap, members[i], members[j])
            for i in range(len(members))
            for j in range(i + 1, len(members))
            if _pair_overlap(overlap, members[i], members[j]) > 0
        ]
        if not pair_values or min(pair_values) < min_pair_overlap:
            remaining[0:0] = members[1:]
            continue
        tiles.append(
            GgptTile(
                image_names=tuple(members),
                n_views=len(members),
                min_pair_overlap=min(pair_values),
            )
        )
    return tiles


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--covisibility",
        required=True,
        help="JSON {image_names: [...], shared_points: [[a,b,count], ...]}",
    )
    parser.add_argument("--poses-locked", action="store_true")
    parser.add_argument("--intrinsics-delta", type=float, default=0.0)
    parser.add_argument("--tile-size", type=int, default=8)
    parser.add_argument("--min-pair-overlap", type=int, default=50)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    payload = json.loads(Path(args.covisibility).read_text(encoding="utf-8"))
    names = payload.get("image_names") or payload.get("images") or []
    raw_pairs = payload.get("shared_points") or payload.get("pairs") or []
    shared: dict[tuple[str, str], int] = {}
    if isinstance(raw_pairs, dict):
        for key, count in raw_pairs.items():
            a, b = key.split(",") if isinstance(key, str) and "," in key else (None, None)
            if a and b:
                shared[(a, b)] = int(count)
    else:
        for row in raw_pairs:
            shared[(str(row[0]), str(row[1]))] = int(row[2])

    plan = plan_ggpt_sidecar(
        image_names=[str(name) for name in names],
        shared_points=shared,
        poses_locked=bool(args.poses_locked),
        intrinsics_delta=args.intrinsics_delta,
        tile_size=args.tile_size,
        min_pair_overlap=args.min_pair_overlap,
    )
    output = {
        "accepted": plan.accepted,
        "role": plan.role,
        "reasons": list(plan.reasons),
        "tiles": [
            {
                "image_names": list(tile.image_names),
                "n_views": tile.n_views,
                "min_pair_overlap": tile.min_pair_overlap,
            }
            for tile in plan.tiles
        ],
        "paper": "Chen et al., GGPT: Geometry-Grounded Point Transformer, CVPR 2026",
    }
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0 if plan.accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
