"""GGPT sidecar admission (Chen et al., CVPR 2026).

GGPT refines dense feed-forward point maps under sparse SfM guidance. It must
not replace S5 global mapping or S8 EDM localization geometry. Fuhe already
failed a sparse-overlap gate: missing co-visibility cannot be hallucinated.
"""
from __future__ import annotations

from ggpt_sidecar import plan_ggpt_sidecar


def test_unlocked_poses_are_rejected_even_with_overlap() -> None:
    plan = plan_ggpt_sidecar(
        image_names=["a", "b", "c", "d"],
        shared_points={("a", "b"): 80, ("b", "c"): 70, ("c", "d"): 60, ("a", "d"): 40},
        poses_locked=False,
        intrinsics_delta=0.0,
    )
    assert plan.accepted is False
    assert plan.role == "rejected"
    assert any("poses" in reason for reason in plan.reasons)
    assert plan.tiles == ()


def test_missing_covisibility_fails_the_overlap_gate() -> None:
    plan = plan_ggpt_sidecar(
        image_names=["p112_a", "p114_a", "p114_b"],
        shared_points={("p112_a", "p114_a"): 12, ("p114_a", "p114_b"): 40},
        poses_locked=True,
        min_pair_overlap=50,
    )
    assert plan.accepted is False
    assert plan.role == "rejected"
    assert any("overlap" in reason for reason in plan.reasons)


def test_pose_locked_tiles_are_visualization_only() -> None:
    names = [f"v{i}" for i in range(8)]
    shared = {
        (names[i], names[j]): 80 - 4 * abs(j - i)
        for i in range(8)
        for j in range(i + 1, 8)
    }
    plan = plan_ggpt_sidecar(
        image_names=names,
        shared_points=shared,
        poses_locked=True,
        intrinsics_delta=0.0,
        tile_size=8,
        min_pair_overlap=50,
    )
    assert plan.accepted is True
    assert plan.role == "visualization_only"
    assert plan.role != "localization_map"
    assert len(plan.tiles) == 1
    assert plan.tiles[0].n_views == 8
    assert plan.tiles[0].min_pair_overlap >= 50


def test_isolated_name_does_not_block_a_later_valid_tile() -> None:
    clique = [f"v{i}" for i in range(8)]
    shared = {
        (clique[i], clique[j]): 80
        for i in range(8)
        for j in range(i + 1, 8)
    }
    plan = plan_ggpt_sidecar(
        image_names=["iso", *clique],
        shared_points=shared,
        poses_locked=True,
        tile_size=8,
        min_pair_overlap=50,
    )
    assert plan.accepted is True
    assert plan.tiles[0].image_names == tuple(clique)


def test_intrinsics_drift_and_invalid_tile_size_are_rejected() -> None:
    names = [f"v{i}" for i in range(8)]
    shared = {
        (names[i], names[j]): 80
        for i in range(8)
        for j in range(i + 1, 8)
    }
    drifted = plan_ggpt_sidecar(
        image_names=names,
        shared_points=shared,
        poses_locked=True,
        intrinsics_delta=1e-3,
    )
    assert drifted.accepted is False
    too_big = plan_ggpt_sidecar(
        image_names=names,
        shared_points=shared,
        poses_locked=True,
        tile_size=32,
    )
    assert too_big.accepted is False


def test_large_maps_are_tiled_instead_of_one_vggt_pass() -> None:
    names = [f"f{i:03d}" for i in range(24)]
    shared = {}
    for i, a in enumerate(names):
        for j in range(i + 1, min(i + 6, 24)):
            shared[(a, names[j])] = 70
    plan = plan_ggpt_sidecar(
        image_names=names,
        shared_points=shared,
        poses_locked=True,
        tile_size=8,
        max_views_per_tile=16,
        min_pair_overlap=50,
    )
    assert plan.accepted is True
    assert plan.role == "visualization_only"
    assert len(plan.tiles) >= 2
    assert all(tile.n_views <= 16 for tile in plan.tiles)
    assert all(tile.n_views >= 4 for tile in plan.tiles)
