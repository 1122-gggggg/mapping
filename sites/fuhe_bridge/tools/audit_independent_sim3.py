#!/usr/bin/env python3
"""Build per-sequence maps and independently verify cross-direction Sim3 bridges."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sqlite3
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from audit_map_geometry import (
    _json_sim3,
    _two_separated_clusters,
    estimate_sim3,
    robust_spatial_span,
)
from audit_dg_graph import classify_accepted_cross_pairs, load_forced_pairs
from finalize_edm_model import (
    FIXED_CAMERA_PARAMS,
    fixed_camera_evidence,
    point_ids_spanning_sequence_edges,
)
from ts_common import (
    BUILD,
    Gate,
    read_fresh_gate_stage_metrics,
    required_check_ids,
    sha256,
)
from resource_guard import required_cli_path, run_global_heavy_job


MAX_IMAGE_ID = 2_147_483_647
MIN_CLUSTER_INLIERS = 30
SEQUENCE_DIRECTIONS = {video.seq: video.direction for video in BUILD}
CANONICAL_CAMERA_ID = 1
PINHOLE_CAMERA_MODEL_ID = 1
CAMERA_SENSOR_TYPE = 0


def accepted_pair_records_for_edges(
    names: list[str],
    pairs,
    scores,
    *,
    threshold: float,
    directions: dict[str, str],
    conditional_pairs: set[tuple[str, str]],
    trusted_edges: set[tuple[str, str]],
) -> dict[tuple[str, str], list[dict[str, object]]]:
    """Select trusted edges after classifying the full DG-accepted pair set."""
    records = classify_accepted_cross_pairs(
        names,
        pairs,
        scores,
        threshold=threshold,
        directions=directions,
        conditional_pairs=conditional_pairs,
    )
    normalized_edges = {tuple(sorted(edge)) for edge in trusted_edges}
    return {
        edge: rows for edge, rows in records.items() if edge in normalized_edges
    }


def image_pair_id(image_id1: int, image_id2: int) -> int:
    """Encode a COLMAP unordered image pair."""
    smaller, larger = sorted((int(image_id1), int(image_id2)))
    return smaller * MAX_IMAGE_ID + larger


def decode_image_pair_id(pair_id: int) -> tuple[int, int]:
    """Decode a COLMAP unordered image pair."""
    return int(pair_id) // MAX_IMAGE_ID, int(pair_id) % MAX_IMAGE_ID


def robust_sim3(
    source: np.ndarray,
    target: np.ndarray,
    *,
    max_error: float,
    iterations: int = 2000,
    seed: int = 0,
) -> dict:
    """Estimate Sim3 with deterministic minimal-set RANSAC and inlier refinement."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must both have shape (N, 3)")
    if len(source) < 3:
        raise ValueError("at least three 3D correspondences are required")
    if not math.isfinite(max_error) or max_error <= 0:
        raise ValueError("max_error must be finite and positive")
    rng = np.random.default_rng(seed)
    best_mask: np.ndarray | None = None
    best_key = (-1, -math.inf)
    for _ in range(iterations):
        sample = rng.choice(len(source), size=3, replace=False)
        try:
            candidate = estimate_sim3(source[sample], target[sample])
        except (ValueError, np.linalg.LinAlgError):
            continue
        if candidate["scale"] <= 0 or not math.isfinite(candidate["scale"]):
            continue
        prediction = (
            candidate["scale"] * (source @ candidate["rotation"].T)
            + candidate["translation"]
        )
        errors = np.linalg.norm(prediction - target, axis=1)
        mask = errors <= max_error
        count = int(mask.sum())
        median = float(np.median(errors[mask])) if count else math.inf
        key = (count, -median)
        if key > best_key:
            best_key = key
            best_mask = mask
    if best_mask is None or int(best_mask.sum()) < 3:
        raise ValueError("RANSAC found fewer than three Sim3 inliers")
    for _ in range(5):
        result = estimate_sim3(source[best_mask], target[best_mask])
        prediction = result["scale"] * (source @ result["rotation"].T) + result[
            "translation"
        ]
        errors = np.linalg.norm(prediction - target, axis=1)
        updated = errors <= max_error
        if np.array_equal(updated, best_mask):
            break
        if int(updated.sum()) < 3:
            break
        best_mask = updated
    result = estimate_sim3(source[best_mask], target[best_mask])
    result["inliers"] = int(best_mask.sum())
    result["correspondences"] = len(source)
    result["inlier_fraction"] = float(best_mask.mean())
    return result


def resolve_quarantined_edges(
    evidence: dict[str, dict], quarantine_counts: dict[str, int]
) -> tuple[bool, list[str]]:
    """Accept only independently passing edges or failed edges with zero final tracks."""
    trusted = []
    accepted: set[str] = set()
    for edge, payload in sorted(evidence.items()):
        try:
            left, right = edge.split("|", 1)
        except ValueError:
            continue
        estimates = payload.get("estimates", [])
        cross_direction = (
            left in SEQUENCE_DIRECTIONS
            and right in SEQUENCE_DIRECTIONS
            and SEQUENCE_DIRECTIONS[left] != SEQUENCE_DIRECTIONS[right]
        )
        supported = (
            payload.get("status") == "PASS"
            and cross_direction
            and len(estimates) == 2
            and all(
                type(estimate.get("inliers")) is int
                and estimate["inliers"] >= MIN_CLUSTER_INLIERS
                for estimate in estimates
            )
        )
        payload["trusted_cross_direction"] = supported
        payload["minimum_cluster_inliers"] = MIN_CLUSTER_INLIERS
        if supported:
            trusted.append(edge)
            accepted.add(edge)
    for edge, count in quarantine_counts.items():
        payload = evidence.get(edge)
        if payload is None:
            continue
        payload["remaining_final_map_shared_points"] = int(count)
        if payload.get("status") == "FAIL" and count == 0:
            payload["raw_status"] = "FAIL"
            payload["status"] = "QUARANTINED"
            accepted.add(edge)
    passed = len(trusted) >= 2 and accepted == set(evidence)
    return passed, trusted


def _canonicalize_local_camera_graph(
    connection: sqlite3.Connection,
    image_ids: set[int],
) -> dict[str, int]:
    """Collapse COLMAP camera/rig metadata to one fixed Fuhe camera sensor."""
    required_columns = {
        "cameras": {
            "camera_id",
            "model",
            "width",
            "height",
            "params",
            "prior_focal_length",
        },
        "images": {"image_id", "camera_id"},
        "rigs": {"rig_id", "ref_sensor_id", "ref_sensor_type"},
        "frames": {"frame_id", "rig_id"},
        "frame_data": {"frame_id", "data_id", "sensor_id", "sensor_type"},
    }
    for table, required in required_columns.items():
        columns = {
            str(row[1])
            for row in connection.execute(f"PRAGMA table_info({table})")
        }
        missing = required - columns
        if missing:
            raise ValueError(f"{table}: missing camera-graph columns {sorted(missing)}")

    params = np.asarray(FIXED_CAMERA_PARAMS, dtype="<f8").tobytes()
    updated = connection.execute(
        "UPDATE cameras SET model = ?, width = ?, height = ?, params = ?, "
        "prior_focal_length = 1 WHERE camera_id = ?",
        (
            PINHOLE_CAMERA_MODEL_ID,
            1920,
            1080,
            params,
            CANONICAL_CAMERA_ID,
        ),
    ).rowcount
    if updated == 0:
        connection.execute(
            "INSERT INTO cameras "
            "(camera_id, model, width, height, params, prior_focal_length) "
            "VALUES (?, ?, ?, ?, ?, 1)",
            (
                CANONICAL_CAMERA_ID,
                PINHOLE_CAMERA_MODEL_ID,
                1920,
                1080,
                params,
            ),
        )
    connection.execute(
        "UPDATE images SET camera_id = ?", (CANONICAL_CAMERA_ID,)
    )
    connection.execute(
        "DELETE FROM cameras WHERE camera_id != ?", (CANONICAL_CAMERA_ID,)
    )

    connection.execute("DELETE FROM frame_data")
    connection.execute("DELETE FROM frames")
    connection.execute("DELETE FROM rigs")
    connection.execute(
        "INSERT INTO rigs (rig_id, ref_sensor_id, ref_sensor_type) VALUES (?, ?, ?)",
        (CANONICAL_CAMERA_ID, CANONICAL_CAMERA_ID, CAMERA_SENSOR_TYPE),
    )
    connection.executemany(
        "INSERT INTO frames (frame_id, rig_id) VALUES (?, ?)",
        ((image_id, CANONICAL_CAMERA_ID) for image_id in sorted(image_ids)),
    )
    connection.executemany(
        "INSERT INTO frame_data "
        "(frame_id, data_id, sensor_id, sensor_type) VALUES (?, ?, ?, ?)",
        (
            (
                image_id,
                image_id,
                CANONICAL_CAMERA_ID,
                CAMERA_SENSOR_TYPE,
            )
            for image_id in sorted(image_ids)
        ),
    )
    camera_rows = connection.execute(
        "SELECT camera_id, model, width, height, params, prior_focal_length "
        "FROM cameras"
    ).fetchall()
    if len(camera_rows) != 1:
        raise ValueError("local database must contain exactly one camera")
    camera_id, model, width, height, raw_params, prior = camera_rows[0]
    actual_params = np.frombuffer(raw_params, dtype="<f8")
    if (
        (camera_id, model, width, height, prior)
        != (CANONICAL_CAMERA_ID, PINHOLE_CAMERA_MODEL_ID, 1920, 1080, 1)
        or not np.array_equal(actual_params, FIXED_CAMERA_PARAMS)
    ):
        raise ValueError("local database fixed camera differs from Fuhe official69")
    expected_images = sorted(image_ids)
    image_camera_rows = connection.execute(
        "SELECT image_id, camera_id FROM images ORDER BY image_id"
    ).fetchall()
    expected_image_camera_rows = [
        (image_id, CANONICAL_CAMERA_ID) for image_id in expected_images
    ]
    if image_camera_rows != expected_image_camera_rows:
        raise ValueError("retained images do not all reference the canonical camera")
    if connection.execute("SELECT * FROM rigs").fetchall() != [
        (CANONICAL_CAMERA_ID, CANONICAL_CAMERA_ID, CAMERA_SENSOR_TYPE)
    ]:
        raise ValueError("local database rig graph is not canonical")
    frame_rows = connection.execute(
        "SELECT frame_id, rig_id FROM frames ORDER BY frame_id"
    ).fetchall()
    expected_frame_rows = [
        (image_id, CANONICAL_CAMERA_ID) for image_id in expected_images
    ]
    frame_data_rows = connection.execute(
        "SELECT frame_id, data_id, sensor_id, sensor_type "
        "FROM frame_data ORDER BY frame_id"
    ).fetchall()
    expected_frame_data_rows = [
        (
            image_id,
            image_id,
            CANONICAL_CAMERA_ID,
            CAMERA_SENSOR_TYPE,
        )
        for image_id in expected_images
    ]
    if frame_rows != expected_frame_rows or frame_data_rows != expected_frame_data_rows:
        raise ValueError("local database frame/sensor relations are not canonical")
    return {
        "cameras": len(camera_rows),
        "rigs": 1,
        "frames": len(frame_rows),
        "frame_data": len(frame_data_rows),
    }


def _filter_database(source: Path, output: Path, sequence: str) -> dict:
    if output.exists():
        output.unlink()
    shutil.copy2(source, output)
    connection = sqlite3.connect(output)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        pose_prior_columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(pose_priors)")
        }
        if pose_prior_columns and "image_id" not in pose_prior_columns:
            pose_prior_count = int(
                connection.execute("SELECT COUNT(*) FROM pose_priors").fetchone()[0]
            )
            if pose_prior_count:
                raise ValueError(
                    "cannot migrate non-empty COLMAP 4 pose_priors to legacy GLOMAP schema"
                )
            connection.execute("DROP TABLE pose_priors")
            connection.execute(
                "CREATE TABLE pose_priors ("
                "image_id INTEGER PRIMARY KEY NOT NULL, "
                "position BLOB, "
                "coordinate_system INTEGER NOT NULL, "
                "position_covariance BLOB, "
                "FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"
            )
        rows = connection.execute(
            "SELECT image_id, name FROM images WHERE name LIKE ? ORDER BY image_id",
            (f"{sequence}/%",),
        ).fetchall()
        keep = {int(row[0]) for row in rows}
        if len(keep) < 3:
            raise ValueError(f"{sequence}: fewer than three database images")
        for table in ("two_view_geometries", "matches"):
            pair_ids = [
                int(row[0])
                for row in connection.execute(f"SELECT pair_id FROM {table}")
                if not set(decode_image_pair_id(int(row[0]))) <= keep
            ]
            connection.executemany(
                f"DELETE FROM {table} WHERE pair_id = ?",
                ((pair_id,) for pair_id in pair_ids),
            )
        placeholders = ",".join("?" for _ in keep)
        parameters = tuple(sorted(keep))
        connection.execute(
            f"DELETE FROM keypoints WHERE image_id NOT IN ({placeholders})", parameters
        )
        connection.execute(
            f"DELETE FROM descriptors WHERE image_id NOT IN ({placeholders})",
            parameters,
        )
        connection.execute(
            f"DELETE FROM images WHERE image_id NOT IN ({placeholders})", parameters
        )
        camera_graph = _canonicalize_local_camera_graph(connection, keep)
        foreign_key_violations = connection.execute(
            "PRAGMA foreign_key_check"
        ).fetchall()
        if foreign_key_violations:
            raise ValueError(
                "canonical local camera graph violates foreign keys: "
                f"{foreign_key_violations[:3]}"
            )
        connection.commit()
        remaining_pairs = int(
            connection.execute("SELECT COUNT(*) FROM two_view_geometries").fetchone()[0]
        )
    finally:
        connection.close()
    return {
        "images": len(keep),
        "two_view_pairs": remaining_pairs,
        "legacy_pose_priors_schema": True,
        **camera_graph,
    }


def _find_model(output: Path) -> Path:
    candidates = [
        path.parent
        for path in output.rglob("images.bin")
        if (path.parent / "cameras.bin").is_file()
        and (path.parent / "points3D.bin").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(f"GLOMAP produced no model under {output}")
    return max(candidates, key=lambda path: (path / "images.bin").stat().st_size)


def _build_local_model(
    *,
    database: Path,
    image_root: Path,
    sequence: str,
    work_dir: Path,
    glomap_bin: Path,
    reuse: bool,
) -> tuple[Path, dict]:
    sequence_dir = work_dir / sequence
    database_out = sequence_dir / "database.db"
    mapper_out = sequence_dir / "model"
    if reuse:
        try:
            return _find_model(mapper_out), {"reused": True}
        except FileNotFoundError:
            pass
    if sequence_dir.exists():
        shutil.rmtree(sequence_dir)
    mapper_out.mkdir(parents=True)
    database_stats = _filter_database(database, database_out, sequence)
    command = [
        str(glomap_bin),
        "mapper",
        "--database_path",
        str(database_out),
        "--image_path",
        str(image_root),
        "--output_path",
        str(mapper_out),
        "--BundleAdjustment.optimize_intrinsics",
        "0",
        "--BundleAdjustment.optimize_principal_point",
        "0",
    ]
    log_path = sequence_dir / "glomap.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(
            f"{sequence}: GLOMAP exited {completed.returncode}; see {log_path}"
        )
    model = _find_model(mapper_out)
    return model, {
        "reused": False,
        "database": database_stats,
        "command": command,
        "log": str(log_path.resolve()),
    }


def _verified_matches(
    connection: sqlite3.Connection, image_id1: int, image_id2: int
) -> np.ndarray:
    row = connection.execute(
        "SELECT rows, cols, data FROM two_view_geometries WHERE pair_id = ?",
        (image_pair_id(image_id1, image_id2),),
    ).fetchone()
    if row is None or int(row[0]) == 0:
        return np.empty((0, 2), dtype=np.uint32)
    matches = np.frombuffer(row[2], dtype=np.uint32).reshape(int(row[0]), int(row[1]))
    return matches if image_id1 < image_id2 else matches[:, ::-1]


def _cross_map_correspondences(
    connection: sqlite3.Connection,
    source_rec,
    target_rec,
    pairs: list[tuple[str, str]],
) -> tuple[np.ndarray, np.ndarray, dict]:
    source_images = {image.name: image for image in source_rec.images.values() if image.has_pose}
    target_images = {image.name: image for image in target_rec.images.values() if image.has_pose}
    correspondences: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    contributing_pairs = 0
    raw_3d_matches = 0
    for source_name, target_name in pairs:
        source_image = source_images.get(source_name)
        target_image = target_images.get(target_name)
        if source_image is None or target_image is None:
            continue
        pair_contributed = False
        for source_index, target_index in _verified_matches(
            connection, int(source_image.image_id), int(target_image.image_id)
        ):
            source_point = source_image.points2D[int(source_index)]
            target_point = target_image.points2D[int(target_index)]
            if not source_point.has_point3D() or not target_point.has_point3D():
                continue
            source_id = int(source_point.point3D_id)
            target_id = int(target_point.point3D_id)
            correspondences[(source_id, target_id)] = (
                np.asarray(source_rec.points3D[source_id].xyz, dtype=np.float64),
                np.asarray(target_rec.points3D[target_id].xyz, dtype=np.float64),
            )
            raw_3d_matches += 1
            pair_contributed = True
        contributing_pairs += pair_contributed
    values = list(correspondences.values())
    if not values:
        return np.empty((0, 3)), np.empty((0, 3)), {
            "contributing_image_pairs": contributing_pairs,
            "raw_3d_matches": raw_3d_matches,
            "unique_3d_correspondences": 0,
        }
    return (
        np.asarray([value[0] for value in values]),
        np.asarray([value[1] for value in values]),
        {
            "contributing_image_pairs": contributing_pairs,
            "raw_3d_matches": raw_3d_matches,
            "unique_3d_correspondences": len(values),
        },
    )


def _main_locked() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--forced-pairs", type=Path, required=True)
    parser.add_argument("--forced-manifest", type=Path, required=True)
    parser.add_argument("--twoview", type=Path, required=True)
    parser.add_argument("--s4-gate", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--glomap-bin", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--final-model", type=Path)
    parser.add_argument(
        "--quarantine-edge",
        nargs=2,
        action="append",
        default=[],
        metavar=("SEQUENCE_A", "SEQUENCE_B"),
    )
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()

    gate_dir = (
        args.out.parent
        if args.out.parent.name == "gates"
        else args.out.parent / "gates"
    )
    predecessor_gate = gate_dir / "S5_fixed_intrinsics.json"
    read_fresh_gate_stage_metrics(
        predecessor_gate, expected_stage="S5_fixed_intrinsics"
    )
    s4 = read_fresh_gate_stage_metrics(
        args.s4_gate, expected_stage="S4_doppelgangers"
    )

    import pycolmap
    import torch

    robust_edges = [tuple(sorted(edge)) for edge in s4["robust_cross_direction_edges"]]
    sequences = sorted({sequence for edge in robust_edges for sequence in edge})
    models = {}
    build_evidence = {}
    for sequence in sequences:
        model, evidence = _build_local_model(
            database=args.database,
            image_root=args.image_root,
            sequence=sequence,
            work_dir=args.work_dir,
            glomap_bin=args.glomap_bin,
            reuse=args.reuse,
        )
        models[sequence] = pycolmap.Reconstruction(str(model))
        camera_evidence = fixed_camera_evidence(models[sequence])
        if not camera_evidence["ok"]:
            raise RuntimeError(
                f"{sequence}: independent model violates the fixed Fuhe camera"
            )
        build_evidence[sequence] = {
            **evidence,
            "model": str(model.resolve()),
            "registered": int(models[sequence].num_reg_images()),
            "points3D": int(models[sequence].num_points3D()),
            "fixed_camera": camera_evidence,
        }

    forced_manifest = json.loads(args.forced_manifest.read_text(encoding="utf-8"))
    directions = {
        **{name: "fwd" for name in forced_manifest["fwd"]},
        **{name: "rev" for name in forced_manifest["rev"]},
    }
    names = sorted(
        path.relative_to(args.image_root).as_posix()
        for path in args.image_root.glob("*/*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    positions = {}
    names_by_sequence: dict[str, list[str]] = defaultdict(list)
    for name in names:
        names_by_sequence[name.split("/", 1)[0]].append(name)
    for sequence_names in names_by_sequence.values():
        denominator = max(1, len(sequence_names) - 1)
        positions.update(
            {name: index / denominator for index, name in enumerate(sequence_names)}
        )
    forced_lines = load_forced_pairs(args.forced_pairs)
    twoview = torch.load(args.twoview, map_location="cpu", weights_only=False)
    accepted_records = accepted_pair_records_for_edges(
        names,
        twoview["pairs"],
        twoview["scores"],
        threshold=args.threshold,
        directions=directions,
        conditional_pairs=forced_lines,
        trusted_edges=set(robust_edges),
    )
    accepted = {
        edge: [(str(row["left"]), str(row["right"])) for row in rows]
        for edge, rows in accepted_records.items()
    }

    database_connection = sqlite3.connect(args.database)
    evidence = {}
    all_pass = True
    try:
        for edge in robust_edges:
            source_sequence, target_sequence = edge
            try:
                clusters = _two_separated_clusters(accepted[edge], positions)
                estimates = []
                cluster_evidence = []
                target_centers = np.asarray(
                    [
                        image.projection_center()
                        for image in models[target_sequence].images.values()
                        if image.has_pose
                    ]
                )
                target_span = robust_spatial_span(target_centers)
                max_error = 0.02 * target_span
                for cluster in clusters:
                    source_points, target_points, stats = _cross_map_correspondences(
                        database_connection,
                        models[source_sequence],
                        models[target_sequence],
                        cluster,
                    )
                    estimate = robust_sim3(
                        source_points, target_points, max_error=max_error
                    )
                    estimates.append(estimate)
                    cluster_evidence.append(stats)
                scale_delta = abs(math.log(estimates[0]["scale"] / estimates[1]["scale"]))
                rotation_delta = math.degrees(
                    (
                        Rotation.from_matrix(estimates[0]["rotation"]).inv()
                        * Rotation.from_matrix(estimates[1]["rotation"])
                    ).magnitude()
                )
                translation_delta = float(
                    np.linalg.norm(
                        estimates[0]["translation"] - estimates[1]["translation"]
                    )
                    / target_span
                )
                normalized_rmse = [estimate["rmse"] / target_span for estimate in estimates]
                support_ok = all(
                    estimate["inliers"] >= 30 and estimate["inlier_fraction"] >= 0.20
                    for estimate in estimates
                )
                passed = (
                    support_ok
                    and scale_delta <= 0.15
                    and rotation_delta <= 15.0
                    and translation_delta <= 0.08
                    and max(normalized_rmse) <= 0.05
                )
                evidence["|".join(edge)] = {
                    "status": "PASS" if passed else "FAIL",
                    "cluster_sizes": [len(cluster) for cluster in clusters],
                    "route_cluster_count": 2,
                    "route_cluster_minimum_normalized_separation": 0.25,
                    "cluster_correspondence_evidence": cluster_evidence,
                    "estimates": [
                        {
                            **_json_sim3(estimate),
                            "inliers": estimate["inliers"],
                            "correspondences": estimate["correspondences"],
                            "inlier_fraction": estimate["inlier_fraction"],
                        }
                        for estimate in estimates
                    ],
                    "target_robust_span": target_span,
                    "ransac_max_error": max_error,
                    "scale_log_delta": scale_delta,
                    "rotation_delta_deg": rotation_delta,
                    "translation_delta_over_span": translation_delta,
                    "normalized_rmse": normalized_rmse,
                    "support_ok": support_ok,
                    "accepted_pair_sources": {
                        source: sum(
                            row["source"] == source
                            for row in accepted_records.get(edge, [])
                        )
                        for source in ("natural", "conditional")
                    },
                    "accepted_pairs": accepted_records.get(edge, []),
                }
            except (ValueError, KeyError, IndexError, np.linalg.LinAlgError) as error:
                passed = False
                evidence["|".join(edge)] = {"status": "FAIL", "error": str(error)}
            all_pass &= passed
    finally:
        database_connection.close()
    quarantine_edges = {tuple(sorted(edge)) for edge in args.quarantine_edge}
    if quarantine_edges and args.final_model is None:
        raise SystemExit("--final-model is required with --quarantine-edge")
    quarantine_counts = {}
    if quarantine_edges:
        final_rec = pycolmap.Reconstruction(str(args.final_model))
        final_sequences = {
            int(image.image_id): str(image.name).split("/", 1)[0]
            for image in final_rec.images.values()
        }
        for edge in sorted(quarantine_edges):
            count = len(
                point_ids_spanning_sequence_edges(
                    final_rec.points3D, final_sequences, {edge}
                )
            )
            quarantine_counts["|".join(edge)] = count
    all_pass, trusted_edges = resolve_quarantined_edges(
        evidence, quarantine_counts
    )
    result = {
        "stage": "S5.7_independent_sim3",
        "status": "PASS" if all_pass and bool(robust_edges) else "FAIL",
        "method": (
            "independent per-sequence GLOMAP plus verified 3D-3D bridge RANSAC; "
            "failed redundant edges require zero shared final-map tracks"
        ),
        "local_models": build_evidence,
        "edges": evidence,
        "trusted_independent_edges": trusted_edges,
        "minimum_trusted_cross_direction_edges": 2,
        "minimum_inliers_per_cluster": MIN_CLUSTER_INLIERS,
        "quarantined_edge_remaining_points": quarantine_counts,
        "database_reuse_contract": {
            "source_database": str(args.database.resolve()),
            "matching_recomputed": False,
            "local_model_scope": "per-sequence only",
            "second_full_glomap": False,
            "pgo": False,
        },
        "conditional_pair_provenance": {
            "role": "source classification only; never an acceptance filter",
            "pair_file": {
                "path": str(args.forced_pairs.resolve()),
                "sha256": sha256(args.forced_pairs),
            },
            "manifest": {
                "path": str(args.forced_manifest.resolve()),
                "sha256": sha256(args.forced_manifest),
            },
        },
        "second_full_glomap": False,
        "pgo": False,
    }
    inputs = {
        "database": args.database,
        "image_root": args.image_root,
        "conditional_pairs": args.forced_pairs,
        "conditional_pair_manifest": args.forced_manifest,
        "two_view_scores": args.twoview,
        "S4_gate": args.s4_gate,
        **{
            f"independent_model/{sequence}": evidence["model"]
            for sequence, evidence in sorted(build_evidence.items())
        },
    }
    if args.final_model is not None:
        inputs["final_model"] = args.final_model
    gate = Gate(
        "S5_7_independent_sim3",
        required_check_ids("S5_7_independent_sim3"),
        script_path=__file__,
        source_files=[
            Path(__file__).with_name("audit_dg_graph.py"),
            Path(__file__).with_name("audit_map_geometry.py"),
            Path(__file__).with_name("finalize_edm_model.py"),
            Path(__file__).with_name("resource_guard.py"),
            Path(__file__).with_name("ts_common.py"),
        ],
        input_artifacts=inputs,
    )
    gate.record_predecessor_gate(
        "S5_fixed_intrinsics",
        predecessor_gate,
        expected_stage="S5_fixed_intrinsics",
    )
    gate.check(
        "G5.7",
        result["status"] == "PASS",
        "independent per-sequence Sim3 has two trusted cross-direction routes",
        trusted_independent_edges=trusted_edges,
        minimum_trusted_cross_direction_edges=2,
        edge_evidence=evidence,
    )
    payload = gate.write(
        args.out.parent,
        output_path=args.out,
        stage_metrics=result,
    )
    print(json.dumps(payload, indent=2), flush=True)


def main() -> None:
    arguments = sys.argv[1:]
    if "--help" in arguments or "-h" in arguments:
        _main_locked()
        return
    disk_path = required_cli_path(arguments, "--work-dir")
    run_global_heavy_job(disk_path, lambda _evidence: _main_locked())


if __name__ == "__main__":
    main()
