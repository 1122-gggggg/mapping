from __future__ import annotations

import importlib.util
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


MODULE_PATH = Path(__file__).with_name("build_localizable_map_core.py")
SPEC = importlib.util.spec_from_file_location("build_localizable_map_core", MODULE_PATH)
assert SPEC and SPEC.loader
core = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(core)


class MotionClassificationTests(unittest.TestCase):
    def cfg(self) -> SimpleNamespace:
        return SimpleNamespace(
            motion_min_tracks=20,
            motion_min_flow_px=3.0,
            motion_rotation_h_over_f=0.85,
            motion_rotation_min_inliers=20,
        )

    def test_classifies_hover_pure_rotation_and_parallax(self) -> None:
        hover = {"tracks": 40, "median_flow_px": 1.0, "h_inliers": 35, "h_over_f": 1.0}
        rotation = {"tracks": 80, "median_flow_px": 20.0, "h_inliers": 70, "h_over_f": 0.92}
        parallax = {"tracks": 80, "median_flow_px": 20.0, "h_inliers": 40, "h_over_f": 0.50}

        self.assertEqual(core.classify_motion_metrics(hover, self.cfg())[0], "hover")
        self.assertEqual(core.classify_motion_metrics(rotation, self.cfg())[0], "pure_rotation")
        self.assertEqual(core.classify_motion_metrics(parallax, self.cfg())[0], "parallax")

    def test_filters_non_parallax_pairs_when_requested(self) -> None:
        pairs = [
            ("seq/000001.jpg", "seq/000002.jpg"),
            ("seq/000002.jpg", "seq/000003.jpg"),
            ("seq/000001.jpg", "seq/000004.jpg"),
        ]
        roles = {
            "seq/000001.jpg": {"motion_class": "parallax"},
            "seq/000002.jpg": {"motion_class": "pure_rotation"},
            "seq/000003.jpg": {"motion_class": "parallax"},
            "seq/000004.jpg": {"motion_class": "hover"},
        }

        kept, stats = core.filter_pairs_by_motion_roles(pairs, roles, exclude_non_parallax=True)

        self.assertEqual(kept, [])
        self.assertEqual(stats["removed_non_parallax_pairs"], 3)
        self.assertEqual(stats["input_pairs"], 3)

    def test_generates_roma_metric3d_large_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = SimpleNamespace(
                work_dir=tmp,
                mpsfm_config_yaml="",
                mpsfm_conf="roma_m3dv2-large",
            )

            conf_arg = core.resolve_mpsfm_conf_arg(cfg)
            conf_path = Path(conf_arg + ".yaml")

            self.assertTrue(conf_path.exists())
            text = conf_path.read_text()
            self.assertIn("defaults/m3dv2-large", text)
            self.assertIn("matcher: roma_outdoor", text)
            self.assertIn("matches_mode: sparse+dense", text)


class PaperMethodIntegrationTests(unittest.TestCase):
    def cfg(self) -> SimpleNamespace:
        return SimpleNamespace(
            dms_min_sampled_matches=24,
            dms_min_inliers=24,
            dms_min_inlier_ratio=0.08,
            dms_rotation_h_over_f=1.5,
        )

    def test_planar_consistency_rescues_homography_dominant_cross_direction_edge(self) -> None:
        metrics = {
            "sampled_matches": 100,
            "f_inliers": 10,
            "h_inliers": 80,
            "f_ratio": 0.10,
            "h_ratio": 0.80,
            "planar_consistent": True,
        }
        keep, reason = core.keep_verified_pair(
            metrics, {"cross_direction": True, "same_video": False}, self.cfg()
        )
        self.assertTrue(keep)
        self.assertEqual(reason, "cross_direction_planar_consistent")

    def test_inconsistent_homography_does_not_rescue_cross_direction_edge(self) -> None:
        metrics = {
            "sampled_matches": 100,
            "f_inliers": 10,
            "h_inliers": 80,
            "f_ratio": 0.10,
            "h_ratio": 0.80,
            "planar_consistent": False,
        }
        keep, reason = core.keep_verified_pair(
            metrics, {"cross_direction": True, "same_video": False}, self.cfg()
        )
        self.assertFalse(keep)
        self.assertEqual(reason, "cross_direction_rotation_like")

    def test_view_pruning_keeps_only_deterministic_largest_component(self) -> None:
        pairs = [("a", "b"), ("b", "c"), ("x", "y")]
        kept, graph = core.prune_outlier_pair_components(
            ["a", "b", "c", "x", "y", "isolated"], pairs
        )
        self.assertEqual(kept, [("a", "b"), ("b", "c")])
        self.assertEqual(graph["component_sizes_before_pruning"], [3, 2, 1])
        self.assertEqual(graph["pruned_views"], ["isolated", "x", "y"])
        self.assertEqual(graph["largest_component_ratio"], 0.5)

    def test_colmap_global_and_graph_filter_are_runtime_defaults(self) -> None:
        self.assertEqual(core.RUNTIME_DEFAULTS["backend"], "colmap_global")
        self.assertEqual(core.RUNTIME_DEFAULTS["pair_verification"], "dms")
        self.assertNotIn("lfoe_mode", core.RUNTIME_DEFAULTS)
        self.assertFalse(core.RUNTIME_DEFAULTS["allow_unlicensed_lfoe"])

    def test_colmap_global_command_fixes_all_intrinsic_groups(self) -> None:
        cfg = SimpleNamespace(
            work_dir="/tmp/colmap-global-command-test",
            skip_bundle_adjustment=False,
            skip_retriangulation=True,
            optimize_intrinsics=0,
            optimize_principal_point=0,
            min_num_view_per_track=3,
            min_triangulation_angle=1.0,
            max_num_tracks=600000,
        )

        command = core.build_global_mapper_cmd(
            cfg, "colmap_global", "/bin/colmap", Path("/tmp/model")
        )

        self.assertEqual(command[:2], ["/bin/colmap", "global_mapper"])
        self.assertIn("--GlobalMapper.ba_refine_focal_length", command)
        self.assertIn("--GlobalMapper.ba_refine_principal_point", command)
        self.assertIn("--GlobalMapper.ba_refine_extra_params", command)
        self.assertNotIn("--TrackEstablishment.max_num_tracks", command)

    def test_lfoe_requires_explicit_deployment_authorization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = SimpleNamespace(
                work_dir=tmp,
                backend="lfoe",
                allow_unlicensed_lfoe=False,
                overwrite=False,
            )

            with self.assertRaisesRegex(SystemExit, "no upstream license grant"):
                core.stage_glomap(cfg)


class PairGraphPerformanceTests(unittest.TestCase):
    def test_default_directional_mode_skips_full_similarity_matrix(self) -> None:
        class NoMatmulArray(np.ndarray):
            def __matmul__(self, other):
                raise AssertionError("directional mode built an unused full similarity matrix")

        class FakeH5File:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def create_group(self, _name):
                return self

            def create_dataset(self, _name, **_kwargs):
                return None

        with tempfile.TemporaryDirectory() as tmp:
            cfg = SimpleNamespace(
                work_dir=tmp,
                template_repo=tmp,
                device="cpu",
                pair_graph_mode="directional",
                same_direction_topk=0,
                num_matched=1,
                seq_window=1,
                cross_topk=0,
                cross_grid=0,
                agg_pair_degree_cap=0,
                agg_intra_degree_cap=0,
                agg_cross_direction_degree_cap=0,
                direction_overrides_json="",
                use_rotation_bridges=False,
            )
            paths = core.cfg_paths(cfg)
            sequence = paths.images / "seq"
            sequence.mkdir(parents=True)
            (sequence / "000001.jpg").touch()
            (sequence / "000002.jpg").touch()
            descriptors = np.ones((2, 3), dtype=np.float32).view(NoMatmulArray)
            megaloc = SimpleNamespace(extract=lambda *_args, **_kwargs: descriptors)
            torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
            h5py = SimpleNamespace(File=lambda *_args, **_kwargs: FakeH5File())

            with mock.patch.dict(
                "sys.modules",
                {"h5py": h5py, "megaloc_lib": megaloc, "torch": torch},
            ):
                core.stage_pairs(cfg)

            self.assertEqual(
                core.read_pairs(paths.pairs),
                [("seq/000001.jpg", "seq/000002.jpg")],
            )

    def test_directional_cross_topk_uses_block_similarity_without_full_matrix(self) -> None:
        class BlockOnlyMatmulArray(np.ndarray):
            full_matrix_built = False

            def __matmul__(self, other):
                other = np.asarray(other)
                if self.shape == (4, 3) and other.shape == (3, 4):
                    type(self).full_matrix_built = True
                    raise AssertionError("directional mode built a full descriptor matrix")
                return np.ndarray.__matmul__(self, other)

        class FakeH5File:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def create_group(self, _name):
                return self

            def create_dataset(self, _name, **_kwargs):
                return None

        with tempfile.TemporaryDirectory() as tmp:
            cfg = SimpleNamespace(
                work_dir=tmp,
                template_repo=tmp,
                device="cpu",
                pair_graph_mode="directional",
                same_direction_topk=0,
                num_matched=1,
                seq_window=1,
                cross_topk=1,
                cross_grid=0,
                agg_pair_degree_cap=0,
                agg_intra_degree_cap=0,
                agg_cross_direction_degree_cap=0,
                direction_overrides_json="",
                use_rotation_bridges=False,
            )
            paths = core.cfg_paths(cfg)
            for folder in ("forward", "reverse"):
                sequence = paths.images / folder
                sequence.mkdir(parents=True)
                (sequence / "000001.jpg").touch()
                (sequence / "000002.jpg").touch()
            descriptors = np.array(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
                dtype=np.float32,
            ).view(BlockOnlyMatmulArray)
            megaloc = SimpleNamespace(extract=lambda *_args, **_kwargs: descriptors)
            torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
            h5py = SimpleNamespace(File=lambda *_args, **_kwargs: FakeH5File())

            with mock.patch.dict(
                "sys.modules",
                {"h5py": h5py, "megaloc_lib": megaloc, "torch": torch},
            ):
                core.stage_pairs(cfg)

            self.assertEqual(
                core.read_pairs(paths.pairs),
                [
                    ("forward/000001.jpg", "forward/000002.jpg"),
                    ("reverse/000001.jpg", "reverse/000002.jpg"),
                    ("forward/000001.jpg", "reverse/000002.jpg"),
                    ("forward/000002.jpg", "reverse/000001.jpg"),
                ],
            )
            self.assertFalse(BlockOnlyMatmulArray.full_matrix_built)


class StrictGateTests(unittest.TestCase):
    def cfg(self, tmp: str) -> SimpleNamespace:
        return SimpleNamespace(
            work_dir=tmp,
            strict_gates=True,
            strict_profile="football_field_1920",
            disable_stage_gates=False,
            dry_run=False,
            gate_min_frames=2,
            gate_min_pairs=2,
            gate_min_registered_images=2,
            gate_min_registered_ratio=0.4,
        )

    def write_pairs(self, path: Path, count: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [f"seq/000{i:03d}.jpg seq/000{i + 1:03d}.jpg" for i in range(count)]
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    def test_strict_pairs_gate_rejects_disconnected_graph(self) -> None:
        import h5py

        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.cfg(tmp)
            p = core.cfg_paths(cfg)
            p.megaloc.mkdir(parents=True, exist_ok=True)
            self.write_pairs(p.pairs, 12)
            with h5py.File(p.global_desc, "w") as fd:
                for i in range(10):
                    fd.create_group(f"seq/000{i:03d}.jpg")
            core.write_json(p.pair_graph_diagnostics, {
                "total_frames": 10,
                "pairs": 12,
                "pair_kinds": {"temporal": 12},
                "relations": {"same_video": 12},
                "connected_components": 3,
                "largest_component": 4,
                "parallax_components_without_bridges": 3,
                "largest_parallax_component_without_bridges": 4,
            })

            with self.assertRaises(SystemExit):
                core.validate_stage_gate("pairs", cfg)

            gate = core.read_json(p.work / "gates" / "pairs.json")
            self.assertFalse(gate["ok"])
            self.assertTrue(any("largest_component_ratio" in r for r in gate["reasons"]))

    def test_strict_doppelgangers_gate_rejects_over_pruning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.cfg(tmp)
            p = core.cfg_paths(cfg)
            self.write_pairs(p.pairs_before_dg, 100)
            self.write_pairs(p.pairs, 10)

            with self.assertRaises(SystemExit):
                core.validate_stage_gate("doppelgangers", cfg)

            gate = core.read_json(p.work / "gates" / "doppelgangers.json")
            self.assertFalse(gate["ok"])
            self.assertTrue(any("doppelgangers_retention_ratio" in r for r in gate["reasons"]))

    def test_strict_glomap_gate_rejects_high_reprojection_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.cfg(tmp)
            p = core.cfg_paths(cfg)
            core.write_json(p.manifest, {"total_frames": 100})
            original = core.glomap_summary
            core.glomap_summary = lambda _path: {
                "exists": True,
                "required": {"cameras.bin": True, "images.bin": True, "points3D.bin": True},
                "registered_images": 95,
                "points3D": 10000,
                "mean_reprojection_error": 3.5,
            }
            try:
                with self.assertRaises(SystemExit):
                    core.validate_stage_gate("glomap", cfg)
            finally:
                core.glomap_summary = original

            gate = core.read_json(p.work / "gates" / "glomap.json")
            self.assertFalse(gate["ok"])
            self.assertTrue(any("mean_reprojection_error" in r for r in gate["reasons"]))

    def test_strict_glomap_gate_rejects_low_point_density(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.cfg(tmp)
            p = core.cfg_paths(cfg)
            core.write_json(p.manifest, {"total_frames": 100})
            original = core.glomap_summary
            core.glomap_summary = lambda _path: {
                "exists": True,
                "required": {"cameras.bin": True, "images.bin": True, "points3D.bin": True},
                "registered_images": 100,
                "points3D": 10000,
                "mean_reprojection_error": 1.0,
            }
            try:
                with self.assertRaises(SystemExit):
                    core.validate_stage_gate("glomap", cfg)
            finally:
                core.glomap_summary = original

            gate = core.read_json(p.work / "gates" / "glomap.json")
            self.assertFalse(gate["ok"])
            self.assertTrue(any("points_per_registered_image" in r for r in gate["reasons"]))

    def test_strict_color_gate_accepts_compact_binary_ply_with_full_vertex_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.cfg(tmp)
            p = core.cfg_paths(cfg)
            p.rgb_ply.parent.mkdir(parents=True, exist_ok=True)
            header = (
                "ply\n"
                "format binary_little_endian 1.0\n"
                "element vertex 100\n"
                "property float x\n"
                "property float y\n"
                "property float z\n"
                "property uchar red\n"
                "property uchar green\n"
                "property uchar blue\n"
                "end_header\n"
            ).encode("ascii")
            p.rgb_ply.write_bytes(header + (b"\0" * (140 * 1024)))
            original = core.glomap_summary
            core.glomap_summary = lambda _path: {"exists": True, "points3D": 100}
            try:
                core.validate_stage_gate("color", cfg)
            finally:
                core.glomap_summary = original

            gate = core.read_json(p.work / "gates" / "color.json")
            self.assertTrue(gate["ok"])
            self.assertEqual(gate["metrics"]["ply_vertices"], 100)

    def test_strict_report_gate_uses_current_gate_pass_over_historical_stage_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.cfg(tmp)
            p = core.cfg_paths(cfg)
            core.write_json(p.work / "gates" / "glomap.json", {"ok": True, "metrics": {}})
            core.write_json(p.stage_times, {
                "stages": [
                    {"stage": "glomap", "status": "failed"},
                    {"stage": "report", "status": "failed"},
                ]
            })
            p.report_json.parent.mkdir(parents=True, exist_ok=True)
            p.report_json.write_text("{}", encoding="utf-8")
            p.report_md.write_text("# report\n", encoding="utf-8")

            core.validate_stage_gate("report", cfg)

            gate = core.read_json(p.work / "gates" / "report.json")
            self.assertTrue(gate["ok"])
            self.assertEqual(gate["metrics"]["latest_failed_stages"], {})


class PycolmapDatabaseCompatTests(unittest.TestCase):
    def test_opens_database_with_path_constructor_api(self) -> None:
        class DirectDatabase:
            def __init__(self, path: str) -> None:
                self.path = path

        module = SimpleNamespace(Database=DirectDatabase)

        db = core.open_pycolmap_database(module, Path("/tmp/test.db"))

        self.assertIsInstance(db, DirectDatabase)
        self.assertEqual(db.path, "/tmp/test.db")

    def test_opens_database_with_open_method_api(self) -> None:
        class OpenDatabase:
            def __init__(self, *args: str) -> None:
                if args:
                    raise TypeError("path constructor unsupported")
                self.opened_path = ""

            @classmethod
            def open(cls, path: str) -> "OpenDatabase":
                db = cls()
                db.opened_path = path
                return db

        module = SimpleNamespace(Database=OpenDatabase)

        db = core.open_pycolmap_database(module, Path("/tmp/test.db"))

        self.assertIsInstance(db, OpenDatabase)
        self.assertEqual(db.opened_path, "/tmp/test.db")


class FuheGluemapBuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        module_path = Path(__file__).with_name("run_fuhe_gluemap_build.py")
        spec = importlib.util.spec_from_file_location("run_fuhe_gluemap_build", module_path)
        assert spec and spec.loader
        cls.fuhe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.fuhe)

    def test_intrinsics_scaling_keeps_distortion_unscaled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "intrinsics.json"
            path.write_text(
                json.dumps(
                    {
                        "image_width": 1280,
                        "image_height": 720,
                        "K": [[960.0, 0.0, 670.0], [0.0, 958.0, 358.0], [0.0, 0.0, 1.0]],
                        "dist": [-0.01, 0.2, -0.003, 0.004, -0.1],
                    }
                ),
                encoding="utf-8",
            )

            scaled_k, dist, map_k, meta = self.fuhe.parse_intrinsics(
                path,
                1920,
                1080,
                variant="current_undistort",
                official_video_hfov_deg=69.0,
            )

            self.assertAlmostEqual(scaled_k[0, 0], 1440.0)
            self.assertAlmostEqual(scaled_k[1, 1], 1437.0)
            self.assertAlmostEqual(scaled_k[0, 2], 1005.0)
            self.assertAlmostEqual(scaled_k[1, 2], 537.0)
            self.assertEqual(dist.tolist(), [-0.01, 0.2, -0.003, 0.004, -0.1, 0.0, 0.0, 0.0])
            self.assertTrue(meta["distortion_coefficients_not_scaled"])
            self.assertEqual(meta["source_model"], "FULL_OPENCV")
            self.assertEqual(meta["scaled_K"], scaled_k.tolist())
            self.assertEqual(meta["map_K"], map_k.tolist())
            self.assertEqual(meta["undistort_alpha"], 0.0)

    def test_pair_graph_metrics_reports_connected_cross_sequence_graph(self) -> None:
        names = ["A/000001.jpg", "A/000002.jpg", "B/000001.jpg", "B/000002.jpg"]
        pairs = [
            ("A/000001.jpg", "A/000002.jpg"),
            ("A/000002.jpg", "B/000001.jpg"),
            ("B/000001.jpg", "B/000002.jpg"),
            ("A/000001.jpg", "B/000002.jpg"),
        ]

        metrics = self.fuhe.pair_graph_metrics(names, pairs)

        self.assertEqual(metrics["connected_components"], 1)
        self.assertEqual(metrics["isolated_images"], 0)
        self.assertEqual(metrics["cross_sequence_pairs"], 2)
        self.assertEqual(metrics["largest_component"], 4)

    def test_rotation_bridge_keep_count_balances_selection_and_parallax_gates(self) -> None:
        metrics = self.fuhe.rotation_bridge_keep_count(
            triangulation_count=422,
            bridge_count=227,
            total_input_images=903,
            max_bridge_ratio=0.29,
            min_parallax_ratio=0.70,
            min_selected_ratio=0.65,
        )

        self.assertTrue(metrics["feasible"])
        self.assertEqual(metrics["target_bridge_count"], 172)
        self.assertGreaterEqual(422 / (422 + metrics["target_bridge_count"]), 0.70)
        self.assertGreaterEqual((422 + metrics["target_bridge_count"]) / 903, 0.65)

    def test_downsample_rotation_bridges_moves_excess_bridge_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            images = run_dir / "images"
            for name in [
                "A/000001.jpg",
                "A/000002.jpg",
                "A/000003.jpg",
                "A/000004.jpg",
                "B/000001.jpg",
                "B/000002.jpg",
                "B/000003.jpg",
            ]:
                path = images / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"x")
            roles = {
                "A/000001.jpg": {"motion_class": "seed", "motion_role": "triangulation"},
                "A/000002.jpg": {"motion_class": "parallax", "motion_role": "triangulation"},
                "A/000003.jpg": {"motion_class": "pure_rotation", "motion_role": "bridge_only"},
                "A/000004.jpg": {"motion_class": "pure_rotation", "motion_role": "bridge_only"},
                "B/000001.jpg": {"motion_class": "seed", "motion_role": "triangulation"},
                "B/000002.jpg": {"motion_class": "pure_rotation", "motion_role": "bridge_only"},
                "B/000003.jpg": {"motion_class": "pure_rotation", "motion_role": "bridge_only"},
            }
            self.fuhe.write_json(
                run_dir / "motion_quality.json",
                {
                    "sequences": {
                        "A": {
                            "total_before": 4,
                            "kept": 4,
                            "rejected": 0,
                            "records": [
                                {"frame": "000002.jpg", "motion_class": "parallax", "kept": True},
                                {"frame": "000003.jpg", "motion_class": "pure_rotation", "kept": True},
                                {"frame": "000004.jpg", "motion_class": "pure_rotation", "kept": True},
                            ],
                        },
                        "B": {
                            "total_before": 3,
                            "kept": 3,
                            "rejected": 0,
                            "records": [
                                {"frame": "000002.jpg", "motion_class": "pure_rotation", "kept": True},
                                {"frame": "000003.jpg", "motion_class": "pure_rotation", "kept": True},
                            ],
                        },
                    }
                },
            )

            summary = self.fuhe.downsample_rotation_bridges(
                run_dir,
                images,
                roles,
                total_input_images=7,
                max_bridge_ratio=0.40,
                min_parallax_ratio=0.60,
                min_selected_ratio=0.0,
            )

            self.assertEqual(summary["kept_bridge_count"], 2)
            self.assertEqual(summary["removed_bridge_count"], 2)
            self.assertEqual(len(list(images.glob("*/*.jpg"))), 5)
            report = self.fuhe.read_json(run_dir / "motion_quality.json")
            self.assertEqual(report["post_bridge_downsample"]["removed_bridge_count"], 2)


class ColmapSqliteSummaryTests(unittest.TestCase):
    def test_reads_camera_models_and_keypoint_coverage_without_pycolmap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "database.db"
            con = sqlite3.connect(db_path)
            try:
                cur = con.cursor()
                cur.execute(
                    "create table cameras (camera_id integer primary key, model integer not null, "
                    "width integer not null, height integer not null, params blob, prior_focal_length integer not null)"
                )
                cur.execute(
                    "create table images (image_id integer primary key, name text not null unique, camera_id integer not null)"
                )
                cur.execute(
                    "create table keypoints (image_id integer primary key, rows integer not null, cols integer not null, data blob)"
                )
                cur.execute("insert into cameras values (1, 6, 1920, 1080, X'', 1)")
                cur.execute("insert into images values (1, 'seq/000001.jpg', 1)")
                cur.execute("insert into images values (2, 'seq/000002.jpg', 1)")
                cur.execute("insert into keypoints values (1, 20000, 2, X'')")
                con.commit()
            finally:
                con.close()

            summary = core.colmap_db_summary_sqlite(db_path)

            self.assertEqual(summary["db_images"], 2)
            self.assertEqual(summary["camera_models"], ["FULL_OPENCV"])
            self.assertEqual(summary["camera_resolutions"][0]["width"], 1920)
            self.assertEqual(summary["images_without_keypoints"], ["seq/000002.jpg"])


if __name__ == "__main__":
    unittest.main()
