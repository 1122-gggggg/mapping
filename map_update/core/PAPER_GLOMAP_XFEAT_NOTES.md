# Paper / GLOMAP / XFeat Notes

Date: 2026-07-02

This note records research ideas and tunable parameters for the incremental
map-update pipeline. Current deployment direction is:

- global retrieval: MegaLoc
- local features: XFeat
- matcher: XFeat LighterGlue
- geometry: PnP, local submap, Sim3 merge, optional later BA

## Papers: Useful Takeaways

| Paper | Main idea | Useful for this pipeline | Action |
|---|---|---|---|
| Map Management for Efficient Long-Term Visual Localization in Outdoor Environments, 2018 | First localize new data against the current map. If it localizes poorly, add richer map data; if it localizes well, update observation/co-observation statistics and keep the map bounded with summarization. | Matches our route logic: high register rate should usually become QA / observation stats, not new geometry. | Add per-session observation statistics to the bundle: which old points/keyframes were repeatedly useful under each new video/domain. |
| Long-term map maintenance pipeline for autonomous vehicles, 2020 | Continuously remove transient/outdated features and promote new candidate features based on repeated geometric evidence. | Useful for changed-region handling. A wall/color/object change should reduce old point health and promote new points only after repeated support. | Add point health: `seen_ok`, `seen_failed`, `last_seen_session`, `support_area`, `domain`. Do not delete after one failure. |
| Long-Term Visual Map Sparsification with Heterogeneous GNN, 2022 | Model SfM as a heterogeneous graph and score stable / useful 3D points for long-term localization; selects stable structures and avoids change-prone points. | Useful for bundle compaction and old-point pruning, but GNN is not needed immediately. | Start with a non-learning score: visibility count, session count, PnP inlier reuse, broad image support, geometry parallax, semantic/static class if available. |
| ORBSLAM-Atlas, 2019 | Spawn submaps when tracking is lost; merge maps when overlap returns; avoid bad pose estimates caused by poor geometric conditioning. | Supports our `submap + Sim3` design and connector frames. | Improve bridge scoring: require enough bridges, broad inlier support, good pose conditioning, and low Sim3 residual, not only raw inlier count. |
| Multi-Session SLAM with Differentiable Wide-Baseline Pose Optimization, 2024 | Use retrieval to find co-visible frames, estimate wide-baseline relative poses in mini-batches, and choose the pair with the best inlier ratio; periodically run global optimization after joining trajectories. | Not a drop-in replacement for our GLOMAP pipeline, but useful when XFeat/LighterGlue bridge search is weak. | Add a bridge-search mode that tries more candidate pairs and ranks by inlier ratio / support area. Consider optical-flow pose only as a later fallback. |

## Immediate Pipeline Additions

1. Treat high-overlap videos as observation sessions.
   - If `register_rate > 0.95`, do not add geometry by default.
   - Still record which base keyframes/points localized well for this new domain.

2. Add old-point health for changed-region detection.
   - For each localized frame, project or match old anchors.
   - Points repeatedly supported across sessions get higher health.
   - Points repeatedly expected but unsupported in a local tile get lower health.
   - Only invalidate/replace after multi-frame and preferably multi-session evidence.

3. Add bridge-quality gates for method 3.
   - Minimum bridges.
   - Minimum geometry bridges, unless connectors provide strong pose support.
   - Median Sim3 residual.
   - Inlier spatial support area.
   - Pose conditioning / parallax score.

4. Add map sparsification after validation, not before.
   - First generate a full updated bundle.
   - Validate held-out localization.
   - Then compact with point/keyframe scores while preserving K-cover style coverage.

## GLOMAP Tunable Parameters

The installed GLOMAP mapper exposes these groups:

### Pipeline Control

| Parameter | Default | Meaning |
|---|---:|---|
| `--constraint_type` | `ONLY_POINTS` | Global positioning constraint type: `ONLY_POINTS`, `ONLY_CAMERAS`, `POINTS_AND_CAMERAS_BALANCED`, `POINTS_AND_CAMERAS`. |
| `--ba_iteration_num` | `3` | Number of BA / retriangulation cycles. |
| `--retriangulation_iteration_num` | `1` | Number of retriangulation cycles. |
| `--skip_preprocessing` | `0` | Skip preprocessing. |
| `--skip_view_graph_calibration` | `0` | Skip view-graph calibration. |
| `--skip_relative_pose_estimation` | `0` | Skip relative pose estimation. |
| `--skip_rotation_averaging` | `0` | Skip global rotation averaging. |
| `--skip_global_positioning` | `0` | Skip global positioning. |
| `--skip_bundle_adjustment` | `0` | Skip bundle adjustment. |
| `--skip_retriangulation` | `0` | Skip retriangulation. |
| `--skip_pruning` | `1` | Skip pruning. |

### View Graph / Relative Pose

| Parameter | Default | Meaning |
|---|---:|---|
| `--ViewGraphCalib.thres_lower_ratio` | `0.1` | Lower camera-ratio threshold for view-graph calibration. |
| `--ViewGraphCalib.thres_higher_ratio` | `10` | Upper camera-ratio threshold for view-graph calibration. |
| `--ViewGraphCalib.thres_two_view_error` | `2` | Two-view error threshold. |
| `--RelPoseEstimation.max_epipolar_error` | `1` | Relative pose epipolar error threshold. |

### Track Establishment

| Parameter | Default | Meaning |
|---|---:|---|
| `--TrackEstablishment.min_num_tracks_per_view` | `-1` | Minimum tracks per image/view; `-1` lets GLOMAP choose. |
| `--TrackEstablishment.min_num_view_per_track` | `3` | Minimum image observations for a 3D track. |
| `--TrackEstablishment.max_num_view_per_track` | `100` | Maximum image observations per track. |
| `--TrackEstablishment.max_num_tracks` | `10000000` | Global track cap. Lower this for memory; raise it for dense maps if memory allows. |

### Global Positioning

| Parameter | Default | Meaning |
|---|---:|---|
| `--GlobalPositioning.use_gpu` | `1` | Use GPU. |
| `--GlobalPositioning.gpu_index` | `-1` | GPU index; `-1` auto. |
| `--GlobalPositioning.optimize_positions` | `1` | Optimize camera centers. |
| `--GlobalPositioning.optimize_points` | `1` | Optimize 3D points. |
| `--GlobalPositioning.optimize_scales` | `1` | Optimize scale variables. |
| `--GlobalPositioning.thres_loss_function` | `0.1` | Robust loss threshold. |
| `--GlobalPositioning.max_num_iterations` | `100` | Max iterations. |

### Bundle Adjustment

| Parameter | Default | Meaning |
|---|---:|---|
| `--BundleAdjustment.use_gpu` | `1` | Use GPU BA. |
| `--BundleAdjustment.gpu_index` | `-1` | GPU index; `-1` auto. |
| `--BundleAdjustment.optimize_rig_poses` | `0` | Optimize rig poses. |
| `--BundleAdjustment.optimize_rotations` | `1` | Optimize camera rotations. |
| `--BundleAdjustment.optimize_translation` | `1` | Optimize camera translations. |
| `--BundleAdjustment.optimize_intrinsics` | `1` | Optimize camera intrinsics. For fixed ANAFI calibration, set this to `0`. |
| `--BundleAdjustment.optimize_principal_point` | `0` | Optimize principal point. Keep `0` for fixed intrinsics. |
| `--BundleAdjustment.optimize_points` | `1` | Optimize 3D points. |
| `--BundleAdjustment.thres_loss_function` | `1` | Robust loss threshold. |
| `--BundleAdjustment.max_num_iterations` | `200` | Max iterations. |

### Triangulation / Filtering Thresholds

| Parameter | Default | Meaning |
|---|---:|---|
| `--Triangulation.complete_max_reproj_error` | `15` | Completion reprojection threshold. |
| `--Triangulation.merge_max_reproj_error` | `15` | Track merge reprojection threshold. |
| `--Triangulation.min_angle` | `1` | Minimum triangulation angle. |
| `--Triangulation.min_num_matches` | `15` | Minimum matches for triangulation. |
| `--Thresholds.max_angle_error` | `1` | Angular error threshold. |
| `--Thresholds.max_reprojection_error` | `0.01` | Reprojection threshold used by GLOMAP filtering. |
| `--Thresholds.min_triangulation_angle` | `1` | Minimum triangulation angle threshold. |
| `--Thresholds.max_epipolar_error_E` | `1` | Essential-matrix epipolar threshold. |
| `--Thresholds.max_epipolar_error_F` | `4` | Fundamental-matrix epipolar threshold. |
| `--Thresholds.max_epipolar_error_H` | `4` | Homography epipolar threshold. |
| `--Thresholds.min_inlier_num` | `30` | Minimum two-view inliers. |
| `--Thresholds.min_inlier_ratio` | `0.25` | Minimum inlier ratio. |
| `--Thresholds.max_rotation_error` | `10` | Maximum rotation error. |

### Practical GLOMAP Presets

Conservative final build with fixed ANAFI intrinsics:

```bash
glomap mapper \
  --database_path DB \
  --image_path IMAGES \
  --output_path OUT \
  --BundleAdjustment.optimize_intrinsics 0 \
  --BundleAdjustment.optimize_principal_point 0 \
  --TrackEstablishment.max_num_tracks 600000
```

Stricter view graph when false matches / repeated structures create splinters:

```bash
glomap mapper \
  --database_path DB \
  --image_path IMAGES \
  --output_path OUT \
  --Thresholds.min_inlier_num 50 \
  --Thresholds.min_inlier_ratio 0.35 \
  --RelPoseEstimation.max_epipolar_error 0.75 \
  --Triangulation.min_angle 2
```

Fast diagnostic build:

```bash
glomap mapper \
  --database_path DB \
  --image_path IMAGES \
  --output_path OUT \
  --skip_retriangulation 1 \
  --TrackEstablishment.max_num_tracks 600000
```

## XFeat / LighterGlue Tunable Parameters

Current source:

- XFeat torch hub entry: `hubconf.py`
- extractor: `modules/xfeat.py`
- LighterGlue wrapper: `modules/lighterglue.py`

### XFeat

| Parameter | Default | Current pipeline name | Meaning |
|---|---:|---|---|
| `top_k` | `4096` | `--qk` | Maximum keypoints per image. Higher improves difficult matching but increases memory/time. |
| `detection_threshold` | `0.05` | not yet exposed | NMS score threshold. Lower gives more weak keypoints; higher keeps only stronger points. |
| NMS `kernel_size` | `5` | not exposed | Local-max suppression radius; hard-coded in `NMS(..., kernel_size=5)`. |
| `detectAndComputeDense(top_k, multiscale=True)` | `multiscale=True` | not used | Dense/coarse mode for XFeat-Star style matching. |
| `extract_dualscale(s1=0.6, s2=1.3)` | `0.6 / 1.3` | not used | Two-scale dense extraction parameters. |
| `match_xfeat(..., min_cossim=-1)` | `-1` | not used in update | Fast mutual-nearest-neighbor matcher without LighterGlue. |
| `match_xfeat_star(..., top_k)` | caller value | not used | Semi-dense XFeat-Star matching path. |

### LighterGlue

| Parameter | Default | Current pipeline name | Meaning |
|---|---:|---|---|
| `min_conf` / `filter_threshold` | `0.1` | `--min-conf` | Match confidence threshold. Lower increases matches and outliers; higher gives cleaner but fewer matches. |
| `width_confidence` | `0.95` | not exposed | Point pruning confidence. Set `-1` to disable if patching the wrapper. |
| `depth_confidence` | `-1` | not exposed | Early stopping confidence; `-1` disables early stopping. |
| `flash` | `True` | not exposed | Use FlashAttention if available. |
| `mp` | `False` | not exposed | Mixed precision. |
| `n_layers` | `6` | model config | Transformer depth. Should not be changed without matching weights. |
| `num_heads` | `1` | model config | Attention heads. Should not be changed without matching weights. |
| `input_dim` | `64` | model config | XFeat descriptor dimension. Fixed. |
| `descriptor_dim` | `96` | model config | Internal descriptor dimension. Fixed by weights. |

### Recommended Sweeps

For map update / bridge search:

| Situation | Try |
|---|---|
| Too few matches / bridges | Increase `--qk` 4096 -> 8192, then increase retrieval `--topk` 15 -> 30. |
| Matches exist but PnP fails | Keep `--min-inliers` stable; check support area and repeated-structure false retrieval before lowering thresholds. |
| Wide-baseline new/old overlap is hard | Lower `--min-conf` from `0.1` to `0.05`, but require stronger PnP and support-area gates. |
| Runtime too slow | Use `top_k=2048` for deployed tracking, but keep `4096` for offline update. |
| Pure rotation connectors | Use them for retrieval/PnP/bundle coverage only; do not create 3D points from them. |

Do not tune by only maximizing raw match count. The real metrics are:

- PnP inliers after RANSAC
- inlier spatial distribution
- pose continuity / conditioning
- Sim3 residual for submap merge
- held-out localization success and median inliers
