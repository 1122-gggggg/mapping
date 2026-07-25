# sfm_reshot25 Build / Update Preservation Notes

Date: 2026-07-01

## Keep

- Final GLOMAP map: `glomap_rebuild_unified_dense/0`
- Frame and camera records: `frame_manifest.json`, `map_intrinsics.json`
- Build history logs: `BUILD_RUN_LOG.md`, `logs/`
- Future one-file full build pipeline: `build_localizable_map.py`
- Incremental update recipe and parameters: `update_pipeline/map_update_tool.py`
- Update frame preparation: `update_pipeline/prepare_update_frames.py`
- Update method notes: `update_pipeline/UPDATE_PIPELINE_METHODS.md`
- Research and tuning notes: `update_pipeline/PAPER_GLOMAP_XFEAT_NOTES.md`
- Post-validation bundle sparsification: `update_pipeline/sparsify_reloc_bundle.py`
- Update validation result: `update_pipeline/validate_p123_126.json`

## Removed Map Outputs

These are generated map artifacts, not the recipe:

- `update_pipeline/reloc_map_updated_P124_P125.pt`
- `update_pipeline/latest_map_P124_P125_realrgb.ply`

They can be regenerated from `update_pipeline/map_update_tool.py` if the same
base map, new frames, and parameters are available.

## Full New-Site Map Build

Use `build_localizable_map.py` for a fresh field:

```bash
/usr/bin/python3 build_localizable_map.py \
  --videos /path/to/site_video_1.mp4 /path/to/site_video_2.mp4 \
  --work-dir /media/cihcilab/新增磁碟區/new_site_map \
  --site-name new_site
```

Default pipeline:

1. Extract frames.
2. Build MegaLoc retrieval, temporal, and cross-video bridge pairs.
3. Run MV-RoMa dense matching with UFM prematch.
4. Aggregate dense matches into COLMAP keypoints/matches.
5. Build a legacy COLMAP database on an ext4-safe temp path.
6. Run GLOMAP.
7. Export RGB point cloud.
8. Build XFeat localizer bundle, tracking metadata, and triangulated bundle.

Important defaults preserved from this run:

- MegaLoc `num_matched=20`
- temporal `seq_window=5`
- cross-video bridge `cross_topk=8`, `cross_grid=10`
- MV-RoMa `roma_cert_thresh=0.35`
- aggregation `agg_maxkp=4000`, `agg_pair_degree_cap=18`, `agg_cross_degree_cap=8`
- GLOMAP `skip_bundle_adjustment=1`, `skip_retriangulation=1`, `max_num_tracks=600000`
- XFeat `topk=2048`, `snap_px=5.0`, triangulation `pair_topk=20`

## Incremental Map Update

Use `update_pipeline/map_update_tool.py` when an existing map only needs added
coverage from new videos. It keeps the old geometry authoritative and adds
register/submap keyframes into a deployable localization bundle.

Validated update defaults in that file:

- deployment VPR `bundle_vpr=megaloc` for new bundles
- `qk=4096`
- `topk=15`
- `min_conf=0.1`
- `min_inliers=50`
- `seq_window=6`
- `retr_nn=10`
- `overlap_thr=0.6`
- `classify_stride=5`
- `min_bridges=4`
- ANAFI SIMPLE_RADIAL default: `focal=1955.5`, `k1=0.002`
- fixed intrinsics by resolution:
  - `2688x1512`: `[1955.5, 1344.0, 756.0, 0.0020]`
  - `1920x1080`: `[1400.0, 960.0, 540.0, 0.0015]`
  - `1280x720`: `[936.5, 640.0, 360.0, 0.0035]`
- frame selection for new videos should prefer split-class extraction:
  `prepare_update_frames.py --motion-filter parallax --split-classes`.
  `geometry/SEQ` contains translation/parallax frames for submap and
  triangulation. `connector/SEQ` contains sparse rotation-like frames for
  retrieval/PnP/deployment bundle only.
- high-overlap videos with `register_rate > 0.95` should not update the map by
  default. Keep them as validation/QA unless they show known changed regions or
  a deployment-critical domain gap.
- high-overlap skipped videos should still be recorded as observation sessions:
  `observation_stats.json` records useful base keyframes, old anchors, local
  tile support, and changed-region candidates without adding geometry.
- if a high-overlap area changed appearance at the same geometry location, route
  it to method 4 (`changed` / `tile_replace`) instead of skipping. Detect this
  by stable PnP to old geometry plus multi-view local support/color/semantic
  inconsistency in a compact tile.
- bridge quality should be judged by bridge count, geometry/connector split,
  median PnP inlier ratio, median inlier support area, pose spread, and Sim3
  residual. Connector bridges can help alignment but do not export geometry.
- adaptive bridge search is enabled by default. When bridge count is low, retry
  with higher XFeat keypoint budget, larger MegaLoc topK, and looser
  LighterGlue confidence. When retrieval looks like false overlap or support is
  spatially weak, retry with stricter match confidence and keep geometric gates
  strict. All attempts are recorded in `bridge_quality_<SEQ>.json`.
- after validation, use `sparsify_reloc_bundle.py` to produce a separate slim
  candidate bundle from observation-hit scores plus ordered K-cover. Validate
  the slim candidate before replacing any deployment bundle.
- MegaLoc deployment bundles must check descriptor health before writing:
  2D array, expected dim 8448, finite values, and L2 norms near 1.0. Norms below
  0.5 indicate corrupt/black frames and should stop the update.
- Use dash-free sequence names such as `P1210121`; avoid raw names like `a-b`
  inside hloc/MV-RoMa style pair keys.
- Updated bundle/PLY outputs are written to a new update directory first. Do
  not overwrite `sfm_glomap/glomap_fused/0` or the deployed bundle until
  held-out localization validation passes.

Current split-update run:

- Frame split root:
  `/media/cihcilab/新增磁碟區/sfm_glomap/update_from_reshot25_build_20260701/new_frames_split_v1`
- Output root:
  `/media/cihcilab/新增磁碟區/sfm_glomap/update_from_reshot25_build_20260701/out_update_megaloc_split_v3`
- Bundle:
  `out_update_megaloc_split_v3/reloc_map_updated.pt`
- PLY:
  `out_update_megaloc_split_v3/latest_map_realrgb.ply`
- Result summary:
  - `P1210121`: register-only, +76 keyframes, +0 points
  - `P1220122`: register-only, +72 keyframes, +0 points
  - `P1240124`: submap+Sim3, +75 keyframes, +13177 points, 58 bridges, residual 0.078u
  - `P1250125`: submap+Sim3, +25 keyframes, +2852 points, 18 bridges, residual 0.024u
- Bundle global descriptors: MegaLoc `(2168, 8448)`, L2 norms approximately 1.0.
- BoQ has been removed from the active update/deployment path. Old bundles can
  still be loaded as geometry/keyframe sources only when their `ref_global`
  descriptors are overridden by a MegaLoc cache.
- Held-out 720p streaming validation against
  `/media/cihcilab/新增磁碟區/補拍影片/test`:
  - result JSON:
    `/media/cihcilab/新增磁碟區/sfm_glomap/update_from_reshot25_build_20260701/out_update_megaloc_split_v3/eval_test_720p_v3_vs_base.json`
  - `P1230123`: base `88.5%` success / median `714` inliers;
    v3 `89.6%` success / median `710` inliers; `+1.1pp`, `3` fail->ok,
    `0` ok->fail.
  - `P1260126`: base `100.0%` success / median `858` inliers;
    v3 `100.0%` success / median `995` inliers; `0` ok->fail.

## Optional Modules

- `UFM`: used indirectly inside MV-RoMa as the prematch model. Keep it if using
  the MV-RoMa stage.
- `Doppelgangers++`: optional pair filter before dense matching. Use when the
  field has repeated or symmetric structures that create plausible but wrong
  image pairs. Skipping it is faster, but risks false edges and split/ghosted
  reconstructions in repetitive scenes.
- `LFOE-GlobalSfM`: optional GLOMAP replacement/filter. Use when the view graph
  has relative-translation outliers or camera positions drift/splinter. Skipping
  it is fine for clean graphs and avoids extra dependency/runtime.
