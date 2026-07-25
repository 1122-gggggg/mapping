# Map Update Pipeline Methods

This update pipeline uses the deployed old map as the anchor:

- Base sparse map: `/media/cihcilab/新增磁碟區/sfm_glomap/glomap_fused/0`
- Base localization bundle: `/media/cihcilab/新增磁碟區/sfm_glomap/deploy/reloc_map_xfeat_tri.pt`
- Base MegaLoc cache: `out_update/base_megaloc.npz` or `/media/cihcilab/新增磁碟區/sfm_glomap/deploy/megaloc_ref_desc_glomap_fused_322.npy`

The future deployment path uses MegaLoc for global retrieval and XFeat +
LighterGlue for local matching. Bundle `ref_global` should therefore be MegaLoc
descriptors. BoQ is no longer part of the update or deployment path.

## Route Selection

新資料進來：

```text
是否能穩定 match 到舊地圖？
  ├─ 是，且重疊高
  │     -> 2. 增量註冊 + 局部三角化
  │
  ├─ 部分重疊，但新區域很多
  │     -> 3. 局部 submap + Sim3/SE3 合併
  │
  └─ 有重疊但大量幾何/語義不一致
        -> 4. 局部區塊替換
```

High-overlap skip rule:

- If `register_rate > 0.95` and there is no evidence of scene change, do not
  update the map. Keep that video as validation/QA.
- If `register_rate > 0.95` but a known region has changed, do not skip it.
  Route it to method 4 (`changed` / `tile_replace`) so old support in that tile
  can be invalidated or down-weighted.

| Task | 適用情境 | 核心做法 | 舊地圖處理方式 |
|---|---|---|---|
| 2. 增量註冊 + 局部三角化 | 新影像和舊地圖高度重疊 | 用舊 3D points 做 PnP / pose registration，成功定位後，只對新 keyframe 做局部 triangulation 與 local BA | 舊圖當 anchor；舊 pose / 舊 points 大多固定，只加入新點或少量調整局部 |
| 3. 局部子圖 + Sim3 合併 | 新資料主要覆蓋新區域，但和舊圖有少量橋接 overlap | 先獨立建一塊小 submap，再用橋幀、共視點、loop constraint 或 cross-map matches 算 Sim3 / SE3 對齊 | 不重建舊圖；只把新 submap 對齊後併入舊 map graph |
| 4. 局部區塊替換 | 場景內容變了，例如裝潢、物件移動、結構更新 | 偵測 changed region / tile，invalidate 舊 points/keyframes，再重建該 tile | 只替換變動區；非變動區保留並作為邊界 anchor |

Terminology:

- overlap: scene-level coverage. How much of the new video sees areas already
  represented by the old map.
- registerable / register_rate: frame-level result. The fraction of sampled new
  frames that can solve PnP against old 3D points with enough inliers.
- inliers: correspondence-level quality. For one localized frame, the number of
  2D-3D matches that agree with the same camera pose under RANSAC.

`register_rate` is a proxy for overlap, not a semantic-change detector. A video
can have high register_rate while one wall/object in the image has changed.
Changed-region detection must inspect local support and appearance consistency.

Practical routing matrix:

| Signals | Interpretation | Action |
|---|---|---|
| overlap high, register_rate high, inliers high | Old area is already covered well | Do not update map; keep as validation/QA |
| overlap high, register_rate high, but local inlier/support drops in one region | Same geometry location may have changed appearance or content | Method 4 changed-region / tile replacement |
| overlap medium, register_rate low, but enough bridge frames exist | New video covers new area with some old-map overlap | Method 3 submap + Sim3 |
| retrieval similarity high, but PnP inliers low | Likely false overlap from repetitive structure | Do not trust retrieval; require geometric verification or reject |
| inliers high but spatially concentrated in a small image corner/tile | Pose may solve, but image coverage is incomplete or locally ambiguous | Check inlier distribution; do not treat as full overlap |

Current automation:

- `map_update_tool.py` uses PnP inlier count to estimate `register_rate`.
- `--skip-overlap-thr 0.95` skips very high-overlap videos by default.
- High-overlap skipped videos are still recorded as observation sessions in
  `observation_stats.json`; this updates which base keyframes / anchors are
  useful without adding new geometry.
- `--force-method SEQ=changed` records a changed-region/tile-replacement need.
- Local tile support is recorded in `observation_stats.json`. Tiles with stable
  global localization but repeatedly low old-map support are listed as
  changed-region candidates.
- Bridge quality is reported through bridge count, geometry/connector split,
  median inlier ratio, median support area, pose spread, and Sim3 residual.
- `--adaptive-params` is enabled by default. If bridge count is too low, the
  tool retries bridge search with more XFeat keypoints, larger MegaLoc topK,
  and looser LighterGlue confidence. If retrieval looks like false overlap or
  bridge support is spatially weak, it retries with stricter confidence and
  records all attempts in `bridge_quality_<SEQ>.json`.

Adaptive parameter policy:

| Situation | Parameters to relax/tighten | Reason |
|---|---|---|
| Few bridge frames but some old-map overlap exists | Increase bridge `qk` to 8192, increase MegaLoc topK to 30/50, allow `min_conf` 0.05 | Recover difficult wide-baseline or low-texture bridges |
| Retrieval similarity high but PnP inliers low | Raise `min_conf` to 0.15/0.2 and keep geometric gates strict | Avoid repeated-structure false overlap |
| Inliers high but concentrated in a small tile | Do not increase trust only from inlier count; require support area and tile health | Prevent pose from being accepted from a tiny image corner |
| High-overlap QA videos | Keep map fixed, record observation health, optionally add only sparse deployment keyframes when domain gap is valuable | Avoid bundle growth and geometry pollution |
| New coverage submap | Use geometry frames for triangulation, connector frames for retrieval/PnP/bridges only | Pure rotation helps linking but not stable 3D point creation |

Current implementation status:

- Method 2 currently performs fast registration and adds localization keyframes.
  Local triangulation/local BA for the register route is the next extension.
- Method 3 is implemented as XFeat+LighterGlue submap reconstruction followed
  by Umeyama Sim3 alignment from bridge frames.
- Method 4 is still not an automatic tile rebuild, but the required detection
  evidence is now produced: `observation_stats.json` contains old anchor health
  and changed-region tile candidates.
- Post-validation compaction is handled by
  `sparsify_reloc_bundle.py`, which writes a separate candidate slim bundle and
  report. Validate the slim bundle before replacing deployment assets.

## Intrinsics Policy

Every resolution uses a fixed SIMPLE_RADIAL calibration. Do not let update-time
tools estimate arbitrary intrinsics for these known camera modes.

| Resolution | SIMPLE_RADIAL params `[f, cx, cy, k1]` |
|---|---|
| 2688x1512 | `[1955.5, 1344.0, 756.0, 0.0020]` |
| 1920x1080 | `[1400.0, 960.0, 540.0, 0.0015]` |
| 1280x720 | `[936.5, 640.0, 360.0, 0.0035]` |

`map_update_tool.py` uses this table for both PnP and submap image import.

## Frame Selection

Preferred order:

1. Replay known `source_idx` from `frame_manifest.json` when reproducing a
   previous build/update run.
2. For new videos, sample by fps and then apply motion/parallax gating.
3. Keep hover near-duplicates sparse.
4. Keep a small number of pure-rotation frames for localization-only coverage,
   but do not rely on them to create new 3D points.
5. For submap/triangulation frames, prefer actual translation baseline and
   parallax; large optical flow alone can be pure rotation.

Use:

```bash
/usr/bin/python3.12 prepare_update_frames.py \
  --manifest /media/cihcilab/新增磁碟區/sfm_reshot25/frame_manifest.json \
  --out-root /path/to/update/new_frames \
  --overwrite
```

For unseen videos without a manifest:

```bash
/usr/bin/python3.12 prepare_update_frames.py \
  --video P2000200=/path/P2000200.MP4 \
  --out-root /path/to/update/new_frames \
  --fps 2 \
  --motion-filter parallax \
  --min-flow-px 0.8 \
  --max-flow-px 180 \
  --keep-rotation-every 8 \
  --split-classes
```

The parallax filter uses sparse feature geometry, not only flow magnitude. It
rejects frames that are too similar, too jumpy, weakly matched, or
homography-dominant (`H/F` inlier ratio high), because those frames are often
hover duplicates or pure rotation. Pure-rotation frames can help localization
coverage, but they should not dominate submap triangulation.

Split output convention:

- `geometry/<SEQ>/`: frames with translation/parallax, used for submap and
  triangulation.
- `connector/<SEQ>/`: sparse rotation-like connector frames, used for
  retrieval, PnP, and deployment bundle only. They should not create new 3D
  points.

Route classification and high-overlap skip use the union of `geometry` and
`connector` frames, because pure-rotation connector frames can still prove the
video is observing the old map. Method 3 reconstruction uses only `geometry`
frames to avoid low-parallax triangulation.

Connector measurement:

1. Sample candidate frames at the configured fps.
2. Match adjacent candidate frames with ORB + BFMatcher ratio test.
3. Estimate both Fundamental matrix `F` and Homography `H` with RANSAC from the
   same matches.
4. Mark a candidate as rotation-like connector when both models have enough
   inliers and `H_inliers / F_inliers >= 0.85`.
5. Keep only every `--keep-rotation-every N` connector candidate. In this run,
   `N=8`.

Connectors are therefore H-dominant / rotation-like frames. They are useful for
MegaLoc retrieval, cross-view matching, PnP, and deployment keyframe coverage,
but are not trusted for creating new 3D points.

Future stricter connector checks can use old-map PnP pose: camera center nearly
unchanged while viewing direction changes significantly, with broad inlier
support across the image.

Connector-assisted bridge policy:

- Connector frames may be included in the temporary submap reconstruction so
  they can obtain a submap pose.
- If the same connector frame also PnPs to the old base map, it becomes a Sim3
  bridge.
- Connector frames are still not allowed to export new geometry. The updated
  tool only exports submap points observed by at least
  `--min-geometry-observations` geometry frames, default 2.
- Connector keyframes in the deployment bundle use old-map inherited xyz from
  PnP, not low-parallax submap points.

This handles cases like P125 where new geometry frames can build a submap but
old-map overlap appears mainly during rotation/turning connector frames.

Run the update with:

```bash
/usr/bin/python3.12 map_update_tool.py \
  --new-data /path/to/new_frames/geometry \
  --connector-data /path/to/new_frames/connector \
  --skip-overlap-thr 0.95
```

The run also writes:

- `observation_stats.json`: observation sessions, old anchor health, tile
  support, and changed-region candidates.
- `bridge_quality_<SEQ>.json`: per-bridge raw matches, PnP inliers, inlier
  ratio, support area, and geometry/connector type.

## ExMaps-Style Stability Layer

The update pipeline now writes an optional `ref_stability` array into updated
bundles. It is aligned with `ref_names` and uses 1.0 as a neutral score. The
score follows the ExMaps idea: old refs decay over update sessions unless new
observation sessions repeatedly support them, while newly-added refs receive a
route-quality prior from the update report.

Current defaults:

- `--stability-half-life-sessions 4.0`
- `--stability-rerank-weight 0.05`
- `sparsify_reloc_bundle.py --stability-weight 0.35`

The score is used conservatively:

- update-time PnP/bridge search reranks only within the raw MegaLoc candidate
  pool;
- deployment localization uses the same mild topK tie-breaker;
- validation stream uses the same behavior, so held-out validation measures the
  deployment path;
- sparsification keeps high-stability refs when trimming a bundle.

Detailed implementation notes, baseline settings, and inspected results are in
`EXMAPS_STABILITY_UPDATE_LOG.md`.

## MegaLoc Deployment Contract

If a bundle has `meta.bundle_vpr = "megaloc"`, then:

- `ref_global` rows are 8448-d L2-normalized MegaLoc descriptors.
- Query-time localization must use MegaLoc query descriptors.
- Descriptor health is a hard gate: shape must be 2D, dim 8448, finite, and
  L2 norms should be near 1.0. Norms below 0.5 indicate corrupt/black frames and
  should stop the update instead of silently entering deployment.

Old bundles whose `ref_global` was built with another VPR must be regenerated
or evaluated with a MegaLoc descriptor cache override before deployment.

## Sim3 vs SE3

Sim3 is usually needed for monocular SfM/SLAM because scale can drift between
submaps. If the input source has known metric scale, such as stereo, RGB-D,
LiDAR, or a metric prior, SE3 may be sufficient.

In this update pipeline, Sim3 is the 7-DoF transform that maps the independent
new submap coordinate system into the old base-map coordinate system:

```text
X_base ~= scale * R * X_submap + t
```

How it is computed:

1. Build the new video submap independently from `geometry` frames.
2. Find bridge frames that have two valid poses:
   - a submap pose from the new reconstruction;
   - a base-map pose from XFeat+LighterGlue 2D-3D PnP against old points.
3. Convert each bridge pose to a camera center in both coordinate systems.
4. Run Umeyama alignment on the paired camera centers:
   `submap_centers -> base_centers`.
5. Apply the resulting `scale, R, t` to all new submap 3D points and per-keypoint
   xyz anchors before adding them to the updated localization bundle / PLY.

The report's `bridges` and `resid(u)` are the health checks. More bridges and
lower residual mean the submap is better anchored to the old map.

## Anchor Policy

"Old map fixed" should not mean the boundary can never move. A practical policy:

- old stable map: fixed or strong prior
- new frames / new points: optimized
- boundary frames / shared points: optionally optimized with robust prior

This keeps the old map authoritative while reducing cracks at the join.

## Changed Region Detection

"同一幾何位置，但外觀變了" should be handled as method 4, not as a normal
submap merge and not as a high-overlap skip.

Detection should use multi-view evidence:

1. Localize new frames to the old map with PnP. If poses are stable and inliers
   are high, the new images are observing the same old geometry.
2. Read `observation_stats.json`.
3. For each tile, compare query keypoint count, old-map matches, and PnP inlier
   count. A tile is only a candidate if global pose is stable but local old-map
   support is repeatedly low.
4. Do not invalidate points from one frame. Require repeated evidence across
   multiple frames and preferably multiple sessions.

## Bundle Sparsification

After held-out validation passes, create a candidate slim bundle:

```bash
/usr/bin/python3.12 sparsify_reloc_bundle.py \
  --bundle /path/to/reloc_map_updated.pt \
  --observation-stats /path/to/observation_stats.json \
  --target-fraction 0.85 \
  --min-per-prefix 40 \
  --out /path/to/reloc_map_updated_slim.pt
```

This uses observation-hit scores plus ordered K-cover per sequence/prefix. It
does not overwrite the validated full bundle. Always run `eval_stream.py` on the
candidate slim bundle before deploying it.
2. Project old 3D points / old local tiles into the new frames.
3. Measure whether support is locally inconsistent:
   - old points project into a region but repeatedly fail local matching;
   - inlier support drops in a compact 3D/2D tile while surrounding regions
     still match;
   - color residuals remain high after exposure/illumination normalization;
   - semantic or DINO/CLIP-style descriptors differ consistently across views.
4. Accumulate votes over several frames and viewing angles. A single-frame
   color difference is not enough because exposure, shadow, reflection, and
   view angle can all produce false positives.

Interpretation:

- Color/texture changed, geometry stable: keep old geometry, update RGB /
  descriptors / support for that tile.
- Object/wall moved or removed: invalidate old points/keyframes in that tile
  and rebuild the tile with boundary anchors.
- Widespread inconsistency: method 4 local replacement, not method 2 register
  and not method 3 blind submap append.

## Update Acceptance Checks

Before replacing a deployed map, inspect:

- frame selection report: each video has enough saved parallax frames and does
  not consist mainly of rejected rotation-like candidates.
- update report: register videos add keyframes; submap videos have enough
  bridge frames, reasonable Sim3 residual, and no `skipped_no_bridges`.
- MegaLoc logs: descriptor shapes/norms are healthy for base and new frames.
- validation localization: held-out forward and reverse videos should not
  regress in success rate or inlier statistics compared with the old bundle.
- artifact policy: write updated bundle/PLY to a new output directory first;
  do not overwrite `glomap_fused/0` or the deployed bundle until validation.

## Known Intrinsics Note

Older project memory also records a 2688x1512 FULL_OPENCV Charuco calibration
around `fx≈2017, fy≈2012, cx≈1408.7, cy≈753.3`. The current update tool keeps
SIMPLE_RADIAL because the deployed localization bundle and current update path
use that model. If final accuracy becomes limited by residuals at joins, run an
intrinsics bake-off before changing the deployed camera model.
