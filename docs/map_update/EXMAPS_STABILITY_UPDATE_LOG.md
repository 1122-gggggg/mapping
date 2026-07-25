# ExMaps-Style Stability Update Log

Date: 2026-07-10 (Taipei local time)

## Status

Implemented the first ExMaps-style stability layer for the incremental map
update pipeline. This change does not rerun the heavy SfM/GPU map update by
itself. It makes future update runs write and use `ref_stability` so new
observation sessions can influence later localization and bundle sparsification.

## Code Changes

| File | Change |
|---|---|
| `source/sfm_reshot25/update_pipeline/stability_scores.py` | New pure scoring module: exponential decay, observation-hit boost, route-quality priors, and stable-ref retrieval rerank. |
| `source/sfm_reshot25/update_pipeline/map_update_tool.py` | Reads prior `ref_stability`, mildly reranks MegaLoc candidates during update-time PnP/bridge matching, and writes aligned `ref_stability` plus metadata into `reloc_map_updated.pt`. |
| `source/sfm_reshot25/update_pipeline/sparsify_reloc_bundle.py` | Uses `ref_stability` during top-up/trimming and preserves aligned `ref_stability` in slim bundles. |
| `pipeline/update_pipeline.py` | Gates optional `ref_stability` shape/finite validity when validating update outputs. |
| `定位/deploy_code/sfm_glomap_deploy/reloc_localizer_xfeat.py` | Deployment localizer loads `ref_stability` from bundles and uses it as a mild MegaLoc topK tie-breaker. Old bundles default to neutral score 1.0. |
| `定位/validation/eval_stream_core.py` | Validation stream now applies the same stability rerank, so `--validate-dir` measures the deployment behavior. |
| `test_stability_scores.py`, `test_sparsify_stability.py`, `test_update_pipeline_sparsify.py` | Added regression tests for stability scoring, sparsify retention, and bundle gate alignment. |

## Stability Model

The bundle now carries:

- `ref_stability`: float32 array aligned with `ref_names`; 1.0 is neutral.
- `meta.ref_stability`: scoring metadata and thresholds.
- `meta.ref_stability_summary`: min/median/max and counts above/below neutral.
- `meta.stability_rerank_weight`: default `0.05`.

Scoring:

```text
existing_score = prior_score * decay + normalized_recent_observation_bonus
new_score      = route_quality_prior
decay          = 0.5 ** (1 / half_life_sessions)
```

Default parameters:

| Parameter | Default | Meaning |
|---|---:|---|
| `--stability-half-life-sessions` | `4.0` | One update session decays old scores by `0.840896`. |
| `observed_bonus` | `0.45` | Max boost for refs repeatedly reused as old-map PnP anchors in the latest observation stats. |
| `min_score` / `max_score` | `0.05` / `2.0` | Bounds to avoid hard deletion or runaway scores. |
| `--stability-rerank-weight` | `0.05` | Applies `similarity + 0.05 * log(stability)` within the raw MegaLoc candidate pool. |
| `--stability-weight` in sparsify | `0.35` | Preserves high-stability refs during bundle compaction. |

Route priors for newly added refs:

| Route / signal | Prior |
|---|---:|
| Good submap with enough geometry bridges | about `1.0` to `1.3` |
| Register-only keyframes | `0.85` |
| Connector-only keyframes | `0.70` |
| Submap with `retrieval_high_but_inliers_low` or fewer than 4 geometry bridges | `0.65` |

This matches the intended behavior:

- High-overlap videos update observation health, not geometry.
- Strong new coverage such as P124 can help future localization.
- Weakly bridged but useful coverage such as P125 is retained, but cannot dominate retrieval.

## Existing Experiment Baseline

Existing run inspected:

- Output: `outputs/verify_update_20260702_tracking_skip_p122`
- Report: `outputs/verify_update_20260702_tracking_skip_p122/update_report.md`
- Prepared frames: `定位/source/sfm_glomap/update_from_reshot25_build_20260701/new_frames_split_v1`
- Historical base bundle: `定位/bundles/base_reloc_map_xfeat_tri.pt` (removed after current-map-only cleanup)
- Output bundle: `outputs/verify_update_20260702_tracking_skip_p122/reloc_map_updated.pt`

Frame preparation settings:

| Setting | Value |
|---|---:|
| fps | `3.0` |
| motion filter | `parallax` |
| min flow px | `0.8` |
| max flow px | `180.0` |
| keep rotation every | `8` |
| split classes | `true` |

Frame selection result:

| Seq | Tested | Geometry | Connector | Kept | Geometry rate | Connector rate |
|---|---:|---:|---:|---:|---:|---:|
| P1210121 | 801 | 17 | 71 | 88 | 2.1% | 8.9% |
| P1220122 | 546 | 22 | 54 | 76 | 4.0% | 9.9% |
| P1240124 | 903 | 41 | 77 | 118 | 4.5% | 8.5% |
| P1250125 | 419 | 33 | 44 | 77 | 7.9% | 10.5% |

Map update settings:

| Setting | Value |
|---|---:|
| `qk` | `4096` |
| `topk` | `15` |
| `min_conf` | `0.1` |
| `min_inliers` | `50` |
| `seq_window` | `6` |
| `retr_nn` | `10` |
| `overlap_thr` | `0.6` |
| `skip_overlap_thr` | `0.95` |
| `min_geometry_observations` | `2` |
| `submap_connector_bridges` | `true` |
| forced route | `P1220122=skip_high_overlap` |

Map update result:

| Seq | Route | Status | Register rate | Geom/Conn frames | Keyframes added | Points added | Bridges (geom/conn) | Sim3 resid |
|---|---|---|---:|---:|---:|---:|---:|---:|
| P1210121 | register | ok | 83% | 17/71 | 76 | 0 | - | - |
| P1220122 | skip_high_overlap | qa_only | - | 22/54 | 0 | 0 | - | - |
| P1240124 | submap | ok | 58% | 41/77 | 74 | 13164 | 60 (19/41) | 0.123 |
| P1250125 | submap | retrieval_high_but_inliers_low | 19% | 33/44 | 25 | 2941 | 18 (2/16) | 0.022 |

Bundle gate:

| Metric | Value |
|---|---:|
| Base keyframes | 1920 |
| Updated keyframes | 2095 |
| Added keyframes | 175 |
| Updated bundle size | 1.4 GB |
| `ref_global` shape | 2095 x 8448 |
| tracking metadata | true |
| PLY size | 7.4 MB |

Observation health:

| Seq | Frames observed | Localized | Success rate | Median support area | Median inlier ratio |
|---|---:|---:|---:|---:|---:|
| P1210121 | 88 | 76 | 86.4% | 0.305 | 0.370 |
| P1220122 | 16 | 15 | 93.8% | 0.367 | 0.459 |
| P1240124 | 95 | 64 | 67.4% | 0.280 | 0.345 |
| P1250125 | 46 | 18 | 39.1% | 0.087 | 0.335 |

Changed-region candidate count: `63`.

## Interpretation

- `P1220122` should stay QA/observation-only unless there is independent
  changed-region evidence. It localizes well enough against the old map and does
  not justify bundle growth.
- `P1240124` is the strongest geometry update in the inspected run: many bridges,
  19 geometry bridges, and 13k geometry-supported points.
- `P1250125` adds useful coverage, but bridge support mostly comes from connector
  frames and only 2 geometry bridges. It is retained with a conservative
  stability prior (`0.65`) so it can help without dominating future retrieval.
- Pure rotation/connector frames are useful for bridge/PnP health, but should not
  generate new 3D points.

## Required Next Experiment

The inspected output did not contain a full `validation_compare.json` under
`outputs/verify_update_20260702_tracking_skip_p122`. Before promoting a future
updated bundle, run held-out validation using videos that were not used to build
the update:

```bash
cd "/media/cihcilab/新增磁碟區/sfm_system/更新地圖"

RUN=update_$(date +%Y%m%d_%H%M)

python3 pipeline/update_pipeline.py \
  --python /usr/bin/python3.12 \
  --video P2000200="/path/to/new_video_1.MP4" \
  --video P2000201="/path/to/new_video_2.MP4" \
  --work-dir "workspaces/$RUN" \
  --out-dir "outputs/$RUN" \
  --fps 3 \
  --motion-filter parallax \
  --min-flow-px 0.8 \
  --max-flow-px 180 \
  --keep-rotation-every 8 \
  --validate-dir "inputs/補拍影片/test" \
  --validate-stride 10 \
  --validate-resize 1280x720 \
  --skip-overlap-thr 0.95
```

Promotion criteria:

- `gates/update_outputs.json` passes.
- `validation_compare.json` shows no increase in `ok_to_fail`.
- `final_success` is at least the configured threshold, or baseline-improved
  when the base is below target.
- `bridge_quality_<SEQ>.json` has enough bridges, broad support area, and
  acceptable Sim3 residual for any submap route.
- For slim bundles, run sparsify only after validation and validate the slim
  candidate separately.

## Verification Performed For This Code Change

Commands:

```bash
pytest -q \
  pipeline/test_update_pipeline_sparsify.py \
  source/sfm_reshot25/update_pipeline/test_update_quality_gates.py \
  source/sfm_reshot25/update_pipeline/test_changed_region_evidence.py \
  source/sfm_reshot25/update_pipeline/test_stability_scores.py \
  source/sfm_reshot25/update_pipeline/test_sparsify_stability.py

/usr/bin/python3.12 -m py_compile \
  pipeline/update_pipeline.py \
  source/sfm_reshot25/update_pipeline/stability_scores.py \
  source/sfm_reshot25/update_pipeline/map_update_tool.py \
  source/sfm_reshot25/update_pipeline/sparsify_reloc_bundle.py \
  ../定位/deploy_code/sfm_glomap_deploy/reloc_localizer_xfeat.py \
  ../定位/validation/eval_stream_core.py
```

Results:

- `pytest`: 8 passed, 4 skipped. Skips are due to the installed pytest tool
  environment lacking numpy/torch; the project Python has those libraries.
- `/usr/bin/python3.12 -m py_compile`: passed.
- Direct `/usr/bin/python3.12` stability smoke test: passed.

Heavy SfM/GPU map update was not rerun as part of this code change.
