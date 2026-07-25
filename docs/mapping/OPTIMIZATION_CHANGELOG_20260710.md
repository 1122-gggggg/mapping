# Optimization Changelog 2026-07-10

This records the gate/tooling changes made after the July 2026 map-build,
localization, update-map, deployment, and data-governance review.

## Scope

- No raw videos, DB/H5 files, bundles, maps, or production symlinks were deleted.
- No expensive MV-RoMa, GlueMap, GLOMAP, localization replay, or GPU training job
  was executed by this change.
- This `sfm_system` directory is not a git repository, so changes were recorded
  here instead of committed.
- Before experiment changes, original file backups were written to
  `docs/experiment_backups_20260710/` with SHA-256 hashes in `manifest.json`.
- After the stricter intrinsics/system-verifier gate landed, a stable checkpoint
  backup was written to `docs/experiment_backups_20260710_intrinsics_gate_current/`
  with SHA-256 hashes in `manifest.sha256`.
- After the stricter DB quality gate landed, a stable checkpoint backup was
  written to `docs/experiment_backups_20260710_database_gate_current/` with
  SHA-256 hashes in `manifest.sha256`.
- After the stricter point-cloud quality gate landed, a stable checkpoint backup
  was written to `docs/experiment_backups_20260710_point_cloud_gate_current/`
  with SHA-256 hashes in `manifest.sha256`.
- After the stricter report/provenance quality gate landed, a stable checkpoint
  backup was written to `docs/experiment_backups_20260710_report_gate_current/`
  with SHA-256 hashes in `manifest.sha256`.
- After the stricter preflight/input quality gate landed, a stable checkpoint
  backup was written to `docs/experiment_backups_20260710_preflight_gate_current/`
  with SHA-256 hashes in `manifest.sha256`.

## Build

- Added `建圖/pipeline/stage_gate_contract.py`.
  - Produces `BUILD_GATE_SUMMARY.json` and `BUILD_GATE_SUMMARY.md`.
  - Normalizes both current stage gates (`stage`/`ok`) and older GlueMap gates
    (`gate`/`passed`).
  - Blocks promotion when a run has a colored point cloud but no localizable
    XFeat/MegaLoc deployment bundle.
  - Adds a required `preflight_quality` handoff stage. The accepted preflight
    package must include readable `preflight_report.json`, per-video
    path/codec/resolution/fps/duration/nb_frames/expected extraction metadata,
    consistent target resolution, sane camera intrinsics, core tool availability,
    >=50GB disk headroom, and passing GPU status when GPU is required.
  - Adds a required `frame_motion_quality` handoff stage. Detailed
    `selection_motion_quality` gates must report selected frames >= 60,
    selected ratio >= 65%, parallax/seed ratio >= 65%, hover ratio <= 5%, and
    bridge frames for multi-group builds; legacy `extract` gates must at least
    report frames, per-group kept counts/ratios, and usable motion classes when
    available.
  - Adds a required `intrinsics_manifest_quality` handoff stage.
    `frame_manifest.json` must report enough frames with coherent frame
    counts/sizes, and `map_intrinsics.json` must provide parseable
    per-resolution or shared camera intrinsics. The gate rejects non-positive
    focal lengths, principal points outside the image, missing camera params,
    and distorted-source inputs without an explicit undistort/no-undistort
    decision.
  - Adds a required `view_graph_quality` handoff stage. The accepted pair graph
    gate must report pairs >= max(500, frames*4), connected components <= 1,
    largest component ratio >= 90%, cross-video/cross-sequence bridge pairs > 0,
    and core connectivity/bridge metrics must be present.
  - Adds a required `database_quality` handoff stage. The accepted DB gate must
    point to or accompany a readable COLMAP SQLite database with
    cameras/images/keypoints/matches/two_view_geometries tables, images >= 60,
    image count equal to `frame_manifest.json`, full image-to-keypoint coverage,
    min keypoints/image >= 50, avg keypoints/image >= 500, match and two-view
    pair counts >= max(500, images*4), nonzero match/two-view ratio >= 90%, and
    avg two-view inliers >= 30. Descriptors are not required because the current
    dense/H5-built DBs legitimately store local descriptors outside SQLite.
  - Adds a required `sfm_reconstruction_quality` handoff stage. The accepted
    sparse/dense-retriangulated model gate must report registered images >= 60,
    registered ratio >= 80%, points3D >= 30000, points/registered >= 500, mean
    reprojection error <= 2px, and all core metrics must be present.
  - Adds a required `point_cloud_quality` handoff stage. The accepted colored PLY
    must have a readable PLY header, supported ascii/binary format, vertex
    element, XYZ and RGB vertex properties, vertices >= 10000, file size >=
    128KB, binary payload size consistent with the header, vertices/points3D >=
    30%, and gate `ply_vertices` must match the PLY header when present.
  - Adds a required `report_package_quality` handoff stage. The accepted report
    package must include readable `build_report.json`, `build_config.json`, and
    `stage_times.json`; output provenance for the sparse model, RGB point cloud,
    frame manifest, intrinsics, config, stage timing, and at least one
    localization bundle output; non-empty parameters; a successful latest
    `report` timing entry; an operator Markdown report with an outputs section;
    and enough preserved `gates/*.json` files for audit.
  - `localization_bundle` now validates bundle gate metadata, not just file
    existence: refs must be positive and unique, `ref_global.shape` must be
    `[refs, 8448]`, tracking metadata must be present, and 3D anchored keypoints
    must be nonzero with mean anchors/ref >= 50.
  - Can now require external validation evidence:
    holdout localization compare JSON, production tracker replay JSON,
    `package_verify.json`, and `system_verify_latest.json`.
- Updated `建圖/pipeline/build_pipeline.py`.
  - Full builds that reach `report` now run the one-click gate summary before
    `建圖/outputs/latest_build` is updated.
  - `latest_build` is not moved when the final one-click gate fails.
  - Added final-gate forwarding flags such as `--final-gate-localization-json`,
    `--require-final-localization`, and `--require-final-system-verify`.
  - Added `--handoff-profile field|candidate`; `field` is the default and
    requires holdout localization, production replay, package verify, and system
    verify evidence before `latest_build` can move. `candidate` is reserved for
    research/parameter sweeps that are not field handoffs.
  - Candidate-profile runs no longer update the formal
    `建圖/outputs/latest_build` pointer. Passing candidate runs update
    `建圖/outputs/latest_candidate_build` instead.
  - Field-handoff production replay defaults are now stricter:
    `--final-gate-min-production-frames 30` and
    `--final-gate-min-production-inliers-p5 30`.
  - Field builds now auto-run package verify and candidate system verify when
    their JSON evidence is not supplied, then rerun the formal final summary.
    Failed summaries are not allowed to move `latest_build`, even when
    `--allow-final-gate-fail` is used for diagnostics.
  - Field builds now reserve the last input video as validation footage when
    localization/production evidence is not supplied, then auto-generate
    `<run>/holdout_localization.json` and `<run>/production_stream_replay.json`.
    The build core receives only the remaining videos; insufficient validation
    footage leaves the final handoff gate failed.
  - Field validation evidence now has a pre-core-build preflight: missing
    validation footage, empty validation files, or reserving all input videos
    blocks before expensive map construction starts.
  - Auto validation symlinks are now normalized to `.MP4` so lower-case `.mp4`
    source files are visible to `eval_stream_core.py`.
- Updated `tools/verify_package.py`.
  - `stage_gate_contract.py` is now a required package file and syntax-checked.
  - Absolute symlinks under old run/output trees are recorded as optional
    archival-output findings instead of failing the portable code package.
- Updated `tools/system_verify.py`.
  - Adds required `latest_build_field_handoff_summary` auditing for
    `建圖/outputs/latest_build/BUILD_GATE_SUMMARY.json`.
  - A formal field handoff now requires `preflight_quality`,
    `frame_motion_quality`, `intrinsics_manifest_quality`, `view_graph_quality`,
    `database_quality`, `sfm_reconstruction_quality`, `point_cloud_quality`,
    `localization_bundle`, `report_package_quality`, `holdout_localization`,
    `production_replay`, `package_verify`, and `system_verify` to be marked
    required and `PASS`/`READY`; old summaries
    with missing quality stages or external stages left as `SKIP` are flagged
    as required failures.
  - Adds `--build-summary-json` candidate mode so a new build can generate its
    own `system_verify.json` before it becomes the formal `latest_build`. This
    mode only ignores the embedded `system_verify` stage while the command is
    producing that evidence; all other build-stage failures and all other field
    handoff pending stages still hard-fail.
- Regenerated strict one-click summary for `建圖/runs/football_field_20260707`:
  - `BUILD_GATE_SUMMARY.json`: 14 PASS, 0 FAIL, 1 SKIP.
  - `holdout_localization`, `package_verify`, and `system_verify` pass.
  - `production_replay` remains skipped until a production-stream replay JSON is
    produced.
- Added `docs/mapping/ONE_CLICK_MAP_BUILD_GATES.md` with the full non-expert promotion
  contract and strict metric guidance.
- Added `建圖/pipeline/plan_db_reuse_sweep.py`.
  - Plans cheap GLOMAP backend sweeps from existing `database_merged.db` or
    `database_mvroma_forced.db`.
  - Writes JSON/Markdown command matrices under `<run>/db_reuse_sweeps`.
  - Does not rerun MV-RoMa/GlueMap.
- Generated Fuhe plan:
  - `建圖/runs/fuhe_full_no_undistort_official69_20260708/db_reuse_sweeps/db_reuse_sweep_plan.json`
  - `建圖/runs/fuhe_full_no_undistort_official69_20260708/db_reuse_sweeps/db_reuse_sweep_plan.md`
- Added `建圖/pipeline/intrinsics_holdout_gate.py`.
  - Treats `no_undistort_official69` as the main candidate only after holdout
    localization passes.
  - Missing holdout eval JSON is a failed gate, not an implicit pass.
- Generated current Fuhe gate report:
  - `docs/intrinsics_holdout_gate_20260710.json`
  - `docs/intrinsics_holdout_gate_20260710.md`
  - Current result is `FAIL`: the candidate localization bundles and holdout
    eval JSONs are missing.
- Added `建圖/pipeline/plan_research_method_experiments.py`.
  - Ordered experiment sequence: baseline GLOMAP DB reuse, LFOE, Doppelgangers++,
    Global-Aware Edge Prioritization, TriP, then GGPT.
  - Doppelgangers++ source recorded:
    `https://doppelgangers25.github.io/doppelgangers_plusplus/static/pdf/doppelgangers_plusplus.pdf`.
  - Current preflight: LFOE and Doppelgangers++ are ready locally; Global-Aware
    Edge Prioritization, TriP, and GGPT are blocked until code/adapters are
    added.
- Generated research experiment plan:
  - `建圖/experiments/research_methods_20260710/research_method_experiment_plan.json`
  - `建圖/experiments/research_methods_20260710/research_method_experiment_plan.md`
  - Doppelgangers++ front-end experiment commands now pass
    `--handoff-profile candidate` so research runs do not masquerade as
    field-deployable handoffs.

## Localization

- Added `定位/pipeline/sweep_localization_params.py`.
  - Default target is P123.
  - Sweeps `topk`, `min_conf`, and `min_inliers`.
  - Summarizes wall time, min final success, max ok-to-fail, max final fail run,
    and min final median inliers.
  - Empty eval result rows are marked as failed sweep outputs.
- Updated `定位/pipeline/localize_pipeline.py`.
  - `production-stream` now writes a quality report and can fail the stage gate.
  - Production-stream gate now fails when no frames are evaluated, success counts
    are inconsistent, or latency is requested but `wall_ms.p90` is missing.
  - Compare quality reports now fail on empty result rows.
  - Production-stream now has a fail-fast preflight for query frames, bundle,
    sparse model files, XFeat cache, and MegaLoc torch/Hugging Face cache before
    tracker model loading can trigger hidden downloads.
  - `production-stream` now honors `--out-json`, supports
    `--production-preflight-only`, and records preflight JSON separately from the
    real replay result.
  - Production-stream quality gates now include evaluated frame count,
    max consecutive failure run, and p5 inliers in addition to success rate and
    optional wall-clock latency.
  - Production preflight now rejects sparse selected-frame directories by
    checking numeric frame-index gaps with `--max-production-frame-gap`.
  - Final `production_replay` now requires the real replay JSON to have a paired
    same-stem `.preflight.json`; this prevents direct benchmark outputs from
    bypassing frame-layout and offline-cache safety checks.
  - Production-stream now accepts `--query-video`; it extracts contiguous
    `frame_000000.jpg` replay frames, writes
    `production_stream_frames_manifest.json`, and then runs the same preflight
    and replay gates. `--production-prepare-only` records the extracted frame
    manifest without running model inference.
  - New gates:
    - `--min-production-success`
    - `--max-production-wall-p90`
    - `--min-production-frames`
    - `--max-production-fail-run`
    - `--min-production-inliers-p5`
    - `--max-production-frame-gap`
    - `--query-video`
    - `--query-video-stride`
    - `--query-video-limit`
    - `--production-prepare-only`
    - `--production-preflight-only`
    - `--skip-production-preflight`
  - Latest report symlinks:
    - `定位/outputs/latest_production_stream.json`
    - `定位/outputs/latest_production_stream_quality_report.md`
- Generated production replay evidence:
  - `定位/outputs/production_stream_p124_preflight_20260710.json`
    records a preflight-only PASS for 41 direct JPG frames and complete
    XFeat/MegaLoc cached assets.
  - `定位/outputs/production_stream_p124_smoke10_20260710.json` is a real
    production-stream smoke replay on the first 10 P124 frames: 10/10 localized,
    success rate 100%, wall_ms p90 174.9 ms with latency gate disabled.
  - `建圖/runs/football_field_20260707/BUILD_GATE_SUMMARY_with_preflight_probe.*`
    verifies that preflight-only JSON is blocked as "production replay did not
    run".
  - `建圖/runs/football_field_20260707/BUILD_GATE_SUMMARY_with_production_smoke10.*`
    verifies that a real replay JSON satisfies the `production_replay` stage.
  - `建圖/runs/football_field_20260707/BUILD_GATE_SUMMARY_with_production_strict_probe.*`
    verifies that a 10-frame smoke replay is blocked when the handoff gate asks
    for at least 30 replay frames.
  - `建圖/runs/football_field_20260707/BUILD_GATE_SUMMARY_with_production_smoke10_strict_metrics.*`
    verifies the same replay passes when explicitly treated as a 10-frame smoke:
    max failure run 0 and inliers p5 57.5.
  - `定位/outputs/production_stream_p124_full41_strict_20260710.json` records a
    failed full replay on the sparse P124 geometry frame directory: 23/41
    localized, success rate 56.1%, max failure run 9, inliers p5 0.0.
  - `定位/outputs/production_stream_p124_geometry_preflight_sparse_20260710.json`
    records the corrected preflight behavior for that same directory:
    `ok=false`, max frame gap 1944, reason "query_dir is sparse, not a stream".
  - `定位/outputs/production_stream_prepare_synthetic_20260710.json` verifies
    query-video preparation on a synthetic MP4: 3 contiguous output frames from
    source frame indices 0, 2, and 4.
  - `定位/outputs/production_stream_query_video_preflight_synthetic_20260710.json`
    verifies a prepared MP4 replay directory passes preflight with max frame
    gap 1.
  - `定位/outputs/production_stream_p124_video_preflight40_20260710.json`
    verifies a real P124 MP4 can be prepared through `--query-video`: 40
    contiguous replay frames, source frame indices 0..312, max frame gap 1.
  - `定位/outputs/production_stream_p124_video_replay40_20260710.json`
    records a real production tracker replay from the same P124 MP4 via
    `--query-video`: 40/40 localized, success rate 100%, max failure run 0,
    inliers p5 50.35, wall_ms p90 96.32 ms. The paired preflight JSON records
    40 direct JPG replay frames, numeric frame count 40, max frame gap 1, and
    complete XFeat/MegaLoc cached assets.
  - `建圖/runs/fuhe_full_no_undistort_official69_20260708/BUILD_GATE_SUMMARY_with_p124_video_replay40_20260710.*`
    verifies the final gate contract marks `production_replay` as PASS for the
    replay40 JSON under strict thresholds (`frames >= 30`, `success >= 90%`,
    `max failure run <= 30`, `inliers p5 >= 30`) and includes paired-preflight
    metrics (`direct_jpg_count=40`, `numeric_frame_count=40`, `max_frame_gap=1`).
    The overall summary remains FAIL for the old
    `fuhe_full_no_undistort_official69_20260708` build run because that run dir
    lacks deploy bundle gate evidence; this does not contradict the replay
    stage result.
  - These are gate-plumbing and smoke artifacts, not a full field promotion;
    full handoff still requires same-field holdout localization, package verify,
    and system verify evidence.
  - Verification after the change:
    - `pytest -q 建圖/pipeline/test_stage_gate_contract.py 建圖/pipeline/test_intrinsics_holdout_gate.py 建圖/pipeline/test_plan_research_method_experiments.py 定位/pipeline/test_localization_gates.py 更新地圖/source/sfm_reshot25/update_pipeline/test_update_quality_gates.py tools/test_verify_package.py tools/test_system_verify.py` -> 93 passed.
    - `python3 -m py_compile 建圖/pipeline/stage_gate_contract.py 建圖/pipeline/build_pipeline.py tools/system_verify.py tools/test_system_verify.py` -> pass.
    - `python3 tools/verify_package.py --allow-extra-top --json-out package_verify.json` -> pass.
    - Temporary football-field summary written outside the repo at
      `/tmp/football_field_gate_check_preflight_20260710.json` verifies the
      real `preflight`, `manifest`, `extract`, `pairs`, `db`, `glomap`, and `color` gates
      pass `preflight_quality`, `intrinsics_manifest_quality`, `frame_motion_quality`,
      `view_graph_quality`, `database_quality`, `sfm_reconstruction_quality`,
      and `point_cloud_quality`: 3 videos, 256.84s total video duration, minimum
      expected extracted frames 120, 1766GB free disk, GPU ok, 262 manifest
      frames, 1920x1080 FULL_OPENCV intrinsics, 262 kept frames, 2046 pairs,
      largest component ratio 1.0,
      DB images/keypoint rows 262/262, 2011 matches, 2011 two-view geometries,
      two-view nonzero ratio 0.994, avg two-view inliers 6174.3, 262 registered
      images, 659024 points3D, 0.879px mean reprojection, and a binary RGB PLY
      with 659024 vertices, 9.89MB, XYZ/RGB properties, and
      vertices/points3D=1.0. This was not promoted over the formal latest
      summary.
    - The same temporary summary verifies `report_package_quality` passes on the
      real `football_field_20260707` run: `build_report.json`,
      `build_config.json`, `stage_times.json`, and
      `BUILD_LOCALIZABLE_MAP_REPORT.md` are readable; output provenance covers
      `glomap_model`, `rgb_point_cloud`, `frame_manifest`, `intrinsics`,
      `config`, `stage_times`, and the triangulated/tracking/snap bundles; 15
      gate JSONs are preserved; latest `report` status is `success`.
      Historical failed attempts in `stage_times.json` are recorded as metrics,
      not treated as current failure when the latest report stage succeeded.
    - `python3 tools/system_verify.py --skip-runtime --allow-blocked` -> FAIL as intended for the current formal pointer:
      `latest_build_field_handoff_summary` fails because
      `建圖/runs/football_field_20260707/BUILD_GATE_SUMMARY.json` still has
      no `preflight_quality` stage, no `frame_motion_quality` stage, no
      `intrinsics_manifest_quality` stage, no `view_graph_quality` stage, no
      `database_quality` stage, no `sfm_reconstruction_quality` stage, no
      `point_cloud_quality` stage, no `report_package_quality` stage, and still
      has `production_replay` as `SKIP` and `required=false`. Sphinx runtime
      remains optional BLOCKED by Secure Boot.

## Update Map

- Added `更新地圖/source/sfm_reshot25/update_pipeline/update_quality_gates.py`.
  - Centralizes quarantine warning parsing and bridge-quality warning logic.
- Updated `map_update_tool.py`.
  - `retrieval_high_but_inliers_low` can now be a report-only quarantine or a
    hard skip:
    - `--quarantine-warnings retrieval_high_but_inliers_low`
    - `--quarantine-action report|skip`
  - Bridge quality can now hard-gate geometry evidence:
    - `--bridge-min-geometry`
    - `--bridge-min-geometry-ratio`
    - existing `--bridge-gate-quality`
  - Submap hard-gate failures roll back connector keyframes from the same
    segment before writing the updated bundle.
  - Forced `changed`/`tile_replace` routes record observation frames so
    multi-session evidence is not lost.
- Added `changed_region_evidence.py`.
  - Aggregates multiple `observation_stats.json` files.
  - Promotes changed-region candidates only after multi-session support.
  - Does not invalidate or delete old points.
  - Default grouping is `seq_tile` to avoid merging unrelated sessions by image
    grid tile alone; cross-session `tile` grouping must be explicit.

## Deployment

- Updated `更新地圖/pipeline/update_pipeline.py`.
  - `--sparsify` now creates a separate slim bundle candidate.
  - Slim bundle must pass metadata/covis/global-descriptor inspection.
  - If stage gates are enabled, `--sparsify` requires `--validate-dir` so the
    slim bundle also passes localization gate before it can be treated as
    deployable.
  - `latest_update` is now updated only after validation and slim gates pass.
  - Bundle inspection now checks finite MegaLoc `ref_global` descriptor shape.

## Data Governance

- Added `tools/sfm_retention_report.py`.
  - Non-destructive scan only.
  - Protects raw inputs, production/current symlink targets, DB/H5/NPZ retuning
    artifacts, and canonical reports/configs/gates.
  - Marks old build/update runs as `prune_to_canonical`, not delete.
  - The scanner now descends into old run directories so DB/H5/NPZ and gate
    files inside them are listed as protected artifacts.
- Generated retention report:
  - `docs/retention_report_20260710_011030.json`
  - `docs/retention_report_20260710_011030.md`

## Suggested Next Commands

Plan cheap GLOMAP backend sweeps:

```bash
python3 建圖/pipeline/plan_db_reuse_sweep.py \
  --run-dir 建圖/runs/fuhe_full_no_undistort_official69_20260708
```

Regenerate the research-method experiment plan:

```bash
python3 建圖/pipeline/plan_research_method_experiments.py \
  --run-dir 建圖/runs/fuhe_full_no_undistort_official69_20260708
```

Run the planned Doppelgangers++ front-end experiment only after baseline and
LFOE measurements are recorded:

```bash
/usr/bin/python3.12 建圖/pipeline/build_pipeline.py \
  --site-name fuhe_dgpp_experiment_20260710 \
  --work-dir 建圖/experiments/research_methods_20260710/02_doppelgangers_pp_frontend \
  --image-root 建圖/runs/fuhe_full_no_undistort_official69_20260708/images \
  --resume \
  --doppelgangers-root 建圖/external_tools/doppelgangers-plusplus \
  --doppelgangers-checkpoint 建圖/external_tools/doppelgangers-plusplus/checkpoints/checkpoint-dg+visym.pth \
  --doppelgangers-threshold 0.7 \
  --doppelgangers-filter-scope cross_video
```

Run P123 localization parameter sweep:

```bash
python3 定位/pipeline/sweep_localization_params.py \
  --include-set P1230123 \
  --topk 20,30,40,50 \
  --min-conf 0.05,0.1,0.15 \
  --min-inliers 40,50,60
```

Run production tracker replay as a gate:

```bash
python3 定位/pipeline/localize_pipeline.py \
  --mode production-stream \
  --query-video <validation_video.MP4> \
  --query-video-stride 8 \
  --query-video-limit 300 \
  --production-preflight-only \
  --out-json 定位/outputs/production_stream_preflight.json

python3 定位/pipeline/localize_pipeline.py \
  --mode production-stream \
  --query-video <validation_video.MP4> \
  --query-video-stride 8 \
  --query-video-limit 300 \
  --min-production-success 0.90 \
  --min-production-frames 30 \
  --max-production-fail-run 30 \
  --min-production-inliers-p5 30 \
  --max-production-frame-gap 60 \
  --max-production-wall-p90 0 \
  --out-json 定位/outputs/production_stream_replay.json
```

Use update quarantine as a hard gate:

```bash
/usr/bin/python3.12 更新地圖/pipeline/update_pipeline.py \
  --frames-root <prepared_geometry_frames> \
  --quarantine-action skip \
  --bridge-gate-quality \
  --bridge-min-geometry 2 \
  --bridge-min-geometry-ratio 0.5
```

Aggregate changed-region evidence:

```bash
python3 更新地圖/source/sfm_reshot25/update_pipeline/changed_region_evidence.py \
  <update1>/observation_stats.json <update2>/observation_stats.json \
  --min-sessions 2
```

Generate a slim bundle candidate with localization gate:

```bash
/usr/bin/python3.12 更新地圖/pipeline/update_pipeline.py \
  --frames-root <prepared_geometry_frames> \
  --validate-dir data/target_site/updates/test \
  --sparsify \
  --sparsify-keep-prefix P1230123 \
  --sparsify-keep-prefix P1260126
```

Regenerate the non-destructive retention report:

```bash
python3 tools/sfm_retention_report.py --root .
```
