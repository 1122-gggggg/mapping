# sfm-map-diagnosis

This tree now lives inside https://github.com/1122-gggggg/mapping under `diagnosis/`.
Install from the mapping repo root: `pip install -e '.[dev]'`.

MapDoctor (`mapdoctor`) and sfm-diagnosis (`sfm_diagnosis`) provide the diagnosis half.
本仓库用一条主命令先诊断地图，再诊断后续视觉定位。

Pipeline: **Stage 0 session selection → Stage 1 map diagnosis → Stage 2 SfM localization**.

This tool does **not** run a localizer. `--logs` is a MapDoctor-schema localization CSV,
and its held-out provenance is not verified by this command; S0/S9 hashes remain the
release authority.

## Install

```bash
pip install -e '.[dev]'
```

Optional extras: `[colmap]` (`pycolmap`), `[viz]` (`plotly`), `[video]` (`opencv-python-headless`, Stage 0 ingest).

## Stage 0: select sessions

Assign roles before the first SfM. Advisory: writes reports and exits 0.

```bash
sfm-qa select-sessions --videos /path/to/videos --output ./qa-select
# optional: --maps /path/to/maps --config config/session_select.yaml
```

See `docs/session_selection.md`, `docs/frozen_core.md`, and `docs/method_lessons.md`.
The default proposal policy is cohort-relative and always keeps a best-available readable
geometry-probe set; heuristic QA references do not authorize or forbid an SfM merge.

## Analyze

```bash
# Stage 1 only: screen the reconstruction
sfm-qa analyze /path/to/sparse/0 --backend gluemap --output ./qa-out

# Stage 1 + Stage 2: map screen, then attribute localization logs
sfm-qa analyze /path/to/sparse/0 --backend colmap --logs loc.csv --output ./qa-out

# Optional build evidence for a richer map stage
sfm-qa analyze /path/to/sparse/0 --backend colmap \
  --database database.db --pairs pairs.csv \
  --images-manifest images.csv --images-dir /path/to/images \
  --output ./qa-out
```

`--backend` is required: `colmap`, `glomap`, or `gluemap`.

Compatibility aliases: `check`, `check-map`, `check-localize`.

`--logs` CSV required columns:

`query,success,inliers,inlier_ratio,reproj_p90_px,hull_coverage,grid4_occupancy,positive_depth_ratio,pose_consensus`

All columns are required. Only `reproj_p90_px` may have an empty value; the remaining
quality fields must be finite and in their documented ranges.

Optional for pose diagnosis: `x,y,z`.

Exit code 0 when `overall_status` is `READY`, `READY_WITH_MAP_WARNINGS`, or
`MAP_SCREENED_LOCALIZATION_UNCHECKED`.

`--output DIR` writes `DIR/map/report.json`, `DIR/sfm/report.json` (if `--logs`), and combined `DIR/report.json`.

## Calibrated failure risk

Raw health/risk scores are ranking signals until validated against held-out localization
outcomes. Cross-fit calibration by independent flight/session whenever possible:

```bash
mapdoctor calibrate-risk loc.csv raw_risk.json \
  --groups session_groups.json --folds 5 \
  --output calibration.json \
  --scores-output calibrated_oof_risk.json

mapdoctor risk-coverage loc.csv calibrated_oof_risk.json \
  --ece-binning equal_mass --confidence 0.95 \
  --target-failure-rate 0.01 0.02 0.05
```

`calibrated_oof_risk.json` is the correct file for evaluating the current dataset. The
`final_calibrator` in `calibration.json` is for future untouched queries:

```bash
mapdoctor apply-risk-calibrator calibration.json future_raw_risk.json \
  --output future_calibrated_risk.json
```

Final deployment threshold selection requires a separate certification set; adjacent
video frames must not be treated as independent trials.

## Graph fragility

```bash
mapdoctor graph-fragility /path/to/sparse/0 --backend gluemap \
  --minimum-shared-landmarks 15 --output graph.json
```

In addition to exact articulation images and bridge edges, the report includes weighted
normalized-Laplacian `lambda2` for soft bottlenecks and a threshold-sensitivity profile.
Shared-landmark counts are computed by blockwise sparse incidence multiplication rather
than Python pair expansion over every track.

See [`docs/diagnostic_reliability.md`](docs/diagnostic_reliability.md) for the mathematics,
validation protocol, assumptions, literature basis, and stronger alternatives.

## Upstream CLIs

The original entry points still work:

```bash
mapdoctor --help
sfm-diagnosis --help
```

## overall_status

| Status | Meaning |
| --- | --- |
| `READY` | Map screening passed and provided-log strict success reached the configured aggregate target (default 95%); release still requires external held-out proof |
| `READY_WITH_MAP_WARNINGS` | Provided localization reached its target; map integrity passed, but one or more advisory map-health heuristics did not |
| `MAP_SCREENED_LOCALIZATION_UNCHECKED` | Map screening passed; no logs given |
| `MAP_SCREENING_FAILED` | Static map checks failed |
| `LOCALIZATION_FAILED` | Map screening passed; provided-log strict success missed its aggregate target |
| `BOTH_FAILED` | Map screening and provided-log aggregate localization target both failed |

## Demo

```bash
python examples/reproducible_demo/generate_demo.py --output /tmp/mapdoctor-demo
sfm-qa analyze /tmp/mapdoctor-demo/sparse/0 --backend gluemap --output /tmp/qa-out
```

See `docs/pipeline.md` for the Stage 0–2 contract.
