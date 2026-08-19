# sfm-map-diagnosis

MapDoctor (`mapdoctor`) and sfm-diagnosis (`sfm_diagnosis`) now live in this repo.

本仓库用一条主命令先诊断地图，再诊断后续视觉定位。

Pipeline: **Stage 0 session selection → Stage 1 map diagnosis → Stage 2 SfM localization**.

This tool does **not** run a localizer. `--logs` is a MapDoctor-schema localization CSV.

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

Optional for pose diagnosis: `x,y,z`.

Exit code 0 only when `overall_status` is `READY` or `MAP_SCREENED_LOCALIZATION_UNCHECKED`.

`--output DIR` writes `DIR/map/report.json`, `DIR/sfm/report.json` (if `--logs`), and combined `DIR/report.json`.

## Upstream CLIs

The original entry points still work:

```bash
mapdoctor --help
sfm-diagnosis --help
```

## overall_status

| Status | Meaning |
| --- | --- |
| `READY` | Map screening passed and every query passed MapDoctor localization gates |
| `MAP_SCREENED_LOCALIZATION_UNCHECKED` | Map screening passed; no logs given |
| `MAP_SCREENING_FAILED` | Static map checks failed |
| `LOCALIZATION_FAILED` | Map screening passed; at least one query failed gates |
| `BOTH_FAILED` | Map screening failed and at least one query failed gates |

## Demo

```bash
python examples/reproducible_demo/generate_demo.py --output /tmp/mapdoctor-demo
sfm-qa analyze /tmp/mapdoctor-demo/sparse/0 --backend gluemap --output /tmp/qa-out
```

See `docs/pipeline.md` for the Stage 0–2 contract.
