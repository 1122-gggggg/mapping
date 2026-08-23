# Map + localization diagnosis

`diagnosis/` is the former `sfm_map_diagnosis` tree. It lives in this repo.
Install from the mapping root:

```bash
pip install -e '.[dev]'
```

Optional extras: `[colmap]`, `[viz]`, `[video]` (Stage 0 video ingest).

This layer is **read-only**. It does not replace the S0–S9 target-site release
contract. S9 held-out localization remains the authoritative project-specific
release gate.

## How the two halves complement each other

| Build (this repo, `sites/`) | Diagnosis (`diagnosis/`) |
|---|---|
| S0 corpus lock (held-out must not leak) | Stage 0 `sfm-qa select-sessions` — advisory role assignment before first SfM |
| S5/S6 final sparse model | Stage 1 `sfm-qa analyze` — MapDoctor health + weak-region screen |
| S9 aggregate held-out rate | Stage 2 `--logs` — per-query attribution. Needs MapDoctor-schema CSV, not S9 aggregates |
| `map_update` promotion | `mapdoctor compare` on the same held-out queries |

S0 and Stage 0 are complementary, not substitutes. S0 is the hard split gate.
Stage 0 is advisory session-role advice.

## After S5/S6: screen the reconstruction

Preferred (MapDoctor + sfm-diagnosis, combined `report.json`):

```bash
python tools/diagnose_map.py \
  --model /path/to/runs/target_site_v1/final_model \
  --backend gluemap \
  --output /path/to/runs/target_site_v1/sfm-qa
```

MapDoctor HTML/CSV only (target_site hook):

```bash
python sites/target_site/tools/run_mapdoctor_qa.py \
  --model /path/to/runs/target_site_v1/final_model \
  --backend gluemap \
  --output /path/to/runs/target_site_v1/mapdoctor
```

`--backend` is required at the `sfm-qa` / `mapdoctor` CLIs: `colmap`, `glomap`, or `gluemap`.

## After S9: attribute localization (optional)

`validate_heldout_localization.py` consumes aggregate per-video results. Those
aggregates are not equivalent to MapDoctor's per-query schema and must not be
converted by inventing missing frame-level fields.

When a real per-frame CSV exists:

```bash
python tools/diagnose_map.py \
  --model /path/to/runs/target_site_v1/final_model \
  --backend gluemap \
  --logs loc.csv \
  --output /path/to/runs/target_site_v1/sfm-qa
```

Required columns:

```text
query,success,inliers,inlier_ratio,reproj_p90_px,hull_coverage,
grid4_occupancy,positive_depth_ratio,pose_consensus
```

Optional `x,y,z` enable pose diagnosis.

## Map-update regression

```bash
mapdoctor compare base.csv candidate.csv --output comparison_report
```

A candidate must not be promoted merely because an intermediate proxy improves
if it creates newly failed held-out queries.

## Scope boundary

Failure to install or run diagnosis does not silently alter the S0–S9 workflow,
map geometry, EDM bundle, camera intrinsics, or promotion state.
