# MapDoctor QA integration

This repository can use [MapDoctor](https://github.com/1122-gggggg/diagnosis_map) as an **optional, read-only post-build QA layer** for final sparse maps.

MapDoctor does not replace the S0–S9 target-site release contract. In particular, S9 held-out localization remains the authoritative project-specific release gate. MapDoctor adds a reusable cross-project view of sparse-map health and, once per-frame localization results are exported, a common held-out benchmark/regression format.

## Static map QA

After S5/S6 produces and validates the final sparse reconstruction:

```bash
python sites/target_site/tools/run_mapdoctor_qa.py \
  --model /path/to/runs/target_site_v1/final_model \
  --backend gluemap \
  --output /path/to/runs/target_site_v1/mapdoctor
```

For a GLOMAP-built candidate, use `--backend glomap`; COLMAP candidates use `--backend colmap`.

The hook writes MapDoctor's JSON, CSV, and HTML reports without modifying the reconstruction. Useful screening fields include reference-image observation support, track length, reprojection statistics, image-space landmark coverage, covisibility connectivity, and weak-reference recapture suggestions.

## Relationship to S9

`sites/target_site/tools/validate_heldout_localization.py` currently consumes aggregate per-video results such as localization rate, inlier p05, continuity, and reference-sequence support. Those aggregates are not equivalent to MapDoctor's per-query benchmark schema and must not be converted by inventing missing frame-level fields.

A proper integration should export the raw per-frame pose-estimation evidence before S9 aggregation. The desired fields are documented in MapDoctor's `docs/BENCHMARK_SCHEMA.md`:

```text
query,success,inliers,inlier_ratio,reproj_p90_px,hull_coverage,
grid4_occupancy,positive_depth_ratio,pose_consensus
```

Optional query positions can be added for weak-region aggregation.

## Map-update regression

Once both a frozen base map and candidate map produce per-frame MapDoctor benchmark rows for the same held-out queries:

```bash
mapdoctor compare base.csv candidate.csv --output comparison_report
```

This directly complements this repository's existing promotion philosophy: a candidate should not be promoted merely because an intermediate proxy metric improves if it creates newly failed held-out queries.

## Installation

Until the package is published to PyPI, install from the MapDoctor repository:

```bash
git clone https://github.com/1122-gggggg/diagnosis_map.git
pip install -e ./diagnosis_map
```

After a PyPI release, the intended installation is `pip install mapdoctor-sfm`.

## Scope boundary

The integration is intentionally optional. Failure to install or run MapDoctor does not silently alter the S0–S9 workflow, map geometry, EDM bundle, camera intrinsics, or promotion state. Promotion decisions continue to require the existing project gates unless the release contract is explicitly revised.
