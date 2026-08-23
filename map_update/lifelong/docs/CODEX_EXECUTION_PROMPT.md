# Codex execution prompt

Copy the following task into Codex after cloning this repository and setting the real data paths.

---

## Mission

Use the existing newest-data GLUEMAP reconstruction as an immutable privileged geometry and use historical sessions only to add still-valid EDM viewpoints/observations. Execute the complete E0–E5 and A1–A11 protocol. Do not merely write a design document.

Set:

```text
REPO_ROOT=<this repository>
BASE_MAP_DIR=<latest GLUEMAP/COLMAP sparse model>
CURRENT_IMAGES_DIR=<RGB files referenced by images.bin/txt>
HISTORICAL_UPDATE_DATA_DIR=<older sessions>
CURRENT_VALIDATION_DATA_DIR=<held-out latest sessions>
EDM_BUNDLE_OR_PIPELINE=<existing EDM/retrieval implementation or precomputed output>
```

## Non-negotiable invariants

- Base current poses, point coordinates, intrinsics, coordinate frame and IDs are immutable.
- Calculate and compare hashes before and after every run.
- Never promote old-only triangulated geometry.
- Never use `VIRTUAL_BA_ONLY` points for PnP.
- Never allow changed or uncertain historical pixels into PnP.
- Count unique `point3D_id` inliers, not duplicated observations.
- Reject competitive pose modes; temporal continuity alone cannot choose a mode.
- A bridge requires multiple anchors or approximately independent paths plus cycle consistency.
- Run change detection again after bridge registration.
- Do not promote a candidate without held-out current-session regression.
- Preserve the 5% common-success inlier non-regression gate; do not loosen gates to improve coverage.

## Work order

1. Audit repository and data; document reusable EDM/GlueMap adapters and any duplicated metrics.
2. Reproduce E0 current-only localization with all per-query diagnostics.
3. Build historical manifest, quality filtering and session keyframes.
4. Direct historical→current retrieval, EDM matching, safe lifting, per-reference PnP and pose clustering.
5. For direct-strong images, run pose-aligned multi-view change localization and create stable/change/uncertain masks.
6. Export only stable historical-pixel→current-point observations into a candidate sidecar.
7. Classify direct failures into bad image, alias, historical change, viewpoint gap and unresolved.
8. For viewpoint gaps, construct a bridge graph. First propagate current point IDs and re-run absolute PnP. If support ends, build a quarantined old-view submap, estimate robust Sim(3), and optimize only historical variables against fixed current anchors.
9. Require multi-anchor/path and cycle evidence. Re-run change detection for every bridged image.
10. Evaluate reference utility on held-out current queries using route-cell EDM success, FIM, stable ratio, redundancy, runtime and risk.
11. Run E0–E5 and A1–A11. Compare success, pose error, confident wrong poses, weak-cell coverage, maximum continuous failure, p95 latency and map size.
12. Produce a versioned candidate sidecar and rollback manifest. Do not merge automatically.

## Required outputs

```text
reports/code_audit.md
reports/data_audit.md
reports/baseline_report.md
reports/direct_registration_report.md
reports/change_detection_report.md
reports/bridge_recovery_report.md
reports/reference_selection_report.md
reports/ablation_report.md
reports/FINAL_REPORT.md

manifests/historical_images.csv
manifests/direct_registration_results.parquet
manifests/change_detection_results.parquet
manifests/bridge_edges.parquet
manifests/reference_candidates.parquet
manifests/active_historical_references.json
manifests/rejected_historical_references.json
manifests/route_cell_metrics.csv
manifests/rollback_manifest.json

output/historical_reference_sidecar/
output/stable_masks/
```

`FINAL_REPORT.md` must state how many images were direct-registered, bridge-recovered, rejected by quality/change/alias, selected as active, removed as redundant, and how route failure topology, accuracy, false accepts, latency and map size changed. It must identify remaining areas that require new current-data capture.

## Initial commands

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp configs/default.yaml configs/local.yaml
# Fill all paths and adapter settings.
update-map validate-config --config configs/local.yaml --require-paths
update-map audit --config configs/local.yaml --output runs/audit
update-map run-direct --config configs/local.yaml --output runs/direct
pytest
```

Implement missing integration code through the existing adapter and module boundaries. Add regression tests before changing a formula or gate. Commit each completed phase separately and leave the work on a feature branch for review.
