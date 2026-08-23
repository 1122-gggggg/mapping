# Experiment protocol

## 1. Data split

Use three non-overlapping roles:

1. `CURRENT_MAP`: newest data used to build the privileged current map.
2. `HISTORICAL_UPDATE`: older sessions used only to propose observations/references.
3. `CURRENT_VALIDATION`: newest held-out sessions used to measure whether historical views improve present-day localization.

Split by flight/session/day. Random adjacent-frame splitting is invalid because it strongly leaks appearance, pose and motion continuity.

If no independent current validation session exists, label every result `PROVISIONAL_PROXY_VALIDATION`.

## 2. Reproducibility manifest

Every run should save:

```text
run_config.yaml
environment.txt
git_commit.txt
input_manifest.json
base_map_hashes.json
model_versions.json
random_seeds.json
```

Cache retrieval, pair matches, lifting results and masks. Changing the localizer, image resizing, intrinsics, retrieval top-K or pose settings creates a new run ID.

## 3. Main experiments

### E0_BASE_CURRENT_ONLY

Original current map and current localizer references. This is the immutable baseline.

### E1_DIRECT_NO_CHANGE_MASK

Add directly registered historical references without change filtering. This intentionally unsafe condition quantifies how much apparent gain and false-pose risk come from ignoring historical change. Offline only.

### E2_DIRECT_CHANGE_AWARE

E1 plus stable-mask filtering. It isolates the contribution of change detection.

### E3_DIRECT_VERIFIED_BRIDGE

E2 plus historical references recovered by a multi-anchor or multi-path bridge and rechecked for change after registration.

### E4_SELECTED_AUGMENTED

E3 with localizer/FIM/K-cover utility selection and redundancy pruning.

### E5_PRODUCTION_CANDIDATE

E4 with current-first historical-on-demand retrieval, source-aware weighted PnP and fail-closed pose multimodality.

## 4. Ablations

| ID | Comparison |
|---|---|
| A1 | no historical references |
| A2 | direct historical references only |
| A3 | direct + change mask |
| A4 | direct + bridge without strict multi-anchor gate; offline only |
| A5 | direct + verified multi-anchor bridge |
| A6 | all candidates vs utility-selected candidates |
| A7 | pooled retrieval vs current-first fallback |
| A8 | unweighted vs confidence/source-aware PnP |
| A9 | no stable-mask filtering vs stable-mask filtering |
| A10 | no FIM utility vs FIM + localizer front-end utility |
| A11 | unmatched-decay vs conflict-only stability updates |

## 5. Metrics

### Pose accuracy

When ground truth exists:

- success at 0.25 m / 2°;
- success at 0.5 m / 5°;
- success at 1.0 m / 10°;
- translation median/p90/p95;
- rotation median/p90/p95.

### Coverage and failure topology

- failed query count;
- failed route-cell count;
- worst-decile route-cell success;
- maximum consecutive failed frames;
- maximum continuous failed route length;
- direct recovery count;
- verified bridge recovery count.

### Pose diagnostics

- unique point3D inliers;
- inlier ratio;
- reprojection p50/p90/p95;
- convex hull;
- 4×4 occupancy;
- positive-depth ratio;
- independent current-reference support;
- pose mode count;
- FIM eigenvalues and covariance;
- leave-one-reference-out stability.

### Safety

- confident wrong pose count;
- new false rejection on healthy baseline queries;
- changed/uncertain pixels entering PnP;
- single-anchor bridge promotion;
- old-only point promotion;
- base-map hash mutation.

### Systems

- retrieval/matching/pose/total p50/p95/p99 latency;
- GPU and host memory;
- disk size;
- current, candidate and active reference counts;
- match pairs per query.

## 6. Promotion gates

Default hard gates:

1. Base-map snapshot is unchanged.
2. No old-only production landmark.
3. No virtual BA-only PnP landmark.
4. Zero new confident wrong poses.
5. Zero new false rejection on healthy common-success queries.
6. Aggregate common-success inlier drop does not exceed 5%.
7. Worst cells or maximum failure run improve by the configured amount.
8. p95 latency and memory stay within budget.
9. Every bridged active reference has multi-anchor/path and cycle evidence.
10. Every historical active reference has a stable mask and provenance.

A mean success-rate increase cannot override a failed safety gate.
