# River v7 decision-gate audit — 2026-08-24

This is an adversarial audit of the v7 product and the generic mapping
session-select / risk-PLY contract after the bidirectional sync. It does
**not** retune gates to preserve last-known-good labels.

Site-absolute paths below are evidence locations. The generic contract
does not depend on them.

## Artifacts opened

| Artifact | Used as |
| --- | --- |
| `outputs/final_map/FINAL_MAP.md` | product claim |
| `outputs/session_selection_staged_optimized_v7_20260823/{selection_receipt,optimized_data_selection,session_edges,session_roles,session_quality}.json/csv` | selector decision |
| `runs/river_core_map_v7_20260823/maps/map_gluemap_baseline/b0_quality.json` | B0 gate + thresholds |
| `runs/P157_V7_FUSION_ANALYSIS_20260824.md` + `runs/river_p157_v7_bridge_analysis_20260824/REPORT.json` | P157 fusion |
| `runs/CORE_MAP_V7_AND_P167_BRIDGE_RESULT_20260824.md` + `runs/river_p167_bridge_audit_v7_20260824/REPORT.json` | P167 bridge |
| `runs/RIVER_CORE_SELECTION_OPTIMIZED_V7_20260823.md` | last-known-good writeup |
| `runs/river_v7_heldout_localization_20260824/REPORT.json` | held-out loc |
| `runs/river_v7_p116_only_fringe_20260824/REPORT.json` | P116 child |
| mapping `diagnosis/src/sfm_qa/session_select/{types,edges,admission,defaults.yaml}` | post-sync gates |
| mapping `diagnosis/src/sfm_diagnosis/{diagnose.py,risk_ply.py}` | FIM / loc thresholds |

Not opened as authority: deleted 8/18 B0+P157 product bins; historical fringe
track counts; MV-RoMa appearance matches.

## Direct vs inferred

Direct:

- Selector `admission_state=HISTORICAL_BEST`, `stop_reason=last_known_good`,
  `exact_pair_probe_present=false`.
- Verified session graph: P119–P120, P118–P119, P118–P120 are `WEAK` with
  `shared_reconstruction_not_independent_geometry`. No `STRONG`/`USABLE` edge.
- FINAL_MAP: 418/418 registered, 237907 points, P90 2.160 px, positive depth 1.0,
  159 bridge-only images with 0 observations. Core trio isolated on the
  *verified* session graph; connectivity is GlueMap-internal.
- P157: 0/73 SIFT geometry; PnP grid median 2/16 (need 6); 0 admitted anchors;
  relaxed Sim3 3/13 folds, residual P50 15.2% vs 5% cap. `NO_GO_LOCAL_FUSION`.
- P167: 275 VPR → 0 geometry survivors. `NO_GO` / `SUBMAP_ONLY`.
- Held-out (strict inliers≥80, ratio≥0.25, hull≥0.15, grid≥6/16, depth≥0.99,
  reproj P90≤3): P116 38/94 +17 provisional; P117 0/52; P157 7/129 +37
  provisional; P167 0/97 +2 provisional.
- P116 child: 38 images / 24423 points, core invariance true, **not promoted**.

Inferred (not evidence):

- That GlueMap-internal clusters (11–13) equal independent bridges.
- That last-known-good `GLOBAL_BA` is independently certified fusion.
- That FIM logdet / λmin predicts localization success.

## Threshold inventory

| Threshold | Where | Class | Must not be called empirical/deployment gate? |
| --- | --- | --- | --- |
| exact-pair scope; independent artifact; group-disjoint fit/holdout; complete finite geometry | mapping `defaults.yaml` `edge.require_*` + `classify_session_edge` | **data-integrity invariant** | No. These are fail-closed identity rules. |
| VPR is not an edge | mapping + river session graph | **data-integrity invariant** | No. |
| B0 registration ≥0.95 | `b0_quality.json` thresholds | **engineering heuristic** used as a map-internal integrity floor | Yes, if marketed as localization readiness. 418/418 is a reconstruction count, not held-out success. |
| B0 positive depth ≥0.99 | same | **engineering heuristic** / near-invariant for cheirality | Borderline. 1.0 here is a reconstruction check, not query success. |
| B0 reproj P90 ≤5 px | same | **engineering heuristic** | Yes. 2.16 px passed; the 5 px cap is unlabeled. |
| B0 cross-video clusters ≥2 | same | **currently uncalibrated** if read as independent bridges | **Yes.** Shared-reconstruction clusters self-certify. Post-sync mapping treats them as `WEAK`. |
| Strict loc: inliers≥80, ratio≥0.25, hull≥0.15, occ≥6/16, depth≥0.99, reproj P90≤3 | held-out REPORT + runner (read-only) | **engineering heuristic** / site operating point | **Yes**, until calibrated on a frozen held-out set with ECE/reliability. |
| P157 PnP occupancy 6/16 | P157 REPORT | **engineering heuristic** | Yes as a probability gate. The NO_GO is still supported because max was 4/16 and Sim3 LOO failed. |
| P157 Sim3 all LOO residual ≤5% | P157 REPORT | **engineering heuristic** | Yes as calibrated accuracy. Observed 15–35% still fails any reasonable residual cap. |
| P116 ≥30 strict, ≥2 ten-second bins, obs in ≥2 core sessions | P116 REPORT | **engineering heuristic** | Yes as a promotion gate. Child was admitted as child and **not** promoted; that part is consistent. |
| mapping `edge.min_cross_tracks_for_verified=30`, `min_verified_pairs_for_usable=8`, consensus/parallax/holdout soft targets | `defaults.yaml` | **engineering heuristic** / some **cohort-relative soft targets** | **Yes.** They never authorize STRONG/USABLE alone after this sync. |
| mapping `DiagnosticThresholds.min_pnp_inliers=25` vs river strict 80 | `diagnose.py` vs loc runner | **uncalibrated inconsistency** | **Yes.** Do not mix the two as one “strict” number. |
| FIM λmin / logdet / condition | selector + old risk map | **observability proxy** | **Yes.** Not success probability. ActLoc proxy is shadow-only. |

## Role / fusion verdicts

| Decision | Verdict | Why |
| --- | --- | --- |
| P119/P120 `BASE_CORE` | **CORRECT BUT OVERSTATED** | Internally strongest hard-valid videos and the operational product. Selector itself says last-known-good; verified graph is disconnected. Not independent-edge certified. |
| P118 `BASE_SUPPORT` + `GLOBAL_BA` | **HEURISTIC-ONLY** | Useful support in the shared reconstruction. Flagged rotation-dominated + ill-conditioned sampled FIM. `GLOBAL_BA` is role-derived admission, not independent fusion. |
| P157 `GEOMETRY_REINFORCEMENT` / `LOCAL_RELATION_ONLY` / `NO_GO_LOCAL_FUSION` | **CORRECT** | New v7 exact-pair evidence fails. Historical 75 DIRECT_STRONG / 18449 fringe points are different model bytes. |
| P167 `NEW_SUBMAP` / `SUBMAP_ONLY` / `NO_GO` | **CORRECT** | 0/275 geometry. High raw matches are not a bridge. |
| P116/P117 `QUARANTINE` as core | **CORRECT** | Contaminated contrast maps; not core geometry. |
| P116 child `ADMITTED_CHILD`, not promoted | **CORRECT** | 38≥30 strict, ≥2 time bins, obs on P118/P119/P120, core invariance true. Promotion withheld is the honest part. |
| P117 as deployable loc | **UNSUPPORTED** | 0/52 strict. |
| P157 loc 7/129 as fusion support | **HEURISTIC-ONLY** | Localization ≠ independent Sim3. Still `LOCAL_RELATION_ONLY`. |
| Old risk map 131/45/8 FIM spheres as calibrated risk | **UNSUPPORTED** | Duplicate FIM, relative 10% cuts, no held-out calibration. Replaced by mapping `risk-ply`. |

## Adversarial challenges

1. **Can same-map tracks self-certify bridges?** Before this sync, mapping
   `trusted_geometry + tracks + groups` could emit `STRONG`. River already
   refused that. Post-sync mapping downgrades shared-map tracks to at most
   `WEAK`. The v7 core still *used* those tracks inside GlueMap. Product
   connectivity is therefore GlueMap-internal, not selector-certified.
2. **Can high raw matches / VPR sneak into geometry?** P167 shows they did
   not on the river probe (275 → 0). Mapping now also refuses VPR promotion.
3. **Does last-known-good mask a disconnected graph?** **Yes.** Receipt
   `stop_reason=last_known_good` with `exact_pair_probe_present=false`.
   FINAL_MAP states the core trio are isolated on the verified session graph.
4. **Can local BA cosmetically lower reproj on wrong aliases?** P157
   explicitly blocked local BA after Sim3 LOO inconsistency. Correct refusal.
5. **Is FIM mistaken for success probability?** The retired river risk script
   ranked λmin/logdet deciles. Mapping `risk-ply` consumes heatmap JSON and
   prints the caveat; it does not recompute FIM.
6. **Are deleted artifacts overclaimed?** 8/18 B0+P157 product bins are gone.
   Historical fringe counts are not v7 admission evidence. P157 report says so.
7. **Build/test leakage?** P116/P117 were reserved as historical loc holdouts
   and were not in v7 Global BA. Held-out queries are those videos. This is
   the intended split; this audit did not re-hash every query frame against
   the 418 map names.

## Post-sync mapping contract

`classify_session_edge` now requires exact-pair independent artifact,
group-disjoint fit/holdout, and complete finite geometry for `STRONG`/`USABLE`.
`connection_is_admissible` no longer treats `WEAK` as mergeable.
`sfm-diagnosis risk-ply` writes map RGB + colored spheres + `legend.json` /
`risk_ply_receipt.json`. ActLoc remains shadow-only.

## Replay / test commands

```bash
# mapping focused contract tests (no GPU / optional extras)
PYTHONPATH=diagnosis/src python3 -m pytest \
  diagnosis/tests/sfm_qa/test_edge_admission.py \
  diagnosis/tests/sfm_qa/test_session_select.py \
  diagnosis/tests/sfm_diagnosis/test_risk_ply.py

# river risk-PLY smoke (wrapper must not import pose_information / own FIM)
MAPPING_SRC=/home/cihcilab/sources/mapping/diagnosis/src \
  python tools/run_v7_localization_risk_map.py
```

## Required remediation

1. Do not describe P119/P120/P118 as independently bridged until exact-pair
   probes exist. Keep the product, label it last-known-good.
2. Keep P157 out of local/Global BA until 6/16 + group-disjoint Sim3 holdout.
3. Do not promote P116 child on the ≥30 heuristic without a frozen loc
   calibration set.
4. Do not treat B0 cluster≥2 as independent bridges.
5. Unify or explicitly fork mapping `min_pnp_inliers=25` vs river strict 80.

## Remaining risks

- Generic `GLOBAL_BA` is still granted when a base role is admitted; that is
  first-map build authorization, not independent-edge certification.
- Camera-sampled canonical heatmap was not required for the first risk-PLY
  refresh; FIM-weak / direction-sensitive classes appear only if a mapping
  heatmap is supplied.
- Held-out JSONL rows are huge; do not copy them into GitHub.
