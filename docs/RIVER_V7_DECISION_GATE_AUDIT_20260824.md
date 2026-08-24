# River v7 decision-gate audit — 2026-08-24

This is the mapping-repo summary of the independent river audit. It does
**not** retune gates to preserve last-known-good labels. Site-absolute paths
are evidence locations. The generic contract does not depend on them.

Authoritative river artifacts (owned by AuditFinisher; do not edit here):

- `/media/cihcilab/新增磁碟區/河濱場域/gluemap_build/runs/RIVER_V7_DECISION_GATE_AUDIT_20260824.md`
- `/media/cihcilab/新增磁碟區/河濱場域/gluemap_build/runs/RIVER_V7_DECISION_GATE_AUDIT_20260824.json`

Verdicts and replay numbers below are copied from that pair. If they diverge,
the river files win.

## Immediate decision change

Do **not** consume on-disk `P116 ADMITTED_CHILD` as scientific multi-group
admission. Audit verdict: `EXPERIMENTAL_CHILD_ADMISSION_REVOKED`.
`scientific_admission=false`. The child is preserved as a core-invariant
experimental append. Product is **not** promoted.

Do **not** change frozen product bytes, P157 `NO_GO`, P167 `NO_GO`, or
P119/P120/P118 membership in the frozen B0. Authorization is last-known-good
plus a successful joint reconstruction, not independently graph-authorized.

Do **not** treat FIM as success probability. ActLoc remains shadow-only.

Held-out success for risk-PLY is the conjunction of outer
`DIRECT_STRONG` and nested `decision.status == ACCEPT`. Nested `REJECT*`
always fails. Nested `ACCEPT` alone is not sufficient: outer
`GEOMETRY_WEAK` or `PROVISIONAL` plus nested `ACCEPT` remains a
weak/provisional marker. If either richer status is present, boolean
`success` cannot override; missing one side is not strict. Boolean
`success` applies only when both richer statuses are absent.

## Artifacts opened

| Artifact | Used as |
| --- | --- |
| river `runs/RIVER_V7_DECISION_GATE_AUDIT_20260824.md/.json` | independent audit authority |
| `outputs/final_map/FINAL_MAP.md` | product claim |
| `outputs/session_selection_staged_optimized_v7_20260823/{selection_receipt,optimized_data_selection,session_edges,session_roles,session_quality,session_graph}` | selector decision |
| `runs/river_core_map_v7_20260823/maps/map_gluemap_baseline/b0_quality.json` | B0 gate + thresholds |
| `runs/P157_V7_FUSION_ANALYSIS_20260824.md` + `runs/river_p157_v7_bridge_analysis_20260824/REPORT.json` | P157 fusion |
| `runs/CORE_MAP_V7_AND_P167_BRIDGE_RESULT_20260824.md` + `runs/river_p167_bridge_audit_v7_20260824/REPORT.json` | P167 bridge |
| `runs/river_v7_heldout_localization_20260824/{REPORT.json,registrations/*.jsonl}` | held-out loc |
| `runs/river_v7_p116_only_fringe_20260824/REPORT.json` | P116 child |
| mapping `diagnosis/src/sfm_qa/session_select/{types,edges,admission,defaults.yaml}` | post-sync gates |
| mapping `diagnosis/src/sfm_diagnosis/{diagnose.py,risk_ply.py}` | FIM / loc / risk-PLY |

Not opened as authority: deleted 8/18 B0+P157 product bins; historical fringe
track counts; MV-RoMa appearance matches; deleted P167 2026-08-17 pair
geometry and independent-map bins; `tools/final_map/defaults.yaml` paths to
those maps.

## Held-out localization (outer vs accepted-strict)

Outer `DIRECT_STRONG` is a metric-threshold label. Nested ambiguity
`decision.status` is the scientific accept. Strict success is both.

| Session | Queries | Outer `DIRECT_STRONG` | Conjunction-strict (outer `DIRECT_STRONG` and `decision.status=ACCEPT`) | Outer provisional | Notes |
| --- | --- | --- | --- | --- | --- |
| P116 | 94 | 38 | **10** | 17 | Nested among the 38 strong: 26 `REJECT_UNVERIFIED_SUPPORT`, 10 `ACCEPT`, 1 `REJECT_LOCAL_DEGENERACY`, 1 `REJECT_LOW_SUPPORT` |
| P117 | 52 | 0 | **0** | 0 | All `GEOMETRY_WEAK` |
| P157 | 129 | 7 | **6** | 37 | Query-to-map lifts, not Sim3. 4/7 see only P119; 3/7 see P119+P120; 0/7 see P118 |
| P167 | 97 | 0 | **0** | 2 | Independent 0/97 agrees with NO_GO |

Do not write “P116 38/94 strict” or “P157 7/129 strict” as accepted-strict
counts. Those are outer labels. Risk-PLY skips only the conjunction
outer `DIRECT_STRONG` and nested `ACCEPT` (and still marks provisional
rows). Nested `ACCEPT` under outer `GEOMETRY_WEAK` is a weak marker.

## Direct vs inferred

Direct:

- Selector `admission_state=HISTORICAL_BEST`, `stop_reason=last_known_good`,
  `exact_pair_probe_present=false`, `scores={}`. Greedy did not run.
- Verified session graph: 7 components, `usable_edge_count=0`. Base trio
  isolated; λ2=0. All 21 edges `independent_artifact=false`. 5 WEAK
  shared-map edges, 16 REJECT (VPR-only or no geometry).
- FINAL_MAP / B0: 418/418 registered, 237907 points, P90 2.159929674553544,
  positive depth 1.0. 159 bridge-only images with 0 observations. Independent
  pycolmap: 161 zero-observation images. Extra two triangulation zeros:
  `P1200120/pts_000103103000_frame_002472.jpg` and
  `P1200120/pts_000104104000_frame_002496.jpg` (pose-repair camera).
- Dominant covisibility component 255 (P118=63, P119=101, P120=91) + 161
  isolates + 1 pair. Map-internal connectivity, not independent pair/Sim3.
- P157 live: 0/73 SIFT geometry; PnP grid median 2/16 (need 6); 0 admitted
  anchors; relaxed Sim3 3/13 folds, residual P50 15.2% vs 5% cap.
  `NO_GO_LOCAL_FUSION`.
- P167 embedded: 275 VPR → 0 geometry. `NO_GO` / `SUBMAP_ONLY`. Cited
  submap 97 / 25158 / P90 1.5536 **not replayable from bins**.
- P116 child construction: 38 images / 24423 points, core invariance true,
  `product_promoted=false`. Connected-fringe rule holds (`b0_view_count` min=1).
- P116 on-disk label remains `ADMITTED_CHILD`. Scientific admission is
  **revoked**. Child bytes are preserved; product hardlinks v7 only.

Inferred (not evidence):

- That GlueMap-internal clusters (11–13) equal independent bridges.
- That last-known-good `GLOBAL_BA` is independently certified fusion.
- That FIM logdet / λmin predicts localization success.
- That 10-second bins or pooled core-session inliers are independent groups.

## Threshold inventory

| Threshold | Where | Class | Must not be called empirical/deployment gate? |
| --- | --- | --- | --- |
| exact-pair scope; independent artifact; group-disjoint fit/holdout; complete finite geometry | mapping `defaults.yaml` `edge.require_*` + `classify_session_edge` | **data-integrity invariant** | No. Fail-closed identity rules. |
| VPR is not an edge | mapping + river session graph | **data-integrity invariant** | No. |
| B0 registration ≥0.95 | `b0_quality.json` | **engineering heuristic** | Yes, if marketed as localization readiness. 418/418 includes 159 zero-obs images. |
| B0 positive depth ≥0.99 | same | **cheirality integrity** | Borderline. Reconstruction check, not query success. |
| B0 reproj P90 ≤5 px | same | **engineering heuristic** | Yes. 2.16 px passed; the 5 px cap is unlabeled. |
| B0 cross-video clusters ≥2 | same | **uncalibrated / self-certifying** | **Yes.** Shared-reconstruction clusters. Post-sync mapping treats them as `WEAK`. |
| Strict loc: inliers≥80, ratio≥0.25, hull≥0.15 (`convex_hull_coverage`), occ≥6/16, depth≥0.99, reproj P90≤3 | held-out runner | **engineering heuristic** | **Yes**, until calibrated. Outer `DIRECT_STRONG` ≠ nested `ACCEPT`. |
| P157 PnP occupancy 6/16 | P157 REPORT | **engineering heuristic** | Yes as a probability gate. NO_GO still supported (max 4/16, Sim3 LOO failed). |
| P157 Sim3 all LOO residual ≤5% | P157 REPORT | **engineering heuristic** | Observed 15–35% still fails any reasonable residual cap. |
| P116 ≥30 outer-strong, ≥2 ten-second bins, obs in ≥2 pooled core sessions | P116 REPORT | **uncalibrated; 10 s bins are a scientifically wrong independence proxy** | Yes as a promotion gate. Construction may stay; scientific admission is revoked. |
| mapping `edge.min_cross_tracks_for_verified=30`, `min_verified_pairs_for_usable=8` | `defaults.yaml` | **engineering heuristic** | **Yes.** They never authorize STRONG/USABLE alone after this sync. |
| mapping `DiagnosticThresholds.min_pnp_inliers=25` vs river strict 80 | `diagnose.py` vs loc runner | **uncalibrated inconsistency** | **Yes.** Do not mix the two as one “strict” number. |
| FIM λmin / logdet / condition | selector + heatmap + old risk map | **observability proxy** | **Yes.** Not success probability. ActLoc proxy is shadow-only. |

## Role / fusion verdicts

| Decision | Verdict | Why |
| --- | --- | --- |
| P119/P120 `BASE_CORE` / `GLOBAL_BA` | **CORRECT BUT OVERSTATED** | Internal strongest hard-valid videos and the operational product. Selector `preserve_last_known_good` short-circuits greedy. Verified graph degree=0. Not independent-edge certified. |
| P118 `BASE_SUPPORT` membership | **CORRECT BUT OVERSTATED** | Joint B0 membership is real. `USABLE` with `FIM_ILL_CONDITIONED` + `ROTATION_DOMINATED`. |
| P118 `GLOBAL_BA` authorization | **HEURISTIC-ONLY** | Granted because `admission_state=HISTORICAL_BEST`, not an independent edge. Mapping-ref would leave unaudited `BASE_SUPPORT` at `LOCAL_RELATION_ONLY`. |
| P157 `GEOMETRY_REINFORCEMENT` / `LOCAL_RELATION_ONLY` / `NO_GO_LOCAL_FUSION` | **CORRECT** | Live v7 exact-pair evidence fails. Historical 75 DIRECT_STRONG / 18449 fringe points are different, now-deleted model bytes. |
| P157 loc 7/129 outer (6/129 accepted-strict) as fusion support | **HEURISTIC-ONLY** | Localization ≠ independent Sim3. Still `LOCAL_RELATION_ONLY`. |
| P167 `NEW_SUBMAP` / `SUBMAP_ONLY` / `NO_GO` | **CORRECT** | 0/275 geometry. High raw matches are not a bridge. Deleted 2026-08-17 sources cannot be replayed. |
| P116/P117 `QUARANTINE` as core | **CORRECT** | Contaminated contrast maps; not core geometry. |
| P116 child construction safety | **CORRECT** | 38 images / 24423 points, core invariance true, product not promoted. |
| P116 `ADMITTED_CHILD` as scientific admission | **`EXPERIMENTAL_CHILD_ADMISSION_REVOKED`** | `scientific_admission=false`. Three numeric gates are uncalibrated and are not independent spatial groups. Child preserved. Do not promote. |
| P117 as deployable loc | **UNSUPPORTED** | 0/52 accepted-strict. |
| Selector “Greedy selected P119/P120/P118” | **UNSUPPORTED** | Receipt is last-known-good before greedy. |
| Historical `independent_bridge_groups` on `independent_artifact=false` edges | **UNSUPPORTED** | Shared-map track clusters. |
| Old risk map 131/45/8 FIM spheres as calibrated risk | **UNSUPPORTED** | Duplicate FIM, relative 10% cuts, no held-out calibration. Replaced by mapping `risk-ply`. |
| FIM as P(success) | **UNSUPPORTED** | Observability only. P116 outer-strong FIM condition p50=1966, effective rank p50=2.285. |

## Adversarial challenges

1. **Can same-map tracks self-certify bridges?** Before this sync, mapping
   `trusted_geometry + tracks + groups` could emit `STRONG`. River already
   refused that. Post-sync mapping downgrades shared-map tracks to at most
   `WEAK`. The v7 core still *used* those tracks inside GlueMap. Product
   connectivity is GlueMap-internal, not selector-certified.
2. **Can high raw matches / VPR sneak into geometry?** P167 shows they did
   not on the surviving embedded funnel (275 → 0). Mapping now also refuses
   VPR promotion. Deleted 2026-08-17 pair files are not live evidence.
3. **Does last-known-good mask a disconnected graph?** **Yes.** Receipt
   `stop_reason=last_known_good` with `exact_pair_probe_present=false`.
4. **Can local BA cosmetically lower reproj on wrong aliases?** P157
   explicitly blocked local BA after Sim3 LOO inconsistency. Correct refusal.
5. **Is FIM mistaken for success probability?** The retired river risk script
   ranked λmin/logdet deciles. Mapping `risk-ply` consumes heatmap JSON and
   prints the caveat; it does not recompute FIM.
6. **Are deleted artifacts overclaimed?** 8/18 B0+P157 product bins are gone.
   P167 2026-08-17 geometry + independent map are gone. Historical fringe
   counts are not v7 admission evidence. Receipts with `*_present: true` are
   historical.
7. **Does outer `DIRECT_STRONG` leak rejected ambiguity?** **Yes.** 26/38
   P116 construction rows are `REJECT_UNVERIFIED_SUPPORT`. Risk-PLY now
   requires outer `DIRECT_STRONG` and nested `ACCEPT`; nested `REJECT*`
   always wins.
8. **Build/test leakage?** P116/P117 were reserved as historical loc holdouts
   and were not in v7 Global BA. This audit did not re-hash every query frame
   against the 418 map names.

## Post-sync mapping contract

`classify_session_edge` requires exact-pair independent artifact,
group-disjoint fit/holdout, and complete finite geometry for `STRONG`/`USABLE`.
`connection_is_admissible` no longer treats `WEAK` as mergeable.
`sfm-diagnosis risk-ply` writes map RGB + colored spheres + `legend.json` /
`risk_ply_receipt.json`. `.jsonl` is one object per line. Zero-observation
`bridge_only` vs triangulation roles are split. Held-out success is the
conjunction of outer `DIRECT_STRONG` and nested `ACCEPT`. Nested
`REJECT*` always wins. ActLoc remains shadow-only. FIM is not recomputed.

## Replay / test commands

```bash
# mapping focused contract tests (no GPU / optional extras)
PYTHONPATH=diagnosis/src python3 -m pytest \
  diagnosis/tests/sfm_qa/test_edge_admission.py \
  diagnosis/tests/sfm_qa/test_session_select.py \
  diagnosis/tests/sfm_diagnosis/test_risk_ply.py

# river risk-PLY smoke (wrapper must not import pose_information / own FIM)
MAPPING_SRC=/home/cihcilab/sources/mapping/diagnosis/src \
  python tools/run_v7_localization_risk_map.py \
  --heatmap outputs/final_map/localization_risk/heatmap \
  --localization runs/river_v7_heldout_localization_20260824/registrations \
  --image-roles runs/river_core_map_v7_20260823/maps/map_gluemap_baseline/frame_manifest.json
```

## Required remediation

1. Do not describe P119/P120/P118 as independently bridged until exact-pair
   probes exist. Keep the product, label it last-known-good.
2. Keep P157 out of local/Global BA until 6/16 + group-disjoint Sim3 holdout.
3. Do not promote the P116 child. On-disk `ADMITTED_CHILD` is
   `EXPERIMENTAL_CHILD_ADMISSION_REVOKED` for scientific admission.
   Construction safety can remain; 10 s bins are not independent groups.
4. Do not treat B0 cluster≥2 as independent bridges.
5. Unify or explicitly fork mapping `min_pnp_inliers=25` vs river strict 80.
6. Do not let `REJECT_UNVERIFIED_SUPPORT` become construction input without
   an explicit heuristic override.
7. Quarantine deleted-artifact citations (P167 2026-08-17, P157 2026-08-18,
   C0/A1, VPR pairs, pair-probe file, stale `tools/final_map/defaults.yaml`).

## Remaining risks

- Generic `GLOBAL_BA` is still granted when a base role is admitted; that is
  first-map build authorization, not independent-edge certification.
- Canonical `sfm-diagnosis heatmap` for v7 used camera-AABB bounds plus
  median-NN spacing grown ×1.25 to 594 positions / 1782 poses
  (`outputs/final_map/localization_risk/heatmap`). It diagnosed 5
  `GEOMETRY_WEAK` poses (also occupancy 5/16). Risk-PLY consumed those rows
  (`heatmap_rows=1782`, `fim_recomputed=false`) as 5 `fim_weak` + 5
  `coverage_hole`. No `direction_sensitive` spread met the 0.25/0.45 gate.
  FIM remains observability, not P(success).
- Held-out JSONL rows are huge; do not copy them into GitHub.
- River `tools/final_map/defaults.yaml` still names deleted 2026-08-15/18
  maps (`images.bin=5625d6cf…`, `points3D.bin=45b53d20…`) and a deleted P157
  child. Not v7 product authority.

Refreshed risk-PLY counts: `unverified_bridge_pose=159`,
`zero_triangulation=2`, `heldout_geometry_weak=300` (all 271
`GEOMETRY_WEAK` plus 29 outer `DIRECT_STRONG` without nested `ACCEPT`;
this set has 0 `GEOMETRY_WEAK`+`ACCEPT` rows, which would also mark),
`heldout_provisional=56`, `fim_weak=5`, `coverage_hole=5`. Conjunction
success is the 16 outer `DIRECT_STRONG` + nested `ACCEPT` rows only.
