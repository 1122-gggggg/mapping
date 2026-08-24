# S0–S9 map and localization quality roadmap — 2026-08-23

This document consolidates three independent code-and-paper audits of the active mapping
repository. It is a roadmap and evidence contract, not a claim that external reconstruction,
EDM inference, local bundle adjustment, or physical recapture ran in this change.

## Evidence labels

- **Implemented:** code and focused tests exist in this repository.
- **Partially wired:** a validator or planner exists, but artifact generation or replay is external.
- **Future:** proposed work; it must not appear as a completed release capability.
- **External dependency:** requires an external repository, weights, GPU, field data, or operator action.

Map-only metrics rank risk and explain failure. Only provenance-bound, held-out localization
can establish deployment quality.

## Release objective

The system should select and release a map by a portfolio objective rather than any single
point-count or reprojection threshold:

```text
maximize:
  held-out localization recall and pose quality
  + route/region/direction/appearance coverage
  + verified geometry, graph connectivity and view diversity
  + long-term landmark stability
minimize:
  false edges, ambiguity, weak-region risk
  + map/runtime cost and redundancy
subject to:
  build/test hash isolation
  finite and internally linked geometry
  fixed camera/intrinsics contracts
  verified cross-session geometry
  no stable-holdout regression
```

Quality preferences are relative within the available cohort. Data integrity, held-out
leakage, geometry validity and release provenance remain hard constraints.


## Status lattice

Implemented receipts use two axes. No paper number is a production threshold.

| Axis | Values | Meaning |
| --- | --- | --- |
| `hard_status` | `VALID`, `HARD_FAIL` | Artifact, identity, finiteness, leakage, gauge, or merge-authority invariants. |
| `evidence_status` | `PASS`, `WARN`, `INSUFFICIENT_EVIDENCE`, `QUALITY_SHORTFALL` | Soft claim support on declared independent units. |

Independent certification units are whole sessions or predeclared route/spatial
blocks, never adjacent frames (Sattler et al. 2018; Sarlin et al. 2022;
Roberts et al. 2017). Comparative MapDoctor summaries are `DESCRIPTIVE_ONLY`.
Target-site release requires every S0–S9 gate `PASS` and `release_lineage_bound`
in `verify_final_release.py`. S9 emits `hard_status` / `evidence_status` but no
`release_authorized` field.

## Current S0–S9 wiring and gaps

| Stage | Current repository path | Status | Highest-value gap |
| --- | --- | --- | --- |
| S0 | `sites/<site>/tools/s0_corpus_lock.py` | Implemented | Hash isolation exists, but route/region/direction/appearance coverage is not a formal corpus contract. |
| S1 | `s1_motion_scan.py` | Implemented | Motion thresholds are site/calibration dependent; direction confidence and cross-condition evidence are incomplete. |
| S2 | `s2_extract.py` | Implemented | Time-uniform/fixed-count extraction does not protect every route, direction, lighting and viewpoint bin. |
| S2b | `s2b_intrinsics_bakeoff.py` | Partially wired | Seed agreement/document integrity does not prove the candidate that maximizes development-held-out localization. |
| S3 | `s3_pairs.py` | Implemented | Candidate count is not retrieval Recall@K or verified route-bin geometry coverage. |
| S4 | `audit_dg_graph.py` | Implemented | Pair-level cycle/cheirality evidence and independent 3D bridge support are not one unified gate. |
| S5 | `run_gluemap_memory_safe.py`, `finalize_edm_model.py` | Implemented | No weak-region targeted track repair, retriangulation and anchored local-BA runner. |
| S5.7 | `audit_independent_sim3.py` | Partially wired | Sim3 fit and validation need a truly disjoint support split. |
| S6 | `audit_map_geometry.py` | Implemented | Shared-track ghost/false-merge evidence and repair closure are incomplete. |
| S7 | `validate_tracking_bundle.py` | Validator implemented | The canonical runner validates a pre-existing bundle; bundle-build receipts are external. |
| S8 | `validate_edm_bundle.py` | Validator implemented | EDM bundle generation is external and is not atomically receipt-bound to S7/S9. |
| S9 | `validate_heldout_localization.py` | Implemented | Results must bind `ts_common.TEST` identities; G9.3 is typed `NOT_APPLICABLE`. The validator still consumes existing result JSON rather than executing the EDM localizer. |
| Diagnosis | `diagnosis/src/sfm_qa`, `sfm_diagnosis`, `mapdoctor` | Implemented/partial | Additive evidence receipts landed. Repair execution and attested S9 replay remain future work. Comparative `READY` on `UNVERIFIED_PROVIDED_LOG` is not certification. |


## Evidence receipts implemented in this change

| Receipt | Code | Implemented fields vs experiment-only |
| --- | --- | --- |
| Prebuild stop | `sfm_qa/session_select/prebuild.py::_stopping_evidence` | `stop_reason`, signed `margin`/`keep_floor`, `hard_status`/`evidence_status`. Does not change proposed IDs. Submodular/ILP replacement is experiment-only. |
| Retrieval vs geometry | `prebuild.py::camera_triplet_scores`, `export.py::PREBUILD_PAIR_COLUMNS` | `evidence_type`, `retrieval_triangle_priority`, `count_field_provenance=num_candidate_pairs`. Manam verified `q_ij` is not computed. |
| Per-edge fusion | `session_select/admission.py::evaluate_geometry_authority` | One `GeometryAuthority` per edge. Fusion needs `exact_pair`, independent artifact, disjoint fit/holdout, and `independent_bridge_groups>=2`. Role alone cannot grant `GLOBAL_BA`. |
| Benchmark completeness | `mapdoctor/benchmark.py::summarize_benchmark` | `metric_evidence`, leave-one-criterion rates, `DESCRIPTIVE_ONLY`. Leave-one rates do not change `passes`. |
| Risk shortfall | `mapdoctor/diagnostics/risk_coverage.py::evaluate_risk_coverage` | Per-target `empirical_status`/`confidence_status`/`bound_shortfall`/`zero_failure_min_independent_units`. `independence_verified=False`. CRC/LTT/nonexchangeable weights are experiment-only. |
| Changed-region review | `map_update/core/changed_region_evidence.py` | `proposal_status`, review-only `REVIEW_CANDIDATE`, signed margins, suppressed-not-erased rows. No geometry invalidation. Automatic replacement is experiment-only. |
| Bridge margins | `map_update/core/update_quality_gates.py::bridge_quality_checks` | `{value,threshold,signed_margin,finite,hard_status,evidence_status}`. Non-finite is `HARD_FAIL`. Merge still uses warning strings. |
| S9 identity/lineage | `sites/target_site/tools/validate_heldout_localization.py`, `verify_final_release.py` | Exact TEST identities, recomputed rate, zero-frame reject, G9.3 `NOT_APPLICABLE`, mandatory S8→S9 hash lineage. G9 numeric defaults unchanged. |

## Improvements implemented in this change

### Query-level localization solutions

Every strict query failure now preserves `diagnose_pose()` recommendations and emits a
structured `resolution_plan`:

- policy is always `EXISTING_DATA_FIRST`;
- initial authorization is `NOT_AUTHORIZED` and counterfactual status is `REQUIRED_NOT_RUN`;
- map-limited failures use map counterfactual repair before conditional recapture;
- localizer-limited failures prioritize retrieval, matching, 2D–3D lifting and PnP;
- alias/PnP failures prioritize independent reference and geometry verification;
- the same failed block must improve without regressing the frozen stable holdout.

When query XYZ is unavailable, the report does not invent a map root cause. It asks for
retrieval-to-PnP instrumentation and pose-local evidence.

### Weak-region solution contract

Every weak-region repair step now contains:

- a stable `stage_id`;
- expected evidence changes, not fabricated numeric gain;
- acceptance checks;
- an existing-data counterfactual stage;
- conditional recapture only after counterfactual completion;
- a frozen weak/stable holdout validation contract.

The static diagnosis never authorizes capture. Only the separate audited recapture planner
may emit `TARGETED_RECAPTURE_REQUIRED` after its hard metrics and safety blockers are handled.

The same contract is written to `repair_plan.json`. Physical capture poses still require
the metric-audited `python -m mapdoctor.recapture plan` workflow, an explicit route frame,
and external safety review.

## Weak-region root cause to solution loop

| Evidence/root cause | Existing-data-first intervention | Acceptance evidence | Escalation condition |
| --- | --- | --- | --- |
| Retrieval gap / wrong condition or direction | Rebuild/reindex references; measure Recall@K and retrieval-to-PnP survival; add existing condition/direction references to the bundle. | Recall@K, top-1 margin, verified match survival, weak/stable success. | Targeted operational-direction capture only if existing references cannot cover the condition. |
| View-graph isolation / critical bridge | Expand weak-to-anchor and cross-route pairs; require independently separated verified bridge groups. | Anchor-track ratio, component integrity, bridge/cycle support, held-out success. | Strong matcher, then separate submap or bridge capture if no valid baseline exists. |
| Matching shortage / low texture | Targeted LightGlue/LoFTR/EDM matching on weak-anchor pairs; preserve image-plane coverage. | Verified matches, unique 2D–3D points, hull/grid coverage, new multi-view tracks. | Detector-free/local reconstruction or lateral/oblique capture when matching cannot create geometry. |
| Matching ambiguity / repeated structure | Doppelgangers++, robust geometric verification, cycle/triplet checks and bad-edge quarantine. | Reference-pose dispersion, cycle residual, positive depth, ghost/teleport audit. | Independent submap when an influential edge has no safe alternate cycle. |
| Track fragmentation | Merge only mutually and geometrically consistent ≥3-view tracks; retriangulate. | Track p50 rises, two-view fraction falls, triangulation support improves, stable holdout does not regress. | New baseline capture if the observations remain two-view/low-angle. |
| High reprojection / inconsistent pose | Prune bad observations, keep intrinsics fixed, constrain healthy anchors, retriangulate and run candidate local BA. | Reprojection p90, anchor pose drift, intrinsics delta, weak/stable replay. | Reject the candidate immediately on stable regression. |
| Low parallax / weak FIM / poor coverage | Search existing long-baseline/cross-route views; repair tracks and bundle associations. | FIM rank/weak mode, angle p10/p50, hull/grid, pose consensus and held-out success. | Audited lateral/height oblique capture after counterfactual failure. |
| Illumination or appearance mismatch with healthy geometry | Add condition-matched references and rebuild retrieval/bundle artifacts, not geometry. | Condition-stratified retrieval, matchability and localization success. | Capture at the missing time/light condition while preserving the same route/view. |
| PnP or reference consensus failure | Balance 2D–3D samples, retain per-reference hypotheses, verify calibration and covariance. | Inliers, positive depth, reprojection, reference dispersion/covariance, jump rate. | Geometry repair only after correspondence/calibration failure is excluded. |

## Required counterfactual protocol

For every declared repair stage:

1. Freeze baseline map, weak/stable query manifests, configuration and content hashes.
2. Write the candidate to a new directory; never mutate the baseline.
3. Record pair, observation, track, pose, intrinsics, bundle and tool/environment provenance.
4. Replay the same weak and stable blocks.
5. Use `mapdoctor compare` semantics: no excessive success drop, new failures, inlier
   regression or reprojection regression on the stable block.
6. Compute observed deficit closure on the weak block.
7. Mark `existing_data_counterfactual_complete=true` only after every predeclared stage ran.
8. Authorize targeted recapture only when repairability remains low and structural evidence
   proves missing physical information.

The default 95% target and regression limits in this repository are project policies, not
universal constants from the papers.

## Priorities

### P0: evidence and release integrity

1. Bind Stage 0 selection to authoritative site S0 build/test hashes.
2. Add route/region/direction/appearance coverage receipts to S0–S3.
3. Require S7/S8 builder receipts containing model, bundle, weights, script and environment hashes.
4. Generate S9 from the actual localizer frame log; do not accept manually summarized results as proof.
5. Extend S9 rows with session/condition/weak-region, retrieval, matching, unique 2D–3D,
   PnP, coverage, reference consensus, continuity and latency evidence.
6. Keep map-only/FIM scores as ranking signals until calibrated on independent sessions.

### P1: repair orchestration

1. Implement a candidate-only weak-region runner: targeted pairs → strong matcher → edge
   verification/quarantine → track repair → retriangulation → fixed-intrinsics anchored BA.
2. Feed graph articulation, bridge, spectral and threshold-sensitivity evidence into
   region-level decisions.
3. Rebuild S7/S8 artifacts atomically for every accepted candidate.
4. Execute weak/stable replay and emit `ComparisonResult` plus measured repairability.
5. Connect the audited recapture planner only after existing-data repair stages complete.

### P2: backend policy and conditional research paths

- Detector-Free SfM or Dense-SfM for local low-texture/fragmented-track regions.
- PixSfM featuremetric refinement for noisy keypoints and appearance change.
- COLMAP 4.x GlobalMapper is the default S5/S5.7 backend; LFOE is blocked without a license grant.
- MP-SfM for proven low-overlap/low-parallax/symmetry regimes.
- RoMo-style dynamic correspondence masking where moving content is material.
- ActLoc/Fisher planning for route and camera direction after site-held-out calibration.
- Appearance-conditioned landmark/reference selection for day/night and long-term change.

Research backends do not become defaults merely because they win a paper benchmark.
Every candidate must pass the same S6/S9, resource, license and gauge contracts.

## Non-portable constants

Do not copy these across sites without development-held-out calibration:

- optical-flow, rotation-rate, parallax and fast-motion thresholds;
- fixed FPS, hover caps and frames per video;
- forced pair count, route separation and neighbors per image;
- intrinsics seed/cross-resolution percentage agreement;
- matcher confidence and input resolution;
- FIM/noise/visibility scale assumptions;
- recapture offsets expressed in map units;
- paper benchmark pose tolerances and K-cover ratios.

Only artifact integrity, non-finite geometry, hash leakage, invalid camera shape and
unverified merge authority remain universal hard failures.

## Validation matrix

Every candidate release should report:

- input videos/frames/pairs and route/direction/condition coverage;
- verified edge, component, bridge, cycle and spectral diagnostics;
- registered images, tracks, point support, triangulation angle and reprojection;
- bundle reference/landmark coverage and artifact identities;
- retrieval Recall@K and stage survival;
- unique 2D–3D, PnP inliers/ratio, hull/grid, positive depth and reference consensus;
- pose error where ground truth exists, continuity, jump/ghost and relocalization time;
- strict success by session, weak/stable region, direction and appearance condition;
- wall time, RAM/VRAM, map size and localization latency;
- calibrated risk–coverage only on independent groups/certification data, with
  `INSUFFICIENT_EVIDENCE` versus `QUALITY_SHORTFALL` and signed shortfalls;
- S9 identity receipt and S8→S9 artifact hashes before any release claim.

Adjacent video frames are not independent certification samples.
Query-level Clopper–Pearson, Wilson, or CRC intervals are descriptive
until groups and provenance are attested.

## Deferred experiment-only work

These remain outside the release path. Each needs development-held-out
calibration and one frozen S9 certification run:

- conformal risk control and nonexchangeable weighting (Angelopoulos et al. 2024; Farinhas et al. 2023; Barber et al. 2023);
- group-level Learn-Then-Test and nested session calibration (Angelopoulos et al. 2025; Kull et al. 2017; Roberts et al. 2017);
- alternate reconstruction backends: PixSfM, Detector-Free SfM, MP-SfM, RoMo, licensed-LFOE only, Light3R-SfM, depth-guided SfM;
- lifelong posterior/FreMEn/BOCPD calibration and automatic tile replacement (Rosen 2016; Krajník et al. 2017; Adams and MacKay 2007);
- replacement of the mixed prebuild objective by a monotone submodular or ILP selector (Nemhauser et al. 1978; Gygli et al. 2015).


## Primary sources

- [Structure-from-Motion Revisited](https://openaccess.thecvf.com/content_cvpr_2016/html/Schonberger_Structure-From-Motion_Revisited_CVPR_2016_paper.html)
- [Skeletal Graphs for Efficient Structure from Motion](https://www.cs.cornell.edu/~snavely/projects/skeletalset/)
- [View-graph Selection Framework for SfM](https://openaccess.thecvf.com/content_ECCV_2018/html/Rajvi_Shah_View-graph_Selection_Framework_ECCV_2018_paper.html)
- [Optimizing the Viewing Graph for Structure-from-Motion](https://openaccess.thecvf.com/content_iccv_2015/html/Sweeney_Optimizing_the_Viewing_ICCV_2015_paper.html)
- [Leveraging Camera Triplets for SfM](https://openaccess.thecvf.com/content/CVPR2024/html/Manam_Leveraging_Camera_Triplets_for_Efficient_and_Accurate_Structure-from-Motion_CVPR_2024_paper.html)
- [GLUEMAP](https://arxiv.org/abs/2605.26103)
- [Doppelgangers++](https://openaccess.thecvf.com/content/CVPR2025/html/Xiangli_Doppelgangers_Improved_Visual_Disambiguation_with_Geometric_3D_Features_CVPR_2025_paper.html)
- [LFOE-GlobalSfM](https://openaccess.thecvf.com/content/CVPR2025/html/Damblon_Learning_to_Filter_Outlier_Edges_in_Global_SfM_CVPR_2025_paper.html)
- [LoFTR](https://openaccess.thecvf.com/content/CVPR2021/html/Sun_LoFTR_Detector-Free_Local_Feature_Matching_With_Transformers_CVPR_2021_paper.html)
- [LightGlue](https://openaccess.thecvf.com/content/ICCV2023/html/Lindenberger_LightGlue_Local_Feature_Matching_at_Light_Speed_ICCV_2023_paper.html)
- [Detector-Free Structure from Motion](https://openaccess.thecvf.com/content/CVPR2024/html/He_Detector-Free_Structure_from_Motion_CVPR_2024_paper.html)
- [Pixel-Perfect SfM](https://arxiv.org/abs/2108.08291)
- [GLOMAP](https://arxiv.org/abs/2407.20219)
- [MP-SfM](https://openaccess.thecvf.com/content/CVPR2025/html/Pataki_MP-SfM_Monocular_Surface_Priors_for_Robust_Structure-from-Motion_CVPR_2025_paper.html)
- [RoMo](https://openaccess.thecvf.com/content/ICCV2025/html/Goli_RoMo_Robust_Motion_Segmentation_Improves_Structure_from_Motion_ICCV_2025_paper.html)
- [MegaLoc](https://arxiv.org/abs/2502.17237)
- [Hierarchical Localization](https://openaccess.thecvf.com/content_CVPR_2019/html/Sarlin_From_Coarse_to_Fine_Robust_Hierarchical_Localization_at_Large_Scale_CVPR_2019_paper.html)
- [Map Quality Evaluation for Visual Localization](https://tisl.cs.utoronto.ca/publication/201705-icra-map_quality_evaluation/icra17-map_quality_evaluation.pdf)
- [Fisher Information Field](https://arxiv.org/abs/2008.03324)
- [Long-Term Visual Map Sparsification](https://openaccess.thecvf.com/content/CVPR2022/html/Chang_Long-Term_Visual_Map_Sparsification_With_Heterogeneous_GNN_CVPR_2022_paper.html)
- [Benchmarking 6DOF Outdoor Visual Localization in Changing Conditions](https://www.microsoft.com/en-us/research/publication/benchmarking-6dof-outdoor-visual-localization-in-changing-conditions/)
- [Conformal Risk Control](https://arxiv.org/abs/2208.02814)
- [LaMAR](https://arxiv.org/abs/2210.10770)
- [RankIQA](https://arxiv.org/abs/1707.08347)
- [Selective Classification (Geifman and El-Yaniv, 2017)](https://proceedings.neurips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html)
- [Clopper and Pearson (1934)](https://doi.org/10.1093/biomet/26.4.404)
- [Roberts et al., structured cross-validation (2017)](https://doi.org/10.1111/ecog.02881)
- [Learn then test (2025)](https://doi.org/10.1214/24-AOAS1998)
- [Non-Exchangeable CRC](https://arxiv.org/abs/2310.01262)
- [ExMaps](https://openaccess.thecvf.com/content/WACV2021/html/Rotsidis_ExMaps_Long-Term_Localization_in_Dynamic_Scenes_Using_Exponential_Decay_WACV_2021_paper.html)
- [Bürki et al., map management (2018)](https://arxiv.org/abs/1808.02658)
- [Berrio et al. (2021)](https://arxiv.org/abs/2008.12449)
