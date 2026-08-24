# Theory and primary-source evidence audit — 2026-08-23

This audit records the follow-up multi-team review of the mapping, diagnosis, and
lifelong-update stack.  It is deliberately stricter than a bibliography: every proposed
change is classified as a direct engineering correction, a literature transfer that still
needs site data, or a non-portable research direction.


A 2026-08-24 follow-up recorded the implemented `hard_status` /
`evidence_status` lattice, independent-unit caveats, and additive
receipts.  No numeric default was changed.  Paper thresholds remain
non-portable.

## Evidence contract

The repository uses four evidence labels:

- **Specification correction:** behavior is fixed by the input format or mathematical
  model.  A paper benchmark is not needed to prove that parsing or projection was wrong.
- **Theory-preserving optimization:** an exact repeated computation is removed without
  changing the objective, graph, threshold, or output.
- **Literature transfer:** a paper supports the principle, but weights, thresholds, and
  deployment benefit must be calibrated on development-held-out sessions and certified on
  untouched S9 sessions.
- **Experiment only:** the method changes the frontend, representation, learned weights,
  or compute regime and cannot enter the release path without an A/B run.

No paper result is treated as proof of performance on this repository's UAV videos.

### Status lattice and independent units

Implemented receipts use two independent axes.  They do not invent a
new operating threshold.

| Axis | Values | Meaning |
| --- | --- | --- |
| `hard_status` | `VALID`, `HARD_FAIL` | Artifact, identity, finiteness, leakage, gauge, or merge-authority invariants. |
| `evidence_status` | `PASS`, `WARN`, `INSUFFICIENT_EVIDENCE`, `QUALITY_SHORTFALL` | Soft claim support on the declared units. |

- `HARD_FAIL`: unreadable, malformed, non-finite, internally inconsistent,
  leaked, misidentified, or unverified merge evidence.
- `PASS`: every predeclared soft claim is supported on the declared
  independent units.  It is not a paper-certified success probability.
- `WARN`: a point policy passes, or a review candidate exists, but
  provenance, independence, or fusion completeness is incomplete.
  Non-authoritative by default.
- `INSUFFICIENT_EVIDENCE`: too few independent units, missing required
  fields, or no resolvable selectivity.  Equivalent action is abstain;
  missing evidence is not a measured miss.
- `QUALITY_SHORTFALL`: enough valid evidence exists to show a miss of a
  site-calibrated soft target or margin.  Distinct from `HARD_FAIL`.

Release authority is not a paper result.  Target-site final release
requires `verify_final_release.py` to see every S0–S9 gate `PASS` and
`release_lineage_bound`.  S9 itself emits `hard_status` and
`evidence_status` but no `release_authorized` field.  MapDoctor
`check_localize` still labels logs
`evaluation_provenance=UNVERIFIED_PROVIDED_LOG` and
`heldout_provenance_verified=False`; `_overall_status` can still return
`READY` / `READY_WITH_MAP_WARNINGS`.  That combined status is a
diagnostic, not a certification claim.

Independent certification units are whole sessions or predeclared
route/spatial blocks, never adjacent frames (Sattler et al. 2018;
Sarlin et al. 2022 LaMAR; Roberts et al. 2017; Clopper and Pearson
1934).  Query-level Clopper–Pearson or Wilson intervals remain
descriptive unless `independence_verified` is attested.  No current
path sets that flag to true.

## Implemented in this audit

| Change | Evidence and theory | Code | Verification |
| --- | --- | --- | --- |
| Skip the unused full `N x N` descriptor matrix in default directional pairing | Source control flow proves that default `directional` mode with `same_direction_topk=0` never reads the matrix. Johnson et al. and Jégou et al. support candidate/block search rather than unnecessary exhaustive similarity materialization. This patch only removes dead work; it does not replace retrieval with ANN. | `map_update/build_localizable_map.py::stage_pairs` | A regression array raises on any full matrix multiplication. Directional temporal-pair output remains identical. The avoided allocation is `4N^2` bytes for float32: about 95.4 MiB at `N=5000`. A local seven-block `N=5000,D=128` probe reduced maximum RSS from 135,696 KiB to 40,316 KiB; its block peak was 2,044,900 bytes. This is an allocation probe, not an S9 runtime claim. |
| Reuse the already computed session-graph Fiedler value | Spectral graph theory and pose-graph sparsification motivate `lambda2`; the objective already obtained the exact value from `session_graph_diagnostics`. Reusing it removes a second eigendecomposition without changing the metric. | `diagnosis/src/sfm_qa/session_select/objective.py::compute_objective_terms` | Regression test counts one `eigvalsh` call per objective evaluation instead of two. |
| Make long-track covisibility support exact by default | A binary image-landmark incidence matrix `B` has exact shared-landmark counts in `B B^T`; map-quality and spectral-graph diagnostics require those supports rather than silently deleting every track longer than 20 observations. Row-block sparse multiplication preserves the graph definition while bounding temporary products. | `diagnosis/src/sfm_diagnosis/graph.py::build_covisibility_graph` | Fifteen 21-view tracks now yield one 21-node component and all 210 expected edges instead of 21 false isolates. Small-track exact/legacy parity is tested; an explicit integer cap remains labeled `legacy_approximation`. |
| Parse COLMAP shared-focal `RADIAL` family correctly | COLMAP defines `RADIAL` as `(f,cx,cy,k1,k2)`, not `(fx,fy,cx,cy,...)`. This is a specification correction. | `diagnosis/src/sfm_qa/bridge.py::_camera_intrinsics` | Synthetic `RADIAL` camera must produce `(fx,fy,cx,cy)=(500,500,320,240)`. |
| Use the same distortion model for projection and PnP quality | COLMAP's radial model and the standard camera-calibration projection equations require distortion in 3D-to-2D projection. OpenCV `projectPoints` and the existing PnP path already use these coefficients. Unsupported fisheye/FOV models now fail closed in this helper instead of silently pretending to be pinhole. | `map_update/lifelong/src/update_map/geometry.py::project_points` | Analytic `RADIAL` projection agrees to numerical precision; existing pinhole pose recovery remains green. |
| Make one session's point evidence mutually exclusive and order-independent | ExMaps motivates visibility- and recency-weighted landmark evidence. A match for the same point/session supersedes a provisional visible-but-unmatched event regardless of processing order; derived last-seen and unmatched-streak state is recomputed from the complete session timeline. Otherwise one physical observation can contribute both positive and negative evidence, or a late replay of an older session can erase newer negative evidence. The decay and retirement policy are unchanged. | `map_update/core/point_evidence_ledger.py::record_session` | Both same-session replay orders retain consistency `1.0`; replaying an older successful `s1` after unmatched `s2` preserves the newer one-session suspect streak and recency weights. |
| Keep lifelong decay time monotonic | Exponential recency decay is a function of non-negative elapsed time. Moving the stored watermark backward makes a later event decay the same interval twice. | `map_update/lifelong/src/update_map/stability.py::advance_time` | An out-of-order timestamp no longer moves the watermark or double-decays currentness. |


## Implemented evidence-status receipts (2026-08-24)

These changes add fields and fail-closed identity checks.  Proposed
sets, numeric defaults, and merge refuse/allow lists are unchanged
except where anonymous or non-finite evidence previously manufactured
authority.

| Change | Evidence and theory | Implemented fields | Code |
| --- | --- | --- | --- |
| Prebuild stopping receipt | Bürki et al. 2018 treat slack as feasibility, not success. Nemhauser/Gygli greedy bounds require a nonnegative monotone submodular objective; the current mixed `ΔJ` is heuristic. | `stopping_evidence.{stop_reason,hard_status,evidence_status,authority=reporting_review_only,grants_selection_or_merge_authority=false}`; optional `next_candidate_id,marginal,best_previous,keep_ratio,keep_floor,margin`. Empty/unreadable/legacy-ineligible → `HARD_FAIL`/`INSUFFICIENT_EVIDENCE`. Nonpositive or relative-marginal stop → `VALID`/`QUALITY_SHORTFALL`. Budget/exhaustion → `VALID`/`PASS`. | `diagnosis/src/sfm_qa/session_select/prebuild.py::_stopping_evidence`, `propose_prebuild_set` |
| Retrieval triangles vs verified triplets | Manam and Govindu score geometrically verified edges with epipolar-inlier support inside camera triplets. Retrieval counts are not that evidence. | Pair fields `evidence_type` ∈ {`retrieval_triangle`,`retrieval_candidate`,`forced_geometry_probe`}, `retrieval_triangle_priority` (bool), `count_field_provenance=num_candidate_pairs`. `triplet_score` remains a retrieval-count score. `requires_geometric_verification` stays true. `HIGH_PROPOSAL_ONLY` was not renamed. | `prebuild.py::camera_triplet_scores`, `propose_prebuild_set`; `export.py::PREBUILD_PAIR_COLUMNS` |
| Per-edge fusion authority | Sweeney defines view-graph edges as pairwise relative geometries. GLOMAP/1DSfM robustify verified relative geometry; they do not turn retrieval into a merge. Fusion needs ≥2 independent disjoint-holdout groups. | `GeometryAuthority` with `authorized`, `hard_status`, `evidence_status`, `independent_artifact`, `evidence_scope`, `geometry_complete`, `group_holdout_disjoint`, `independent_bridge_groups`, `fit_evidence_ids`, `holdout_evidence_ids`, `geometry_artifact_sha256`, `reasons`. `evaluate_geometry_authority` never reads another edge. `incident_fusion_authorization` grants from the first self-contained authorized edge; role alone cannot set `base_admitted`. Local `usable_geometry_ready(min_groups=1)` is `WARN`/`HARD_FAIL`, not fusion. | `diagnosis/src/sfm_qa/session_select/admission.py::evaluate_geometry_authority`, `incident_fusion_authorization`, `classify_fusion_authorization`; `export.py::build_role_rows` |
| Benchmark metric completeness | Missing metrics are not successes (Wilson 1927; Chow 1957). Leave-one-criterion rates are diagnostics only. | `BenchmarkSummary.metric_evidence[metric].{present,failed,fail_rate}` with `fail_rate=None` if `present=0`; `leave_one_criterion_strict_success_rates`; `interpretation=DESCRIPTIVE_ONLY`; `independent_units_verified=False`. Weak cells with XYZ: `INSUFFICIENT_EVIDENCE` if `n<2`, else `QUALITY_SHORTFALL`; `authority=DESCRIPTIVE_ONLY`. Cells without XYZ are omitted, not labeled healthy. `passes`/`failures` polarity is unchanged. | `diagnosis/src/mapdoctor/benchmark.py::QueryLocalizationResult`, `summarize_benchmark` |
| Risk-target shortfall statuses | Selective classification trades risk for coverage (Geifman and El-Yaniv 2017). Clopper–Pearson assumes fixed independent Bernoulli units (1934). Do not invent a threshold when no complete-tie point is feasible. | `target_diagnostics[target].{empirical_status,confidence_status,hard_status,evidence_status,bound_shortfall,zero_failure_min_independent_units,accept_all_baseline,independence_verified=False,authority=reporting}`. `NO_RESOLVABLE_SELECTIVITY` or unverified bound → `INSUFFICIENT_EVIDENCE`. Empirical miss → `QUALITY_SHORTFALL`. A bound under unverified assumptions → `WARN`. `safe_operating_points` stay `None` when no complete-tie bound meets the existing target. | `diagnosis/src/mapdoctor/diagnostics/risk_coverage.py::evaluate_risk_coverage`, `_target_evidence_status` |
| Changed-region review-only sufficiency | Repeated comparable multi-session evidence should precede map mutation (Biber and Duckett 2005; Berrio et al. 2021). Default `seq_tile` keys include `seq`, so a normal group has one session. | Report `hard_status=VALID`, `proposal_status` ∈ {`EMPTY_NO_CANDIDATES`,`REVIEW_CANDIDATES`,`QUALITY_SHORTFALL`,`INSUFFICIENT_EVIDENCE`}, `authority=review_only`, `geometry_invalidated=false`, `independence_attested=false`, `lineage_attested=false`. Rows: `INSUFFICIENT_EVIDENCE` / `QUALITY_SHORTFALL` / `REVIEW_CANDIDATE` with signed session and support margins. Subthreshold rows are counted in `suppressed`, not erased. No `PASS`. CLI `0` still uses the product total and labels `min_total_source`. | `map_update/core/changed_region_evidence.py::aggregate_changed_region_evidence` |
| Structured bridge margins | Non-finite comparisons must not silently pass. Umeyama/MAGSAC assume valid correspondences; numeric cutoffs stay site policy. | `bridge_quality_checks` records `{name,value,threshold,direction=gte,signed_margin,finite,enabled,passed,reason,hard_status,evidence_status,authority}`. Non-finite → `HARD_FAIL`/`INSUFFICIENT_EVIDENCE`. Measured miss → `QUALITY_SHORTFALL`. Merge refuse/allow still uses `bridge_quality_warnings` string reasons. | `map_update/core/update_quality_gates.py::bridge_quality_checks`, `bridge_quality_warnings` |
| Target S9 identity and lineage | Sattler/LaMAR evaluate changing conditions on separated mapping/query sessions. Artifact hash lineage is a repository hard invariant. | `evaluate_results` binds exactly `ts_common.TEST` identities; rejects duplicates, non-TEST, zero frames, and `rate != localized/frames`. Emits `hard_status`, `evidence_status`, `independent_session_count`, `identity_receipt`, `soft_checks.*.signed_margin`. G9.3 is typed `NOT_APPLICABLE` because TEST directions are unknown. G9 numeric defaults are unchanged. `main` hashes optional EDM/tracking/package artifacts. `verify_final_release.py::release_lineage_checks` requires S8/S9 hash binding; missing S9 hashes fail `release_lineage_bound`. | `sites/target_site/tools/validate_heldout_localization.py::evaluate_results`, `main`; `verify_final_release.py::release_lineage_checks`, `s8_s9_freshness_ok` |

Not landed in this wave, and therefore not claimed: singleton rank
neutralization in `relative_quality.py` (still `1.0`); renaming
`HIGH_PROPOSAL_ONLY`; a `release_authorized` field; MapDoctor
`comparison.py` `COMPARATIVE_ONLY`; `point_evidence_ledger`
`retirement_eligible` / `insufficient_evidence` labels.

## Claims that remain local policies, not paper facts

These values are intentionally unchanged.  The literature supports the variables or
method family, not the repository's constants:

| Local claim | Code location | Required evidence before a stronger claim |
| --- | --- | --- |
| Relative percentiles, `Delta J` weights, and the marginal-stop ratio improve localization | `diagnosis/src/sfm_qa/session_select/prebuild.py`, `relative_quality.py` | A0–A5 ablation on development-held-out sessions, followed by one frozen S9 certification run. Stopping receipts do not change membership. |
| Singleton rank `1.0` is exceptional quality | `diagnosis/src/sfm_qa/relative_quality.py::percentile_ranks` | RankIQA supports order only inside comparable groups. Neutral `0.5` remains a recommended specification correction, not applied. |
| FIM and structural localizability proxies predict actual success | `diagnosis/src/sfm_diagnosis/fisher.py`, `actloc.py` | Group-held-out calibration by route/session/region; report calibration and risk–coverage, not only correlation. |
| `SESSION_DECAY=0.5` and two unmatched sessions are appropriate retirement rules | `map_update/core/point_evidence_ledger.py` | Site-specific half-life and false-retirement study with real point visibility, opportunity, and matching identities. Opportunity-aware consecutiveness is not implemented. |
| Query-level conformal/Bonferroni intervals retain deployment guarantees under temporal or spatial dependence | `diagnosis/src/mapdoctor/diagnostics/risk_coverage.py` | Independent group units or a declared non-exchangeable weighting experiment; `independence_verified` stays false. Query-level resampling is insufficient. |
| Two sessions and two low-support frames establish a changed region | `map_update/core/changed_region_evidence.py` | Multi-session change labels, precision/recall, false-replacement rate, and frozen stable-holdout replay. Current promotions are review-only. |
| Sim(3) bridge thresholds substitute for a disjoint hold-out | `map_update/lifelong/src/update_map/bridge.py`, site S5.7 tools | Fit and validation anchors must be disjoint, with site-calibrated cycle/pose thresholds and S9 replay. Structured margins do not retune those values. |

## Experiment queue

| Priority | Method | Primary-source support | Decision and release gate |
| --- | --- | --- | --- |
| P1 | Cache adaptive bridge query preparation and remove the final selected-parameter replay | HLoc and LightGlue support reducing expensive local matching after retrieval. Code inspection shows up to 10 passes (`1 + 4 + 4 + 1`) over the same frames. | Implement only with a real EDM/XFeat GPU fixture. Require identical bridge set, Sim(3) residuals, observation payload, and S9 outcome while measuring calls, wall time, RAM, and VRAM. |
| P1 | Expose point-evidence half-life and retirement policy in a receipt | ExMaps supports recency-weighted point stability, not the fixed values `0.5` and `2`. | Add configuration only together with real visibility identities and a calibration protocol; do not turn unconstrained knobs into production defaults. |
| P1 | Add measured frontend matchability to reference/map selection | Map Point Selection for Visual SLAM, long-term heterogeneous-GNN sparsification, and ExMaps jointly support localization utility, coverage, and stability rather than raw point count. | Measure retrieval -> matching -> lifting -> PnP survival. Compare all references, FIM-only, and FIM+matchability on fixed weak/stable holdouts. |
| P2 | PixSfM featuremetric refinement | Pixel-Perfect SfM supports featuremetric keypoint and geometry refinement under detector noise and appearance change. | External A/B only: fixed corpus/intrinsics; run S6–S9; report track/angle/reprojection, compute, weak-block gain, and stable-block regression. |
| P2 | Non-exchangeable conformal risk control | CRC (Angelopoulos et al. 2024 / arXiv:2208.02814) needs exchangeable bounded right-continuous monotone loss and gives expected-risk, not a single-release high-probability certificate. Farinhas et al. 2023 require declared relevance weights that are not fit to certification losses. Barber et al. 2023 and Chernozhukov et al. 2018 likewise need declared weights or block permutations that preserve dependence. | **Deferred, experiment-only.** Offline route/session/spatial-block study with four disjoint group roles. Reject undeclared data-dependent weights. Never claim exchangeable-query validity for adjacent frames. |
| P2 | Group-level Learn-Then-Test / nested calibration | Angelopoulos et al. 2025 LTT needs finite predeclared candidates and valid bounded-loss p-values. Kull beta and isotonic calibration need a separate representative sample (Kull et al. 2017; Zadrozny and Elkan 2002). Roberts et al. 2017 require structure-aware folds. | **Deferred, experiment-only.** Nested session-held-out calibration and group LTT remain literature transfers. Current OOF calibration is development evidence only. |
| P2 | LFOE, MP-SfM, Detector-Free SfM, PixSfM, RoMo, Light3R-SfM, depth-guided SfM | Each paper targets a diagnosed regime (translation outliers, low overlap/parallax, detector failure, keypoint noise, dynamic content, feed-forward cost, or small-parallax movies). Liu et al. 2022 explicitly tune COLMAP++ separately and exclude heavy blur. | **Deferred, experiment-only.** Trigger by diagnosed failure only. Every backend must later pass the same S6/S9, gauge, resource, and artifact-provenance gates. No default switch. |
| P2 | Lifelong posterior / FreMEn / BOCPD calibration | ExMaps ranks matches with decayed visibility; it does not justify two-miss retirement. Rosen persistence needs calibrated miss/false-alarm likelihoods. FreMEn (Krajník et al. 2017) selects order from future observations. Adams and MacKay 2007 BOCPD assumes piecewise i.i.d. runs. | **Deferred.** Opportunity-aware ledger labels and half-life exposure are literature transfers requiring held-out calibration. BOCPD and automatic tile replacement are experiment-only. |
| Not portable | RTMap or 3DGS change-localization architecture | These papers support multi-view/multi-traversal evidence accumulation and uncertainty-aware change handling. | Transfer the evidence principle only. HD-map/3DGS elements, sensors, and online fusion are not the sparse SfM/EDM representation used here. |
| Not now | Replacement selection objective or ILP | Nemhauser et al. 1978 and Gygli et al. 2015 require nonnegative monotone submodular components. The current mixed `ΔJ` has negative risk/cost/redundancy and average diversity. | Replacing the heuristic is a literature transfer requiring held-out A/B. Additive stop receipts are the only now-safe change. |

## Primary-source set reviewed

### Reconstruction, graph quality, and efficiency

- Schönberger and Frahm, [*Structure-from-Motion Revisited*](https://openaccess.thecvf.com/content_cvpr_2016/html/Schonberger_Structure-From-Motion_Revisited_CVPR_2016_paper.html), CVPR 2016.
- Sweeney et al., [*Optimizing the Viewing Graph for Structure-from-Motion*](https://openaccess.thecvf.com/content_iccv_2015/html/Sweeney_Optimizing_the_Viewing_ICCV_2015_paper.html), ICCV 2015.
- Manam and Govindu, [*Leveraging Camera Triplets for Efficient and Accurate Structure-from-Motion*](https://openaccess.thecvf.com/content/CVPR2024/html/Manam_Leveraging_Camera_Triplets_for_Efficient_and_Accurate_Structure-from-Motion_CVPR_2024_paper.html), CVPR 2024.
- Pan, Schönberger, and Pollefeys, [*Global Structure-from-Motion Meets Feedforward Reconstruction*](https://arxiv.org/abs/2605.26103), CVPR 2026 / GLUEMAP.
- Pan et al., [*Global Structure-from-Motion Revisited*](https://arxiv.org/abs/2407.20219), ECCV 2024 / GLOMAP.
- He et al., [*Detector-Free Structure from Motion*](https://openaccess.thecvf.com/content/CVPR2024/html/He_Detector-Free_Structure_from_Motion_CVPR_2024_paper.html), CVPR 2024.
- Lindenberger et al., [*Pixel-Perfect Structure-from-Motion with Featuremetric Refinement*](https://openaccess.thecvf.com/content/ICCV2021/html/Lindenberger_Pixel-Perfect_Structure-From-Motion_With_Featuremetric_Refinement_ICCV_2021_paper.html), ICCV 2021.
- Pataki et al., [*MP-SfM: Monocular Surface Priors for Robust Structure-from-Motion*](https://openaccess.thecvf.com/content/CVPR2025/html/Pataki_MP-SfM_Monocular_Surface_Priors_for_Robust_Structure-from-Motion_CVPR_2025_paper.html), CVPR 2025.
- Xiangli et al., [*Doppelgangers++*](https://openaccess.thecvf.com/content/CVPR2025/html/Xiangli_Doppelgangers_Improved_Visual_Disambiguation_with_Geometric_3D_Features_CVPR_2025_paper.html), CVPR 2025.
- Damblon et al., [*Learning to Filter Outlier Edges in Global SfM*](https://openaccess.thecvf.com/content/CVPR2025/html/Damblon_Learning_to_Filter_Outlier_Edges_in_Global_SfM_CVPR_2025_paper.html), CVPR 2025.
- Goli et al., [*RoMo: Robust Motion Segmentation Improves Structure from Motion*](https://openaccess.thecvf.com/content/ICCV2025/html/Goli_RoMo_Robust_Motion_Segmentation_Improves_Structure_from_Motion_ICCV_2025_paper.html), ICCV 2025.
- Johnson, Douze, and Jégou, [*Billion-scale Similarity Search with GPUs*](https://arxiv.org/abs/1702.08734), 2017.
- Jégou, Douze, and Schmid, [*Product Quantization for Nearest Neighbor Search*](https://doi.org/10.1109/TPAMI.2010.57), TPAMI 2011.
- Wilson and Snavely, [*Robust Global Translations with 1DSfM*](https://research.cs.cornell.edu/1dsfm/docs/1DSfM_ECCV14.pdf), ECCV 2014.
- Fiedler, [*Algebraic connectivity of graphs*](https://doi.org/10.21136/CMJ.1973.101168), Czechoslovak Mathematical Journal 1973.
- Nemhauser, Wolsey, and Fisher, [*An analysis of approximations for maximizing submodular set functions—I*](https://doi.org/10.1007/BF01588971), Mathematical Programming 1978.

### Localization, map value, and long-term maintenance

- Sarlin et al., [*From Coarse to Fine: Robust Hierarchical Localization at Large Scale*](https://openaccess.thecvf.com/content_CVPR_2019/html/Sarlin_From_Coarse_to_Fine_Robust_Hierarchical_Localization_at_Large_Scale_CVPR_2019_paper.html), CVPR 2019.
- Lindenberger et al., [*LightGlue: Local Feature Matching at Light Speed*](https://openaccess.thecvf.com/content/ICCV2023/html/Lindenberger_LightGlue_Local_Feature_Matching_at_Light_Speed_ICCV_2023_paper.html), ICCV 2023.
- Zaffar et al., [*On the Estimation of Image-matching Uncertainty in Visual Place Recognition*](https://openaccess.thecvf.com/content/CVPR2024/html/Zaffar_On_the_Estimation_of_Image-matching_Uncertainty_in_Visual_Place_Recognition_CVPR_2024_paper.html), CVPR 2024.
- Sattler et al., [*Benchmarking 6DOF Outdoor Visual Localization in Changing Conditions*](https://openaccess.thecvf.com/content_cvpr_2018/html/Sattler_Benchmarking_6DOF_Outdoor_CVPR_2018_paper.html), CVPR 2018.
- Chang et al., [*Long-Term Visual Map Sparsification with Heterogeneous GNN*](https://openaccess.thecvf.com/content/CVPR2022/html/Chang_Long-Term_Visual_Map_Sparsification_With_Heterogeneous_GNN_CVPR_2022_paper.html), CVPR 2022.
- Rotsidis et al., [*ExMaps: Long-Term Localization in Dynamic Scenes Using Exponential Decay*](https://openaccess.thecvf.com/content/WACV2021/html/Rotsidis_ExMaps_Long-Term_Localization_in_Dynamic_Scenes_Using_Exponential_Decay_WACV_2021_paper.html), WACV 2021.
- Müller and van Daalen, [*Map Point Selection for Visual SLAM*](https://arxiv.org/abs/2306.12901), 2023.
- Bürki et al., [*Map Management for Efficient Long-Term Visual Localization in Outdoor Environments*](https://arxiv.org/abs/1808.02658), IEEE IV 2018.
- Berrio et al., [*Long-Term Map Maintenance Pipeline for Autonomous Vehicles*](https://arxiv.org/abs/2008.12449), IEEE T-ITS 2021/2022.
- Rosen, Mason, and Leonard, [*Towards Lifelong Feature-Based Mapping in Semi-Static Environments*](https://david-m-rosen.github.io/publication/persistencefilter-icra/PersistenceFilter-ICRA.pdf), ICRA 2016.
- Biber and Duckett, [*Dynamic Maps for Long-Term Operation of Mobile Service Robots*](https://www.roboticsproceedings.org/rss01/p03.pdf), RSS 2005.
- Krajník et al., [*FreMEn*](https://doi.org/10.1109/TRO.2017.2665664), IEEE T-RO 2017.

### Reliability, calibration, and change

- Zhang, [*A Flexible New Technique for Camera Calibration*](https://doi.org/10.1109/34.888718), TPAMI 2000.
- COLMAP, [camera model specification](https://github.com/colmap/colmap/blob/main/doc/cameras.rst) and [model definitions](https://github.com/colmap/colmap/blob/main/src/colmap/sensor/models.h).
- Angelopoulos et al., [*Conformal Risk Control*](https://arxiv.org/abs/2208.02814), 2022.
- Farinhas et al., [*Non-Exchangeable Conformal Risk Control*](https://arxiv.org/abs/2310.01262), 2023.
- Galappaththige et al., [*Multi-View Pose-Agnostic Change Localization with Zero Labels*](https://openaccess.thecvf.com/content/CVPR2025/html/Galappaththige_Multi-View_Pose-Agnostic_Change_Localization_with_Zero_Labels_CVPR_2025_paper.html), CVPR 2025.
- Du et al., [*RTMap: Real-Time Recursive Mapping with Change Detection and Localization*](https://openaccess.thecvf.com/content/ICCV2025/html/Du_RTMap_Real-Time_Recursive_Mapping_with_Change_Detection_and_Localization_ICCV_2025_paper.html), ICCV 2025.
- Geifman and El-Yaniv, [*Selective Classification for Deep Neural Networks*](https://proceedings.neurips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html), NeurIPS 2017.
- El-Yaniv and Wiener, [*On the Foundations of Noise-free Selective Classification*](https://jmlr.org/papers/v11/el-yaniv10a.html), JMLR 2010.
- Clopper and Pearson, [*The use of confidence or fiducial limits illustrated in the case of the binomial*](https://doi.org/10.1093/biomet/26.4.404), Biometrika 1934.
- Wilson, [*Probable Inference, the Law of Succession, and Statistical Inference*](https://doi.org/10.1080/01621459.1927.10502953), JASA 1927.
- Roberts et al., [*Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure*](https://doi.org/10.1111/ecog.02881), Ecography 2017.
- Angelopoulos et al., [*Learn then test*](https://doi.org/10.1214/24-AOAS1998), AOAS 2025.
- Angelopoulos et al., [*Conformal Risk Control*](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html), ICLR 2024.
- Barber et al., [*Conformal prediction beyond exchangeability*](https://doi.org/10.1214/23-AOS2276), Annals of Statistics 2023.
- Schuirmann, [*A comparison of the two one-sided tests procedure and the power approach*](https://doi.org/10.1007/BF01068419), J Pharmacokinet Biopharm 1987.
- Chow, [*An optimum character recognition system using decision functions*](https://doi.org/10.1109/TEC.1957.5222035), IRE TEC 1957.
- Kull, Silva Filho, and Flach, [*Beta calibration*](https://proceedings.mlr.press/v54/kull17a.html), AISTATS 2017.
- Liu, van de Weijer, and Bagdanov, [*RankIQA*](https://arxiv.org/abs/1707.08347), ICCV 2017.

### Robust estimation, multi-session alignment, and graph optimization

- Fischler and Bolles, [*Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography*](https://doi.org/10.1145/358669.358692), CACM 1981.
- Umeyama, [*Least-Squares Estimation of Transformation Parameters Between Two Point Patterns*](https://doi.org/10.1109/34.88573), TPAMI 1991.
- Barath et al., [*MAGSAC++, a Fast, Reliable and Accurate Robust Estimator*](https://openaccess.thecvf.com/content_CVPR_2020/html/Barath_MAGSAC_a_Fast_Reliable_and_Accurate_Robust_Estimator_CVPR_2020_paper.html), CVPR 2020.
- Kümmerle et al., [*g2o: A General Framework for Graph Optimization*](https://doi.org/10.1109/ICRA.2011.5979949), ICRA 2011.
- Sünderhauf and Protzel, [*Switchable Constraints for Robust Pose Graph SLAM*](https://doi.org/10.1109/IROS.2012.6385590), IROS 2012.
- Schneider et al., [*Real-time 6-DOF Multi-session Visual SLAM over Large Scale Environments*](https://publications.ri.cmu.edu/real-time-6-dof-multi-session-visual-slam-over-large-scale-environments), ICRA 2013.
- Elvira et al., [*ORBSLAM-Atlas: A Robust and Accurate Multi-map System*](https://arxiv.org/abs/1908.11585), 2019.
- Campos et al., [*ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial, and Multi-Map SLAM*](https://doi.org/10.1109/TRO.2021.3075644), TRO 2021.
- Lipson et al., [*Multi-Session SLAM with Differentiable Wide-Baseline Pose Optimization*](https://openaccess.thecvf.com/content/CVPR2024/html/Lipson_Multi-Session_SLAM_with_Differentiable_Wide-Baseline_Pose_Optimization_CVPR_2024_paper.html), CVPR 2024.

### Selection, diagnostic quality, and evaluation protocol

- Snavely et al., [*Skeletal Graphs for Efficient Structure from Motion*](https://www.cs.cornell.edu/~snavely/projects/skeletalset/), CVPR 2008.
- Park and Yoon, [*Optimal Key-frame Selection for Video-based Structure-from-Motion*](https://doi.org/10.1049/el.2011.2674), Electronics Letters 2011.
- Shah et al., [*A Unified View-Graph Selection Framework for Structure from Motion*](https://arxiv.org/abs/1708.01125), ECCV 2018.
- Doherty et al., [*Spectral Measurement Sparsification for Pose-Graph SLAM*](https://arxiv.org/abs/2203.13897), IROS 2022.
- Merzić et al., [*Map Quality Evaluation for Visual Localization*](https://tisl.cs.utoronto.ca/publication/201705-icra-map_quality_evaluation/icra17-map_quality_evaluation.pdf), ICRA 2017.
- Hartmann et al., [*Predicting Matchability*](https://openaccess.thecvf.com/content_cvpr_2014/papers/Hartmann_Predicting_Matchability_2014_CVPR_paper.pdf), CVPR 2014.
- Sarlin et al., [*LaMAR: Benchmarking Localization and Mapping for Augmented Reality*](https://arxiv.org/abs/2210.10770), ECCV 2022.
- Shi et al., [*Are We Ready for Service Robots? The OpenLORIS-Scene Datasets for Lifelong SLAM*](https://arxiv.org/abs/1911.05603), ICRA 2020.
- Piasco et al., [*Long-Term Visual Localization Revisited*](https://research.chalmers.se/publication/529149/file/529149_Fulltext.pdf), TPAMI 2022.
- Wenzel et al., [*4Seasons: Benchmarking Visual SLAM and Long-Term Localization for Autonomous Driving in Challenging Conditions*](https://arxiv.org/abs/2301.01147), 2023.
- Gygli, Grabner, and Van Gool, [*Video Summarization by Learning Submodular Mixtures of Objectives*](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Gygli_Video_Summarization_by_2015_CVPR_paper.html), CVPR 2015.

## Stage-to-evidence coverage matrix

This matrix defines the strongest claim that source evidence alone permits.  A source can
justify a variable, invariant, or experiment; it cannot certify a repository-specific
threshold or improvement without the frozen A/B protocol below.

| System part | Primary theoretical support | Repository anchors | Permitted claim and boundary |
| --- | --- | --- | --- |
| S0 corpus and held-out isolation | Aachen changing-conditions benchmark, LaMAR, OpenLORIS, 4Seasons | `sites/*/tools/s0_corpus_lock.py`, `map_update/lifelong/src/update_map/splits.py` | Whole-session and condition-separated evaluation is necessary.  A 95% target and the current session identities remain deployment policy. |
| S1/S2 motion-aware frame selection | Optimal key-frame selection, RANSAC/MAGSAC++, *Structure-from-Motion Revisited* | `sites/*/tools/s1_motion_scan.py`, `s2_extract.py`, `diagnosis/src/sfm_qa/session_select/motion.py` | Baseline, degeneracy, blur, and redundancy are valid selection signals.  Their weights and cutoffs require development-held-out calibration. |
| S2b camera and projection contract | Zhang calibration, COLMAP camera definitions, OpenCV projection equations | `sites/*/tools/ts_intrinsics.py`, `s2b_intrinsics_bakeoff.py`, `sfm_qa/bridge.py`, `update_map/geometry.py` | Parameter order and projection equations are specification facts.  Choosing PINHOLE versus RADIAL for a site remains an empirical bake-off. |
| S3 proposal/view graph | Skeletal graphs, viewing-graph optimization, unified view-graph selection, Manam verified triplets, HLoc, Bürki slack | `stage_pairs`, `s3_pairs.py`, `session_select/prebuild.py` | Sparse proposal graphs can reduce work.  Retrieval triangles (`num_candidate_pairs`) may reorder probes only.  Stopping receipts report slack, not success.  Candidate counts never become geometric merge authority. |
| S4 repeated-structure/outlier control | Doppelgangers++, LFOE-GlobalSfM, switchable constraints | `stage_doppelgangers`, `audit_dg_graph.py`, session critical-bridge diagnostics | Alias and high-influence edges deserve explicit rejection or quarantine.  Learned scores remain proposal priority.  Thresholds require the same S6/S9 A/B as any frontend change. |
| S5 reconstruction and refinement | COLMAP, GLOMAP, GLUEMAP, PixSfM, Detector-Free SfM, MP-SfM, RoMo | `stage_glomap`, site GlueMap launchers, `finalize_edm_model.py` | Register/triangulate/filter/BA and fixed-calibration gates are grounded.  Alternate learned backends remain experiment-only until resource and held-out parity. |
| S5.7 bridge, Sim(3), and gauge | Umeyama, RANSAC, 1DSfM, GLOMAP, g2o, multi-session SLAM | `admission.py::evaluate_geometry_authority`, `update_quality_gates.py::bridge_quality_checks`, `audit_independent_sim3.py` | Fusion authority is per exact-pair edge with ≥2 independent disjoint-holdout groups.  Structured margins do not retune residuals.  Site-calibrated Sim(3) cutoffs cannot replace S9. |
| S6 map and graph diagnosis | Map-quality evaluation, matchability prediction, Fiedler algebraic connectivity | `sfm_diagnosis`, `mapdoctor/diagnostics`, `session_select/critical_bridges.py` | Support, matchability, bridges, articulation, and algebraic connectivity are defensible diagnostics.  They are proxies, not calibrated success probabilities.  Compute them on authorized edges only when claiming topology authority. |
| S7/S8 reference and point selection | HLoc, map-point selection, heterogeneous-GNN sparsification, ExMaps | `sparsify_reloc_bundle.py`, `selection.py`, `point_evidence_ledger.py` | Coverage, information, matchability, and recency should be measured per cost.  Learned ranks, half-lives, and retirement streaks require frozen candidate-only A/B. |
| S9 localization and risk reporting | Sattler changing-conditions, LaMAR, Geifman selective risk, Clopper–Pearson, Roberts structured CV | `validate_heldout_localization.py`, `verify_final_release.py`, `risk_coverage.py`, `benchmark.py` | Identity/hash lineage is a specification correction.  Pose targets, coverage curves, and group-held-out evaluation are appropriate.  Adjacent frames are not independent certification samples.  CRC/LTT/nonexchangeable weights remain experiment-only. |
| Lifelong change and map promotion | ExMaps, Berrio, Biber and Duckett, RTMap | `changed_region_evidence.py`, `update_quality_gates.py`, `stability.py` | Multi-view, multi-session evidence should precede invalidation.  Implemented promotions are review-only.  Automatic replacement, FreMEn retirement, and BOCPD remain experiment-only. |

## Required system-level validation

Unit tests establish parser, projection, state, and exact-computation invariants.  They do
not establish localization improvement.  A candidate release still requires:

1. frozen build, development-held-out, and certification manifests by whole session;
2. identical intrinsics, matcher/localizer configuration, and artifact hashes across A/B;
3. pair count, graph connectivity, track/angle/reprojection, wall time, RAM, and VRAM;
4. S9 strict success and pose accuracy by route, direction, appearance, and weak/stable block;
5. no stable-holdout regression and no gauge, scale, gravity, bundle, or provenance break.
6. independent-unit counts, signed soft-gate margins, and an explicit
   `INSUFFICIENT_EVIDENCE` versus `QUALITY_SHORTFALL` split;
7. no claim that query-level CP/Wilson/CRC intervals certify release
   unless groups and provenance were frozen before certification labels.
