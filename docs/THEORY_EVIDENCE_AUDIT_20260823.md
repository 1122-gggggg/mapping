# Theory and primary-source evidence audit — 2026-08-23

This audit records the follow-up multi-team review of the mapping, diagnosis, and
lifelong-update stack.  It is deliberately stricter than a bibliography: every proposed
change is classified as a direct engineering correction, a literature transfer that still
needs site data, or a non-portable research direction.

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

## Claims that remain local policies, not paper facts

These values are intentionally unchanged.  The literature supports the variables or
method family, not the repository's constants:

| Local claim | Code location | Required evidence before a stronger claim |
| --- | --- | --- |
| Relative percentiles, `Delta J` weights, and the marginal-stop ratio improve localization | `diagnosis/src/sfm_qa/session_select/prebuild.py`, `relative_quality.py` | A0–A5 ablation on development-held-out sessions, followed by one frozen S9 certification run. |
| FIM and structural localizability proxies predict actual success | `diagnosis/src/sfm_diagnosis/fisher.py`, `actloc.py` | Group-held-out calibration by route/session/region; report calibration and risk–coverage, not only correlation. |
| `SESSION_DECAY=0.5` and two unmatched sessions are appropriate retirement rules | `map_update/core/point_evidence_ledger.py` | Site-specific half-life and false-retirement study with real point visibility and matching identities. |
| Query-level conformal/Bonferroni intervals retain deployment guarantees under temporal or spatial dependence | `diagnosis/src/mapdoctor/diagnostics/risk_coverage.py` | Independent group units or a declared non-exchangeable weighting experiment; query-level resampling is insufficient. |
| Two sessions and two low-support frames establish a changed region | `map_update/core/changed_region_evidence.py` | Multi-session change labels, precision/recall, false-replacement rate, and frozen stable-holdout replay. |
| Sim(3) bridge thresholds substitute for a disjoint hold-out | `map_update/lifelong/src/update_map/bridge.py`, site S5.7 tools | Fit and validation anchors must be disjoint, with site-calibrated cycle/pose thresholds and S9 replay. |

## Experiment queue

| Priority | Method | Primary-source support | Decision and release gate |
| --- | --- | --- | --- |
| P1 | Cache adaptive bridge query preparation and remove the final selected-parameter replay | HLoc and LightGlue support reducing expensive local matching after retrieval. Code inspection shows up to 10 passes (`1 + 4 + 4 + 1`) over the same frames. | Implement only with a real EDM/XFeat GPU fixture. Require identical bridge set, Sim(3) residuals, observation payload, and S9 outcome while measuring calls, wall time, RAM, and VRAM. |
| P1 | Expose point-evidence half-life and retirement policy in a receipt | ExMaps supports recency-weighted point stability, not the fixed values `0.5` and `2`. | Add configuration only together with real visibility identities and a calibration protocol; do not turn unconstrained knobs into production defaults. |
| P1 | Add measured frontend matchability to reference/map selection | Map Point Selection for Visual SLAM, long-term heterogeneous-GNN sparsification, and ExMaps jointly support localization utility, coverage, and stability rather than raw point count. | Measure retrieval -> matching -> lifting -> PnP survival. Compare all references, FIM-only, and FIM+matchability on fixed weak/stable holdouts. |
| P2 | PixSfM featuremetric refinement | Pixel-Perfect SfM supports featuremetric keypoint and geometry refinement under detector noise and appearance change. | External A/B only: fixed corpus/intrinsics; run S6–S9; report track/angle/reprojection, compute, weak-block gain, and stable-block regression. |
| P2 | Non-exchangeable conformal risk control | Conformal Risk Control gives finite-sample guarantees under its assumptions; non-exchangeable CRC studies relevance-weighted losses under shift. | Offline route/session/spatial-block experiment. Reject undeclared data-dependent weights and never claim exchangeable-query validity for adjacent frames. |
| P2 | LFOE, MP-SfM, Detector-Free SfM, RoMo, Light3R-SfM | The papers target translation outliers, low overlap/parallax, detector failure, dynamic content, and feed-forward graph cost respectively. | Trigger by diagnosed failure regime only. Every backend must pass the same S6/S9, gauge, resource, and artifact-provenance gates. |
| Not portable | RTMap or 3DGS change-localization architecture | These papers support multi-view/multi-traversal evidence accumulation and uncertainty-aware change handling. | Transfer the evidence principle only. HD-map/3DGS elements, sensors, and online fusion are not the sparse SfM/EDM representation used here. |

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

### Localization, map value, and long-term maintenance

- Sarlin et al., [*From Coarse to Fine: Robust Hierarchical Localization at Large Scale*](https://openaccess.thecvf.com/content_CVPR_2019/html/Sarlin_From_Coarse_to_Fine_Robust_Hierarchical_Localization_at_Large_Scale_CVPR_2019_paper.html), CVPR 2019.
- Lindenberger et al., [*LightGlue: Local Feature Matching at Light Speed*](https://openaccess.thecvf.com/content/ICCV2023/html/Lindenberger_LightGlue_Local_Feature_Matching_at_Light_Speed_ICCV_2023_paper.html), ICCV 2023.
- Zaffar et al., [*On the Estimation of Image-matching Uncertainty in Visual Place Recognition*](https://openaccess.thecvf.com/content/CVPR2024/html/Zaffar_On_the_Estimation_of_Image-matching_Uncertainty_in_Visual_Place_Recognition_CVPR_2024_paper.html), CVPR 2024.
- Sattler et al., [*Benchmarking 6DOF Outdoor Visual Localization in Changing Conditions*](https://openaccess.thecvf.com/content_cvpr_2018/html/Sattler_Benchmarking_6DOF_Outdoor_CVPR_2018_paper.html), CVPR 2018.
- Chang et al., [*Long-Term Visual Map Sparsification with Heterogeneous GNN*](https://openaccess.thecvf.com/content/CVPR2022/html/Chang_Long-Term_Visual_Map_Sparsification_With_Heterogeneous_GNN_CVPR_2022_paper.html), CVPR 2022.
- Rotsidis et al., [*ExMaps: Long-Term Localization in Dynamic Scenes Using Exponential Decay*](https://openaccess.thecvf.com/content/WACV2021/html/Rotsidis_ExMaps_Long-Term_Localization_in_Dynamic_Scenes_Using_Exponential_Decay_WACV_2021_paper.html), WACV 2021.
- Müller and van Daalen, [*Map Point Selection for Visual SLAM*](https://arxiv.org/abs/2306.12901), 2023.

### Reliability, calibration, and change

- Zhang, [*A Flexible New Technique for Camera Calibration*](https://doi.org/10.1109/34.888718), TPAMI 2000.
- COLMAP, [camera model specification](https://github.com/colmap/colmap/blob/main/doc/cameras.rst) and [model definitions](https://github.com/colmap/colmap/blob/main/src/colmap/sensor/models.h).
- Angelopoulos et al., [*Conformal Risk Control*](https://arxiv.org/abs/2208.02814), 2022.
- Farinhas et al., [*Non-Exchangeable Conformal Risk Control*](https://arxiv.org/abs/2310.01262), 2023.
- Galappaththige et al., [*Multi-View Pose-Agnostic Change Localization with Zero Labels*](https://openaccess.thecvf.com/content/CVPR2025/html/Galappaththige_Multi-View_Pose-Agnostic_Change_Localization_with_Zero_Labels_CVPR_2025_paper.html), CVPR 2025.
- Du et al., [*RTMap: Real-Time Recursive Mapping with Change Detection and Localization*](https://openaccess.thecvf.com/content/ICCV2025/html/Du_RTMap_Real-Time_Recursive_Mapping_with_Change_Detection_and_Localization_ICCV_2025_paper.html), ICCV 2025.

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

## Stage-to-evidence coverage matrix

This matrix defines the strongest claim that source evidence alone permits.  A source can
justify a variable, invariant, or experiment; it cannot certify a repository-specific
threshold or improvement without the frozen A/B protocol below.

| System part | Primary theoretical support | Repository anchors | Permitted claim and boundary |
| --- | --- | --- | --- |
| S0 corpus and held-out isolation | Aachen changing-conditions benchmark, LaMAR, OpenLORIS, 4Seasons | `sites/*/tools/s0_corpus_lock.py`, `map_update/lifelong/src/update_map/splits.py` | Whole-session and condition-separated evaluation is necessary.  A 95% target and the current session identities remain deployment policy. |
| S1/S2 motion-aware frame selection | Optimal key-frame selection, RANSAC/MAGSAC++, *Structure-from-Motion Revisited* | `sites/*/tools/s1_motion_scan.py`, `s2_extract.py`, `diagnosis/src/sfm_qa/session_select/motion.py` | Baseline, degeneracy, blur, and redundancy are valid selection signals.  Their weights and cutoffs require development-held-out calibration. |
| S2b camera and projection contract | Zhang calibration, COLMAP camera definitions, OpenCV projection equations | `sites/*/tools/ts_intrinsics.py`, `s2b_intrinsics_bakeoff.py`, `sfm_qa/bridge.py`, `update_map/geometry.py` | Parameter order and projection equations are specification facts.  Choosing PINHOLE versus RADIAL for a site remains an empirical bake-off. |
| S3 proposal/view graph | Skeletal graphs, viewing-graph optimization, unified view-graph selection, camera triplets, HLoc | `stage_pairs`, `s3_pairs.py`, `session_select/prebuild.py` | Sparse proposal graphs and triplet/cycle ranking can reduce work.  Retrieval or candidate counts never become geometric merge authority. |
| S4 repeated-structure/outlier control | Doppelgangers++, LFOE-GlobalSfM, switchable constraints | `stage_doppelgangers`, `audit_dg_graph.py`, session critical-bridge diagnostics | Alias and high-influence edges deserve explicit rejection or quarantine.  Learned scores and thresholds require the same S6/S9 A/B as any frontend change. |
| S5 reconstruction and refinement | COLMAP, GLOMAP, GLUEMAP, PixSfM, Detector-Free SfM, MP-SfM, RoMo | `stage_glomap`, site GlueMap launchers, `finalize_edm_model.py` | Register/triangulate/filter/BA and fixed-calibration gates are grounded.  Alternate learned backends remain experiment-only until resource and held-out parity. |
| S5.7 bridge, Sim(3), and gauge | Umeyama, RANSAC, g2o, multi-session SLAM, ORB-SLAM Atlas/3, differentiable wide-baseline optimization | `update_map/geometry.py`, `bridge.py`, `audit_independent_sim3.py`, gauge verifiers | Robust Sim(3), fixed anchors, and disjoint validation are sound safety patterns.  Bridge residual thresholds are site-calibrated and cannot replace S9. |
| S6 map and graph diagnosis | Map-quality evaluation, matchability prediction, spectral measurement sparsification | `sfm_diagnosis`, `mapdoctor/diagnostics`, `session_select/critical_bridges.py` | Support, matchability, bridges, articulation, and algebraic connectivity are defensible diagnostics.  They are proxies, not calibrated success probabilities. |
| S7/S8 reference and point selection | HLoc, map-point selection, heterogeneous-GNN sparsification, ExMaps | `sparsify_reloc_bundle.py`, `selection.py`, `point_evidence_ledger.py` | Coverage, information, matchability, and recency should be measured per cost.  Learned ranks, half-lives, and retirement streaks require frozen candidate-only A/B. |
| S9 localization and risk reporting | Changing-conditions benchmark, LaMAR, conformal risk control, non-exchangeable CRC | `validate_heldout_localization.py`, `risk_coverage.py`, `calibration.py` | Pose thresholds, coverage curves, and group-held-out evaluation are appropriate.  Adjacent frames are not independent certification samples; formal guarantees need declared group/exchangeability assumptions. |
| Lifelong change and map promotion | ExMaps, multi-view pose-agnostic change localization, RTMap, long-term localization benchmarks | `changed_region_evidence.py`, `stability.py`, `CurrentFirstLocalizer`, `CandidateBundleManager` | Multi-view, multi-session evidence should precede invalidation or replacement.  The sparse-map system must not claim an HD-map/3DGS method or dataset result as direct production validation. |

## Required system-level validation

Unit tests establish parser, projection, state, and exact-computation invariants.  They do
not establish localization improvement.  A candidate release still requires:

1. frozen build, development-held-out, and certification manifests by whole session;
2. identical intrinsics, matcher/localizer configuration, and artifact hashes across A/B;
3. pair count, graph connectivity, track/angle/reprojection, wall time, RAM, and VRAM;
4. S9 strict success and pose accuracy by route, direction, appearance, and weak/stable block;
5. no stable-holdout regression and no gauge, scale, gravity, bundle, or provenance break.
