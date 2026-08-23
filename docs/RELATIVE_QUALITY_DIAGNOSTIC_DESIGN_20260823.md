# Relative-quality mapping and diagnosis design — 2026-08-23

This note records the evidence behind the best-available selection policy used by
`sfm-qa`. It separates claims made by primary sources from engineering inferences made
for this repository.

## 1. Decision

Quality preferences are soft, cohort-relative evidence. Data-integrity and geometry
authority remain hard invariants.

```text
readable videos
  → observed-metric scores + evidence completeness
  → cohort-relative ranks
  → coverage/diversity/redundancy portfolio
  → prioritized geometry probes
  → verified session graph
  → relative marginal-utility map subset
  → untouched held-out localization
  → relative risk–coverage diagnosis
```

The system must not return an empty proposal merely because every video is below a
heuristic quality reference. It returns the best available non-empty probe set and marks
the fallback. That set still cannot enter a joint reconstruction until cross-session
geometry is verified.

The following stay hard because a relative score cannot make them true:

- an input artifact must be readable and structurally valid;
- build and held-out query sessions must not leak into each other;
- retrieval similarity is not a geometric edge;
- non-finite geometry, an unverified relative pose, or an ambiguous unique bridge cannot
  authorize a merge;
- release claims require untouched held-out localization.

## 2. Relative evidence model

For each observed metric in the current cohort, the implementation uses a tie-aware
percentile rank. Missing values remain missing.

```text
r_ij = midrank(x_ij) / (n_j - 1)
Q_i  = Σ(j observed) w_j r_ij / Σ(j observed) w_j
E_i  = Σ(j observed) w_j / Σ(all j) w_j
```

`Q_i` is relative quality and `E_i` is evidence completeness. An absolute observed score
is retained as a weak prior so an extremely poor but top-ranked member of a uniformly bad
cohort is not mislabeled as objectively good.

For a candidate video `i` and already selected set `S`, the proposal layer uses:

```text
ΔJ(i | S) = wq quality(i)
           + wb bridgeability(i,S)
           + wc new_graph_coverage(i,S)
           + wd motion_diversity(i,S)
           + wa exposure_diversity(i,S)
           + wt triplet_support(i,S)
           + wm multi_link_support(i,S)
           - wr risk(i)
           - wk frame_cost(i)
           - wn redundancy(i,S)
```

The selector stops after the marginal score collapses relative to earlier additions, not
when it crosses a universal quality number. It always reserves a validation candidate
when the available pool permits it.

After reconstruction, mapped sessions use:

```text
U(S) = coverage + quality + connectivity + redundancy + information
     + view_diversity - track_cost - risk
```

Mapped `WEAK` sessions may compete under this objective. A weak but complementary,
geometrically verified session can beat a redundant strong session. A video-only weak
row cannot silently become map geometry.

## 3. Primary-source evidence by stage

### 3.1 Video, frame, and mapping-session selection

| Primary source | Source result | Transfer used here |
| --- | --- | --- |
| [Ahmed et al., Robust Key Frame Extraction for 3D Reconstruction from Video Streams, VISAPP 2010](https://www.scitepress.org/PublishedPapers/2010/28369/) | Correspondence ratio, GRIC model choice, and point-to-epipolar-line cost help avoid rotation, planar degeneracy, and blur in video key-frame selection. | Treat image and two-view geometry as weighted evidence; do not copy its fixed cutoffs across cameras. |
| [Park & Yoon, Optimal key-frame selection for video-based SfM, 2011](https://doi.org/10.1049/el.2011.2674) | Feature lifetime, baseline, redundancy, and degeneracy jointly matter for key-frame selection. | Keep quality, motion, and redundancy as separate terms rather than one blur gate. |
| [Snavely et al., Skeletal Graphs for Efficient Structure from Motion, CVPR 2008](https://www.cs.cornell.edu/~snavely/projects/skeletalset/) | A small view skeleton can preserve coverage and bounded uncertainty, with remaining images registered later. | Build a bounded session portfolio first; keep nonselected sessions for validation, appearance support, or later registration. |
| [Shah et al., View-graph Selection Framework for SfM, ECCV 2018](https://openaccess.thecvf.com/content_ECCV_2018/html/Rajvi_Shah_View-graph_Selection_Framework_ECCV_2018_paper.html) | Task-specific image/pair costs can target accuracy, efficiency, coverage, or disambiguation through an approximate network-flow formulation. | Use a multi-term portfolio objective and preserve explicit image/edge evidence. |
| [Manam & Govindu, Leveraging Camera Triplets for Efficient and Accurate SfM, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Manam_Leveraging_Camera_Triplets_for_Efficient_and_Accurate_Structure-from-Motion_CVPR_2024_paper.html) | Edge support relative to the strongest edge in a camera triplet helps remove redundancy and false edges while maintaining reconstruction quality. | Candidate-count triplet scores prioritize probes only; verified image-level triplets remain the stronger later-stage evidence. |
| [Sarlin et al., LaMAR: Benchmarking Localization and Mapping for Augmented Reality, ECCV 2022](https://www.microsoft.com/en-us/research/uploads/prod/2022/10/sarlin2022eccv.pdf) | Long-term, multi-floor mapping/query sequence selection is formulated around cross-sequence coverage, with day/night, viewpoint, device, and time changes. | Reserve whole sessions as queries and assess condition coverage rather than randomly splitting adjacent frames. |
| [Gygli et al., Video Summarization by Learning Submodular Mixtures of Objectives, CVPR 2015](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Gygli_Video_Summarization_by_2015_CVPR_paper.html) | Submodular mixtures can jointly optimize summary objectives such as importance, representativeness, and uniformity under a length budget. | Greedy marginal selection is appropriate for the inexpensive proposal stage; it is not merge authority. |

### 3.2 View graph and intermediate reconstruction

| Primary source | Source result | Transfer used here |
| --- | --- | --- |
| [Schönberger & Frahm, Structure-from-Motion Revisited, CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/html/Schonberger_Structure-From-Motion_Revisited_CVPR_2016_paper.html) | Registration and triangulation benefit from visible-point count, uniform image-plane distribution, triangulation angle, positive depth, reprojection control, and iterative refinement. | Track distribution, not raw point count alone; rank weak regions using track, angle, coverage, and reprojection evidence. |
| [Sweeney et al., Optimizing the Viewing Graph for Structure-from-Motion, ICCV 2015](https://openaccess.thecvf.com/content_iccv_2015/html/Sweeney_Optimizing_the_Viewing_ICCV_2015_paper.html) | Triplet point-transfer consistency can identify and improve inaccurate epipolar geometry before global reconstruction. | Cycle residuals are soft edge penalties and investigation priorities; an unverified edge still cannot merge sessions. |
| [Cui et al., Tracks Selection for Robust, Efficient and Scalable Large-Scale Structure from Motion, 2017](https://doi.org/10.1016/j.patcog.2017.08.002) | Compactness, accuracy, and connectedness can select far fewer tracks while preserving reconstruction quality and scalable bundle adjustment. | Penalize redundant track cost while protecting coverage and graph connectivity. |
| [Doherty et al., Spectral Measurement Sparsification for Pose-Graph SLAM, IROS 2022](https://arxiv.org/abs/2203.13897) | Algebraic connectivity can guide edge-budget selection while preserving pose-graph estimation quality. | Report Fiedler value, bridges, articulation points, and threshold sensitivity as within-map fragility ranks. |
| [Zhang & Scaramuzza, Fisher Information Field, 2020](https://arxiv.org/abs/2008.03324) | A sum of visible-landmark Fisher information models pose observability and is useful for planning/localization quality. | Keep FIM rank, log-determinant, condition, and weak direction as relative observability evidence, not a success probability. |
| [He et al., Detector-Free Structure from Motion, CVPR 2024](https://arxiv.org/abs/2306.15669) | A tolerant coarse reconstruction followed by multiview track and geometry refinement improves difficult texture/illumination/viewpoint cases. | Low-feature videos become low-confidence probes or stronger-frontend candidates rather than automatic early deletions. |
| [Xiangli et al., Doppelgangers++: Improved Visual Disambiguation with Geometric 3D Features, CVPR 2025](https://arxiv.org/abs/2412.05826) | Visual aliasing can produce spurious matches, misplaced or distorted geometry, and incorrectly fused elements. | Relative match counts never override repeated-structure and independent-geometry checks. |

### 3.3 Localization value and map compression

| Primary source | Source result | Transfer used here |
| --- | --- | --- |
| [Sattler et al., Benchmarking 6DOF Outdoor Visual Localization in Changing Conditions, CVPR 2018](https://arxiv.org/abs/1707.09092) | Long-term benchmarks evaluate pose recall across illumination, weather, season, and viewpoint shifts using multiple translation/rotation tolerances. | Hold out entire sessions and report both aggregate and condition-stratified localization recall. |
| [Taira et al., InLoc: Indoor Visual Localization with Dense Matching and View Synthesis, CVPR 2018](https://arxiv.org/abs/1803.10368) | Retrieval, matching, PnP, and pose verification are distinct failure stages under occlusion, viewpoint, and appearance change. | Attribute failures by stage; do not collapse every failure into a map-quality label. |
| [Hartmann et al., Predicting Matchability, CVPR 2014](https://openaccess.thecvf.com/content_cvpr_2014/html/Hartmann_Predicting_Matchability_2014_CVPR_paper.html) | Detector response alone does not determine whether a descriptor will match successfully. | Prefer measured matchability and verified match ratios over feature count or sharpness alone. |
| [Meržić et al., Map Quality Evaluation for Visual Localization, ICRA 2017](https://tisl.cs.utoronto.ca/publication/201705-icra-map_quality_evaluation/icra17-map_quality_evaluation.pdf) | Orientation-aware nearby observers and repeat-observed landmarks can rank localizability; the operating crossover is cross-validated, not universal. | Map-only support remains a ranking signal calibrated against held-out localization. |
| [Ferranti et al., Can You Trust Your Pose? Confidence Estimation in Visual Localization, ICPR 2020](https://arxiv.org/abs/2010.00347) | High inlier count can accompany a wrong pose in repetitive scenes; spatial inlier coverage improves confidence ranking. | Query quality combines inliers, inlier ratio, hull/grid coverage, reprojection, positive depth, and pose consensus. |
| [Dymczyk et al., Keep It Brief, IROS 2015](https://doi.org/10.1109/IROS.2015.7353722) | Localization-map compression is a constrained coverage problem; slack keeps it feasible under limited map budgets. | Best-available selection uses penalties/slack semantics instead of returning no data when ideal coverage is impossible. |
| [Camposeco et al., Hybrid Scene Compression for Visual Localization, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Camposeco_Hybrid_Scene_Compression_for_Visual_Localization_CVPR_2019_paper.html) | Weighted K-cover and grid coverage prevent selected landmarks from concentrating only in highly textured regions. | Coverage must be spatially distributed across images/regions, not maximized only in total. |
| [Chang et al., Long-Term Visual Map Sparsification with Heterogeneous GNN, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Chang_Long-Term_Visual_Map_Sparsification_With_Heterogeneous_GNN_CVPR_2022_paper.html) | Query-fitting plus K-cover can select stable, broadly visible structure at a fixed map size; training and test queries are spatially separated. | Preserve point/session coverage per cost and validate on spatially/session-isolated queries before considering a learned selector. |
| [Rotsidis et al., ExMaps: Long-Term Localization in Dynamic Scenes Using Exponential Decay, WACV 2021](https://richardt.name/publications/exmaps/) | Exponentially decayed visibility evidence favors landmarks repeatedly useful across recent sessions in dynamic scenes. | Maintain point-level temporal evidence and treat recency/stability as ranking evidence, not a permanent binary label. |

## 4. Post-map relative localization diagnosis

Each query in the provided log receives cohort-relative ranks for:

- PnP inlier count and inlier ratio;
- reprojection p90, reversed so lower is better;
- hull and 4×4 grid coverage;
- positive-depth ratio;
- pose consensus;

The report emits every complete tie-group prefix as a risk–coverage curve. This guarantees
non-zero coverage and exposes the tradeoff without selecting a universal query threshold.
Raw/strict success labels are used only to evaluate each prefix, not to construct the
ranking. Missing metrics lower evidence completeness and mildly shrink the score rather
than becoming positive evidence. Individual strict failures remain visible for attribution. This post-hoc relative
score is still not a calibrated future-failure probability; use MapDoctor's group-aware
calibration path for that claim.

Run acceptance uses `strict_success_rate >= min_strict_success_rate` (default 95%) rather
than requiring every query to pass. When provided localization meets its target but an
advisory map-health heuristic fails, the combined status is
`READY_WITH_MAP_WARNINGS`; the warning is retained and the downstream outcome is not
overridden. Structural map integrity remains hard. The diagnostic command labels input
logs as `UNVERIFIED_PROVIDED_LOG`; S0/S9 hashes or an external manifest must prove they
are actually untouched held-out data.

## 5. Required validation protocol

1. Freeze camera intrinsics, builder, matcher, localizer, and runtime parameters.
2. Split by whole video/session/route/spatial block. Adjacent frames cannot cross folds.
3. Use build sessions for reconstruction, development-held-out sessions for tuning, and a
   final untouched certification set for release.
4. Compare at least:
   - all readable videos;
   - absolute QA only;
   - relative QA/motion;
   - relative QA plus retrieval proposal graph;
   - full portfolio plus verified triplet/cycle checks.
5. Report registered images, graph components/bridges, track and triangulation statistics,
   reprojection, RAM/VRAM, runtime, map size, and held-out localization.
6. For localization, report retrieval Recall@K, strict success rate, pose recall at several
   translation/rotation tolerances where ground truth exists, p50/p90 errors, query
   risk–coverage, and performance by illumination/viewpoint/session/spatial group.
7. Select weights on development data only. Run certification once after selection logic,
   thresholds, and risk calibration are frozen.

## 6. Evidence versus inference

Supported directly by the cited literature:

- no single blur, point-count, reprojection, track, or inlier metric determines mapping or
  localization quality;
- coverage, connectivity, diversity, uncertainty, matchability, and cost are legitimate
  joint selection objectives;
- long-term appearance and viewpoint changes require session-separated evaluation;
- map-only scores require downstream localization validation;
- repetitive structure can invalidate apparently strong pair counts.

Repository-specific engineering inferences:

- use tie-aware percentile ranks as the common normalization;
- combine absolute score and cohort rank with explicit evidence completeness;
- keep at least a best-available non-empty geometry-probe set;
- use relative marginal collapse as the default proposal stop;
- allow mapped `WEAK` sessions to compete under the joint objective;
- treat 95% as this project's default aggregate held-out target because the active S9
  contract already uses that target. It is not a universal number from the papers.

No paper score or threshold above is copied into production as a universal constant.
