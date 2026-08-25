# Paper-driven mapping-system optimization — 2026-08-23

This document records architecture decisions for the mapping/diagnosis/update stack. A paper benchmark alone never authorizes replacement: the implementation must also have an explicit usable license, a compatible COLMAP boundary, fixed-calibration controls, bounded resources, and the same S6/S9 release evidence.

## 1. Target failure modes

The system is optimized for monocular multi-video UAV mapping where the dominant failures are not simply “insufficient point count”:

1. low-parallax / hover / pure-rotation / motion-blurred data entering SfM;
2. too many redundant videos/tracks causing unnecessary memory and optimization cost;
3. forward/reverse traversals that VPR may fail to connect;
4. repeated structures generating false cross-session bridges and ghost geometry;
5. one weak session or bridge deforming a globally optimized map;
6. a reconstruction that has low reprojection error but poor held-out localizability;
7. long-term scene changes and stale landmarks;
8. update operations that silently move the old gauge and invalidate localization-side products.

The governing principle is therefore:

```text
DATA EVIDENCE
    ↓
PROPOSAL GRAPH
    ↓
VERIFIED GEOMETRIC GRAPH
    ↓
GLOBAL RECONSTRUCTION
    ↓
INDEPENDENT / POST-MAP DIAGNOSIS
    ↓
LOCALIZATION-READINESS TEST
    ↓
GAUGE-AWARE MAP UPDATE
```

No upstream score is allowed to skip a downstream geometric/release gate.

---

## 2. Implemented optimization: two-phase video admission

The later relative-quality audit and its expanded primary-source matrix are recorded in
[`RELATIVE_QUALITY_DIAGNOSTIC_DESIGN_20260823.md`](RELATIVE_QUALITY_DIAGNOSTIC_DESIGN_20260823.md).

### 2.1 Phase A — proposal only

`sfm-qa select-sessions` now computes video-level quality and motion evidence before deciding where to spend pair matching / reconstruction budget.

Signals include:

- sharpness p10 and median;
- under/over-exposure ratio;
- near-duplicate ratio;
- parallax, low-parallax, hover, pure-rotation, fast-motion and unproven interval ratios;
- Essential-matrix inlier ratio / epipolar-outlier proxy;
- optional retrieval candidate counts;
- graph-neighbourhood coverage;
- motion-profile diversity;
- frame cost.

The proposal score is intentionally heuristic:

```text
A_i = weighted(video quality, parallax, sharpness, non-duplicate,
               exposure, motion evidence, epipolar consistency)

R_i = weighted(degenerate motion, duplicates, bad exposure,
               epipolar outliers, unproven intervals)
```

A greedy marginal objective then selects a bounded candidate set:

```text
ΔJ(i | S) = wq A_i + wb B_i + wc C_i + wd D_i + wa E_i
           + wt T_i + wm M_i - wr R_i - wk K_i - wn N_i
```

where `B` is proposal bridgeability, `C` uncovered graph neighbourhood, `D` measured motion diversity, `E` measured exposure diversity, `T` camera-triplet support, `M` multi-link support, `K` frame cost and `N` redundancy.

The production default now ranks the current cohort instead of applying
`min_video_score` as an eligibility gate. Missing measurements are omitted and reported
through evidence completeness. Measured exposure diversity and a redundancy penalty are
part of the portfolio objective. If every video is weak under the legacy references, the
selector emits the best available non-empty geometry-probe set with
`relative_fallback_used=true`.

**Important:** this output is not `BASE_CORE`. It is a queue of videos/pairs that deserve geometric verification.

### 2.2 Phase B — geometric admission

The existing fail-closed session graph remains authoritative. Current admission enforces verified geometry, independent bridge groups, and pose/scale consensus. Spatial-support, reprojection, cycle, and disjoint hold-out Sim(3) checks remain separate diagnostic/release checks where evidence is available. Only a real geometric edge can let greedy `U(S)` assign both sessions to `BASE_CORE` or `BASE_SUPPORT`.

This resolves the previous structural problem: video-only mode could safely choose a single seed but could not responsibly recommend a multi-video base because retrieval was correctly forbidden from acting as geometry. The new proposal layer fills that gap without weakening the geometric contract.

Mapped `WEAK` sessions now remain candidates under the joint utility when real
reconstruction evidence exists. If no mapped `STRONG`/`USABLE` seed exists, the best
mapped weak session is an explicitly labeled fallback. Unverified VPR edges, ambiguous
unique bridges, and held-out leakage remain non-negotiable blockers.

### 2.3 Post-map relative diagnosis

Provided localization reports now include a cohort-relative query score and complete
risk–coverage curve. Aggregate acceptance uses the configured strict-success target
(default 95%) rather than requiring all queries to pass. A passing provided log can
produce `READY_WITH_MAP_WARNINGS` when map integrity passes but advisory map-only metrics
disagree. The command marks provenance unverified; S0/S9 hashes must prove held-out
isolation before release.

---

## 3. Implemented correctness fixes

### 3.1 OpenCV inlier-mask counting

OpenCV masks are not guaranteed to be Boolean/0-1. Summing a 0/255 mask can inflate one inlier to 255. Motion analysis now uses `count_nonzero()` for homography and Essential/pose masks. This directly affects pure-rotation, low-parallax and epipolar-consistency decisions.

### 3.2 Configuration/runtime drift

Image-QA thresholds had been partly hard-coded while YAML advertised another set. `evaluate_video()` now receives the configured sharpness/exposure/near-duplicate gates so reports and execution share the same thresholds.

### 3.3 Occupancy normalization

`grid_occupancy_4x4` may arrive as a fraction in `[0,1]` or an occupied-cell count in `[0,16]`. The objective now normalizes both representations before combining them with convex-hull coverage.

### 3.4 View diversity is no longer a time proxy

Capture timestamp does not encode viewpoint diversity. When motion histograms exist, the selector uses total-variation distance between measured motion profiles. Timestamp remains metadata only.

### 3.5 CI location

GitHub executes workflows only from repository-root `.github/workflows/`. A root workflow has been added for the selection/diagnosis core; the nested historical workflow is no longer the only CI definition.

---

## 4. Paper-to-stage mapping

### 4.1 GLUEMAP — retained as an explicit alternative

Pan, Schönberger & Pollefeys, *Global Structure-from-Motion Meets Feedforward Reconstruction*, CVPR 2026.

GLUEMAP combines retrieval, optional Doppelgangers++ disambiguation, feed-forward local reconstruction, global averaging/BA and refinement. It remains COLMAP-compatible and useful for controlled comparison.

**Decision:** the maintained BSD-licensed COLMAP 4.x `global_mapper` is the S5 default. GlueMap remains an explicit whole-pipeline alternative; it does not consume the existing match database and its default transitive learned dependencies include non-commercial components.

### 4.2 Camera triplets — use for graph quality and sparsification

Manam & Govindu, *Leveraging Camera Triplets for Efficient and Accurate Structure-from-Motion*, CVPR 2024.

Triplets provide redundant local consistency: a candidate edge that is weak relative to the other two edges in a triangle is suspicious and/or low-value. This supports both graph sparsification and false-edge filtering.

**Implemented:** a session-level proposal analogue

```text
q(e,t) = n_e / max(n_1,n_2,n_3)
q(e)   = mean_t q(e,t)
```

using candidate counts only to rank geometry work. S3/S4 still owns true geometric verification.

**Stronger extension:** compute the same concept from verified pair inliers/relative poses at image level and combine it with rotation-cycle residuals before GlobalMapper.

### 4.3 Doppelgangers++ — keep as the repeated-structure primary gate

Xiangli et al., *Doppelgangers++: Improved Visual Disambiguation with Geometric 3D Features*, CVPR 2025.

The method targets visual aliasing where distinct surfaces look alike and generate spurious SfM matches. It uses 3D-aware MASt3R features and is explicitly compatible with SfM pipelines.

**Decision:** S4 Doppelgangers++ remains the first repeated-structure disambiguator. It should not be bypassed for scenes that show symmetry/repeated fences/buildings/roadside structures.

### 4.4 LFOE-GlobalSfM — isolated research backend only

Damblon et al., *Learning to Filter Outlier Edges in Global SfM*, CVPR 2025.

The method models relative-translation edges as vertices in a clustered line graph and classifies them before translation averaging. Its command-compatible `glomap_filter` consumes a COLMAP database and writes a COLMAP model.

**Replacement audit:** the official repository has no top-level license grant. The released executable also bundles the now-deprecated standalone GLOMAP fork. Therefore:

- COLMAP 4.x `global_mapper` replaces LFOE in S5 and S5.7;
- all focal, principal-point and extra-parameter refinement flags are disabled;
- LFOE remains blocked in the experiment planner unless a local/upstream license file is present;
- this is a legal/maintenance cutover, not an accuracy claim—COLMAP lacks LFOE's learned translation-edge classifier, so S4/S6/S9 remain mandatory.

### 4.5 G-MASt3R-SfM — transfer graph pruning, not its optimizer stack

Watanabe et al., *G-MASt3R-SfM: Graph-based View Pruning and Multi-stage Optimization for Robust SfM*, 2026.

GVP geometrically verifies MASt3R edges, detects outlier view groups, and removes them before reconstruction. MSO then expands optimization from community-local to global.

**Implemented transfer:** dense-match verification removes every view component outside the deterministic largest verified component before database aggregation. The retained-component ratio is a hard gate. MSO was not copied: COLMAP GlobalMapper already performs global positioning plus bundle adjustment.

### 4.6 Planar-SfM — distinguish a valid plane from pure rotation

Pragier et al., *Planar-SfM: Camera Pose Estimation via Homography Graph Embeddings*, 2026.

Planar-SfM scores homography hypotheses by image support and agreement between homography-derived and essential-matrix-derived rotations, then selects a mutually consistent graph backbone.

**Implemented transfer:** dense pair verification decomposes the normalized homography, compares every homography rotation candidate with the essential rotation, and only rescues a homography-dominant cross-video/cross-direction edge when the rotation geodesic is within the configured bound. This replaces the old rule that treated every such cross-direction edge as rotation-like. Full spectral pose recovery is not duplicated because the downstream global mapper already owns pose estimation.

### 4.7 Detector-Free SfM — fallback when the feature graph is the bottleneck

He et al., *Detector-Free Structure from Motion*, CVPR 2024.

Detector-Free SfM starts from quantized detector-free matches, builds a coarse reconstruction and iteratively improves tracks/geometry. It is especially relevant when local keypoint repeatability is poor.

**Decision:** sealed A/B candidate for texture-poor or detector-failure segments, not a default. Apache-2.0 code and public weights exist, and fixed-pose/fixed-intrinsics triangulation writes COLMAP models, but the released refinement depends on a custom COLMAP/CUDA stack and has no target-site S9 evidence.

### 4.8 MP-SfM — fallback for low overlap / low parallax / symmetry

*MP-SfM: Monocular Surface Priors for Robust Structure-from-Motion*, CVPR 2025.

The key idea is to inject monocular geometric priors such as depth/normals with uncertainty to stabilize cases where multiview constraints alone become weak or ambiguous.

**Decision:** use only as an experimental fallback on segments diagnosed as low-parallax/low-overlap/symmetric and compare against GlueMap on the same held-out localization set. A monocular prior must never silently define metric scale; the final map remains a similarity-gauge reconstruction unless external scale is supplied.

### 4.9 RoMo — dynamic-scene escalation path

*RoMo: Robust Motion Segmentation Improves Structure from Motion*, ICCV 2025.

RoMo combines optical flow, epipolar cues and pretrained video segmentation to reject dynamic-scene correspondences.

**Implemented subset:** lightweight optical-flow/Essential consistency enters video admission as a risk signal.

**Not implemented:** semantic/video segmentation. Trigger a full RoMo-style backend only when dynamic contamination is material; static infrastructure scenes should not pay this cost by default.

### 4.10 Light3R-SfM — speed/memory research alternative

*Light3R-SfM: Towards Feed-forward Structure-from-Motion*, CVPR 2025, uses retrieval-guided sparse scene graphs / tree-style connectivity to reduce feed-forward reconstruction cost.

**Decision:** benchmark as an alternative when GlueMap local inference or graph size, rather than BA, dominates runtime/memory. Do not substitute it into release maps without S6/S9 parity.

### 4.11 Long-term map sparsification — preserve localization support, not raw point count

Chang et al., *Long-Term Visual Map Sparsification with Heterogeneous GNN*, CVPR 2022.

The work frames sparse localization-map design with a K-Cover constraint: camera positions should retain enough useful 3D support under a total map budget. It further learns long-term point importance on a heterogeneous SfM graph.

**Adopted principle:** pre-build/session selection and later bundle sparsification should optimize coverage/support per cost. “More frames / more tracks / more points” is not itself a release objective.

### 4.12 ExMaps — point-level temporal stability

Rotsidis et al., *ExMaps: Long-Term Localization in Dynamic Scenes using Exponential Decay*, WACV 2021.

The useful concept is a recency-weighted visibility/matching history rather than a one-shot permanent landmark label. A generic decay takes the form

```text
N(t) = 2^(-lambda * t)
```

and point stability aggregates decayed evidence across sessions.

**Current state:** `map_update/core/point_evidence_ledger.py` has point identities and session evidence but is not fully wired into the live update matcher. Do not claim production point-level ExMaps until that identity/evidence hand-off is complete.

### 4.13 Multi-session wide-baseline optimization

*Multi-Session SLAM with Differentiable Wide-Baseline Pose Optimization*, CVPR 2024.

The important transferable principle is that disconnected/weakly connected sessions need robust wide-baseline geometry and similarity alignment; temporal adjacency cannot substitute for cross-session evidence.

**Future/release-gate decision:** preserve independent submap reconstruction and add disjoint hold-out Sim(3) validation as the safety pattern for weak bridges. The current session admission code does not yet implement the disjoint hold-out fit/validation split.

### 4.14 Change detection / map freshness

Galappaththige et al., *Multi-View Pose-Agnostic Change Localization with Zero Labels*, CVPR 2025, shows the benefit of accumulating multiview evidence for viewpoint-robust change localization.

Du et al., *RTMap: Real-Time Recursive Mapping with Change Detection and Localization*, ICCV 2025, jointly treats prior-map localization, uncertainty and structural change over repeated traversals.

**Future-work decision:** these support accumulating multi-session evidence before replacing geometry. They do **not** justify copying an HD-map/3DGS representation into this sparse SfM localization system. Current code emits review evidence and `needs_tile_replace`; tile rebuilding, boundary-anchor optimization and revalidation are not implemented.

### 4.15 Global-Aware Edge Prioritization — replace kNN pair init, not S3/S4

Wei, Tolias, Matas & Barath, *Global-Aware Edge Prioritization for Pose Graph Initialization*, CVPR 2026.

The paper's claim that matters here is not the GNN itself. On IMC23/MegaDepth, **multi-MST selection already beats per-image kNN** for every retrieval backbone, including MegaLoc, and the gap is largest at k=1–2. Connectivity-aware hop-distance modulation then adds long-range chords that kNN never proposes.

**Replace:** GlueMap/COLMAP `num_neighbors` kNN as the *candidate-edge* initializer, using `pipeline/pose_graph_init.py`.

**Do not replace:**

- S3 VPR-blind forced reverse-direction bridges. MegaLoc similarity on this site is 0.10–0.17 reverse vs 0.31–0.59 same-direction; a MegaDepth-trained GNN will not invent that geometry.
- S4 Doppelgangers++. The paper reports DG++ adds nothing *on VisymScenes facade duplicates after their ranks*. UAV forward/reverse aliasing is a different failure mode.

**Implemented:** Kruskal multi-MST + hop-distance modulation + required-edge union. GNN rank CSV is an optional drop-in `score` column, not a production default.

### 4.16 GGPT — dense visualization sidecar, never the localization map

Chen, Wang, Zhang, Prokudin & Tang, *GGPT: Geometry-Grounded Point Transformer*, CVPR 2026.

GGPT refines a dense feed-forward point map `Xd` with sparse SfM points `Xs` in 3D. The accompanying SfM (RoMa/UFM, sparse BA, DLT) is a 4–16 view recipe. That is not S5: this system maps 10²–10³ UAV frames with locked PINHOLE/SIMPLE_RADIAL intrinsics and a global mapper.

Fuhe already ran a GGPT/Pi3 sidecar preflight and failed the sparse-overlap gate. The transformer cannot hallucinate missing co-visibility.

**Do not replace:** S5 COLMAP GlobalMapper, S8 EDM cell-anchor geometry, or S9.

**Supplement:** after poses and intrinsics are locked, tile 8-view subsets that already share ≥50 sparse points and emit a visualization/QA cloud. `pipeline/ggpt_sidecar.py` is the admission contract. Checkpoints stay external.

### 4.17 RIC-Loc — selective posed-reference scores, not the flight localizer

Kang, Kim, Lee & Kim, *Reference-Induced Consensus for Selective Posed-Reference Visual Localization*, arXiv:2607.04722.

RIC-Loc lifts one map-frame SE(3) hypothesis per posed reference from a frozen VGGT pass, then fuses with robust SO(3)×R³ consensus. The unique product is ground-truth-free failure ranking: `σ_disp` (hypothesis spread) and `σ_cons` (track-conditioned covariance), fused as

```text
σ_joint = max(σ_cons / median_heldout(σ_cons), σ_disp / median_heldout(σ_disp))
```

on the covariance-eligible set. Table 1 shows the MegaLoc top-1−top-2 gap is near-random (AUROC 0.52–0.55) for pose failure. Cambridge outdoor translation still trails structure-based hloc (40 cm vs 16 cm).

**Do not replace:** EDM 2D–3D cell-anchor localization. Outdoor metric translation and the 1.9–3.4 s VGGT pass are the wrong operating point for the flight stack.

**Supplement:** S9 / `sfm-diagnosis consensus` already had RIC-Loc-style Student-t fusion. This change adds held-out `σ_joint` (`--held-out-hypotheses`) and refuses to treat retrieval-score gap as a reject signal. Covariance-ineligible queries have no `σ_joint` and must abstain.

---

## 5. Recommended production path after this optimization

```text
raw videos
  ↓
Stage 0A: video QA + motion/epipolar scan
  ↓
optional VPR proposal graph
  ↓
budgeted prebuild candidates + reserved validation
  ↓
forced/priority geometry-verification queue
  ↓
S0 corpus lock / content-hash held-out isolation
  ↓
S1/S2 motion-aware extraction
  ↓
S2b intrinsics bake-off
  ↓
S3 verified cross-video pairs
     (forced reverse-direction ∪ multi-MST retrieval chords)
  ↓
S4 Doppelgangers++ + graph/cycle audit
  ↓
verified-edge H/E planar consistency + largest-component view pruning
  ↓
S5 maintained COLMAP GlobalMapper + fixed-intrinsics final BA
  ↓
S5.7 independent per-session COLMAP GlobalMapper + Sim(3) release gate
  ↓
S6 ghost / duplicate geometry audit
  ↓
S7 MegaLoc bundle
  ↓
S8 EDM cell-anchor 2D→3D bundle
  ↓
S9 untouched held-out localization
  ↓
MapDoctor weak-region / failure-risk diagnosis
  ↓
update routing with gauge / scale / gravity release gates
```

Conditional research escalation:

```text
texture/keypoint failure        → Detector-Free SfM or PixSfM sealed A/B
pairwise rescue failure         → RoMa v2 sealed A/B
low overlap/parallax/symmetry   → MP-SfM A/B
translation-edge outliers       → licensed LFOE build only, never silent fallback
material dynamic contamination  → RoMo-style segmentation A/B
dense visual QA after locked poses  → GGPT sidecar (overlap-gated tiles only)
pair-graph sparsity / long-range chords → multi-MST edge prioritization
held-out pose rejection             → RIC-Loc σ_joint (not retrieval-score gap)
```

Every A/B comparison must use the same build corpus and untouched S9 sessions. The winner is the method with better localization robustness under equivalent compute/memory constraints, not the method with the lowest training/reprojection loss alone.

---

## 6. What should be measured in ablations

For the new video selector, do not validate only by reconstruction registration rate. At minimum compare:

1. number of input videos / extracted frames / pair candidates / verified pairs;
2. peak RAM/VRAM and wall-clock time;
3. largest connected component and number of critical bridges;
4. rotation-cycle / Sim(3) hold-out residuals;
5. registered image ratio and 3D-point count;
6. reprojection RMSE/p90;
7. graph fragility (articulation points, bridge edges, normalized-Laplacian `lambda2`);
8. S9 held-out localization success rate;
9. PnP inliers, inlier ratio, reprojection p90, hull/grid support and pose consensus;
10. number/length of contiguous localization-failure regions;
11. calibration error of failure-risk predictions;
12. localization accuracy/runtime after EDM bundle sparsification.

Required ablations:

```text
A0 all usable videos
A1 video QA only
A2 QA + motion/epipolar
A3 A2 + retrieval proposal graph
A4 A3 + triplet/budgeted proposal
A5 A4 + S4 Doppelgangers++
A6 A5 + independent disjoint-hold-out Sim3 gate (future)
```

This isolates whether the selector saves computation without sacrificing S9 coverage, and whether false-edge gates improve geometry/localization independently of raw video count.

---

## 7. Stronger methods do exist — but “stronger” is regime-dependent

| Failure regime | Strong candidate | Why it can be stronger | Why it is not default here |
| --- | --- | --- | --- |
| weak texture / detector failure | Detector-Free SfM | Apache-2.0, public weights, fixed-pose/intrinsics COLMAP output | custom COLMAP/CUDA stack; experiment until S9 parity |
| noisy keypoints / geometry | PixSfM | Apache-2.0 featuremetric refinement of existing COLMAP models | COLMAP 3.8/pyceres port and multi-GB cache |
| pairwise rescue | RoMa v2 | MIT, public code/weights, covariance-aware dense matches | pairwise output does not replace MV-RoMa multi-view tracks |
| repeated structures | Doppelgangers++ | explicit visual-alias disambiguation | CC BY-NC-SA; retain only where deployment terms allow |
| global translation-edge outliers | LFOE-GlobalSfM | learned clustered line-graph edge filtering | no upstream license grant; experiment blocked by default |
| planar-dominant pairs | Planar-SfM consistency transfer | H/E rotation agreement separates supported planes from rotation-only edges | no official code for full spectral solver |
| feed-forward speed/memory | InstantSfM / Light3R-SfM | GPU-native or sparse feed-forward reconstruction | InstantSfM is CC BY-NC; Light3R has no released compatible module |
| long-term sparse map | Heterogeneous-GNN K-Cover | learns future localization value under map budget | training/query-domain requirement; more complex maintenance stack |

### 7.1 2026-08-24 online replacement audit

| Candidate | Boundary | License/release status | Decision |
|---|---|---|---|
| COLMAP 4.0.4 `global_mapper` / GLOMAP ECCV 2024 | Existing COLMAP DB → COLMAP sparse model | New BSD, installed CUDA binary, maintained upstream | **REPLACE_NOW** for S5/S5.7 |
| Detector-Free SfM (CVPR 2024) | Fixed-pose/intrinsics dense triangulation → COLMAP model | Apache-2.0 + weights; custom COLMAP fork | **EXPERIMENT_ONLY** |
| PixSfM (ICCV 2021) | Existing COLMAP model → featuremetric refined model | Apache-2.0 + weights; existing isolated runner | **EXPERIMENT_ONLY** |
| RoMa v2 (2025) | Pairwise dense rescue matches | MIT + weights; no multi-view-track contract | **EXPERIMENT_ONLY** |
| MGSfM (ICCV 2025) | COLMAP DB → model | BSD-3; assumes synchronized rigid multi-camera sequences | **EXPERIMENT_ONLY** |
| LFOE / Dense-SfM | Mapper / dense refinement | No top-level license; Dense-SfM also omits the paper's GS extension | **REJECT as default** |
| InstantSfM | Images/depth → model | CC BY-NC 4.0; active development | **REJECT as production replacement** |
| Planar-SfM / G-MASt3R-SfM / DATAP-SfM | Pair graph / full SfM | No usable official module or code not released | **REJECT until release** |

The release criterion therefore stays empirical and deployment-oriented:

```text
better method = passes geometry gates
              + improves untouched localization
              + fits runtime/memory budget
              + does not introduce gauge/release regressions
```

---

## 8. References

- Pan, L., Schönberger, J. L., Pollefeys, M. [*Global Structure-from-Motion Meets Feedforward Reconstruction*](https://arxiv.org/abs/2605.26103). CVPR 2026 / GLUEMAP.
- Pan, L. et al. [*Global Structure-from-Motion Revisited*](https://arxiv.org/abs/2407.20219). ECCV 2024 / maintained in COLMAP 4.x.
- Manam, B., Govindu, V. M. [*Leveraging Camera Triplets for Efficient and Accurate Structure-from-Motion*](https://openaccess.thecvf.com/content/CVPR2024/html/Manam_Leveraging_Camera_Triplets_for_Efficient_and_Accurate_Structure-from-Motion_CVPR_2024_paper.html). CVPR 2024.
- Xiangli, Y. et al. [*Doppelgangers++: Improved Visual Disambiguation with Geometric 3D Features*](https://openaccess.thecvf.com/content/CVPR2025/html/Xiangli_Doppelgangers_Improved_Visual_Disambiguation_with_Geometric_3D_Features_CVPR_2025_paper.html). CVPR 2025.
- He, X. et al. [*Detector-Free Structure from Motion*](https://arxiv.org/abs/2306.15669). CVPR 2024.
- Pataki et al. [*MP-SfM: Monocular Surface Priors for Robust Structure-from-Motion*](https://openaccess.thecvf.com/content/CVPR2025/html/Pataki_MP-SfM_Monocular_Surface_Priors_for_Robust_Structure-from-Motion_CVPR_2025_paper.html). CVPR 2025.
- Damblon et al. [*Learning to Filter Outlier Edges in Global SfM*](https://openaccess.thecvf.com/content/CVPR2025/html/Damblon_Learning_to_Filter_Outlier_Edges_in_Global_SfM_CVPR_2025_paper.html). CVPR 2025.
- Lee, J. et al. [*Dense-SfM: Structure from Motion with Dense Consistent Matching*](https://openaccess.thecvf.com/content/CVPR2025/html/Lee_Dense-SfM_Structure_from_Motion_with_Dense_Consistent_Matching_CVPR_2025_paper.html). CVPR 2025.
- Zhong, J. et al. [*InstantSfM: Towards GPU-Native SfM for the Deep Learning Era*](https://arxiv.org/abs/2510.13310). 2025.
- Edstedt, J. et al. [*RoMa v2: Harder Better Faster Denser Feature Matching*](https://arxiv.org/abs/2511.15706). 2025.
- Watanabe, T. et al. [*G-MASt3R-SfM: Graph-based View Pruning and Multi-stage Optimization for Robust SfM*](https://arxiv.org/abs/2606.22856). 2026.
- Pragier, G. et al. [*Planar-SfM: Camera Pose Estimation via Homography Graph Embeddings*](https://arxiv.org/abs/2606.31979). 2026.
- Goli et al. [*RoMo: Robust Motion Segmentation Improves Structure from Motion*](https://openaccess.thecvf.com/content/ICCV2025/html/Goli_RoMo_Robust_Motion_Segmentation_Improves_Structure_from_Motion_ICCV_2025_paper.html). ICCV 2025.
- [*Light3R-SfM: Towards Feed-forward Structure-from-Motion*](https://cvpr.thecvf.com/virtual/2025/poster/32660). CVPR 2025.
- Chang, M.-F. et al. [*Long-Term Visual Map Sparsification with Heterogeneous GNN*](https://openaccess.thecvf.com/content/CVPR2022/html/Chang_Long-Term_Visual_Map_Sparsification_With_Heterogeneous_GNN_CVPR_2022_paper.html). CVPR 2022.
- Rotsidis et al. [*ExMaps: Long-Term Localization in Dynamic Scenes Using Exponential Decay*](https://richardt.name/publications/exmaps/). WACV 2021.
- Lipson, L., Deng, J. [*Multi-Session SLAM with Differentiable Wide-Baseline Pose Optimization*](https://openaccess.thecvf.com/content/CVPR2024/html/Lipson_Multi-Session_SLAM_with_Differentiable_Wide-Baseline_Pose_Optimization_CVPR_2024_paper.html). CVPR 2024.
- Galappaththige, C. J. et al. [*Multi-View Pose-Agnostic Change Localization with Zero Labels*](https://openaccess.thecvf.com/content/CVPR2025/html/Galappaththige_Multi-View_Pose-Agnostic_Change_Localization_with_Zero_Labels_CVPR_2025_paper.html). CVPR 2025.
- Du, Y. et al. [*RTMap: Real-Time Recursive Mapping with Change Detection and Localization*](https://openaccess.thecvf.com/content/ICCV2025/html/Du_RTMap_Real-Time_Recursive_Mapping_with_Change_Detection_and_Localization_ICCV_2025_paper.html). ICCV 2025.
- Wei, T. et al. [*Global-Aware Edge Prioritization for Pose Graph Initialization*](https://openaccess.thecvf.com/content/CVPR2026/html/Wei_Global-Aware_Edge_Prioritization_for_Pose_Graph_Initialization_CVPR_2026_paper.html). CVPR 2026.
- Chen, Y. et al. [*GGPT: Geometry-Grounded Point Transformer*](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_GGPT_Geometry-Grounded_Point_Transformer_CVPR_2026_paper.html). CVPR 2026.
- Kang, W. et al. [*Reference-Induced Consensus for Selective Posed-Reference Visual Localization*](https://arxiv.org/abs/2607.04722). arXiv:2607.04722, 2026.
