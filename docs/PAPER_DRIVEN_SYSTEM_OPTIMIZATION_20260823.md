# Paper-driven mapping-system optimization — 2026-08-23

This document records the architecture decisions used to optimize the mapping/diagnosis/update stack. It distinguishes **implemented production-path changes** from **conditional research backends** and **future work**. A paper being stronger on a benchmark is not, by itself, a reason to replace the validated GlueMap → EDM deployment path.

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
ΔJ(i | S) = wq A_i + wb B_i + wc C_i + wd D_i
           + wt T_i + wm M_i - wr R_i - wk K_i
```

where `B` is proposal bridgeability, `C` uncovered graph neighbourhood, `D` measured motion diversity, `T` camera-triplet support, `M` multi-link support and `K` frame cost.

**Important:** this output is not `BASE_CORE`. It is a queue of videos/pairs that deserve geometric verification.

### 2.2 Phase B — geometric admission

The existing fail-closed session graph remains authoritative. A cross-session edge requires verified geometry: matches/tracks, pose/scale consensus, spatial support, independent bridge groups and hold-out Sim(3) validation where applicable. Only then can greedy `U(S)` assign `BASE_CORE` or `BASE_SUPPORT`.

This resolves the previous structural problem: video-only mode could safely choose a single seed but could not responsibly recommend a multi-video base because retrieval was correctly forbidden from acting as geometry. The new proposal layer fills that gap without weakening the geometric contract.

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

### 4.1 GLUEMAP — keep as the primary reconstruction backend

Pan, Schönberger & Pollefeys, *Global Structure-from-Motion Meets Feedforward Reconstruction*, CVPR 2026.

GLUEMAP combines retrieval, optional Doppelgangers++ disambiguation, feed-forward local reconstruction, global averaging/BA and refinement. It is a good match for the system because it provides local robustness while retaining a globally optimized COLMAP-compatible map.

**Decision:** do not replace the validated GlueMap path by default. Improve what enters GlueMap and independently test what comes out.

### 4.2 Camera triplets — use for graph quality and sparsification

Manam & Govindu, *Leveraging Camera Triplets for Efficient and Accurate Structure-from-Motion*, CVPR 2024.

Triplets provide redundant local consistency: a candidate edge that is weak relative to the other two edges in a triangle is suspicious and/or low-value. This supports both graph sparsification and false-edge filtering.

**Implemented:** a session-level proposal analogue

```text
q(e,t) = n_e / max(n_1,n_2,n_3)
q(e)   = mean_t q(e,t)
```

using candidate counts only to rank geometry work. S3/S4 still owns true geometric verification.

**Stronger extension:** compute the same concept from verified pair inliers/relative poses at image level and combine it with rotation-cycle residuals before GlueMap.

### 4.3 Doppelgangers++ — keep as the repeated-structure primary gate

Xiangli et al., *Doppelgangers++: Improved Visual Disambiguation with Geometric 3D Features*, CVPR 2025.

The method targets visual aliasing where distinct surfaces look alike and generate spurious SfM matches. It uses 3D-aware MASt3R features and is explicitly compatible with SfM pipelines.

**Decision:** S4 Doppelgangers++ remains the first repeated-structure disambiguator. It should not be bypassed for scenes that show symmetry/repeated fences/buildings/roadside structures.

### 4.4 LFOE-GlobalSfM — conditional second opinion on translation edges

*Learning to Filter Outlier Edges in Global Structure-from-Motion*, CVPR 2025.

The method models the line graph of camera-pair relations and predicts inaccurate translation edges, with clustering for scalability.

**Decision:** keep the pinned backend as a research/diagnostic fallback when:

- S4 leaves a small number of high-influence ambiguous edges;
- independent Sim(3) disagrees across sessions;
- cycle errors remain high after Doppelgangers++;
- a candidate map has ghost geometry despite apparently good pairwise scores.

Do not make it a mandatory pass on every normal build until its effect is calibrated on held-out localization.

### 4.5 Detector-Free SfM — fallback when the feature graph is the bottleneck

He et al., *Detector-Free Structure from Motion*, CVPR 2024.

Detector-Free SfM starts from quantized detector-free matches, builds a coarse reconstruction and iteratively improves tracks/geometry. It is especially relevant when local keypoint repeatability is poor.

**Decision:** research fallback for texture-poor or detector-failure segments, not an always-on replacement. The current GlueMap feed-forward local geometry already addresses part of this failure regime, so replacing the whole pipeline would duplicate cost unless S3/S5 diagnostics identify frontend failure.

### 4.6 MP-SfM — fallback for low overlap / low parallax / symmetry

*MP-SfM: Monocular Priors for Robust Structure from Motion*, CVPR 2025.

The key idea is to inject monocular geometric priors such as depth/normals with uncertainty to stabilize cases where multiview constraints alone become weak or ambiguous.

**Decision:** use only as an experimental fallback on segments diagnosed as low-parallax/low-overlap/symmetric and compare against GlueMap on the same held-out localization set. A monocular prior must never silently define metric scale; the final map remains a similarity-gauge reconstruction unless external scale is supplied.

### 4.7 RoMo — dynamic-scene escalation path

*RoMo: Robust Motion Segmentation for Structure from Motion*, ICCV 2025.

RoMo combines optical flow, epipolar cues and pretrained video segmentation to reject dynamic-scene correspondences.

**Implemented subset:** lightweight optical-flow/Essential consistency enters video admission as a risk signal.

**Not implemented:** semantic/video segmentation. Trigger a full RoMo-style backend only when dynamic contamination is material; static infrastructure scenes should not pay this cost by default.

### 4.8 Light3R-SfM — speed/memory research alternative

*Light3R-SfM*, CVPR 2025, uses retrieval-guided sparse scene graphs / tree-style connectivity to reduce feed-forward reconstruction cost.

**Decision:** benchmark as an alternative when GlueMap local inference or graph size, rather than BA, dominates runtime/memory. Do not substitute it into release maps without S6/S9 parity.

### 4.9 Long-term map sparsification — preserve localization support, not raw point count

Chang et al., *Long-Term Visual Map Sparsification with Heterogeneous GNN*, CVPR 2022.

The work frames sparse localization-map design with a K-Cover constraint: camera positions should retain enough useful 3D support under a total map budget. It further learns long-term point importance on a heterogeneous SfM graph.

**Adopted principle:** pre-build/session selection and later bundle sparsification should optimize coverage/support per cost. “More frames / more tracks / more points” is not itself a release objective.

### 4.10 ExMaps — point-level temporal stability

Rotsidis et al., *ExMaps: Long-Term Localization in Dynamic Scenes using Exponential Decay*, WACV 2021.

The useful concept is a recency-weighted visibility/matching history rather than a one-shot permanent landmark label. A generic decay takes the form

```text
N(t) = 2^(-lambda * t)
```

and point stability aggregates decayed evidence across sessions.

**Current state:** `map_update/core/point_evidence_ledger.py` has point identities and session evidence but is not fully wired into the live update matcher. Do not claim production point-level ExMaps until that identity/evidence hand-off is complete.

### 4.11 Multi-session wide-baseline optimization

*Multi-Session SLAM with Differentiable Wide-Baseline Pose Optimization*, CVPR 2024.

The important transferable principle is that disconnected/weakly connected sessions need robust wide-baseline geometry and similarity alignment; temporal adjacency cannot substitute for cross-session evidence.

**Decision:** preserve independent submap reconstruction + held-out Sim(3) validation as a hard safety pattern for weak bridges.

### 4.12 Change detection / map freshness

Galappaththige et al., *Multi-View Pose-Agnostic Change Localization with Zero Labels*, CVPR 2025, shows the benefit of accumulating multiview evidence for viewpoint-robust change localization.

Du et al., *RTMap: Real-Time Recursive Mapping with Change Detection and Localization*, ICCV 2025, jointly treats prior-map localization, uncertainty and structural change over repeated traversals.

**Decision:** these support the architecture of accumulating multi-session evidence before replacing geometry. They do **not** justify copying an HD-map/3DGS representation into this sparse SfM localization system. Route 4 remains evidence-first: detect, localize, confirm across views/sessions, then replace a local tile with boundary anchors and revalidate.

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
  ↓
S4 Doppelgangers++ + graph/cycle audit
  ├─ ambiguous/high-influence → optional LFOE second opinion
  ↓
S5 GlueMap + fixed-intrinsics final BA
  ↓
S5.7 independent per-session/submap Sim(3)
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
texture/keypoint failure        → Detector-Free SfM A/B
low overlap/parallax/symmetry   → MP-SfM A/B
residual bad translation edges  → LFOE A/B
material dynamic contamination  → RoMo-style segmentation A/B
graph/inference memory bottleneck → Light3R-SfM A/B
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
A6 A5 + independent Sim3 gate
```

This isolates whether the selector saves computation without sacrificing S9 coverage, and whether false-edge gates improve geometry/localization independently of raw video count.

---

## 7. Stronger methods do exist — but “stronger” is regime-dependent

| Failure regime | Strong candidate | Why it can be stronger | Why it is not default here |
| --- | --- | --- | --- |
| weak texture / detector failure | Detector-Free SfM | detector-free dense matching and iterative multiview refinement | extra frontend/reconstruction cost; GlueMap already adds feed-forward local robustness |
| low overlap / low parallax / symmetry | MP-SfM | monocular depth/normal priors regularize weak multiview geometry | prior bias/domain shift; needs site A/B and gauge care |
| repeated structures | Doppelgangers++ | explicit visual-alias disambiguation | already integrated; still needs graph connectivity checks |
| global translation-edge outliers | LFOE-GlobalSfM | learned line-graph edge filtering | learned-domain dependency; best as second opinion initially |
| dynamic scene | RoMo | motion segmentation using flow + epipolar + learned video cues | unnecessary cost in mostly static infrastructure scenes |
| feed-forward speed/memory | Light3R-SfM | sparse retrieval-guided graph | release parity with GlueMap/EDM not yet established for this site |
| long-term sparse map | Heterogeneous-GNN K-Cover | learns future localization value under map budget | training/query-domain requirement; more complex maintenance stack |

The release criterion therefore stays empirical and deployment-oriented:

```text
better method = passes geometry gates
              + improves untouched localization
              + fits runtime/memory budget
              + does not introduce gauge/release regressions
```

---

## 8. References

- Pan, L., Schönberger, J. L., Pollefeys, M. *Global Structure-from-Motion Meets Feedforward Reconstruction*. CVPR 2026 / GLUEMAP.
- Manam, B., Govindu, V. M. *Leveraging Camera Triplets for Efficient and Accurate Structure-from-Motion*. CVPR 2024.
- Xiangli, Y. et al. *Doppelgangers++: Improved Visual Disambiguation with Geometric 3D Features*. CVPR 2025.
- He, X. et al. *Detector-Free Structure from Motion*. CVPR 2024.
- *MP-SfM: Monocular Priors for Robust Structure from Motion*. CVPR 2025.
- *Learning to Filter Outlier Edges in Global Structure-from-Motion*. CVPR 2025.
- *RoMo: Robust Motion Segmentation for Structure from Motion*. ICCV 2025.
- *Light3R-SfM*. CVPR 2025.
- Chang, M.-F. et al. *Long-Term Visual Map Sparsification with Heterogeneous GNN*. CVPR 2022.
- Rotsidis et al. *ExMaps: Long-Term Localization in Dynamic Scenes Using Exponential Decay*. WACV 2021.
- *Multi-Session SLAM with Differentiable Wide-Baseline Pose Optimization*. CVPR 2024.
- Galappaththige, C. J. et al. *Multi-View Pose-Agnostic Change Localization with Zero Labels*. CVPR 2025.
- Du, Y. et al. *RTMap: Real-Time Recursive Mapping with Change Detection and Localization*. ICCV 2025.
