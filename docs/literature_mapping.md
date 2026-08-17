# Literature mapping

This repository is a synthesis. No cited paper directly evaluates the complete combination of a privileged GLUEMAP base geometry, EDM detector-free localization, historical-view augmentation, change-aware masks, multi-session bridges and route-specific promotion gates.

## Multi-Session SLAM with Differentiable Wide-Baseline Pose Optimization

- Link: https://arxiv.org/abs/2404.15263
- Used for: disconnected-session candidate retrieval, wide-baseline relative geometry, monocular Sim(3) joining and global/pose-graph consistency.
- Adaptation: the paper's sessions are not treated symmetrically here. Current GLUEMAP poses are fixed anchors; only historical poses/submaps may move.

## Multi-View Pose-Agnostic Change Localization with Zero Labels

- Link: https://openaccess.thecvf.com/content/CVPR2025/papers/Galappaththige_Multi-View_Pose-Agnostic_Change_Localization_with_Zero_Labels_CVPR_2025_paper.pdf
- Used for: pose-aligned feature/structure comparison and multi-view change evidence.
- Adaptation: temporal direction is reversed. Historical-only content is rejected because the current map represents the production scene.

## RTMap: Real-Time Recursive Mapping with Change Detection and Localization

- Link: https://openaccess.thecvf.com/content/ICCV2025/papers/Du_RTMap_Real-Time_Recursive_Mapping_with_Change_Detection_and_Localization_ICCV_2025_paper.pdf
- Used for: matched/outdated/new association separation, uncertainty and removing change events before localization optimization.
- Adaptation: the map element representation differs; the repository applies the classification principle to EDM correspondences and historical references.

## Long-term Visual Map Sparsification with Heterogeneous GNN

- Link: https://arxiv.org/abs/2203.15182
- Used for: future-query utility, heterogeneous visibility relations, K-cover and map-size control.
- Adaptation: selection is implemented as an interpretable greedy EDM/FIM/K-cover baseline. A GNN can replace the scoring model without changing the hard invariants.

## Predictive and Adaptive Maps for Long-Term Visual Navigation

- Link: https://arxiv.org/abs/2603.12460
- Used for: privileged experience, history-aware feature management and avoiding repeated map-to-map degradation.
- Adaptation: the newest GLUEMAP is the privileged canonical geometry; historical data cannot recursively redefine it.

## ExMaps: Long-Term Localization in Dynamic Scenes Using Exponential Decay

- Link: https://openaccess.thecvf.com/content/WACV2021/html/Rotsidis_ExMaps_Long-Term_Localization_in_Dynamic_Scenes_Using_Exponential_Decay_WACV_2021_paper.html
- Used for: persistence/history scores and gradual retirement.
- Adaptation: age does not directly penalize viewpoint utility. A very old but still geometrically valid side view may be valuable. Repeated conflict is stronger evidence than a single unmatched observation.

## Map Point Selection for Visual SLAM

- Link: https://arxiv.org/abs/2306.12901
- Used for: combining back-end information with front-end feature/matching utility.
- Adaptation: EDM matchability and held-out current-query gain are measured alongside FIM, rather than selecting references by information alone.

## GLUEMAP and EDM

The repository assumes a GLUEMAP-derived COLMAP-compatible model and an existing EDM localization implementation. GlueMap's augmented/virtual constraints must not automatically become physical PnP landmarks. EDM produces 2D–2D correspondences; this project provides the controlled 2D–current-3D lifting and downstream update policy.
