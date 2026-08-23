# Implementation status

The repository started with only an empty README. The current implementation is therefore a clean reference architecture rather than a patch over an existing application.

## Implemented and executable

- COLMAP/GLUEMAP text and binary model readers.
- Geometry provenance sidecars and exclusion of `VIRTUAL_BA_ONLY` landmarks.
- Immutable base-map SHA-256 snapshots and verification.
- Historical image manifests, quality metrics and conservative keyframe filtering.
- Retrieval and matcher adapter protocols, including precomputed and external-command adapters.
- EDM 2D–2D to current-map 2D–3D lifting with stable-mask checks, unique landmark aggregation and multi-reference support.
- Per-reference PnP, SE(3) mode clustering, pooled dominant-mode refinement, normalized FIM and leave-one-reference-out diagnostics.
- Pose-aligned photometric/SSIM change-mask baseline, precomputed mask loading and multi-view mask fusion.
- Bridge graph construction, current-landmark ID propagation, disjoint-path checks, Sim(3) estimation and anchored pose-graph optimization.
- Route-view cells, front-end-aware utility scoring, K-cover reference selection and redundancy pruning.
- Conflict-driven historical reference stability state machine.
- E0–E5 protocol definitions, A1–A11 ablation matrix and non-regression gates.
- Current-first/historical-on-demand online controller.
- Versioned candidate bundle staging, promotion and rollback.
- Synthetic end-to-end protocol and automated tests.

## Requires project data or external model integration

- Actual EDM model inference. Connect it through `MatcherAdapter` or provide precomputed `.npz` matches.
- Actual retrieval model inference. Connect it through `RetrieverAdapter` or provide retrieval JSON.
- DINOv2/MV-3DCD feature extraction. The change module accepts an injected dense-feature extractor or precomputed masks.
- Real bridge-submap reconstruction. Existing SfM/GLUEMAP/COLMAP commands can be called through the external adapter and their output evaluated by this repository.
- Real E0–E5 results. They require the user's base map, historical sessions and independent current validation queries.

The software never reports synthetic results as real localization improvement. Missing data or adapters are emitted as explicit blockers in the protocol report.
