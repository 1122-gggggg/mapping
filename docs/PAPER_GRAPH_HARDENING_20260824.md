# S3.5 Paper-Graph Hardening

This stage inserts a **fail-closed graph hardening pass after exact-pair geometric
verification and before Doppelgangers++ / GlueMap optimization**. It adapts useful
parts of:

- **G-MASt3R-SfM: Graph-based View Pruning and Multi-stage Optimization for Robust
  SfM**, arXiv:2606.22856.
- **Planar-SfM: Camera Pose Estimation via Homography Graph Embeddings**,
  arXiv:2606.31979v2.

It is not a reimplementation of either paper. The unit of analysis in this repo is
usually a **session/video edge**, not every image pair, and existing exact-pair
geometry remains the only merge authority.

## What was integrated

### From G-MASt3R-SfM

1. Build a scene/session graph only from geometrically authorized edges.
2. Run deterministic Louvain community detection and a weighted spring layout.
3. Measure community separation relative to the graph layout scale.
4. Quarantine a small, weak, weakly attached community instead of allowing it to
   contaminate the Base.
5. Keep a strong disconnected community as `NEW_SUBMAP_CANDIDATE`, rather than
   pretending it is connected to the Base.
6. Emit a three-level optimization schedule:
   `LOCAL -> NEIGHBOR -> GLOBAL_COMPONENT`.

The schedule is a plan only. A stage becomes evidence only after the real BA
runner writes convergence/residual receipts.

### From Planar-SfM

1. Consume one or more optional homography hypotheses per session pair.
2. Score hypotheses using available shared-inlier support, plane-normal
   consistency, and agreement with an independently estimated essential-matrix
   rotation.
3. Construct a hypothesis line graph and compute a spectral real-line embedding.
4. Find a minimum-range spanning backbone: among connected edge intervals, choose
   the interval with the smallest embedding range, then compute the
   maximum-reliability spanning tree inside it.
5. Add only low-inconsistency redundant edges that reduce articulation points.
6. Downgrade a replaceable, non-bridge embedding outlier to `AMBIGUOUS`.

Planarity is therefore **not automatically treated as a failure**. It contributes
positive evidence only when at least two independent planar-consistency terms are
present. Missing planar evidence never upgrades an edge.

## Safety invariants

- VPR/retrieval-only pairs never enter the geometric graph.
- The stage never upgrades `REJECT`, `WEAK`, or `AMBIGUOUS` to `USABLE/STRONG`.
- It never removes an original graph bridge solely because of spectral embedding.
- Strong disconnected components are retained as new-submap candidates.
- A pruned `STRONG/USABLE` row is only downgraded to `AMBIGUOUS`; the original
  status is retained in `status_before_paper_graph`.
- The hardening output does not replace S4 repeated-structure checking, S5 fixed-
  intrinsics BA, S5.7 independent Sim(3), or S9 held-out localization.

## Command

After `sfm-qa select-sessions` and exact-pair probe generation:

```bash
sfm-graph-harden \
  --edges /path/to/session_edges.csv \
  --probe-metrics /path/to/exact_pair_probe_metrics.json \
  --sessions /path/to/all_sessions.txt \
  --protected-sessions /path/to/frozen_base_sessions.txt \
  --config diagnosis/src/sfm_qa/session_select/defaults.yaml \
  --output /path/to/s3_5_paper_graph
```

Without installing the package:

```bash
PYTHONPATH=diagnosis/src python -m sfm_qa.session_select.paper_graph_cli \
  --edges /path/to/session_edges.csv \
  --output /path/to/s3_5_paper_graph
```

## Required edge fields

An edge enters the graph only when all of the following are true:

```text
status in {STRONG, USABLE}
num_verified_pairs / num_cross_session_tracks / inlier_count > 0
independent_bridge_groups > 0
evidence_scope == exact_pair
independent_artifact == true
geometry_complete == true
group_holdout_disjoint == true
```

These are the same fail-closed concepts already used by session selection. A row
that only has `num_candidate_pairs` is retrieval evidence and remains outside the
geometric graph.

## Optional planar fields

Pair-level summary fields:

```text
homography_shared_inlier_ratio
homography_inlier_ratio
homography_support
plane_normal_similarity
plane_normal_angle_deg
homography_rotation_error_deg
essential_rotation_agreement_deg
```

Multiple hypotheses can be supplied as JSON in `homography_hypotheses`:

```json
[
  {
    "hypothesis_id": "wall_0",
    "confidence": 0.91,
    "homography_shared_inlier_ratio": 0.78,
    "plane_normal": [0.02, -0.01, 0.999],
    "homography_rotation_error_deg": 1.3,
    "inlier_ids": ["q12-r55", "q18-r61"]
  }
]
```

The hardener consumes these summaries; it does not estimate homographies from raw
images. Homography estimation/decomposition must remain in an independent exact-
pair probe so fit and holdout evidence can be audited.

## Outputs

| File | Use |
|---|---|
| `paper_graph_report.json` | complete communities, embedding, backbone, pruning, and invariants |
| `session_edges_hardened.csv` | original edge table plus graph diagnostics and fail-closed downgrades |
| `retained_geometry_pairs.txt` | pair allowlist for S4/S5 candidate consumption |
| `quarantined_geometry_pairs.txt` | pair denylist with graph-level reasons |
| `quarantined_sessions.txt` | weak separated sessions excluded from the Base |
| `new_submap_candidates.txt` | strong disconnected sessions that should be reconstructed independently |
| `mso_optimization_schedule.json` | local/neighbor/component-global optimization plan |

## Pipeline placement

```text
S3 pair proposal
  -> exact-pair geometry probes
  -> S3.5 paper graph hardening (this stage)
  -> S4 Doppelgangers++ / repeated-structure audit
  -> S5 GlueMap + fixed-intrinsics BA
  -> S5.7 independent Sim(3)
  -> S9 held-out localization
```

The retained pair list is an additional filter, not a bypass. S4 may reject more
edges; S5 must still preserve intrinsics and emit BA receipts; S9 remains release
authority.
