# Stage 0: multi-session selection

Objective: decide which videos are worth geometric verification, which verified sessions should enter the first SfM, which should wait as frozen-base references/update candidates, and which must stay out.

`sfm-qa select-sessions` is intentionally split into **two phases**:

1. **Stage 0A — pre-build proposal.** Video QA, motion/epipolar evidence and optional retrieval candidates rank where expensive cross-session geometry should be spent. This stage may propose multiple videos even before a map exists.
2. **Stage 0B — verified admission.** Only geometrically verified session edges can create `BASE_CORE` / `BASE_SUPPORT`. VPR alone never authorizes a merge.

The command is advisory: it writes reports and exits 0. It does not run SfM or mutate a map. The site-specific S0 corpus lock remains the authoritative build/test split and must hash-prove held-out isolation before release.

## Stage 0A: which videos should be tested for the base?

For each video, the proposal layer uses measured signals rather than timestamp:

- Laplacian sharpness p10 / median
- under/over-exposure ratio
- near-duplicate ratio
- parallax / low-parallax / hover / pure-rotation / fast-motion ratios
- essential-matrix inlier ratio and a lightweight epipolar-outlier proxy
- fraction of motion intervals that remain unproven
- optional retrieval candidate connectivity
- video/frame cost

Missing/ambiguous measurements do not become positive evidence. The numerical gates in `defaults.yaml` are engineering heuristics and must be calibrated on held-out sessions before being interpreted probabilistically.

By default, those heuristics are **not eligibility requirements**. Each observed
metric is converted to a tie-aware percentile inside the current video cohort:

```text
r_i = midrank(x_i) / (n_observed - 1)
Q_i = Σ(observed w_j r_ij) / Σ(observed w_j)
```

The absolute measured score remains a weak prior, and the report separately emits
`evidence_completeness`; a missing metric is omitted, never counted as healthy. `WEAK`,
`REJECT`, and `INCONSISTENT` contribute increasing risk penalties but do not empty the
proposal pool. Only an unrecoverable input artifact (for example a confirmed unreadable
zero-frame video) is removed before the geometry-probe queue.

The proposal objective is a budgeted marginal score:

```text
ΔJ(i | S) = wq video_quality
           + wb bridgeability
           + wc graph_neighbourhood_coverage
           + wd motion_diversity
           + wa measured_exposure_diversity
           + wt triplet_support
           + wm multi_link_support
           - wr risk
           - wk frame_cost
           - wn redundancy
```

This is not a learned score. Its purpose is to spend S3/S4 geometry on the most useful sessions first, not to declare them correct.

### Camera-triplet proposal score

For a complete session triangle `t={a,b,c}`, candidate-count support uses the session-level analogue of camera-triplet scoring:

```text
q(e,t) = n_e / max(n_ab, n_ac, n_bc)
q(e)   = mean_t q(e,t)
```

A weak edge inside otherwise strong triangles gets lower priority. At proposal time `n_e` may be retrieval candidate count, therefore `q(e)` is still **not geometry**. The same idea becomes stronger when computed from verified matches/tracks later.

### Budgeted coverage

The selector prefers a video that covers previously uncovered graph neighbourhoods to another highly redundant video. This follows the same design principle as K-Cover visual-map sparsification: preserve enough support/coverage under a finite map or matching budget rather than maximizing raw video count.

If no retrieval graph exists, the selector does not invent all-pairs edges. It proposes only a small geometry-probe subset and emits forced verification pairs. Selection stops at a **relative marginal-gain collapse**, not at a fixed score. If every video is weak under the old references, it still returns the least-bad non-empty probe set and labels `relative_fallback_used=true`; this never grants merge authority. At least one session is reserved as a proposal-stage validation candidate when the pool is large enough; the later site S0 corpus lock is still the final authority on hold-out leakage.

Outputs:

```text
prebuild_plan.json
prebuild_base_candidates.txt
prebuild_validation_candidates.txt
prebuild_rejected_sessions.txt
prebuild_deferred_sessions.txt
prebuild_verification_pairs.csv
```

`prebuild_verification_pairs.csv` is the important hand-off to S3: every row has `requires_geometric_verification=true`.

## Stage 0B objective `U(S)`

After geometric evidence exists, the actual base selector maximizes

```text
U(S) = α coverage + β quality + γ connectivity + δ redundancy
     + ε information + ζ view_diversity − η track_cost − θ risk
```

The implementation normalizes `grid_occupancy_4x4` whether upstream supplied a `[0,1]` fraction or an integer `0..16` occupied-cell count. View diversity uses measured motion-profile diversity when available; **capture timestamp is never used as a view-diversity proxy or ranking key**.

Greedy procedure:

1. Score each session internally. Missing geometry never invents `STRONG`.
2. Seed `S` with the highest internal-quality `STRONG`/`USABLE` session. If every reconstructed session is `WEAK`, use the best mapped `WEAK` session as an explicitly labeled relative fallback.
3. Rank mapped `WEAK` sessions alongside stronger sessions under `U(S)`; risk penalties can lower them, but a complementary verified session is not rejected for missing one heuristic target.
4. Add only an admissible geometric neighbor with largest positive measured `ΔU`, subject to track/observation budgets.
5. Split `S` into `BASE_CORE` (load-bearing) and `BASE_SUPPORT` (helpful but not load-bearing).
6. Classify leftovers into update/reference/submap/quarantine roles.

Prefer **one fewer video** over a cheap merge that adds tracks without independent geometry.

## Fail-closed bridges

A session-to-session link is a geometric **edge** only after verification:

- verified pairs / shared tracks above heuristic floors
- rotation, translation-direction and scale consensus
- cross-session reprojection consistency
- at least one independent bridge group for a usable edge; the default `STRONG` label
  requires at least two temporally/spatially separated groups
- Sim(3) fit on a support set and validation on a disjoint hold-out
- cycle/graph checks where redundant paths exist

Fail-closed rules:

| Condition | Action |
| --- | --- |
| Uncertain metric or missing evidence | `QUARANTINE` |
| No reliable geometric edge to the base | `NEW_SUBMAP` |
| Only `REJECT` / `AMBIGUOUS` links | do not merge |
| Unique critical bridge or one unverified critical bridge | do not force-merge |
| Retrieval high but geometry absent | proposal only; queue S3 verification |

A critical bridge is the unique usable connector of two groups of size at least 2. Never promote a session into the base on that single hinge.

## VPR is not an edge

Visual place recognition / retrieval only proposes candidate pairs. A high similarity or many retrieved pairs is **not** covisibility, not a verified relative pose, and not merge authority. This distinction is especially important for forward/reverse flights, repeated structures and appearance changes.

## Track budgets

Optional resource budgets (`max_base_sessions`, `max_total_tracks`, `max_total_observations`, `max_tracks_per_session`) stop the greedy walk before reconstruction becomes track-heavy. They are compute/memory constraints, not quality pass criteria. Diminishing relative coverage/information gain with rising track cost is a reason to stop adding videos.

## Role vocabulary

| Role | Meaning |
| --- | --- |
| `BASE_CORE` | Load-bearing sessions for the first SfM. |
| `BASE_SUPPORT` | Helpful verified sessions; may enter initial SfM when needed. |
| `APPEARANCE_REF` | Same geometry, useful appearance. Localize vs frozen base; do not rebuild. |
| `GEOMETRY_REINFORCEMENT` | Extra geometry for a known weak region, only with independent bridges. |
| `UPDATE_CANDIDATE` | Scene change vs frozen base. Not “newer timestamp”. |
| `NEW_SUBMAP` | Internally usable but no reliable edge to base. Build separately. |
| `QUARANTINE` | Uncertain/high-influence/single-hinge evidence. |
| `REJECT` | Internally inconsistent or unusable. |
| `VALIDATION_ONLY` | Hold-out. Never fit the base it is supposed to test. |

Only `BASE_CORE` plus needed `BASE_SUPPORT` enter the initial SfM.

## CLI

```bash
sfm-qa select-sessions \
  --videos DIR \
  --output DIR \
  [--vpr-candidates session_pairs.json] \
  [--maps DIR] \
  [--config PATH]
```

| Flag | Required | Meaning |
| --- | --- | --- |
| `--videos` | yes | Input videos; normally one video per session. |
| `--output` | yes | Stage 0A/0B reports. |
| `--vpr-candidates` | no | Retrieval candidate JSON. Proposal ranking only; never geometric authority. |
| `--maps` | no | Existing reconstructions/verified tracks for Stage 0B. |
| `--config` | no | YAML overlay merged onto heuristic defaults. |

Python entry:

```text
sfm_qa.session_select.run.select_sessions(
    video_dir,
    output_dir,
    config=None,
    maps_dir=None,
    vpr_candidates=None,
)
```

Video decoding uses the optional extra:

```bash
pip install -e '.[video]'
```

## Literature design notes

The selector deliberately borrows **principles**, not paper scores, from several lines of
work. The expanded primary-source matrix, evidence/inference split, and validation protocol are
in [`../../docs/RELATIVE_QUALITY_DIAGNOSTIC_DESIGN_20260823.md`](../../docs/RELATIVE_QUALITY_DIAGNOSTIC_DESIGN_20260823.md).

- Manam & Govindu, *Leveraging Camera Triplets for Efficient and Accurate Structure-from-Motion*, CVPR 2024: triangle support can simultaneously expose weak graph edges and reduce graph density.
- Shah et al., *View-graph Selection Framework for SfM*, ECCV 2018: task-specific image/edge costs can target accuracy, efficiency, coverage, or disambiguation instead of chaining fixed thresholds.
- Snavely et al., *Skeletal Graphs for Efficient Structure from Motion*, CVPR 2008: a small uncertainty-preserving view skeleton can be reconstructed first, then remaining images registered later.
- Sarlin et al., *LaMAR*, ECCV 2022: mapping/query sequence selection must account for cross-session coverage in multi-floor, long-term, appearance-changing environments.
- Dymczyk et al., *Keep It Brief*, IROS 2015: K-cover with slack preserves localization support under a finite map budget and remains feasible when perfect coverage is impossible.
- Chang et al., *Long-Term Visual Map Sparsification with Heterogeneous GNN*, CVPR 2022: K-Cover motivates preserving localization support under a finite map budget instead of retaining every observation.
- He et al., *Detector-Free Structure from Motion*, CVPR 2024: difficult texture/overlap should trigger a stronger geometric frontend rather than be hidden by a weak feature graph.
- Pan et al., *Global Structure-from-Motion Meets Feedforward Reconstruction (GLUEMAP)*, CVPR 2026: global consistency and scalable optimization still require a reliable image/session graph; feed-forward local robustness does not eliminate graph-quality control.
- Xiangli et al., *Doppelgangers++*, CVPR 2025: repeated visual structure is an explicit false-edge failure mode, therefore graph admission remains fail-closed until S4 disambiguation.

These papers do **not** validate the numerical thresholds in `defaults.yaml`. Those thresholds remain site-calibration parameters and should be ablated against held-out localization success/failure.
