# Predictive and adaptive feature-map memory

This module implements the map-management part of **Predictive and Adaptive Maps for Long-Term Visual Navigation in Changing Environments** (`arXiv:2603.12460`) and adapts it to this repository's hard invariant:

> Current-map geometry is immutable. Long-term learning changes only a sidecar active set, feature scores and temporal models. Historical-only or otherwise unverified geometry is quarantined and cannot become a production pose landmark.

Implementation: `src/update_map/lifelong/`. Configuration: `lifelong:` in `configs/default.yaml`.

## Event supervision from localizer pose estimation

The original paper uses image-feature displacement voting. This repository keeps the paper's map-management logic but derives stronger labels from the configured matcher/2D-to-current-3D/robust-pose pipeline:

- PnP inlier: `correct`;
- matched correspondence rejected by the trusted geometric estimator: `incorrect`;
- eligible feature offered to matching but receiving no correspondence: `unmatched`.

`classify_feature_events(...)` performs this classification. `eligible_feature_ids` must contain only features genuinely eligible in the current view: projected into the FOV, not occluded, outside changed-region masks and accepted by image-quality gates. It must not be the whole map.

## Score-based adaptive map

For feature \(i\),

\[
S_i^{k+1}=\operatorname{clip}(S_i^k+r(e_i^k),S_{\min},S_{\max}),
\]

with

\[
r(e)=\begin{cases}
+s_c,&e=\text{correct},\\
-s_i,&e=\text{incorrect},\\
-s_n,&e=\text{unmatched}.
\end{cases}
\]

Defaults reproduce the strongest paper setting: \(s_c=1\), \(s_i=1\), \(s_n=0\), 500 features and 5% gradual exchange per trusted traversal. Thus a full map exchanges 25 entries.

`unmatched_penalty=0` does not discard the observation. It writes a zero-valued temporal sample. Absence is therefore not negative geometric evidence, while repeated time-dependent absence remains useful for visibility prediction.

The `score` policy ranks by cumulative \(S_i\). It retires the lowest-ranked exchange set and activates the same number of verified candidates. It never retires an exchange entry without an admissible replacement, and it can fill an initially incomplete sidecar up to `map_budget` without retirement.

## FreMEn temporal prediction

Each trusted observation gives timestamped value \(y_{i,k}=r(e_i^k)\) at time \(t_{i,k}\), measured in days. The model is

\[
\hat y_i(t)=\beta_{i,0}+\sum_{j=1}^{H}
\left[a_{i,j}\cos\left(\frac{2\pi(t-t_0)}{T_j}\right)
+b_{i,j}\sin\left(\frac{2\pi(t-t_0)}{T_j}\right)\right].
\]

For every candidate period \(T_j\), `fit_fremen_model(...)` performs irregular-time least-squares harmonic regression and ranks the component by

\[
P_j=a_j^2+b_j^2.
\]

The strongest sufficiently separated periods are refit jointly with ridge regularization:

\[
\hat\beta=(X^TX+\lambda R)^{-1}X^Ty,
\]

where the intercept is not regularized. Candidate periods are configurable; defaults include half-day, daily, weekly, monthly and yearly cycles. An optional geometric frequency grid supports unknown cycles.

At localization time, `select_features(timestamp_days, strategy="fremen")` ranks active features by \(\hat y_i(t)\). For retirement, the implementation follows the paper's long-term rule and ranks by mean historical temporal value, not by instantaneous phase. A night feature is therefore not permanently deleted merely because a daytime query is being processed.

## Descriptor-uniqueness admission

For candidate descriptor \(d_q\),

\[
U(d_q)=\min_{d\in\mathcal D_M}\operatorname{dist}(d_q,d).
\]

Higher values are preferred. `rank_candidates_by_uniqueness(...)` uses greedy farthest-first ranking: each selected descriptor joins the reference set before the next candidate is scored, preventing a batch of near-duplicates from occupying the exchange set.

Metrics:

- `cosine` for learned float descriptors;
- `l2` for Euclidean descriptors;
- `hamming` for binary descriptors such as BRIEF.

Uniqueness cannot override provenance. A candidate needs `verified_current_geometry=True` and a trusted `point3d_id`; otherwise it becomes `QUARANTINED` and cannot replace an active feature.

## Fail-closed update

```python
plan = manager.update_session(
    events=events,
    timestamp_days=timestamp_days,
    candidates=candidates,
    gate_passed=pose_estimate.quality.passed,
    gate_reason=",".join(pose_estimate.quality.failed_gates),
)
```

When `gate_passed=False`, this is an exact no-op: no score/history write, retirement, activation or quarantine. The plan records ignored IDs for audit. The caller must use the complete pose-quality decision, including spatial coverage, reprojection, positive depth, FIM conditioning, multi-reference consensus and ambiguous multi-modal rejection—not raw PnP success alone.

## Executable policies

| Strategy | Behavior |
|---|---|
| `static` | Ignore observations/candidates; frozen baseline |
| `latest` | Replace the active set with latest verified candidates; destructive ablation |
| `aggressive` | Replace incorrect and unmatched features when replacements exist |
| `strict` | Replace only incorrectly matched features |
| `summary` | Retire incorrect features and add all verified candidates; growing-map baseline |
| `score` | Gradually exchange lowest cumulative-score features |
| `fremen` | Harmonic prediction plus mean-score retirement; default |

The paper's separate multiple-map experience strategy is intentionally not flattened into this feature manager because that would destroy experience boundaries. This repository already handles multiple references/sessions at the reference-selection layer; a multiple-experience ablation belongs there.

## Integration contract

1. Build eligible IDs from active sidecar features visible in the query and accepted by stable/change/occlusion masks.
2. Associate each accepted localizer correspondence with its feature-memory ID.
3. Convert the PnP inlier mask:

```python
events = classify_feature_events(eligible_ids, matched_feature_ids, pnp_inlier_mask)
```

4. Convert new stable 2D-to-current-3D associations into `FeatureCandidate`. A recommended ID is `<reference_id>:<point2d_idx>:<point3d_id>` so multiple appearance observations of one landmark remain distinct.
5. Call `update_session(...)` only after all registration/change gates pass.
6. Persist with `manager.save(...)` beside the versioned candidate bundle. No current-map camera, image or point file is rewritten.
7. At query time expose only IDs returned by `select_features(...)` to the matcher/reference index.

`MapUpdatePlan` reports exact observed, activated, retired, quarantined and ignored IDs, exchange target/applied count, and active counts before/after.

## Required evaluation

At minimum compare `static`, `score` and `fremen` on identical held-out current-query sessions. Ablate exchange rates 0%, 1%, 5%, 10%, 100%; zero versus positive unmatched penalty; scalar versus temporal ranking; random versus uniqueness admission; and gate enabled versus disabled. Report common-success PnP inlier non-regression, weak-cell success, longest failure run, confident-wrong-pose count, map size and p95 latency. Base-map hashes must remain unchanged.

## Limitations

- Periodicity needs enough cycles and timestamp diversity; below `min_temporal_samples` prediction falls back to empirical mean.
- Harmonics model recurring conditions, not permanent structural change. Multi-view change evidence remains authoritative.
- Descriptor distance may indicate uniqueness or instability; geometry verification, repeatability and held-out localization utility remain necessary.
- `latest`, `aggressive` and `summary` are ablations and must not bypass candidate-bundle promotion gates.
