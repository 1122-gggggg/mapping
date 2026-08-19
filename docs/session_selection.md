# Stage 0: multi-session selection

Objective: decide which videos should enter the first SfM, which should wait as
frozen-base localizers or appearance references, and which must stay out.

Selection is **advisory**. `sfm-qa select-sessions` always exits 0 after writing
reports. It does not run SfM, does not mutate maps, and does not authorize a
merge.

## Objective `U(S)`

For a candidate base set `S`, the greedy selector maximizes

```
U(S) = α coverage + β quality + γ connectivity + δ redundancy
     + ε information + ζ view_diversity − η track_cost − θ risk
```

Weights live in `config/session_select.yaml` and are labeled **heuristic**. They
are not a fitted site score.

Greedy procedure:

1. Score each session internally (registration ratio, parallax, reprojection,
   sharpness, cycle error). Missing metrics fail closed — never invent `STRONG`.
2. Seed `S` with the highest internal-quality `STRONG`/`USABLE` session.
   **Timestamp is never a ranking key.**
3. Add the admissible neighbor with largest `ΔU`, subject to track/observation
   budgets and diminishing-return stop gates (`min_information_gain`,
   `min_coverage_gain`).
4. Split `S` into `BASE_CORE` (removing it drops coverage / connectivity /
   information a lot) and `BASE_SUPPORT` (helps, but is not load-bearing).
5. Classify leftovers into remainder roles.

Prefer **one fewer video** over a cheap merge that adds tracks without
independent geometry.

## Fail-closed bridges

A session-to-session link is a geometric **edge** only after verification:

- shared tracks / verified pairs above heuristic floors
- rotation, translation-direction, and scale consensus
- cross-session reprojection
- **≥ 2 independent bridge groups** (temporal and spatial separation)
- Sim3 fit on a support set, validated on a **hold-out** (never fit and
  validate on the same points)

Fail-closed rules:

| Condition | Action |
| --- | --- |
| Uncertain metric or missing evidence | `QUARANTINE` |
| No reliable geometric edge to the base | `NEW_SUBMAP` |
| Only `REJECT` / `AMBIGUOUS` links | do not merge |
| Unique critical bridge, or one unverified critical bridge | do not force-merge |

A critical bridge is the unique usable connector of two groups of size ≥ 2.
Never promote a session into the base on that single hinge.

## VPR is not an edge

Visual place recognition / retrieval (and any similar candidate generator)
only proposes pairs. A high retrieval score is **not** a geometric edge, not
covisibility, and not merge authority. Candidates still need independent
bridges and a Sim3 hold-out.

## Track budgets

Optional hard caps in the YAML (`max_base_sessions`, `max_total_tracks`,
`max_total_observations`, `max_tracks_per_session`) stop the greedy walk
before the reconstruction becomes track-heavy. When a candidate would blow a
budget, skip it. Diminishing `Δcoverage` / `Δinformation` with a rising track
cost is a stop, not a reason to keep adding video.

## Role vocabulary

| Role | Meaning |
| --- | --- |
| `BASE_CORE` | Load-bearing sessions for the first SfM. |
| `BASE_SUPPORT` | Helpful but not load-bearing; may enter initial SfM when needed. |
| `APPEARANCE_REF` | Same geometry, different appearance. Localize vs the frozen base; do not rebuild. |
| `GEOMETRY_REINFORCEMENT` | Extra geometry for a known weak region, only with ≥ 2 independent bridges. |
| `UPDATE_CANDIDATE` | Scene change vs the frozen base. Not “newer timestamp”. |
| `NEW_SUBMAP` | Internally usable, no reliable edge to the base. Own map, not a forced merge. |
| `QUARANTINE` | Uncertain, high-influence without support, or a single critical/ambiguous hinge. |
| `REJECT` | Internally inconsistent or unusable. |
| `VALIDATION_ONLY` | Hold-out. Never used to fit the base it is supposed to test. |

Only `BASE_CORE` plus needed `BASE_SUPPORT` enter the initial SfM. Everything
else waits: localize vs the frozen base, keep as appearance/fringe evidence,
open a new submap, or stay quarantined.

## CLI

```bash
sfm-qa select-sessions --videos DIR --output DIR [--maps DIR] [--config PATH]
```

| Flag | Required | Meaning |
| --- | --- | --- |
| `--videos` | yes | Directory of input videos (one session per video unless ingest says otherwise). |
| `--output` | yes | Directory for Stage 0 reports. |
| `--maps` | no | Optional existing reconstructions to score against. |
| `--config` | no | YAML overlay merged onto `config/session_select.yaml`. |

Prints per-role counts. Exit code is always 0 after reports are written.

Python entry: `sfm_qa.session_select.run.select_sessions(video_dir, output_dir, config=None, maps_dir=None)`.
Package import root: `sfm_qa.session_select`.

Video decoding uses the optional extra:

```bash
pip install -e '.[video]'
```
