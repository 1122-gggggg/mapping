# Method lessons (site-agnostic)

These are constraints on how Stage 0–2 may use evidence. They are not a
scorecard and they do not name a site, flight, or successor map as authority.

## Keep HG / HA / HL separate

Do not collapse reconstruction into one number.

| Axis | What it measures | What it is not |
| --- | --- | --- |
| **HG** geometry | Parallax, track length, cycle / Sim3 consistency, independent bridges | “Looks like the same place” |
| **HA** appearance | Illumination, season, weather, recapture matchability | Proof the scene geometry is unchanged |
| **HL** localization | Query pose gates against a *frozen* map | Proof the map should absorb the query video |

A session can be a strong appearance reference (HA) while adding no geometry
(HG). A session can localize well (HL) while being a bad base member. Mixing
the three into one rank is how timestamp and retrieval sneak into Base
selection.

## FIM logdet / condition is observability, not loc probability

Fisher information (`logdet`, condition number, λ_min) describes how
observable a pose or point is *under the measurement model you assumed*.
It is useful for ranking coverage and for spotting degenerate directions.

It is **not**:

- P(this query will localize)
- a calibrated success probability
- a reason to merge two maps

Treat FIM as an observability / information term inside `U(S)` and inside
map diagnosis. Calibrate localization probability only from held-out query
outcomes (Stage 2 logs), never from FIM alone.

## Recapture localizability ≠ map-information gain

A leftover that PnPs cleanly onto the frozen base is *localizable*. That does
not mean adding it to the reconstruction increases map information.

- Localizable + same geometry + different appearance → `APPEARANCE_REF`
- Localizable + independent new parallax on a known weak region → maybe
  `GEOMETRY_REINFORCEMENT`
- Localizable + scene change → `UPDATE_CANDIDATE`
- Localizable on a single hinge / ambiguous pair → `QUARANTINE`

Information gain needs new, well-conditioned tracks that survive hold-out.
A successful recapture is necessary for some roles and sufficient for none
of the merge roles.

## VPR is a candidate only

Retrieval proposes pairs. Before any merge or base extension:

1. cluster candidates into **≥ 2 independent bridges** (separated in time
   and in reference-camera space)
2. fit Sim3 on a support set
3. require the **hold-out** residual to stay under the heuristic relative
   cap
4. refuse a unique critical bridge or a single `AMBIGUOUS` link

No independent model ⇒ no Sim3 ⇒ no merge. Fail closed. `map_fusion`
stays unauthorized until those gates pass; even then Stage 0 only *advises*.

## Timestamp ≠ Base / Update

Capture time does not rank `BASE_CORE` or `BASE_SUPPORT`. A later video is
not an update; an earlier video is not automatically the core.

`UPDATE_CANDIDATE` is for measured scene change versus the frozen base, not
for “filmed afterwards”. If time is the only difference and geometry holds,
the leftover is an appearance or validation resource.

## Worst case: use one fewer video

When a candidate is cheap in retrieval score but expensive in risk — one
critical bridge, weak independent support, high influence without evidence,
track-budget blow-up, or a non-positive `ΔU` — drop it.

The failure mode this system is built to avoid is an over-complete base that
looks connected and will not localize. Prefer a smaller frozen core plus
`NEW_SUBMAP` / `QUARANTINE` leftovers over a forced N+1 merge.
