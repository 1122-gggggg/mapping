# Frozen base / core

After Stage 0, the selected `BASE_CORE` (+ needed `BASE_SUPPORT`) is
reconstructed once and then **frozen**. Later sessions do not get a vote to
move those cameras or points.

## Freeze the base

A frozen base is a read-only geometry authority for everything that follows:

- registered base poses stay put
- existing 3D points stay put
- later work may *observe* those points; it may not bundle-adjust them by
  default

If a leftover session cannot live with that freeze, it is not a silent
in-place update. It is `UPDATE_CANDIDATE`, `NEW_SUBMAP`, or `QUARANTINE`.

## Fringe points must still see the frozen base

New 3D points (fringe triangulation after a leftover localizes) are admitted
only when the track still has a **frozen-base view**:

1. Localize leftover cameras against the frozen base (do not rebuild).
2. Append observations of *existing* frozen points if the 2D evidence already
   exists.
3. Triangulate new points only from tracks that reproject cleanly into at
   least one frozen-base camera **and** the new views.

A fringe point that only exists among leftover cameras is not a base
extension. It is a disconnected splinter: keep it on a `NEW_SUBMAP` or
quarantine it.

## Localize vs the frozen base, not just-hung history

Appearance references, validation hold-outs, and update probes register
against the **frozen** reconstruction, not against:

- a map that already absorbed the session under test
- a just-hung incremental reconstruction of “everything so far”
- another leftover’s private submap, unless that submap was itself frozen
  and independently bridged

Using a moving history as the reference hides drift and makes Stage 2 logs
incomparable. Stage 2 (`sfm-qa analyze --logs`) should attribute queries
against the same frozen base Stage 0 selected.

## Never N+1 joint BA by default

Do not add one more video and jointly bundle-adjust it with the entire base.

That N+1 joint BA:

- lets a weak or changed session pull frozen geometry
- turns a single critical/ambiguous bridge into a silent merge
- conflates recapture localizability with map-information gain

Default path: freeze → localize leftovers → optional connected-fringe
triangulation → optional independent submap. Joint refinement of base +
newcomer is an explicit, separately justified experiment, not the pipeline
default. When unsure, use one fewer video.
