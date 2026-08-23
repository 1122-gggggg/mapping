# Pipeline

Primary commands: `sfm-qa select-sessions` (Stage 0) and `sfm-qa analyze` (Stage 1–2).

## Stage 0: multi-session selection

Runs from videos, before any initial SfM.

```
ALL VIDEOS
  → per-session QA
  → cross-session graph (VPR = candidates only; geometry = edges)
  → greedy Base-core (maximize U(S); timestamp never ranks)
  → remainder roles
  → only BASE_CORE + needed BASE_SUPPORT enter initial SfM
  → leftovers localize vs the frozen base
  → APPEARANCE_REF / fringe / NEW_SUBMAP / QUARANTINE
```

Fail-closed: uncertain → `QUARANTINE`; no reliable geometric edge → `NEW_SUBMAP`;
never force-merge on one critical or `AMBIGUOUS` bridge.

```bash
sfm-qa select-sessions --videos DIR --output DIR [--maps DIR] [--config PATH]
```

Selection is advisory: reports are written and the process exits 0. See
`docs/session_selection.md`, `docs/frozen_core.md`, and `docs/method_lessons.md`.

## Stage 1: map diagnosis

Always runs for `sfm-qa analyze` (and `check` / `check-map` / `check-localize`).

1. Load the reconstruction through the requested MapDoctor adapter (`colmap`, `glomap`, or `gluemap`).
2. Score MapDoctor readiness / health / covisibility fragility.
3. Convert the same in-memory model to `sfm_diagnosis.MapData`.
4. If `--database`, `--pairs`, `--images-manifest`, or `--images-dir` is given, load optional build evidence and pass it into `analyze_weak_regions`.
5. Write `DIR/map/report.json`. The combined report keeps the same payload under the `map` key.

`diagnostic_mode` comes from the weak-region summary. It is present even when no build evidence is supplied.

## Stage 2: SfM localization

Runs only when `--logs` is given.

1. Score the MapDoctor-schema localization CSV.
2. Attribute failed queries with `diagnose_pose` when `x,y,z` exist.
3. Write `DIR/sfm/report.json`.

This stage does not run a localizer. The CSV is an already-computed result log.

If `--logs` is omitted, Stage 2 is skipped and `overall_status` is `MAP_SCREENED_LOCALIZATION_UNCHECKED` (or `MAP_SCREENING_FAILED`).

Attribute leftover / recapture queries against the **frozen** Stage 0 base, not a just-hung N+1 reconstruction.

## Combined report

`DIR/report.json` contains `overall_status`, `map`, and `localization` (`null` when Stage 2 is skipped).

Exit 0 only for `READY` and `MAP_SCREENED_LOCALIZATION_UNCHECKED`.
(`select-sessions` is outside this table and always exits 0 after writing.)
