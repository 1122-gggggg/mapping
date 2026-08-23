# Map + localization diagnosis

`diagnosis/` is the former `sfm_map_diagnosis` tree. It lives in this repo.
Install from the mapping root:

```bash
pip install -e '.[dev]'
```

Optional extras: `[colmap]`, `[viz]`, `[video]` (Stage 0 video ingest).

This layer is **read-only** and venue-agnostic. It does not replace a deployment's
immutable build/test split or release contract.

## How the two halves complement each other

| Lifecycle | Diagnosis (`diagnosis/`) |
|---|---|
| Before mapping | Stage 0 `sfm-qa select-sessions` — advisory role assignment before first SfM |
| During/after mapping | Stage 1 `sfm-qa analyze` — MapDoctor health + weak-region screen through `MapModel` |
| After localization | Stage 2 `--logs` — method-agnostic per-query attribution |
| Map update | `mapdoctor compare` on the same immutable query universe |

A deployment's corpus lock and Stage 0 are complementary, not substitutes. The corpus
lock is the hard split gate; Stage 0 is advisory session-role advice.

## After S5/S6: screen the reconstruction

Preferred (MapDoctor + sfm-diagnosis, combined `report.json`):

```bash
python tools/diagnose_map.py \
  --model /path/to/map \
  --map-adapter package.module:AdapterClass \
  --output /path/to/diagnosis
```

MapDoctor HTML/CSV only:

```bash
mapdoctor analyze /path/to/map \
  --map-adapter package.module:AdapterClass \
  --output /path/to/mapdoctor
```

Built-ins are `colmap`, `glomap`, and `gluemap`. Other formats implement `MapAdapter` and
use `package.module:AdapterClass`; no diagnosis-core edit is required.

## After localization: attribute failures (optional)

When a per-query CSV or JSON exists:

```bash
python tools/diagnose_map.py \
  --model /path/to/map \
  --map-adapter package.module:AdapterClass \
  --logs loc.csv \
  --output /path/to/diagnosis
```

Required columns:

```text
query,success
```

Optional standardized quality columns enable their corresponding gates. `localizer` is an
arbitrary provenance label and JSON `metrics` may contain method-specific scalar evidence.
Optional `x,y,z` enable pose diagnosis. A deployment can force selected standardized fields
to be present with `localization.required_metrics`.

## Map-update regression

```bash
mapdoctor compare base.csv candidate.csv --output comparison_report
```

A candidate must not be promoted merely because an intermediate proxy improves
if it creates newly failed held-out queries.

## Scope boundary

Failure to install or run diagnosis does not silently alter map geometry, localizer assets,
camera intrinsics, or promotion state.
