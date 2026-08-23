# Data contracts

## 1. Current map

The configured `adapters.map_loader` must return `update_map.models.BaseMap`.
The built-in COLMAP-compatible loader accepts:

```text
<base>/cameras.bin images.bin points3D.bin
<base>/cameras.txt images.txt points3D.txt
<base>/sparse/0/cameras.bin images.bin points3D.bin
```

Optional GLUEMAP provenance sidecars:

### `point_provenance.json`

```json
{
  "1": "CURRENT_REAL",
  "2": "CURRENT_FEEDFORWARD_VERIFIED",
  "9012": "VIRTUAL_BA_ONLY"
}
```

### `virtual_point_ids.txt`

```text
9012
9013
9014
```

Without provenance metadata, all imported points are conservatively represented as current real points because standard COLMAP files do not encode GlueMap's internal track origin. Production use should export provenance explicitly.

## 2. Image manifest

CSV fields:

```text
image_id,path,source,session_id,sequence_id,frame_index,timestamp,
camera_id,width,height,quality_status,quality_metrics,metadata
```

`source` is one of `CURRENT_MAP`, `CURRENT_VALIDATION`, `HISTORICAL_UPDATE`.

## 3. Retrieval

```json
{
  "queries": {
    "historical_update:abc123": [
      {"reference": "images/ref_0001.jpg", "score": 0.91},
      {"reference": "images/ref_0042.jpg", "score": 0.84}
    ]
  }
}
```

Reference IDs must resolve to the `name` field in the current `BaseMap`, or to its numeric image ID.

## 4. Localizer pair file

Compressed NumPy `.npz`:

```python
query_xy      # float32/64 [N,2]
reference_xy  # float32/64 [N,2]
confidence    # float32/64 [N]
sigma         # optional [N] or [N,2]
```

Default filename is the SHA-1 of `query_id + "\n" + reference_id`. An optional `index.json` may map pair keys to arbitrary paths:

```json
{
  "pairs": {
    "query_id\nreference_id": "session/q0001__r0042.npz"
  }
}
```

## 5. Change mask

`.npz`:

```python
labels       # uint8 HxW: 0 invalid, 1 stable, 2 changed, 3 uncertain
confidence   # optional float HxW
```

Only label 1 is PnP-eligible.

## 6. Historical sidecar

`historical_references.json` records pose, provenance, stable ratio, bridge depth, anchors, current point IDs and state.

`observations.json` records:

```text
historical_image_id
historical_xy
current_point3d_id
confidence
supporting_references
provenance
```

No XYZ coordinates are stored for new historical-only production points because such points are forbidden.

## 7. Coordinate convention

All stored absolute camera poses are world-to-camera. External adapters that use camera-to-world must invert them before returning `Pose`.

## 8. Dataset split audit

`manifests/split_audit.json` records exact-path overlap, optional SHA-256 duplicate-content overlap, session/sequence name collisions and the resulting validation grade. A real production claim requires an independent current session; absent current-map source images or detected overlap is downgraded to `PROVISIONAL_PROXY_VALIDATION`.

## 9. Candidate registry

A staged production bundle contains only localization sidecars, masks and manifests. Any embedded `cameras.*`, `images.*` or `points3D.*` reconstruction is rejected so that historical-only geometry cannot be activated accidentally. Promotion requires a regression JSON with `"passed": true`, verifies the frozen base-map snapshot and atomically updates `active_bundle.json`.
