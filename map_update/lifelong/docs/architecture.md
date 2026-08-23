# Architecture

## 1. Map ownership model

The system treats the latest adapter-normalized reconstruction as a privileged geometry:

\[
M_{core}=\{K_i,T_i^{current},X_j^{current},\mathcal O_j^{current}\}
\]

It is read-only. Historical data are stored in a separate localization layer:

\[
M_{loc}=\{I_r,T_r,A_r,M_r^{stable},q_r,\text{provenance}_r\}
\]

where `A_r` maps historical reference pixels to existing current `point3D_id` values. A historical observation may increase the set of views from which a current landmark can be matched, but it does not move that landmark.

## 2. Packages

| Module | Responsibility |
|---|---|
| `map_adapters` | Built-in COLMAP-compatible loader plus importable `BaseMap` loaders |
| `io.colmap` | Pure-Python implementation of the built-in text/binary loader |
| `io.hashing` | Base-map snapshot and mutation detection |
| `io.manifests` | Session-aware input manifests |
| `quality` | Blur/exposure/entropy/near-duplicate filtering |
| `adapters` | Precomputed, Python callable, or command-based localizer integration |
| `lifting` | Matcher reference pixel to fixed current point3D lifting |
| `pose` | Per-reference PnP, clustering, multimodality rejection, refinement |
| `metrics` | Reprojection, spatial support, positive depth, FIM and gates |
| `change` | Stable/changed/uncertain masks and multiview fusion |
| `bridge` | Image graph, current-ID propagation, Sim(3), cycle gates, anchored pose graph |
| `selection` | Route-cell K-cover, front-end/FIM utility, redundancy and budget |
| `stability` | Currentness and historical-view utility histories |
| `experiments` | E0–E5, A1–A11 and non-regression evaluation |
| `pipeline` | Direct historical registration and sidecar export |

## 3. Coordinate conventions

`Pose` is world-to-camera:

\[
x_c = R_{cw}x_w+t_{cw}.
\]

The camera center is:

\[
C_w=-R_{cw}^{\top}t_{cw}.
\]

`Sim3` maps a source submap to the current map:

\[
x_{current}=sRx_{old}+t.
\]

Every adapter must state its convention before converting to these types. Tests should fail when a transform is silently transposed or inverted.

## 4. Provenance barriers

Each current point carries one of:

- `CURRENT_REAL`
- `CURRENT_FEEDFORWARD_VERIFIED`
- `VIRTUAL_BA_ONLY`
- `HISTORICAL_GEOMETRY_ONLY`

The default lifting config accepts only the first two. `VIRTUAL_BA_ONLY` may constrain GLUEMAP bundle adjustment but is not a physical PnP landmark. Historical-only points may initialize a candidate submap but cannot be exported in the production sidecar.

If the GlueMap exporter does not preserve provenance, add either:

```text
point_provenance.json
```

with `{point_id: provenance}` entries, or:

```text
virtual_point_ids.txt
```

before trusting the localization map.

## 5. Current-first online path

1. Retrieve and match current references.
2. If current PnP passes all gates, return it.
3. Only when current support is weak or the route cell is known to be weak, retrieve selected historical active references.
4. Filter every historical reference match by its stable mask.
5. Lift only to current point IDs.
6. Estimate per-reference poses, cluster in SE(3), and accept only a unique dominant mode.
7. Refine a source-aware weighted pose and return it only if all production gates pass.

This controls latency and prevents a large historical pool from increasing aliasing on already healthy queries.

## 6. Sidecar output

The augmented bundle is deliberately not a rewritten COLMAP model. It contains:

```text
historical_reference_sidecar/
├── manifest.json
├── historical_references.json
├── observations.json
└── stable_masks/
```

A deployment adapter can transform this sidecar into the format expected by any localization service. The original base map remains independently hash-verifiable and rollback requires selecting the previous sidecar version rather than reconstructing geometry.

## Production activation layer

`update_map.online.CurrentFirstLocalizer` implements the production policy: current references are tried first, selected active historical references are queried only when the current pass is weak, and a multi-modal fallback is rejected fail-closed.

`update_map.bundle.CandidateBundleManager` stages versioned sidecars, verifies the frozen base-map snapshot before promotion, maintains an atomic active-bundle pointer, and supports rollback without rewriting current-map geometry.
