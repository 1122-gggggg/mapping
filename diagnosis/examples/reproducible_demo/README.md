# Reproducible end-to-end demo

This example exercises MapDoctor without downloading a private or third-party dataset. It deterministically generates a synthetic sparse reconstruction and a frozen held-out query benchmark, then runs the complete workflow.

The demo is intended for **software reproducibility and interface validation**, not as evidence that the default thresholds are universally optimal for real scenes.

## Run

From an editable MapDoctor checkout:

```bash
python examples/reproducible_demo/run_demo.py --output /tmp/mapdoctor-demo
```

The runner performs all of these checks:

1. Generates one COLMAP-format sparse reconstruction with 8 registered reference views and 80 tracked 3D points.
2. Loads the same reconstruction independently through:
   - `ColmapAdapter`
   - `GlomapAdapter`
   - `GluemapAdapter`
3. Verifies producer provenance is preserved in each report.
4. Generates 20 frozen held-out query rows for a clean base benchmark.
5. Generates the same 20 query IDs for a candidate benchmark, deliberately degrading `query_007.jpg` and `query_015.jpg`.
6. Verifies the base strict success rate is 100%.
7. Verifies the candidate strict success rate is 90%.
8. Runs `mapdoctor compare` and verifies the candidate is rejected by the regression gate.
9. Verifies the two newly failed query IDs are surfaced explicitly.

Expected console summary:

```text
MapDoctor reproducible demo: PASS
Three adapters: COLMAP / GLOMAP / GLUEMAP
Base benchmark: 100.0% strict success
Candidate benchmark: 90.0% strict success
Regression gate: FAIL (expected)
```

## Generated artifacts

The output directory contains:

```text
data/
  sparse/0/{cameras.txt,images.txt,points3D.txt}
  base.csv
  candidate.csv
map_health/
  colmap/{report.json,report.html,weak_images.csv}
  glomap/{report.json,report.html,weak_images.csv}
  gluemap/{report.json,report.html,weak_images.csv}
benchmark/
  base/{benchmark.json,benchmark.html,queries.csv}
  candidate/{benchmark.json,benchmark.html,queries.csv}
comparison/
  comparison.json
  comparison.html
  query_deltas.csv
```

Because all input data is generated from source code, this example can run in CI and can be reproduced by users without accepting a dataset license or downloading large assets.

For real-world scale and motivation, see `docs/CASE_STUDY_TARGET_SITE.md`. For external/localizer interoperability, see the hloc exporter in `docs/INTEGRATIONS.md`.
