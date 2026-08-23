# Limitations and non-claims

1. A direct localization failure is not enough to distinguish viewpoint gap from structural change. Bridge connectivity, old-old consistency and pose-aligned change evidence are required.
2. The included aligned-image change detector is an executable baseline, not a replacement for the full multi-view dense-feature method in the cited CVPR paper.
3. Standard COLMAP files do not encode GlueMap virtual-track provenance. Export a sidecar before using the map for strict production filtering.
4. FIM is a local linearized uncertainty approximation. It does not model retrieval failure, descriptor aliasing or all calibration/systematic errors; EDM front-end measurements remain mandatory.
5. PnP gates in `configs/default.yaml` are bootstrap engineering values. Calibrate them on the actual camera, 720p preprocessing, map scale and safety requirement.
6. A bridge proves coordinate connectivity, not current scene validity. Change detection after bridge registration is mandatory.
7. Monocular Sim(3) alignment can be numerically valid but semantically wrong under repeated structures. Multi-anchor and independent-path evidence are therefore hard gates.
8. The package does not ship EDM/GLUEMAP weights or datasets. Adapters integrate an existing implementation.
9. Without independent current validation, reference utility and promotion claims are provisional.
10. The repository does not rewrite the privileged map. A downstream application must load the sidecar or convert it into its localization bundle format.
