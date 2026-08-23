# Validation Summary - 2026-07-02

> Archived snapshot. References to the former operator/one-click build pipeline describe
> the 2026-07-02 external workspace and are not current repository entrypoints. Current
> builds use the site-specific S0-S9 tools.

## Package

Command:

```bash
/usr/bin/python3.12 /media/cihcilab/新增磁碟區/sfm_system/tools/verify_package.py
```

Result: PASS

- external disk root contains only `sfm_system`;
- no broken symlinks;
- no absolute symlinks;
- JSON configs parse;
- Python entrypoints compile;
- `operator_pipeline.py verify --allow-blocked` produces `system_verify_latest.md/json`;
- Sphinx system tools installed;
- Secure Boot is enabled, so Sphinx firmware launch remains blocked.

## Build Readiness

Result: PASS

- build pipeline CLI is callable from `operator_pipeline.py build --help`;
- MV-RoMa root exists;
- MV-RoMa `outdoor_final.pth` exists, 3.3GB;
- UFM root exists;
- LFOE `glomap_filter` exists and is executable;
- Doppelgangers++ source exists.

Optional:

- Doppelgangers++ checkpoint is not present. This is only required when repeated
  structure filtering is explicitly enabled.

Base map artifacts:

- `base_glomap_fused_0/cameras.bin`: present
- `base_glomap_fused_0/images.bin`: present
- `base_glomap_fused_0/points3D.bin`: present
- `base_reloc_map_xfeat_tri.pt`: present
- `base_megaloc_cache_v3.npz`: present

## Updated Map Candidate

Output:

```text
/media/cihcilab/新增磁碟區/sfm_system/更新地圖/outputs/verify_update_20260702
```

Superseded by the flight-deployable tracking-ready candidate:

```text
/media/cihcilab/新增磁碟區/sfm_system/更新地圖/outputs/verify_update_20260702_tracking_skip_p122
```

Artifacts:

- `reloc_map_updated.pt`
- `latest_map_realrgb.ply`
- `update_report.md`

Update gate: PASS

Summary:

- base bundle: 1920 keyframes
- updated bundle: 2095 keyframes, with complete `ref_centers/ref_yaws/covis`
  tracking metadata for every keyframe
- P1210121: register-only
- P1220122: high-overlap QA skip
- P1240124: submap + Sim3, 60 bridges, residual 0.123u
- P1250125: submap + Sim3, 18 bridges, residual 0.022u

## Localization Validation

Validation uses MegaLoc retrieval top-30 plus XFeat/LighterGlue/PnP at 720p.
This matches the production boot/lost localization setting.

### 補拍影片/test

Report:

```text
/media/cihcilab/新增磁碟區/sfm_system/定位/outputs/compare_verify_update_tracking_skip_p122_test_topk30_20260702.quality_report.md
```

Result: PASS

| Set | Base success | Updated success | ok->fail | maxfail updated |
|---|---:|---:|---:|---:|
| P1230123 | 89.6% | 90.0% | 0 | 27 |
| P1260126 | 100.0% | 100.0% | 0 | 0 |

P123 is a baseline-improved borderline pass: the old map is below 90%, the
updated map improves success rate, has no ok-to-fail regression, and reduces the
longest failure run from 28 to 27.

### Downloads Validation

Report:

```text
/media/cihcilab/新增磁碟區/sfm_system/定位/outputs/compare_verify_update_tracking_skip_p122_downloads_topk30_20260702.quality_report.md
```

Result: PASS

| Set | Base success | Updated success | ok->fail | maxfail updated |
|---|---:|---:|---:|---:|
| P0720072 | 99.9% | 99.9% | 0 | 1 |
| P0730073 | 100.0% | 100.0% | 0 | 0 |

## Mission / Flight Control

Commands:

```bash
/usr/bin/python3.12 /media/cihcilab/新增磁碟區/sfm_system/operator_pipeline.py mission --mode flight-selftest
/usr/bin/python3.12 /media/cihcilab/新增磁碟區/sfm_system/operator_pipeline.py mission --mode dry-run
```

Result: PASS

- heading fusion sanity check passed;
- PCMD forward/yaw/gaz sign sanity check passed;
- dry-run reached route complete and entered LANDING.

## Sphinx / Olympe

Report:

```text
/media/cihcilab/新增磁碟區/sfm_system/定位/mission/outputs/sphinx_smoke_20260702.md
```

Installed:

- `parrot-sphinx` 2.25.2
- `parrot-ue4-empty` 2.25.2
- Olympe 8.4.0 wheelhouse

Smoke result:

- Olympe import and command construction: PASS
- Sphinx core, Gazebo, UE4, and firmwared connection: reached
- ANAFI firmware launch: BLOCKED by `SecureBoot enabled`

Required external action:

```text
Disable Secure Boot in BIOS/UEFI, then log out and log back in.
```
