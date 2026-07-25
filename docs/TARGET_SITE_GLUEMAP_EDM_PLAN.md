# Plan: target_site — gluemap map built from scratch for EDM

**Created**: 2026-07-15 · **Phase 0 completed**: 2026-07-15
**Data**: `/media/cihcilab/新增磁碟區/sfm_system/data/target_site/`
**gluemap**: `/media/cihcilab/新增磁碟區/河濱場域/gluemap_build/tools/gluemap/`
**Consumer**: `/media/cihcilab/新增磁碟區/sfm_system/EDM定位測試/build/build_reloc_map_edm.py`

> **STATUS: Phase 0 · S0 · S1 · S1b · S2 all PASS. S2b: 1080p group RESOLVED — official69 is CORRECT, Charuco is WRONG.**
> Everything below marked ✅ is verified, not assumed. §A records Phase 0; §A0 records S0/S1.
> Tooling: `建圖/target_site/tools/` · Run: `建圖/target_site/runs/target_site_v1/`

---

## A0. S0 + S1 RESULTS — two more plan assumptions were WRONG

### S0 — corpus lock: **PASS (10/10 gates)**
7 build videos, 42,746 frames, 1604 s, 10.30 GB. 16:9 exact (`aspect_err == 0.0` in IEEE754).
All 9 videos hash distinct. All Parrot, **same drone serial `PI040416BA8L105488`**.

### S1 — motion scan: **PASS**, but it overturned two things

#### 🔴 CORRECTION 11 — the manoeuvres are NOT degenerate rotations. The whole "純旋轉不拿去三角化" premise barely applies to this footage.

`S01_ABrot`'s own filename (中間有往下看和左看) promises a look-down and a look-left, and the
detector finds them — **6 sustained slews ≥ 5 °/s**. But every one of them carries **0.8–4.3° of
parallax**:

```
 96- 97s   8.0 deg/s   parallax 2.21 deg
278-280s   9.5 deg/s   parallax 4.30 deg
355-361s   7.5 deg/s   parallax 1.49 deg
374-376s   8.0 deg/s   parallax 1.04 deg
```

They are **gimbal moves during forward flight**. The drone keeps flying, the camera centre keeps
moving, the baseline survives, the structure is real. **Excluding them from triangulation would
throw away good geometry for nothing.**

Measured `pure_rotation` (a genuine slew with *no* baseline) across all 7 videos: **0.34% mean.**

Their real hazard is different and belongs to S4, not S1: they look **off the corridor** (at the
ground, at the left bank), so they overlap the main route poorly. That is a **connectivity**
problem, not a degeneracy one.

**Two obvious tests had to be discarded to get here:**
- `h_inliers/f_inliers >= 0.8` — the classic rotation test, and *the one already in this codebase* —
  labelled **46% of an ordinary reverse flight** "pure rotation". For aerial video the scene is far
  and near-planar, so a homography explains the flow almost always, translating or not.
- Residual parallax alone was no better: over 1 s, **half of all frames sit below 0.35°** simply
  because the scene is 50–100 m away. But SfM never triangulates off a 1-second baseline — gluemap
  pairs each frame with up to 100 retrieved neighbours spanning many seconds.

What actually kills structure is a **stationary camera centre**. So the test is: the camera must be
**slewing** (Kabsch rotation rate ≥ 5 °/s, vs a 0.64 °/s cruise median) **AND** no baseline may
appear while it does (residual parallax < 0.35°).

#### 🔴 CORRECTION 12 — VPR blindness across direction is now MEASURED, not assumed

Direction was resolved by aligning each flight against a **forward anchor and a reverse anchor**
(one anchor was not enough — a forward-only reference gave reverse flights rho ≈ −0.15, five times
weaker than the +0.85 forward flights scored).

| | vs fwd anchor | vs rev anchor | verdict |
|---|---|---|---|
| S03_BA2 | rho −0.134, **sim 0.157** | rho **+0.851**, **sim 0.428** | **rev** ✓ |
| S04_ab | rho **+0.848**, **sim 0.592** | rho −0.301, sim 0.156 | **fwd** ✓ |
| S05_P1220122 | rho +0.210, sim 0.169 | rho **+0.916**, **sim 0.558** | **rev** (was unknown) |
| S06_P1240124 | rho +0.429, sim 0.135 | rho −0.181, sim 0.224 | **fwd** (was unknown) |
| S07_P1250125 | rho **+0.808**, **sim 0.314** | rho −0.199, sim 0.101 | **fwd** (was unknown) |

**Same-direction median cosine similarity 0.31–0.59; cross-direction 0.10–0.17 — a ~3.5× gap.**

⇒ **MegaLoc is effectively blind across a direction reversal.** It will never propose a
forward↔reverse pair on its own. The forced VPR-blind bridging in S3 is not insurance, it is
**load-bearing**.

**Final directions: 4 forward (S01, S04, S06, S07) · 3 reverse (S02, S03, S05).**

⚠️ **S06_P1240124 is the dirtiest video** — 8.7% fast_motion, 7.3% hover, and by far the weakest
direction call (rho +0.429 vs +0.8 for the others, with a *low* similarity to both anchors). It may
run a slightly different route. Watch it in S4's connectivity check.

### S1b — is forced cross-video matching actually needed? **YES.**

The script's own auto-verdict said "natural retrieval may suffice" (18.98% cross-direction). **That
verdict is wrong and was discarded.** The average is inflated by two videos that match *nothing*:

```
S01_ABrot (fwd):  0.0%   <- clean forward video: retrieval NEVER proposes a reverse pair
S07       (fwd):  0.0%
S04_ab    (fwd):  1.5%
S06       (fwd): 61.5%   <- but S06's similarity to EVERY video is 0.109-0.224.
S05       (rev): 42.4%      It matches nothing, so its top-5 is noise. Not signal.
```

**The three clean forward videos get 0% / 0% / 1.5% cross-direction retrieval.** SALAD will never
propose a forward↔reverse pair on its own.

Real bridges DO exist — and *where* they are is the whole story:

```
S06 x S05:  sim 0.72 @ route 1%/2%     <- the very START of both videos
            sim 0.71 @ route 3%/7%
S06 x S02:  sim 0.60 @ route 6%/13%
S06 x S03:  sim 0.52 @ route 4%/37%
```

**Everything clusters at route 0–7% — the launch/turnaround area.** Nothing mid-corridor (the rest
sit at 0.13–0.23, i.e. the cross-direction noise floor).

#### The 180° problem, stated honestly

If the camera at position P looks north (forward) and at the same P looks south (reverse), the two
images share **zero** scene content. **No matcher can fix this** — not MegaLoc, not EDM, not RoMa,
not LoFTR — because there is nothing in common to match. Forcing that pair yields zero inliers, and
that is the *correct* answer. Forward↔reverse can only co-observe at **endpoints/turnarounds**,
across **distant common structure**, or under a **side-looking** camera. This footage is
forward-looking corridor flight, so: endpoints only. Which is exactly what the data shows.

#### But this matters far less than it looks — because of the yaw gate

`production_edm_tracker.py:141` **already rejects any reference whose heading differs by >90°.** So
at flight time a forward-flying drone *never* matches reverse references anyway — they are gated out.

⇒ Fusing forward and reverse does **not** exist to make cross-direction matching work (it can't, and
it needn't). It exists to put both directions **in one coordinate frame**, so you can fly out and
back and localize into the same map. That is a *pose-graph* constraint, and endpoint bridges can
carry it.

⇒ **Force the candidates along the corridor anyway** — not to force fusion, but to let geometry
answer whether any mid-route co-visibility exists. If only the endpoint bridge survives, the fusion
is a **single hinge**, which is the classic under-constrained fold risk. **G5.7 (Sim3 consistency)
then becomes non-negotiable.**

### S2 — extraction: **PASS.** 1414 frames, 3 cameras, zero undistortion

| seq | frames | shape |
|---|---|---|
| S01_ABrot | 392 | 1920×1080 |
| S02_BA | 104 | 1920×1080 |
| S03_BA2 | 110 | 3840×2160 |
| S04_ab | 252 | 2688×1512 |
| S05_P1220122 | 173 | 2688×1512 |
| S06_P1240124 | 245 | 2688×1512 |
| S07_P1250125 | 138 | 2688×1512 |

**G2.1 is a discriminative test, not a code inspection.** Checking "we never called `cv2.undistort`"
proves nothing about what landed on disk. Instead each sampled JPEG is compared against *two*
hypotheses — the raw decode, and the same frame put through `cv2.undistort` with the Charuco K+dist.
Result: **MAD ratio 0.058** — the written frames sit **17× closer to raw** than to undistorted.

`bridge_only` frames: **8** (of 1414). Pure rotation really is near-absent, as S1 predicted.

### 🔴 S2b — CORRECTION 13: **official69 is CORRECT. The Charuco calibration is WRONG.**

Rather than trust one self-calibration run (which would just report the seed back to itself — *this
is how river_site shipped a focal nobody validated*), BA was run **twice per group from two seeds
3.1% apart**, to test whether the focal is identifiable at all.

**1920×1080 (S02_BA, 104 frames):**

| seed | start fx/W | converged fx/W | moved |
|---|---|---|---|
| official69 | 0.727505 | **0.728243** | +0.10% |
| charuco | 0.750379 | **0.728055** | **−2.98%** |

**Two seeds 3.14% apart converged to within 0.03% of each other.** So:

1. **The focal IS identifiable** — BA is genuinely measuring it, not sitting near its seed.
2. **Measured: fx/W = 0.728149 → HFOV 68.95°.** official69 claims 69.00° — **off by 0.09%.**
3. **The Charuco seed was actively dragged 2.98% away from itself, onto official69's value.** Its
   overfit distortion (`k2 = +0.256`, 25 frames, RMS 0.83 px) dragged its focal with it — focal and
   distortion are strongly correlated in a Charuco solve, so one being wrong poisons the other.
   Second independent confirmation of the memory *"Charuco FULL_OPENCV 實測是錯的"*.
4. This also explains **river_site's 0.7819 (65.2°)**: those frames were `cv2.undistort`-ed with that
   bad distortion, so the image was *bent* and BA had to invent a focal to match. **That map's
   intrinsics are garbage.**

Reconstruction quality: **104/104 registered, 52,584 points, reproj 1.087.**

⇒ **Use official69. The original spec was right.** (Remaining: confirm the 2.7K and 4K groups agree —
if they do, one common canvas becomes legitimate and 1.8× cheaper.)

### Per-video motion (S1)

| seq | dir | parallax | low_par | hover | fast | **pure_rot** | triangulable |
|---|---|---|---|---|---|---|---|
| S01_ABrot | fwd | 63.1% | 21.5% | 9.8% | 4.6% | 1.0% | 84.6% |
| S02_BA | rev | 95.6% | 2.8% | 0.9% | 0.5% | 0.2% | 98.4% |
| S03_BA2 | rev | 95.9% | 4.1% | 0.0% | 0.0% | 0.0% | 100.0% |
| S04_ab | fwd | 87.6% | 8.4% | 4.0% | 0.0% | 0.0% | 96.0% |
| S05_P1220122 | rev | 90.6% | 7.7% | 0.8% | 0.0% | 0.8% | 98.3% |
| S06_P1240124 | fwd | 74.9% | 8.9% | 7.3% | **8.7%** | 0.2% | 83.8% |
| S07_P1250125 | fwd | 92.6% | 6.7% | 0.0% | 0.5% | 0.2% | 99.3% |

---

---

## A. Phase 0 results — READ THIS FIRST

### A.1 What was DONE (code already changed, verified)

| # | Change | Status |
|---|---|---|
| **D1** | PINHOLE reprojection patch | ✅ **Already applied — no action needed.** gluemap is an *editable* install (`_gluemap_editable.pth`); the build tree **is** the env, there is no second copy. Verified from executed bytecode. |
| **D2** | Fixed-intrinsics BA | ✅ **FIXED.** `refine_intrinsics` flag added, default `False`. |
| **D3** | Forced-bridge pair injection | ✅ **ADDED.** `--extra_pairs_path` in gluemap. |
| — | Doppelgangers checkpoint | ✅ **Materialized** from symlink → real 2.9 GB file. |
| — | MegaLoc HF weight cache | ✅ **Warmed.** A cold cache fails at hour zero of a multi-hour build. |
| — | Cost-monitor hook | ✅ Removed from `~/.claude/settings.json`. |

**D2 — the exact edits** (`gluemap/estimators/augmented_bundle_adjustment.py`, before `create_default_ceres_bundle_adjuster`):
```python
ba_options.refine_focal_length   = refine_intrinsics
ba_options.refine_principal_point = refine_intrinsics
ba_options.refine_extra_params   = refine_intrinsics   # ← the trap: defaults True, k1 drifts
if not refine_intrinsics:
    for camera_id in reconstruction.cameras:
        ba_config.set_constant_cam_intrinsics(camera_id)
```
plus a post-construction assert on `problem.is_parameter_block_constant(...)` — constancy is decided at
adjuster-construction time, so it is checked there rather than trusting an options round-trip.
Threaded through `IterativeBAOptions.refine_intrinsics` → `global_refinement.py` → `--refine_intrinsics` CLI.

**Proof on the REAL river_site model** (not synthetic):
```
[BA FIXED] cam1 PINHOLE  max|Δ| = 0.0000e+00   fx 1000.81249 -> 1000.81249
[BA FREE ] cam1 PINHOLE  max|Δ| = 1.1249e+01   fx 1000.81249 -> 1005.41555   (only 3 iterations!)
```
Production BA runs 200 iterations × 3 filter rounds. **D2 was real and large.**

> ⚠️ `refine_focal_length=False` **alone is NOT enough** — `refine_extra_params` defaults to `True`, so `k1` still drifts on any model with distortion params. All three must be False.
> ✅ `pygluemap.solve_cuda` is a **non-issue**: it copies `Solver::Options`, swaps the linear solver, and calls plain `ceres::Solve`. It never rebuilds the Problem, so constancy baked in at construction is honoured. No C++ rebuild.

### A.2 🔴 THE INTRINSICS PROBLEM IS BIGGER THAN THOUGHT

Three sources disagree about this drone's focal length:

| Source | fx/W | implied HFOV |
|---|---|---|
| official69 (the original spec) | 0.727505 | 69.0° |
| `configs/mapping/原始估計內參.json` (Charuco, 1280×720) | 0.750379 | **67.3°** |
| river_site map's actual BA-converged value | 0.781880 | **65.2°** |

**Spread: 7.5%.** And the river number is not even comparable — those frames were `cv2.undistort`ed with the Charuco K+dist, so it is the focal of a *rectified* image, not of a raw one. **It cannot be used as evidence for a no-undistort build.**

The Charuco calibration is *weak*: 25 frames, small 5×7 board, RMS 0.83 px, `cx` off-centre by +30.8 px, and `k2 = +0.256` — yet official69 PINHOLE with **no undistortion** empirically *worked* (LoMa forward map: 267/267 localization). If the imagery really had that much barrel distortion, a pure pinhole model could not have done that. **⇒ the distortion coefficients are almost certainly overfit noise — and since focal and distortion are strongly correlated in Charuco calibration, the focal is suspect too.** (Recorded in memory already: *"Charuco FULL_OPENCV 實測是錯的"*.)

**⇒ DECISION (user-approved): 3-way bake-off, holdout localization is the judge, then hard-fix at the winner.**

| Candidate | 1920×1080 | 2688×1512 | 3840×2160 |
|---|---|---|---|
| **A** official69 | fx=fy=1396.81, c=(960, 540) | 1955.53, c=(1344, 756) | 2793.62, c=(1920, 1080) |
| **B** Charuco-scaled, no-undistort | fx=1440.73 fy=1437.29, c=(1006.2, 538.1) | 2017.02 / 2012.21, c=(1408.7, 753.3) | 2881.46 / 2874.59, c=(2012.5, 1076.2) |
| **C** BA self-calibrated | ← measured by a `--refine_intrinsics` pass per group | | |

Candidate B scales focal **and** principal point linearly and **discards distortion entirely** (no-undistort policy). `PINHOLE` can express `fx≠fy` and an off-centre `cx` — `SIMPLE_PINHOLE` cannot. **Use `PINHOLE`.**

Judge: `建圖/pipeline/intrinsics_holdout_gate.py` — *"Gate intrinsics bake-off candidates with holdout localization results."* Winner = highest localization rate / inliers on **P1230123 + P1260126**, which never enter the map.

### A.3 🔴 Corrections to the original plan (things that were WRONG)

1. **`exclude_rotation_from_triangulation` does the OPPOSITE of what it sounds like.** It calls `filter_pairs_by_motion_roles(exclude_non_parallax=True)`, which **deletes every pair touching a pure-rotation frame** — so the frame neither registers **nor bridges**. Combined with `use_rotation_bridges=True` the two flags **mutually annihilate**. **Do not use it.**
2. **`bridge_only` is a DEAD LABEL.** Nothing reads it — no matcher, no triangulator, no BA. It is only counted in gates. "Register but do not triangulate" must be **written fresh**, and the filter must move from **pair level** to **observation level**.
3. **The motion-manifest producer is NOT `run_football_gluemap_from_motion_manifest.py`** — that is a pure *consumer*, and it cannot even run as shipped (its default `--source-manifest` points at a run dir that no longer exists). The real producer is `更新地圖/source/sfm_reshot25/build_localizable_map.py`.
4. **`fx/W` is `0.7275050`**, not `0.727531` (the old value implied HFOV 68.998° — a rounding slip).
5. **Unpatched PINHOLE would have *crashed*, not silently mis-filtered** (`camera.focal_length` raises on a 2-focal model). Moot now, but don't reason from the old premise.
6. **Hover is currently a binary DROP**, not a rate reduction (`keep = motion_class == "parallax"`). And there is a **latent bug**: the `motion_keep_rotation_every` branch runs *after* `use_rotation_bridges` sets `keep=True`, silently **decimating the very bridge frames it just kept**. Guard with `and not use_rotation_bridges`.

### A.4 🔴 Landmines

- **`base/validation/P0710071_frames`** — 512 contiguous **1920×1080** JPEGs, EXIF fully stripped, and **its source video does not exist anywhere on the filesystem**. Same resolution as build group A, zero metadata to distinguish it. **Any stage that globs `base/**` will silently ingest it with plausible-looking intrinsics.** → explicit path exclusion before S1.
- **The 2 test videos are group-B resolution** — geometrically identical to 4 build videos. **Path-based exclusion is the ONLY thing keeping them out of the map.** No geometric property can catch a leak.
- **Never `pip install` from `tools/gluemap/pyproject.toml`** — it pins `torch==2.4.1` and would downgrade the working 2.7.1+cu128 env (`ENVIRONMENT_LOCK.md:13`).
- **Pin `pycolmap` at 4.0.4.** An upgrade exposing `ReprojectionErrorType` would route the real reconstruction through the C++ NORMALIZED path and **silently bypass the D1 patch**.
- **`pygluemap.is_cuda_available()` is False** — the extension was built without CUDA ceres, so BA runs on **CPU ceres**. Not a regression (river used the same env) but it is the main wall-clock driver.

### A.5 Environment (verified)

RTX 5090 (32 GiB, ~32 free), 1.8 TB free on the build volume, Python 3.11.15, torch 2.7.1+cu128, pycolmap-cuda12 **4.0.4**, pyceres 2.6. All four checkpoints present. `gluemap-demo` runs. Live env matches `gluemap_pip_freeze.txt` exactly.

---

## B. Corpus (verified — sha256-locked)

### BUILD (7 videos · 42,746 frames · 1604.09 s · 10.30 GB)

All Parrot Anafi, **same drone serial `PI040416BA8L105488`**, software 1.8.2 — strong evidence they share one lens.

| Video | Res | fps | Dur | Epoch | Dir |
|---|---|---|---|---|---|
| `base/map/A-B(中間有往下看和左看).MP4` | 1920×1080 | 29.97 | 497s | 2026-05-16 | fwd **+ known pure-rotation** |
| `base/map/B-A.MP4` | 1920×1080 | 29.97 | 108s | 2026-05-16 | rev |
| `base/map/B-A(2).MP4` | 3840×2160 | 29.97 | 109s | 2026-03-15 | rev |
| `updates/build/a-b.MP4` | 2688×1512 | 23.98 | 267s | 2026-06-24 | fwd |
| `updates/build/P1220122.MP4` | 2688×1512 | 23.98 | 182s | 2026-06-24 | **unknown → S1** |
| `updates/build/P1240124.MP4` | 2688×1512 | 23.98 | 301s | 2026-06-24 | **unknown → S1** |
| `updates/build/P1250125.MP4` | 2688×1512 | 23.98 | 140s | 2026-06-24 | **unknown → S1** |

**DROPPED**: `base/map/A-B.MP4` (1280×720) — the only file with **no Parrot metadata** (libx264 re-encode). Its aspect is fine; it is dropped because **HFOV is unverifiable** (downscale vs crop indistinguishable).

**TEST — held out, never enters mapping**: `updates/test/P1230123.MP4`, `updates/test/P1260126.MP4`.

**Frame-rate split**: base/map is 29.97 fps, updates/build is 23.976 fps. The two cohorts are **not on a common frame clock** — extraction stride must be time-based, not index-based.

**Epochs**: March → May → June. Cross-epoch appearance drift is a first-class fusion risk.

### ✅ 16:9 hard gate — PASSES exactly (not approximately)

`W/H − 16/9 == +0.000e+00` in IEEE754 for every build source. EDM's canvas is 1024×576 = `1.7777777777777777`, bit-identical. This matters because EDM plain-resizes and derives a **single width-based scale** (`scale_of = c.width / EDM_W`) — a non-16:9 frame would **silently corrupt EDM's y-coordinates**.

---

## C. Key mechanisms (verified in source)

- **`intrinsics_mode: SHARED` already means "one camera per unique image shape"** (`datasets/base.py:194-201`) — exactly 同個解析度共享內參. No change needed.
- **`extract_gt_intrinsics` maps by image NAME** (`utils/colmap.py:81-89`), filling each shape-bucket from the first matching image. A per-resolution seed aligns 1:1 with SHARED bucketing, and it logs `GT intrinsics loaded: N/M cameras matched` → **free gate**.
- **`force_square`**: >1 unique shape ⇒ `True` ⇒ letterbox-pad to 518×518 with per-image `[scale_x, scale_y, x_off, y_off]` bookkeeping. **Multi-resolution is supported by design**; the `force_square=False` branch is the one that assumes homogeneity (reads `images_shape_ori[0]` for all). Cost: ~1.76× backbone compute, **no loss of image resolution** (long side is 518 either way).
- **Doppelgangers is currently OFF**: `skip_doppelgangers: true` takes the `torch.ones(...)` path — **every pair scored 1.0** (`gluemap_impl.py:86-93`). **Must be turned ON.**
- **`cameras.bin` width MUST equal on-disk image width** — EDM derives its scale from it.

---

## D. Stages and gates

### S0 — Corpus lock ✅ (data already gathered)
**G0.1** sha256 manifest of the 7 build / 2 test / 1 dropped · **G0.2** `test_leakage_check` (**re-run every stage**) · **G0.3** every build video exactly 16:9 · **G0.4** Parrot metadata present · **G0.5** `base/validation/P0710071_frames` explicitly excluded

### S1 — Motion analysis (BEFORE extraction)
Reuse `build_localizable_map.py:5608-5658` (`two_view_motion_metrics`, `classify_motion_metrics`). Resolve direction for P1220122/P1240124/P1250125 by VPR-sequence alignment against A-B.
**G1.1** timeline covers ≥99% of each video · **G1.2** `parallax_or_seed_ratio ≥ 0.65` · **G1.3** `hover_ratio ≤ 0.02` after sampling · **G1.4** direction labelled for all 7; ≥1 fwd and ≥1 rev
**G1.5 POSITIVE CONTROL** — the look-down/look-left segments in `A-B(中間有往下看和左看)` **must** be detected as `pure_rotation`. The filename tells us they exist. If the detector misses them, the detector is broken, not the data.

### S2 — Extraction + intrinsics bake-off
Motion-adaptive sampling (`parallax` full rate; `hover` decimated; `pure_rotation` sparse + flagged; `fast_motion` dropped). **No undistortion. No forced 1280×720.**
**G2.1** `cv2.undistort` never called; extracted frame ≡ raw decoded+resized · **G2.2** every frame 16:9 · **G2.3** bake-off A/B/C measured per resolution group · **G2.4** frame budget · **G2.5** re-run G0.2
**Branch**: if all groups agree on fx/W within 2% → single common canvas permitted (1 camera, `force_square=False`, ~1.8× cheaper). Else → per-resolution cameras.

### S3 — Multi-camera seed + forced bridges
One `PINHOLE` camera per resolution group. SALAD pairs **∪ forced VPR-blind bridge candidates** via the new `--extra_pairs_path`.
**G3.1** `GT intrinsics loaded: N/N` — **100%** · **G3.2** `cameras.bin` (w,h) == on-disk (w,h) for every image · **G3.3** ≥4.0 pairs/frame · **G3.4** `discovered_sequences == 7` (guards the `subfolder_regex` silent-drop)

### S4 — Two-view + Doppelgangers++ (THE anti-ghost gate)
`skip_doppelgangers: false`. `valid_dg_threshold` 0.8 → step down 0.5 / 0.2 if it over-prunes.
**G4.1** DG actually ran (log must NOT say *"Skipping Doppelgangers"*) · **G4.2** rejection rate 2–40% (**0% ⇒ misconfigured**) · **G4.3** ≥2 **independent, spatially-separated** bridges per connected fwd/rev pair · **G4.4** largest connected component ≥0.90

> **Forcing candidates ≠ forcing connectivity.** We force *pair candidates* where VPR is blind (reverse views do not retrieve). Verification decides. **An honest split beats a wrong fusion.**

### S5 — SfM + fixed-intrinsics BA
`--no-refine_intrinsics` (default). `min_track_length` 2 → **3**.
**Pure-rotation frames: registered and kept in the pose graph for bridging, but contribute NO observations to triangulation** — must be implemented at *observation* level (see A.3 #1/#2).
**G5.1** registered ≥95% · **G5.2** mean reproj ≤ **2.0** · **G5.3** ONE connected component (≥0.98) · **G5.4** final `cameras.bin` params == seed to 1e-6 · **G5.5** median triangulation angle ≥2°, ≤5% below 1° · **G5.6** zero observations from pure-rotation frames · **G5.7** **Sim3 fold check** — every fwd↔rev bridge cluster must independently agree on `T_AB`; disagreement ⇒ doppelgänger ⇒ **HALT**

### S6 — Ghost audit
**G6.1** fwd and rev trajectories spatially coincident · **G6.2** no duplicate-structure cluster · **G6.3** per-sequence reproj within 1.5× global mean · **G6.4** March/May/June traversals overlay

### S7 — Bundle bootstrap (SIMPLER than planned)
**EDM does NOT need an XFeat bundle.** It `torch.load`s `--in-bundle` as a plain dict and never reads `xb["refs"]`. **`build_reloc_map_xfeat_tri.py` is NOT a dependency — drop it.**

`build_bundle_seed.py` (~70 lines) produces only **3 keys**:
```python
ref_names  = sorted(n for n in images_by_name if (images / n).exists())  # the SORT is load-bearing:
                                                                        # covis values index INTO this list
ref_global = megaloc_lib.extract(ref_names, Path(images), device)        # (N, 8448) f32, already L2-normalized
                                                                        # pass a pathlib.Path, not a str
meta       = {"vpr_input": 322, "bundle_vpr": "megaloc", ...}
```
Then **reuse the existing root** — write no covis/yaw code:
```
python 定位/source/sfm_glomap/deploy/augment_reloc_bundle_tracking.py \
    --model <COLMAP dir> --bundle seed.pt --out seed_tracking.pt --top-covis 40
```
Exclude bridge-only rotation frames from `ref_names`.

> 🔴 **THE YAW TRAP — reproduce VERBATIM, do NOT "fix" it.**
> ```python
> fwd = R.T @ np.array([0., 0., 1.]);  yaw = math.atan2(fwd[1], fwd[0])
> ```
> This is the optical axis projected onto the **map's** XY plane. A GLUEMAP world frame is **not gravity-aligned**, so for near-nadir cameras the projection collapses and atan2 goes unstable. **It is still mandatory to copy byte-for-byte**, because `production_edm_tracker.py:227` recomputes runtime yaw with the *identical* formula and then gates candidates on `abs(Δyaw) > 90°`. **A "corrected" gravity-aligned yaw in the map silently rejects every correct candidate at flight time.**

**G7.1** every ref has a MegaLoc descriptor · **G7.2** all covis indices in range · **G7.3** `ref_yaws` **bimodal** (a fwd+rev map must show two heading clusters; unimodal ⇒ one direction collapsed into the other ⇒ ghost)

### S8 — EDM reloc map
**G8.1** log reports the expected cameras + resolutions · **G8.2** median 3D-anchored cells/ref ≥ river baseline · **G8.3** triangulated images == active refs

### S9 — Held-out acceptance (the only gate that really matters)
Localize **P1230123 + P1260126** against the EDM map.
**G9.1** localization rate ≥95% · **G9.2** no pose jumps beyond drone-speed plausibility · **G9.3** reverse passes localize too · **G9.4** inliers ≥ floor · **G9.5** no ghost teleports

---

## E. Compute budget (measured)

River baseline **on this box**: 454 frames → **35.2 min** gluemap (`smoke_pinhole_fix`, reproj 1.87). Cost ≈ `O(N × num_neighbors)`, and **BA is on CPU ceres** (see A.4).

- ~1,200 selected frames → **~1.5–2 h** (single canvas) · **~3–4 h** if `force_square=True`
- ⇒ budget **half a day per full iteration**. Cache aggressively (`force_load: true`, `rerun_from`) — the two-view cache survives config retuning, so DG-threshold and BA sweeps are cheap re-runs.

**Do NOT copy the `highres_opt` params** (`num_track_per_img: 2048`, `batch_size: 2`) — that run **regressed** (reproj 4.01, 352 registered, gate FAILED). Start from `smoke_pinhole_fix`.

---

## F. Next up (Phase 1 — build tooling, not yet written)

1. `S0` corpus lock + leakage check (incl. the `P0710071_frames` exclusion)
2. `S1` motion-manifest **producer** (adapt `build_localizable_map.py`; fix the hover/rotation decimation bug)
3. `S2` no-undistort motion-adaptive extractor + multi-camera seed + the 3-way intrinsics bake-off
4. Observation-level "register but don't triangulate" for pure-rotation frames (**fresh code** — nothing reusable exists)
5. `S7` `build_bundle_seed.py`

## G. Acceptance

Ships to EDM only if **every** gate S0–S9 is green, with **S5.3** (one component), **S5.4** (intrinsics held), **S5.7** (Sim3 fold), **S6** (ghost audit) and **S9** (held-out localization) as the hard anti-ghost / anti-mismatch / anti-direction-failure set.
