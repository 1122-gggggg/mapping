# Build run log — sfm_reshot25 (autonomous)

Live journal of THIS build run. Monitored every ~20 min; auto-advances phases,
auto-fixes stalls/crashes, ends by delivering the localizable map + validation.

## Plan (auto-executed, gated)
- P2 MV-RoMa (running) → P2b aggregate (py312) → P3 DB(sfmdb)+GLOMAP(sfm) → connectivity GATE
  → P4 deploy artifacts (M / megaloc cache / cloud) → P5 localization validation.
- **Validation:** held-out test P1230123,P1260126 (blind) + **OLD data robustness**
  (localize old-field frames against new M — cross-time/lighting stress).
- **Deliverable:** `reloc_map_xfeat_tri.pt` (localizer map M) + megaloc cache + cloud + glomap/0.

## Auto-fix rules
- MV-RoMa stalled (done_sources unchanged ~20 min & proc dead): restart `02_match_mvroma_mixed.py`
  (resume via done_sources sentinel — skips finished sources).
- Any stage crash: read log, diagnose, fix minimally, retry; if blocked, STOP + report.
- GLOMAP splinter (not 1 component): inject targeted RoMa-bridge pairs across the split, re-match+map.

## Events
- 15:27 P2 MV-RoMa started (33258 pairs, ~0.4 s/pair, resume-safe).
- 16:17 P2 MV-RoMa 481/1923 (~5s/src, healthy, ~2h left). No action.
- 16:38 P2 MV-RoMa 744/1923 (~4.8s/src, healthy, ~1.6h left). No action.
- 16:59 P2 MV-RoMa 1093/1923 (healthy, ~1h left). No action.
- 17:20 P2 MV-RoMa 1356/1923 (healthy, ~45min left). No action.
- 17:41 P2 MV-RoMa 1673/1923 (healthy, h5 4.3GB, ~20min left). No action.
- 17:53 P2 MV-RoMa DONE (1923/1923, 32033 pairs, h5 ~4.3GB). -> P2b aggregate.
- 17:54 P2b aggregate running (32033/33258 pairs, 1225 skipped=non-overlap forced→rejected as designed), ~35min.
- 17:55 P2b CRASH: h5py segfault at agg 29% (corrupt pair group from interrupted matches0 append). FIX: repair_dense_h5_raw -> re-aggregate.
- 18:06 P2-repair running (failed=0, raw kpts intact) ~3min -> then re-run 02b aggregate on clean h5.
- 18:10 P2-repair OK foreground (clean h5, 32033 pairs). Re-launch 02b via nohup (run_in_background got killed; nohup survives).
- 18:32 P2b OOM-killed at 43% (rss 29GB/30GB; AGG_MAXKP=12000 too high for 1924 dense imgs; thrash→3s/it). FIX: AGG_MAXKP=6000. Re-repair + re-run 02b nohup.
- 18:36 P2b re-run OK: AGG_MAXKP=6000 → 17.6 it/s (no thrash), mem 5G/30G, ETA ~30min. Watch mem (lower to 4000 if it climbs near OOM).
- 18:58 P2b died @43% (15G free, NOT OOM → systemd-oomd PSI pressure from 6G swap churn by idle dashboard+blender). FIX: killed leftovers, AGG_MAXKP=4000, re-repair+re-run.
- 19:03 P2b re-run AGG_MAXKP=4000 (leftovers killed, swap freed): 18.7 it/s, mem 4G/16G free. Watch pair ~13858 (43%, prior death point).
- 19:17 Hermes fix: start P2b aggregate with AGG_PAIR_DEG_CAP=12 AGG_MAXKP=4000 (pair-degree cap prevents repeated OOM).
- 19:21 Hermes check: P2b managed aggregate is progressing with AGG_PAIR_DEG_CAP=12 AGG_MAXKP=4000; old Claude Code monitor/session killed to avoid duplicate launches; only managed build left running.
- 19:33 P2b aggregate DONE with pair_cap=12 AGG_MAXKP=4000.
- 19:33 P3 DB build started (sfmdb, taskset -c 0).
- 19:35 P3 DB build DONE: /media/cihcilab/新增磁碟區/sfm_reshot25/fair_mvroma/mvroma/database_mvroma_forced.db.
- 19:35 GLOMAP mapper started (skip BA, skip retriangulation, max_tracks=600000).
- 19:36 GLOMAP mapper DONE: /media/cihcilab/新增磁碟區/sfm_reshot25/glomap_fused_mvroma_noba_full/0.
- 19:36 Model summary written: /media/cihcilab/新增磁碟區/sfm_reshot25/glomap_fused_mvroma_noba_full/SUMMARY.json.
- 19:36 Hermes managed recovery DONE through GLOMAP; next P4/P5 artifacts/validation.
- 19:39 Hermes fix: start P2b aggregate balanced caps intra=12 cross=16 total=28 pair_cap=0 AGG_MAXKP=4000.
- 19:56 P2b aggregate DONE with cap_spec=12,16,28 AGG_MAXKP=4000.
- 19:56 P3 DB build started (sfmdb, taskset -c 0).
- 19:57 P3 DB build DONE: /media/cihcilab/新增磁碟區/sfm_reshot25/fair_mvroma/mvroma/database_mvroma_forced.db.
- 19:57 GLOMAP mapper started (skip BA, skip retriangulation, max_tracks=600000).
- 19:58 GLOMAP mapper DONE: /media/cihcilab/新增磁碟區/sfm_reshot25/glomap_fused_mvroma_noba_full/0.
- 19:58 Model summary written: /media/cihcilab/新增磁碟區/sfm_reshot25/glomap_fused_mvroma_noba_full/SUMMARY.json.
- 19:58 Hermes managed recovery DONE through GLOMAP; next P4/P5 artifacts/validation.
- 14:55 Hermes fix: start P2b aggregate balanced caps intra=10 cross=8 total=18 pair_cap=0 AGG_MAXKP=4000.
- 15:24 P2b aggregate DONE with cap_spec=10,8,18 AGG_MAXKP=4000.
- 15:24 P3 DB build started (sfmdb, taskset -c 0).
- 15:26 P3 DB build DONE: /media/cihcilab/新增磁碟區/sfm_reshot25/fair_mvroma/mvroma/database_mvroma_forced.db.
- 15:26 GLOMAP mapper started (skip BA, skip retriangulation, max_tracks=600000).
