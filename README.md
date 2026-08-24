# mapping — 建圖 + 診斷

這個 repo 是 `sfm_system` 的**建圖側**與**地圖／定位診斷**。
飛行時的即時定位（tracker / PCMD / EDM runtime）**不在這裡**，在另一個 repo。

界線是這樣劃的：**產生地圖與 bundle 的東西在這裡；消費 bundle 去飛的東西不在這裡；
讀完地圖之後做健康檢查與失敗歸因的東西也在這裡。**
所以 `finalize_edm_model.py`、`validate_heldout_localization.py` 都在，
`diagnosis/`（MapDoctor + sfm-diagnosis + `sfm-qa`）也在。

原始工作目錄是 `/media/cihcilab/新增磁碟區/sfm_system/`。這裡不含任何 `runs/` 產物
（那裡有 265 GB 的影像、database、model、log）。

```bash
pip install -e '.[dev]'
```

### 驗證矩陣

根目錄的輕量核心由 `.github/workflows/ci.yml` 驗證。各 site 的 `tools/` 保留歷史上的
同名裸模組（例如 `ts_common`、`audit_dg_graph`），而且部分測試需要場域資料、GPU 或
選用依賴；因此目前不能把整庫 `pytest` 單一程序視為有效契約。請依邊界分開執行：

```bash
python3 -m pytest diagnosis/tests tools/test_diagnose_map.py tools/test_diagnosis_layout.py
PYTHONPATH=map_update/core python3 -m pytest map_update/core/test_point_evidence_ledger.py map_update/core/test_point_evidence_recency.py
PYTHONPATH=map_update/lifelong/src python3 -m pytest map_update/lifelong/tests
PYTHONPATH=pipeline python3 -m pytest pipeline/test_build_localizable_map_core.py::PairGraphPerformanceTests

# 只有在該場域的選用依賴與外部 fixtures 都存在時執行。
PYTHONPATH=sites/<site> python3 -m pytest sites/<site>/tests
```

site 測試必須各自在新程序執行，避免 Python 模組快取把另一個 site 的同名工具誤當成
本場域實作。要支援真正的 root-wide suite，需先把三套 site tools 遷移成命名空間套件，
並為 `torch`、`pycolmap`、`lmdb`、`h5py` 與外部資料建立明確的 extras／skip 契約。

---

## 現行建圖入口

`sites/<site>/tools/` 的 **S0–S9** 是唯一正式建圖與 release 流程。舊的一鍵 wrapper、
其專屬 gate/config/test 與不可執行文件已移除，避免新場域誤走已失效的外部路徑。

`pipeline/run_fuhe_gluemap_build.py`、
`pipeline/run_football_gluemap_from_motion_manifest.py`、
`pipeline/repair_fuhe_gluemap_fixed_ba.py` 也都是實際跑過的活程式。

`pipeline/build_localizable_map_core.py` 是 symlink，指到
`map_update/build_localizable_map.py`；這條可攜式全域 SfM 路徑預設用 COLMAP 4.x `global_mapper`，並在聚合前執行 H/E 平面一致性與 view-component pruning。

---

## 現行方法：S0–S9

每個 site 一份 `tools/`。三份高度重疊（`ts_common.py` / `ts_env.py` / `ts_intrinsics.py`
是共用骨架的三份副本），差異在各場域的實際限制。**target_site 是最完整、最新的那份**。

| Stage | 工具 | 做什麼 | 硬 gate |
|---|---|---|---|
| **S0** | `s0_corpus_lock.py` | 鎖定影片語料，並在動任何東西之前證明 build/test split | held-out 影片不得洩漏進 map |
| **S1** | `s1_motion_scan.py` | **抽幀之前**先分類運動、判定飛行方向 | pure rotation / hover 不得主導 |
| **S1b** | `s1b_bridge_feasibility.py` | 到底需不需要 forced cross-video pairs、需要在哪 | 見下方「MegaLoc 對反向是盲的」 |
| **S2** | `s2_extract.py` | motion-adaptive 抽幀。**不做 undistortion、不做強制 resize** | 每段抽出的幀數與視差比例 |
| **S2b** | `s2b_intrinsics_bakeoff.py` | crash-safe 雙 seed 內參可辨識性烘培 | 多解析度 × 多相機模型獨立 replay |
| **S3** | `s3_pairs.py` | forced VPR-blind bridge candidates + real-loader 契約 gate | 跨影片 bridge pairs > 0 |
| **S4** | `audit_dg_graph.py` + Doppelgangers++ | **反鬼影主閘**。拒絕重複結構造成的假橋接 | 最大 component ratio、不得清光跨向邊 |
| **S5** | COLMAP 4.x `global_mapper` → `finalize_edm_model.py` | 維護中的 GLOMAP 全域建圖 → 移除 pure-rotation observations → **固定內參 BA** | focal/principal/extra 全部固定，並與 seed 逐參數比對 |
| **S5.7** | `audit_independent_sim3.py` + COLMAP GlobalMapper | 各 sequence 獨立建圖，再獨立驗證跨方向 Sim3 橋接 | 不靠同一次建圖自我背書 |
| **S6** | `audit_map_geometry.py` | 鬼影稽核 | G6.1–G6.4 |
| **S7** | `build_bundle_seed.py` → `validate_tracking_bundle.py` | MegaLoc seed bundle → tracking bundle | `ref_global.shape == [refs, 8448]` |
| **S8** | `finalize_edm_model.py` → `validate_edm_bundle.py` | EDM detector-free 固定姿態重三角化 → bundle | cell-anchor round-trip |
| **S9** | `validate_heldout_localization.py` | **唯一真正重要的 gate**：未參與建圖的影片 | target_site 要 ≥95% |
| — | `build_gravity_alignment.py` | 從 model 推 `T_align_gravity.json`（相機 x 軸 ⊥ 重力） | G-GRAV-1a/1b/2/3/4 |
| — | `verify_gauge_invariance.py` | 更新前後逐元素比對舊 pose / 舊 point3D | G-U1a/1b/1c、G-U2a/2b |
| — | `recompute_site_scale.py` | 重算 `S` 與五個尺度參數 | G-U3 |
| — | `verify_final_release.py` | S0–S9 全綠才發 release | 缺一不發 |

### 為什麼是這個順序

- **S1 在 S2 之前**是刻意的。先分類運動再抽幀，才不會把 hover 重複幀和 pure rotation
  送進昂貴前端。`s1_motion_scan.py` 同時判定飛行方向，S1b 才有辦法問「需不需要強制橋接」。
- **S2 不做 undistortion**。內參政策是「固定、不讓下游估」，任何 resize / undistort 決策
  都必須明確記錄在 manifest 裡。
- **S2b 存在的原因**：ANAFI 到底該用 PINHOLE（韌體已去畸變）還是 SIMPLE_RADIAL，
  規格書講不清楚。三個焦距來源互差 7.5%，所以用實測 bake-off 定案，不用猜的。
- **S5 只跑維護中的 COLMAP GlobalMapper。** 它直接重用現有 COLMAP database 的隔離副本；
  最終模型仍須通過固定內參、S6 與 S9。
- **S9 才是驗收**。「有輸出檔」不等於完成。

### 論文方法整合

- **G-MASt3R-SfM**：移植 geometrically verified view pruning；dense pair filter 只保留最大的 verified component，component ratio 不足直接 fail。沒有再加 MSO/PGO，因為下游 global positioning + BA 已涵蓋較完整的目標。
- **Planar-SfM**：homography-dominant edge 不再一律當 pure rotation；只有 homography decomposition 與 essential rotation 的 geodesic agreement 通過才救回。
- **LFOE-GlobalSfM**：論文方法仍可做隔離 A/B，但官方 repository 沒有 license grant，不能作 production default；由 BSD 授權且持續維護的 COLMAP 4.x `global_mapper` 取代。
- **其他候選**：DFSfM／PixSfM 保留為固定 pose/intrinsics 的隔離候選；Dense-SfM 缺 license 且未釋出論文的 GS track extension；InstantSfM 為 CC BY-NC；Planar-SfM、G-MASt3R-SfM、DATAP-SfM 尚無可直接替換的完整官方模組。

## 診斷：Stage 0–2

`diagnosis/` 是原本獨立的 [sfm_map_diagnosis](https://github.com/1122-gggggg/sfm_map_diagnosis)。
它不依賴 `sites/<site>`、場地 ID、固定原點／邊界或特定定位器。建圖前、建圖中、建圖後
都透過通用資料契約工作：session/image evidence、`MapModel`、以及 localization result。
現有 `sites/` 是另外一組歷史建圖／release 工具，不是診斷核心的依賴。

| Stage | 命令 | 通用輸入 | 硬 gate？ |
|---|---|---|---|
| **0：建圖前** | `sfm-qa select-sessions` | sessions、images、選用的既有地圖證據 | 否，advisory |
| **1：建圖中／後** | `sfm-qa analyze MAP --map-adapter ADAPTER` | 任意 adapter 正規化出的 `MapModel` | 地圖完整性是硬檢查 |
| **2：定位後** | 同上再加 `--logs loc.csv` | 任意定位器輸出的 `query,success` 與選用指標 | 由部署設定決定 |

Stage 0 預設採 cohort-relative portfolio：品質門檻只作風險參考，整批影片都偏弱時仍會
輸出現有資料中的最佳非空 geometry-probe 組合，並明示 `relative_fallback_used`；VPR
仍不能當幾何邊。Stage 2 會輸出 query-relative risk–coverage；對由部署方 immutable
manifest 證明為 held-out 的 log，以 strict success rate（預設 95%）判定。品質欄位只在
存在時啟用 gate；部署方可用 `localization.required_metrics` 強制指定必填證據。完整論文與設計依據見
`docs/RELATIVE_QUALITY_DIAGNOSTIC_DESIGN_20260823.md`；逐項區分規格修正、理論移植、
場域政策與 experiment-only 方法的後續稽核見 `docs/THEORY_EVIDENCE_AUDIT_20260823.md`。

內建 map adapter 有 `colmap`、`glomap`、`gluemap`；其他格式可直接傳
`--map-adapter package.module:AdapterClass`，不用改診斷核心。定位結果的 `localizer` 是任意
provenance label，不會切換 HLOC、EDM、SCR 或任何方法專屬邏輯。

弱區域與失敗 query 會輸出 existing-data-first 解方、counterfactual 與 conditional
recapture 驗收契約。完整 S0–S9 缺口與優先序見
`docs/S0_S9_LOCALIZATION_QUALITY_ROADMAP_20260823.md`。

```bash
# 建圖中或建圖後：任何可轉成 MapModel 的地圖
sfm-qa analyze /path/to/map \
  --map-adapter package.module:AdapterClass \
  --output /path/to/diagnosis

# 定位後：任何定位方法，最小結果契約只有 query,success
sfm-qa analyze /path/to/map \
  --map-adapter package.module:AdapterClass \
  --logs /path/to/localization-results.csv \
  --output /path/to/diagnosis

# 風險球：消費 heatmap / weak-region / logs，不再算第二套 FIM
sfm-diagnosis risk-ply /path/to/map --map-adapter colmap --output /path/to/risk --logs loc.jsonl
```
細節見 `docs/MAPDOCTOR_QA_INTEGRATION.md` 與 `diagnosis/docs/pipeline.md`。

---

## 踩過的坑（這些是這套流程存在的理由）

**MegaLoc 對「同一條航線的正反向」是可測量地盲目的。**
正↔反的 retrieval similarity 只有 0.10–0.17，同向是 0.31–0.59。
所以 S1b/S3 的 forced VPR-blind bridging 不是保險，是**承重結構**。
target_site 靠 446 條 accepted forced edges 把三組雙區域骨幹接起來。

**「純旋轉」段落常常不是退化的。** 雲台在動的時候仍有 0.8–4.3° 視差。
不要用「看起來像轉彎」就丟掉整段——用 `s1_motion_scan.py` 的 H/F inlier ratio 判。

**GlueMap BA 會默默漂移內參。** 見上面 S5。

**GlueMap 沒有原生 append-existing-map 契約。** 這是下面「更新地圖」那節的核心問題。

**pure-rotation 影像可以保留 pose，但最終 3D map 必須是零 observation。**

---

## 更新地圖（新拍資料進來怎麼辦）

`map_update/` 是這條線的全部程式碼與研究紀錄。

`map_update/lifelong/` 已整合原 `1122-gggggg/update_map` 的完整 Git 歷史，提供
change-aware historical-view augmentation、immutable base-map snapshot、長期 feature
memory、candidate bundle promotion／rollback 與 `update-map` CLI。它使用和診斷層相同的
開放邊界：`adapters.map_loader` 可是內建名稱或 `package.module:loader`，定位方法名稱只作
provenance，retriever／matcher 透過 precomputed、Python callable 或 external-command adapter
接入，不選擇方法專屬邏輯。

```bash
update-map synthetic-demo --output /tmp/update-map-demo
update-map inspect-map /path/to/map --map-adapter package.module:loader
```

- `docs/map_update/EDM_INCREMENTAL_UPDATE_DESIGN.md` — **從這裡開始讀**。EDM 時代的具體
  增量更新設計：以 gauge 是否被動過分流、U0–U5 流程、G-U1~G-U6 gates、P0–P8 實作順序。
- `map_update/MAP_UPDATE_STRATEGY_RECORD.md` — 長期架構論證 (v2, 2026-07-15)。
  7 篇論文的可移植 / 不可照搬對照、point-level evidence ledger 設計、release gates。
  上面那份設計不取代它，是把它落到 EDM 的具體形狀。
- `docs/map_update/UPDATE_PIPELINE_METHODS.md` — 目前**已實作**的路由邏輯與參數。
- `map_update/core/` — 實際程式：`map_update_tool.py`（60 KB，主工具）、
  `prepare_update_frames.py`（geometry/connector 分流）、`sparsify_reloc_bundle.py`、
  `stability_scores.py`、`changed_region_evidence.py`、`update_quality_gates.py`。

### 四條路由

```
新資料進來 → 對舊地圖定位 → 看 register_rate
  ├─ >0.95 且無場景變更  → 1. 不更新地圖，留作 validation/QA，只記 observation
  ├─ 高重疊              → 2. 增量註冊 + 局部三角化（PnP 舊 3D 點，固定舊 pose）
  ├─ 部分重疊、新區域多  → 3. 獨立 submap + bridge frames 算 Sim3 併入
  └─ 有重疊但幾何/語義不符 → 4. 局部 tile 替換（invalidate 舊點，邊界當 anchor）
```

`register_rate` 是 overlap 的 proxy，**不是** 場景變更偵測器。
一段影片可以 register_rate 很高、同時某面牆已經變了。

### ✅ 已遷移到 EDM（對舊圖匹配的部分）

定位主線已經定案為 **EDM**（detector-free + cell-anchor LUT + MegaLoc 檢索）。
但 `map_update/core/map_update_tool.py` 從頭到尾用 **XFeat + LighterGlue**：

- 對舊地圖 PnP（判 `register_rate`、找 bridge frames）→ `extract_xfeat` + `match_lighterglue`
- 建新 submap → XFeat + LighterGlue → hloc 三角化
- 輸出 bundle → `reloc_map_xfeat_tri.pt`

**已改**：`map_update/core/update_matcher.py` 把匹配前端抽成後端介面，
`--matcher edm` 為預設。對舊圖定位／路由判定／register 路徑與 bundle keyframe
全部走 EDM；`--matcher xfeat` 保留只為 A/B。

**尚未遷移**：route 3 的 submap 重建仍是 XFeat + hloc。在 `--matcher edm` 下會
**直接擋下**並要求改走 register/skip，或明示 `--allow-xfeat-submap` 接受混合前端
候選（必須留在 quarantine）。不會靜默混用。

（本 repo 已刪掉純 XFeat 定位驗證器 `eval_stream.py` 與 XFeat bundle 匯出器
`export_track_landmarks.py`；`map_update_tool.py` 裡的 XFeat 沒有刪，因為它是**更新時的
匹配前端**，不是飛行定位器——刪掉等於刪掉整條更新線。）

### ⚠️ 目前的實作落差（別誤以為這套已經完成）

| 路由 | 宣稱 | 實際 |
|---|---|---|
| 2. register | 增量註冊 + **局部三角化** | 只做 PnP + 加定位 keyframe。**不產生新 3D 點，不做 joint BA** |
| 3. submap | submap + Sim3 合併 | 有實作（XFeat+LighterGlue 建 submap → Umeyama Sim3），但**合併後沒有 joint BA** |
| 4. tile replace | 自動 tile 重建 | **沒有**。只產生偵測證據（`observation_stats.json` 的 changed-region candidates） |
| stability | ExMaps | 是 **reference-level** 衰減，不是 ExMaps 的 **3D point-level** visibility/recency |

而且 `map_update/update_pipeline.py` 預設的 base model / bundle / MegaLoc cache 路徑
指向已經不存在的 `sfm_glomap/`。這是 `MAP_UPDATE_STRATEGY_RECORD.md` 裡的 **P0**：
在恢復 map snapshot contract 之前，任何更新都不該標成 production。

### 更新完之後，定位端一定要拿到的三樣東西

這是**建圖 → 定位的交付契約**。每次更新地圖都必須重新產生（或明確證明不必重產）：

| # | 交付物 | 為什麼不能沿用 |
|---|---|---|
| 1 | **EDM bundle** | detector-free。2D→3D 層是用「該次 COLMAP 位姿」把 cell-anchor 打到 3D 的，位姿一動整層就失效 |
| 2 | **五個尺度參數 + 到達容差** | 全是 map units。新地圖的 `S` 不同就得重算 |
| 3 | **重力對齊 `T_align_gravity.json`** | 新重建 = 新 gauge，重力在新座標裡是別的方向 |

第 3 樣由 `pipeline/build_gravity_alignment.py` 產生（見下）。已產好的兩份在
`reference/gravity/`。

尺度參數的定義（`EDM定位測試/build/make_transfer_package.py` 的 `derive_site_profile()`）：

```
S = 2 · p95( ‖ center_i − componentwise_median(all centers) ‖ )      # robust_camera_span()

radius                    = 0.16   · S
max_jump                  = 0.40   · S
adaptive_jump_floor       = 0.0006 · S
adaptive_jump_bootstrap   = 0.004  · S
adaptive_jump_ceiling     = 0.0016 · S
```

比例是**無因次、全場域共用**；`S` 是每張地圖各自算的。已知 `S`：
target_site `5.0065`、football_field `1.8328`、fuhe_bridge `2.0271`。
`EDMConfig` 的預設值（0.8 / 2.0 / 0.003 / 0.02 / 0.008）正好是這條規則在 target_site
尺度的取值，**不是通用常數**——沒有自己 profile 的場域會默默沿用它們。

#### 這三樣東西的失效條件不一樣，別一律當成「要全部重來」

| 更新方式 | gauge | 相機集合 | EDM bundle | 5 參數 | T_align_gravity |
|---|---|---|---|---|---|
| 方法 2：固定舊 pose 增量註冊 + 局部三角化 | 不變 | **變大** | 只需補新 keyframe | **必須重算** | ✅ 不變 |
| 方法 3：submap + Sim3 併入（含 joint BA） | **變** | 變大 | 全部重建 | 全部重算 | **必須重求** |
| 方法 4：局部 tile 替換（邊界 anchor + BA） | 邊界會動 | 可能不變 | 全部重建 | 重算 | 視 BA 是否動到全域 |
| 週期性 `global_mapper` 全域重解 | **全變** | 變 | 全部重建 | 全部重算 | **必須重求** |

兩個非直覺的點：

- **`S` 對「純加點」也敏感。** 它是 `p95` 掃過**全部**相機中心算的，所以就算 gauge
  一動也沒動，只要往新區域加了 keyframe，`p95` 就位移、五個參數全部跟著變。
  「舊 pose 沒變 ⇒ 尺度參數不用改」是錯的。
- **反過來，`T_align_gravity` 對「純加點」是免疫的。** 只要舊相機的 pose 逐值不變，
  座標系就沒轉，重力方向仍指同一邊。

所以方法 2 值錢的地方不是省時間，是**它保住了 gauge**——三樣交付物只有一樣要真的重建。

#### 要靠這點，就得證明舊 pose 真的沒動

`colmap mapper --Mapper.fix_existing_frames 1` 只是「請求」，不是證據。
建圖側已經有這個模式可以照抄：`finalize_edm_model.py` 對內參做逐參數比對
（容差 `1e-6`，target_site 實測 delta `0.0`）。**對舊相機 pose 做一模一樣的 gate**：

```
G-UPDATE-1  舊 image 的 R,t 對更新前後逐值比對，max delta ≤ 1e-9   → 通過才可宣稱 gauge 不變
G-UPDATE-2  舊 point3D 的 xyz 逐值比對（未被 tile 替換的部分）
G-UPDATE-3  重算 S，寫進 update report；即使 gauge 不變也要重發 site profile
G-UPDATE-4  T_align_gravity：G-UPDATE-1 通過才可沿用，否則標記為 stale 並擋下交付
```

沒有 G-UPDATE-1 就別宣稱 gauge 不變——BA 只要碰到一點舊 pose，重力對齊就悄悄歪掉，
而這種歪法在定位成功率上**看不出來**（成功率照樣 99%，只是飛機以為的「上」不是上）。

### 往 EDM 遷移的狀態

| # | 項目 | 狀態 |
|---|---|---|
| 1 | 匹配前端 `extract_xfeat`/`match_lighterglue` → EDM cell-anchor | ✅ `update_matcher.py` |
| 2 | bundle keyframe → `xyz_by_cell` + `image_jpg`（EDMRelocMap schema） | ✅ |
| 3 | route 3 submap 重建 | ⛔ 未遷移，`--matcher edm` 下會擋下 |
| 4 | 交付 site profile + `T_align_gravity.json` | ✅ 工具已有（`recompute_site_scale.py` / `build_gravity_alignment.py`），尚未接進更新流程收尾 |
| 5 | 檢索 MegaLoc（8448 維 / 322×322） | ✅ 不變，整層沿用 |

**第 1 項不是純換 API**：XFeat 給的是可跨 pair 重用的 per-image keypoints，
EDM 是 pair-specific detector-free。`round(kpt/8)` cell 就是為了在 detector-free 上
造出穩定 keypoint 身分才存在的，dedup 就靠它。

一個實作陷阱：`correspondences_by_ref` 回傳的是**相機像素**（1280 寬），
不是 EDM canvas（1024 寬）。量化成 cell 之前要先除以 `scale = 1.25`，
否則會索引到錯的 `xyz_by_cell` 槽。已有回歸測試。

**不要在沒過 round-trip / resize / multi-resolution gate 之前改 input canvas。**

第 1 項不是純換 API：XFeat 給的是可跨 pair 重用的 per-image keypoints/descriptors，
EDM 是 pair-specific 的 detector-free 匹配。cell-anchor 就是為了在 detector-free 上
造出穩定的 keypoint identity 才存在的——`round(kpt/8)` 精確可逆。
**不要在沒過 round-trip / resize / multi-resolution gate 之前改 input canvas。**

### 重送原始建圖資料時

正確結果應該接近 **no-op**，不是把它當新的長期觀測。content hash 相同的影像要對應既有
image identity，不新增 ID、不更新 `last_seen_session`、不進 held-out set。
否則同一批舊資料會讓過期點被永久「續命」，而且驗收指標會有 test leakage。

---

## 沒有 vendor 進來的外部工具

`建圖/external_tools/` 有 7.8 GB 的第三方 repo，這裡**不含**。要重現請自己 clone
（下面是原機器上的 pinned commit）：

| 工具 | Upstream | Commit |
|---|---|---|
| MV-RoMa | https://github.com/IceTea-CV/MV-RoMa | `acb09ef` |
| Doppelgangers++ | https://github.com/doppelgangers25/doppelgangers-plusplus | `f58d86a` |
| LFOE-GlobalSfM | https://github.com/DmblnNicole/LFOE-GlobalSfM | `a80c845` |

| DetectorFreeSfM | https://github.com/zju3dv/DetectorFreeSfM | `4a370f1` |
| GGPT | https://github.com/ChenYutongTHU/GGPT | `2f39fcf` |
| SLiM | https://github.com/Band-127/SLiM | `8b34762` |
| SplatHLoc | https://github.com/HqiTao/SplatHLoc | `c73657e` |
| DeViLoc | https://github.com/TruongKhang/DeViLoc | `c9e990b` |
| scrstudio (R-SCoRe) | https://github.com/cvg/scrstudio | `a2b40bd` |
| colmap (mpsfm ext) | https://github.com/Zador-Pataki/colmap | `739d84b` |
| LoFTR / UFM | (原機器上非 git checkout) | — |
| EDM | https://github.com/chicleee/EDM | (在定位 repo) |
| GlueMap | https://github.com/colmap/gluemap | — |

LFOE 此 revision 未提供 top-level license；只允許在取得明確授權後進入 experiment queue，不是 production dependency。

`sites/target_site/tools/preflight_research_backends.py` 會檢查這些後端在不在。

---

## 環境（原機器實測）

| 用途 | 直譯器 |
|---|---|
| COLMAP 4.0.4 GlobalMapper（CUDA） | `~/micromamba/envs/sfm` |
| GlueMap / GLOMAP / pycolmap 4.0.4 | `~/micromamba/envs/target-site-gluemap-run` |
| EDM bundle 產生 | `sfm_system/EDM定位測試/env/edm_eval_py312`（`/usr/bin/python3.12` venv + `yacs`） |
| sm_120 CUDA 編譯 | `CUDA_HOME=~/miniconda3/envs/gsplat`（唯一完整的 CUDA 12.8） |

- shell 打 `python3` 會抓到 miniconda base (3.13)，**那支沒有 pycolmap**。
- `/usr/bin/python3` (3.12) 有 torch 2.11.0+cu128 與 pycolmap 4.0.4，**但沒有 yacs**。
- `/usr/bin/nvcc` 是 CUDA **12.0**，編 sm_120 會 `Unsupported gpu architecture 'compute_120'`。
- `micromamba/envs/target-site-gluemap-run` 有 nvcc 12.8 **但沒有 toolkit headers**。
  **nvcc 在 PATH 上不等於有 toolkit。**

pycolmap 4.0.4 的 rig model 會擋掉部分 CLI / model surgery，繞法是走 text model。

---

## 絕對路徑

約 50 個檔案裡寫死 `/media/cihcilab/新增磁碟區/...` 或 `/home/cihcilab/...`。
這是刻意的——原系統的每份 `*receipt*.json` / `gates/*.json` 都用絕對路徑 + SHA-256
記錄輸入輸出，搬動目錄會讓證據無法重新驗證。

在別台機器上跑之前先 grep：

```bash
grep -rn '/media/cihcilab\|/home/cihcilab' --include='*.py' --include='*.json' .
```

---

## 原始路徑對照

| 這個 repo | 原始磁碟位置 |
|---|---|
| `pipeline/` | `sfm_system/建圖/pipeline/` |
| `sites/<site>/` | `sfm_system/建圖/<site>/` （不含 `runs/`） |
| `map_update/` | `sfm_system/更新地圖/pipeline/` + `更新地圖/source/sfm_reshot25/` |
| `map_update/core/` | `sfm_system/更新地圖/source/sfm_reshot25/update_pipeline/` |
| `configs/` | `sfm_system/configs/` |
| `docs/` | `sfm_system/docs/` |
| `diagnosis/` | 原獨立 repo `sfm_map_diagnosis`（MapDoctor + sfm-diagnosis） |

**沒有推上來的**：`runs/`（265 GB 產物與證據）、`external_tools/`（7.8 GB 第三方）、
`定位/` 與 `EDM定位測試/`（定位側，在另一個 repo）、`LoMa建圖測試/`（另一條建圖路線的實驗）。
