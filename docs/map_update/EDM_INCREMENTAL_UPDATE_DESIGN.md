# EDM 時代的地圖增量更新設計

_2026-07-26 · 取代 `UPDATE_PIPELINE_METHODS.md` 的 XFeat 假設，不取代
`map_update/MAP_UPDATE_STRATEGY_RECORD.md` 的長期架構論證_

---

## TL;DR

**用 gauge（座標系）有沒有被動過來分流，不要用「新資料多不多」來分流。**

理由是交付契約決定的。每次更新完，定位端要拿三樣東西：EDM bundle、五個尺度參數 +
到達容差、`T_align_gravity.json`。這三樣的失效條件不一樣：

- 只要有任何一個舊相機 pose 動了 → 三樣全部作廢，你已經付掉大部分重建的成本
- 舊 pose 逐值不變 → `T_align_gravity` 直接沿用，EDM bundle 只補新的，只有尺度參數必須重算

所以真正該問的不是「新資料多不多」，是**「這次更新能不能不動到任何舊 pose」**。

而「固定 pose 做三角化」這件事，**你已經有現成的程式**：
`EDM定位測試/build/build_reloc_map_edm.py` 的 stage E 就是 hloc fixed-pose triangulation。
它本來是拿來把 EDM cell anchor 打成 3D 的，換個輸入就是增量更新的局部三角化。

---

## 標準答案：三個不同頻率的迴圈

長期地圖維護的公認有效解（Bürki et al., IV 2018）不是「一種更新方式」，
是**用成本不對稱把工作分層**：便宜的事天天做，昂貴的事只在**量到**需要時才做。

| 迴圈 | 觸發 | 做什麼 | 成本 | 三樣交付物 |
|---|---|---|---|---|
| **L1 observation session** | **每次飛行，無條件** | 只記錄哪些 ref / 3D 點被命中。**不加任何幾何** | ~0 | 全部不動 |
| **L2 rich session（保 gauge）** | 某區域定位品質**掉到門檻以下** | 固定舊 pose，只對該區局部三角化 | 分鐘級 | bundle 補 delta；**尺度參數重算**；重力沿用 |
| **L3 controlled rebuild** | gauge 必須改 / 漂移累積 / 大範圍變更 | 完整重建 + S0–S9 驗收 | 6–8 小時 | 全部重出 |

原論文的流程就是：新資料先對現有地圖離線定位 → **定位表現低於預設門檻**就當
rich session（三角化新 landmark 加進地圖），**高於門檻**就只當 observation session
（只更新共視統計，不加 landmark）→ 每次 rich session 之後跑一次
**offline summarization** 把 landmark 總數壓回固定上限。

論文的具體設定：門檻是 **10 cm translation RMS**，summarization 上限
City-Environment **75k**、Parking-Lot **150k** landmarks；驗證涵蓋停車場整年季節變化
與市區全日夜光照變化。

### 這三件事各自為什麼重要

1. **L1 讓 L2 的判斷有依據，而且免費。** 沒有持續累積的 observation 統計，
   你根本分不清「這區真的變了」還是「這次拍攝角度不好」。
   單次證據不足以退役任何 3D 點。
2. **L2 只在量到退化時才做。** 不是排程、不是「有新影片就更新」。
   「每次拍完就 merge 進地圖」是最常見也最貴的錯誤：地圖無界成長、
   同一批 appearance 重複加權、定位反而變慢變差。
3. **每次 L2 之後必須 summarize，否則地圖無界成長。**
   `map_update/core/sparsify_reloc_bundle.py` 已經在做這件事
   （observation-hit score + 每 sequence/prefix 的 ordered K-cover），
   這正是 Dymczyk 式 summarization 的形狀。**它目前是選用的，應該變成 L2 的強制收尾。**

### 要照抄，但有一處不能照抄

論文用 **10 cm translation RMS**（來自輪速計）當 rich / observation 的分界。
**你沒有輪速計，也沒有公制尺度**，這個門檻搬不過來。

替代品你已經有了——就是 **S9 的驗收契約**，套在新影片上：

```
成功率            < 90%           → 這段需要 L2
inlier p5         < 30            → 幾何支撐不足
最大連續失敗       > 30 frames     → 有連續盲區
失敗在空間上聚集   （非隨機分布）   → 指出要局部更新的區域
```

**改一點：論文是整趟 sortie 二選一，你應該做到 per-tile。** 你的航線是線狀的，
一小段爛不該觸發整趟 rich session。用失敗幀的空間聚集去圈出要動的區域，
其餘區域維持 observation-only——這樣 L2 的範圍最小，gauge 也最容易保住。

---

## 為什麼不是「先定位 → 找定位不到的區域 → 局部三角化」就好

你的直覺方向是對的，而且就是文獻上的 method 2 / method 3。但有兩個坑：

**坑 1：`register_rate` 不是場景變更偵測器。**
一段影片可以 register_rate 99%、同時某面牆已經改建。定位得到 ≠ 地圖還對。
「定位不到的區域」只找得到**新增覆蓋**，找不到**已變更**的區域——後者反而是
定位得到、但局部 inlier 支撐持續偏低的地方。這兩種要用不同證據分辨：

| 訊號 | 解讀 | 動作 |
|---|---|---|
| 定位不到 + 有足夠 bridge | 新覆蓋區 | 增量三角化（本文主線） |
| 定位得到、pose 穩、但某 tile 的 old-map support 反覆偏低 | 同一幾何位置外觀/內容變了 | tile 替換（要多 session 證據） |
| retrieval 相似度高但 PnP inlier 低 | 重複結構造成的假重疊 | 拒絕，不要信 retrieval |
| inlier 多但擠在畫面一角 | pose 解得出來但覆蓋不完整 | 不當作完整重疊 |

**坑 2：局部三角化的「局部」如果用了 BA，就不局部了。**
只要 BA 的可調參數集合碰到舊相機，gauge 就變了，`T_align_gravity` 立刻 stale——
而且這種歪法**在定位成功率上完全看不出來**。成功率照樣 99%，只是飛機以為的「上」不是上。

---

## 建議流程

沿用建圖側的 S 階段命名習慣，用 U 開頭。**每一階都盡量重用既有程式**。

### U0 — corpus lock + no-op 守門

重用 `sites/target_site/tools/s0_corpus_lock.py` 的模式。

- content hash 新影片；與既有 build sequence 相同的 → **no-op**，不是新 session
- 近重複影格由 global descriptor + 時序 + 幾何 overlap 擋下
- 重送的舊資料不得更新 `last_seen_session`、不得進 held-out set

**為什麼重要**：漏掉這關，重送同一批舊影片會讓過期點被永久「續命」，
而且驗收指標會有 test leakage，你會看到假的好成績。

### U1 — motion scan

**直接重用 `s1_motion_scan.py`**，不用改。

輸出 geometry / connector 二分：

- `geometry/`：有平移視差，可以造新 3D 點
- `connector/`：H-dominant（`H_inliers / F_inliers >= 0.85`）的轉彎幀，
  只能用來檢索、PnP、當 bridge，**不得造新點**

注意建圖時學到的教訓：**「純旋轉」段落常常不是退化的**，雲台在動時仍有 0.8–4.3° 視差。
別用「看起來像轉彎」整段丟掉。

### U2 — 用**部署中的 EDM localizer** 對舊地圖定位

這裡刻意不用 XFeat、也不用 MV-RoMa。理由：這一階要量的是
**「真正會飛的那套，在這裡找不找得到自己」**。換 matcher 就不是在量那件事。

每幀輸出：pose、inliers、inlier 的**影像空間分布**、命中的舊 keyframe、tile support。
彙總成 `observation_stats.json`（欄位沿用現有格式，`map_update/core/` 已有）。

### U3 — 路由，**主軸是 gauge**

```
                    ┌─ register_rate > 0.95 且無變更證據
                    │     → R0 不動地圖。只記 observation。三樣交付物都不用重發
                    │
新資料 ── U2 定位 ──┼─ 新覆蓋區，且新幀能靠舊 3D 點 PnP 定出 pose
                    │     → R1 保 gauge 增量（主線，見下）
                    │
                    ├─ 新區域與舊圖只剩少量 bridge，新幀 PnP 不穩
                    │     → R2 quarantine。這是 controlled rebuild，不是 incremental update
                    │
                    └─ 定位穩但特定 tile 支撐持續偏低（跨多 session）
                          → R3 tile 替換。邊界 anchor 會動 → 視同 gauge 變更處理
```

**R0 / R1 是便宜的；R2 / R3 一律當成重建等級的事情**。別讓 R2 偽裝成 R1。

### U4 — R1 主線：保 gauge 的增量三角化

這是整份文件的核心，五步：

1. **新幀求 pose**：EDM 匹配到舊 keyframe → 2D-3D → PnP。
   舊 3D 點與舊 pose 全程唯讀。
2. **凍結舊側**：
   ```
   colmap mapper \
     --input_path  <old_sparse> \
     --Mapper.fix_existing_frames 1 \
     --Mapper.ba_refine_focal_length 0 \
     --Mapper.ba_refine_principal_point 0 \
     --Mapper.ba_refine_extra_params 0
   ```
   `fix_existing_frames` 是**請求**，不是證據。證據是 U5 的 G-U1。
3. **新點三角化**：把新幀（只用 `geometry/`）與其 co-visible 舊幀組 pair，
   走 `build_reloc_map_edm.py` 的 **stage A→E** 產生新 3D 點。
   Stage E 本來就是 fixed-pose triangulation，**這正是需要的語意**。
   只匯出被 ≥2 張 geometry 幀觀測到的點（現有 `--min-geometry-observations` 語意）。
4. **connector 幀**：可以進 bundle 當定位 keyframe，xyz 用 PnP 從舊圖繼承，
   **不得**用低視差的新點。
5. **不做 joint BA**。想做 joint BA 的那一刻，你就離開 R1 了。

> **這一步和現況的差距**：`map_update_tool.py` 的 register 路徑目前**只做 PnP 加 keyframe，
> 不產生新 3D 點**。上面第 3 步是要補的東西，而它不需要從零寫——
> `build_reloc_map_edm.py` 的 stage A–E 已經是這個形狀。

### U5 — Gauge gate 與交付物

| Gate | 檢查 | 不過的後果 |
|---|---|---|
| **G-U1** | 舊 image 的 `R, t` 更新前後逐值比對，max delta ≤ `1e-9` | 不得宣稱 gauge 不變 → `T_align_gravity` 標為 stale，擋下交付 |
| **G-U2** | 未被替換的舊 `points3D` xyz 逐值比對 | 同上 |
| **G-U3** | 重算 `S = 2·p95(‖center − componentwise_median‖)`，重發五個尺度參數 | **即使 G-U1 通過也必須做**（見下） |
| **G-U4** | EDM bundle：新 keyframe 的 cell→xyz round-trip、`ref_global` 為 `[refs, 8448]` | 不得交付 |
| **G-U5** | held-out 定位：用沒進過建圖也沒進過這次更新的影片，成功率不得退步 | 不得取代 production |
| **G-U6** | 舊場域 regression set 不得出現新增失敗 | 不得取代 production |

G-U1 的作法直接抄建圖側：`finalize_edm_model.py` 對內參做的就是逐參數比對
（容差 `1e-6`，target_site 實測 delta `0.0`）。同一個模式套到 pose 上。

**G-U3 為什麼「即使 gauge 沒變也要做」**：`S` 是對**全部**相機中心取 `p95` 算的。
往新區域加 keyframe 會讓 `p95` 位移，五個參數跟著全變。
「舊 pose 沒動 ⇒ 尺度參數不用改」是錯的，這是最容易漏的一條。

---

## 交付物失效矩陣

| 路由 | gauge | 相機集合 | EDM bundle | 5 尺度參數 | T_align_gravity |
|---|---|---|---|---|---|
| R0 observation only | 不變 | 不變 | 不動 | 不動 | 不動 |
| **R1 保 gauge 增量** | **不變** | 變大 | **只補新 keyframe** | **必須重算** | **✅ 沿用** |
| R2 submap + Sim3 + joint BA | 變 | 變大 | 全部重建 | 全部重算 | 必須重求 |
| R3 tile 替換 | 邊界動 | 可能不變 | 全部重建 | 重算 | 視 BA 範圍 |
| 週期性 `global_mapper` 全域重解 | 全變 | 變 | 全部重建 | 全部重算 | 必須重求 |

R1 的價值不在省時間，在**三樣交付物只有一樣要真的重建**。

---

## Matcher 分工

| 用途 | 用什麼 | 為什麼 |
|---|---|---|
| U2 對舊圖定位 / 路由判定 | **部署中的 EDM** | 要量的就是「會飛的那套找不找得到自己」，換 matcher 就失去意義 |
| U4 新點三角化 | **EDM cell-anchor**（stage A–E） | 程式已存在；與 bundle 同一套 identity，不用維護第二條 |
| 低重疊 / 寬基線救援 | RoMa v2 pair rescue | 只在 EDM 造不出足夠 track 時離線補 |
| 大規模 controlled rebuild | MV-RoMa multi-view tracks | 見 `MAP_UPDATE_STRATEGY_RECORD.md`；**不是** incremental 路徑 |

先全部用 EDM。只有在實測「新區域三角化點數/角度不足」時才引入 MV-RoMa——
而那時你多半已經在 R2（controlled rebuild）了，本來就該用比較強的前端。

**cell-anchor 的硬約束**：`round(kpt/8)` 精確可逆是整套 identity 的地基。
未經 round-trip / resize / multi-resolution gate，**不得改 input canvas**。

---

## 實作順序

| # | 工作 | 依賴 | 備註 |
|---|---|---|---|
| **P0** | 修 base path。`map_update/update_pipeline.py` 預設指向已不存在的 `sfm_glomap/` | — | 沒有可重現的 canonical map，後面全部沒有意義 |
| **P1** | U2 換成 EDM localizer（目前是 XFeat+LighterGlue） | P0 | |
| **P2** | **G-U1 / G-U2 gauge gate** | P0 | 抄 `finalize_edm_model.py` 的逐值比對。**先做這個**，它讓後面所有路徑可驗證 |
| **P3** | G-U3 尺度參數重算，接 `derive_site_profile()` | P2 | 純量計算，最便宜、最容易漏 |
| **P4** | U4 新點三角化：接 `build_reloc_map_edm.py` stage A–E | P1, P2 | 補上 register 路徑缺的「局部三角化」 |
| **P5** | EDM bundle 增量打包（只補新 keyframe） | P4 | |
| **P6** | `T_align_gravity` 的 carry / re-derive 決策，綁 G-U1 | P2 | |
| **P7** | point-level evidence ledger（ExMaps 式） | P0–P2 | 目前 `stability_scores.py` 是 reference-level，不是 point-level |
| **P8** | R3 tile 替換自動化 | P7 | 需要多 session 證據才有意義，排最後 |

P2 排在功能之前是刻意的：**先讓「有沒有動到 gauge」變成可驗證的事實**，
之後每條路徑才有辦法誠實分類。

---

## 明確不要做的事

- **不要用 `.pt` bundle 反推或更新 canonical geometry。** `geometry/` 是唯一真相，
  `deployment/` 可丟棄重建。
- **不要在影像集合改變後用 GlueMap `force_load=True`。** 目前 cache 只看檔案在不在，
  沒有 image-list / hash 驗證，會載到舊影像集合的結果。
- **不要只靠一次 Sim3 或一次拍攝就覆蓋正式地圖。** 低重疊 submap 先進 quarantine。
- **不要用單張 frame 的顏色差異刪 3D 點。** 曝光、陰影、反光、視角都會造成假陽性；
  要多 frame、多視角、最好多 session 的累積證據。
- **不要把「未匹配」一律當成過期。** 只有在「投影在視野內、深度/遮擋合理、該幀定位可靠」
  卻仍持續匹配不到時，才算有效負面證據。
- **不要用 frame 當衰減單位。** 一段長影片會對同一個點重複投票數千次。用 session。
- **不要在沒有 RTK/GCP 的情況下，把 inlier 數或跨方法一致性講成公分級絕對精度。**

---

## 已知風險

- **沒有外部 pose ground truth。** 所有 map-unit 誤差都是 proxy。
- **MegaLoc 對正反向是可測量地盲目的**（相似度 0.10–0.17 vs 同向 0.31–0.59）。
  更新時的 bridge 搜尋同樣會踩到，forced VPR-blind bridging 在更新階段一樣是承重結構。
- **GlueMap 沒有原生 append-existing-map 契約。** R1 是繞過去（固定 pose + 只加新點），
  不是 GlueMap 支援的操作。真要走 R2 得先做一次 canonicalization
  （見 `MAP_UPDATE_STRATEGY_RECORD.md` 的「已有 GlueMap map」一節）。
- **`T_align_gravity.json` 目前不在建圖側產生。** 它需要一個明確的產生者與 provenance，
  否則 R2/R3 之後沒有東西可以重求。
