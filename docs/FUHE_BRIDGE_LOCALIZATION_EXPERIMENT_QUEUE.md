# 福和橋定位實驗清單

更新日期：2026-07-23

> **狀態：STOPPED / FROZEN（使用者於 2026-07-23 指示停止）**  
> 不再排程福和橋建圖、GGPT、SURE、SLiM 或 EDM 定位。以下清單保留為
> 可重現紀錄，不代表仍待背景執行。

目的：在同一份通過 release gates 的福和橋幾何上，比較 EDM balanced、SURE 與 SLiM 的定位成功率、精度代理、速度及資源用量。P1130113 僅作 held-out query，不得加入任何地圖或 matcher-specific association 的建立過程。

## 執行前置門檻

- [ ] `fuhe_bridge_probe_v5` 完成 GlueMap/GLOMAP。
- [ ] G4.0 Doppelgangers 圖結構門檻通過。
- [ ] 固定內參模型定案，後續實驗不得改 camera poses 或 intrinsics。
- [ ] 獨立 Sim3 與鬼影檢查通過。
- [ ] EDM bundle 與 held-out P1130113 測試 manifest 凍結並記錄內容 hash。

前置門檻未全部通過時，不啟動 SURE/SLiM GPU 實驗，也不建立完整場域的衍生地圖。

## 實驗佇列

| ID | 方法 | 地圖／association | 狀態 | 執行條件 |
|---|---|---|---|---|
| FBL-E0 | EDM balanced | v5 標準 EDM map | 停止／未執行 | v5 未通過 G6.1，不可建立 promotion baseline |
| FBL-E1 | [SURE](https://arxiv.org/abs/2603.04869v1) | 既有 COLMAP observations 的 matcher-neutral pixel 2D–3D association | 停止／未執行 | 建圖前置門檻未通過 |
| FBL-E2 | SURE | SURE-specific 定位 overlay | 停止／未執行 | FBL-E1 未啟動 |
| FBL-E3 | [SLiM](https://openaccess.thecvf.com/content/CVPR2026/html/Choo_Scalable_Feature_Matching_via_State_Space_Modeling_and_Sparse_Correlation_CVPR_2026_paper.html) | single-pair adapter + matcher-neutral pixel 2D–3D association | 停止／未執行 | 建圖前置門檻未通過 |
| FBL-E4 | SLiM | SLiM-specific 定位 overlay | 停止／未執行 | FBL-E3 未啟動 |
| FBL-E5 | 三方法公平比較 | 各方法通過 gate 的 map/association | 停止／未執行 | 無可 release 的福和橋定位圖 |

## 專用定位地圖規則

「專用地圖」預設只指 matcher-specific 2D–3D overlay，不是重新跑完整 GlueMap、GLOMAP 或全域 bundle adjustment。

建立順序固定如下：

1. 先用最終 v5 的既有 COLMAP observations 建立 matcher-neutral pixel association。
2. 若量測證明 2D match 到既有 3D observation 的轉換率是主要瓶頸，才以 SURE 或 SLiM 在 map images 間建立 tracks。
3. 固定 v5 camera poses 與 intrinsics，只三角化新的 3D points；不得讓 test sequence P1130113 參與。
4. 如需優化，只允許 point-only refinement；任何 camera BA 都視為另一個建圖實驗，不能混入本定位比較。
5. overlay 必須通過 reprojection、cheirality、track length、重複點與鬼影區域檢查後才能用於 held-out 評測。

## 統一量測欄位

- 定位成功率與連續失敗長度。
- PnP raw correspondences、unique 3D anchors、inliers、inlier ratio 與 reprojection error。
- 有獨立姿態參考時才報 translation/rotation error；沒有 pose GT 時不得把軌跡平滑度或 inlier 數宣稱為絕對精度。
- retrieval、matching、2D–3D conversion、PnP 與總延遲的 median/p95。
- FPS、峰值 VRAM、RAM、模型載入時間與衍生 map 大小。
- 重複橋墩、弱紋理、逆向 traversal 與已知鬼影風險區段的分層結果。

## Promotion 規則

- 新方法必須在相同 held-out manifest 上不降低定位成功率，且精度證據不得只依賴 inlier 數。
- 速度 promotion 以部署硬體 RTX 5060 的實測為準；RTX 5090 結果只作開發篩選。
- SURE 的 uncertainty filtering 與 SLiM 的 sparse-correlation/filtering 參數必須記入 run config，不可只保存最終摘要。
- 任何方法若需要改相機姿態、固定內參或原始 v5 幾何，立即退出本清單，另立建圖實驗。

## 既有證據

- target_site 已做過 SURE、JamMa 與 SLiM static gate；結果與限制記錄於 [`SCREENING_REPORT.md`](../建圖/target_site/runs/target_site_v1/experiments/map_method_screen_20260722/SCREENING_REPORT.md)。福和橋仍須使用自己的 frozen split 重新量測，不得直接沿用 target_site 結論。
- SURE 官方程式：<https://github.com/LSC-ALAN/SURE>
- SLiM 官方程式：<https://github.com/Band-127/SLiM>

## 停止時的最終證據

### 保留的最強模型

- 路徑：
  `建圖/fuhe_bridge/runs/fuhe_bridge_probe_v5/experiments/p114_fixed_pose_retriangulation_v3_from_aba/final_fixed`
- registered images：239/240。
- points3D：114,910。
- mean reprojection error：0.9265 px。
- 固定內參：PASS；獨立 Sim3：PASS；G6.1 symmetric-overlap：FAIL。
- 此模型是研究紀錄的最強候選，但不是可交付定位圖。

### v4／v5 為何不升版

- v4 track completion 使整體結果退化。
- v5 clear-points 將 P110 邊改善到約 0.04482，但 P112 仍約 0.08，
  registered images 降至 238/240，仍未通過 G6.1。
- P110↔P114 尚需 59 次可靠合併，最大可見支援 78，理論可達；
  P112↔P114 尚需 124 次，現有語料最大可見支援只有 26，無法達標。

### P112↔P114 人工共同視野搜尋

- 強共同視野：P112 126.0 秒 ↔ P114 25.5 秒，在 1280 寬度得到
  7,411 inliers、96.6% inlier ratio、16/16 grid coverage；資料庫內相鄰既有
  pair 已有 1,978 verified matches，因此它不是缺失的鬼影跨區連接。
- 鬼影風險窗最佳候選：P112 60.0 秒 ↔ P114 114.5 秒，只有 18 inliers、
  56.3% ratio、5/4 grid coverage；相鄰正式 frames 只有 12 inliers，且 pair
  不在資料庫。這不足以安全地強制合併。

### 尚未執行的實驗與原因

- P112 稠密補幀／LoFTR promotion：現有 ghost-window 支援遠低於 G6.1
  所需量；繼續加入弱邊有製造重疊鬼影的風險。
- GGPT Pi3 sidecar：preflight 已建立，但 sparse overlap gate 未通過；
  GGPT 不能把缺少的真實共同視野憑空補出來。
- EDM／SURE／SLiM 定位：沒有通過 release gates 的 map，執行定位數字會把
  鬼影風險掩蓋成看似正常的 PnP 結果，因此停止。

### 重新啟動條件

只有取得一段連續、清楚、跨越 P112 鬼影區至 P114 的新 map traversal，且
query/held-out 仍保持隔離時才重啟。未達此條件前，福和橋所有產物保持只讀，
不再作跨場域回歸或方法 promotion。
