# target_site GlueMap → EDM 實作與驗收紀錄

> 狀態：進行中（最後只有 S0–S9 全數 PASS 才能標示已完成）  
> 執行日期：2026-07-17 起  
> 目標：交付可直接供 EDM localization 使用的 target_site map，並以 P1230123/P1260126 留出影像驗收。

## 驗收原則

- 建圖與檢測流程是 global SfM，不額外疊加 pose-graph optimization。
- 不把 forced candidate 當成 forced connectivity；Doppelgangers++ 仍可拒絕錯誤橋接。
- 不因為「有輸出檔」就稱為完成；S0–S9 任一 hard gate 未通過就不交付。
- pure-rotation 影像保留 pose，但最終 3D map 必須是零 observation。
- 內參在最終 BA 固定，並與 seed 逐參數比對（容差 `1e-6`）。
- EDM bundle 建立後，一定用未進入建圖資料的 P1230123/P1260126 做最終留出驗收。

## 已成功階段

### S0–S3：資料、動態抽幀、內參與 pair contract

- 7 條 build sequence，共 1,414 張影像；P1230123/P1260126 只屬於 test，未洩漏進 map。
- S2b 內參烘培對 3 種解析度、6 組模型做獨立 replay，全部通過。
- S0–S3 release replay：25/25 PASS。
- 2026-07-17 11:35 時的正式 config SHA-256：
  `ce4d151864bafd3bd1618007ab998ad9ef7594a55d4a2cc2dfef771ebafc7dc0`
- memory-safe launcher SHA-256：
  `f13de43cb177d3147790bef00d12dc9513d909f692710eed3dec170eedc273cb`

### S4：Doppelgangers++

- candidate pairs：126,706
- threshold 0.8 接受 105,876，拒絕 20,830（16.44%）。
- 最大 image component：1,395/1,414（98.66%）。
- 單區域弱跨向邊全數移除後，sequence graph 仍由三組雙區域骨幹完全連通：
  - S06_P1240124 ↔ S02_BA：133 條 accepted forced edges
  - S06_P1240124 ↔ S03_BA2：122 條 accepted forced edges
  - S06_P1240124 ↔ S05_P1220122：191 條 accepted forced edges
- S4 gate：G4.1–G4.4 全數 PASS。
- 證據：`runs/target_site_v1/gates/S4_doppelgangers.json`。

### GlueMap 正式建圖與精煉

- 正式 detached pipeline 完成，總耗時 15,532.26 秒。
- 第二輪 refinement 最終保留 397,525 條 real tracks（5,919,683
  observations）與 61,551 條 virtual tracks（1,108,273 observations）。
- real reprojection angle mean `0.0725°`、median `0.0397°`，99.2%
  低於 `0.5°`。
- 第二輪第一次 BA 在 201 iterations 時為 `NO_CONVERGENCE`，但
  cost 由 `1.406420e7` 穩定降至 `1.063752e7`；緊接的 BA 在 16
  iterations 收旂（`1.062201e7 → 1.062140e7`），下游結構可正常讀取。
- 模型：`runs/target_site_v1/gluemap/gluemap_aba`；完整 log：
  `runs/target_site_v1/recovery/s4_s5_memory_safe_detached_20260717_1247.log`。

### S5 第一次修復版（後續發現仍需更嚴格主分量裁切）

- registered 1,414/1,414，points3D 393,932，mean reprojection
  `1.521834818 px`，invalid-depth observations 0。
- 內參已完全復原為 seed，maximum delta `0.0`；pure-rotation
  observations 0。
- median triangulation angle `9.0958°`，低於 `1°` 的比例 0.5077%。
- 當時 G5.3 以 1,383/1,407 = 98.294% 達到原定的 98% 門檻，
  因此 S5.1–S5.6 形式上 PASS；後續 robust geometry audit 顯示這個
  容差會保留一個真實的錯誤次分量，故不把此版當作交付版。

### S5 最終嚴格主分量版：PASS

- 從原始 `gluemap_aba` 重做一次可重現 finalization，並在 BA 前移除
  S06 的 24 張錯誤次分量/零觀測姿態。
- registered 1,390/1,414 = 98.303%；其中 1,383 張可定位影像為
  1,383/1,383 的單一 track component，非預期 zero-observation registered
  image 為 0。另 7 張 pure-rotation 保留 pose 但為零 observation。
- points3D 391,776；mean reprojection `1.451463486 px`；invalid-depth
  observations 0。各 sequence mean 為 1.180–1.797 px。
- median triangulation angle `9.0229°`；低於 `1°` 的比例 0.5557%。
- final camera params 與 seed 完全一致，maximum intrinsics delta `0.0`。
- fixed-intrinsics BA：101 iterations，cost `1.02937 → 0.989939 px`，
  `NO_CONVERGENCE`。它不是以 termination label 單獨放行；是因 cost 有限且
  下降，且 G5.1–G5.6 全數 PASS 才收下。
- gate：`runs/target_site_v1/gates/S5_fixed_intrinsics.json`；log：
  `runs/target_site_v1/recovery/s5_component_prune_20260718_0745.log`。
- **runtime 核對**：S5 使用 sibling `target-site-gluemap` env，而 source lock
  指定 `target-site-gluemap-run`。兩者的 pycolmap 皆為 4.0.4，且實際
  `_core.cpython-311-x86_64-linux-gnu.so` SHA-256 同為
  `107e972d5b1b2e8f8a35f8a61e84d6ca1e945d526088a9f8af21bece578c65f3`。
  sibling 的 NumPy 為 2.4.6，canonical 為 2.4.3；BA/模型修改是同一
  pycolmap C++ core 執行，最終 JSON 門檻還會由 canonical runtime 重讀模型
  稽核。

### S5 邊隔離後最終交付模型：PASS

- canonical runtime 從原始 `gluemap_aba` 重做，裁掉 24 張 S06
  錯誤姿態，並刪除 3,426 個 S03/S06 共觀 points（98,763
  observations）。final 殘留該邊 points = 0。
- registered 1,390/1,414；可定位影像主分量 1,383/1,383；
  points3D 388,353。
- mean reprojection `1.443595993 px`，invalid observations 0，各 sequence
  mean 1.180–1.772 px。
- median triangulation angle `8.9601°`，<1° 比例 0.5606%；內參差 0。
- BA：101 iterations，cost `1.02474 → 0.984924 px`，所有 G5.1–G5.6
  PASS。log：`runs/target_site_v1/recovery/s5_edge_quarantine_20260718_0652.log`。

### S5.7 + S6：PASS

- trusted independent Sim3 edges：`S02_BA|S06_P1240124`、
  `S05_P1220122|S06_P1240124`。
- `S03_BA2|S06_P1240124` 為 `QUARANTINED`，raw status FAIL，final
  shared points 0；不參與 trusted backbone。
- robust camera span `5.00649`；正反向 symmetric overlap median/p90 為
  `0.02662/0.08906` span，G6.1 PASS。
- 正反向共享 40,757 points（10.495%），G6.2 PASS。
- 全域 reprojection `1.44360 px`，7 條 sequence 全在 1.5× 範圍，G6.3
  PASS。
- 2026-03/05/06 三組 epoch pair 的 subset median 為
  `0.02081/0.01194/0.02309` span，G6.4 PASS。
- gates：`gates/S5_7_independent_sim3.json`、`gates/S5_7_S6_geometry.json`。

### S7：PASS

- MegaLoc active refs 1,383，descriptor shape `(1383, 8448)`，全數 finite
  且 L2-normalized。
- pose metadata 1,383/1,383，median covisibility degree 40，零 pure-rotation
  reference。
- yaw 兩群分離 `175.9909°`，counts 860/523，小群佔 37.82%，
  concentrations 0.837/0.769。
- gate：`gates/S7_tracking_bundle.json`；bundle：
  `edm/target_site_v1_seed_tracking.pt`（67 MiB）。

## 失敗、根因與修正

### F1：原始 star/refinement 在進 BA 前發生記憶體飽和

- **症狀**：可用 RAM 只剩約 2.7 GiB，swap-out 持續 54–69 MB/s。
- **處置**：以 Ctrl-C 安全中止，保留 DG/SALAD 快取。
- **根因**：約 14.26M SIFT keypoints / 20.68M track keypoints 在 `max_num_tracks`
  生效前就已載入；真正 bottleneck 是上游 feature/track load，不是 DG。
- **修正**：`num_track_per_img=512`、`SIFT max_num_features=2048`、
  `max_num_tracks=400000`，並加入 fail-closed memory-safe launcher。
- **未採用**：不降 DG 0.8 threshold，因為那不會解決上游記憶體瓶頸，還會增加幽靈結構風險。

### F2：修改外部 GlueMap 會破壞 S2b 歷史 source hash

- **症狀**：直接在外部 GlueMap 加 SIFT cap 後，與內參烘培時鎖定的 source hash 不同。
- **修正**：外部 repo 完全復原成使用者原本的 dirty state；改用
  target-site 專用 launcher 在 import GlueMap 前 monkeypatch `pycolmap.extract_features`。
- **驗證**：外部 `gluemap/utils/colmap.py` SHA-256 復原為
  `bf5644dab3048ca87623a49125a74d6afcc20ee8847d08218a612d3872137971`；S2b 六組獨立 replay 重新 PASS。

### F3：互動 PTY 在 star inference 80% 被新訊息終止

- **症狀**：第一次 memory-safe 正式 star run 在 1,130/1,414 時，主 PID 與監控
  session 因新使用者訊息後的互動 session 回收而消失。
- **資料完整性**：DG/SALAD 快取完整；不是 OOM，swap-out 全程為 0。
- **為何不能從 80% 續跑**：GlueMap 原生 `star_result.pth` 只在全批 inference
  完成後才 atomic save，中間不有 per-image checkpoint。
- **修正**：用 `nohup + setsid` 將正式 PID 11691 與對話完全脫鉤，並從 star
  重跑；DG 不重算。

### F4：第一個 detached Bash resource guard 立即失效

- **症狀**：guard log 為空，背景 PID 立即結束。
- **根因**：`bash -c` 在脫鉤啟動的多層引號解析不可靠。
- **修正**：改寫 `tools/resource_guard.py`，將低記憶體與連續 swap-out 決策做成
  可測試函式；3 個單元測試與 `ruff` 全數通過。
- **正式保護**：PID 12630 監控主 PID 11691；低於 4 GiB 或 swap-out
  ≥32 MiB/s 連續兩次時送出 SIGINT。

### F5：S4 gate 的 forced-pair parser 不接受註解

- **症狀**：`forced_bridges.txt` 第一行是 `#` 註解，解析器誤當成壞資料。
- **修正**：先加 regression test，再使 parser 忽略註解與空行。
- **結果**：S4 工具 4 個測試 PASS，正式快取可完整解析。

### F6：過度嚴格的 S4 G4.3 定義會把「存在弱邊」誤判成「骨幹依賴弱邊」

- **第一版結果**：0.8/0.7/0.6 都 FAIL，因為某些非必要跨向 sequence pair 只有一個
  時間區域的 accepted edge。
- **問題**：此定義忽略了 sequence graph 可透過其他雙區域骨幹連通；它會因一條
  「可刪的偶發邊」否決一張不依賴該邊的圖。
- **修正定義**：刪掉所有不足兩個分離橋群的跨向邊後，7-sequence graph 仍必須
  100% 連通，且至少保留一條 robust cross-direction edge。
- **測試與結果**：新增「非必要單鉸鏈被刪除但骨幹仍連通」測試；總計 5 測試
  PASS，正式 G4.3 PASS。

### F7：第一輪大型 CUDA BA 出現多次 non-finite step 警告

- **症狀**：400,000 real + 124,340 virtual tracks，約 1,717 萬 residual 的
  Ceres BA 在 01:31–01:58 間多次報告 `Linear solver failure. Failed to
  compute a finite step.`。
- **審核方式**：不以單條 warning 當作 crash；同時監控 PID、CPU/GPU、RAM、
  swap-out 與最終 Ceres termination。期間主程序持續使用約 300% CPU，
  RAM 可用約 11.7 GiB，swap-out 為 0。
- **結果**：求解器在 121 次迭代後回報 `CONVERGENCE`；cost 由
  `3.371967e+07` 降至 `1.415186e+07`，並成功進入 filtering 與第二輪 BA。
  因此未誤殺可收旂的正式作業。
- **預防性 fallback**：target-site launcher 新增 recovery config 可選的
  `ba_max_num_iterations` / `ba_max_filter_iterations`，不改外部 GlueMap，且可只
  重跑 refinement。4 個 launcher 單元測試與系統 `ruff` 均 PASS；此 fallback
  本次尚未啟用。
- **工具路徑經驗**：GlueMap micromamba env 沒有內建 `ruff`；改用系統
  `/home/cihcilab/.local/bin/ruff` 後檢查 PASS。這是工具路徑問題，非程式邏輯失敗。

### F8：S5 原始統計被 9 筆負深度 observation 汙染

- **症狀**：`compute_mean_reprojection_error()` 因 9 筆負深度 observation
  回傳約 `1e149`；同時 fixed BA 後內參有 `9.66e-5` 差異，連通性分母
  還誤含了刻意零 observation 的 pure-rotation images。
- **保全**：失敗模型放在
  `runs/target_site_v1/recovery/S5_failed_model_20260718_051617`，原始 log 為
  `recovery/s5_finalize_20260718_043103.log`。
- **修正**：在 BA 前後移除非正深度與 >8 px observations，刪除短於
  3 的 tracks，用手動 positive-depth pixel residual 重算，並在 BA 後逐參數
  復原 seed intrinsics。
- **驗證**：`test_finalize_edm_model.py` 8/8 PASS，`ruff` PASS；修復版
  S5 後來得到 1.522 px 與零 invalid observations。

### F9：原 G5.7/S6 正規化掩蓋 S06 錯誤次分量

- **首次警報**：G6.1–G6.4 全數 PASS，但 G5.7 三組橋接的 Sim3
  rotation 差為 60.16°、98.47°、147.75°，G5.7 FAIL。
- **稽核實作問題**：原工具用近乎一維的 corridor camera centers 估完整
  3D Sim3，繞路徑軸的旋轉本來就不可觀；而 forced image pairs 是視覺
  overlap，不是同步、同位置的 trajectory correspondences。
- **更嚴重的資料問題**：全圖 max–min span 為 27,503,643，主因是
  `S06_P1240124/000245.jpg` 零 observation 姿態被 BA 推至約 2,750 萬
  units 外。`000190.jpg`–`000199.jpg` 另形成有 658–848 observations、但
  離主圖約 983–987 units 的次分量。以全圖 max–min 正規化後，
  所有 overlap residual 都被壓到 `1e-8` 量級，導致 G6 假 PASS。
- **修正中**：finalizer 改為在 BA 前後只保留非 pure-rotation 影像的
  唯一最大 track component，G5.3 改為主分量 100% 且零非預期
  zero-observation registered image。S6 尺度改用相機中心至中位數中心
  的 95th-percentile robust span。
- **G5.7 正確替代**：新增 per-sequence 獨立 GLOMAP，再從已驗證
  cross-pair features 建立 3D↔3D correspondences，對兩個空間分離橋群各自
  RANSAC Sim3；這才能在未共享全域 frame 前比較 `T_AB`。
- **測試**：finalizer/seed 16 測試、geometry 4 測試、independent
  Sim3 3 測試全數 PASS；後續用 source-selected canonical runtime
  完整重跑 target-site suite，146/146 PASS，全部 tools/tests `ruff` PASS。
- **工具路徑經驗**：第一次 audit 從 `tools/` 工作目錄執行時用了
  相對 model path，因此找不到模型；另一次將實際檔名
  `forced_bridges.json` 誤寫成 `forced_bridges_manifest.json`。兩者皆在建圖
  前置路徑階段即失敗，未修改地圖；後續強制以 run-dir 為基準的
  完整路徑。

### F10：獨立 GLOMAP 無法開啟 COLMAP 4 `pose_priors` schema

- **症狀**：S02 獨立 mapper 在讀取 DB 後立即 `SIGABRT`，log 顯示
  `PrepareSQLStatements(): SQLite error: SQL logic error`。主程式 exit `-6`，
  尚未進入 reconstruction。
- **定位**：`PRAGMA integrity_check` 為 `ok`，104 images、2,713 two-view pairs
  都完整；相同 GLOMAP 連原始 production DB 也在同一行失敗，因此非
  sequence filter 破壞 DB。從 binary SQL strings 確認，此 GLOMAP 要求舊版
  `pose_priors(image_id, ...)`，但 pycolmap 4.0.4 DB 是
  `pose_priors(pose_prior_id, corr_data_id, ...)`。在 prepare 不存在的 `image_id`
  statement 時就失敗。
- **修正**：filtered audit DB 的 `pose_priors` 為空表；只在確認 row count
  為 0 後，將這張臨時表轉為 legacy schema。原始 production DB 不修改。
  如果日後表內有 pose prior，工具會 fail closed，不會靜默丟資料。
- **驗證**：DB filter regression test 同時檢查 image/pair 裁切與 schema
  轉換；independent Sim3 3/3 tests 與 `ruff` PASS，後啟動 retry。

### F11：S03↔S06 第二橋群缺少獨立 3D↔3D 支持

- **結果**：獨立子圖為 S02 104 images/52,703 points、S03 110/51,815、
  S05 173/86,391、S06 220/123,644。
- **已 PASS 骨幹**：
  - S02↔S06：兩橋群 83/85 與 227/257 RANSAC inliers，scale log
    delta `0.00244`、rotation `0.0955°`、translation/span `0.000192`。
  - S05↔S06：781/815 與 1,220/1,270 inliers，scale log delta
    `0.000339`、rotation `0.1037°`、translation/span `0.000241`。
- **失敗邊**：S03↔S06 第一橋群有 144 unique 3D↔3D correspondences，
  第二橋群只有 1，無法獨立解 Sim3。這不是容差太嚴，而是該邊不具
  兩個可觀的獨立鎖點，不可留在 trusted backbone。
- **final-map 影響量**：S03/S06 共觀 3,413 points（0.87% of map）、
  98,227 observations；其中 1,007 points 只有 S03/S06，2,406 另有
  S02 或 S05 支持。
- **fail-closed 修正**：不將闃檻改鬆。直接刪除所有 3,413 個
  S03/S06 共觀 tracks，用其餘幾何重做 fixed-intrinsics BA。後續
  G5.7 只信任兩條獨立 PASS 邊；S03↔S06 必須在 final map 仍為
  0 shared points 才可標記 `QUARANTINED`，否則仍 FAIL。
- **測試**：quarantine track selection、零殘留邊與至少兩條 trusted edge
  都有 regression tests；finalizer 11/11、independent audit 5/5，`ruff` PASS。

### F12：S7 validator 的 `numpy.bool_` 無法 JSON serialize

- **症狀**：MegaLoc 與 tracking bundle 已完整寫入，但 validator 在
  `json.dumps()` 報 `Object of type bool is not JSON serializable`；實際類型是
  NumPy 2 的 `numpy.bool_`。
- **根因**：`np.isfinite(...).all()` / `np.allclose()` 使 `descriptor_ok`、
  `covis_ok` 可能保留 NumPy scalar，而非內建 `bool`。
- **修正**：在 gate boundary 明確 `bool(...)`正規化；3/3 validator tests
  與 `ruff` PASS。重跑後 G7.1–G7.3 全數 PASS，bundle 未重算也無汙染。

### F13：S8 誤用系統 Python，缺少 EDM `yacs`

- **症狀**：S8 讀完 model 與 16,928 covis pairs，但在建立
  `EDMMatcher` 時立即 `ModuleNotFoundError: No module named 'yacs'`。當時尚未
  產生 EDM matches，final model/bundles 未被修改。
- **根因**：`/usr/bin/python3` 有 torch 2.11/cu128、pycolmap 4.0.4 與
  hloc，但沒有 EDM config dependency `yacs`。
- **修正**：改用專案已存在的 `EDM定位測試/env/.venv_edm`；實測
  torch `2.11.0+cu128`、CUDA 12.8、RTX 5090 available、pycolmap 4.0.4、
  hloc 1.5、yacs 0.1.8、kornia 0.8.3。以 `--overwrite-edm` 從配對階段
  乾淨重跑。

### F14：hloc 要求 pose-only reference 也有 feature group

- **已成功的前半段**：EDM 16,928 pairs 在 140 秒完成，速度約
  120.5 pairs/s；建立 53,667,670 keypoints、8,985,379 anchor cells與
  44,682,291 matches。
- **失敗點**：hloc 的 `import_features()` 遍歷 reference model 所有
  1,390 registered images，但 S7 正確排除了 7 張 pure-rotation refs，因此
  `feats-edm.h5` 沒有這些 group，第一個缺檔為
  `S01_ABrot/000373.jpg`。
- **修正**：不把 pure-rotation 加回 EDM，也不修改 final model。在
  `work/active_reference_model` 寫一份三角化專用副本，只 deregister
  缺少 EDM features 的 pose-only images，並強制
  `num_reg_images == len(active refs)`。
- **快取復用**：保留並重用已完成的 `edm_pairs.h5`，不再跑 16,928
  對 GPU matching；失敗的部分 triangulation model 已移至 recovery。
- **測試**：新增 pose-only deregistration regression test，1/1 PASS，`ruff`
  PASS。

## 進行中

- S8 復用 EDM matches、active-reference fixed-pose triangulation retry：PID 56427；
  resource guard：PID 56519。
- 依使用者指示，長時間作業每 20 分鐘人工檢查一次，階段完成或
  錯誤時才提早回報；resource guard 仍持續 fail-closed 保護。
- 後續依序：S5.7 independent per-sequence Sim3 → S6 robust ghost audit →
  S7 MegaLoc/covis/yaw → S8 EDM re-triangulation → S9 P123/P126 held-out
  localization。

## 待補最終結果

- S5 注冊率、reprojection error、連通性、內參差、三角化角、pure-rotation observation。
- S5.7 兩個獨立橋群的 Sim3 scale/rotation/translation/residual 一致性。
- S6 正反向與 3/5/6 月軌跡重疊、per-sequence reprojection、duplicate-structure audit。
- S7 MegaLoc descriptor 覆蓋率、covis index、yaw bimodality。
- S8 EDM anchored cells/ref 與三角化 active refs。
- S9 P123/P126 localization rate、inliers、軌跡連續與 ghost teleport。
