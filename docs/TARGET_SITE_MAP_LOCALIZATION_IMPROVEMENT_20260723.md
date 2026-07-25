# target_site 建圖／定位改良執行紀錄

更新日期：2026-07-24（Asia/Taipei）

## 不可變條件

- 核心模型固定為
  `建圖/target_site/runs/target_site_v1/final_model`；任何 sidecar 實驗不得
  覆寫相機、內參、既有 point IDs 或 tracks。
- build data 為 7 條既有建圖 sequence；`P1230123.MP4`、
  `P1260126.MP4` 永遠只作 held-out query。
- production 主線維持 EDM balanced；重型方法只在連續 WEAK 或 LOST 時
  擔任救援，除非另行批准更換主線。
- 沒有外部 pose ground truth，不把 inlier、軌跡平滑或跨方法一致性冒充
  公尺／公分級絕對精度。
- 福和橋已停止，不再用於本輪跨場域實驗；football_field 只作靜態回歸。

## 凍結基準

- GlueMap/GLOMAP：1390/1414 registered，其中 1383 張可定位、7 張
  pure-rotation；388,353 points；mean reprojection 1.443595993 px。
- 固定內參、獨立 Sim3、G6.1–G6.4、S0–S9 release gates 全部通過。
- EDM balanced profile：
  `EDM定位測試/deploy/profiles/balanced_rtx5060_candidate.json`
  （原始檔 SHA-256
  `15523cc131704df4b20faf671cf6a3bf62d5686b28f3bb792d68d5bd050d374d`；
  canonical JSON digest
  `193ac4738642ba5cd25fc2ddffaab5caabd4a2b6dcf4b01b43c6b86b9c008d89`）。
- P123：2595/2684 = 96.684%，median latency 19.168 ms。
- P126：3402/3408 = 99.824%，median latency 19.129 ms。
- normal LOO：312/312；P124 hard LOO：1/647。

## 實驗狀態

| ID | 實驗 | 狀態 | 備註 |
|---|---|---|---|
| B00 | baseline、split、hash 凍結 | 已完成 | 沿用 frozen manifest 與 sealed provenance |
| B01 | matcher→map conversion funnel | 已完成 | runtime 已回報 raw/direction/anchor conversion |
| B02 | MVS workspace audit | 已完成 | 1383/1383 geometric depth 與 normal；已修正遞迴計數缺陷 |
| M01/M02 | GLOMAP DB-reuse quality/coverage | terminal stopped | 此 ledger 項以封存的 DB-reuse attempts v4/v5/v7 對應；v4 retriangulation rig-reference abort，v5/v7 均 1389 < 1390 raw-model gate，無下游建圖／定位 |
| M03 | MVS sparse-first EDM grid overlay | 已完成／不升級主地圖 | 原 anchor 0 變更；full 與三個 consistency variant 均已封存 |
| M04 | Dense-SfM augmentation-only | 停止 | 既有 Dense-SfM replacement/refinement 與本輪 MVS all-frame 都未通過 |
| M05/M06 | GGPT tiles／GGPT+MVS consensus | terminal scientific fail | blind geometry QA（absolute reprojection/per-ID world/spread/secondary ghost）廣泛失敗；extent ratio 本身通過，但不進 map/EDM/localization |
| M07 | SplatHLoc FGS map | terminal blocked | 缺 target-site Feature Gaussian scene assets、MixVPR/masking/runtime 資產與 deployment license；屬獨立約 30k iterations 的地圖工作流 |
| M08 | R-SCoRe learned map | terminal denied | frozen protocol 未完成且 checkpoint 損毀、無有效模型；禁止擴展為三 seeds／full-site |
| M09 | ACE mode／GLACE learned localization | ACE mode 已完成、research-only、不升級；GLACE global-feature 暫時 blocked | ACE mode (`--global_feat False`) 有獨立、sealed LOO artifact；不能泛化成完整 GLACE。R2Former 官方權重遭 Drive quota block，故 GLACE global-feature 尚未開始、不使用替代 descriptor |
| L00 | EDM balanced control | 已完成 | 唯一 production baseline |
| L02–L04 | EDM + dense overlays | 已完成小樣本篩選 | all-frame 退步；output-only rescue 有小幅 hard 改善但未升 production |
| L05 | SURE frozen configuration | terminal no-rerun | 凍結證據為 <30 FPS，拒絕重新跑 |
| L06 | SLiM single-pair adapter | terminal blocked | SM120 無官方可用 runtime；pair identity 也不是 EDM grid localization contract |
| L07 | JamMa frozen configuration | terminal no-rerun | 凍結證據為 hard 0/60，且沒有 JamMa-specific 2D→3D map |
| L08/L09 | DeViLoc／ImLoc-style depth lift | terminal blocked | DeViLoc D0 source-pin/checkpoint contract 未成立（D1 5/5；D2–D4 未跑）；截至檢查日，current HLoc 未找到可重現 ImLoc 官方完整 implementation |
| L10 | SplatHLoc rescue | terminal closed with M07 blocker | 無 scene assets／license 前不能觸發 |
| L11 | R-SCoRe standalone/rescue | terminal closed with M08 | frozen protocol 明確禁止擴展至三 seeds／full-site |

## MVS sparse-first 實測結論

- 原始 EDM anchors：5,566,486；full MVS 只填 NaN，新增 1,746,321，
  對既有 anchors、reference images、global descriptors、centers、yaws、covis
  的變更都是 0。
- 最嚴格保留版本為 `edm_mvs_d100_r2_x002.pt`：新增 615,273 anchors，
  最終 6,181,759；SHA-256
  `3a9b9930ae0c23a39adbb5e756061d5da03e20a9401de4d97c29ea08d7b85e68`。
- all-frame full MVS 在 normal 60-frame window 雖把 anchor conversion
  0.816 提到 0.930，PnP inlier/anchor 卻從 0.993 降到 0.500；
  hard window 從 23/60 降到 16/60。因此稠密點不能直接取代原 EDM LUT。
- 三個 consistency variant 在 hard window 只有 15–16/60，也全部拒絕。

## 救援式整合實測

救援 arbiter 的固定規則：EDM 已成功時不呼叫；只在 WEAK/LOST；至少
30 inliers、median reprojection ≤5 px、cheirality ≥0.95、合法旋轉、
相對上個 EDM pose 位移 ≤0.008 map-unit；候選 pose 不與 EDM 混合。
預設為 output-only，避免 sidecar pose 改變 EDM 內部狀態。

| hard 60-frame 版本 | localized | wall FPS (RTX 5090) | tracker p50/p95 | 結論 |
|---|---:|---:|---:|---|
| EDM balanced baseline | 23/60 | 41.23 | 12.87/45.29 ms | production control |
| MVS rescue topk=3、每幀 | 27/60 | 24.66 | 31.32/84.37 ms | 精度較高、速度失敗 |
| MVS rescue topk=3、cooldown=3 | 26/60 | 29.13 | 29.25/55.73 ms | 仍低於 30 FPS |
| MVS rescue topk=1、cooldown=3 | 25/60 | 34.29 | 22.50/52.64 ms | 小窗口結果；full 3-seed 結案為 `REJECTED_NO_PROMOTION` |

`topk=1/cooldown=3` 共嘗試 15 次、接受 2 次；兩次均使用主 EDM 當幀
reference `S07_P1250125/000099.jpg`，分別有 37/31 inliers，沒有超過
0.008 的 pose jump。normal 60-frame control 為 60/60；沒有觸發救援。
hard longest LOST run 仍是 7 frames；後續 full 3-seed shadow 完成後，V009
在每 seed 仍為 15→15，故完整結果是 `REJECTED_NO_PROMOTION`。RTX5060
的完整直接驗證亦未封存，故不更新 production profile。

## 研究後端可執行狀態

- `backend_preflight.json` 是啟動重型工作前的 fail-closed gate。
- GGPT：FP32 P2B dense-only run 已完成，但 blind geometry QA 與 secondary
  ghost gate fail（extent ratio 本身 pass）；sidecar 不得改動 GlueMap、EDM
  或 localization。
- SLiM：SM120 無官方 CUDA 12 runtime，且官方 pair matching 不等同於
  2D→3D/PnP localization；兩個條件同時補齊前不重開。
- DeViLoc：D0 是 TopicFM source-pin/checkpoint contract mismatch，不是
  DeViLoc 方法失敗；D1 fixed-intrinsics adapter 的 5 個測試通過，因 D0
  未成立而禁止 D2–D4 forward。
- R-SCoRe/scrstudio：frozen protocol 的 64/4 smoke 不可 promotion；後續
  5k attempt 未完成且無有效模型，終止且不准擴展至 full-site 或三 seeds。
- ACE mode：pinned GLACE source 的 `--global_feat False` LOO 實驗已完成，僅限
  non-commercial research、不進 EDM／production；權威終止檔是
  `EXP/research_backends/ace_target_site_loo_20260724_v1/terminal_audit_v2.json`
  （`9403137c1626a7c9b451cf6730a9314d797feb9e6f33408ae133cea45ee0fb47`）。
  這不是完整 GLACE global-feature 結果，也不外推 R-SCoRe 結論。
- GLACE global-feature：獨立 append-only preflight 已封存（source `e704a8`、官方
  R2Former Drive ID 固定），但官方檔案遇 Google Drive quota；在 exact weight
  strict-load 成功前，256D features、smoke、30k science、評估和 P123/P126 都不跑，
  不得以任何替代 global descriptor 或 ACE mode 冒充。
- SplatHLoc：需要獨立 Feature Gaussian scene map/MixVPR（約 30k iterations
  的場域訓練工作流）與 deployment license；不是 GlueMap 的 in-place refinement。
- ImLoc：截至本次檢查日，current HLoc 未找到可驗證、可重現的 ImLoc 官方
  source、weights、config 和 depth-map/GPU LO-RANSAC entrypoint 組合；
  不把一般 depth lift 冒稱為 ImLoc。

## 2026-07-24 terminal closeout

本節是本輪的終止帳本。`EXP` 固定指
`建圖/target_site/runs/target_site_v1/experiments/map_localization_improvement_20260723`；
`RUN` 固定指 `建圖/target_site/runs/target_site_v1`。下列 SHA-256 對應原始
receipt/terminal JSON，而非由本文件重算的摘要。任何「median 改善」或單幀／
代理指標均不構成定位精度宣稱。

### Dense-MVS 三 seed offline shadow

- aggregate receipt：`EXP/benchmarks/shadow_full_aggregate_balanced_20260723_v3/aggregate.json`
  — `f86bcc94dd6292d745d16bc041b5b6a67e30c09b8cee2ec2ccf22baadd32dc1f`，
  `failed`；V009 要求至少兩個 seed 嚴格縮短最長 unlocalized segment，未達成。
- 三個 seed receipt 分別是
  `EXP/benchmarks/shadow_full_offline_balanced_s17_20260723_warm_v3/receipt.json`
  — `7b37f972b3d9530c4da4a090457574681600a071a93d19de83682c6675f8e84e`，
  `EXP/benchmarks/shadow_full_offline_balanced_s47_20260723_warm_v3/receipt.json`
  — `1b5d305fd183d3b0f643c84b2a2fd3a8662cdba4fd98b74b52347cafd8e52201`，以及
  `EXP/benchmarks/shadow_full_offline_balanced_s83_20260723_warm_v3/receipt.json`
  — `aa0c0400acdb56e224d8797b9d88a808695239493e530141e878e2ab89d65ebd`。
- 六個 result 共 18,276 scored：primary_ok 17,875、output_ok 17,893，
  僅 +18（0.09849 percentage point）；18 個 fixes 全是 WEAK，LOST fixes=0。
  每 seed 最長 unlocalized 均 15→15，故 quality gate fail。模擬 online FPS
  61.804–65.948、warmed p95 18.071–29.863 ms 只說明此 shadow 配置的吞吐，
  不足以抵銷 stability gate，也不等同正式 RTX5060 localization 結果。
- 結論：`REJECTED_NO_PROMOTION`。若沒有改變 V009 的已授權科學合約，
  不重跑同一配置。

### GlueMap/GLOMAP DB-reuse retune

- v4：`RUN/experiments/glomap_db_reuse_retune_20260723_v4/terminal_status.json`
  — `fe1642a5712f8335c9bce16c0331682ddd2398cafc5c0b8dee303aae08475950`，
  `STOP_NONZERO_MAPPER_AND_NONEMPTY_WAL`；retriangulation 時 rig reference
  sensor mismatch abort，沒有可審核 raw model。
- v5：`RUN/experiments/glomap_db_reuse_retune_20260723_v5/terminal_status.json`
  — `5a8a64fa2d8f88bbe12ce7bf4bfdab66a2b80d263e82e640de433069049eb41b`，
  `STOPPED_RAW_MODEL_GATE`；mapper exit 0 但 registered_images=1389 < 1390，
  且 frozen audit API 不相容；所有下游停止。
- v7：`RUN/experiments/glomap_db_reuse_retune_20260723_v7/terminal_status.json`
  — `292cd359b8594ea9da4faf8584b557e2362e4fd49833bb4f8114e1e7db15d72e`，
  `STOPPED_RAW_MODEL_REGISTERED_IMAGE_FLOOR`；完整 attestation 後仍為
  1389 < 1390，禁止 finalize、Sim3、ghost gate 與 localization。
- 重試條件：須先獲新授權，針對 upstream rig/frame-ID mismatch 以有
  retriangulation 的全新隔離實驗修正；不可放寬 1390 門檻，亦不可把
  skip-retriangulation workaround 當作 root-equivalent。

### 幾何與 learned-localization 後端

- GGPT v10 terminal receipt：
  `EXP/research_backends/ggpt_target_site_tile_20260724_v10/terminal_receipt.json`
  — `44a1ece8cd0b3f399859f02152497608fddea2eef9cbc9c4f6d218777ac73b78`，
  `SCIENTIFIC_FAIL_GHOST_EXTENT`，不可重試。相應 science report
  `EXP/research_backends/ggpt_target_site_tile_20260724_v10/science_report.json`
  — `97097ab629689349af864889dd3106eff12e4f6b8f8c4cd0d35b26f805e32504`：
  `EXP/research_backends/ggpt_target_site_tile_20260724_v10/qa_report.json`
  — `4fe9f7fde1ce7786b7b389395b1f316840f0c222f2eafe8158261b64005359a7`：
  blind fixed-camera reprojection q50 5.042→3.116，但 absolute gate 為
  q50 ≤2、q95 ≤8，q95 17.038→44.785；per-ID world residual q50
  0.11208→0.06958（限值 ≤0.04508）、q95 13.15946→11.45850（限值 ≤0.22539）；
  multi-view spread q50 0.02499→0.02282、q95 0.8437→1.4177（限值 ≤0.22539）。
  dense radius q99−q01=12.509 的適用 gate 是 dense/sparse extent ratio；
  ratio=0.19155 ≤3 本身通過，但 secondary mode fail。
  因此是 blind geometry QA 的廣泛 fail；這只是一個 P2B dense-only
  geometry evidence，沒有 EDM 或定位速度／精度提升主張。
- R-SCoRe bounded smoke receipt：
  `EXP/research_backends/rscore_port_preflight_20260724_v1/smoke_slice/receipt_p2_smoke_20260724.json`
  — `d786faab68b05a7d944beaf2657e2ae40bc60d34b30a3dcb9d746a1721c81497`；
  是 frozen protocol 的 non-promotable diagnostic（rotation 10° 的 4/4
  為 0/4；60-step in-sample 記錄 median translation raw-map=1.73993、
  rotation=136.9248°），不是一個可與 EDM 比較的 localization-accuracy
  結論。
- R-SCoRe 5k terminal receipt：
  `EXP/research_backends/rscore_convergence_20260724_v1/terminal_receipt_5k_20260724.json`
  — `15f4c271015bb0ccd02f322de7c72c5ce5159d39165476e12e71899acc25ab3e`，
  `TERMINAL_DENY_FULL_RSCORE`。predeclared 5000-step run 在 TB iteration
  1499 前結束、無 controller report，唯一 `head.pt` 為 truncated archive；
  提前結束原因未知，最大觀測 VRAM 9954 MiB 不能推斷原因或成功完成。此為
  frozen protocol 的 incomplete/corrupt-checkpoint terminal，沒有有效模型可做
  定位評估；禁止 tuning、重試、三 seeds、full-site、held-out decode 或 production map。
- DeViLoc terminal receipt：
  `EXP/research_backends/deviloc_target_site_pose_smoke_20260724_v1/receipts/terminal_receipt.json`
  — `08f700d1221b4caf338b75fe5e18a08fb0ac3eafc02a44738d24fe6f576b05d7`；
  D0 `D0_STRICT_TOPICFM_SOURCE_CHECKPOINT_COMPATIBILITY` 是 source-pin／
  checkpoint contract mismatch：`aaai23_ver` README 對應自己的 `model_best`，
  而 current master 才把 Drive fast ID 指給 TopicFM-fast，未證明兩者為
  可嚴格載入的官方 pair。D1 adapter 5/5 pass；D2–D4 未跑，未 decode query、
  未 GPU forward、未產生 pose。因此沒有任何 DeViLoc localization、FPS 或
  pose-accuracy 結論，也不可將其稱為 method failure。僅在明確指定官方相容
  source+checkpoint contract 後才能新開 receipt。

### ACE mode of pinned GLACE（research-only；不升級）

- 權威結果為
  `EXP/research_backends/ace_target_site_loo_20260724_v1/terminal_audit_v2.json`
  — `9403137c1626a7c9b451cf6730a9314d797feb9e6f33408ae133cea45ee0fb47`；原
  `terminal_audit.json` 的 identity=false 是比較器問題，不能用它判定 production
  被修改。ACE mode 是 pinned GLACE source 的 `--global_feat False`，不是完整
  GLACE global-feature/R2Former 實驗；每個 full run 只可稱為請求 30,000 次
  training-loop step calls，未封存 30,000 次成功 AdamW updates。
- LOO pseudo-GT 診斷（非 external absolute accuracy）：normal S04，252 frames，
  2°/5cm=31.349%、5°/10cm=42.063%、10°/1m=75.000%，median
  0.3332°/0.18566 m；hard S06，221 frames，22.172%、23.529%、28.507%，median
  40.1677°/1.97228 m。hard 結果不具取代 EDM 條件。
- `pose_success=100%` 僅表示 finite estimated error 且 DSAC inliers >0，不能當作
  100% 定位精度或成功率。hard median inliers=31，低於 balanced EDM `track_min_inliers=50`。
  43.404/60.830 ms 是 RTX5090 上 DataLoader yield 之後的 batch timer，涵蓋 transfer、
  inference、DSAC/PnP、pose/error 和 logging；不含 decode、resize、DataLoader wait、
  model load、startup，故欄名 `end_to_end_latency_ms` 不可當端到端延遲，也不可直接與
  EDM 影片 wall-time 比較。
- P123/P126 為 `NOT_RUN`。release 已有 1280×720 PINHOLE
  K=[931.2057783503648, 931.2057783503648, 640, 360]；舊 D9 所稱「沒有固定內參」錯誤，
  但 sealed D9 本身不覆寫。真正限制是沒有 external/frozen-LOO video pose GT、不能報
  accuracy，且 hard LOO quality gate 差；任何無 GT 的 inlier/funnel/trajectory-only probe
  需另行授權，且不得改 production。

### 已關閉的 matcher／scene-map 前置條件

- JamMa/SURE：`EXP/receipts/jamma_sure_existing_evidence_terminal_20260724_v1.json`
  — `019ab905cbd25728e60143d2376f15e0a1c139def9c02124fa754f40932a0ae9`，
  `ALREADY_COMPLETED_REJECTED_NO_RERUN`；SURE <30 FPS，JamMa hard 60-frame
  0/60。凍結證據只關閉兩者的已拒絕配置，不泛化為專門地圖永遠無法幫助。
- ImLoc：`EXP/receipts/imloc_2601_04185_implementation_preflight_terminal_20260724_v1.json`
  — `7c4aee056645e7fbdcf2cb6963e36de71f8b5c81313fb1383e5de9c792a3a5b1`，
  `BLOCKED_OFFICIAL_IMPLEMENTATION_ABSENT`。需具可重現且明確授權的官方
  source、weights、config 和 depth-map/GPU LO-RANSAC entrypoint 才能新建 receipt。
- SLiM：`EXP/receipts/slim_target_site_official_port_preflight_terminal_20260724_v1.json`
  — `4c509facd2634d21350c7e5e516d806c145b45fd21f326851672649e81835a30`，
  `BLOCKED_OFFICIAL_RUNTIME_UNSUPPORTED_SM120`。另須有效的 SLiM 專用
  2D→3D+PnP contract；single pair 的 match identity 本身不是 localization。
- SplatHLoc：`EXP/receipts/splathloc_target_site_feasibility_terminal_20260724_v1.json`
  — `3ca6e4befe6c49c9264a528096dd80c556da39db780d661953ba203af9e76fc8`，
  `BLOCKED_SCENE_MAP_ASSET_AND_DEPLOYMENT_LICENSE_BOUNDARY`。必須先有
  deployment license、target-site Feature Gaussian scene map、MixVPR、
  masking/preprocess 和相容 runtime；因此約 30k iterations 的獨立場域地圖仍是
  未獲授權／未具資產的 blocker，不能把它當作 GlueMap 的後處理。

### 尚未解除的外部條件

- RTX5060 上的完整 P123/P126、normal/hard LOO 直接驗證尚未取得可封存的
  artifact；因此仍是 production promotion 的外部 blocker。上述任何 RTX5090、
  simulated-online 或 proxy timing 都不得替代它。
- 所有未啟動的後續工作都受各 receipt 中的 reopen condition 約束；它們是
  已關閉的 stage，而不是可暗中視為「所有 empirical 實驗都已完成」。

## Promotion gates

- normal LOO 必須維持 312/312。
- P123 至少 2644/2684；P126 不低於 3402/3408。
- P123 最長 LOST run 至少縮短 50%；hard LOO 至少提升至 10/647。
- RTX5060 TRACK p95 ≤33.3 ms、持續 ≥30 FPS、peak VRAM ≤7600 MiB。
- sidecar 必須保持核心模型 hash；新 GLOMAP map 必須重跑全部 geometry gates。
- 每個實驗都須保存 raw matches → anchored matches → unique 3D → PnP
  inliers，避免用 raw match 數量誤判。

## 資源規則

- GGPT、MVS、SplatHLoc、R-SCoRe 與大型 BA 不並行。
- RAM 使用超過 80% 時不啟動另一個重型工作；正式工作記錄 RAM、swap、
  VRAM 與 wall time。
- 現有 93 GiB MVS workspace 只讀重用，不重跑 PatchMatch；production
  package 不直接攜帶 depth/normal 工作檔。

## 本輪決策

**唯一 production 組合仍是 GlueMap map + EDM balanced。** final model、
EDM bundle、profile 均未被任何實驗覆寫；本輪沒有候選可 promotion。
歷史上的 MVS `topk=1/cooldown=3/output-only` 只保留作封存的
research evidence，不是下一階段預設候選，也不得包裝成可選 rescue sidecar。

ACE mode 已完成但只屬 research-only/no-promotion；GLACE global-feature 有獨立
append-only artifact，正受官方 R2Former Drive quota 外部 blocker，不能以 ACE mode
或替代 descriptor 取代。RTX5060 直接驗證亦仍是外部硬體 blocker。其餘重開需要新的
官方資產／license／相容性合約／明確授權。不得以
較好的 median、單幀結果、match count、proxy 或 workaround 取代 frozen
accuracy、stability、full-corpus 與 RTX5060 promotion gates。

> **2026-07-25 更新：本節上方兩個 blocker 敘述已過期。** GLACE 的 R2Former
> Drive quota 已解除、SplatHLoc 的 30k 場景訓練已實際執行完畢。詳見下一節。
> production 決策本身未變：仍是 GlueMap map + EDM balanced，無候選可 promotion。

## 2026-07-25 續辦紀錄（成功與失敗並列）

本節補記 07-24 終止帳本之後實際發生的事。失敗與成功同樣入帳，避免下一位
操作者重走同樣的死路。

### Phase-B true-LOO 已結案

- S04 component v2 終局：
  `RUN/experiments/target_site_true_loo_phaseB_S04_components_20260725_v2/receipts/P13_final_index.json`
  — `CLOSED_NOT_LOCALIZATION_ELIGIBLE_P10_FAIL`。1144 影像的 sealed accepted-TVG
  component 已在固定 K 下獨立重建並通過 raw map gates，但未過 frozen fit-only Sim3 gate。
- 因此 S04 不再是「topology 待決」，而是與 S06 一樣 closed；所有下游方法
  （EDM／GGPT／SURE／JamMa／SLiM／DeViLoc／ACE）在 Phase-B 路徑上維持
  `NOT_EXECUTED_UPSTREAM_MAP_GATE`。

### GLACE global-feature：已執行完，research-only

- 終局 receipt：
  `EXP/research_backends/glace_target_site_loo_20260724_v1/receipts/GLACE_R2Former_target_site_terminal_20260725_v1.json`
  — `COMPLETED_RESEARCH_ONLY_NO_PROMOTION`。官方 R2Former 權重已下載並通過
  NumPy-safe strict load，normal/hard 兩個 head 都完成訓練與評估。
- reference-images-excluded pseudo-GT（非 external absolute accuracy）：normal S04
  252 frames，median 0.4966°/0.36709 m、10°/1m 60.714%；hard S06 221 frames，
  median 57.700°/1.74976 m、10°/1m 21.719%。hard 明顯不具取代 EDM 條件。
- `method_decision`：維持 sealed GlueMap + EDM balanced，不 promote GLACE head。
- 時間欄位仍不可當端到端延遲：`glace_core_latency_ms` 61.514 從 DataLoader
  yield 之後起算，且不含 R2Former query descriptor 抽取（features.npy 為預先計算）。

### SplatHLoc：30k 訓練完成，D5 localizer preflight 於第四次嘗試才跑通

- D4 官方預設 30k train-only 已封存：
  `EXP/research_backends/splathloc_target_site_loo_20260725_v2/receipts/D4f_..._terminal_provenance_20260725_v2.json`
  — `PASS_RESEARCH_PSEUDOGT_OFFICIAL30K_SEALED`。這只是地圖訓練結果，沒有任何
  定位／速度／true-LOO／promotion 主張。
- D5 的四次嘗試全部保留為證據，lineage 收據：
  `.../receipts/D5_attempt_lineage_20260725_v1.json`（+ v4 outcome addendum）。

  | 嘗試 | 結果 | 根因 |
  |---|---|---|
  | v1 | FAIL_CUDA_TOOLCHAIN | 未釘 `CUDA_HOME`，torch 抓到 `/usr/bin/nvcc` 12.0 → `Unsupported gpu architecture 'compute_120'` |
  | v2 | FAIL_CUDA_TOOLCHAIN | `CUDA_HOME` 指向只有 nvcc、沒有 `targets/x86_64-linux/include` 的環境 → `cuda_runtime.h: No such file` |
  | v3 | FAIL_JAMMA_JEGO_PARITY | CUDA 已修好（extension 186.97 s 編成），改撞 `scan_jego` 的 1760 vs 1840 |
  | v4 | PASS（train-anchor 技術性 preflight） | 以 reflect-pad 640×368 + padding-strip masking 修正 |

- **JEGO parity 的通則**：`scan_jego` 用 `h_8 // 2` 配置 `xs`，卻寫入
  `ceil(h_8 / 2)` 列，因此 JamMa 要求影像高度可被 16 整除。target_site 是
  1280×720，SplatHLoc 的 `longest_edge=640` 給出 640×360 → h_8=45（奇數）、
  w_8=80，22×80=1760 對上 23×80=1840。官方 JamMa 自己的 dataloader 會 pad 成
  832×832（h_8=104）所以碰不到；SplatHLoc 的 `splathloc.py` 直接餵 resize 結果，
  **4:3 可過、16:9 必炸**。這是上游對長寬比的限制，不是 probe 的缺陷。
- 修法採 reflect-pad（operator 於 07-25 在三個選項中選定），理由是 padding 正是
  官方 JamMa 的慣例，可保留官方 `longest_edge=640` 的尺度語意；被否決的是
  `longest_edge=768`／`512`（等於改掉官方預設解析度）。padding 只加在下／右，
  原點與主點不變，故不需調整內參。
- v4 實測：JamMa coarse 3044／fine 3040，**落在 padding 區而被濾掉的為 0**
  （BORDER_RM=2 本來就會丟掉 padded 的最下兩列）；peak GPU 3.76 GiB；
  torch 2.7.1+cu128 / CUDA 12.8 / RTX 5090；network blocked attempts=0。
- 邊界效應：BORDER_RM=2 會連帶丟掉真實影像 y∈[352,360) 的 coarse candidate；
  未 padding 時則是丟 y∈[344,360)。
- **v4 只是 train-anchor 技術性 preflight**：它證明 runtime 能端到端跑完，
  其中 PnP 697 inliers 是 LightGlue 路徑的自洽檢查，不是 Normal20 定位結果，
  更不是精度主張。下一關仍是 frozen D4a/D4b 的 Normal20 localizer 合約。

### 仍未解除的外部條件（未變）

- RTX5060 上的完整 P123/P126 與 normal/hard LOO 直接驗證，仍是 production
  promotion 的唯一外部 blocker；RTX5090 與 simulated-online 數字都不可替代。
- GLOMAP DB-reuse 的 1390 registered-image floor 未放寬；v8 亦 fail
  （1389，且 intrinsics delta 9.658953104008106e-05 > 1e-06）。
