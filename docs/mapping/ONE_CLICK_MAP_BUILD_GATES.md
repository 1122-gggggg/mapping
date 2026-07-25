# One-Click Localizable Map Gates

這份文件定義「外行人一鍵建圖」的通過條件。目標不是只產生漂亮點雲，而是產生可定位、可驗證、可交付的地圖包。

機器可讀總 gate 由 `建圖/pipeline/stage_gate_contract.py` 產生：

```bash
python3 建圖/pipeline/stage_gate_contract.py --run-dir 建圖/runs/<site_run>
```

輸出：

- `BUILD_GATE_SUMMARY.json`
- `BUILD_GATE_SUMMARY.md`

完整 `operator_pipeline.py build` / `build_pipeline.py` 跑到 `report` 階段時，會自動產生這份總結；只有 `--handoff-profile field` 的總 gate 通過才會更新 `建圖/outputs/latest_build`。`--handoff-profile candidate` 通過時只更新 `建圖/outputs/latest_candidate_build`。

部署或交給非專家前，應把外部驗證證據一併納入：

```bash
python3 建圖/pipeline/stage_gate_contract.py \
  --run-dir 建圖/runs/<site_run> \
  --require-localization \
  --localization-json 定位/validation/eval_test_720p_v3_vs_base.json \
  --localization-json 定位/validation/eval_downloads_720p_v3_vs_base.json \
  --require-production \
  --production-json 定位/outputs/production_stream_replay.json \
  --require-package-verify \
  --package-verify-json package_verify.json \
  --require-system-verify \
  --system-verify-json system_verify_latest.json
```

## Promotion Rule

只有 `BUILD_GATE_SUMMARY.json.overall_ok == true` 才能交給非專家使用者。

如果只有 `point_cloud` 通過、但 `localization_bundle` 未通過，這只是 inspection-only 點雲，不是可定位地圖。

`operator_pipeline.py verify` / `tools/system_verify.py` 也會重新檢查 `建圖/outputs/latest_build/BUILD_GATE_SUMMARY.json`。正式 handoff 不能只看舊版 `overall_ok`；`preflight_quality`、`frame_motion_quality`、`intrinsics_manifest_quality`、`view_graph_quality`、`database_quality`、`sfm_reconstruction_quality`、`point_cloud_quality`、`localization_bundle`、`report_package_quality`、`holdout_localization`、`production_replay`、`package_verify`、`system_verify` 都必須標為 required，且狀態為 `PASS` 或 `READY`。

`build_pipeline.py --handoff-profile field` 會在沒有手動提供 validation/package/system verify 證據時自動做 evidence generation 與 two-pass promotion：

1. 多段 `--videos` 輸入時，預設保留最後 1 段作 validation，不送進建圖前端；其餘影片進入建圖。保留影片會在 `<run>/validation_videos/` 產生標準化 `.MP4` symlink，避免原始副檔名大小寫造成驗證程式掃不到。
2. 用保留影片跑 final-bundle 對 final-bundle 的 holdout localization，寫 `<run>/holdout_localization.json`。這是新場域的 absolute localization gate，不是舊/新地圖升級比較。
3. 用同一段保留影片跑 production tracker replay，寫 `<run>/production_stream_replay.json` 與 paired `.preflight.json`。
4. 跑 `tools/verify_package.py --json-out <run>/package_verify.json`。
5. 產生 `<run>/BUILD_GATE_SUMMARY.pre_system_verify.json`，此時只暫不要求 `system_verify` stage。
6. 用 pre-system summary 跑候選 `tools/system_verify.py --build-summary-json ... --json-out <run>/system_verify.json`。
7. 用 `<run>/system_verify.json` 重跑正式 `BUILD_GATE_SUMMARY.json`；只有正式 summary PASS 才更新 `latest_build`。

如果影片不足以保留 holdout，系統不會用建圖影片偽造 holdout evidence；正式 final gate 會因缺 localization / production replay evidence 失敗。
這個 validation 可行性會在昂貴建圖前先做 preflight；例如 field profile 只輸入單段影片且沒有 `--validation-videos` 或外部 JSON evidence 時，流程會直接停止並要求補 validation footage。

如需手動診斷同一流程，可直接用候選 summary 產生 system verify 證據，避免要求候選已經是正式 symlink：

```bash
python3 tools/system_verify.py \
  --build-summary-json 建圖/runs/<site_run>/BUILD_GATE_SUMMARY.json \
  --json-out 建圖/runs/<site_run>/system_verify.json \
  --md-out 建圖/runs/<site_run>/system_verify.md \
  --skip-runtime \
  --allow-blocked
```

這個候選模式只允許 summary 內嵌的 `system_verify` stage 暫時 pending，因為此命令正在產生那份證據；其他 build stages 不得失敗，且 field handoff stages 仍必須 required 且 PASS。`--allow-final-gate-fail` 只用於留下診斷 summary，不得讓失敗 build 更新正式 latest pointer。

## Gate Contract

| Stage | Required | Core Evidence | Hard Failure Examples | Operator Fix |
|---|---:|---|---|---|
| `preflight` | yes | `gates/preflight.json` | 影片不存在、工具缺失、GPU/磁碟不足、解析度/內參不明 | 修正輸入與環境後重跑，不進入昂貴前端 |
| `preflight_quality` | yes | same accepted preflight gate plus `preflight_report.json` | `preflight_report.json` 缺失、影片 metadata 缺失、影片不存在或太短、每段預期抽幀 < 15、target resolution 太低或不一致、核心工具缺失、disk < 50GB、GPU required 但未通過、相機內參不合理 | 先修正輸入影片、核心工具、GPU/磁碟與相機設定；不要把不可讀或低資訊影片送進抽幀/匹配 |
| `frame_quality` | yes | `gates/extract.json` or `gates/selection_motion_quality.json` | frame/motion gate 缺失或失敗 | 補拍有平移視差的影片，或調整抽幀/品質門檻 |
| `frame_motion_quality` | yes | same accepted frame/motion gate metrics | selected frames < 60、selected ratio < 65%、parallax/seed ratio < 65%、hover ratio > 5%、multi-video 沒 bridge frames、舊 extract 缺 motion_gate 或每段 kept 太少 | 重新拍攝有平移視差的慢速航線與 connector footage，不把 hover/pure-rotation 主導影格交給後段 |
| `intrinsics_manifest` | yes | `gates/manifest.json` or `gates/intrinsics.json` | manifest 缺失、解析度不一致、內參來源不明、undistort 決策未記錄 | 固化相機模型與解析度縮放，先做 holdout localization |
| `intrinsics_manifest_quality` | yes | `frame_manifest.json`, `map_intrinsics.json`, same accepted manifest/intrinsics gate | manifest frame count 不一致、缺 frame size、map intrinsics 缺 per-resolution/shared camera、camera model/params 不可解析、fx/fy 非正、principal point 超出影像、distorted source 缺 undistort/no-undistort 決策 | 重新產生 manifest/intrinsics；若 distortion vs firmware undistortion 不明，先做 intrinsics bake-off 並用 holdout localization 固化結論 |
| `pair_graph` | yes | `gates/pairs.json` or `gates/gluemap_pair_graph.json` | pair graph gate 缺失或失敗 | 增加重疊拍攝、提高 retrieval topk、補 bridge 視角 |
| `view_graph_quality` | yes | same accepted pair graph gate metrics | `pairs < max(500, frames*4)`、`largest_component_ratio < 90%`、`connected_components > 1`、`isolated_images > 0`、跨影片/跨序列 bridge pairs 為 0、關鍵 connectivity/bridge metrics 缺失 | 補 cross-video/cross-sequence bridge、提高 retrieval topk、重抽 connector footage；不要在斷裂 view graph 上直接跑昂貴 dense matching/SfM |
| `pair_filtering_optional` | no, but fail-if-present | `gates/doppelgangers.json`, `gates/verify_pairs.json` | Doppelgangers++ 移除所有跨影片邊、DMS 保留率過低 | 檢查重複結構/錯誤 retrieval，不盲目放寬 |
| `dense_matching_optional` | no, but fail-if-present | `gates/mvroma.json`, `gates/aggregate.json` | dense H5 空、match group 不可讀、aggregation 後 pair 為 0 | 重跑 dense matching 或從保留的 DB/H5 診斷 |
| `database` | yes | `gates/db.json` or `gates/gluemap_database.json` | DB 太小、image/keypoint/match/two_view_geometry 缺失 | 優先用既有 DB/H5 重建 DB，不急著重跑前端 |
| `database_quality` | yes | same accepted DB gate plus COLMAP SQLite DB | DB 檔缺失、缺 cameras/images/keypoints/matches/two_view_geometries 表、images < 60、DB images 與 frame manifest 不一致、有 image 沒 keypoints、keypoints 太少、match/two-view pair 不足、two_view_geometries 非零比例 < 90%、平均 two-view inliers 過低 | 先用保留的 DB/H5/NPZ 重建或修復 DB；若 two-view geometry 太弱，回頭檢查 pair verification / dense matching，不直接進 mapper |
| `sparse_model` | yes | `gates/dense_retriangulated_model.json`, `gates/glomap.json`, or `gates/gluemap_model.json` | 模型 gate 缺失或失敗、COLMAP/GLOMAP binaries 不完整 | 優先用既有 DB/H5 重建或掃 GLOMAP/BA/triangulation |
| `sfm_reconstruction_quality` | yes | same accepted sparse model gate metrics | `registered_images < 60`、`registered_ratio < 80%`、`points3D < 30000`、`points/registered < 500`、`mean_reprojection_error > 2px`、關鍵 metrics 缺失 | 先掃 GLOMAP/BA/triangulation；若 pair graph 夠連通但仍低品質，再用 LFOE/Doppelgangers++ 診斷 outlier/repeated-structure edges |
| `point_cloud` | yes | `gates/color.json` or `gates/rgb_ply.json`, `deploy/map_rgb*.ply` | PLY 缺失、點數過少、RGB 點數與 model points 不一致 | 重新 colorize 或檢查模型/PLY 對應 |
| `point_cloud_quality` | yes | same accepted point cloud gate plus PLY header | PLY 不可讀、`end_header` 缺失、缺 XYZ/RGB vertex properties、vertices < 10000、PLY < 128KB、binary PLY bytes 小於 header 宣告、vertices / points3D < 30%、gate 記錄的 vertices 與 PLY header 不一致 | 從 accepted sparse model 重新 colorize；不要交付 tiny/non-RGB/mismatched PLY |
| `localization_bundle` | yes | `gates/triangulate.json` or `gates/tracking.json`, `deploy/reloc_map_xfeat_*.pt` | 無 MegaLoc/XFeat bundle、refs=0、unique refs 不一致、`ref_global` 非 `[refs, 8448]`、tracking metadata 缺失、3D anchor 太少 | 重建 tracking/triangulated bundle，不能只交付點雲或空 bundle |
| `report_package` | yes | `gates/report.json` or build report | 缺少 config/report/stage timing/gate provenance | 補齊報告後才能封包 |
| `report_package_quality` | yes | same accepted report gate plus `build_report.json`, `build_config.json`, `stage_times.json`, operator Markdown, and `gates/*.json` | machine-readable report/config/timing 缺失、operator Markdown 沒輸出段落、必要 output keys 缺失、沒有任何 localization bundle output、輸出路徑缺失或空檔、latest report stage 不是 success、gate JSON 少於 8 個 | 重跑 report stage 或重新產生 provenance package；不要交付無法追溯參數、輸出與 gate 的 build |
| `holdout_localization` | deploy required | `--localization-json` compare result | 空 rows、success 低於門檻、old-ok/new-fail 退步、連續失敗過長 | 用未參與建圖的影片跑 `localize_pipeline.py --mode compare` |
| `production_replay` | deploy recommended / runtime required | `--production-json` replay result plus paired `.preflight.json`, optional `production_stream_frames_manifest.json` | preflight 失敗、缺 paired preflight、只有 preflight-only JSON、production state-machine success 低、latency 缺失或超標、success/n 不一致 | 優先用 `--query-video` 自動抽連續 replay frames；先跑 `--production-preflight-only`，通過後跑正式 replay |
| `package_verify` | deploy required | `package_verify.json` | 必要路徑缺失、JSON/py syntax 錯、symlink/permission 問題 | 跑 `tools/verify_package.py` 並修正封包 |
| `system_verify` | field required | `system_verify_latest.json` | required check 失敗；optional blocker 只記錄不擋 map package | 跑 `tools/system_verify.py`; 實飛前不要跳過 runtime |

## Recommended Strict Metrics

這些數字是預設安全線；每個場域可有 profile，但不得移除相同類型的檢查。

| Area | Recommended Gate |
|---|---|
| Input videos / preflight quality | `preflight_report.json` 必須存在；videos list 不得空白；每段影片要有 path/codec/resolution/fps/duration/nb_frames/target resolution/sanitized stem/expected extracted frames；duration >= 10s；raw fps >= 1；expected extracted frames >= 15；target resolution >= 640x480 且每段一致；核心工具 `ffmpeg`、`ffprobe`、`glomap`、`python_sfm`、`python_sfmdb` 必須可用；disk_free_gb >= 50；GPU required 時必須 ok；preflight camera intrinsics 必須通過焦距/主點 sanity check |
| Disk/GPU | 至少 50GB 可用空間；需要 GPU 的 stage 必須確認 CUDA 裝置可用 |
| Frame selection | gate 必須存在且通過；正式 handoff 也要求 `frame_motion_quality` 數值 gate |
| Frame motion quality | detailed selection gate: selected >= 60、selected ratio >= 65%、parallax/seed ratio >= 65%、hover ratio <= 5%、multi-group bridge_frames > 0、每組 >= 15 frames；legacy extract gate: frames >= 60、每組 kept >= 15、keep ratio >= 15%、若有 motion class 則 parallax/seed ratio >= 10%、hover ratio <= 50%；缺核心 metrics 視為失敗 |
| Intrinsics manifest quality | `frame_manifest.json.total_frames` >= 60；若有 `frames` list 必須與 total 一致且每張有 name/width/height；`map_intrinsics.json` 必須提供 per-resolution intrinsics 或 shared camera；支援 PINHOLE/SIMPLE_PINHOLE/SIMPLE_RADIAL/OPENCV/FULL_OPENCV；fx/fy > 0；cx/cy 在影像範圍內；有 distorted source 時必須明確記錄 undistort 或 no-undistort 決策 |
| Pair graph | gate 必須存在且通過；正式 handoff 也要求 `view_graph_quality` 數值 gate |
| View graph quality | pairs >= max(500, frames * 4)；connected components <= 1；largest component >= 90%；若有 isolated image metric 必須為 0；跨影片/跨序列 bridge pairs > 0；缺 connectivity/bridge 核心 metric 視為失敗 |
| Doppelgangers++ | retention >= 30%；不得清掉所有 cross-video/cross-direction pair |
| Dense match | dense groups / input pairs >= 85%；missing dense ratio <= 1% |
| DB | DB > 10MB；DB image count == feature groups；不得有無 keypoint image；正式 handoff 也要求 `database_quality` 數值 gate |
| Database quality | COLMAP SQLite DB 必須可開啟；必須有 cameras/images/keypoints/matches/two_view_geometries；images >= 60 且等於 frame_manifest total；所有 images 必須有 keypoints；min keypoints/image >= 50；avg keypoints/image >= 500；matches 與 two_view_geometries pair 數 >= max(500, images*4)；matches/two-view 非零比例 >= 90%；avg two-view inliers >= 30 |
| Sparse model | model gate 必須存在且通過；正式 handoff 也要求 `sfm_reconstruction_quality` 數值 gate |
| SfM reconstruction quality | registered ratio >= 80%；registered images >= 60；points3D >= 30000；points/registered image >= 500；mean reprojection <= 2px；缺任一核心 metric 視為失敗 |
| Colored PLY | PLY > 128KB；正式 handoff 也要求 `point_cloud_quality` 數值 gate：PLY header 可讀、format 是 ascii/binary little/big endian、必須有 vertex element、XYZ 與 RGB vertex properties、vertices >= 10000、binary bytes 至少符合 header 宣告、vertices / points3D >= 30%、gate `ply_vertices` 與 header 一致 |
| Bundle | refs > 0；unique refs == refs；`ref_global.shape == [refs, 8448]`；tracking metadata complete |
| Local feature 3D anchors | mean anchored keypoints/ref >= 50；total anchored keypoints > 0；缺 gate metadata 不得只靠 bundle 檔案存在通過 |
| Report package quality | `build_report.json`、`build_config.json`、`stage_times.json` 必須是可讀 JSON；`outputs` 必須包含 `glomap_model`、`rgb_point_cloud`、`frame_manifest`、`intrinsics`、`config`、`stage_times`，且至少包含 `triangulated_bundle`、`tracking_bundle`、`snap_bundle` 之一；輸出路徑必須存在，檔案輸出不得為空；`parameters` 不得空白；`stage_times.total_seconds > 0` 且 latest `report` status 必須 success/pass；operator Markdown 必須包含輸出段落；`gates/` 至少保留 8 份 JSON |
| Holdout localization | final success >= 90%，或 baseline 已低於目標時 final 不退步；ok-to-fail <= 0；max fail run <= 30 |
| Production replay | validation MP4 應由 `--query-video` 轉成連續 replay frames 並保存 `production_stream_frames_manifest.json`；正式 replay JSON 必須有同 stem `.preflight.json`；preflight 必須確認 frames 是連續 stream-like frame dir（不得是 sparse geometry/keyframe dir）、reloc bundle、XFeat Torch Hub code/weights、MegaLoc code/HF weights完整；replay 的 calibration 與 2D-3D anchors 由 bundle 提供，不要求 sparse model；若明確指定 production descriptor cache，preflight 先檢查檔案非空，runtime loader 再驗 schema、ordered names、dimension 與 values；正式 replay 建議 frames >= 30；success >= 90%；max failure run <= 30；inliers p5 >= 30；若啟用 latency gate，必須有 `wall_ms.p90` 且低於門檻 |
| Automatic validation split | field profile 多段 `--videos` 預設保留最後 1 段 validation video；`--validation-videos` 可顯式指定；保留影片 symlink 標準化為 `.MP4`；昂貴建圖前會先確認可產生 localization / production evidence；`--skip-auto-validation-evidence` 只用於診斷，缺 evidence 不得 field handoff |
| Package verify | `package_verify.json.ok == true`；新增 gate 腳本也必須通過 py_compile |
| System verify | 所有 `required == true` checks 必須是 `PASS` 或 `READY`；optional `BLOCKED` 只列為現場限制 |

## Generalization Policy

- 多做檢查可以接受；少做檢查不接受。
- optional research modules 可以是 `SKIP`，但只要有執行就必須通過自己的 gate。
- DB/H5/NPZ 是低成本重調參的核心資料，不因清理或封包而刪除。
- `latest_build`、production symlink、deployment bundle 只能指向 field profile 總 gate 通過的 run。
- `latest_candidate_build` 可以指向 candidate profile 通過的研究/掃參 run，但不得當成外行人交付輸出。
- `build_pipeline.py --handoff-profile field` 是預設正式交付 profile，必須要求 holdout localization、production replay、package verify、system verify；缺任一項不得更新 `latest_build`。
- field profile 會自動從多段 `--videos` 保留 validation 影片並產生 holdout localization / production replay evidence；單段影片或 validation 不足時必須失敗，不得用訓練影格自我驗證。
- field validation preflight 必須在核心建圖前執行；缺 validation footage、validation 檔案不存在或為空、或保留後沒有建圖影片時，都不得進入昂貴前端。
- field profile 會自動補 package verify 與候選 system verify 證據；只有已提供可信外部證據時才需要手動傳 `--final-gate-package-verify-json` 或 `--final-gate-system-verify-json`。
- `tools/system_verify.py` 會把 `latest_build` summary 內這些 field handoff stage 的 `SKIP`、`required=false`、`ok=false` 視為 required failure；目前正式要求包含 `preflight_quality`、`frame_motion_quality`、`intrinsics_manifest_quality`、`view_graph_quality`、`database_quality`、`sfm_reconstruction_quality`、`point_cloud_quality`、`localization_bundle`、`report_package_quality`、`holdout_localization`、`production_replay`、`package_verify`、`system_verify`。舊版寬鬆 summary 需要補外部驗證後重新產生。
- `tools/system_verify.py --build-summary-json <run>/BUILD_GATE_SUMMARY.json` 是候選 build promotion 前的模式，只放行內嵌 `system_verify` 這一項 pending；其他任何 build stage 失敗、或定位、production replay、package、bundle gate pending 都會擋下。
- 沒有 `holdout_localization` 證據的 build 只能用 `--handoff-profile candidate` 當候選或研究輸出，不得標記為 field-deployable。
- production replay 未納入前，只能說 per-frame validation 通過，不能說現場 state machine 已驗證。
- `production_stream_preflight` 只能用來診斷缺資產/缺 frames；即使 preflight 通過，也不能取代正式 replay。
- 新場域失敗時，報告要指出第一個 blocked stage，而不是讓操作者猜是哪個演算法壞掉。
