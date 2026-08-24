#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
 map_update_tool.py  —  舊地圖 + 新資料 → 新地圖  (單檔高效更新工具)
================================================================================

一行輸入舊地圖 + 新影片幀,自動分類每支新影片、選路由、更新，輸出新的定位
bundle + 真實顏色點雲 PLY + 更新報告。不需重跑 6-8 小時的完整重建。

--------------------------------------------------------------------------------
 這是什麼 / 為什麼
--------------------------------------------------------------------------------
完整重建 (MegaLoc → MV-RoMa → GLOMAP + 全域 BA) 每次要 6-8 小時。多數「補拍」
只是加幾支影片，大部分場景沒變。本工具用「分類—路由」把更新降到分鐘級：

  新影片 vs 舊地圖
  ├─ 重疊 (register_rate ≥ 門檻) ─→ 方法2 REGISTER
  │     每幀 PnP 註冊進舊座標系；配到的 keypoint 直接「繼承」舊地圖的 3D 點；
  │     只把 keyframe 加進 reloc bundle。舊幾何 0 改動（權威不動）。點雲不增點。
  │
  └─ 新覆蓋 (register_rate < 門檻) ─→ 方法3 SUBMAP + SIM3
        用新影片自建獨立 submap (XFeat+LighterGlue → hloc 三角化)；用能同時對
        到舊地圖的「橋接幀」算 Umeyama Sim3(相似轉換)，把 submap 對齊進舊座標
        系；submap 的 keyframe + 3D 點(轉換+抽色)併入。無聯合 BA。

輸出的「地圖」有兩個面向：
  (1) 定位 bundle (.pt)  — 部署用。每個 keyframe = XFeat 特徵 + 每點 3D 錨點
      (xyz) + MegaLoc 全域描述子。更新 = 往 bundle 加 keyframe（避開 COLMAP 4.0.4
      rig model 手術牆）。這才是「定位地圖」。
  (2) 真色點雲 (.ply)    — 人眼檢視用。base(抽色) + 各新覆蓋 submap(Sim3+抽色)。

驗證過 (獨立 held-out、inlier 去重、無時序洩漏)：補覆蓋讓定位成功率 +2.7pp、
已覆蓋處中位 inlier +63%、0 退化。結論：**覆蓋是槓桿，不需重做 BA**。

--------------------------------------------------------------------------------
 用法
--------------------------------------------------------------------------------
  # 用預設路徑重現這次 P124/P125 更新（新資料放好即可）
  python3.12 map_update_tool.py \
      --new-data /media/cihcilab/新增磁碟區/sfm_reshot25/images \
      --videos P1240124 P1250125 \
      --out-dir ./out_update

  # 一般情形：新資料 = 一個資料夾，內含每支影片一個子夾(裡面是 .jpg 幀)
  python3.12 map_update_tool.py \
      --base-bundle  <舊定位bundle.pt> \
      --base-model   <舊COLMAP模型資料夾, e.g. glomap_fused/0> \
      --base-images  <舊模型影像根目錄, e.g. images_fused> \
      --new-data     <新影片幀根目錄> \
      --out-dir      <輸出資料夾>
      # --videos 省略 = 自動抓 new-data 下所有子資料夾

  # 只更新 bundle 不出點雲：加 --no-ply
  # 強制某片走特定方法：--force-method P1240124=submap  (或 =register / =skip_high_overlap)
  # 雙集合更新：
  #   --new-data       <geometry frames root>   # submap/triangulation only
  #   --connector-data <connector frames root>  # pure-rotation connectors for PnP/bundle only

輸出 (out-dir/)：
  reloc_map_updated.pt          更新後的定位 bundle（部署用）
  latest_map_realrgb.ply        整張真色點雲（base + 新覆蓋）
  update_report.md              路由決策 / 各片統計 / 參數快照

--------------------------------------------------------------------------------
 參數設置（全部可用 CLI 覆寫；預設為已驗證值）
--------------------------------------------------------------------------------
 特徵 / 匹配
   --qk            4096   每幀 XFeat keypoint 數 (deploy 原生，~10ms/pair)
   --topk          15     每個 query 用 MegaLoc 取前 K 個 ref 做匹配（去重後累積 2D-3D）
   --min-conf      0.1    LighterGlue 最小匹配信心
 註冊 / 橋接 (PnP)
   --min-inliers   50     PnP 接受門檻(ADD)。低於此該幀視為未註冊；橋接也用此門檻
   --seq-window    6      submap 序列鄰接配對窗 (i..i+window)
   --retr-nn       10     submap 每幀 MegaLoc 檢索鄰居配對數
 路由
   --overlap-thr   0.6    register_rate ≥ 此值 → 方法2(重疊)；否則 → 方法3(新覆蓋)
   --skip-overlap-thr 0.95 register_rate > 此值 → 不更新地圖，保留作驗證/QA
   --skip-min-median-inliers 80  high-overlap skip 也要求 sampled median inliers 夠高
   --classify-stride 5    分類抽樣步長（每 N 幀試註冊一次估 register_rate）
   --min-bridges   4      算 Sim3 至少要的橋接幀數（不足則跳過該片並記錄）
 相機內參 (ANAFI，凍結)
   --focal         1955.5 SIMPLE_RADIAL 焦距(px)。⚠ 若新資料解析度不同須等比縮放
   --k1            0.002  徑向畸變
   （主點固定取影像中心 W/2, H/2）
 其他
   --bundle-vpr    megaloc  輸出 bundle 的 ref_global 固定為 MegaLoc。
   --no-ply               不輸出點雲，只更新 bundle
   --ply-stride    1      點雲抽點步長(檢視用，>1 可縮檔)
   --force-method  片名=register|submap|skip_high_overlap  跳過自動分類，強制路由

--------------------------------------------------------------------------------
 執行環境
--------------------------------------------------------------------------------
  直譯器: /usr/bin/python3.12  (含 hloc + torch.hub XFeat/MegaLoc + pycolmap 4.0.4)
  相依:   torch, opencv, h5py, numpy, pycolmap, hloc, 以及 sfm_glomap 的
          megaloc_lib / reloc_localizer_xfeat（見 --repo）
  GPU:    CUDA。序列執行（單 GPU），多片自動一片接一片。
  注意:   所有 SfM 暫存 DB 放 /tmp(ext4)。NTFS3 會弄壞 SQLite。

--------------------------------------------------------------------------------
 已知限制 / 何時該回頭做完整重建
--------------------------------------------------------------------------------
  * 方法3 是 Sim3 對齊，非聯合 BA → 接縫可能輕微雙層(P124 resid ~0.13u)。
    定位夠用；要幾何完美再補一輪錨定 BA。
  * 場景「實體變了」(施工/物件移除) 本工具不覆寫舊幾何 → 需 method4 tile-replace
    (變更偵測 old_support_ratio)，不在本工具範圍。
  * 新影片和舊地圖完全無重疊(算不出 Sim3) → 無法對齊，會跳過並在報告標記。
  * 累積更新多次後建議定期做一次完整重建，避免 Sim3 誤差堆疊。
================================================================================
"""
import sys, os, time, json, shutil, argparse, math
from collections import Counter, defaultdict
import numpy as np, torch, cv2
from pathlib import Path
import pycolmap
from megaloc_cache import (
    MEGALOC_DESCRIPTOR_DIM,
    load_aligned_megaloc_cache,
    write_megaloc_cache,
)
from stability_scores import build_ref_stability, rerank_indices_by_stability, stability_summary
from update_quality_gates import bridge_hard_fail_reasons, bridge_quality_checks, bridge_quality_warnings, matched_warnings

def find_system_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if p.name == "sfm_system":
            return p
    return Path("/media/cihcilab/新增磁碟區/sfm_system")


# ---- defaults (validated) ----
SYSTEM_ROOT = find_system_root(Path(__file__).resolve())
DEF_REPO_PATH = SYSTEM_ROOT / "定位" / "source" / "sfm_glomap"
DEF_REPO   = str(DEF_REPO_PATH)
DEF_BUNDLE = str(DEF_REPO_PATH / "deploy" / "reloc_map_xfeat_tri.pt")
DEF_MODEL  = str(DEF_REPO_PATH / "glomap_fused" / "0")
DEF_IMAGES = str(DEF_REPO_PATH / "update_from_reshot25_build_20260701" / "base_images_fused")

def require_explicit_if_default_missing(flag, provided, default):
    if provided:
        return provided
    default_path=Path(default)
    if not default_path.exists():
        raise SystemExit(f"{flag} default is missing ({default_path}); pass {flag} explicitly")
    return str(default_path)

FIXED_INTRINSICS = {
    # width, height: SIMPLE_RADIAL params [f, cx, cy, k1]
    (2688, 1512): [1955.5, 1344.0, 756.0, 0.0020],
    (1920, 1080): [1400.0, 960.0, 540.0, 0.0015],
    (1280, 720): [936.5, 640.0, 360.0, 0.0035],
}

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def n2p(a,b): return "/".join((a.replace("/","-"), b.replace("/","-")))

def fixed_intrinsics(W,H,fallback_focal=None,fallback_k1=None):
    p=FIXED_INTRINSICS.get((int(W),int(H)))
    if p is not None:
        return list(p)
    if fallback_focal is not None:
        # ANAFI fallback: scale 2688-wide calibration by image width.
        f=float(fallback_focal)*float(W)/2688.0
        return [f,float(W)/2.0,float(H)/2.0,float(fallback_k1 if fallback_k1 is not None else 0.002)]
    raise ValueError(f"no fixed intrinsics for resolution {W}x{H}")

def make_camera(W,H,args):
    params=fixed_intrinsics(W,H,args.focal,args.k1)
    cam=pycolmap.Camera.create_from_model_id(0,pycolmap.CameraModelId.SIMPLE_RADIAL,params[0],W,H)
    cam.params=params
    return cam, params

def normalize_desc(label, desc, expected_dim=8448):
    desc=np.asarray(desc,np.float32)
    if desc.ndim!=2:
        raise ValueError(f"{label}: descriptor array must be 2D, got shape {desc.shape}")
    if expected_dim is not None and desc.shape[1]!=expected_dim:
        raise ValueError(f"{label}: expected MegaLoc dim {expected_dim}, got {desc.shape[1]}")
    if not np.isfinite(desc).all():
        raise ValueError(f"{label}: descriptor array contains NaN/Inf")
    norms=np.linalg.norm(desc,axis=1)
    log(f"{label}: desc shape={desc.shape}, norm min/med/max={norms.min():.3f}/{np.median(norms):.3f}/{norms.max():.3f}")
    bad=np.where(norms<0.5)[0]
    if len(bad):
        raise ValueError(f"{label}: {len(bad)} descriptors have norm <0.5; likely corrupt/black frames")
    return desc/(norms[:,None]+1e-9)

def umeyama(src, dst):
    """相似轉換 dst ≈ s*R*src + t (Umeyama SVD)."""
    ms=src.mean(0); md=dst.mean(0); S=src-ms; D=dst-md; cov=D.T@S/len(src)
    U,d,Vt=np.linalg.svd(cov); Sd=np.eye(3)
    if np.linalg.det(U)*np.linalg.det(Vt)<0: Sd[2,2]=-1
    R=U@Sd@Vt; var=(S**2).sum()/len(src); s=np.trace(np.diag(d)@Sd)/var
    return s, R, md - s*R@ms

def pose_center_yaw_from_transform(T):
    T = T() if callable(T) else T
    Rm = T.rotation.matrix()
    tt = np.asarray(T.translation)
    C = -Rm.T @ tt
    fwd = Rm.T @ np.array([0.0, 0.0, 1.0])
    yaw = math.atan2(float(fwd[1]), float(fwd[0]))
    return C.astype(np.float32), float(yaw), fwd.astype(np.float32)

def pose_center_yaw_from_result(result):
    return pose_center_yaw_from_transform(result["cam_from_world"])

def pose_center_yaw_from_image(image):
    return pose_center_yaw_from_transform(image.cam_from_world)

def transform_center_yaw(C, fwd, s, R, t):
    Cb = (float(s) * (R @ np.asarray(C, float)) + np.asarray(t, float)).astype(np.float32)
    fb = R @ np.asarray(fwd, float)
    yaw = math.atan2(float(fb[1]), float(fb[0]))
    return Cb, float(yaw)

def complete_covisibility(ref_names, centers, covis, top_covis):
    centers = np.asarray(centers, np.float32)
    n = len(ref_names)
    out = {}
    for i, name in enumerate(ref_names):
        existing = []
        for j in (covis or {}).get(name, []):
            try:
                jj = int(j)
            except Exception:
                continue
            if 0 <= jj < n and jj != i and jj not in existing:
                existing.append(jj)
            if len(existing) >= top_covis:
                break
        if len(existing) < top_covis and centers.shape == (n, 3) and np.isfinite(centers[i]).all():
            d = np.linalg.norm(centers - centers[i][None, :], axis=1)
            d[i] = np.inf
            for jj in np.argsort(d):
                jj = int(jj)
                if not np.isfinite(d[jj]):
                    continue
                if jj not in existing:
                    existing.append(jj)
                if len(existing) >= top_covis:
                    break
        out[name] = existing[:top_covis]
    return out

def inlier_indices(result, n):
    if not result:
        return np.array([], dtype=np.int64)
    for key in ("inlier_mask","inliers","inlier_idxs","inlier_indices"):
        if isinstance(result,dict) and key in result:
            mask=np.asarray(result[key])
            if mask.dtype==bool and len(mask)==n:
                return np.where(mask)[0].astype(np.int64)
            if np.issubdtype(mask.dtype,np.integer):
                return mask[(mask>=0)&(mask<n)].astype(np.int64)
    nin=int(result.get("num_inliers",0)) if isinstance(result,dict) else 0
    return np.arange(min(n,nin), dtype=np.int64)

def tile_key(pt, W, H, grid):
    x=float(pt[0]); y=float(pt[1])
    gx=min(grid-1,max(0,int(x/max(1.0,float(W))*grid)))
    gy=min(grid-1,max(0,int(y/max(1.0,float(H))*grid)))
    return f"{gx},{gy}"

def counter_top(c, n=50):
    return [{"key":k,"count":int(v)} for k,v in c.most_common(n)]

def parse_int_list(text):
    return [int(x) for x in str(text).replace(";",",").split(",") if x.strip()]

def parse_float_list(text):
    return [float(x) for x in str(text).replace(";",",").split(",") if x.strip()]

def main():
    ap=argparse.ArgumentParser(add_help=False)
    ap.add_argument("-h","--help",action="store_true")
    ap.add_argument("--repo",default=None)
    ap.add_argument("--matcher",choices=["edm","xfeat"],default="edm",
                    help="Update-time correspondence frontend. Default edm, matching the "
                         "deployed localizer. xfeat is legacy and feeds a bundle format that "
                         "is no longer flown -- A/B only.")
    ap.add_argument("--allow-xfeat-submap",action="store_true",
                    help="Permit the not-yet-migrated XFeat submap route while --matcher edm. "
                         "Output is mixed-frontend and must stay in quarantine.")
    ap.add_argument("--edm-deploy",default=None,
                    help="EDM deploy dir (reloc_localizer_edm.py / edm_matcher.py). "
                         "Defaults to <repo>/../EDM定位測試/deploy.")
    ap.add_argument("--base-bundle",default=None)
    ap.add_argument("--base-model",default=None)
    ap.add_argument("--base-images",default=None)
    ap.add_argument("--base-megaloc-cache",default="",
                    help="optional precomputed base MegaLoc npz cache; copied/reused instead of extracting base refs")
    ap.add_argument("--new-data",default=str(SYSTEM_ROOT / "更新地圖" / "source" / "sfm_reshot25" / "images"))
    ap.add_argument("--connector-data",default=None,
                    help="Optional connector frame root. These frames are PnP/bundle-only and are not triangulated.")
    ap.add_argument("--videos",nargs="*",default=None)
    ap.add_argument("--out-dir",default="./out_update")
    ap.add_argument("--qk",type=int,default=4096)
    ap.add_argument("--topk",type=int,default=15)
    ap.add_argument("--min-conf",type=float,default=0.1)
    ap.add_argument("--min-inliers",type=int,default=50)
    ap.add_argument("--seq-window",type=int,default=6)
    ap.add_argument("--retr-nn",type=int,default=10)
    ap.add_argument("--overlap-thr",type=float,default=0.6)
    ap.add_argument("--skip-overlap-thr",type=float,default=0.95,
                    help="If non-forced register_rate is greater than this, skip map update and keep the video for QA.")
    ap.add_argument("--skip-min-median-inliers",type=int,default=80,
                    help="High-overlap skip also requires sampled median PnP inliers at least this value.")
    ap.add_argument("--min-support-area",type=float,default=0.05,
                    help="Warn when accepted 2D-3D support covers less than this bbox area fraction of the image.")
    ap.add_argument("--observation-stride",type=int,default=5,
                    help="Stride for recording high-overlap observation sessions without adding geometry.")
    ap.add_argument("--health-grid",type=int,default=4,
                    help="Image grid per side for old-point health / changed-region support stats.")
    ap.add_argument("--health-min-tile-kp",type=int,default=80,
                    help="Minimum query keypoints in a tile before it can be flagged as low-support.")
    ap.add_argument("--health-low-support-ratio",type=float,default=0.015,
                    help="Tile inliers/query-keypoints ratio below this is a changed-region candidate signal.")
    ap.add_argument("--bridge-min-support-area",type=float,default=0.05,
                    help="Bridge quality warning threshold for median inlier bbox area fraction.")
    ap.add_argument("--bridge-min-inlier-ratio",type=float,default=0.25,
                    help="Bridge quality warning threshold for median PnP inlier/raw-match ratio.")
    ap.add_argument("--bridge-min-geometry",type=int,default=4,
                    help="Bridge quality warning threshold for minimum geometry-frame bridges.")
    ap.add_argument("--bridge-min-geometry-ratio",type=float,default=0.0,
                    help="Bridge quality warning threshold for geometry bridges / total bridges.")
    ap.add_argument("--bridge-gate-quality",action=argparse.BooleanOptionalAction,default=True,
                    help="Bridge quality warnings reject the submap instead of only warning.")
    ap.add_argument("--quarantine-warnings",default="retrieval_high_but_inliers_low",
                    help="Comma-separated classify warnings that should quarantine low-quality segments.")
    ap.add_argument("--quarantine-action",choices=["report","skip"],default="report",
                    help="report = mark the segment; skip = hard gate and do not add keyframes/points.")
    ap.add_argument("--adaptive-params",action=argparse.BooleanOptionalAction,default=True,
                    help="Automatically retune retrieval/local matching parameters by route and failure mode.")
    ap.add_argument("--adaptive-bridge-qk",type=int,default=8192,
                    help="XFeat top_k used for difficult bridge-search retries.")
    ap.add_argument("--adaptive-bridge-topks",default="30,50",
                    help="MegaLoc retrieval topK sweep when bridge count is insufficient.")
    ap.add_argument("--adaptive-bridge-min-confs",default="0.05,0.1",
                    help="LighterGlue min_conf sweep for loose difficult bridge search.")
    ap.add_argument("--adaptive-strict-min-confs",default="0.15,0.2",
                    help="LighterGlue min_conf sweep when retrieval looks false or bridge quality is weak.")
    ap.add_argument("--adaptive-strict-topks",default="15,30",
                    help="MegaLoc retrieval topK sweep for strict false-overlap / repetitive-structure checks.")
    ap.add_argument("--classify-stride",type=int,default=5)
    ap.add_argument("--min-bridges",type=int,default=4)
    ap.add_argument("--submap-connector-bridges",action=argparse.BooleanOptionalAction,default=True,
                    help="Use connector frames inside submap reconstruction to obtain poses for Sim3 bridges, but do not export connector geometry.")
    ap.add_argument("--min-geometry-observations",type=int,default=2,
                    help="When connector bridge frames are used, export only submap points observed by at least this many geometry frames.")
    ap.add_argument("--focal",type=float,default=1955.5)
    ap.add_argument("--k1",type=float,default=0.002)
    ap.add_argument("--bundle-vpr",choices=["megaloc"],default="megaloc")
    ap.add_argument("--stability-half-life-sessions",type=float,default=4.0,
                    help="Half-life, in update sessions, for ExMaps-style ref stability decay.")
    ap.add_argument("--stability-rerank-weight",type=float,default=0.05,
                    help="Mild log-stability weight used after MegaLoc retrieval candidate selection; 0 disables rerank.")
    ap.add_argument("--no-ply",action="store_true")
    ap.add_argument("--ply-stride",type=int,default=1)
    ap.add_argument("--force-method",nargs="*",default=[])
    a=ap.parse_args()
    if a.help:
        print(__doc__); return
    a.repo=require_explicit_if_default_missing("--repo",a.repo,Path(DEF_REPO))
    a.base_bundle=require_explicit_if_default_missing("--base-bundle",a.base_bundle,Path(DEF_BUNDLE))
    a.base_model=require_explicit_if_default_missing("--base-model",a.base_model,Path(DEF_MODEL))
    a.base_images=require_explicit_if_default_missing("--base-images",a.base_images,Path(DEF_IMAGES))

    sys.path.insert(0, a.repo+"/deploy"); sys.path.insert(0, a.repo+"/scripts")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import megaloc_lib
    from update_matcher import dedup_anchored
    import pycolmap, h5py
    import hloc.reconstruction as RC
    DEV="cuda"; QK=a.qk; ADD=a.min_inliers
    forced={}
    for item in a.force_method:
        if "=" not in item:
            raise SystemExit(f"--force-method must be SEQ=register|submap|skip_high_overlap|changed, got {item}")
        k,v=item.split("=",1)
        if v == "qa_only":
            v = "skip_high_overlap"
        if v not in {"register","submap","skip_high_overlap","changed","tile_replace"}:
            raise SystemExit(f"unknown forced method {v}; use register, submap, skip_high_overlap, changed, or tile_replace")
        forced[k]=v
    NB=Path(a.new_data)
    CB=Path(a.connector_data) if a.connector_data else None
    if a.videos:
        vids=a.videos
    else:
        vv={p.name for p in NB.iterdir() if p.is_dir()}
        if CB and CB.exists():
            vv.update(p.name for p in CB.iterdir() if p.is_dir())
        vids=sorted(vv)
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    log(f"videos to add: {vids}")

    b=torch.load(a.base_bundle,map_location='cpu',weights_only=False)
    refs=dict(b['refs']); base_names=list(b['ref_names'])
    base_stability=np.asarray(b.get('ref_stability',np.ones(len(base_names),np.float32)),np.float32)
    if base_stability.shape!=(len(base_names),):
        log(f"base ref_stability shape {base_stability.shape} does not match refs; using neutral scores")
        base_stability=np.ones(len(base_names),np.float32)
    obs_sessions={}

    def get_session(seq):
        if seq not in obs_sessions:
            obs_sessions[seq]={
                'frames':0,'localized':0,'failed':0,
                'support_areas':[],'inlier_ratios':[],
                'base_ref_hits':Counter(),'anchor_hits':Counter(),
                'tile':defaultdict(lambda:{'query_kp':0,'matches':0,'inliers':0,'frames':0,'low_support_frames':0})
            }
        return obs_sessions[seq]
    # base MegaLoc for retrieval. Prefer an existing package-level cache to avoid
    # re-extracting all base keyframes on every update run.
    cache=Path(a.base_megaloc_cache) if a.base_megaloc_cache else out/"base_megaloc.npz"
    if cache.exists():
        bdesc=load_aligned_megaloc_cache(
            cache,
            base_names,
            expected_dim=MEGALOC_DESCRIPTOR_DIM,
            expected_input_size=322,
        )
    else:
        log("extracting base MegaLoc (one-time)...")
        bdesc=megaloc_lib.extract(base_names,Path(a.base_images),DEV).astype(np.float32)
        write_megaloc_cache(cache,bdesc,base_names,input_size=322)
    out_cache=out/"base_megaloc.npz"
    if cache != out_cache:
        write_megaloc_cache(out_cache,bdesc,base_names,input_size=322)
    bdesc=normalize_desc("base MegaLoc",bdesc)

    if a.matcher=="edm":
        edm_deploy=Path(a.edm_deploy) if a.edm_deploy else (SYSTEM_ROOT/"EDM定位測試"/"deploy")
        if not edm_deploy.is_dir():
            raise SystemExit(f"--edm-deploy not found: {edm_deploy}. The EDM localizer lives in "
                             f"the localization repo; pass its deploy dir explicitly.")
        from update_matcher import EDMUpdateMatcher
        cam0,_=make_camera(a.width,a.height,a) if hasattr(a,"width") else (None,None)
        matcher_backend=EDMUpdateMatcher.from_deploy(
            edm_deploy, Path(a.base_bundle), cam0,
            topk=a.topk, min_conf=a.min_conf,
            pnp_max_error=a.pnp_max_error if hasattr(a,"pnp_max_error") else 8.0,
            min_inliers=a.min_inliers)
    else:
        log("WARNING: --matcher xfeat is legacy. The resulting bundle is NOT the deployed "
            "format; use this for A/B measurement only.")
        from update_matcher import XFeatUpdateMatcher
        matcher_backend=XFeatUpdateMatcher.from_deploy(
            Path(a.repo)/"deploy", Path(a.repo)/"scripts", refs,
            qk=a.qk, topk=a.topk, min_conf=a.min_conf)
    log(f"update matcher backend: {matcher_backend.name}")
    def mnorm(names,base):
        return normalize_desc(f"query MegaLoc {base}",megaloc_lib.extract(names,base,DEV).astype(np.float32))

    def collect_anchored(query,qg,topk=None,min_conf=None,matcher=None):
        """dedup 2D-3D vs base bundle: one best anchor per query observation.

        Backend-neutral: the matcher supplies a stable per-image identity for
        each query observation (XFeat keypoint index / EDM query cell), so the
        inlier count stays comparable across frontends.
        """
        topk=topk or a.topk; min_conf=a.min_conf if min_conf is None else min_conf
        matcher=matcher or matcher_backend
        sims=bdesc@qg
        idx=rerank_indices_by_stability(sims,base_stability,topk,stability_weight=a.stability_rerank_weight)
        rows=matcher.correspondences(query,[base_names[ti] for ti in idx],[int(ti) for ti in idx],
                                     min_conf=min_conf)
        return dedup_anchored(rows).as_tuple()

    def collect(query,qg,topk=None,min_conf=None):
        P2,P3,_=collect_anchored(query,qg,topk,min_conf)
        return P2,P3

    def pnp(P2,P3,W,H):
        cam,_=make_camera(W,H,a)
        try: return pycolmap.estimate_and_refine_absolute_pose(P2.astype(float),P3.astype(float),cam)
        except Exception: return None

    def support_area(P2,W,H,result=None):
        pts=np.asarray(P2,float)
        if result is not None:
            mask=None
            for key in ("inlier_mask","inliers","inlier_idxs","inlier_indices"):
                if isinstance(result,dict) and key in result:
                    mask=np.asarray(result[key])
                    break
            if mask is not None and len(mask):
                if mask.dtype==bool and len(mask)==len(pts):
                    pts=pts[mask]
                elif np.issubdtype(mask.dtype,np.integer):
                    pts=pts[mask[(mask>=0)&(mask<len(pts))]]
        if len(pts)<4:
            return 0.0
        x0,y0=pts.min(axis=0); x1,y1=pts.max(axis=0)
        return float(max(0.0,x1-x0)*max(0.0,y1-y0)/max(1.0,float(W*H)))

    def record_observation(seq, frame_name, W, H, qk, P2, meta, result, label):
        sess=get_session(seq)
        sess['frames']+=1
        localized=bool(result and int(result.get('num_inliers',0))>=ADD)
        if localized: sess['localized']+=1
        else: sess['failed']+=1
        grid=max(1,int(a.health_grid))
        frame_q=Counter(); frame_m=Counter(); frame_i=Counter()
        for pt in np.asarray(qk,float):
            tk=tile_key(pt,W,H,grid)
            frame_q[tk]+=1
        if P2 is not None and len(P2):
            for pt in np.asarray(P2,float):
                tk=tile_key(pt,W,H,grid)
                frame_m[tk]+=1
            ii=inlier_indices(result,len(P2)) if localized else np.array([],dtype=np.int64)
            for idx in ii:
                m=meta[int(idx)]
                sess['base_ref_hits'][m['ref_name']]+=1
                sess['anchor_hits'][f"{m['ref_name']}#{m['ref_kp']}"]+=1
                tk=tile_key(P2[int(idx)],W,H,grid)
                frame_i[tk]+=1
            sess['support_areas'].append(support_area(P2,W,H,result) if localized else 0.0)
            sess['inlier_ratios'].append((len(ii)/max(1,len(P2))) if len(P2) else 0.0)
        for tk,qn in frame_q.items():
            tile=sess['tile'][tk]
            tile['frames']+=1
            tile['query_kp']+=int(qn)
            tile['matches']+=int(frame_m.get(tk,0))
            tile['inliers']+=int(frame_i.get(tk,0))
            if localized and qn>=a.health_min_tile_kp and (frame_i.get(tk,0)/max(1,qn))<a.health_low_support_ratio:
                tile['low_support_frames']+=1

    def record_observation_frames(fr, names, em, seq, label):
        kept=0
        for k in range(0,len(fr),max(1,a.observation_stride)):
            p=fr[k]; nm=names[k]
            rgb=load_rgb(p); H,W=rgb.shape[:2]
            query=matcher_backend.prepare_query(rgb)
            P2,P3,meta=collect_anchored(query,em[k])
            qk=query.keypoints_or_empty()
            result=None
            if P2 is not None and len(P2)>=6:
                result=pnp(P2,P3,W,H)
            record_observation(seq,nm,W,H,qk,P2 if P2 is not None else np.zeros((0,2)),meta,result,label)
            kept+=1
        log(f"[{seq}] {label}: recorded {kept} observation frames for health/change stats")

    def bridge_search_attempt(seq, sub_items, sub_pose, qk_limit, topk, min_conf, label, record_geom):
        src=[]; dst=[]; quality=[]; bridge_geom=0; bridge_conn=0
        for p,nm,is_geom,qg in sub_items:
            if nm not in sub_pose:
                continue
            rgb=load_rgb(p); H,W=rgb.shape[:2]
            query=matcher_backend.prepare_query(rgb,**({'qk':int(qk_limit)} if matcher_backend.name=='xfeat' else {}))
            P2,P3,meta=collect_anchored(query,qg,topk=int(topk),min_conf=float(min_conf))
            qk=query.keypoints_or_empty()
            if P2 is None or len(P2)<ADD:
                continue
            r=pnp(P2,P3,W,H)
            if not r or r.get('num_inliers',0)<ADD:
                continue
            nin=int(r.get('num_inliers',0))
            area=support_area(P2,W,H,r)
            ratio=float(nin/max(1,len(P2)))
            quality.append({'name':nm,'geometry':bool(is_geom),'raw_matches':int(len(P2)),
                            'inliers':nin,'inlier_ratio':ratio,'support_area':area,
                            'qk':int(qk_limit),'topk':int(topk),'min_conf':float(min_conf),
                            'attempt':label})
            if is_geom and record_geom:
                record_observation(seq,nm,W,H,qk,P2,meta,r,f"BRIDGE_{label}")
            src.append(sub_pose[nm]); dst.append(np.asarray(r['cam_from_world'].inverse().translation))
            if is_geom: bridge_geom+=1
            else: bridge_conn+=1
        out={'label':label,'qk':int(qk_limit),'topk':int(topk),'min_conf':float(min_conf),
             'src':src,'dst':dst,'quality':quality,'bridge_geometry':bridge_geom,'bridge_connector':bridge_conn}
        if len(src)>=a.min_bridges:
            src_arr=np.asarray(src); dst_arr=np.asarray(dst)
            s0,R0,t0=umeyama(src_arr,dst_arr)
            resid0=float(np.median(np.linalg.norm(dst_arr-(s0*(src_arr@R0.T)+t0),axis=1)))
            med_ratio=float(np.median([x['inlier_ratio'] for x in quality])) if quality else 0.0
            med_area=float(np.median([x['support_area'] for x in quality])) if quality else 0.0
            score=1000.0 - 50.0*resid0 + 100.0*med_ratio + 25.0*med_area + min(75.0,float(len(src)))
            out.update({'sim3_scale':float(s0),'sim3_R':R0,'sim3_t':t0,'sim3_resid_u':resid0,
                        'bridge_median_inlier_ratio':med_ratio,'bridge_median_support_area':med_area,
                        'score':score})
        else:
            out.update({'bridge_median_inlier_ratio':0.0,'bridge_median_support_area':0.0,
                        'sim3_resid_u':None,'score':float(len(src))})
        return out

    def choose_bridge_attempt(seq, sub_items, sub_pose, classify_warnings):
        attempts=[]
        initial=bridge_search_attempt(seq,sub_items,sub_pose,a.qk,a.topk,a.min_conf,"default",False)
        attempts.append(initial)
        need_loose=len(initial['src'])<a.min_bridges
        need_strict=(
            "retrieval_high_but_inliers_low" in classify_warnings or
            (len(initial['src'])>=a.min_bridges and (
                initial['bridge_median_inlier_ratio']<a.bridge_min_inlier_ratio or
                initial['bridge_median_support_area']<a.bridge_min_support_area
            ))
        )
        if a.adaptive_params and need_loose:
            for topk in parse_int_list(a.adaptive_bridge_topks):
                for conf in parse_float_list(a.adaptive_bridge_min_confs):
                    attempts.append(bridge_search_attempt(seq,sub_items,sub_pose,a.adaptive_bridge_qk,topk,conf,f"loose_q{a.adaptive_bridge_qk}_k{topk}_c{conf}",False))
        if a.adaptive_params and need_strict:
            for topk in parse_int_list(a.adaptive_strict_topks):
                for conf in parse_float_list(a.adaptive_strict_min_confs):
                    attempts.append(bridge_search_attempt(seq,sub_items,sub_pose,a.qk,topk,conf,f"strict_q{a.qk}_k{topk}_c{conf}",False))
        best=max(attempts,key=lambda x:x['score'])
        # Record observations only for the selected bridge attempt to avoid double-counting.
        if best['quality']:
            replay=bridge_search_attempt(seq,sub_items,sub_pose,best['qk'],best['topk'],best['min_conf'],best['label'],True)
            # Keep selected attempt summary/Sim3 from replay so observation and output match.
            best=replay if replay['score']>=best['score']-1e-6 else best
        return best, attempts

    def load_rgb(p): return cv2.cvtColor(cv2.imread(str(p),cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
    def bundle_global(rgb, name, em, name_to_idx):
        return em[name_to_idx[name]].astype(np.float32)

    def materialize_images(items, root):
        root=Path(root)
        seen=set()
        for p,nm,_,_ in items:
            if nm in seen:
                continue
            seen.add(nm)
            dst=root/nm
            dst.parent.mkdir(parents=True,exist_ok=True)
            if dst.exists():
                continue
            try:
                os.symlink(Path(p).resolve(),dst)
            except OSError:
                shutil.copy2(p,dst)

    def add_registered_keyframes(fr, names, em, name_to_idx, seq, label):
        nreg=0
        inliers=[]
        for k,(p,nm) in enumerate(zip(fr,names)):
            rgb=load_rgb(p); H,W=rgb.shape[:2]
            query=matcher_backend.prepare_query(rgb)
            P2,P3,meta=collect_anchored(query,em[k])
            qk=query.keypoints_or_empty()
            if P2 is None or len(P2)<ADD:
                record_observation(seq,nm,W,H,qk,np.zeros((0,2)),[],None,label)
                continue
            r=pnp(P2,P3,W,H)
            record_observation(seq,nm,W,H,qk,P2,meta,r,label)
            if not r or r.get('num_inliers',0)<ADD: continue
            new_kf.append((nm,matcher_backend.bundle_keyframe(query,meta,rgb,nm)))
            C,yaw,_=pose_center_yaw_from_result(r)
            ii=inlier_indices(r,len(meta))
            cov=[meta[int(j)]['ref_index'] for j in ii if int(j)<len(meta)]
            new_centers.append(C); new_yaws.append(yaw); new_covis[nm]=list(dict.fromkeys(int(x) for x in cov))
            new_glob.append(bundle_global(rgb,nm,em,name_to_idx)); nreg+=1; inliers.append(int(r.get('num_inliers',0)))
        med=float(np.median(inliers)) if inliers else None
        log(f"[{seq}] {label}: +{nreg} keyframes (0 new points), median_inliers={med}")
        return nreg, med

    new_kf=[]; new_glob=[]; new_centers=[]; new_yaws=[]; new_covis={}; ply_xyz=[]; ply_rgb=[]; report=[]

    def snapshot_new_state():
        return (len(new_kf), len(new_glob), len(new_centers), len(new_yaws), set(new_covis.keys()))

    def rollback_new_state(state):
        kf_n, glob_n, center_n, yaw_n, covis_keys = state
        del new_kf[kf_n:]
        del new_glob[glob_n:]
        del new_centers[center_n:]
        del new_yaws[yaw_n:]
        for key in list(new_covis.keys()):
            if key not in covis_keys:
                del new_covis[key]

    for seq in vids:
        fr=sorted((NB/seq).glob("*.jpg")); names=[f"{seq}/{p.name}" for p in fr]
        cfr=sorted((CB/seq).glob("*.jpg")) if CB and (CB/seq).exists() else []
        cnames=[f"{seq}/{p.name}" for p in cfr]
        if not fr and not cfr: log(f"[{seq}] no frames, skip"); continue
        em=mnorm(names,NB) if fr else np.zeros((0,8448),np.float32)
        cem=mnorm(cnames,CB) if cfr else np.zeros((0,8448),np.float32)
        name_to_idx={n:i for i,n in enumerate(names)}
        cname_to_idx={n:i for i,n in enumerate(cnames)}
        classify_fr=fr+cfr
        classify_names=names+cnames
        classify_em=np.vstack([em,cem]) if len(fr)+len(cfr) else np.zeros((0,8448),np.float32)
        # ---- classify (unless forced) ----
        route=forced.get(seq)
        rr=None
        if route is None and classify_fr:
            samp=list(range(0,len(classify_fr),a.classify_stride)); ok=0; attempts=0
            sample_inliers=[]; sample_areas=[]; warnings=[]
            for i in samp:
                rgb=load_rgb(classify_fr[i]); H,W=rgb.shape[:2]
                query=matcher_backend.prepare_query(rgb)
                P2,P3=collect(query,classify_em[i])
                if P2 is not None and len(P2)>=6:
                    attempts+=1
                    r=pnp(P2,P3,W,H)
                    if r:
                        nin=int(r.get('num_inliers',0))
                        sample_inliers.append(nin)
                        if nin>=ADD:
                            ok+=1
                            sample_areas.append(support_area(P2,W,H,r))
            rr=ok/max(1,len(samp))
            attempt_rate=attempts/max(1,len(samp))
            med_in=float(np.median(sample_inliers)) if sample_inliers else 0.0
            med_area=float(np.median(sample_areas)) if sample_areas else 0.0
            if attempt_rate>=0.70 and rr<a.overlap_thr and med_in<ADD:
                warnings.append("retrieval_high_but_inliers_low")
            if rr>=a.overlap_thr and med_area<a.min_support_area:
                warnings.append("inliers_spatially_concentrated")
            if rr>a.skip_overlap_thr and med_in>=a.skip_min_median_inliers and med_area>=a.min_support_area:
                route="skip_high_overlap"
            else:
                route="register" if rr>=a.overlap_thr else "submap"
            classify_metrics={'sampled':len(samp),'pnp_attempts':attempts,'attempt_rate':attempt_rate,
                              'sampled_median_inliers':med_in,'sampled_median_support_area':med_area,
                              'classify_warnings':",".join(warnings)}
            log(f"[{seq}] register_rate {rr:.0%} ({ok}/{len(samp)}), med_inliers={med_in:.0f}, support_area={med_area:.3f} -> {route.upper()}{' WARN '+','.join(warnings) if warnings else ''}")
        else:
            route = route or ("connector_only" if cfr else "submap")
            classify_metrics={}
            log(f"[{seq}] forced -> {route.upper()}" if seq in forced else f"[{seq}] route -> {route.upper()}")

        if route in {"changed","tile_replace"}:
            log(f"[{seq}] changed-region route requested; method4 detection/rebuild is documented but not implemented in this tool yet")
            if classify_fr:
                record_observation_frames(classify_fr,classify_names,classify_em,seq,"CHANGED_REGION_OBSERVATION")
            report.append({'seq':seq,'route':'changed-region','status':'needs_tile_replace','register_rate':rr,'geometry_frames':len(fr),'connector_frames':len(cfr),'keyframes_added':0,'points_added':0,**classify_metrics})
            continue

        if route=="skip_high_overlap":
            record_observation_frames(classify_fr,classify_names,classify_em,seq,"QA_OBSERVATION")
            report.append({'seq':seq,'route':'skip_high_overlap','status':'qa_only','register_rate':rr,'geometry_frames':len(fr),'connector_frames':len(cfr),'keyframes_added':0,'points_added':0,**classify_metrics})
            rr_text = "-" if rr is None else f"{rr:.0%}"
            log(f"[{seq}] SKIP: register_rate {rr_text}; keep as validation/QA, not map update")
            continue

        quarantine_status=''
        quarantine_hits=matched_warnings(classify_metrics.get('classify_warnings',''),a.quarantine_warnings)
        if quarantine_hits and route in {"register","submap","connector_only"}:
            status="quarantined:"+",".join(quarantine_hits)
            if a.quarantine_action=="skip":
                record_observation_frames(classify_fr,classify_names,classify_em,seq,"QUARANTINE_OBSERVATION")
                report.append({'seq':seq,'route':route,'status':status,'register_rate':rr,'geometry_frames':len(fr),'connector_frames':len(cfr),'keyframes_added':0,'points_added':0,**classify_metrics})
                log(f"[{seq}] QUARANTINE hard gate: {','.join(quarantine_hits)}; no keyframes or points added")
                continue
            quarantine_status=status
            log(f"[{seq}] QUARANTINE report-only: {','.join(quarantine_hits)}")

        seq_state=snapshot_new_state()
        if cfr and route in {"register","submap","connector_only"}:
            cn, cmed = add_registered_keyframes(cfr, cnames, cem, cname_to_idx, seq, "CONNECTOR")
        else:
            cn, cmed = 0, None

        if route=="connector_only":
            status=quarantine_status or classify_metrics.get('classify_warnings') or 'ok'
            report.append({'seq':seq,'route':'connector_only','status':status,'register_rate':rr,'geometry_frames':len(fr),'connector_frames':len(cfr),'keyframes_added':cn,'connector_keyframes_added':cn,'connector_median_inliers':cmed,'points_added':0,**classify_metrics})
            continue

        if route=="register":
            # ---- 方法2: 每幀 PnP 註冊，繼承 base xyz，加 keyframe ----
            nreg, rmed = add_registered_keyframes(fr, names, em, name_to_idx, seq, "REGISTER")
            status=quarantine_status or classify_metrics.get('classify_warnings') or 'ok'
            report.append({'seq':seq,'route':'register','status':status,'register_rate':rr,'geometry_frames':len(fr),'connector_frames':len(cfr),'keyframes_added':nreg+cn,'register_keyframes_added':nreg,'connector_keyframes_added':cn,'register_median_inliers':rmed,'connector_median_inliers':cmed,'points_added':0,**classify_metrics})

        else:
            if not fr:
                rollback_new_state(seq_state)
                report.append({'seq':seq,'route':'submap','status':'skipped_no_geometry_frames','register_rate':rr,'geometry_frames':0,'connector_frames':len(cfr),'keyframes_added':0,'connector_keyframes_added':0,'connector_keyframes_rolled_back':cn,'connector_median_inliers':cmed,'points_added':0,**classify_metrics})
                log(f"[{seq}] no geometry frames for SUBMAP; connector keyframes rolled back")
                continue
            # ---- 方法3: 自建 submap → Sim3 → 加 keyframe + 點 ----
            W_=Path(f"/tmp/mapupd_{seq}"); shutil.rmtree(W_,ignore_errors=True); W_.mkdir(parents=True)
            FE=W_/"f.h5"; MA=W_/"m.h5"; PA=W_/"p.txt"; feats={}
            sub_items=[(p,nm,True,em[k]) for k,(p,nm) in enumerate(zip(fr,names))]
            if a.submap_connector_bridges and cfr:
                sub_items += [(p,nm,False,cem[k]) for k,(p,nm) in enumerate(zip(cfr,cnames))]
            sub_items.sort(key=lambda x:x[1])
            sub_names=[x[1] for x in sub_items]
            sub_em=np.stack([x[3] for x in sub_items]).astype(np.float32)
            geom_names=set(names)
            sub_img_root=NB
            if a.matcher!="xfeat" and not a.allow_xfeat_submap:
                raise SystemExit(
                    "route 3 (submap + Sim3) still builds its submap with XFeat+LighterGlue; "
                    "it has NOT been migrated to EDM. Running it under --matcher edm would "
                    "silently mix frontends and emit a submap the deployed stack never sees. "
                    "Either route this sequence to register/skip, or pass --allow-xfeat-submap "
                    "to accept a mixed-frontend candidate that must stay in quarantine.")
            # The submap route is the last XFeat holdout; import it here so an
            # EDM-only run never loads XFeat at all.
            from reloc_localizer_xfeat import load_xfeat, extract_xfeat
            xf=load_xfeat(QK)
            if any(not is_geom for _,_,is_geom,_ in sub_items):
                sub_img_root=W_/"images"
                materialize_images(sub_items,sub_img_root)
            with h5py.File(FE,"w") as fd:
                for p,nm,_,_ in sub_items:
                    rgb=load_rgb(p); o=extract_xfeat(xf,rgb,QK)
                    feats[nm]={'kp':o['keypoints'].detach().cpu(),'desc':o['descriptors'].detach().cpu(),'sc':o['scores'].detach().cpu() if 'scores' in o else None,'wh':(rgb.shape[1],rgb.shape[0])}
                    fd.create_group(nm).create_dataset("keypoints",data=np.asarray(o['keypoints'].detach().cpu(),np.float32))
            pairs=set()
            for i in range(len(sub_names)):
                for j in range(i+1,min(i+a.seq_window,len(sub_names))): pairs.add((i,j))
                sims=sub_em@sub_em[i]; sims[i]=-1
                for j in np.argsort(-sims)[:a.retr_nn]: pairs.add(tuple(sorted((i,int(j)))))
            PA.write_text("\n".join(f"{sub_names[i]} {sub_names[j]}" for i,j in sorted(pairs))+"\n")
            def ff(nm):
                f=feats[nm]; d={'keypoints':f['kp'].to(DEV),'descriptors':f['desc'].to(DEV),'image_size':f['wh']}
                if f['sc'] is not None: d['scores']=f['sc'].to(DEV)
                return d
            with h5py.File(MA,"w") as fm:
                for i,j in sorted(pairs):
                    a_,b_=sub_names[i],sub_names[j]; n0=feats[a_]['kp'].shape[0]
                    try:_,_,mi=xf.match_lighterglue(ff(a_),ff(b_),min_conf=a.min_conf)
                    except Exception:mi=None
                    m0=np.full((n0,),-1,np.int32)
                    if mi is not None and len(mi):
                        arr=np.asarray(mi.detach().cpu() if hasattr(mi,'detach') else mi).astype(np.int64); m0[arr[:,0]]=arr[:,1].astype(np.int32)
                    g=fm.create_group(n2p(a_,b_)); g.create_dataset("matches0",data=m0); g.create_dataset("matching_scores0",data=np.ones((n0,),np.float32))
            H0,W0=load_rgb(fr[0]).shape[:2]
            intr=fixed_intrinsics(W0,H0,a.focal,a.k1)
            image_opts={"camera_model":"SIMPLE_RADIAL","camera_params":",".join(str(x) for x in intr)}
            rec=RC.main(W_/"sub",sub_img_root,PA,FE,MA,camera_mode=pycolmap.CameraMode.SINGLE,
                        skip_geometric_verification=True,image_list=sub_names,image_options=image_opts)
            log(f"[{seq}] submap {rec.num_reg_images()}/{len(sub_names)} imgs ({len(fr)} geometry, {len(cfr) if a.submap_connector_bridges else 0} connector), {rec.num_points3D()} pts")
            sub_pose={rec.images[i].name:rec.images[i].projection_center() for i in rec.images if rec.images[i].has_pose}
            classify_warning_list=[x for x in str(classify_metrics.get('classify_warnings','')).split(',') if x]
            bridge_best, bridge_attempts=choose_bridge_attempt(seq,sub_items,sub_pose,classify_warning_list)
            src=bridge_best['src']; dst=bridge_best['dst']; bridge_quality=bridge_best['quality']
            bridge_geom=bridge_best['bridge_geometry']; bridge_conn=bridge_best['bridge_connector']
            if len(src)<a.min_bridges:
                log(f"[{seq}] only {len(src)} bridges ({bridge_geom} geometry, {bridge_conn} connector) < {a.min_bridges} -> SKIP (no overlap with base)")
                (out/f"bridge_quality_{seq}.json").write_text(json.dumps({'seq':seq,'selected_attempt':bridge_best['label'],'attempts':[{k:v for k,v in x.items() if k not in {'src','dst','sim3_R','sim3_t'}} for x in bridge_attempts]},indent=2),encoding='utf-8')
                rollback_new_state(seq_state)
                report.append({'seq':seq,'route':'submap','status':'skipped_no_bridges','register_rate':rr,'geometry_frames':len(fr),'connector_frames':len(cfr),'keyframes_added':0,'connector_keyframes_added':0,'connector_keyframes_rolled_back':cn,'connector_median_inliers':cmed,'points_added':0,'bridges':len(src),'bridge_geometry':bridge_geom,'bridge_connector':bridge_conn,'bridge_attempt':bridge_best['label'],'bridge_qk':bridge_best['qk'],'bridge_topk':bridge_best['topk'],'bridge_min_conf':bridge_best['min_conf'],**classify_metrics}); continue
            src_arr=np.asarray(src); dst_arr=np.asarray(dst)
            s,R,t=bridge_best['sim3_scale'],bridge_best['sim3_R'],bridge_best['sim3_t']
            resid=float(bridge_best['sim3_resid_u'])
            med_bridge_ratio=float(bridge_best['bridge_median_inlier_ratio'])
            med_bridge_area=float(bridge_best['bridge_median_support_area'])
            sub_pose_spread=float(np.median(np.linalg.norm(src_arr-src_arr.mean(0),axis=1))) if len(src_arr)>1 else 0.0
            base_pose_spread=float(np.median(np.linalg.norm(dst_arr-dst_arr.mean(0),axis=1))) if len(dst_arr)>1 else 0.0
            bridge_checks=bridge_quality_checks(
                bridge_geometry=bridge_geom,
                total_bridges=len(src),
                median_inlier_ratio=med_bridge_ratio,
                median_support_area=med_bridge_area,
                min_inlier_ratio=a.bridge_min_inlier_ratio,
                min_support_area=a.bridge_min_support_area,
                min_geometry=a.bridge_min_geometry,
                min_geometry_ratio=a.bridge_min_geometry_ratio,
            )
            bridge_warnings=bridge_quality_warnings(
                bridge_geometry=bridge_geom,
                total_bridges=len(src),
                median_inlier_ratio=med_bridge_ratio,
                median_support_area=med_bridge_area,
                min_inlier_ratio=a.bridge_min_inlier_ratio,
                min_support_area=a.bridge_min_support_area,
                min_geometry=a.bridge_min_geometry,
                min_geometry_ratio=a.bridge_min_geometry_ratio,
            )
            hard_fail_reasons=bridge_hard_fail_reasons(bridge_checks)
            (out/f"bridge_quality_{seq}.json").write_text(json.dumps({'seq':seq,'selected_attempt':bridge_best['label'],'bridges':bridge_quality,'attempts':[{k:v for k,v in x.items() if k not in {'src','dst','sim3_R','sim3_t'}} for x in bridge_attempts],'checks':bridge_checks},indent=2),encoding='utf-8')
            log(f"[{seq}] Sim3 s={s:.3f} bridges={len(src)} ({bridge_geom} geometry, {bridge_conn} connector) resid={resid:.3f}u, bridge_ratio={med_bridge_ratio:.2f}, bridge_area={med_bridge_area:.3f}, attempt={bridge_best['label']} qk={bridge_best['qk']} topk={bridge_best['topk']} conf={bridge_best['min_conf']}")
            if hard_fail_reasons or (bridge_warnings and a.bridge_gate_quality):
                skip_reasons=bridge_warnings if a.bridge_gate_quality else hard_fail_reasons
                log(f"[{seq}] bridge quality gate failed: {','.join(skip_reasons)}")
                rollback_new_state(seq_state)
                report.append({'seq':seq,'route':'submap','status':"skipped_bridge_quality:"+",".join(skip_reasons),'register_rate':rr,'geometry_frames':len(fr),'connector_frames':len(cfr),'keyframes_added':0,'connector_keyframes_added':0,'connector_keyframes_rolled_back':cn,'connector_median_inliers':cmed,'points_added':0,'bridges':len(src),'bridge_geometry':bridge_geom,'bridge_connector':bridge_conn,'bridge_median_inlier_ratio':med_bridge_ratio,'bridge_median_support_area':med_bridge_area,'bridge_sub_pose_spread':sub_pose_spread,'bridge_base_pose_spread':base_pose_spread,'bridge_attempt':bridge_best['label'],'bridge_qk':bridge_best['qk'],'bridge_topk':bridge_best['topk'],'bridge_min_conf':bridge_best['min_conf'],'sim3_resid_u':resid,**classify_metrics}); continue
            geom_obs_by_pid={}
            valid_point_ids=set()
            for pid,p3 in rec.points3D.items():
                nobs=sum(1 for e in p3.track.elements if rec.images[e.image_id].name in geom_names)
                geom_obs_by_pid[int(pid)]=nobs
                if nobs>=a.min_geometry_observations:
                    valid_point_ids.add(int(pid))
            # keyframes (feats + Sim3'd per-kp xyz + selected deployment VPR)
            nkf=0
            for iid in rec.images:
                im=rec.images[iid]
                if not im.has_pose: continue
                nm=im.name
                if nm not in geom_names:
                    continue
                rgb=load_rgb(sub_img_root/nm); H,W=rgb.shape[:2]; o=extract_xfeat(xf,rgb,QK)
                kp=np.asarray(o['keypoints'].detach().cpu()); xyz=np.full((len(kp),3),np.nan,np.float32)
                for pi,p2 in enumerate(im.points2D):
                    if pi>=len(kp): break
                    if p2.has_point3D() and int(p2.point3D_id) in valid_point_ids:
                        xyz[pi]=s*(R@np.asarray(rec.point3D(p2.point3D_id).xyz))+t
                new_kf.append((nm,{'feats':{'keypoints':o['keypoints'].detach().cpu(),'descriptors':o['descriptors'].detach().cpu(),'scores':o['scores'].detach().cpu() if 'scores' in o else None,'image_size':(W,H)},'xyz':xyz}))
                Csub,_yaw_sub,fwd_sub=pose_center_yaw_from_image(im)
                Cbase,yaw_base=transform_center_yaw(Csub,fwd_sub,s,R,t)
                new_centers.append(Cbase); new_yaws.append(yaw_base); new_covis[nm]=[]
                new_glob.append(bundle_global(rgb,nm,em,name_to_idx)); nkf+=1
            # points for PLY (Sim3'd + colored)
            npt=0
            if not a.no_ply:
                rec.extract_colors_for_all_images(str(sub_img_root))
                kept_pts=[p for pid,p in rec.points3D.items() if int(pid) in valid_point_ids]
                sxyz=np.array([p.xyz for p in kept_pts]); srgb=np.array([p.color for p in kept_pts])
                if len(sxyz): ply_xyz.append(s*(sxyz@R.T)+t); ply_rgb.append(srgb); npt=len(sxyz)
            status=quarantine_status or classify_metrics.get('classify_warnings') or (";".join(bridge_warnings) if bridge_warnings else 'ok')
            report.append({'seq':seq,'route':'submap','status':status,'register_rate':rr,'geometry_frames':len(fr),'connector_frames':len(cfr),'sim3_scale':float(s),'bridges':len(src),'bridge_geometry':bridge_geom,'bridge_connector':bridge_conn,'bridge_median_inlier_ratio':med_bridge_ratio,'bridge_median_support_area':med_bridge_area,'bridge_sub_pose_spread':sub_pose_spread,'bridge_base_pose_spread':base_pose_spread,'bridge_attempt':bridge_best['label'],'bridge_qk':bridge_best['qk'],'bridge_topk':bridge_best['topk'],'bridge_min_conf':bridge_best['min_conf'],'sim3_resid_u':resid,'keyframes_added':nkf+cn,'submap_keyframes_added':nkf,'connector_keyframes_added':cn,'connector_median_inliers':cmed,'points_added':npt,'raw_submap_points':rec.num_points3D(),'geometry_supported_points':len(valid_point_ids),**classify_metrics})
            (out/f"bridge_quality_{seq}.json").write_text(json.dumps({'seq':seq,'selected_attempt':bridge_best['label'],'bridges':bridge_quality,'attempts':[{k:v for k,v in x.items() if k not in {'src','dst','sim3_R','sim3_t'}} for x in bridge_attempts],'checks':bridge_checks},indent=2),encoding='utf-8')
            log(f"[{seq}] SUBMAP done: +{nkf} submap keyframes, +{cn} connector keyframes, +{npt} points")

    # ---- write observation / old-point-health stats ----
    obs_payload={'version':1,'grid':int(a.health_grid),'sessions':{},'changed_region_candidates':[]}
    global_ref_hits=Counter(); global_anchor_hits=Counter()
    for seq,s in obs_sessions.items():
        global_ref_hits.update(s['base_ref_hits']); global_anchor_hits.update(s['anchor_hits'])
        tiles=[]
        for tk,t in sorted(s['tile'].items()):
            support_ratio=float(t['inliers']/max(1,t['query_kp']))
            match_ratio=float(t['matches']/max(1,t['query_kp']))
            row={'tile':tk,'frames':int(t['frames']),'query_kp':int(t['query_kp']),
                 'matches':int(t['matches']),'inliers':int(t['inliers']),
                 'match_ratio':match_ratio,'support_ratio':support_ratio,
                 'low_support_frames':int(t['low_support_frames'])}
            tiles.append(row)
            if t['low_support_frames']>=2:
                cand=dict(row); cand['seq']=seq
                cand['reason']='localized_frames_with_low_old-map_tile_support'
                obs_payload['changed_region_candidates'].append(cand)
        obs_payload['sessions'][seq]={
            'frames':int(s['frames']),'localized':int(s['localized']),'failed':int(s['failed']),
            'success_rate':float(s['localized']/max(1,s['frames'])),
            'median_support_area':float(np.median(s['support_areas'])) if s['support_areas'] else 0.0,
            'median_inlier_ratio':float(np.median(s['inlier_ratios'])) if s['inlier_ratios'] else 0.0,
            'top_base_ref_hits':counter_top(s['base_ref_hits'],50),
            'top_anchor_hits':counter_top(s['anchor_hits'],200),
            'tiles':tiles
        }
    obs_payload['global_top_base_ref_hits']=counter_top(global_ref_hits,100)
    obs_payload['global_top_anchor_hits']=counter_top(global_anchor_hits,500)
    obs_payload['changed_region_candidates']=sorted(
        obs_payload['changed_region_candidates'],
        key=lambda x:(-x['low_support_frames'],x['support_ratio'],x['seq'],x['tile'])
    )
    obsp=out/"observation_stats.json"
    obsp.write_text(json.dumps(obs_payload,indent=2),encoding='utf-8')
    log(f"OBSERVATION STATS -> {obsp}")

    # ---- write updated bundle ----
    for nm,r in new_kf: refs[nm]=r
    new_names=[nm for nm,_ in new_kf]
    if len(new_centers)!=len(new_kf) or len(new_yaws)!=len(new_kf):
        raise RuntimeError(f"tracking metadata count mismatch: keyframes={len(new_kf)} centers={len(new_centers)} yaws={len(new_yaws)}")
    base_centers=np.asarray(b.get('ref_centers',[]),np.float32)
    base_yaws=np.asarray(b.get('ref_yaws',[]),np.float32)
    if base_centers.shape!=(len(base_names),3) or base_yaws.shape!=(len(base_names),):
        raise RuntimeError("base bundle must contain complete ref_centers/ref_yaws before producing a flight-deployable update")
    b2=dict(b); b2['refs']=refs; b2['ref_names']=base_names+new_names
    b2['ref_global']=np.vstack([bdesc]+([np.stack(new_glob).astype(np.float32)] if new_glob else []))
    b2['ref_centers']=np.vstack([base_centers]+([np.stack(new_centers).astype(np.float32)] if new_centers else []))
    b2['ref_yaws']=np.concatenate([base_yaws]+([np.asarray(new_yaws,np.float32)] if new_yaws else []))
    ref_stability, stability_meta=build_ref_stability(
        b2['ref_names'],
        base_names,
        base_stability,
        obs_payload,
        report,
        half_life_sessions=a.stability_half_life_sessions,
    )
    b2['ref_stability']=ref_stability
    covis=dict(b.get('covis',{}) or {})
    covis.update(new_covis)
    b2['covis']=complete_covisibility(b2['ref_names'],b2['ref_centers'],covis,int(b.get('meta',{}).get('tracking_top_covis',40) or 40))
    covis_lens=[len(v) for v in b2['covis'].values()]
    meta=dict(b2.get('meta',{}))
    meta.update({"vpr":"megaloc-8448","vpr_input":322,"bundle_vpr":"megaloc",
                 "global_descriptor_source":"MegaLoc for retrieval and deployment"})
    meta.update({"fixed_intrinsics_by_resolution":{f"{w}x{h}":p for (w,h),p in FIXED_INTRINSICS.items()}})
    meta.update({"observation_stats":"observation_stats.json",
                 "changed_region_candidates":len(obs_payload['changed_region_candidates'])})
    meta.update({"ref_stability":stability_meta,
                 "ref_stability_summary":stability_summary(ref_stability),
                 "stability_rerank_weight":float(a.stability_rerank_weight)})
    meta.update({"tracking_metadata":True,
                 "tracking_model":str(a.base_model),
                 "tracking_metadata_source":"base inherited; registered keyframes from PnP; submap keyframes from Sim3-aligned submap poses; covis completed by nearest camera centers",
                 "tracking_top_covis":int(b.get('meta',{}).get('tracking_top_covis',40) or 40),
                 "tracking_median_covis_degree":float(np.median(covis_lens)) if covis_lens else 0.0,
                 "tracking_keyframes":len(b2['ref_names'])})
    b2['meta']=meta
    outb=out/"reloc_map_updated.pt"; torch.save(b2,outb)
    log(f"UPDATED bundle: base {len(base_names)} + {len(new_kf)} new kf -> {outb}")

    # ---- write real-RGB PLY (base colored + new-coverage points) ----
    if not a.no_ply:
        base=pycolmap.Reconstruction(a.base_model); base.extract_colors_for_all_images(a.base_images)
        bx=np.array([p.xyz for p in base.points3D.values()]); bc=np.array([p.color for p in base.points3D.values()])
        X=np.vstack([bx]+ply_xyz)[::a.ply_stride]; C=np.vstack([bc]+ply_rgb).astype(np.uint8)[::a.ply_stride]
        outp=out/"latest_map_realrgb.ply"
        with open(outp,"wb") as f:
            f.write(f"ply\nformat binary_little_endian 1.0\nelement vertex {len(X)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n".encode())
            buf=np.empty(len(X),dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('r','u1'),('g','u1'),('b','u1')])
            buf['x'],buf['y'],buf['z']=X[:,0],X[:,1],X[:,2]; buf['r'],buf['g'],buf['b']=C[:,0],C[:,1],C[:,2]
            f.write(buf.tobytes())
        log(f"PLY: {len(X)} pts -> {outp}")

    # ---- report ----
    rep=out/"update_report.md"; lines=[f"# Map Update Report  ({time.strftime('%Y-%m-%d %H:%M')})","",
        f"- base bundle: `{a.base_bundle}`  ({len(base_names)} keyframes)",
        f"- new data: `{a.new_data}`  videos: {vids}",
        f"- connector data: `{a.connector_data}`",
        f"- output bundle: `{outb}`  ({len(b2['ref_names'])} keyframes, +{len(new_kf)})","",
        "## 路由與統計","","| 影片 | 路由 | 狀態 | register_rate | sampled med inliers | support area | geom/conn frames | keyframes+ | conn kf+ | points+ | raw/geom-supported pts | Sim3 s | bridges(g/c) | bridge params | bridge ratio | bridge area | pose spread sub/base | resid(u) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    def fmt_rr(v): return "-" if v is None else f"{v:.0%}"
    def fmt(v): return "-" if v is None else (f"{v:.3f}" if isinstance(v,float) else str(v))
    for r in report:
        gf=r.get('geometry_frames'); cf=r.get('connector_frames')
        frames="-" if gf is None and cf is None else f"{fmt(gf)}/{fmt(cf)}"
        rawpts="-" if r.get('raw_submap_points') is None else f"{fmt(r.get('raw_submap_points'))}/{fmt(r.get('geometry_supported_points'))}"
        bridges="-" if r.get('bridges') is None else f"{fmt(r.get('bridges'))} ({fmt(r.get('bridge_geometry'))}/{fmt(r.get('bridge_connector'))})"
        spread="-" if r.get('bridge_sub_pose_spread') is None else f"{fmt(r.get('bridge_sub_pose_spread'))}/{fmt(r.get('bridge_base_pose_spread'))}"
        bparams="-" if r.get('bridge_attempt') is None else f"{r.get('bridge_attempt')} q{r.get('bridge_qk')}/k{r.get('bridge_topk')}/c{r.get('bridge_min_conf')}"
        lines.append(f"| {r['seq']} | {r['route']} | {r.get('status','ok')} | {fmt_rr(r.get('register_rate'))} | {fmt(r.get('sampled_median_inliers'))} | {fmt(r.get('sampled_median_support_area'))} | {frames} | {fmt(r.get('keyframes_added'))} | {fmt(r.get('connector_keyframes_added'))} | {fmt(r.get('points_added'))} | {rawpts} | {fmt(r.get('sim3_scale'))} | {bridges} | {bparams} | {fmt(r.get('bridge_median_inlier_ratio'))} | {fmt(r.get('bridge_median_support_area'))} | {spread} | {fmt(r.get('sim3_resid_u'))} |")
    lines+=["","## 參數快照",""]
    for k,v in sorted(vars(a).items()): lines.append(f"- `{k}` = `{v}`")
    rep.write_text("\n".join(lines))
    summary=out/"update_summary.json"
    summary.write_text(json.dumps({
        "version":1,
        "created_at":time.strftime('%Y-%m-%d %H:%M:%S'),
        "rows":report,
    },indent=2),encoding='utf-8')
    log(f"REPORT -> {rep}")
    log(f"SUMMARY -> {summary}")
    log("DONE.")

if __name__=="__main__":
    main()
