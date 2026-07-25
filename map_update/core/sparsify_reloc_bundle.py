#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Post-validation localization-bundle sparsification.

This is intentionally conservative: it creates a separate candidate bundle and
report. It never overwrites the validated input bundle. The first-stage scoring
uses observation stats from map_update_tool.py when available, plus an ordered
K-cover per sequence/prefix so every video segment keeps coverage.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def prefix_of(name: str) -> str:
    return name.split("/", 1)[0]


def frame_number(name: str) -> int:
    m = re.search(r"frame_(\d+)", name)
    return int(m.group(1)) if m else 0


def load_ref_scores(obs_path: str | None) -> dict[str, float]:
    if not obs_path:
        return {}
    p = Path(obs_path)
    if not p.exists():
        raise FileNotFoundError(p)
    data = json.loads(p.read_text(encoding="utf-8"))
    scores = defaultdict(float)
    for item in data.get("global_top_base_ref_hits", []):
        scores[item["key"]] += float(item["count"])
    for sess in data.get("sessions", {}).values():
        for item in sess.get("top_base_ref_hits", []):
            scores[item["key"]] += 0.25 * float(item["count"])
    if scores:
        mx = max(scores.values())
        if mx > 0:
            for k in list(scores):
                scores[k] = scores[k] / mx
    return dict(scores)


def combined_score(name: str, scores: dict[str, float], stability: dict[str, float],
                   stability_weight: float) -> float:
    return float(scores.get(name, 0.0)) + float(stability_weight) * float(stability.get(name, 1.0))


def select_indices(names, scores, target_fraction, min_per_prefix, keep_prefixes,
                   stability: dict[str, float] | None = None, stability_weight: float = 0.35):
    stability = stability or {}
    groups = defaultdict(list)
    for i, name in enumerate(names):
        groups[prefix_of(name)].append(i)

    keep = set()
    per_prefix = {}
    for pref, idxs in sorted(groups.items()):
        idxs = sorted(idxs, key=lambda i: (frame_number(names[i]), names[i]))
        if pref in keep_prefixes:
            quota = len(idxs)
        else:
            quota = max(min_per_prefix, int(math.ceil(len(idxs) * target_fraction)))
            quota = min(quota, len(idxs))

        # Ordered K-cover: keep approximately uniform temporal coverage.
        if quota > 0:
            for j in np.linspace(0, len(idxs) - 1, num=quota, dtype=int):
                keep.add(idxs[int(j)])

        # Observation-score top-up: preserve refs repeatedly useful for PnP.
        score_quota = max(1, quota // 4) if quota else 0
        ranked = sorted(
            idxs,
            key=lambda i: combined_score(names[i], scores, stability, stability_weight),
            reverse=True,
        )
        keep.update(ranked[:score_quota])

        # Trim back to quota only if the prefix is not forced to keep all.
        if pref not in keep_prefixes:
            kept = [i for i in idxs if i in keep]
            if len(kept) > quota:
                ranked_keep = sorted(
                    kept,
                    key=lambda i: (
                        combined_score(names[i], scores, stability, stability_weight),
                        -abs(idxs.index(i) - len(idxs) / 2),
                    ),
                    reverse=True,
                )
                for i in ranked_keep[quota:]:
                    keep.discard(i)
        per_prefix[pref] = {"input": len(idxs), "kept": len([i for i in idxs if i in keep])}

    return sorted(keep), per_prefix


def subset_aligned(value, keep_idx):
    if isinstance(value, np.ndarray) and len(value) >= max(keep_idx, default=-1) + 1:
        return value[keep_idx]
    if torch.is_tensor(value) and value.shape[0] >= max(keep_idx, default=-1) + 1:
        return value[keep_idx]
    if isinstance(value, list) and len(value) >= max(keep_idx, default=-1) + 1:
        return [value[i] for i in keep_idx]
    return value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--observation-stats")
    ap.add_argument("--target-fraction", type=float, default=0.85)
    ap.add_argument("--min-per-prefix", type=int, default=40)
    ap.add_argument("--keep-prefix", action="append", default=[],
                    help="Prefix/sequence to keep completely; may be repeated.")
    ap.add_argument("--stability-weight", type=float, default=0.35,
                    help="Weight for ref_stability during score top-up and trimming.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    bundle_path = Path(args.bundle)
    out_path = Path(args.out)
    report_path = out_path.with_suffix(out_path.suffix + ".sparsify_report.json")

    b = torch.load(bundle_path, map_location="cpu", weights_only=False)
    names = list(b["ref_names"])
    scores = load_ref_scores(args.observation_stats)
    ref_stability = np.asarray(b.get("ref_stability", np.ones(len(names), dtype=np.float32)), dtype=np.float32)
    if ref_stability.shape != (len(names),):
        ref_stability = np.ones(len(names), dtype=np.float32)
    stability = {name: float(ref_stability[i]) for i, name in enumerate(names)}
    keep_idx, per_prefix = select_indices(
        names,
        scores,
        args.target_fraction,
        args.min_per_prefix,
        set(args.keep_prefix),
        stability=stability,
        stability_weight=args.stability_weight,
    )
    keep_names = [names[i] for i in keep_idx]

    report = {
        "bundle": str(bundle_path),
        "out": str(out_path),
        "dry_run": bool(args.dry_run),
        "input_keyframes": len(names),
        "kept_keyframes": len(keep_names),
        "removed_keyframes": len(names) - len(keep_names),
        "target_fraction": args.target_fraction,
        "min_per_prefix": args.min_per_prefix,
        "keep_prefix": args.keep_prefix,
        "stability_weight": args.stability_weight,
        "stability_summary": {
            "min": float(ref_stability.min()) if len(ref_stability) else 0.0,
            "median": float(np.median(ref_stability)) if len(ref_stability) else 0.0,
            "max": float(ref_stability.max()) if len(ref_stability) else 0.0,
        },
        "per_prefix": per_prefix,
    }

    if not args.dry_run:
        out = dict(b)
        out["ref_names"] = keep_names
        out["refs"] = {name: b["refs"][name] for name in keep_names if name in b["refs"]}
        out["ref_global"] = np.asarray(b["ref_global"])[keep_idx].astype(np.float32)
        for key in ("ref_centers", "ref_yaws"):
            if key in out:
                out[key] = subset_aligned(out[key], keep_idx)
        if "ref_stability" in out:
            out["ref_stability"] = subset_aligned(out["ref_stability"], keep_idx)
        if "covis" in out and isinstance(out["covis"], dict):
            old_to_new = {old: new for new, old in enumerate(keep_idx)}
            covis = {}
            for name in keep_names:
                vals = b["covis"].get(name, [])
                covis[name] = [old_to_new[i] for i in vals if i in old_to_new]
            out["covis"] = covis
        meta = dict(out.get("meta", {}))
        meta.update({
            "sparsified_from": str(bundle_path),
            "sparsify_input_keyframes": len(names),
            "sparsify_kept_keyframes": len(keep_names),
            "sparsify_target_fraction": args.target_fraction,
            "sparsify_observation_stats": args.observation_stats,
            "sparsify_stability_weight": args.stability_weight,
        })
        out["meta"] = meta
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(out, out_path)

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"report -> {report_path}")
    if args.dry_run:
        log(f"DRY RUN kept {len(keep_names)}/{len(names)} keyframes")
    else:
        log(f"wrote {out_path} kept {len(keep_names)}/{len(names)} keyframes")


if __name__ == "__main__":
    main()
