#!/usr/bin/env python3
"""S1b -- do we actually NEED forced cross-video pairs, and where?

Layer 1 of the cross-flight co-visibility discipline: cheap VPR candidates first,
before paying for any geometry.

THE QUESTION THIS ANSWERS
  The map must come out as ONE connected component. The 4 forward videos will link
  to each other easily; so will the 3 reverse ones. Everything hinges on whether
  the forward cluster reaches the reverse cluster at all.

  S1 measured that MegaLoc cannot RETRIEVE across a direction reversal (same-
  direction cosine 0.31-0.59, cross-direction 0.10-0.17). But that is not the same
  claim as "forward and reverse never see the same structure". Two very different
  worlds follow:

    (a) they genuinely never co-observe -> forcing pairs yields ZERO valid matches.
        It costs compute and buys nothing. Accept a split, or drop a direction.
    (b) they DO co-observe somewhere (a turnaround, an open stretch, distant
        skyline) but VPR cannot find it -> forcing is the ONLY way to link them,
        and it is load-bearing.

  Guessing wrong is expensive in both directions, so measure.

WHAT IT REPORTS
  - the cross-video similarity matrix, split same-direction vs cross-direction
  - for each fwd x rev pair: the best candidate frame pairs and WHERE in the route
    they sit (a turnaround shows up as candidates clustered at one end)
  - a verdict: are there enough plausible cross-direction candidates to be worth
    forcing, and if so, over which time ranges

NOTE ON THE DESCRIPTOR
  gluemap retrieves with DINO-SALAD, not MegaLoc. Both are global descriptors and
  both fail the same way on a reversed viewpoint, so MegaLoc is a fair proxy for
  "will retrieval find this on its own" -- and it is what the deployed localizer
  uses anyway.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ts_common import BUILD, RUNS, Gate, log, read_json, sha256, write_json  # noqa: E402
import s1_motion_scan as s1  # noqa: E402

TOPK = 5


def retrieval_verdict(cross_frac: float, total_mutual: int) -> str:
    if total_mutual == 0:
        return "FORCE REQUIRED: zero mutual cross-direction candidates"
    if cross_frac < 0.10:
        return (
            "FORCE REQUIRED: retrieval proposes cross-direction "
            "pairs too rarely to rely on"
        )
    return "NATURAL RETRIEVAL MAY SUFFICE: cross-direction pairs are proposed often"


def cache_paths(run_dir: Path) -> tuple[Path, Path]:
    root = Path(run_dir).resolve() / "cache" / "s1b"
    return root / "megaloc_descriptors.pkl", root / "megaloc_descriptors.meta.json"


def _material_digest(paths: list[Path], config: dict) -> str:
    material = {
        "inputs": [
            {"path": str(path.resolve()), "sha256": sha256(path)}
            for path in sorted((Path(path) for path in paths), key=lambda item: str(item.resolve()))
        ],
        "config": config,
    }
    return hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def write_cache_metadata(
    metadata_path: Path,
    cache_path: Path,
    material_paths: list[Path],
    config: dict,
) -> None:
    write_json(
        metadata_path,
        {
            "schema_version": "run-local-cache-v1",
            "cache_path": str(cache_path.resolve()),
            "cache_sha256": sha256(cache_path) if cache_path.is_file() else None,
            "material_sha256": _material_digest(material_paths, config),
            "config": config,
        },
    )


def cache_is_fresh(
    metadata_path: Path, material_paths: list[Path], config: dict
) -> bool:
    try:
        metadata = read_json(metadata_path)
        cache_path = Path(metadata["cache_path"])
        return (
            metadata.get("schema_version") == "run-local-cache-v1"
            and metadata.get("material_sha256") == _material_digest(material_paths, config)
            and cache_path.is_file()
            and metadata.get("cache_sha256") == sha256(cache_path)
        )
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError):
        return False


def descriptors(run_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    cache, metadata = cache_paths(run_dir)
    inputs = [run_dir / "motion_manifest.json", Path(s1.__file__).resolve()]
    config = {"topk": TOPK, "sequences": [video.seq for video in BUILD]}
    if cache_is_fresh(metadata, inputs, config):
        return pickle.loads(cache.read_bytes())
    out = {}
    for v in BUILD:
        log(f"MegaLoc {v.seq} ...")
        out[v.seq] = s1.megaloc_descriptors(v)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(out))
    write_cache_metadata(metadata, cache, inputs, config)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-name", default="target_site_v1")
    args = ap.parse_args()
    run_dir = RUNS / args.run_name

    mm = read_json(run_dir / "motion_manifest.json")
    direction = {v.seq: v.direction for v in BUILD}
    for seq, d in mm.get("directions", {}).items():
        direction[seq] = d["direction"]
    log(f"directions: {direction}")

    desc = descriptors(run_dir)
    seqs = [v.seq for v in BUILD]

    # --- the similarity landscape -------------------------------------------
    log("\n=== pairwise video similarity (median of each frame's best match) ===")
    log(f"{'':>14s} " + " ".join(f"{s[:9]:>9s}" for s in seqs))
    pair_stats = {}
    for a in seqs:
        row = []
        for b in seqs:
            if a == b:
                row.append(float("nan"))
                continue
            sim = desc[a][1] @ desc[b][1].T
            best = sim.max(axis=1)
            row.append(float(np.median(best)))
            pair_stats[(a, b)] = {
                "median_best": float(np.median(best)),
                "p90_best": float(np.percentile(best, 90)),
                "max": float(sim.max()),
                "same_direction": direction[a] == direction[b],
            }
        log(f"{a:>14s} " + " ".join("      -  " if np.isnan(x) else f"{x:9.3f}" for x in row))

    same = [v["median_best"] for v in pair_stats.values() if v["same_direction"]]
    cross = [v["median_best"] for v in pair_stats.values() if not v["same_direction"]]
    log(f"\nsame-direction  median-best: min={min(same):.3f} med={np.median(same):.3f} max={max(same):.3f}  (n={len(same)})")
    log(f"cross-direction median-best: min={min(cross):.3f} med={np.median(cross):.3f} max={max(cross):.3f}  (n={len(cross)})")
    log(f"  -> retrieval gap: {np.median(same) / max(np.median(cross), 1e-9):.1f}x")

    # --- would SALAD-style top-K retrieval EVER propose a cross-direction pair? -
    # This is the whole question. Pool every frame of every video, retrieve each
    # frame's top-K nearest neighbours from OTHER videos, and count how many of
    # those land on a video flying the other way.
    log(f"\n=== of each frame's top-{TOPK} cross-video neighbours, how many are cross-direction? ===")
    natural = {}
    for a in seqs:
        da = desc[a][1]
        others = [(b, desc[b][1]) for b in seqs if b != a]
        offs, mats, owner = [], [], []
        for b, db in others:
            offs.append(len(owner))
            mats.append(db)
            owner += [b] * len(db)
        sim = da @ np.vstack(mats).T
        topk = np.argsort(-sim, axis=1)[:, :TOPK]
        own = np.array(owner)
        hit = sum(1 for r in topk.reshape(-1) if direction[own[r]] != direction[a])
        tot = topk.size
        natural[a] = {"cross_direction_neighbours": hit, "total": tot, "frac": hit / tot}
        log(f"  {a:>14s} ({direction[a]}): {hit:5d}/{tot:5d} = {hit / tot * 100:5.1f}% cross-direction")

    # --- where are the plausible cross-direction candidates? -----------------
    # A real turnaround shows up as candidates clustered at one end of the route.
    log("\n=== best cross-direction candidate pairs (fwd x rev), and WHERE ===")
    fwd = [s for s in seqs if direction[s] == "fwd"]
    rev = [s for s in seqs if direction[s] == "rev"]
    bridges = {}
    for a in fwd:
        ta, da = desc[a]
        for b in rev:
            tb, db = desc[b]
            sim = da @ db.T
            # A candidate only counts if BOTH sides retrieve each other (mutual
            # top-1). One-sided similarity is what vegetation aliasing looks like.
            ab = sim.argmax(axis=1)
            ba = sim.argmax(axis=0)
            mutual = [(i, int(ab[i])) for i in range(len(ab)) if int(ba[int(ab[i])]) == i]
            scored = sorted(mutual, key=lambda ij: -sim[ij[0], ij[1]])
            top = [{
                "t_fwd": float(ta[i]), "t_rev": float(tb[j]),
                "sim": float(sim[i, j]),
                "frac_fwd": float(ta[i] / ta[-1]), "frac_rev": float(tb[j] / tb[-1]),
            } for i, j in scored[:8]]
            bridges[f"{a}|{b}"] = {"n_mutual": len(mutual), "top": top}
            if top:
                where = ", ".join(
                    f"{c['frac_fwd'] * 100:.0f}%/{c['frac_rev'] * 100:.0f}% s={c['sim']:.2f}"
                    for c in top[:5]
                )
                log(f"  {a:>12s} x {b:<12s} mutual={len(mutual):3d}  best(route%): {where}")
            else:
                log(f"  {a:>12s} x {b:<12s} mutual=  0  -- NO mutual candidate at all")

    # --- verdict --------------------------------------------------------------
    total_mutual = sum(v["n_mutual"] for v in bridges.values())
    strong = sum(1 for v in bridges.values() for c in v["top"] if c["sim"] > 0.35)
    cross_frac = float(np.mean([v["frac"] for v in natural.values()]))

    verdict = retrieval_verdict(cross_frac, total_mutual)
    log(f"VERDICT: {verdict}")
    log(f"  natural cross-direction retrieval : {cross_frac * 100:.2f}% of top-{TOPK} neighbours")
    log(f"  mutual fwd x rev candidates       : {total_mutual} across {len(bridges)} video pairs")
    log(f"  ... of which similarity > 0.35    : {strong}")
    if total_mutual == 0:
        log("  WARNING: zero mutual candidates ANYWHERE. Forward and reverse may")
        log("           genuinely share no co-visible surface. Forcing pairs would")
        log("           then produce no valid matches -- expect an honest split.")
    log("=" * 72)

    write_json(run_dir / "bridge_feasibility.json", {
        "topk": TOPK,
        "directions": direction,
        "pair_stats": {f"{a}|{b}": v for (a, b), v in pair_stats.items()},
        "natural_cross_direction_retrieval": natural,
        "cross_direction_fraction": cross_frac,
        "fwd_x_rev_bridges": bridges,
        "total_mutual": total_mutual,
        "verdict": verdict,
    })
    log(f"\nwrote {run_dir / 'bridge_feasibility.json'}")
    g = Gate(
        "S1b_bridge_feasibility",
        {"G1b.1"},
        script_path=Path(__file__),
        input_artifacts={"motion_manifest": run_dir / "motion_manifest.json"},
        source_files=[Path(__file__).with_name("ts_common.py")],
    )
    g.record_predecessor_gate(
        "S1_motion",
        run_dir / "gates" / "S1_motion.json",
        expected_stage="S1_motion",
    )
    g.check(
        "G1b.1",
        total_mutual > 0,
        "at least one mutual fwd x rev candidate exists",
        total_mutual=total_mutual,
        cross_frac=cross_frac,
        verdict=verdict,
    )
    g.write(run_dir)


if __name__ == "__main__":
    main()
