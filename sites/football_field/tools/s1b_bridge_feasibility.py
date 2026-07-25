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
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ts_common import BUILD, RUNS, log, read_json, write_json  # noqa: E402
import s1_motion_scan as s1  # noqa: E402

TOPK = 5
CACHE = Path("/tmp/claude-1000/-home-cihcilab/3fd611d2-d089-48b6-aa46-06dcdc84a329/scratchpad/ts_megaloc.pkl")


def descriptors(run_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if CACHE.exists():
        return pickle.loads(CACHE.read_bytes())
    out = {}
    for v in BUILD:
        log(f"MegaLoc {v.seq} ...")
        out[v.seq] = s1.megaloc_descriptors(v)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_bytes(pickle.dumps(out))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-name", default="football_field_v1")
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

    log("\n" + "=" * 72)
    if cross_frac < 0.02:
        verdict = ("FORCE REQUIRED: retrieval essentially never proposes a "
                   "cross-direction pair on its own")
    elif cross_frac < 0.10:
        verdict = ("FORCE STRONGLY ADVISED: retrieval proposes cross-direction "
                   "pairs too rarely to rely on")
    else:
        verdict = "NATURAL RETRIEVAL MAY SUFFICE: cross-direction pairs are proposed often"
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


if __name__ == "__main__":
    main()
