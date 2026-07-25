#!/usr/bin/env python3
"""S0 -- lock the corpus and prove the build/test split before anything is built.

Writes corpus_manifest.json (sha256 + probed streams) and runs the S0 gates.

The 16:9 gate is not cosmetic. EDM plain-resizes every reference into a 1024x576
canvas and rescales keypoints by a SINGLE width-derived factor applied to both
axes (build_reloc_map_edm.py:386). That is exact only while every source is 16:9.
A 4:3 source would be squashed and its y-coordinates would be wrong by a third,
silently, with a perfectly healthy-looking map.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ts_common import (  # noqa: E402
    ASPECT, BUILD, DROPPED, EXCLUDED_DIRS, RUNS, SUBFOLDER_REGEX, TEST,
    Gate, ffprobe, log, required_check_ids, resolution_groups, sha256,
    stage_material_artifacts, write_json,
)
from ts_intrinsics import CANDIDATES, cameras_for  # noqa: E402


TS_COMMON = Path(__file__).with_name("ts_common.py")
TS_INTRINSICS = Path(__file__).with_name("ts_intrinsics.py")


def probe_all(videos, *, want_hash: bool) -> list[dict]:
    out = []
    for v in videos:
        if not v.path.exists():
            raise SystemExit(f"missing video: {v.path}")
        p = ffprobe(v.path)
        tags = p["tags"]
        rec = {
            "seq": v.seq,
            "rel": v.rel,
            "declared": {"width": v.width, "height": v.height, "fps": round(v.fps, 4)},
            "probed": {k: p[k] for k in ("width", "height", "fps", "nb_frames", "duration", "codec")},
            "aspect": p["width"] / p["height"],
            "aspect_err": p["width"] / p["height"] - ASPECT,
            "parrot": tags.get("make", "") == "Parrot",
            "make": tags.get("make", ""),
            "creation_time": tags.get("creation_time", ""),
            "direction": v.direction,
            "epoch": v.epoch,
            "bytes": v.path.stat().st_size,
        }
        if want_hash:
            log(f"  sha256 {v.rel} ({rec['bytes'] / 1e9:.2f} GB) ...")
            rec["sha256"] = sha256(v.path)
            rec["source_rel"] = v.rel
            rec["source_sha256"] = rec["sha256"]
        out.append(rec)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-name", default="target_site_v1")
    ap.add_argument("--no-hash", action="store_true",
                    help="skip sha256 (10 GB of reads); for iterating only")
    args = ap.parse_args()

    run_dir = RUNS / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log(f"S0 corpus lock -> {run_dir}")

    build = probe_all(BUILD, want_hash=not args.no_hash)
    test = probe_all(TEST, want_hash=not args.no_hash)

    g = Gate(
        "S0_corpus",
        required_check_ids("S0_corpus"),
        script_path=Path(__file__),
        input_artifacts=stage_material_artifacts("S0_corpus", run_dir),
        source_files=[TS_COMMON, TS_INTRINSICS],
    )

    # G0.1 -- the build set is exactly what we declared, and it probes as declared
    mismatched = [
        r["rel"] for r in build
        if (r["probed"]["width"], r["probed"]["height"]) != (r["declared"]["width"], r["declared"]["height"])
    ]
    g.check("G0.1a", len(build) == 7 and not mismatched,
            "7 build videos, all probing at their declared resolution"
            if not mismatched else f"resolution mismatch: {mismatched}",
            n_build=len(build), mismatched=mismatched)

    hashes = [r.get("source_sha256") for r in build + test]
    if all(hashes):
        g.check("G0.1b", len(set(hashes)) == len(hashes),
                f"all {len(hashes)} videos are fully hashed and content-distinct",
                n_hashed=len(hashes), n_unique=len(set(hashes)))
    else:
        g.incomplete(
            "G0.1b",
            "--no-hash is diagnostic only; release corpus requires all nine hashes",
            n_hashed=sum(bool(h) for h in hashes),
            n_required=len(build) + len(test),
        )

    # G0.2 -- no test video is in the build set, by path OR by content
    build_rels = {r["rel"] for r in build}
    test_rels = {r["rel"] for r in test}
    overlap = build_rels & test_rels
    bh = {r["sha256"] for r in build if r.get("sha256")}
    th = {r["sha256"] for r in test if r.get("sha256")}
    g.check("G0.2a", not overlap,
            "test video paths are disjoint from the seven build paths"
            if not overlap else f"PATH LEAK: {sorted(overlap)}",
            build_paths=sorted(build_rels), test_paths=sorted(test_rels),
            path_overlap=sorted(overlap))
    if all(hashes):
        g.check("G0.2b", not (bh & th),
                "all nine hashes are present and build/test content is disjoint"
                if not (bh & th) else f"CONTENT LEAK: {sorted(bh & th)}",
                n_build_hashes=len(bh), n_test_hashes=len(th),
                hash_overlap=sorted(bh & th))
    else:
        g.incomplete(
            "G0.2b",
            "build/test content separation cannot be proved without all nine hashes",
            n_build_hashes=len(bh), n_test_hashes=len(th),
        )

    # G0.3 -- 16:9, exactly. EDM's single width-derived scale depends on it.
    bad = [(r["rel"], r["aspect"], r["aspect_err"]) for r in build if r["aspect_err"] != 0.0]
    g.check("G0.3", not bad,
            f"all {len(build)} build videos are exactly 16:9 (aspect_err == 0.0 in IEEE754)"
            if not bad else f"NOT 16:9 -- EDM would silently corrupt y-coords: {bad}",
            worst_aspect_err=max((abs(r["aspect_err"]) for r in build), default=0.0))

    # G0.4 -- Parrot provenance (the one file without it is the one we dropped)
    non_parrot = [r["rel"] for r in build if not r["parrot"]]
    g.check("G0.4", not non_parrot,
            f"all {len(build)} build videos carry Parrot metadata"
            if not non_parrot else f"non-Parrot in build set: {non_parrot}",
            dropped=list(DROPPED))

    # G0.5 -- positive proof that P071 is outside every enumerated source root
    # and absent from every source-lineage record written by this stage.
    declared_inputs = [v.path.resolve() for v in [*BUILD, *TEST]]
    declared_roots = sorted({p.parent for p in declared_inputs}, key=str)
    excluded = [d.resolve() for d in EXCLUDED_DIRS]
    excluded_under_input_roots = [
        str(d) for d in excluded
        if any(d == root or root in d.parents for root in declared_roots)
    ]
    declared_under_excluded = [
        str(p) for p in declared_inputs
        if any(p == d or d in p.parents for d in excluded)
    ]
    manifest_rels = {r.get("source_rel", r["rel"]) for r in build + test}
    p071_lineage = sorted(rel for rel in manifest_rels if "P0710071" in rel)
    excluded_ok = (
        all(d.is_dir() for d in excluded)
        and all(p.is_file() for p in declared_inputs)
        and not excluded_under_input_roots
        and not declared_under_excluded
        and not p071_lineage
    )
    g.check(
        "G0.5",
        excluded_ok,
        "P071 is outside every enumerated input root and absent from output lineage"
        if excluded_ok else "P071 exclusion proof failed",
        excluded=[str(d) for d in excluded],
        enumerated_input_roots=[str(p) for p in declared_roots],
        excluded_under_input_roots=excluded_under_input_roots,
        declared_inputs_under_excluded=declared_under_excluded,
        p071_lineage=p071_lineage,
    )

    # G0.6 -- sequence names must match gluemap's subfolder_regex, or its
    # multi-sequence loader discovers ZERO sequences and drops them silently
    import re
    rx = re.compile(SUBFOLDER_REGEX)
    unmatched = [v.seq for v in BUILD if not rx.match(v.seq)]
    g.check("G0.6", not unmatched,
            f"all 7 sequence names match subfolder_regex {SUBFOLDER_REGEX!r}"
            if not unmatched else f"would be SILENTLY DROPPED by gluemap: {unmatched}",
            regex=SUBFOLDER_REGEX)

    # G0.7 -- resolution groups -> one camera each
    groups = resolution_groups()
    g.check("G0.7", len(groups) == 3,
            f"{len(groups)} resolution groups -> {len(groups)} PINHOLE cameras: "
            + ", ".join(f"{w}x{h}({len(v)})" for (w, h), v in groups.items()),
            groups={f"{w}x{h}": [x.seq for x in v] for (w, h), v in groups.items()})

    # G0.8 -- both directions present, or forward<->reverse fusion is moot
    dirs = {v.direction for v in BUILD}
    g.check("G0.8", "fwd" in dirs and "rev" in dirs,
            f"directions present: {sorted(dirs)} (unknown ones resolved by S1)",
            directions={v.seq: v.direction for v in BUILD},
            unknown=[v.seq for v in BUILD if v.direction == "unknown"])

    manifest = {
        "run_name": args.run_name,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "build": build,
        "test": test,
        "dropped": DROPPED,
        "excluded_dirs": [str(d) for d in EXCLUDED_DIRS],
        "subfolder_regex": SUBFOLDER_REGEX,
        "resolution_groups": {
            f"{w}x{h}": {
                "videos": [x.seq for x in v],
                "candidates": {
                    c: cameras_for(c)[(w, h)].__dict__ for c in CANDIDATES
                },
            }
            for (w, h), v in groups.items()
        },
        "totals": {
            "build_videos": len(build),
            "build_frames": sum(r["probed"]["nb_frames"] for r in build),
            "build_seconds": round(sum(r["probed"]["duration"] for r in build), 2),
            "build_bytes": sum(r["bytes"] for r in build),
            "test_videos": len(test),
        },
    }
    write_json(run_dir / "corpus_manifest.json", manifest)
    g.write(run_dir)

    t = manifest["totals"]
    log(f"S0 PASS -- {t['build_videos']} build videos, {t['build_frames']} frames, "
        f"{t['build_seconds']:.0f}s, {t['build_bytes'] / 1e9:.2f} GB; "
        f"{t['test_videos']} held out")
    log(f"  manifest: {run_dir / 'corpus_manifest.json'}")


if __name__ == "__main__":
    main()
