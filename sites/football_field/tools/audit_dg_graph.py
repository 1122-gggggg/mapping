#!/usr/bin/env python3
"""Recompute the target-site S4 Doppelgangers graph acceptance gate."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


def largest_component_fraction(
    node_count: int, edges: Iterable[tuple[int, int]]
) -> tuple[float, int]:
    if node_count <= 0:
        return 0.0, 0
    parent = list(range(node_count))
    size = [1] * node_count

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for left, right in edges:
        a, b = find(int(left)), find(int(right))
        if a == b:
            continue
        if size[a] < size[b]:
            a, b = b, a
        parent[b] = a
        size[a] += size[b]
    largest = max(size[find(node)] for node in range(node_count))
    return largest / node_count, largest


def independent_bridge_count(
    normalized_pairs: list[tuple[float, float]], *, minimum_separation: float
) -> int:
    """Return two only when bridge support separates on both traversals."""
    if not normalized_pairs:
        return 0
    for index, (left_a, right_a) in enumerate(normalized_pairs):
        for left_b, right_b in normalized_pairs[index + 1 :]:
            if (
                abs(left_a - left_b) >= minimum_separation
                and abs(right_a - right_b) >= minimum_separation
            ):
                return 2
    return 1


def robust_sequence_component_fraction(
    directions: dict[str, str],
    sequence_edges: set[tuple[str, str]],
    bridge_counts: dict[tuple[str, str], int],
) -> tuple[float, set[tuple[str, str]]]:
    """Drop weak cross-direction hinges and score the remaining backbone."""
    names = sorted(directions)
    index = {name: position for position, name in enumerate(names)}
    retained = set()
    for raw_left, raw_right in sequence_edges:
        left, right = sorted((raw_left, raw_right))
        if (
            directions[left] == directions[right]
            or bridge_counts.get((left, right), 0) >= 2
        ):
            retained.add((left, right))
    fraction, _ = largest_component_fraction(
        len(names), ((index[left], index[right]) for left, right in retained)
    )
    return fraction, retained


def load_forced_pairs(path: Path) -> set[tuple[str, str]]:
    pairs = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"invalid forced-pair line: {line!r}")
        pairs.add(tuple(sorted((fields[0], fields[1]))))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--twoview", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--forced-pairs", type=Path, required=True)
    parser.add_argument("--forced-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--minimum-separation", type=float, default=0.25)
    args = parser.parse_args()

    import torch

    names = sorted(
        path.relative_to(args.image_root).as_posix()
        for path in args.image_root.glob("*/*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    payload = torch.load(args.twoview, map_location="cpu", weights_only=False)
    pairs = np.asarray(payload["pairs"], dtype=np.int64)
    scores = np.asarray(payload["scores"], dtype=np.float64)
    if len(scores) != len(pairs):
        raise SystemExit("two-view score/pair length mismatch")
    valid_mask = scores > args.threshold
    valid_pairs = pairs[valid_mask]
    component_fraction, component_size = largest_component_fraction(
        len(names), ((int(a), int(b)) for a, b in valid_pairs)
    )
    rejection_rate = float(np.mean(~valid_mask))

    forced = load_forced_pairs(args.forced_pairs)
    manifest = json.loads(args.forced_manifest.read_text(encoding="utf-8"))
    forward = set(manifest["fwd"])
    reverse = set(manifest["rev"])
    directions = {
        **{sequence: "fwd" for sequence in forward},
        **{sequence: "rev" for sequence in reverse},
    }
    local_position: dict[str, float] = {}
    by_sequence: dict[str, list[str]] = defaultdict(list)
    for name in names:
        by_sequence[name.split("/", 1)[0]].append(name)
    for sequence_names in by_sequence.values():
        denominator = max(1, len(sequence_names) - 1)
        for index, name in enumerate(sequence_names):
            local_position[name] = index / denominator

    cross_valid: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    sequence_edges: set[tuple[str, str]] = set()
    for left_index, right_index in valid_pairs:
        left, right = names[int(left_index)], names[int(right_index)]
        left_sequence, right_sequence = left.split("/", 1)[0], right.split("/", 1)[0]
        if left_sequence == right_sequence:
            continue
        sequence_edges.add(tuple(sorted((left_sequence, right_sequence))))
        if left_sequence in reverse and right_sequence in forward:
            left, right = right, left
            left_sequence, right_sequence = right_sequence, left_sequence
        if left_sequence in forward and right_sequence in reverse:
            cross_valid[(left_sequence, right_sequence)].append((left, right))

    bridge_evidence = {}
    bridge_counts: dict[tuple[str, str], int] = {}
    for sequence_pair, accepted in sorted(cross_valid.items()):
        accepted_forced = [
            (left, right)
            for left, right in accepted
            if tuple(sorted((left, right))) in forced
        ]
        normalized = [
            (local_position[left], local_position[right])
            for left, right in accepted_forced
        ]
        count = independent_bridge_count(
            normalized, minimum_separation=args.minimum_separation
        )
        bridge_counts[tuple(sorted(sequence_pair))] = count
        bridge_evidence["|".join(sequence_pair)] = {
            "accepted_cross_edges": len(accepted),
            "accepted_forced_edges": len(accepted_forced),
            "independent_bridge_count": count,
        }

    robust_fraction, retained_sequence_edges = robust_sequence_component_fraction(
        directions, sequence_edges, bridge_counts
    )
    robust_cross_edges = [
        edge
        for edge in retained_sequence_edges
        if directions[edge[0]] != directions[edge[1]]
    ]

    checks = {
        "G4.1": bool(
            len(scores)
            and float(np.min(scores)) < args.threshold < float(np.max(scores))
            and float(np.std(scores)) > 1e-6
        ),
        "G4.2": 0.02 <= rejection_rate <= 0.40,
        "G4.3": robust_fraction == 1.0 and bool(robust_cross_edges),
        "G4.4": component_fraction >= 0.90,
    }
    result = {
        "stage": "S4_doppelgangers",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "threshold": args.threshold,
        "candidate_pairs": len(scores),
        "accepted_pairs": int(np.sum(valid_mask)),
        "rejected_pairs": int(np.sum(~valid_mask)),
        "rejection_rate": rejection_rate,
        "largest_component_images": component_size,
        "largest_component_fraction": component_fraction,
        "robust_sequence_component_fraction": robust_fraction,
        "retained_sequence_edges": [list(edge) for edge in sorted(retained_sequence_edges)],
        "robust_cross_direction_edges": [list(edge) for edge in sorted(robust_cross_edges)],
        "bridge_evidence": bridge_evidence,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    if result["status"] != "PASS":
        raise SystemExit("S4 gate failed")


if __name__ == "__main__":
    main()
