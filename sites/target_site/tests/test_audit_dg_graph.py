from __future__ import annotations

import sys
from pathlib import Path


TARGET_SITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TARGET_SITE / "tools"))

from audit_dg_graph import (  # noqa: E402
    load_forced_pairs,
    independent_bridge_count,
    largest_component_fraction,
    robust_sequence_component_fraction,
)


def test_largest_component_fraction_counts_isolates() -> None:
    fraction, size = largest_component_fraction(5, [(0, 1), (1, 2), (3, 4)])

    assert fraction == 0.6
    assert size == 3


def test_independent_bridges_require_separation_on_both_sequences() -> None:
    pairs = [(0.05, 0.95), (0.10, 0.90), (0.85, 0.15), (0.90, 0.10)]

    assert independent_bridge_count(pairs, minimum_separation=0.25) == 2


def test_clustered_bridges_count_as_one() -> None:
    pairs = [(0.05, 0.95), (0.10, 0.90), (0.15, 0.85)]

    assert independent_bridge_count(pairs, minimum_separation=0.25) == 1


def test_forced_pair_parser_ignores_comments_and_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "pairs.txt"
    path.write_text("# generated candidates\n\nS01/a.jpg S02/b.jpg\n", encoding="utf-8")

    assert load_forced_pairs(path) == {("S01/a.jpg", "S02/b.jpg")}


def test_robust_sequence_graph_ignores_nonessential_single_hinges() -> None:
    directions = {"F1": "fwd", "F2": "fwd", "R1": "rev", "R2": "rev"}
    edges = {("F1", "F2"), ("F1", "R2"), ("F2", "R1"), ("R1", "R2")}
    bridge_counts = {("F1", "R2"): 1, ("F2", "R1"): 2}

    fraction, retained = robust_sequence_component_fraction(
        directions, edges, bridge_counts
    )

    assert fraction == 1.0
    assert ("F1", "R2") not in retained
    assert ("F2", "R1") in retained
