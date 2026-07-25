from __future__ import annotations

import pytest

pytest.importorskip("numpy")

from sparsify_reloc_bundle import select_indices


def test_sparsify_top_up_uses_ref_stability():
    names = [f"P200/frame_{i:06d}.jpg" for i in range(10)]
    stability = {name: 1.0 for name in names}
    stability[names[-1]] = 5.0

    keep_idx, per_prefix = select_indices(
        names,
        scores={},
        target_fraction=0.5,
        min_per_prefix=1,
        keep_prefixes=set(),
        stability=stability,
        stability_weight=0.35,
    )

    assert len(keep_idx) == 5
    assert names[-1] in [names[i] for i in keep_idx]
    assert per_prefix["P200"] == {"input": 10, "kept": 5}
