import numpy as np

from update_map.change import AlignedImageChangeDetector, fuse_multiview_masks
from update_map.config import ChangeConfig
from update_map.models import MaskBundle
from update_map.states import MaskLabel


def test_change_detector_never_marks_threshold_band_stable() -> None:
    current = np.full((80, 100, 3), 120, dtype=np.uint8)
    historical = current.copy()
    historical[20:60, 30:70] = 220
    config = ChangeConfig(absdiff_threshold=0.1, structural_threshold=0.1, min_component_area=10)
    result = AlignedImageChangeDetector(config).detect(current, historical)
    assert np.any(result.labels == int(MaskLabel.CHANGED))
    assert result.ratio(int(MaskLabel.STABLE)) < 1.0


def test_multiview_fusion_requires_votes() -> None:
    stable = np.full((10, 10), int(MaskLabel.STABLE), dtype=np.uint8)
    changed = stable.copy()
    changed[2:5, 2:5] = int(MaskLabel.CHANGED)
    fused = fuse_multiview_masks(
        [MaskBundle(changed), MaskBundle(changed), MaskBundle(stable)], min_votes=2
    )
    assert np.all(fused.labels[2:5, 2:5] == int(MaskLabel.CHANGED))
