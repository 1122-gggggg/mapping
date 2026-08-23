import numpy as np

from update_map.config import LiftingConfig
from update_map.lifting import ReferenceObservationIndex, aggregate_lifted_correspondences, lift_match_set
from update_map.models import BaseMap, Camera, Landmark, MapImage, MaskBundle, MatchSet, Observation, Pose
from update_map.states import GeometryProvenance, MaskLabel


def make_map() -> BaseMap:
    camera = Camera(1, "PINHOLE", 100, 100, np.array([80.0, 80.0, 50.0, 50.0]))
    image = MapImage(
        1,
        "ref.jpg",
        1,
        Pose.identity(),
        np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]]),
        np.array([1, 2, 3]),
    )
    points = {
        1: Landmark(1, np.array([0.0, 0.0, 5.0])),
        2: Landmark(2, np.array([1.0, 0.0, 5.0]), provenance=GeometryProvenance.VIRTUAL_BA_ONLY),
        3: Landmark(3, np.array([0.0, 1.0, 5.0])),
    }
    return BaseMap({1: camera}, {1: image}, points)


def test_changed_and_virtual_observations_never_lift() -> None:
    model = make_map()
    observations = model.observations_for_image(1)
    index = ReferenceObservationIndex(observations, model)
    matches = MatchSet(
        "q",
        "ref.jpg",
        np.array([[5.0, 5.0], [6.0, 6.0], [7.0, 7.0]]),
        np.array([[10.1, 10.0], [20.1, 20.0], [30.1, 30.0]]),
        np.ones(3),
    )
    labels = np.full((100, 100), int(MaskLabel.STABLE), dtype=np.uint8)
    labels[30, 30] = int(MaskLabel.CHANGED)
    lifted, diagnostics = lift_match_set(matches, index, LiftingConfig(), MaskBundle(labels))
    assert [item.point3d_id for item in lifted] == [1]
    assert diagnostics.forbidden_provenance == 1
    assert diagnostics.outside_stable_mask == 1


def test_aggregate_correspondences_counts_unique_point_ids() -> None:
    model = make_map()
    index = ReferenceObservationIndex(model.observations_for_image(1), model)
    config = LiftingConfig(require_stable_mask=False)
    first = MatchSet("q", "a", np.array([[1.0, 1.0]]), np.array([[10.0, 10.0]]), np.array([0.8]))
    second = MatchSet("q", "b", np.array([[1.2, 1.1]]), np.array([[10.0, 10.0]]), np.array([0.9]))
    group_a, _ = lift_match_set(first, index, config)
    group_b, _ = lift_match_set(second, index, config)
    aggregate, diagnostics = aggregate_lifted_correspondences([group_a, group_b])
    assert len(aggregate) == 1
    assert aggregate[0].reference_support == 2
    assert diagnostics.duplicate_point3d == 1
