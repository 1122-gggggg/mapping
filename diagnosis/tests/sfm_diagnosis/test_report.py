from test_diagnose import healthy_map

from sfm_diagnosis.report import map_health_summary


def test_map_report_contains_covisibility_audit():
    report = map_health_summary(healthy_map(), covisibility_min_shared=10)
    assert report["num_images"] == 5
    assert report["num_points3D"] > 100
    assert report["covisibility"]["connected_components"] == 1
    assert report["covisibility"]["isolated_images"] == 0
