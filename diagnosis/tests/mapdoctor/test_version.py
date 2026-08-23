from importlib.metadata import PackageNotFoundError, version

import mapdoctor


def test_runtime_version_matches_package_metadata():
    assert mapdoctor.__version__
    for name in ("sfm-mapping", "sfm-map-diagnosis"):
        try:
            assert version(name) == "0.1.0"
            return
        except PackageNotFoundError:
            continue
    raise AssertionError("neither sfm-mapping nor sfm-map-diagnosis is installed")
