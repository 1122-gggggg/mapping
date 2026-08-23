from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_diagnosis_src_layout_is_in_repo() -> None:
    src = ROOT / "diagnosis" / "src"
    assert (src / "mapdoctor" / "cli.py").is_file()
    assert (src / "sfm_diagnosis" / "cli.py").is_file()
    assert (src / "sfm_qa" / "pipeline.py").is_file()
    assert (src / "sfm_qa" / "session_select" / "defaults.yaml").is_file()


def test_diagnosis_packages_import() -> None:
    import mapdoctor
    import sfm_diagnosis
    import sfm_qa

    assert mapdoctor.__file__
    assert sfm_diagnosis.__file__
    assert sfm_qa.__file__
    assert Path(mapdoctor.__file__).is_relative_to(ROOT / "diagnosis" / "src")
