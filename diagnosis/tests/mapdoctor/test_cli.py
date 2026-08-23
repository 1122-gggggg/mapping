from pathlib import Path

from mapdoctor.cli import main

FIXTURE = Path(__file__).parent / "fixtures" / "colmap_text"


def test_unified_analyze_writes_reports(tmp_path):
    code = main(["analyze", str(FIXTURE), "--backend", "gluemap", "--output", str(tmp_path)])
    assert code == 0
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "report.html").exists()
    assert (tmp_path / "weak_images.csv").exists()


def test_three_backend_cli_interfaces(tmp_path):
    for backend in ("colmap", "glomap", "gluemap"):
        output = tmp_path / backend
        assert main([backend, str(FIXTURE), "--output", str(output)]) == 0
        assert (output / "report.json").exists()
