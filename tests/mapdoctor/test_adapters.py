import shutil
from pathlib import Path

from mapdoctor import ColmapAdapter, GlomapAdapter, GluemapAdapter, load_colmap, load_glomap, load_gluemap
from mapdoctor.adapters import get_adapter, list_adapters
from mapdoctor.metrics import analyze

FIXTURE = Path(__file__).parent / "fixtures" / "colmap_text"


def test_three_explicit_adapters():
    expected = {"colmap": ColmapAdapter, "glomap": GlomapAdapter, "gluemap": GluemapAdapter}
    assert list_adapters() == ("colmap", "glomap", "gluemap")
    for backend, adapter_type in expected.items():
        adapter = get_adapter(backend)
        assert isinstance(adapter, adapter_type)
        model = adapter.load(FIXTURE)
        assert model.source == backend
        assert model.metadata["adapter"] == adapter_type.__name__


def test_convenience_loaders_preserve_provenance():
    assert load_colmap(FIXTURE).source == "colmap"
    assert load_glomap(FIXTURE).source == "glomap"
    assert load_gluemap(FIXTURE).source == "gluemap"


def test_gluemap_sparse_only_provenance():
    model = GluemapAdapter().load(FIXTURE)
    provenance = model.metadata["gluemap_provenance"]
    assert provenance["mode"] == "sparse-only"
    assert provenance["detected_artifacts"] == []
    metrics = analyze(model)
    assert metrics.producer_provenance["gluemap"]["mode"] == "sparse-only"


def test_gluemap_workspace_artifacts_are_detected_without_deserialization(tmp_path):
    workspace = tmp_path / "run"
    model_dir = workspace / "refined" / "0"
    shutil.copytree(FIXTURE, model_dir)
    for artifact in (
        "twoview_result.pth",
        "star_result.pth",
        "pipeline_timing.pth",
        "database_sift.db",
    ):
        (workspace / artifact).write_bytes(b"not deserialized by MapDoctor")
    (workspace / "coarse").mkdir()
    (workspace / "coarse_trial").mkdir()

    model = GluemapAdapter().load(model_dir)
    provenance = model.metadata["gluemap_provenance"]
    assert provenance["mode"] == "workspace-artifacts"
    assert Path(provenance["workspace"]) == workspace
    assert provenance["detected_artifacts"] == [
        "database_sift.db",
        "pipeline_timing.pth",
        "star_result.pth",
        "twoview_result.pth",
    ]
    assert provenance["coarse_reconstructions"] == ["coarse", "coarse_trial"]
    assert "twoview_inference" in provenance["detected_stages"]
    assert "star_inference" in provenance["detected_stages"]
    assert "global_mapping_coarse_output" in provenance["detected_stages"]
    assert "refinement_preparation" in provenance["detected_stages"]

    metrics = analyze(model)
    assert metrics.producer_provenance["gluemap"]["mode"] == "workspace-artifacts"
