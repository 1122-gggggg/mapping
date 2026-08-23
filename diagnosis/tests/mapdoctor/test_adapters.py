import shutil
import sys
from pathlib import Path
from types import ModuleType

from mapdoctor import ColmapAdapter, GlomapAdapter, GluemapAdapter, load_colmap, load_glomap, load_gluemap
from mapdoctor.adapters import AdapterInspection, MapAdapter, get_adapter, list_adapters
from mapdoctor.cli import main as mapdoctor_main
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


def test_external_adapter_can_run_full_diagnosis_without_core_changes(monkeypatch, tmp_path):
    class ExternalAdapter(MapAdapter):
        backend = "external"
        display_name = "External test map"

        def inspect(self, path):
            return AdapterInspection(self.backend, Path(path), "external")

        def load(self, path):
            model = ColmapAdapter().load(path)
            model.source = self.backend
            model.format = "external"
            return model

    module = ModuleType("mapdoctor_test_external_adapter")
    module.ExternalAdapter = ExternalAdapter
    monkeypatch.setitem(sys.modules, module.__name__, module)

    adapter = get_adapter(f"{module.__name__}:ExternalAdapter")

    assert isinstance(adapter, ExternalAdapter)
    assert adapter.load(FIXTURE).format == "external"
    output = tmp_path / "external-report"
    assert mapdoctor_main(
        [
            "analyze",
            str(FIXTURE),
            "--map-adapter",
            f"{module.__name__}:ExternalAdapter",
            "--output",
            str(output),
        ]
    ) == 0
    assert (output / "report.json").exists()


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
