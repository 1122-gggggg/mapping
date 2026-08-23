from __future__ import annotations

import sys
from types import ModuleType

from update_map.map_adapters import load_map
from update_map.models import BaseMap


def test_external_map_adapter_is_not_a_closed_format_enum(monkeypatch, tmp_path) -> None:
    def load_external(path):
        assert path == tmp_path
        return BaseMap({}, {}, {}, source_format="external")

    module = ModuleType("update_map_test_external_loader")
    module.load_external = load_external
    monkeypatch.setitem(sys.modules, module.__name__, module)

    model = load_map(tmp_path, f"{module.__name__}:load_external")

    assert model.source_format == "external"
