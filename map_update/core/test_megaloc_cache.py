from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from megaloc_cache import load_aligned_megaloc_cache, write_megaloc_cache
import megaloc_cache


def _shared_module_available() -> bool:
    try:
        megaloc_cache._load_shared_module()
    except ImportError:
        return False
    return True


requires_shared = pytest.mark.skipif(
    not _shared_module_available(),
    reason="sfm_system megaloc_cache_io is machine-local",
)



MEGALOC_DIM = 8448


def megaloc_row() -> np.ndarray:
    row = np.zeros((1, MEGALOC_DIM), dtype=np.float32)
    row[0, 0] = 1.0
    return row


@requires_shared
def test_load_aligned_megaloc_cache_reorders_descriptors_by_name(tmp_path: Path):
    cache = tmp_path / "base_megaloc.npz"
    np.savez(
        cache,
        names=np.array(["b.jpg", "a.jpg"]),
        desc=np.array([[2.0, 20.0], [1.0, 10.0]], dtype=np.float32),
    )

    desc = load_aligned_megaloc_cache(cache, ["a.jpg", "b.jpg"])

    assert desc.tolist() == [[1.0, 10.0], [2.0, 20.0]]


@requires_shared
def test_load_aligned_megaloc_cache_requires_names(tmp_path: Path):
    cache = tmp_path / "base_megaloc.npz"
    np.savez(cache, desc=np.ones((1, 2), dtype=np.float32))

    with pytest.raises(ValueError, match="missing names"):
        load_aligned_megaloc_cache(cache, ["a.jpg"])


@requires_shared
def test_load_aligned_megaloc_cache_rejects_missing_expected_ref(tmp_path: Path):
    cache = tmp_path / "base_megaloc.npz"
    np.savez(
        cache,
        names=np.array(["a.jpg"]),
        desc=np.ones((1, 2), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="cache missing 1 .*refs"):
        load_aligned_megaloc_cache(cache, ["a.jpg", "b.jpg"])


@requires_shared
def test_update_loader_supports_legacy_npy_with_name_sidecar(tmp_path: Path):
    cache = tmp_path / "base_megaloc.npy"
    rows = np.concatenate([megaloc_row(), np.roll(megaloc_row(), 1, axis=1)])
    np.save(cache, rows, allow_pickle=False)
    cache.with_suffix(".json").write_text(
        json.dumps({"ref_names": ["b.jpg", "a.jpg"], "input_size": 322}),
        encoding="utf-8",
    )

    desc = load_aligned_megaloc_cache(
        cache,
        ["a.jpg", "b.jpg"],
        expected_dim=MEGALOC_DIM,
        expected_input_size=322,
    )

    assert int(np.argmax(desc[0])) == 1
    assert int(np.argmax(desc[1])) == 0


@requires_shared
def test_update_loader_rejects_non_float32_cache(tmp_path: Path):
    cache = tmp_path / "base_megaloc.npz"
    np.savez(cache, desc=np.ones((1, 2), dtype=np.int32), names=np.array(["a.jpg"]))

    with pytest.raises(ValueError, match="stored as float32"):
        load_aligned_megaloc_cache(cache, ["a.jpg"])


@requires_shared
def test_update_writer_emits_canonical_npz(tmp_path: Path):
    cache = tmp_path / "base_megaloc.npz"
    rows = megaloc_row()

    write_megaloc_cache(cache, rows, ["a.jpg"], input_size=322)

    assert cache.read_bytes().startswith(b"PK")
    with np.load(cache, allow_pickle=False) as archive:
        assert archive["schema_name"].item() == "sfm_system.megaloc_cache"
        assert int(archive["schema_version"].item()) == 1
        np.testing.assert_array_equal(archive["desc"], rows)


def test_importing_megaloc_cache_does_not_require_sfm_system():
    assert megaloc_cache.load_aligned_megaloc_cache is not None
    assert megaloc_cache.write_megaloc_cache is not None


def test_load_without_shared_module_names_missing_path(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MEGALOC_CACHE_IO", raising=False)
    monkeypatch.setattr(megaloc_cache, "_system_root", lambda: (_ for _ in ()).throw(
        ImportError(
            "cannot find sfm_system root from /tmp; "
            "missing 定位/deploy_code/sfm_glomap_deploy/megaloc_cache_io.py; "
            "pass MEGALOC_CACHE_IO or override is required"
        )
    ))
    with pytest.raises(ImportError, match="MEGALOC_CACHE_IO|override is required") as excinfo:
        megaloc_cache.load_aligned_megaloc_cache("missing.npz", ["a.jpg"])
    assert "megaloc_cache_io.py" in str(excinfo.value)


def test_megaloc_cache_io_env_override_is_required_when_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    missing = tmp_path / "no_such_megaloc_cache_io.py"
    monkeypatch.setenv("MEGALOC_CACHE_IO", str(missing))
    with pytest.raises(ImportError, match=str(missing)) as excinfo:
        megaloc_cache.write_megaloc_cache()
    assert "override is required" in str(excinfo.value)


@requires_shared
def test_update_adapter_imports_shared_module_by_exact_path(tmp_path: Path):
    sentinel = tmp_path / "fake-import-executed"
    fake_dir = tmp_path / "fake"
    fake_dir.mkdir()
    (fake_dir / "megaloc_cache_io.py").write_text(
        "from pathlib import Path\n"
        "import os\n"
        "Path(os.environ['FAKE_SENTINEL']).write_text('executed')\n",
        encoding="utf-8",
    )
    update_dir = Path(__file__).resolve().parent
    shared_dir = (
        update_dir.parents[3] / "定位" / "deploy_code" / "sfm_glomap_deploy"
    )
    env = os.environ.copy()
    env["FAKE_SENTINEL"] = str(sentinel)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(update_dir), str(fake_dir), str(shared_dir)]
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import megaloc_cache; print(megaloc_cache._shared.SCHEMA_NAME)",
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "sfm_system.megaloc_cache"
    assert not sentinel.exists()
