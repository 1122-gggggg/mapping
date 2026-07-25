from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from megaloc_cache import load_aligned_megaloc_cache, write_megaloc_cache


MEGALOC_DIM = 8448


def megaloc_row() -> np.ndarray:
    row = np.zeros((1, MEGALOC_DIM), dtype=np.float32)
    row[0, 0] = 1.0
    return row


def test_load_aligned_megaloc_cache_reorders_descriptors_by_name(tmp_path: Path):
    cache = tmp_path / "base_megaloc.npz"
    np.savez(
        cache,
        names=np.array(["b.jpg", "a.jpg"]),
        desc=np.array([[2.0, 20.0], [1.0, 10.0]], dtype=np.float32),
    )

    desc = load_aligned_megaloc_cache(cache, ["a.jpg", "b.jpg"])

    assert desc.tolist() == [[1.0, 10.0], [2.0, 20.0]]


def test_load_aligned_megaloc_cache_requires_names(tmp_path: Path):
    cache = tmp_path / "base_megaloc.npz"
    np.savez(cache, desc=np.ones((1, 2), dtype=np.float32))

    with pytest.raises(ValueError, match="missing names"):
        load_aligned_megaloc_cache(cache, ["a.jpg"])


def test_load_aligned_megaloc_cache_rejects_missing_expected_ref(tmp_path: Path):
    cache = tmp_path / "base_megaloc.npz"
    np.savez(
        cache,
        names=np.array(["a.jpg"]),
        desc=np.ones((1, 2), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="cache missing 1 .*refs"):
        load_aligned_megaloc_cache(cache, ["a.jpg", "b.jpg"])


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


def test_update_loader_rejects_non_float32_cache(tmp_path: Path):
    cache = tmp_path / "base_megaloc.npz"
    np.savez(cache, desc=np.ones((1, 2), dtype=np.int32), names=np.array(["a.jpg"]))

    with pytest.raises(ValueError, match="stored as float32"):
        load_aligned_megaloc_cache(cache, ["a.jpg"])


def test_update_writer_emits_canonical_npz(tmp_path: Path):
    cache = tmp_path / "base_megaloc.npz"
    rows = megaloc_row()

    write_megaloc_cache(cache, rows, ["a.jpg"], input_size=322)

    assert cache.read_bytes().startswith(b"PK")
    with np.load(cache, allow_pickle=False) as archive:
        assert archive["schema_name"].item() == "sfm_system.megaloc_cache"
        assert int(archive["schema_version"].item()) == 1
        np.testing.assert_array_equal(archive["desc"], rows)


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
