from __future__ import annotations

import importlib
import importlib.util
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import SimpleNamespace

import pytest


MODULE_PATH = Path(__file__).with_name("build_localizable_map_core.py")
SPEC = importlib.util.spec_from_file_location("build_localizable_map_core_attestation", MODULE_PATH)
assert SPEC and SPEC.loader
core = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(core)


def _sha256(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def _expected_ufm_assets(snapshot: Path, dinov2_weights: Path) -> dict[str, dict]:
    return {
        "ufm_config": core.mvroma_file_content_identity(snapshot / "config.json"),
        "ufm_weights": core.mvroma_file_content_identity(
            snapshot / "model.safetensors"
        ),
        "dinov2_weights": core.mvroma_file_content_identity(dinov2_weights),
    }


def test_file_content_identity_detects_restored_mtime_mutation_and_is_path_free(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model.bin"
    path.write_bytes(b"original")
    frozen_mtime = path.stat().st_mtime_ns

    first = core.mvroma_file_content_identity(path)
    path.write_bytes(b"mutated!")
    os.utime(path, ns=(frozen_mtime, frozen_mtime))
    second = core.mvroma_file_content_identity(path)

    assert set(first) == {"size", "sha256"}
    assert first["size"] == second["size"]
    assert first["sha256"] != second["sha256"]
    assert str(tmp_path) not in json.dumps(first)

    alias = tmp_path / "snapshot-link"
    alias.symlink_to(path)
    assert core.mvroma_file_content_identity(alias) == second


def test_loaded_orchestrator_capture_rejects_path_replacement(
    tmp_path: Path,
) -> None:
    source = tmp_path / "orchestrator.py"
    replacement = tmp_path / "replacement.py"
    source.write_bytes(b"print('held')\n")
    replacement.write_bytes(b"print('new!')\n")
    capture = core._capture_mvroma_loaded_source(source)
    try:
        assert core.attest_mvroma_loaded_source(capture) == {
            "size": len(b"print('held')\n"),
            "sha256": _sha256(b"print('held')\n"),
        }
        source.rename(tmp_path / "original.py")
        replacement.rename(source)
        with pytest.raises(RuntimeError, match="loaded .*source (inode|path) changed"):
            core.attest_mvroma_loaded_source(capture)
    finally:
        os.close(capture.fd)


def test_file_content_identity_opens_with_nonblocking_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "model.bin"
    path.write_bytes(b"model")
    original_open = core.os.open
    observed_flags: list[int] = []

    def checked_open(open_path: object, flags: int, *args: object) -> int:
        observed_flags.append(flags)
        return original_open(open_path, flags, *args)

    monkeypatch.setattr(core.os, "open", checked_open)

    assert core.mvroma_file_content_identity(path)["sha256"] == _sha256(b"model")
    assert len(observed_flags) == 1
    assert observed_flags[0] & os.O_NONBLOCK


def test_tree_identity_is_relocation_invariant_and_binds_names_and_contents(
    tmp_path: Path,
) -> None:
    roots = [tmp_path / "one", tmp_path / "two"]
    for root in roots:
        (root / "pkg").mkdir(parents=True)
        (root / "pkg" / "a.py").write_text("VALUE = 1\n")
        (root / "pkg" / "ignored.pyc").write_bytes(b"cache")
        (root / "__pycache__").mkdir()
        (root / "__pycache__" / "a.py").write_text("ignored = True\n")

    first = core.mvroma_tree_content_identity(roots[0], python_only=True)
    second = core.mvroma_tree_content_identity(roots[1], python_only=True)
    assert first == second
    assert first["file_count"] == 1
    assert str(tmp_path) not in json.dumps(first)
    leaf_sha256 = _sha256(b"VALUE = 1\n")
    expected_sha256sum = _sha256(f"{leaf_sha256}  ./pkg/a.py\n".encode())
    assert first["sha256sum_sha256"] == expected_sha256sum

    (roots[1] / "pkg" / "a.py").write_text("VALUE = 2\n")
    assert core.mvroma_tree_content_identity(roots[1], python_only=True) != first
    (roots[1] / "pkg" / "a.py").write_text("VALUE = 1\n")
    (roots[1] / "pkg" / "extra.py").write_text("EXTRA = 1\n")
    assert core.mvroma_tree_content_identity(roots[1], python_only=True) != first


def test_frozen_source_tree_uses_exact_o101_domain_and_selected_paths(
    tmp_path: Path,
) -> None:
    root = tmp_path / "source"
    root.mkdir()
    (root / "a.py").write_text("A = 1\n")
    (root / "b.py").write_text("B = 2\n")
    (root / "untracked.py").write_text("UNTRACKED = True\n")

    identity = core.build_mvroma_frozen_source_tree(root, ["b.py", "a.py"])
    import hashlib

    expected = hashlib.sha256()
    expected.update(b"o101-source-tree-v1\0")
    for relative in ("a.py", "b.py"):
        leaf = _sha256((root / relative).read_bytes())
        expected.update(relative.encode())
        expected.update(b"\0")
        expected.update(leaf.encode("ascii"))
        expected.update(b"\n")

    assert identity["schema"] == "o101-source-tree/v1"
    assert identity["file_count"] == 2
    assert identity["tree_sha256"] == expected.hexdigest()
    assert [row[0] for row in identity["files"]] == ["a.py", "b.py"]
    assert "untracked.py" not in json.dumps(identity)
    assert str(tmp_path) not in json.dumps(identity)


@pytest.mark.parametrize(
    "paths",
    [
        ["a.py", "a.py"],
        ["../escape.py"],
        ["/absolute.py"],
        ["missing.py"],
    ],
)
def test_frozen_source_tree_rejects_duplicate_unsafe_or_missing_paths(
    tmp_path: Path, paths: list[str]
) -> None:
    root = tmp_path / "source"
    root.mkdir()
    (root / "a.py").write_text("A = 1\n")

    with pytest.raises((ValueError, FileNotFoundError)):
        core.build_mvroma_frozen_source_tree(root, paths)


def test_private_source_tree_copies_verified_selection_read_only(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    selected = source / "pkg" / "selected.py"
    selected.parent.mkdir()
    selected.write_text("VALUE = 'frozen'\n")
    (source / "untracked.py").write_text("VALUE = 'ignored'\n")
    identity = core.build_mvroma_frozen_source_tree(source, ["pkg/selected.py"])

    with core.private_attested_mvroma_source_tree(source, identity) as private:
        private = Path(private)
        assert private.resolve() != source.resolve()
        assert private.stat().st_mode & 0o777 == 0o700
        copied = private / "pkg" / "selected.py"
        assert copied.read_text() == "VALUE = 'frozen'\n"
        assert copied.stat().st_mode & 0o777 == 0o400
        assert not (private / "untracked.py").exists()
        selected.write_text("VALUE = 'changed'\n")
        assert copied.read_text() == "VALUE = 'frozen'\n"


def test_private_source_tree_rejects_symlink_selected_file(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    real = source / "real.py"
    real.write_text("VALUE = 1\n")
    (source / "alias.py").symlink_to(real)

    with pytest.raises(ValueError, match="symlink"):
        core.build_mvroma_frozen_source_tree(source, ["alias.py"])


def test_python_source_root_attestation_uses_git_tracked_mvroma_and_all_ufm_python(
    tmp_path: Path,
) -> None:
    mvroma = tmp_path / "mvroma"
    ufm = tmp_path / "ufm"
    mvroma.mkdir()
    ufm.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=mvroma, check=True)
    (mvroma / "tracked.py").write_text("TRACKED = 1\n")
    (mvroma / "untracked.py").write_text("UNTRACKED = 1\n")
    subprocess.run(["git", "add", "tracked.py"], cwd=mvroma, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=O101",
            "-c",
            "user.email=o101@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=mvroma,
        check=True,
    )
    (ufm / "root.py").write_text("ROOT = 1\n")
    (ufm / "pkg").mkdir()
    (ufm / "pkg" / "child.py").write_text("CHILD = 1\n")
    (ufm / "ignored.txt").write_text("ignored\n")

    result = core.attest_mvroma_python_source_roots(mvroma, ufm)

    assert [row[0] for row in result["identity"]["mvroma"]["files"]] == [
        "tracked.py"
    ]
    assert [row[0] for row in result["identity"]["ufm"]["files"]] == [
        "pkg/child.py",
        "root.py",
    ]
    assert result["provenance"]["mvroma_git_head"] == subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=mvroma,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert str(tmp_path) not in json.dumps(result["identity"])

    with pytest.raises(RuntimeError, match="frozen.*MV-RoMa|MV-RoMa.*frozen"):
        core.attest_mvroma_python_source_roots(
            mvroma,
            ufm,
            expected={
                "mvroma_git_head": "0" * 40,
                "mvroma_file_count": 1,
                "mvroma_tree_sha256": result["identity"]["mvroma"]["tree_sha256"],
                "ufm_file_count": 2,
                "ufm_tree_sha256": result["identity"]["ufm"]["tree_sha256"],
            },
        )


def test_private_import_environment_uses_source_loader_and_restores_process_state(
    tmp_path: Path,
) -> None:
    import importlib

    mvroma = tmp_path / "private-mvroma"
    ufm = tmp_path / "private-ufm"
    (mvroma / "src").mkdir(parents=True)
    (mvroma / "src" / "probe.py").write_text("VALUE = 'private'\n")
    (ufm / "uniflowmatch").mkdir(parents=True)
    (ufm / "uniflowmatch" / "__init__.py").write_text("VALUE = 'ufm'\n")
    (ufm / "UniCeption" / "uniception").mkdir(parents=True)
    (ufm / "UniCeption" / "uniception" / "__init__.py").write_text(
        "VALUE = 'uniception'\n"
    )
    before_path = list(sys.path)
    before_cwd = Path.cwd()
    before_dont_write = sys.dont_write_bytecode
    before_pycache = sys.pycache_prefix

    with core.private_mvroma_import_environment(mvroma, ufm):
        module = importlib.import_module("src.probe")
        assert module.VALUE == "private"
        assert Path(module.__file__).suffix == ".py"
        assert isinstance(module.__loader__, SourceFileLoader)
        assert Path(module.__file__).resolve().is_relative_to(mvroma.resolve())
        assert sys.dont_write_bytecode is True
        assert sys.pycache_prefix is not None
        assert Path.cwd() == mvroma.resolve()

    assert "src.probe" not in sys.modules
    assert list(sys.path) == before_path
    assert Path.cwd() == before_cwd
    assert sys.dont_write_bytecode is before_dont_write
    assert sys.pycache_prefix == before_pycache


def test_private_import_environment_rejects_preloaded_target_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mvroma = tmp_path / "mvroma"
    ufm = tmp_path / "ufm"
    mvroma.mkdir()
    ufm.mkdir()
    monkeypatch.setitem(
        sys.modules,
        "uniflowmatch.shadow",
        SimpleNamespace(__file__="/tmp/shadow.py"),
    )

    with pytest.raises(RuntimeError, match="preloaded.*uniflowmatch"):
        with core.private_mvroma_import_environment(mvroma, ufm):
            pass


def _image_jobs() -> list[dict]:
    return [
        {
            "source_index": 0,
            "source": "seq/a.jpg",
            "targets": ["seq/b.jpg", "seq/b.jpg"],
            "chunks": [["seq/b.jpg"]],
            "shard_name": "000000-deadbeef.h5",
        }
    ]


def test_image_tree_hashes_references_once_and_detects_same_metadata_mutation(
    tmp_path: Path,
) -> None:
    image_root = tmp_path / "images"
    (image_root / "seq").mkdir(parents=True)
    source = image_root / "seq" / "a.jpg"
    target = image_root / "seq" / "b.jpg"
    source.write_bytes(b"source00")
    target.write_bytes(b"target00")
    frozen_mtime = target.stat().st_mtime_ns

    first = core.build_mvroma_image_sha256_tree(image_root, _image_jobs())
    assert first["file_count"] == 2
    assert sorted(first["by_name"]) == ["seq/a.jpg", "seq/b.jpg"]
    assert str(tmp_path) not in json.dumps(first)

    target.write_bytes(b"changed0")
    os.utime(target, ns=(frozen_mtime, frozen_mtime))
    second = core.build_mvroma_image_sha256_tree(image_root, _image_jobs())
    assert second["by_name"]["seq/a.jpg"] == first["by_name"]["seq/a.jpg"]
    assert second["by_name"]["seq/b.jpg"] != first["by_name"]["seq/b.jpg"]

    (image_root / "unreferenced.jpg").write_bytes(b"ignored")
    assert core.build_mvroma_image_sha256_tree(image_root, _image_jobs()) == second


@pytest.mark.parametrize("bad_name", ["../escape.jpg", "/absolute.jpg", "seq/missing.jpg"])
def test_image_tree_fails_closed_on_escape_absolute_or_missing(
    tmp_path: Path, bad_name: str
) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    outside = tmp_path / "escape.jpg"
    outside.write_bytes(b"outside")
    job = {
        "source_index": 0,
        "source": bad_name,
        "targets": [],
        "chunks": [],
        "shard_name": "000000-deadbeef.h5",
    }

    with pytest.raises((ValueError, FileNotFoundError)):
        core.build_mvroma_image_sha256_tree(image_root, [job])


def test_image_tree_rejects_symlink_even_when_target_is_inside_root(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    real = image_root / "real.jpg"
    real.write_bytes(b"image")
    (image_root / "alias.jpg").symlink_to(real)
    job = {
        "source_index": 0,
        "source": "alias.jpg",
        "targets": [],
        "chunks": [],
        "shard_name": "000000-deadbeef.h5",
    }

    with pytest.raises(ValueError, match="symlink"):
        core.build_mvroma_image_sha256_tree(image_root, [job])


@pytest.mark.parametrize("field", ["by_name", "file_count", "tree_sha256"])
def test_image_tree_validator_rejects_internal_drift_before_model_load(
    tmp_path: Path, field: str
) -> None:
    jobs, image_root, image_tree, asset_paths, assets, _values = (
        _attested_execution_fixture(tmp_path)
    )
    changed = json.loads(json.dumps(image_tree))
    if field == "by_name":
        changed["by_name"]["source.jpg"] = "0" * 64
    elif field == "file_count":
        changed["file_count"] += 1
    else:
        changed["tree_sha256"] = "0" * 64
    load_calls = 0

    def load_runner() -> object:
        nonlocal load_calls
        load_calls += 1
        return lambda job: {}

    with pytest.raises(ValueError, match="image tree"):
        core.execute_attested_mvroma_resume(
            jobs,
            tmp_path / "shards",
            tmp_path / "final.h5",
            "d" * 64,
            image_root,
            changed,
            asset_paths,
            assets,
            max_correspondences=4000,
            mvroma_resume=True,
            overwrite=False,
            runner_loader=load_runner,
        )

    assert load_calls == 0
    assert not (tmp_path / "shards").exists()


def test_image_tree_validator_returns_detached_complete_snapshot(tmp_path: Path) -> None:
    jobs, _image_root, image_tree, _asset_paths, _assets, _values = (
        _attested_execution_fixture(tmp_path)
    )
    snapshot = core.snapshot_mvroma_image_sha256_tree(image_tree, jobs)
    image_tree["by_name"]["source.jpg"] = "0" * 64
    image_tree["files"][0][2] = "0" * 64

    assert snapshot["by_name"]["source.jpg"] != "0" * 64
    assert snapshot["files"][0][2] != "0" * 64


def _post_model_expectation_ref() -> dict[str, str]:
    return {
        "schema": "o101-post-model-contract-ref/v1",
        "contract_schema": "o101-post-model-contract/v3",
        "sha256": "b" * 64,
        "base_sha256": "e" * 64,
    }


def _candidate_inference_contract() -> dict[str, object]:
    return {
        "coarse_res_hw": [560, 560],
        "target_res_hw": [560, 840],
        "chunk_size": 32,
        "sample_mode": "score_grid",
        "sample_grid": [8, 12],
        "certainty_threshold": 0.35,
        "max_correspondences": 4000,
        "upsample_preds": True,
        "num_cluster": 512,
        "prematcher": "ufm",
    }


def _candidate_runtime_contract(device: str) -> dict[str, object]:
    return {
        "phase": "post_import_pre_model",
        "gpu": {"selected_device": device},
    }


def test_stage_contract_is_canonical_path_free_and_rejects_nonfinite() -> None:
    implementation = {"orchestrator": {"size": 1, "sha256": "a" * 64}}
    models = {"mvroma": {"size": 2, "sha256": "b" * 64}}
    inference = {"coarse_res_hw": [560, 560], "threshold": 0.35}
    runtime = {"phase": "post_import_pre_model", "torch": "2.7.1+cu128"}

    first = core.build_mvroma_stage_contract(
        implementation=implementation,
        models=models,
        inference=inference,
        runtime=runtime,
        post_model_expectation_ref=_post_model_expectation_ref(),
    )
    second = core.build_mvroma_stage_contract(
        implementation=json.loads(json.dumps(implementation)),
        models=json.loads(json.dumps(models)),
        inference=json.loads(json.dumps(inference)),
        runtime=json.loads(json.dumps(runtime)),
        post_model_expectation_ref=_post_model_expectation_ref(),
    )
    assert first == second
    assert set(first) == {"payload", "sha256"}
    assert first["payload"]["schema"] == "mvroma-stage-contract/v2"
    assert first["payload"]["post_model_expectation_ref"] == (
        _post_model_expectation_ref()
    )
    assert "/tmp" not in json.dumps(first)

    implementation["orchestrator"]["sha256"] = "c" * 64
    models["mvroma"]["sha256"] = "d" * 64
    inference["threshold"] = 0.99
    runtime["torch"] = "mutated"
    assert first["payload"]["implementation"]["orchestrator"]["sha256"] == "a" * 64
    assert first["payload"]["models"]["mvroma"]["sha256"] == "b" * 64
    assert first["payload"]["inference"]["threshold"] == 0.35
    assert first["payload"]["runtime"]["torch"] == "2.7.1+cu128"
    assert first["sha256"] == _sha256(
        json.dumps(
            first["payload"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    )

    implementation = {"orchestrator": {"size": 1, "sha256": "a" * 64}}
    models = {"mvroma": {"size": 2, "sha256": "b" * 64}}
    inference = {"coarse_res_hw": [560, 560], "threshold": 0.35}
    runtime = {"phase": "post_import_pre_model", "torch": "2.7.1+cu128"}

    changed = core.build_mvroma_stage_contract(
        implementation=implementation,
        models=models,
        inference={**inference, "threshold": 0.36},
        runtime=runtime,
        post_model_expectation_ref=_post_model_expectation_ref(),
    )
    assert changed["sha256"] != first["sha256"]
    with pytest.raises(ValueError, match="finite"):
        core.build_mvroma_stage_contract(
            implementation=implementation,
            models=models,
            inference={"threshold": float("nan")},
            runtime=runtime,
            post_model_expectation_ref=_post_model_expectation_ref(),
        )
    with pytest.raises(ValueError, match="provenance"):
        core.build_mvroma_stage_contract(
            implementation=implementation,
            models=models,
            inference=inference,
            runtime={
                "phase": "post_import_pre_model",
                "provenance": {"TORCH_HOME": "/tmp/cache"},
            },
            post_model_expectation_ref=_post_model_expectation_ref(),
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.pop("sha256"),
        lambda value: value.update({"extra": "forbidden"}),
        lambda value: value.update({"schema": "wrong"}),
        lambda value: value.update({"contract_schema": "wrong"}),
        lambda value: value.update({"sha256": "B" * 64}),
        lambda value: value.update({"base_sha256": "short"}),
    ],
)
def test_stage_contract_rejects_invalid_post_model_expectation_ref(
    mutate: object,
) -> None:
    reference = _post_model_expectation_ref()
    mutate(reference)
    with pytest.raises(ValueError, match="post-model expectation reference"):
        core.build_mvroma_stage_contract(
            implementation={},
            models={},
            inference={},
            runtime={"phase": "post_import_pre_model"},
            post_model_expectation_ref=reference,
        )


def test_stage_contract_binds_post_model_expectation_ref_transitively() -> None:
    reference = _post_model_expectation_ref()
    first = core.build_mvroma_stage_contract(
        implementation={},
        models={},
        inference={},
        runtime={"phase": "post_import_pre_model"},
        post_model_expectation_ref=reference,
    )
    reference["sha256"] = "c" * 64
    assert first["payload"]["post_model_expectation_ref"]["sha256"] == "b" * 64

    changed = core.build_mvroma_stage_contract(
        implementation={},
        models={},
        inference={},
        runtime={"phase": "post_import_pre_model"},
        post_model_expectation_ref=reference,
    )
    assert changed["sha256"] != first["sha256"]


def test_snapshot_stage_contract_rehashes_and_detaches_prepared_payload() -> None:
    contract = core.build_mvroma_stage_contract(
        implementation={},
        models={},
        inference={},
        runtime={"phase": "post_import_pre_model"},
        post_model_expectation_ref=_post_model_expectation_ref(),
    )
    snapshot = core.snapshot_mvroma_stage_contract(contract)
    contract["payload"]["post_model_expectation_ref"]["sha256"] = "c" * 64
    assert snapshot["payload"]["post_model_expectation_ref"]["sha256"] == "b" * 64

    with pytest.raises(RuntimeError, match="stage contract hash mismatch"):
        core.snapshot_mvroma_stage_contract(contract)

    invalid_schema = json.loads(json.dumps(snapshot))
    invalid_schema["payload"]["schema"] = "mvroma-stage-contract/v1"
    invalid_schema["sha256"] = _sha256(
        json.dumps(
            invalid_schema["payload"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    )
    with pytest.raises(ValueError, match="stage contract payload"):
        core.snapshot_mvroma_stage_contract(invalid_schema)


def test_candidate_cfg_snapshot_rejects_post_prepare_semantic_drift() -> None:
    cfg = SimpleNamespace(
        device="cuda:0",
        mvroma_grid_h=560,
        mvroma_grid_w=840,
        mvroma_chunk=32,
        mvroma_sample_mode="score_grid",
        mvroma_sample_grid="8x12",
        roma_cert_thresh=0.35,
        agg_maxkp=4000,
    )
    stage_payload = {
        "runtime": {"gpu": {"selected_device": "cuda:0"}},
        "inference": {
            "coarse_res_hw": [560, 560],
            "target_res_hw": [560, 840],
            "chunk_size": 32,
            "sample_mode": "score_grid",
            "sample_grid": [8, 12],
            "certainty_threshold": 0.35,
            "max_correspondences": 4000,
            "upsample_preds": True,
            "num_cluster": 512,
            "prematcher": "ufm",
        },
    }

    snapshot = core.snapshot_mvroma_candidate_cfg(cfg, stage_payload)
    assert snapshot.device == "cuda:0"
    assert snapshot.mvroma_grid_h == 560
    assert snapshot.mvroma_grid_w == 840
    assert snapshot.mvroma_sample_grid == "8x12"

    cfg.roma_cert_thresh = 0.36
    with pytest.raises(RuntimeError, match="candidate cfg drift"):
        core.snapshot_mvroma_candidate_cfg(cfg, stage_payload)


def _post_model_identity_fixture() -> dict:
    object_id = {
        "type": {"module": "pkg.model", "qualname": "Model"},
        "mro": [
            {"module": "pkg.model", "qualname": "Model"},
            {"module": "builtins", "qualname": "object"},
        ],
    }
    load_id = {
        "schema": "mvroma-state-load/v1",
        "strict": False,
        "missing_keys": [],
        "unexpected_keys": ["legacy.key"],
    }
    runtime = {
        "phase": "post_model_pre_publish",
        "versions": {"torch": "2.test"},
        "torch_flags": {"flash_sdp_enabled": True},
        "gpu": {"name": "GPU-A", "uuid": "GPU-1", "driver": "1"},
        "environment": {"CUDA_VISIBLE_DEVICES": "0"},
        "attention_backend": {
            "uniception_has_fused_attn": True,
            "uniception_use_fused_attn_raw": 1,
            "uniception_use_fused_attn": True,
            "mvroma_attention_xformers_available": False,
            "mvroma_block_xformers_available": False,
            "dino_attention_xformers_enabled": True,
            "dino_attention_xformers_available": False,
            "dino_block_xformers_enabled": True,
            "dino_block_xformers_available": False,
            "dino_swiglu_xformers_enabled": True,
            "dino_swiglu_xformers_available": False,
        },
    }
    result = {
        "schema": "o101-post-model-identity/v1",
        "mvroma": object_id,
        "mvroma_state_load": load_id,
        "ufm_runner_class": {
            "module": "pkg.ufm",
            "qualname": "VendoredUFM",
        },
        "ufm_runner_is_vendored_class": True,
        "ufm": object_id,
        "ufm_state_load": {
            **load_id,
            "schema": "ufm-safetensors-state-load/v1",
            "unexpected_keys": [],
        },
        "encoder": object_id,
        "dinov2_model": object_id,
        "fused_attention": [
            {"name": "info.0.attn", "identity": object_id, "fused_attn": True}
        ],
        "dinov2_blocks": [
            {"index": 0, "block": object_id, "attention": object_id}
        ],
        "module_identity": {
            "schema": "mvroma-module-origins/v1",
            "modules": [
                {
                    "module": "pkg.model",
                    "role": "mvroma_vendored",
                    "relative_path": "pkg/model.py",
                    "loader": "SourceFileLoader",
                    "size": 10,
                    "sha256": "a" * 64,
                }
            ],
            "sha256": "",
        },
        "model_state_identity": {
            "schema": "o101-model-state-identity/v2",
            "nonpersistent_content_encoding": "numpy-c-order-bytes/v1",
            "maximum_nonpersistent_content_bytes_per_model": 1048576,
            "models": {
                role: {
                    "module_count": 1,
                    "training_modules": [],
                    "parameters": [
                        {
                            "name": "weight",
                            "shape": [1],
                            "dtype": "torch.float32",
                            "requires_grad": True,
                            "alias_group": 0,
                        }
                    ],
                    "buffers": [],
                    "nonpersistent_content_bytes": 0,
                }
                for role in ("mvroma", "ufm")
            },
            "sha256": "",
        },
        "runtime_identity": runtime,
        "post_model_assets": {
            "schema": "mvroma-runtime-assets/v1",
            "files": {"model": {"size": 1, "sha256": "c" * 64}},
            "dinov2_source": {"tree_sha256": "d" * 64},
        },
    }
    import hashlib

    module_payload = {
        "schema": result["module_identity"]["schema"],
        "modules": result["module_identity"]["modules"],
    }
    result["module_identity"]["sha256"] = hashlib.sha256(
        b"mvroma-module-origins-v1\0"
        + json.dumps(
            module_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()
    state_payload = {
        "schema": result["model_state_identity"]["schema"],
        "nonpersistent_content_encoding": result["model_state_identity"][
            "nonpersistent_content_encoding"
        ],
        "maximum_nonpersistent_content_bytes_per_model": result[
            "model_state_identity"
        ]["maximum_nonpersistent_content_bytes_per_model"],
        "models": result["model_state_identity"]["models"],
    }
    result["model_state_identity"]["sha256"] = hashlib.sha256(
        b"o101-model-state-identity-v2\0"
        + json.dumps(
            state_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()
    return result


def _refresh_post_model_module_hash(value: dict) -> None:
    import hashlib

    module_payload = {
        "schema": value["module_identity"]["schema"],
        "modules": value["module_identity"]["modules"],
    }
    value["module_identity"]["sha256"] = hashlib.sha256(
        b"mvroma-module-origins-v1\0"
        + json.dumps(
            module_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()


def test_post_model_contract_freezes_structure_but_not_dynamic_runtime() -> None:
    first = _post_model_identity_fixture()
    frozen = core.build_mvroma_post_model_contract(first)

    assert frozen["payload"]["schema"] == "o101-post-model-contract/v3"
    assert set(frozen) == {"payload", "sha256", "base_sha256"}
    assert "/tmp" not in json.dumps(frozen)

    dynamic = json.loads(json.dumps(first))
    dynamic["runtime_identity"]["versions"]["torch"] = "3.test"
    dynamic["runtime_identity"]["torch_flags"]["flash_sdp_enabled"] = False
    dynamic["runtime_identity"]["gpu"] = {
        "name": "GPU-B",
        "uuid": "GPU-2",
        "driver": "2",
    }
    dynamic["runtime_identity"]["environment"]["CUDA_VISIBLE_DEVICES"] = "1"
    dynamic["post_model_assets"]["files"]["model"]["sha256"] = "e" * 64
    assert core.build_mvroma_post_model_contract(dynamic) == frozen

    for field, mutate in (
        (
            "mvroma_state_load",
            lambda value: value.update(unexpected_keys=["drift", "legacy.key"]),
        ),
        ("fused_attention", lambda value: value[0].update(fused_attn=False)),
        (
            "runtime_identity",
            lambda value: value["attention_backend"].update(
                uniception_use_fused_attn=False
            ),
        ),
    ):
        changed = json.loads(json.dumps(first))
        mutate(changed[field])
        assert core.build_mvroma_post_model_contract(changed)["sha256"] != frozen[
            "sha256"
        ]
    changed_module = json.loads(json.dumps(first))
    changed_module["module_identity"]["modules"][0]["sha256"] = "f" * 64
    _refresh_post_model_module_hash(changed_module)
    assert core.build_mvroma_post_model_contract(changed_module)["sha256"] != frozen[
        "sha256"
    ]
    changed_state = json.loads(json.dumps(first))
    changed_state["model_state_identity"]["models"]["mvroma"]["parameters"][0][
        "dtype"
    ] = "torch.float64"
    state_payload = {
        "schema": changed_state["model_state_identity"]["schema"],
        "nonpersistent_content_encoding": changed_state["model_state_identity"][
            "nonpersistent_content_encoding"
        ],
        "maximum_nonpersistent_content_bytes_per_model": changed_state[
            "model_state_identity"
        ]["maximum_nonpersistent_content_bytes_per_model"],
        "models": changed_state["model_state_identity"]["models"],
    }
    import hashlib

    changed_state["model_state_identity"]["sha256"] = hashlib.sha256(
        b"o101-model-state-identity-v2\0"
        + json.dumps(
            state_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()
    changed_state_contract = core.build_mvroma_post_model_contract(changed_state)
    assert changed_state_contract["sha256"] != frozen["sha256"]
    assert changed_state_contract["base_sha256"] == frozen["base_sha256"]


def test_post_model_contract_rejects_schema_and_absolute_path_drift() -> None:
    missing = _post_model_identity_fixture()
    missing.pop("encoder")
    with pytest.raises(ValueError, match="post-model identity keys"):
        core.build_mvroma_post_model_contract(missing)

    invalid_phase = _post_model_identity_fixture()
    invalid_phase["runtime_identity"]["phase"] = "post_model"
    with pytest.raises(ValueError, match="post-model runtime phase"):
        core.build_mvroma_post_model_contract(invalid_phase)

    absolute = _post_model_identity_fixture()
    absolute["module_identity"]["modules"][0]["relative_path"] = "/tmp/model.py"
    _refresh_post_model_module_hash(absolute)
    with pytest.raises(ValueError, match="absolute path"):
        core.build_mvroma_post_model_contract(absolute)

    bad_module_hash = _post_model_identity_fixture()
    bad_module_hash["module_identity"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="module identity hash"):
        core.build_mvroma_post_model_contract(bad_module_hash)

    duplicate_load_key = _post_model_identity_fixture()
    duplicate_load_key["mvroma_state_load"]["unexpected_keys"] = ["x", "x"]
    with pytest.raises(ValueError, match="load-key identity"):
        core.build_mvroma_post_model_contract(duplicate_load_key)

    no_fused_attention = _post_model_identity_fixture()
    no_fused_attention["fused_attention"] = []
    with pytest.raises(ValueError, match="fused-attention"):
        core.build_mvroma_post_model_contract(no_fused_attention)


def test_post_model_runtime_transition_allows_backend_materialization_only() -> None:
    post = _post_model_identity_fixture()["runtime_identity"]
    pre = json.loads(json.dumps(post))
    pre["phase"] = "post_import_pre_model"
    for key in core.MVROMA_DINO_MATERIALIZED_BACKEND_KEYS:
        pre["attention_backend"][key] = None

    core.validate_mvroma_post_model_runtime(pre, post)

    for key in ("versions", "torch_flags", "gpu", "environment"):
        changed = json.loads(json.dumps(post))
        changed[key]["drift"] = True
        with pytest.raises(RuntimeError, match=f"runtime {key} drift"):
            core.validate_mvroma_post_model_runtime(pre, changed)

    invalid = json.loads(json.dumps(post))
    invalid["phase"] = "post_model"
    with pytest.raises(ValueError, match="post-model runtime phase"):
        core.validate_mvroma_post_model_runtime(pre, invalid)

    backend_drift = json.loads(json.dumps(post))
    backend_drift["attention_backend"]["uniception_use_fused_attn"] = False
    with pytest.raises(RuntimeError, match="attention backend drift"):
        core.validate_mvroma_post_model_runtime(pre, backend_drift)

    materialized_before_import = json.loads(json.dumps(pre))
    materialized_before_import["attention_backend"][
        "dino_attention_xformers_available"
    ] = False
    with pytest.raises(RuntimeError, match="DINO backend transition"):
        core.validate_mvroma_post_model_runtime(materialized_before_import, post)


def test_post_model_expectation_rebuilds_and_compares_complete_reference() -> None:
    post_model = _post_model_identity_fixture()
    pre_model_runtime = json.loads(json.dumps(post_model["runtime_identity"]))
    pre_model_runtime["phase"] = "post_import_pre_model"
    for key in core.MVROMA_DINO_MATERIALIZED_BACKEND_KEYS:
        pre_model_runtime["attention_backend"][key] = None
    contract = core.build_mvroma_post_model_contract(post_model)
    expected_ref = {
        "schema": "o101-post-model-contract-ref/v1",
        "contract_schema": contract["payload"]["schema"],
        "sha256": contract["sha256"],
        "base_sha256": contract["base_sha256"],
    }

    assert core.verify_mvroma_post_model_expectation(
        post_model,
        pre_model_runtime=pre_model_runtime,
        expected_ref=expected_ref,
    ) == contract

    wrong_ref = {**expected_ref, "sha256": "0" * 64}
    with pytest.raises(RuntimeError, match="post-model expectation mismatch"):
        core.verify_mvroma_post_model_expectation(
            post_model,
            pre_model_runtime=pre_model_runtime,
            expected_ref=wrong_ref,
        )

    runtime_drift = json.loads(json.dumps(pre_model_runtime))
    runtime_drift["environment"]["PYTHONHASHSEED"] = "drift"
    with pytest.raises(RuntimeError, match="runtime environment drift"):
        core.verify_mvroma_post_model_expectation(
            post_model,
            pre_model_runtime=runtime_drift,
            expected_ref=expected_ref,
        )


def test_collect_prepared_post_model_identity_is_contract_compatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _post_model_identity_fixture()
    events: list[str] = []

    class FusedAttention:
        fused_attn = True

    class DinoBlock:
        def __init__(self) -> None:
            self.attn = object()

    class DinoModel:
        def __init__(self) -> None:
            self.blocks = [DinoBlock()]

    class Encoder:
        def __init__(self) -> None:
            self.model = DinoModel()

    class PrematchModel:
        def __init__(self) -> None:
            self.encoder = Encoder()
            self.attention = FusedAttention()

        def named_modules(self) -> list[tuple[str, object]]:
            return [("attention", self.attention)]

    class MVModel:
        pass

    prematch = PrematchModel()

    def collect_modules(*args: object, **kwargs: object) -> dict:
        events.append("modules")
        return {"identity": fixture["module_identity"], "provenance": {}}

    def collect_state(models: dict, *, expected_device: str) -> dict:
        events.append("state")
        assert set(models) == {"mvroma", "ufm"}
        assert expected_device == "cuda:0"
        return {"identity": fixture["model_state_identity"], "provenance": {}}

    def probe_runtime(*args: object, **kwargs: object) -> dict:
        events.append("runtime")
        assert kwargs["phase"] == "post_model_pre_publish"
        return {"identity": fixture["runtime_identity"], "provenance": {}}

    monkeypatch.setattr(core, "collect_mvroma_module_identity", collect_modules)
    monkeypatch.setattr(core, "collect_mvroma_model_state_identity", collect_state)
    monkeypatch.setattr(core, "probe_mvroma_effective_runtime", probe_runtime)

    runtime = SimpleNamespace(
        runner_ufm_class=PrematchModel,
        vendored_ufm_class=PrematchModel,
        torch=object(),
    )
    prepared = SimpleNamespace(
        runtime_objects=runtime,
        source_roots={
            "identity": {
                "mvroma": {"files": []},
                "ufm": {"files": []},
            }
        },
        dino_source_identity={"files": []},
        private_mvroma_root="private-mvroma",
        private_ufm_root="private-ufm",
        private_dinov2_root="private-dino",
        initial_assets=fixture["post_model_assets"],
    )
    collected = core.collect_prepared_mvroma_post_model_identity(
        SimpleNamespace(device="cuda:0"),
        prepared,
        mvroma_loaded=SimpleNamespace(
            model=MVModel(),
            load_identity=fixture["mvroma_state_load"],
        ),
        ufm_loaded=SimpleNamespace(
            prematch=[None, prematch],
            load_identity=fixture["ufm_state_load"],
        ),
        post_model_assets=fixture["post_model_assets"],
    )

    assert events == ["modules", "state", "runtime"]
    assert collected.identity["runtime_identity"] == fixture["runtime_identity"]
    assert collected.identity["post_model_assets"] == fixture["post_model_assets"]
    assert core.build_mvroma_post_model_contract(collected.identity)["payload"][
        "schema"
    ] == "o101-post-model-contract/v3"


class _FakeTensor:
    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        dtype: str = "torch.float32",
        device: str = "cuda:0",
        requires_grad: bool = False,
        content: bytes = b"\x00\x00\x80>",
        reported_element_size: int | None = None,
    ) -> None:
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.requires_grad = requires_grad
        self.content = content
        self.reported_element_size = reported_element_size
        self.cpu_calls = 0

    def numel(self) -> int:
        count = 1
        for size in self.shape:
            count *= size
        return count

    def element_size(self) -> int:
        if self.reported_element_size is not None:
            return self.reported_element_size
        return len(self.content) // max(self.numel(), 1)

    def detach(self) -> "_FakeTensor":
        return self

    def cpu(self) -> "_FakeTensor":
        self.cpu_calls += 1
        return self

    def contiguous(self) -> "_FakeTensor":
        return self

    def numpy(self) -> object:
        return SimpleNamespace(tobytes=lambda order="C": self.content)


class _FakeStateModule:
    def __init__(
        self,
        *,
        training: bool = False,
        parameter_dtype: str = "torch.float32",
        device: str = "cuda:0",
        nonpersistent_content: bytes = b"\x00\x00\x80>",
        nonpersistent_element_size: int | None = None,
    ) -> None:
        self.training = training
        shared = _FakeTensor(
            (2, 3), dtype=parameter_dtype, device=device, requires_grad=True
        )
        persistent = _FakeTensor((3,), device=device)
        transient = _FakeTensor(
            (),
            device=device,
            content=nonpersistent_content,
            reported_element_size=nonpersistent_element_size,
        )
        self.child = SimpleNamespace(
            training=training,
            _buffers={"transient": transient},
            _non_persistent_buffers_set={"transient"},
        )
        self._buffers = {"running": persistent}
        self._non_persistent_buffers_set: set[str] = set()
        self._parameters = [("alias", shared), ("weight", shared)]

    def named_modules(self) -> list[tuple[str, object]]:
        return [("", self), ("child", self.child)]

    def named_parameters(
        self, *, remove_duplicate: bool = True
    ) -> list[tuple[str, _FakeTensor]]:
        if remove_duplicate:
            return [self._parameters[0]]
        return list(self._parameters)

    def named_buffers(
        self, *, remove_duplicate: bool = True
    ) -> list[tuple[str, _FakeTensor]]:
        return [
            ("running", self._buffers["running"]),
            ("child.transient", self.child._buffers["transient"]),
        ]


def test_model_state_identity_binds_structure_dtype_alias_and_persistence() -> None:
    first = core.collect_mvroma_model_state_identity(
        {
            "mvroma": _FakeStateModule(),
            "ufm": _FakeStateModule(),
        },
        expected_device="cuda:0",
    )

    assert first["identity"]["schema"] == "o101-model-state-identity/v2"
    assert set(first) == {"identity", "provenance"}
    assert first["identity"]["models"]["mvroma"]["training_modules"] == []
    parameters = first["identity"]["models"]["mvroma"]["parameters"]
    assert [row["alias_group"] for row in parameters] == [0, 0]
    buffers = first["identity"]["models"]["mvroma"]["buffers"]
    assert [row["persistent"] for row in buffers] == [False, True]
    assert buffers[0]["content_nbytes"] == 4
    assert buffers[0]["content_sha256"] == _sha256(b"\x00\x00\x80>")
    assert buffers[1]["content_nbytes"] is None
    assert buffers[1]["content_sha256"] is None
    assert "cuda" not in json.dumps(first["identity"])
    assert first["provenance"]["expected_device"] == "cuda:0"

    relocated = core.collect_mvroma_model_state_identity(
        {
            "mvroma": _FakeStateModule(device="cuda:1"),
            "ufm": _FakeStateModule(device="cuda:1"),
        },
        expected_device="cuda:1",
    )
    assert relocated["identity"] == first["identity"]
    assert relocated["provenance"] != first["provenance"]

    changed_dtype = core.collect_mvroma_model_state_identity(
        {
            "mvroma": _FakeStateModule(parameter_dtype="torch.float64"),
            "ufm": _FakeStateModule(),
        },
        expected_device="cuda:0",
    )
    assert changed_dtype["identity"]["sha256"] != first["identity"]["sha256"]

    changed_content = core.collect_mvroma_model_state_identity(
        {
            "mvroma": _FakeStateModule(
                nonpersistent_content=b"\x00\x00\x00?"
            ),
            "ufm": _FakeStateModule(),
        },
        expected_device="cuda:0",
    )
    assert changed_content["identity"]["sha256"] != first["identity"]["sha256"]


def test_model_state_identity_rejects_training_or_wrong_device() -> None:
    with pytest.raises(RuntimeError, match="training modules"):
        core.collect_mvroma_model_state_identity(
            {
                "mvroma": _FakeStateModule(training=True),
                "ufm": _FakeStateModule(),
            },
            expected_device="cuda:0",
        )
    with pytest.raises(RuntimeError, match="device mismatch"):
        core.collect_mvroma_model_state_identity(
            {
                "mvroma": _FakeStateModule(device="cpu"),
                "ufm": _FakeStateModule(),
            },
            expected_device="cuda:0",
        )
    oversized = _FakeStateModule(
        nonpersistent_content=b"x" * (1024 * 1024 + 1)
    )
    with pytest.raises(RuntimeError, match="non-persistent buffer content cap"):
        core.collect_mvroma_model_state_identity(
            {
                "mvroma": oversized,
                "ufm": _FakeStateModule(),
            },
            expected_device="cuda:0",
        )
    assert oversized.child._buffers["transient"].cpu_calls == 0

    with pytest.raises(RuntimeError, match="buffer byte count mismatch"):
        core.collect_mvroma_model_state_identity(
            {
                "mvroma": _FakeStateModule(
                    nonpersistent_content=b"bad",
                    nonpersistent_element_size=4,
                ),
                "ufm": _FakeStateModule(),
            },
            expected_device="cuda:0",
        )


def _module_fixture(root: Path) -> tuple[Path, Path, dict[str, object]]:
    mvroma = root / "mvroma"
    ufm = root / "ufm"
    relative_paths = {
        "src.build_model": "src/build_model.py",
        "src.mvroma": "src/mvroma/__init__.py",
        "src.run_model": "src/run_model.py",
        "src.matchers": "src/matchers/__init__.py",
        "src.matchers.run_matcher_path": "src/matchers/run_matcher_path.py",
        "src.matchers.uniflowmatch": "src/matchers/uniflowmatch/__init__.py",
        "src.matchers.uniflowmatch.models": "src/matchers/uniflowmatch/models/__init__.py",
        "src.matchers.uniflowmatch.models.ufm": "src/matchers/uniflowmatch/models/ufm.py",
        "src.matchers.uniflowmatch.models.base": "src/matchers/uniflowmatch/models/base.py",
        "src.mvroma.models.pipeline": "src/mvroma/models/pipeline.py",
        "src.mvroma.utils.grids": "src/mvroma/utils/grids.py",
        "src.track_cluster": "src/track_cluster.py",
        "uniflowmatch": "uniflowmatch/__init__.py",
        "uniflowmatch.models": "uniflowmatch/models/__init__.py",
        "uniflowmatch.models.base": "uniflowmatch/models/base.py",
        "uniflowmatch.models.ufm": "uniflowmatch/models/ufm.py",
        "uniflowmatch.models.unet_encoder": "uniflowmatch/models/unet_encoder.py",
        "uniflowmatch.models.utils": "uniflowmatch/models/utils.py",
        "uniflowmatch.utils": "uniflowmatch/utils/__init__.py",
        "uniflowmatch.utils.flow_resizing": "uniflowmatch/utils/flow_resizing.py",
        "uniflowmatch.utils.geometry": "uniflowmatch/utils/geometry.py",
        "uniception": "UniCeption/uniception/__init__.py",
        "uniception.models": "UniCeption/uniception/models/__init__.py",
        "uniception.models.encoders": "UniCeption/uniception/models/encoders/__init__.py",
        "uniception.models.encoders.base": "UniCeption/uniception/models/encoders/base.py",
        "uniception.models.encoders.dinov2": "UniCeption/uniception/models/encoders/dinov2.py",
        "uniception.models.encoders.image_normalizations": "UniCeption/uniception/models/encoders/image_normalizations.py",
        "uniception.models.info_sharing": "UniCeption/uniception/models/info_sharing/__init__.py",
        "uniception.models.info_sharing.global_attention_transformer": "UniCeption/uniception/models/info_sharing/global_attention_transformer.py",
        "uniception.models.prediction_heads": "UniCeption/uniception/models/prediction_heads/__init__.py",
        "uniception.models.prediction_heads.adaptors": "UniCeption/uniception/models/prediction_heads/adaptors.py",
        "uniception.models.prediction_heads.base": "UniCeption/uniception/models/prediction_heads/base.py",
        "uniception.models.prediction_heads.dpt": "UniCeption/uniception/models/prediction_heads/dpt.py",
        "uniception.models.prediction_heads.mlp_feature": "UniCeption/uniception/models/prediction_heads/mlp_feature.py",
        "uniception.models.prediction_heads.moge_conv": "UniCeption/uniception/models/prediction_heads/moge_conv.py",
        "uniception.models.utils.config": "UniCeption/uniception/models/utils/config.py",
        "uniception.models.utils.intermediate_feature_return": "UniCeption/uniception/models/utils/intermediate_feature_return.py",
        "uniception.models.utils.transformer_blocks": "UniCeption/uniception/models/utils/transformer_blocks.py",
    }
    modules: dict[str, object] = {}
    for name, relative in relative_paths.items():
        base = mvroma if name.startswith("src.") else ufm
        path = base / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"MODULE = {name!r}\n")
        modules[name] = SimpleNamespace(
            __file__=str(path),
            __loader__=SourceFileLoader(name, str(path)),
        )
    return mvroma, ufm, modules


def test_hybrid_module_identity_is_relocatable_and_rejects_shadowing(
    tmp_path: Path,
) -> None:
    first_mv, first_ufm, first_modules = _module_fixture(tmp_path / "first")
    second_mv, second_ufm, second_modules = _module_fixture(tmp_path / "second")

    first = core.collect_mvroma_module_identity(
        first_modules, mvroma_root=first_mv, ufm_root=first_ufm
    )
    second = core.collect_mvroma_module_identity(
        second_modules, mvroma_root=second_mv, ufm_root=second_ufm
    )
    assert first["identity"] == second["identity"]
    assert first["provenance"] != second["provenance"]
    assert str(tmp_path) not in json.dumps(first["identity"])

    shadow = tmp_path / "mvroma-shadow" / "src" / "build_model.py"
    shadow.parent.mkdir(parents=True)
    shadow.write_text("SHADOW = True\n")
    first_modules["src.build_model"] = SimpleNamespace(__file__=str(shadow))
    with pytest.raises(RuntimeError, match="origin"):
        core.collect_mvroma_module_identity(
            first_modules, mvroma_root=first_mv, ufm_root=first_ufm
        )

    wrong_relative = first_mv / "src" / "wrong_build_model.py"
    wrong_relative.write_text("SHADOW = True\n")
    _first_mv, _first_ufm, first_modules = _module_fixture(tmp_path / "first")
    first_modules["src.build_model"] = SimpleNamespace(__file__=str(wrong_relative))
    with pytest.raises(RuntimeError, match="relative|origin"):
        core.collect_mvroma_module_identity(
            first_modules, mvroma_root=first_mv, ufm_root=first_ufm
        )


@pytest.mark.parametrize(
    "required_name",
    [
        "src.mvroma",
        "src.matchers.run_matcher_path",
        "uniflowmatch.models.utils",
        "uniflowmatch.utils.flow_resizing",
        "uniception.models.encoders.dinov2",
        "uniception.models.info_sharing.global_attention_transformer",
        "uniception.models.prediction_heads.adaptors",
        "uniception.models.utils.config",
        "uniception.models.utils.transformer_blocks",
    ],
)
def test_hybrid_module_identity_requires_stage_and_transitive_sentinels(
    tmp_path: Path, required_name: str
) -> None:
    mvroma, ufm, modules = _module_fixture(tmp_path)
    modules.pop(required_name)

    with pytest.raises(RuntimeError, match="missing required"):
        core.collect_mvroma_module_identity(
            modules, mvroma_root=mvroma, ufm_root=ufm
        )


def _dinov2_module_fixture(root: Path) -> dict[str, object]:
    relative_paths = {
        "dinov2": "dinov2/__init__.py",
        "dinov2.hub": "dinov2/hub/__init__.py",
        "dinov2.hub.backbones": "dinov2/hub/backbones.py",
        "dinov2.hub.utils": "dinov2/hub/utils.py",
        "dinov2.models": "dinov2/models/__init__.py",
        "dinov2.models.vision_transformer": "dinov2/models/vision_transformer.py",
        "dinov2.layers": "dinov2/layers/__init__.py",
        "dinov2.layers.attention": "dinov2/layers/attention.py",
        "dinov2.layers.block": "dinov2/layers/block.py",
        "dinov2.layers.drop_path": "dinov2/layers/drop_path.py",
        "dinov2.layers.layer_scale": "dinov2/layers/layer_scale.py",
        "dinov2.layers.mlp": "dinov2/layers/mlp.py",
        "dinov2.layers.patch_embed": "dinov2/layers/patch_embed.py",
        "dinov2.layers.swiglu_ffn": "dinov2/layers/swiglu_ffn.py",
    }
    modules: dict[str, object] = {}
    for name, relative in relative_paths.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"MODULE = {name!r}\n")
        modules[name] = SimpleNamespace(
            __file__=str(path),
            __loader__=SourceFileLoader(name, str(path)),
        )
    return modules


def test_post_model_module_identity_requires_and_attests_dinov2_tree(
    tmp_path: Path,
) -> None:
    mvroma, ufm, modules = _module_fixture(tmp_path / "runtime")
    dinov2 = tmp_path / "dinov2-source"
    modules.update(_dinov2_module_fixture(dinov2))

    identity = core.collect_mvroma_module_identity(
        modules,
        mvroma_root=mvroma,
        ufm_root=ufm,
        dinov2_root=dinov2,
    )

    rows = {row["module"]: row for row in identity["identity"]["modules"]}
    assert rows["dinov2.models.vision_transformer"]["role"] == "dinov2"
    assert str(tmp_path) not in json.dumps(identity["identity"])

    modules.pop("dinov2.layers.attention")
    with pytest.raises(RuntimeError, match="missing.*dinov2.layers.attention"):
        core.collect_mvroma_module_identity(
            modules,
            mvroma_root=mvroma,
            ufm_root=ufm,
            dinov2_root=dinov2,
        )


@pytest.mark.parametrize(
    "namespace_name",
    ["dinov2.hub.cell_dino", "dinov2.hub.xray_dino"],
)
def test_post_model_module_identity_attests_exact_dinov2_hub_namespaces(
    tmp_path: Path, namespace_name: str
) -> None:
    mvroma, ufm, modules = _module_fixture(tmp_path / "runtime")
    dinov2 = tmp_path / "dinov2-source"
    modules.update(_dinov2_module_fixture(dinov2))
    relative = namespace_name.replace(".", "/")
    namespace_dir = dinov2 / relative
    namespace_dir.mkdir(parents=True)
    (namespace_dir / "backbones.py").write_text("MODEL = 'frozen'\n")

    saved_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "dinov2" or name.startswith("dinov2.")
    }
    for name in saved_modules:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(dinov2))
    try:
        namespace_module = importlib.import_module(namespace_name)
        modules[namespace_name] = namespace_module
        identity = core.collect_mvroma_module_identity(
            modules,
            mvroma_root=mvroma,
            ufm_root=ufm,
            dinov2_root=dinov2,
        )
    finally:
        sys.path.remove(str(dinov2))
        for name in list(sys.modules):
            if name == "dinov2" or name.startswith("dinov2."):
                sys.modules.pop(name, None)
        sys.modules.update(saved_modules)

    rows = {row["module"]: row for row in identity["identity"]["modules"]}
    assert rows[namespace_name] == {
        "module": namespace_name,
        "role": "dinov2",
        "relative_path": relative,
        "loader": "NamespaceLoader",
        "namespace": True,
        "sentinel": {
            "relative_path": f"{relative}/backbones.py",
            **core.mvroma_file_content_identity(namespace_dir / "backbones.py"),
        },
    }


def test_post_model_module_identity_rejects_namespace_shadow_variants(
    tmp_path: Path,
) -> None:
    mvroma, ufm, modules = _module_fixture(tmp_path / "runtime")
    dinov2 = tmp_path / "dinov2-source"
    modules.update(_dinov2_module_fixture(dinov2))
    namespace_name = "dinov2.hub.cell_dino"
    namespace_dir = dinov2 / "dinov2/hub/cell_dino"
    namespace_dir.mkdir(parents=True)
    (namespace_dir / "backbones.py").write_text("MODEL = 'frozen'\n")

    saved_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "dinov2" or name.startswith("dinov2.")
    }
    for name in saved_modules:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(dinov2))
    try:
        namespace_module = importlib.import_module(namespace_name)
        modules[namespace_name] = namespace_module
        original_loader = namespace_module.__loader__
        namespace_module.__loader__ = object()
        with pytest.raises(RuntimeError, match="namespace.*loader|loader.*namespace"):
            core.collect_mvroma_module_identity(
                modules,
                mvroma_root=mvroma,
                ufm_root=ufm,
                dinov2_root=dinov2,
            )
        namespace_module.__loader__ = original_loader

        original_path = namespace_module.__path__
        outside = tmp_path / "outside"
        outside.mkdir()
        namespace_module.__path__ = [str(outside)]
        with pytest.raises(RuntimeError, match="namespace.*origin|origin.*namespace"):
            core.collect_mvroma_module_identity(
                modules,
                mvroma_root=mvroma,
                ufm_root=ufm,
                dinov2_root=dinov2,
            )
        namespace_module.__path__ = original_path

        modules.pop(namespace_name)
        modules["dinov2.hub.unlisted"] = namespace_module
        with pytest.raises(RuntimeError, match="no origin"):
            core.collect_mvroma_module_identity(
                modules,
                mvroma_root=mvroma,
                ufm_root=ufm,
                dinov2_root=dinov2,
            )
    finally:
        sys.path.remove(str(dinov2))
        for name in list(sys.modules):
            if name == "dinov2" or name.startswith("dinov2."):
                sys.modules.pop(name, None)
        sys.modules.update(saved_modules)


def test_module_identity_rejects_loaded_dinov2_without_attested_root(
    tmp_path: Path,
) -> None:
    mvroma, ufm, modules = _module_fixture(tmp_path / "runtime")
    modules.update(_dinov2_module_fixture(tmp_path / "dinov2-source"))

    with pytest.raises(RuntimeError, match="DINOv2.*root|root.*DINOv2"):
        core.collect_mvroma_module_identity(
            modules,
            mvroma_root=mvroma,
            ufm_root=ufm,
        )


def test_hybrid_module_identity_rejects_bytecode_origin(tmp_path: Path) -> None:
    mvroma, ufm, modules = _module_fixture(tmp_path)
    pyc = mvroma / "src" / "__pycache__" / "build_model.cpython-310.pyc"
    pyc.parent.mkdir(exist_ok=True)
    pyc.write_bytes(b"stale bytecode")
    modules["src.build_model"] = SimpleNamespace(__file__=str(pyc))

    with pytest.raises(RuntimeError, match=r"bytecode|\.pyc"):
        core.collect_mvroma_module_identity(
            modules, mvroma_root=mvroma, ufm_root=ufm
        )


def test_exact_local_ufm_snapshot_ignores_refs_and_binds_expected_files(
    tmp_path: Path,
) -> None:
    revision = "1" * 40
    cache = tmp_path / "hub"
    snapshot = cache / "models--infinity1096--UFM-Refine" / "snapshots" / revision
    snapshot.mkdir(parents=True)
    config = b'{"model":"ufm"}'
    weights = b"weights"
    (snapshot / "config.json").write_bytes(config)
    (snapshot / "model.safetensors").write_bytes(weights)
    refs = snapshot.parents[1] / "refs"
    refs.mkdir()
    (refs / "main").write_text("f" * 40)
    expected = {
        "config.json": {"size": len(config), "sha256": _sha256(config)},
        "model.safetensors": {"size": len(weights), "sha256": _sha256(weights)},
    }

    resolved = core.resolve_local_ufm_snapshot(
        cache, revision=revision, expected_files=expected
    )
    assert resolved["revision"] == revision
    assert resolved["identity"]["files"] == expected
    assert resolved["snapshot_path"] == str(snapshot.resolve())

    (snapshot / "model.safetensors").write_bytes(b"WEIGHTS")
    with pytest.raises(RuntimeError, match="model.safetensors"):
        core.resolve_local_ufm_snapshot(
            cache, revision=revision, expected_files=expected
        )


def test_runtime_probe_uses_exact_keys_and_explicit_nulls() -> None:
    cuda_backend = SimpleNamespace(
        matmul=SimpleNamespace(
            allow_tf32=True,
            allow_fp16_reduced_precision_reduction=True,
            allow_bf16_reduced_precision_reduction=True,
        ),
        flash_sdp_enabled=lambda: True,
        mem_efficient_sdp_enabled=lambda: True,
        math_sdp_enabled=lambda: True,
        cudnn_sdp_enabled=lambda: True,
    )
    fake_torch = SimpleNamespace(
        __version__="2.test",
        version=SimpleNamespace(cuda=None),
        backends=SimpleNamespace(
            cuda=cuda_backend,
            cudnn=SimpleNamespace(
                allow_tf32=True,
                benchmark=False,
                deterministic=False,
                version=lambda: None,
            ),
        ),
        are_deterministic_algorithms_enabled=lambda: False,
        get_float32_matmul_precision=lambda: "high",
        get_default_dtype=lambda: "torch.float32",
    )
    versions = {
        key: None
        for key in (
            "python_implementation",
            "python",
            "torch",
            "cuda",
            "cudnn",
            "numpy",
            "h5py",
            "opencv",
            "pillow",
            "torchvision",
            "transformers",
            "huggingface_hub",
            "safetensors",
            "timm",
            "einops",
            "scipy",
        )
    }
    versions["torch"] = "2.test"
    config = SimpleNamespace(
        _HAS_FUSED_ATTN=True,
        _USE_FUSED_ATTN=0,
    )
    config.use_fused_attn = lambda: config._USE_FUSED_ATTN > 0
    modules = {"uniception.models.utils.config": config}

    result = core.probe_mvroma_effective_runtime(
        fake_torch,
        device="cpu",
        versions=lambda: versions,
        environ={
            "UNICEPTION_FUSED_ATTN": "0",
            "XFORMERS_DISABLED": "1",
            "PYTHONHASHSEED": "0",
            "TORCH_HOME": "/tmp/cache-a",
        },
        modules=modules,
    )

    identity = result["identity"]
    assert identity["phase"] == "post_import_pre_model"
    assert identity["versions"] == versions
    assert identity["versions"]["transformers"] is None
    assert identity["gpu"] == {
        "selected_device": "cpu",
        "name": None,
        "uuid": None,
        "compute_capability": None,
        "driver": None,
    }
    assert identity["environment"]["UNICEPTION_FUSED_ATTN"] == "0"
    assert identity["environment"]["XFORMERS_DISABLED"] == "1"
    assert identity["environment"]["PYTHONHASHSEED"] == "0"
    assert "TORCH_HOME" not in identity["environment"]
    assert result["provenance"]["environment_paths"]["TORCH_HOME"] == "/tmp/cache-a"
    assert identity["attention_backend"]["uniception_has_fused_attn"] is True
    assert identity["attention_backend"]["uniception_use_fused_attn"] is False
    assert identity["attention_backend"]["dino_attention_xformers_enabled"] is None
    assert set(identity["torch_flags"]) == set(core.MVROMA_TORCH_FLAG_KEYS)
    assert identity["torch_flags"]["cuda_matmul_allow_fp16_accumulation"] is None
    assert identity["torch_flags"]["cudnn_benchmark_limit"] is None
    assert identity["torch_flags"]["default_dtype"] == "torch.float32"

    config._USE_FUSED_ATTN = 1
    changed = core.probe_mvroma_effective_runtime(
        fake_torch,
        device="cpu",
        versions=lambda: versions,
        environ={
            "UNICEPTION_FUSED_ATTN": "1",
            "XFORMERS_DISABLED": "1",
            "TORCH_HOME": "/another/relocated/cache",
        },
        modules=modules,
    )
    assert changed["identity"] != identity
    assert changed["identity"]["attention_backend"]["uniception_use_fused_attn"] is True
    relocated_only = core.probe_mvroma_effective_runtime(
        fake_torch,
        device="cpu",
        versions=lambda: versions,
        environ={
            "UNICEPTION_FUSED_ATTN": "1",
            "XFORMERS_DISABLED": "1",
            "TORCH_HOME": "/third/cache/location",
        },
        modules=modules,
    )
    assert relocated_only["identity"] == changed["identity"]
    assert relocated_only["provenance"] != changed["provenance"]

    post_model = core.probe_mvroma_effective_runtime(
        fake_torch,
        device="cpu",
        phase="post_model_pre_publish",
        versions=lambda: versions,
        environ={
            "UNICEPTION_FUSED_ATTN": "1",
            "XFORMERS_DISABLED": "1",
        },
        modules=modules,
    )
    assert post_model["identity"]["phase"] == "post_model_pre_publish"
    with pytest.raises(ValueError, match="runtime phase"):
        core.probe_mvroma_effective_runtime(
            fake_torch,
            device="cpu",
            phase="post_model",
            versions=lambda: versions,
            environ={},
            modules=modules,
        )


def test_asset_attestation_detects_same_size_restored_mtime_drift(tmp_path: Path) -> None:
    paths = {}
    for key in ("mvroma_checkpoint", "ufm_config", "ufm_weights", "dinov2_weights"):
        path = tmp_path / key
        path.write_bytes((key + "00000000").encode())
        paths[key] = path
    dino = tmp_path / "dinov2_source"
    dino.mkdir()
    (dino / "hubconf.py").write_text("MODEL = 1\n")
    paths["dinov2_source"] = dino

    baseline = core.attest_mvroma_runtime_assets(paths, phase="initial")
    weight = paths["ufm_weights"]
    frozen_mtime = weight.stat().st_mtime_ns
    original = weight.read_bytes()
    weight.write_bytes(b"X" * len(original))
    os.utime(weight, ns=(frozen_mtime, frozen_mtime))

    with pytest.raises(RuntimeError, match="ufm_weights.*before_merge"):
        core.attest_mvroma_runtime_assets(
            paths, expected=baseline, phase="before_merge"
        )


def test_pinned_dinov2_guards_rewrite_local_load_and_bypass_decoy_cache(
    tmp_path: Path,
) -> None:
    source = tmp_path / "dinov2"
    source.mkdir()
    checkpoint = tmp_path / "pinned" / "dinov2_vitl14_pretrain.pth"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"PINNED")
    decoy = tmp_path / "torch-home" / "hub" / "checkpoints" / checkpoint.name
    decoy.parent.mkdir(parents=True)
    decoy.write_bytes(b"DECOY!")
    calls: list[tuple] = []
    state_delegate_calls = 0

    hub = SimpleNamespace()

    def original_state_delegate(url: str, **kwargs: object) -> bytes:
        nonlocal state_delegate_calls
        state_delegate_calls += 1
        return decoy.read_bytes()

    def original_load(repo: str, entrypoint: str, **kwargs: object) -> bytes:
        calls.append((repo, entrypoint, dict(kwargs)))
        return hub.load_state_dict_from_url(
            Path(str(kwargs["weights"])).resolve().as_uri(),
            map_location="cpu",
            check_hash=False,
            weights_only=True,
        )

    def direct_load(source_value: object, **kwargs: object) -> bytes:
        calls.append(("torch.load", hasattr(source_value, "read"), dict(kwargs)))
        assert hasattr(source_value, "read")
        return source_value.read()

    hub.load = original_load
    hub.load_state_dict_from_url = original_state_delegate
    fake_torch = SimpleNamespace(hub=hub, load=direct_load)

    with core.pinned_dinov2_torch_hub(
        fake_torch,
        source_root=source,
        checkpoint=checkpoint,
        expected_checkpoint=core.mvroma_file_content_identity(checkpoint),
    ):
        result = hub.load(
            "facebookresearch/dinov2", "dinov2_vitl14", force_reload=False
        )
        with pytest.raises(RuntimeError, match="repository"):
            hub.load("someone/else", "dinov2_vitl14")

    assert result == b"PINNED"
    assert state_delegate_calls == 0
    assert calls[0] == (
        str(source.resolve()),
        "dinov2_vitl14",
        {
            "source": "local",
            "pretrained": True,
            "weights": str(checkpoint.resolve()),
        },
    )
    assert calls[1] == (
        "torch.load",
        True,
        {"map_location": "cpu", "weights_only": True},
    )
    assert hub.load is original_load
    assert hub.load_state_dict_from_url is original_state_delegate


def test_pinned_ufm_loader_uses_exact_snapshot_kwargs_and_restores_guards(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot"
    source = tmp_path / "dinov2"
    checkpoint = tmp_path / "dino.pth"
    snapshot.mkdir()
    source.mkdir()
    (snapshot / "config.json").write_text("{}")
    (snapshot / "model.safetensors").write_bytes(b"ufm")
    (source / "hubconf.py").write_text("# fake\n")
    checkpoint.write_bytes(b"weights")
    calls: list[tuple] = []
    snapshot_observation: dict[str, object] = {}
    state_delegate_calls = 0

    hub = SimpleNamespace()

    def original_load(repo: str, entrypoint: str, **kwargs: object) -> bytes:
        return hub.load_state_dict_from_url(
            Path(str(kwargs["weights"])).resolve().as_uri(),
            map_location="cpu",
            weights_only=True,
        )

    def original_state(*args: object, **kwargs: object) -> None:
        nonlocal state_delegate_calls
        state_delegate_calls += 1

    def direct_load(source_value: object, **kwargs: object) -> bytes:
        assert hasattr(source_value, "read")
        return source_value.read()

    hub.load = original_load
    hub.load_state_dict_from_url = original_state
    fake_torch = SimpleNamespace(hub=hub, load=direct_load)

    class FakeModel:
        def eval(self) -> "FakeModel":
            calls.append(("eval",))
            return self

        def to(self, device: str) -> "FakeModel":
            calls.append(("to", device))
            return self

    class FakeUFM:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: object) -> FakeModel:
            calls.append(("from_pretrained", model_id, dict(kwargs)))
            private_snapshot = Path(model_id)
            snapshot_observation.update(
                {
                    "is_private": private_snapshot.resolve() != snapshot.resolve(),
                    "mode": private_snapshot.stat().st_mode & 0o777,
                    "config": (private_snapshot / "config.json").read_text(),
                    "weights": (private_snapshot / "model.safetensors").read_bytes(),
                }
            )
            fake_torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vitl14", force_reload=False
            )
            return FakeModel()

    loaded = core.load_pinned_ufm_model(
        FakeUFM,
        snapshot_path=snapshot,
        device="cuda:0",
        torch_module=fake_torch,
        dinov2_source=source,
        dinov2_weights=checkpoint,
        expected_assets=_expected_ufm_assets(snapshot, checkpoint),
    )

    assert loaded[0] is None
    assert isinstance(loaded[1], FakeModel)
    assert calls[0][0] == "from_pretrained"
    assert calls[0][2] == {
        "local_files_only": True,
        "map_location": "cpu",
        "strict": False,
    }
    assert calls[1:] == [("eval",), ("to", "cuda:0")]
    assert snapshot_observation == {
        "is_private": True,
        "mode": 0o700,
        "config": "{}",
        "weights": b"ufm",
    }
    assert state_delegate_calls == 0
    assert fake_torch.hub.load is original_load
    assert fake_torch.hub.load_state_dict_from_url is original_state


def test_pinned_mvroma_loader_uses_held_checkpoint_and_records_key_sets(
    tmp_path: Path,
) -> None:
    from argparse import Namespace

    checkpoint = tmp_path / "mvroma.pth"
    checkpoint.write_bytes(b"PINNED-MVROMA")
    events: list[object] = []

    class FakeConfig:
        num_cluster = 0

    class FakeModel:
        def load_state_dict(self, state: object, *, strict: bool) -> object:
            events.append(("load_state_dict", state, strict))
            return SimpleNamespace(
                missing_keys=["decoder.missing"],
                unexpected_keys=["legacy.unexpected"],
            )

        def eval(self) -> "FakeModel":
            events.append("eval")
            return self

        def to(self, device: str) -> "FakeModel":
            events.append(("to", device))
            return self

    model = FakeModel()

    def build_our_model(args: object, model_cfg: object, **kwargs: object) -> tuple:
        events.append(("build", vars(args), model_cfg.num_cluster, kwargs))
        return model, model_cfg

    def torch_load(source: object, **kwargs: object) -> bytes:
        assert hasattr(source, "read")
        events.append(("torch.load", kwargs))
        return source.read()

    runtime = SimpleNamespace(
        Namespace=Namespace,
        ModelConfig=FakeConfig,
        build_our_model=build_our_model,
        torch=SimpleNamespace(load=torch_load),
    )

    loaded = core.load_pinned_mvroma_model(
        runtime,
        checkpoint=checkpoint,
        expected_checkpoint=core.mvroma_file_content_identity(checkpoint),
        device="cuda:0",
    )

    assert loaded.model is model
    assert loaded.model_config.num_cluster == 512
    assert loaded.load_identity == {
        "schema": "mvroma-state-load/v1",
        "strict": False,
        "missing_keys": ["decoder.missing"],
        "unexpected_keys": ["legacy.unexpected"],
    }
    assert events == [
        (
            "build",
            {
                "use_dinov2": True,
                "train_until_16x": False,
                "train_refiner": False,
                "train_all_model": False,
                "num_cluster": 512,
            },
            512,
            {"use_dinov2": True},
        ),
        ("torch.load", {"map_location": "cpu", "weights_only": True}),
        ("load_state_dict", b"PINNED-MVROMA", False),
        "eval",
        ("to", "cuda:0"),
    ]


def test_pinned_mvroma_loader_rejects_key_drift_before_eval_or_device_move(
    tmp_path: Path,
) -> None:
    from argparse import Namespace

    checkpoint = tmp_path / "mvroma.pth"
    checkpoint.write_bytes(b"weights")
    events: list[str] = []

    class FakeConfig:
        num_cluster = 0

    class FakeModel:
        def load_state_dict(self, state: object, *, strict: bool) -> object:
            return SimpleNamespace(missing_keys=["actual"], unexpected_keys=[])

        def eval(self) -> "FakeModel":
            events.append("eval")
            return self

        def to(self, device: str) -> "FakeModel":
            events.append("to")
            return self

    runtime = SimpleNamespace(
        Namespace=Namespace,
        ModelConfig=FakeConfig,
        build_our_model=lambda *args, **kwargs: (FakeModel(), args[1]),
        torch=SimpleNamespace(load=lambda source, **kwargs: source.read()),
    )
    expected_load = {
        "schema": "mvroma-state-load/v1",
        "strict": False,
        "missing_keys": ["expected"],
        "unexpected_keys": [],
    }

    with pytest.raises(RuntimeError, match="key.*identity|identity.*key"):
        core.load_pinned_mvroma_model(
            runtime,
            checkpoint=checkpoint,
            expected_checkpoint=core.mvroma_file_content_identity(checkpoint),
            expected_load_identity=expected_load,
            device="cuda:0",
        )

    assert events == []


def test_pinned_ufm_loader_captures_one_safetensors_key_identity_and_restores(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot"
    dinov2_source = tmp_path / "dinov2"
    dinov2_weights = tmp_path / "dino.pth"
    snapshot.mkdir()
    dinov2_source.mkdir()
    (snapshot / "config.json").write_text("{}")
    (snapshot / "model.safetensors").write_bytes(b"ufm")
    (dinov2_source / "hubconf.py").write_text("# fake\n")
    dinov2_weights.write_bytes(b"dino")
    calls: list[tuple] = []

    class FakeModel:
        def eval(self) -> "FakeModel":
            return self

        def to(self, device: str) -> "FakeModel":
            return self

    model = FakeModel()

    def original_safetensors_load(
        loaded_model: object,
        filename: str,
        *,
        strict: bool,
        device: str,
        **kwargs: object,
    ) -> tuple[list[str], list[str]]:
        calls.append((loaded_model, filename, strict, device, kwargs))
        return ["ufm.missing"], ["ufm.unexpected"]

    fake_safetensors = SimpleNamespace(load_model=original_safetensors_load)
    hub = SimpleNamespace()

    def original_hub_load(repo: str, entrypoint: str, **kwargs: object) -> bytes:
        return hub.load_state_dict_from_url(
            Path(str(kwargs["weights"])).resolve().as_uri(),
            map_location="cpu",
            weights_only=True,
        )

    hub.load = original_hub_load
    hub.load_state_dict_from_url = lambda *args, **kwargs: b"unused"
    fake_torch = SimpleNamespace(
        hub=hub,
        load=lambda source, **kwargs: source.read(),
    )

    class FakeUFM:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: object) -> FakeModel:
            fake_safetensors.load_model(
                model,
                str(Path(model_id) / "model.safetensors"),
                strict=bool(kwargs["strict"]),
                device=str(kwargs["map_location"]),
            )
            fake_torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
            return model

    loaded = core.load_pinned_ufm_model_with_identity(
        FakeUFM,
        snapshot_path=snapshot,
        device="cuda:0",
        torch_module=fake_torch,
        dinov2_source=dinov2_source,
        dinov2_weights=dinov2_weights,
        expected_assets=_expected_ufm_assets(snapshot, dinov2_weights),
        safetensors_torch=fake_safetensors,
    )

    assert loaded.prematch == [None, model]
    assert loaded.load_identity == {
        "schema": "ufm-safetensors-state-load/v1",
        "strict": False,
        "missing_keys": ["ufm.missing"],
        "unexpected_keys": ["ufm.unexpected"],
    }
    assert len(calls) == 1
    assert calls[0][0] is model
    assert calls[0][2:] == (False, "cpu", {})
    assert Path(calls[0][1]).name == "model.safetensors"
    assert fake_safetensors.load_model is original_safetensors_load


def test_pinned_ufm_loader_rejects_key_drift_before_eval_or_device_move(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot"
    dinov2_source = tmp_path / "dinov2"
    dinov2_weights = tmp_path / "dino.pth"
    snapshot.mkdir()
    dinov2_source.mkdir()
    (snapshot / "config.json").write_text("{}")
    (snapshot / "model.safetensors").write_bytes(b"ufm")
    (dinov2_source / "hubconf.py").write_text("# fake\n")
    dinov2_weights.write_bytes(b"dino")
    events: list[str] = []

    class FakeModel:
        def eval(self) -> "FakeModel":
            events.append("eval")
            return self

        def to(self, device: str) -> "FakeModel":
            events.append(f"to:{device}")
            return self

    model = FakeModel()

    def original_safetensors_load(
        loaded_model: object,
        filename: str,
        *,
        strict: bool,
        device: str,
        **kwargs: object,
    ) -> tuple[list[str], list[str]]:
        return ["actual.missing"], []

    fake_safetensors = SimpleNamespace(load_model=original_safetensors_load)
    hub = SimpleNamespace()

    def original_hub_load(repo: str, entrypoint: str, **kwargs: object) -> bytes:
        return hub.load_state_dict_from_url(
            Path(str(kwargs["weights"])).resolve().as_uri(),
            map_location="cpu",
            weights_only=True,
        )

    hub.load = original_hub_load
    hub.load_state_dict_from_url = lambda *args, **kwargs: b"unused"
    fake_torch = SimpleNamespace(
        hub=hub,
        load=lambda source, **kwargs: source.read(),
    )

    class FakeUFM:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: object) -> FakeModel:
            fake_safetensors.load_model(
                model,
                str(Path(model_id) / "model.safetensors"),
                strict=bool(kwargs["strict"]),
                device=str(kwargs["map_location"]),
            )
            fake_torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
            return model

    expected_load = {
        "schema": "ufm-safetensors-state-load/v1",
        "strict": False,
        "missing_keys": ["expected.missing"],
        "unexpected_keys": [],
    }
    with pytest.raises(RuntimeError, match="key.*identity|identity.*key"):
        core.load_pinned_ufm_model_with_identity(
            FakeUFM,
            snapshot_path=snapshot,
            device="cuda:0",
            torch_module=fake_torch,
            dinov2_source=dinov2_source,
            dinov2_weights=dinov2_weights,
            expected_assets=_expected_ufm_assets(snapshot, dinov2_weights),
            expected_load_identity=expected_load,
            safetensors_torch=fake_safetensors,
        )

    assert events == []
    assert fake_safetensors.load_model is original_safetensors_load


def test_pinned_dinov2_rejects_preexisting_package_modules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "dinov2-source"
    source.mkdir()
    checkpoint = tmp_path / "dino.pth"
    checkpoint.write_bytes(b"weights")
    hub = SimpleNamespace(
        load=lambda *args, **kwargs: b"unexpected",
        load_state_dict_from_url=lambda *args, **kwargs: b"unexpected",
    )
    fake_torch = SimpleNamespace(hub=hub, load=lambda *args, **kwargs: b"unexpected")
    monkeypatch.setitem(
        sys.modules,
        "dinov2",
        SimpleNamespace(__file__="/tmp/untrusted/dinov2/__init__.py"),
    )

    with core.pinned_dinov2_torch_hub(
        fake_torch,
        source_root=source,
        checkpoint=checkpoint,
        expected_checkpoint=core.mvroma_file_content_identity(checkpoint),
    ):
        with pytest.raises(RuntimeError, match="preexisting.*dinov2"):
            hub.load("facebookresearch/dinov2", "dinov2_vitl14")


def test_pinned_ufm_loader_requires_exactly_one_dino_and_state_load(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot"
    source = tmp_path / "dinov2"
    checkpoint = tmp_path / "dino.pth"
    snapshot.mkdir()
    source.mkdir()
    (snapshot / "config.json").write_text("{}")
    (snapshot / "model.safetensors").write_bytes(b"ufm")
    (source / "hubconf.py").write_text("# fake\n")
    checkpoint.write_bytes(b"weights")

    class FakeModel:
        def eval(self) -> "FakeModel":
            return self

        def to(self, device: str) -> "FakeModel":
            return self

    class NoDinoUFM:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: object) -> FakeModel:
            return FakeModel()

    fake_torch = SimpleNamespace(
        hub=SimpleNamespace(
            load=lambda *args, **kwargs: None,
            load_state_dict_from_url=lambda *args, **kwargs: None,
        ),
        load=lambda *args, **kwargs: None,
    )

    with pytest.raises(RuntimeError, match="exactly one.*DINOv2"):
        core.load_pinned_ufm_model(
            NoDinoUFM,
            snapshot_path=snapshot,
            device="cpu",
            torch_module=fake_torch,
            dinov2_source=source,
            dinov2_weights=checkpoint,
            expected_assets=_expected_ufm_assets(snapshot, checkpoint),
        )


def test_pinned_dinov2_state_load_uses_verified_open_inode(
    tmp_path: Path,
) -> None:
    source = tmp_path / "dinov2"
    source.mkdir()
    (source / "hubconf.py").write_text("# fake\n")
    checkpoint = tmp_path / "dinov2_vitl14_pretrain.pth"
    checkpoint.write_bytes(b"PINNED")
    hub = SimpleNamespace()

    def original_load(repo: str, entrypoint: str, **kwargs: object) -> bytes:
        return hub.load_state_dict_from_url(
            Path(str(kwargs["weights"])).resolve().as_uri(),
            map_location="cpu",
            weights_only=True,
        )

    def adversarial_torch_load(source_value: object, **kwargs: object) -> bytes:
        if hasattr(source_value, "read"):
            return source_value.read()
        original = checkpoint.read_bytes()
        checkpoint.write_bytes(b"DECOY!")
        try:
            return Path(source_value).read_bytes()
        finally:
            checkpoint.write_bytes(original)

    hub.load = original_load
    hub.load_state_dict_from_url = lambda *args, **kwargs: b"wrong delegate"
    fake_torch = SimpleNamespace(hub=hub, load=adversarial_torch_load)

    with core.pinned_dinov2_torch_hub(
        fake_torch,
        source_root=source,
        checkpoint=checkpoint,
        expected_checkpoint=core.mvroma_file_content_identity(checkpoint),
    ):
        result = hub.load("facebookresearch/dinov2", "dinov2_vitl14")

    assert result == b"PINNED"


def test_pinned_dinov2_rejects_checkpoint_drift_before_hub_load(
    tmp_path: Path,
) -> None:
    source = tmp_path / "dinov2"
    source.mkdir()
    (source / "hubconf.py").write_text("# fake\n")
    checkpoint = tmp_path / "dinov2_vitl14_pretrain.pth"
    checkpoint.write_bytes(b"PINNED")
    expected = core.mvroma_file_content_identity(checkpoint)
    checkpoint.write_bytes(b"DRIFTED")
    events: list[str] = []
    hub = SimpleNamespace(
        load=lambda *args, **kwargs: events.append("hub.load"),
        load_state_dict_from_url=lambda *args, **kwargs: events.append("state.load"),
    )
    fake_torch = SimpleNamespace(
        hub=hub,
        load=lambda *args, **kwargs: events.append("torch.load"),
    )

    with pytest.raises(RuntimeError, match="identity|changed"):
        with core.pinned_dinov2_torch_hub(
            fake_torch,
            source_root=source,
            checkpoint=checkpoint,
            expected_checkpoint=expected,
        ):
            events.append("body")

    assert events == []


def test_pinned_ufm_loader_rejects_snapshot_drift_before_model_construction(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot"
    source = tmp_path / "dinov2"
    checkpoint = tmp_path / "dino.pth"
    snapshot.mkdir()
    source.mkdir()
    config = snapshot / "config.json"
    weights = snapshot / "model.safetensors"
    config.write_text('{"model":"pinned"}')
    weights.write_bytes(b"PINNED-UFM")
    (source / "hubconf.py").write_text("# fake\n")
    checkpoint.write_bytes(b"PINNED-DINO")
    expected_assets = {
        "ufm_config": core.mvroma_file_content_identity(config),
        "ufm_weights": core.mvroma_file_content_identity(weights),
        "dinov2_weights": core.mvroma_file_content_identity(checkpoint),
    }
    config.write_text('{"model":"drifted"}')
    events: list[str] = []

    class FakeUFM:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: object) -> object:
            events.append("from_pretrained")
            raise AssertionError("model construction must not start after asset drift")

    fake_torch = SimpleNamespace(
        hub=SimpleNamespace(
            load=lambda *args, **kwargs: events.append("hub.load"),
            load_state_dict_from_url=lambda *args, **kwargs: events.append("state.load"),
        ),
        load=lambda *args, **kwargs: events.append("torch.load"),
    )

    with pytest.raises(RuntimeError, match="identity|changed"):
        core.load_pinned_ufm_model(
            FakeUFM,
            snapshot_path=snapshot,
            device="cpu",
            torch_module=fake_torch,
            dinov2_source=source,
            dinov2_weights=checkpoint,
            expected_assets=expected_assets,
        )

    assert events == []


def test_pinned_ufm_loader_binds_snapshot_inodes_against_path_swap(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot"
    source = tmp_path / "dinov2"
    checkpoint = tmp_path / "dino.pth"
    snapshot.mkdir()
    source.mkdir()
    config = snapshot / "config.json"
    weights = snapshot / "model.safetensors"
    config.write_text("{}")
    weights.write_bytes(b"PINNED-UFM")
    (source / "hubconf.py").write_text("# fake\n")
    checkpoint.write_bytes(b"DINO")
    hub = SimpleNamespace()

    def state_load(repo: str, entrypoint: str, **kwargs: object) -> bytes:
        return hub.load_state_dict_from_url(
            Path(str(kwargs["weights"])).resolve().as_uri(),
            map_location="cpu",
            weights_only=True,
        )

    hub.load = state_load
    hub.load_state_dict_from_url = lambda *args, **kwargs: b"unused"

    def direct_load(source_value: object, **kwargs: object) -> bytes:
        return source_value.read() if hasattr(source_value, "read") else Path(source_value).read_bytes()

    fake_torch = SimpleNamespace(hub=hub, load=direct_load)
    observed_payloads: list[bytes] = []

    class FakeModel:
        def __init__(self, payload: bytes):
            self.payload = payload

        def eval(self) -> "FakeModel":
            return self

        def to(self, device: str) -> "FakeModel":
            return self

    class SwappingUFM:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: object) -> FakeModel:
            backup = snapshot / "model.original"
            decoy = tmp_path / "model.decoy"
            decoy.write_bytes(b"DECOY-UFM!")
            os.replace(weights, backup)
            os.replace(decoy, weights)
            try:
                payload = (Path(model_id) / "model.safetensors").read_bytes()
                observed_payloads.append(payload)
                fake_torch.hub.load(
                    "facebookresearch/dinov2", "dinov2_vitl14", force_reload=False
                )
            finally:
                os.replace(weights, decoy)
                os.replace(backup, weights)
            return FakeModel(payload)

    with pytest.raises(RuntimeError, match="changed.*model.safetensors|model.safetensors.*changed"):
        core.load_pinned_ufm_model(
            SwappingUFM,
            snapshot_path=snapshot,
            device="cpu",
            torch_module=fake_torch,
            dinov2_source=source,
            dinov2_weights=checkpoint,
            expected_assets=_expected_ufm_assets(snapshot, checkpoint),
        )

    assert observed_payloads == [b"PINNED-UFM"]


def test_pinned_ufm_loader_rejects_same_inode_mutation_even_if_bytes_restore(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot"
    source = tmp_path / "dinov2"
    checkpoint = tmp_path / "dino.pth"
    snapshot.mkdir()
    source.mkdir()
    config = snapshot / "config.json"
    weights = snapshot / "model.safetensors"
    config.write_text("{}")
    weights.write_bytes(b"PINNED-UFM")
    (source / "hubconf.py").write_text("# fake\n")
    checkpoint.write_bytes(b"DINO")
    hub = SimpleNamespace()

    def original_load(repo: str, entrypoint: str, **kwargs: object) -> bytes:
        return hub.load_state_dict_from_url(
            Path(str(kwargs["weights"])).resolve().as_uri(),
            map_location="cpu",
            weights_only=True,
        )

    hub.load = original_load
    hub.load_state_dict_from_url = lambda *args, **kwargs: b"unused"
    fake_torch = SimpleNamespace(
        hub=hub,
        load=lambda source_value, **kwargs: source_value.read(),
    )

    class MutatingUFM:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: object) -> object:
            original = weights.read_bytes()
            weights.write_bytes(b"MUTATED-UF")
            _ = (Path(model_id) / "model.safetensors").read_bytes()
            weights.write_bytes(original)
            fake_torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
            return SimpleNamespace(eval=lambda: None, to=lambda device: None)

    with pytest.raises(RuntimeError, match="changed.*model.safetensors|model.safetensors.*changed"):
        core.load_pinned_ufm_model(
            MutatingUFM,
            snapshot_path=snapshot,
            device="cpu",
            torch_module=fake_torch,
            dinov2_source=source,
            dinov2_weights=checkpoint,
            expected_assets=_expected_ufm_assets(snapshot, checkpoint),
        )


def _attested_execution_fixture(tmp_path: Path) -> tuple:
    import numpy as np

    image_root = tmp_path / "images"
    image_root.mkdir()
    (image_root / "source.jpg").write_bytes(b"source-image")
    (image_root / "target.jpg").write_bytes(b"target-image")
    jobs = core.build_mvroma_source_jobs(
        ["source.jpg target.jpg"], limit_src=0, chunk_size=1
    )
    image_tree = core.build_mvroma_image_sha256_tree(image_root, jobs)
    asset_paths: dict[str, Path] = {}
    for key in ("mvroma_checkpoint", "ufm_config", "ufm_weights", "dinov2_weights"):
        path = tmp_path / key
        path.write_bytes(key.encode())
        asset_paths[key] = path
    dino = tmp_path / "dinov2_source"
    dino.mkdir()
    (dino / "hubconf.py").write_text("MODEL = 1\n")
    asset_paths["dinov2_source"] = dino
    assets = core.attest_mvroma_runtime_assets(asset_paths, phase="initial")
    values = (
        np.arange(32, dtype=np.float32).reshape(16, 2),
        np.arange(32, dtype=np.float32).reshape(16, 2) + 0.5,
        np.linspace(0.1, 0.9, 16, dtype=np.float32),
    )
    return jobs, image_root, image_tree, asset_paths, assets, values


def _raise_primary_marker(error: BaseException) -> None:
    raise error


def _traceback_function_names(error: BaseException) -> list[str]:
    names: list[str] = []
    current = error.__traceback__
    while current is not None:
        names.append(current.tb_frame.f_code.co_name)
        current = current.tb_next
    return names


def _exception_graph_has_cycle(root: BaseException) -> bool:
    attributes = ("__cause__", "__context__")
    states: dict[int, int] = {id(root): 1}
    stack: list[tuple[BaseException, int]] = [(root, 0)]
    while stack:
        error, index = stack[-1]
        if index >= len(attributes):
            states[id(error)] = 2
            stack.pop()
            continue
        attribute = attributes[index]
        stack[-1] = (error, index + 1)
        linked = getattr(error, attribute)
        if linked is None:
            continue
        state = states.get(id(linked), 0)
        if state == 1:
            return True
        if state == 0:
            states[id(linked)] = 1
            stack.append((linked, 0))
    return False


def _exception_graph_messages(root: BaseException) -> set[str]:
    messages: set[str] = set()
    pending = [root]
    visited: set[int] = set()
    while pending:
        error = pending.pop()
        if id(error) in visited:
            continue
        visited.add(id(error))
        messages.add(str(error))
        if error.__cause__ is not None:
            pending.append(error.__cause__)
        if error.__context__ is not None:
            pending.append(error.__context__)
    return messages


def test_cleanup_error_attachment_preserves_existing_chain_without_cycles() -> None:
    primary = RuntimeError("runner primary")
    original_cause = ValueError("original cause")
    cleanup_a = RuntimeError("image cleanup a")
    cleanup_b = RuntimeError("image cleanup b")
    outer_cleanup = RuntimeError("outer root cleanup")
    primary.__cause__ = original_cause
    cleanup_a.__context__ = cleanup_b
    cleanup_b.__context__ = primary

    core._attach_mvroma_cleanup_error(primary, cleanup_a)
    core._attach_mvroma_cleanup_error(primary, outer_cleanup)

    assert primary.__cause__ is original_cause
    assert _exception_graph_has_cycle(primary) is False
    assert _exception_graph_messages(primary) == {
        "runner primary",
        "original cause",
        "image cleanup a",
        "image cleanup b",
        "outer root cleanup",
    }


def test_cleanup_error_attachment_is_visible_after_suppressed_context() -> None:
    import traceback

    try:
        try:
            raise ValueError("intentionally hidden context")
        except ValueError:
            raise RuntimeError("runner primary") from None
    except RuntimeError as caught:
        primary = caught

    cleanup = RuntimeError("visible cleanup failure")
    core._attach_mvroma_cleanup_error(primary, cleanup)

    rendered = "".join(
        traceback.format_exception(type(primary), primary, primary.__traceback__)
    )
    assert "runner primary" in rendered
    assert "visible cleanup failure" in rendered
    assert "intentionally hidden context" not in rendered


def test_cleanup_error_attachment_handles_deep_chain_iteratively() -> None:
    primary = RuntimeError("runner primary")
    cleanup = RuntimeError("cleanup 0")
    tail = cleanup
    for index in range(1, 1501):
        linked = RuntimeError(f"cleanup {index}")
        tail.__context__ = linked
        tail = linked

    core._attach_mvroma_cleanup_error(primary, cleanup)

    assert primary.__cause__ is cleanup
    assert _exception_graph_has_cycle(primary) is False
    assert "cleanup 1500" in _exception_graph_messages(primary)


def test_cleanup_error_attachment_repairs_existing_primary_cycle() -> None:
    primary = RuntimeError("runner primary")
    cleanup = RuntimeError("reused cleanup")
    primary.__cause__ = cleanup
    cleanup.__context__ = primary

    core._attach_mvroma_cleanup_error(primary, cleanup)

    assert primary.__cause__ is cleanup
    assert cleanup.__context__ is None
    assert _exception_graph_has_cycle(primary) is False


@pytest.mark.parametrize(
    ("error_type", "message"),
    [
        (KeyboardInterrupt, "runner interrupted"),
        (SystemExit, "runner requested exit"),
        (RuntimeError, "runner failed"),
        (RuntimeError, "CUDA error: device-side assert triggered"),
    ],
)
def test_attested_job_guard_cleanup_preserves_runner_primary(
    tmp_path: Path,
    error_type: type[BaseException],
    message: str,
) -> None:
    jobs, image_root, image_tree, asset_paths, assets, _values = (
        _attested_execution_fixture(tmp_path)
    )
    primary = error_type(message)
    cleanup = RuntimeError("job image cleanup failed")

    @contextmanager
    def failing_job_guard(
        root: Path, job: dict, expected_tree: dict
    ):
        del root, expected_tree
        try:
            yield dict(job)
        finally:
            raise cleanup

    def load_runner() -> object:
        def run(job: dict) -> dict:
            del job
            _raise_primary_marker(primary)

        return run

    with pytest.raises(BaseException) as caught:
        core.execute_attested_mvroma_resume(
            jobs,
            tmp_path / "shards",
            tmp_path / "final.h5",
            "d" * 64,
            image_root,
            image_tree,
            asset_paths,
            assets,
            max_correspondences=4000,
            mvroma_resume=True,
            overwrite=False,
            runner_loader=load_runner,
            job_image_guard=failing_job_guard,
        )

    assert caught.value is primary
    assert type(caught.value) is error_type
    assert str(caught.value) == message
    assert "_raise_primary_marker" in _traceback_function_names(caught.value)
    assert caught.value.__cause__ is cleanup
    assert not (tmp_path / "final.h5").exists()


def test_attested_default_image_guard_does_not_wrap_runner_runtime_error(
    tmp_path: Path,
) -> None:
    jobs, image_root, image_tree, asset_paths, assets, _values = (
        _attested_execution_fixture(tmp_path)
    )
    primary = RuntimeError("runner inference failed")

    def load_runner() -> object:
        def run(job: dict) -> dict:
            del job
            _raise_primary_marker(primary)

        return run

    with pytest.raises(RuntimeError) as caught:
        core.execute_attested_mvroma_resume(
            jobs,
            tmp_path / "shards",
            tmp_path / "final.h5",
            "d" * 64,
            image_root,
            image_tree,
            asset_paths,
            assets,
            max_correspondences=4000,
            mvroma_resume=True,
            overwrite=False,
            runner_loader=load_runner,
        )

    assert caught.value is primary
    assert "_raise_primary_marker" in _traceback_function_names(caught.value)
    assert caught.value.__cause__ is None
    assert not (tmp_path / "final.h5").exists()


def test_attested_job_guard_cleanup_without_primary_remains_primary(
    tmp_path: Path,
) -> None:
    jobs, image_root, image_tree, asset_paths, assets, values = (
        _attested_execution_fixture(tmp_path)
    )
    cleanup = RuntimeError("job image cleanup failed")

    @contextmanager
    def failing_job_guard(
        root: Path, job: dict, expected_tree: dict
    ):
        del root, expected_tree
        try:
            yield dict(job)
        finally:
            raise cleanup

    def load_runner() -> object:
        def run(job: dict) -> dict:
            return {str(job["targets"][0]): values}

        return run

    with pytest.raises(RuntimeError) as caught:
        core.execute_attested_mvroma_resume(
            jobs,
            tmp_path / "shards",
            tmp_path / "final.h5",
            "d" * 64,
            image_root,
            image_tree,
            asset_paths,
            assets,
            max_correspondences=4000,
            mvroma_resume=True,
            overwrite=False,
            runner_loader=load_runner,
            job_image_guard=failing_job_guard,
        )

    assert caught.value is cleanup
    assert not (tmp_path / "final.h5").exists()


def test_prepare_close_failure_is_cause_not_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = KeyboardInterrupt("runtime import interrupted")
    cleanup = RuntimeError("prepare stack close failed")
    pairs = tmp_path / "pairs.txt"
    pairs.write_text("source.jpg target.jpg\n")
    paths = SimpleNamespace(pairs=pairs, images=tmp_path / "images")
    cfg = SimpleNamespace(
        o101_mvroma_candidate=True,
        mvroma_chunk=1,
        limit_src=0,
        device="cpu",
    )
    mvroma_identity = {"sha256": "mvroma"}
    ufm_config_identity = {"sha256": "ufm-config"}
    ufm_weights_identity = {"sha256": "ufm-weights"}
    initial_assets = {
        "files": {
            "mvroma_checkpoint": mvroma_identity,
            "ufm_config": ufm_config_identity,
            "ufm_weights": ufm_weights_identity,
        }
    }
    resolved = SimpleNamespace(
        mvroma_root=tmp_path / "mvroma",
        ufm_root=tmp_path / "ufm",
        dinov2_source=tmp_path / "dino",
        dinov2_weights=tmp_path / "dino.pth",
        asset_paths={},
        ufm_snapshot={"identity": {}},
    )
    source_roots = {
        "identity": {
            "mvroma": {"files": []},
            "ufm": {"files": []},
        },
        "provenance": {},
    }

    class FailingStack:
        def enter_context(self, manager: object) -> object:
            return manager.__enter__()

        def close(self) -> None:
            raise cleanup

    @contextmanager
    def value_context(value: object):
        yield value

    def fail_import() -> None:
        _raise_primary_marker(primary)

    monkeypatch.setattr(core, "ExitStack", FailingStack)
    monkeypatch.setattr(core, "resolve_mvroma_runtime_paths", lambda _cfg: resolved)
    monkeypatch.setattr(
        core, "build_mvroma_image_sha256_tree", lambda _root, _jobs: {}
    )
    monkeypatch.setattr(
        core,
        "attest_mvroma_python_source_roots",
        lambda *args, **kwargs: source_roots,
    )
    monkeypatch.setattr(
        core,
        "_attest_dinov2_frozen_source",
        lambda _root: ({}, {"files": []}),
    )
    monkeypatch.setattr(
        core, "attest_mvroma_runtime_assets", lambda *args, **kwargs: initial_assets
    )
    monkeypatch.setattr(core, "MVROMA_CHECKPOINT_EXPECTED", mvroma_identity)
    monkeypatch.setattr(
        core,
        "MVROMA_UFM_EXPECTED_FILES",
        {
            "config.json": ufm_config_identity,
            "model.safetensors": ufm_weights_identity,
        },
    )
    monkeypatch.setattr(
        core,
        "private_attested_mvroma_source_tree",
        lambda root, _identity: value_context(root),
    )
    monkeypatch.setattr(
        core,
        "private_mvroma_import_environment",
        lambda _mvroma, _ufm: value_context(None),
    )
    monkeypatch.setattr(core, "import_mvroma_runtime_modules", fail_import)

    with pytest.raises(BaseException) as caught:
        core.prepare_mvroma_stage_runtime(cfg, paths)

    assert caught.value is primary
    assert "_raise_primary_marker" in _traceback_function_names(caught.value)
    assert caught.value.__cause__ is cleanup


def test_attested_execute_orders_load_image_guards_publish_and_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs, image_root, image_tree, asset_paths, assets, values = (
        _attested_execution_fixture(tmp_path)
    )
    events: list[str] = []

    def full_attestor(
        paths: dict, *, phase: str, expected: dict, **kwargs: object
    ) -> dict:
        events.append(f"full:{phase}")
        return core.attest_mvroma_runtime_assets(
            paths, phase=phase, expected=expected
        )

    @contextmanager
    def image_guard(
        root: Path, job: dict, expected_tree: dict
    ):
        events.append(f"image:enter:{job['source_index']}")
        with core.open_attested_mvroma_job_images(root, job, expected_tree) as bound:
            yield bound
        events.append(f"image:exit:{job['source_index']}")

    def global_image_attestor(root: Path, job_list: list[dict]) -> dict:
        events.append("global_image:pre_merge")
        return core.build_mvroma_image_sha256_tree(root, job_list)

    def finalize_runtime() -> None:
        events.append("runtime:finalize")

    def verify_post_model(post_model_assets: dict) -> None:
        assert post_model_assets == assets
        events.append("post_model:verify")

    def load_runner() -> object:
        events.append("model_load")

        def run(job: dict) -> dict:
            events.append("inference")
            return {str(job["targets"][0]): values}

        return run

    real_publish = core.publish_mvroma_source_shard_atomic
    real_merge = core.merge_mvroma_shards_atomic

    def publish(*args: object, **kwargs: object) -> dict:
        events.append("publish")
        return real_publish(*args, **kwargs)

    def merge(*args: object, **kwargs: object) -> dict:
        events.append("merge")
        return real_merge(*args, **kwargs)

    monkeypatch.setattr(core, "publish_mvroma_source_shard_atomic", publish)
    monkeypatch.setattr(core, "merge_mvroma_shards_atomic", merge)

    result = core.execute_attested_mvroma_resume(
        jobs,
        tmp_path / "shards",
        tmp_path / "final.h5",
        "d" * 64,
        image_root,
        image_tree,
        asset_paths,
        assets,
        max_correspondences=4000,
        mvroma_resume=True,
        overwrite=False,
        runner_loader=load_runner,
        full_attestor=full_attestor,
        job_image_guard=image_guard,
        global_image_attestor=global_image_attestor,
        pre_merge_runtime_finalizer=finalize_runtime,
        post_model_verifier=verify_post_model,
    )

    assert result["model_builds"] == 1
    assert events == [
        "model_load",
        "full:post_model_pre_publish",
        "post_model:verify",
        "image:enter:0",
        "inference",
        "image:exit:0",
        "publish",
        "full:pre_merge",
        "global_image:pre_merge",
        "runtime:finalize",
        "merge",
    ]


def test_attested_execute_postload_failure_publishes_nothing(
    tmp_path: Path,
) -> None:
    jobs, image_root, image_tree, asset_paths, assets, _values = (
        _attested_execution_fixture(tmp_path)
    )
    load_calls = 0

    def load_runner() -> object:
        nonlocal load_calls
        load_calls += 1
        return lambda job: {}

    def full_attestor(
        paths: dict, *, phase: str, expected: dict, **kwargs: object
    ) -> dict:
        if phase == "post_model_pre_publish":
            raise RuntimeError("post-load attestation failure")
        return core.attest_mvroma_runtime_assets(
            paths, phase=phase, expected=expected
        )

    with pytest.raises(RuntimeError, match="post-load"):
        core.execute_attested_mvroma_resume(
            jobs,
            tmp_path / "shards",
            tmp_path / "final.h5",
            "d" * 64,
            image_root,
            image_tree,
            asset_paths,
            assets,
            max_correspondences=4000,
            mvroma_resume=True,
            overwrite=False,
            runner_loader=load_runner,
            full_attestor=full_attestor,
        )

    assert load_calls == 1
    assert not list((tmp_path / "shards").glob("*.h5"))
    assert not (tmp_path / "final.h5").exists()


def test_attested_execute_post_model_verifier_failure_precedes_assets_and_inference(
    tmp_path: Path,
) -> None:
    jobs, image_root, image_tree, asset_paths, assets, _values = (
        _attested_execution_fixture(tmp_path)
    )
    events: list[str] = []

    def load_runner() -> object:
        events.append("model_load")

        def run(job: dict) -> dict:
            events.append("inference")
            return {}

        return run

    def reject_post_model(post_model_assets: dict) -> None:
        assert post_model_assets == assets
        events.append("post_model:verify")
        raise RuntimeError("post-model expectation mismatch")

    def full_attestor(*args: object, **kwargs: object) -> dict:
        events.append("asset_attestation")
        return assets

    with pytest.raises(RuntimeError, match="expectation mismatch"):
        core.execute_attested_mvroma_resume(
            jobs,
            tmp_path / "shards",
            tmp_path / "final.h5",
            "d" * 64,
            image_root,
            image_tree,
            asset_paths,
            assets,
            max_correspondences=4000,
            mvroma_resume=True,
            overwrite=False,
            runner_loader=load_runner,
            post_model_verifier=reject_post_model,
            full_attestor=full_attestor,
        )

    assert events == ["model_load", "asset_attestation", "post_model:verify"]
    assert not list((tmp_path / "shards").glob("*.h5"))
    assert not (tmp_path / "final.h5").exists()


def test_attested_execute_detects_image_drift_before_shard_publish(
    tmp_path: Path,
) -> None:
    jobs, image_root, image_tree, asset_paths, assets, values = (
        _attested_execution_fixture(tmp_path)
    )
    source = image_root / "source.jpg"
    frozen_mtime = source.stat().st_mtime_ns

    def load_runner() -> object:
        def run(job: dict) -> dict:
            old = source.read_bytes()
            source.write_bytes(b"X" * len(old))
            os.utime(source, ns=(frozen_mtime, frozen_mtime))
            return {str(job["targets"][0]): values}

        return run

    with pytest.raises(RuntimeError, match="image.*source 0"):
        core.execute_attested_mvroma_resume(
            jobs,
            tmp_path / "shards",
            tmp_path / "final.h5",
            "d" * 64,
            image_root,
            image_tree,
            asset_paths,
            assets,
            max_correspondences=4000,
            mvroma_resume=True,
            overwrite=False,
            runner_loader=load_runner,
        )

    assert not list((tmp_path / "shards").glob("*.h5"))
    assert not (tmp_path / "final.h5").exists()


def test_attested_execute_full_cache_hit_rechecks_before_merge_without_model(
    tmp_path: Path,
) -> None:
    jobs, image_root, image_tree, asset_paths, assets, values = (
        _attested_execution_fixture(tmp_path)
    )
    shard_dir = tmp_path / "shards"
    final = tmp_path / "final.h5"
    core.execute_attested_mvroma_resume(
        jobs,
        shard_dir,
        final,
        "d" * 64,
        image_root,
        image_tree,
        asset_paths,
        assets,
        max_correspondences=4000,
        mvroma_resume=True,
        overwrite=False,
        runner_loader=lambda: (
            lambda job: {str(job["targets"][0]): values}
        ),
    )
    phases: list[str] = []
    global_image_calls = 0
    job_guard_calls = 0
    runtime_finalizer_calls = 0

    def full_attestor(
        paths: dict, *, phase: str, expected: dict, **kwargs: object
    ) -> dict:
        phases.append(phase)
        return core.attest_mvroma_runtime_assets(
            paths, phase=phase, expected=expected
        )

    @contextmanager
    def forbidden_job_guard(root: Path, job: dict, expected_tree: dict):
        nonlocal job_guard_calls
        job_guard_calls += 1
        raise AssertionError("full cache hit must not open job images")
        yield

    def global_image_attestor(root: Path, job_list: list[dict]) -> dict:
        nonlocal global_image_calls
        global_image_calls += 1
        return core.build_mvroma_image_sha256_tree(root, job_list)

    def forbidden_loader() -> object:
        raise AssertionError("full cache hit must not load models")

    def forbidden_post_model_verifier(post_model_assets: dict) -> None:
        raise AssertionError("full cache hit must not verify post-model identity")

    def finalize_runtime() -> None:
        nonlocal runtime_finalizer_calls
        runtime_finalizer_calls += 1

    result = core.execute_attested_mvroma_resume(
        jobs,
        shard_dir,
        final,
        "d" * 64,
        image_root,
        image_tree,
        asset_paths,
        assets,
        max_correspondences=4000,
        mvroma_resume=True,
        overwrite=False,
        runner_loader=forbidden_loader,
        full_attestor=full_attestor,
        job_image_guard=forbidden_job_guard,
        global_image_attestor=global_image_attestor,
        pre_merge_runtime_finalizer=finalize_runtime,
        post_model_verifier=forbidden_post_model_verifier,
    )

    assert result["model_builds"] == 0
    assert result["reused_sources"] == 1
    assert phases == ["pre_merge"]
    assert global_image_calls == 1
    assert job_guard_calls == 0
    assert runtime_finalizer_calls == 1


def _synthetic_attested_jobs(count: int) -> tuple[list[dict], dict, tuple]:
    import numpy as np

    lines = [f"source-{index:03d}.jpg target-{index:03d}.jpg" for index in range(count)]
    jobs = core.build_mvroma_source_jobs(lines, limit_src=0, chunk_size=1)
    names = sorted(
        {
            name
            for job in jobs
            for name in [str(job["source"]), *[str(value) for value in job["targets"]]]
        }
    )
    rows = [[name, 1, _sha256(name.encode())] for name in names]
    payload = {"schema": "mvroma-image-tree/v1", "files": rows}
    image_tree = {
        **payload,
        "file_count": len(rows),
        "tree_sha256": _sha256(
            b"mvroma-image-tree-v1\0"
            + json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode()
        ),
        "by_name": {name: sha256_value for name, _size, sha256_value in rows},
    }
    values = (
        np.arange(32, dtype=np.float32).reshape(16, 2),
        np.arange(32, dtype=np.float32).reshape(16, 2) + 0.5,
        np.linspace(0.1, 0.9, 16, dtype=np.float32),
    )
    return jobs, image_tree, values


def test_attested_execute_full_attestation_volume_is_constant_for_1_and_100_sources(
    tmp_path: Path,
) -> None:
    observations: dict[int, tuple[list[tuple[str, int]], int]] = {}
    initial_assets = {"simulated_hashed_bytes": 6_592_000_000}

    for count in (1, 100):
        jobs, image_tree, values = _synthetic_attested_jobs(count)
        full_calls: list[tuple[str, int]] = []
        job_calls = 0

        def full_attestor(
            paths: dict, *, phase: str, expected: dict, **kwargs: object
        ) -> dict:
            full_calls.append((phase, int(expected["simulated_hashed_bytes"])))
            return expected

        @contextmanager
        def job_guard(root: Path, job: dict, expected_tree: dict):
            nonlocal job_calls
            job_calls += 1
            yield job

        result = core.execute_attested_mvroma_resume(
            jobs,
            tmp_path / f"shards-{count}",
            tmp_path / f"final-{count}.h5",
            "d" * 64,
            tmp_path,
            image_tree,
            {},
            initial_assets,
            max_correspondences=4000,
            mvroma_resume=False,
            overwrite=False,
            runner_loader=lambda: (
                lambda job: {str(job["targets"][0]): values}
            ),
            full_attestor=full_attestor,
            job_image_guard=job_guard,
            global_image_attestor=lambda root, job_list: image_tree,
        )
        assert result["model_builds"] == 1
        assert result["recomputed_sources"] == count
        observations[count] = (full_calls, job_calls)

    expected_full = [
        ("post_model_pre_publish", 6_592_000_000),
        ("pre_merge", 6_592_000_000),
    ]
    assert observations[1] == (expected_full, 1)
    assert observations[100] == (expected_full, 100)


def test_attested_execute_mixed_cache_guards_only_pending_source(tmp_path: Path) -> None:
    jobs, image_tree, values = _synthetic_attested_jobs(2)
    shard_dir = tmp_path / "shards"

    @contextmanager
    def passthrough_guard(root: Path, job: dict, expected_tree: dict):
        yield job

    core.execute_attested_mvroma_resume(
        jobs,
        shard_dir,
        tmp_path / "first.h5",
        "d" * 64,
        tmp_path,
        image_tree,
        {},
        {},
        max_correspondences=4000,
        mvroma_resume=True,
        overwrite=False,
        runner_loader=lambda: lambda job: {str(job["targets"][0]): values},
        full_attestor=lambda *args, **kwargs: {},
        job_image_guard=passthrough_guard,
        global_image_attestor=lambda root, job_list: image_tree,
    )
    (shard_dir / jobs[1]["shard_name"]).unlink()
    guarded: list[int] = []
    inferred: list[int] = []

    @contextmanager
    def job_guard(root: Path, job: dict, expected_tree: dict):
        guarded.append(int(job["source_index"]))
        yield job

    result = core.execute_attested_mvroma_resume(
        jobs,
        shard_dir,
        tmp_path / "second.h5",
        "d" * 64,
        tmp_path,
        image_tree,
        {},
        {},
        max_correspondences=4000,
        mvroma_resume=True,
        overwrite=False,
        runner_loader=lambda: lambda job: (
            inferred.append(int(job["source_index"]))
            or {str(job["targets"][0]): values}
        ),
        full_attestor=lambda *args, **kwargs: {},
        job_image_guard=job_guard,
        global_image_attestor=lambda root, job_list: image_tree,
    )

    assert result["reused_sources"] == 1
    assert result["recomputed_sources"] == 1
    assert guarded == [1]
    assert inferred == [1]


def test_attested_execute_image_precheck_failure_never_calls_runner(tmp_path: Path) -> None:
    jobs, image_root, image_tree, asset_paths, assets, _values = (
        _attested_execution_fixture(tmp_path)
    )
    (image_root / "source.jpg").write_bytes(b"changed-image")
    inference_calls = 0

    def runner_loader() -> object:
        def runner(job: dict) -> dict:
            nonlocal inference_calls
            inference_calls += 1
            return {}

        return runner

    with pytest.raises(RuntimeError, match="image guard.*source 0"):
        core.execute_attested_mvroma_resume(
            jobs,
            tmp_path / "shards",
            tmp_path / "final.h5",
            "d" * 64,
            image_root,
            image_tree,
            asset_paths,
            assets,
            max_correspondences=4000,
            mvroma_resume=True,
            overwrite=False,
            runner_loader=runner_loader,
        )

    assert inference_calls == 0
    assert not list((tmp_path / "shards").glob("*.h5"))


def test_attested_execute_path_swap_reads_held_image_then_aborts_publish(
    tmp_path: Path,
) -> None:
    jobs, image_root, image_tree, asset_paths, assets, values = (
        _attested_execution_fixture(tmp_path)
    )
    source = image_root / "source.jpg"
    observed: list[bytes] = []

    def runner_loader() -> object:
        def runner(job: dict) -> dict:
            backup = image_root / "source.original"
            decoy = image_root / "source.decoy"
            decoy.write_bytes(b"decoy-image!")
            os.replace(source, backup)
            os.replace(decoy, source)
            try:
                observed.append(Path(job["bound_source_path"]).read_bytes())
            finally:
                os.replace(source, decoy)
                os.replace(backup, source)
            return {str(job["targets"][0]): values}

        return runner

    with pytest.raises(RuntimeError, match="image guard.*source 0"):
        core.execute_attested_mvroma_resume(
            jobs,
            tmp_path / "shards",
            tmp_path / "final.h5",
            "d" * 64,
            image_root,
            image_tree,
            asset_paths,
            assets,
            max_correspondences=4000,
            mvroma_resume=True,
            overwrite=False,
            runner_loader=runner_loader,
        )

    assert observed == [b"source-image"]
    assert not list((tmp_path / "shards").glob("*.h5"))


def test_attested_execute_premerge_fail_keeps_shard_but_no_final(tmp_path: Path) -> None:
    jobs, image_root, image_tree, asset_paths, assets, values = (
        _attested_execution_fixture(tmp_path)
    )

    def full_attestor(
        paths: dict, *, phase: str, expected: dict, **kwargs: object
    ) -> dict:
        if phase == "pre_merge":
            raise RuntimeError("pre-merge drift")
        return core.attest_mvroma_runtime_assets(paths, phase=phase, expected=expected)

    with pytest.raises(RuntimeError, match="pre-merge drift"):
        core.execute_attested_mvroma_resume(
            jobs,
            tmp_path / "shards",
            tmp_path / "final.h5",
            "d" * 64,
            image_root,
            image_tree,
            asset_paths,
            assets,
            max_correspondences=4000,
            mvroma_resume=True,
            overwrite=False,
            runner_loader=lambda: lambda job: {str(job["targets"][0]): values},
            full_attestor=full_attestor,
        )

    assert len(list((tmp_path / "shards").glob("*.h5"))) == 1
    assert not (tmp_path / "final.h5").exists()


def test_attested_execute_runtime_finalize_failure_keeps_old_final(
    tmp_path: Path,
) -> None:
    jobs, image_root, image_tree, asset_paths, assets, values = (
        _attested_execution_fixture(tmp_path)
    )
    final_path = tmp_path / "final.h5"
    final_path.write_bytes(b"accepted-final")
    events: list[str] = []

    def finalize_runtime() -> None:
        events.append("runtime:finalize")
        raise RuntimeError("private runtime cleanup failed")

    with pytest.raises(RuntimeError, match="runtime cleanup failed"):
        core.execute_attested_mvroma_resume(
            jobs,
            tmp_path / "shards",
            final_path,
            "d" * 64,
            image_root,
            image_tree,
            asset_paths,
            assets,
            max_correspondences=4000,
            mvroma_resume=True,
            overwrite=False,
            runner_loader=lambda: lambda job: {
                str(job["targets"][0]): values
            },
            pre_merge_runtime_finalizer=finalize_runtime,
        )

    assert events == ["runtime:finalize"]
    assert len(list((tmp_path / "shards").glob("*.h5"))) == 1
    assert final_path.read_bytes() == b"accepted-final"


def test_attested_execute_releases_runner_before_runtime_finalizer(
    tmp_path: Path,
) -> None:
    import gc
    import weakref

    jobs, image_root, image_tree, asset_paths, assets, values = (
        _attested_execution_fixture(tmp_path)
    )
    payload_refs: list[weakref.ReferenceType[object]] = []

    class ModelPayload:
        pass

    def load_runner() -> object:
        payload = ModelPayload()
        payload_refs.append(weakref.ref(payload))

        def run(job: dict) -> dict:
            assert payload is not None
            return {str(job["targets"][0]): values}

        return run

    def finalize_runtime() -> None:
        gc.collect()
        assert payload_refs[0]() is None, "runner still retains model payload"

    result = core.execute_attested_mvroma_resume(
        jobs,
        tmp_path / "shards",
        tmp_path / "final.h5",
        "d" * 64,
        image_root,
        image_tree,
        asset_paths,
        assets,
        max_correspondences=4000,
        mvroma_resume=True,
        overwrite=False,
        runner_loader=load_runner,
        pre_merge_runtime_finalizer=finalize_runtime,
    )

    assert result["recomputed_sources"] == 1
    assert payload_refs[0]() is None


def test_attested_execute_full_cache_image_drift_blocks_new_final(tmp_path: Path) -> None:
    jobs, image_root, image_tree, asset_paths, assets, values = (
        _attested_execution_fixture(tmp_path)
    )
    shard_dir = tmp_path / "shards"
    core.execute_attested_mvroma_resume(
        jobs,
        shard_dir,
        tmp_path / "first.h5",
        "d" * 64,
        image_root,
        image_tree,
        asset_paths,
        assets,
        max_correspondences=4000,
        mvroma_resume=True,
        overwrite=False,
        runner_loader=lambda: lambda job: {str(job["targets"][0]): values},
    )
    (image_root / "source.jpg").write_bytes(b"changed-image")
    runtime_finalizer_calls = 0

    def finalize_runtime() -> None:
        nonlocal runtime_finalizer_calls
        runtime_finalizer_calls += 1

    with pytest.raises(RuntimeError, match="global image tree changed"):
        core.execute_attested_mvroma_resume(
            jobs,
            shard_dir,
            tmp_path / "second.h5",
            "d" * 64,
            image_root,
            image_tree,
            asset_paths,
            assets,
            max_correspondences=4000,
            mvroma_resume=True,
            overwrite=False,
            runner_loader=lambda: (_ for _ in ()).throw(
                AssertionError("full cache hit loaded model")
            ),
            pre_merge_runtime_finalizer=finalize_runtime,
        )

    assert not (tmp_path / "second.h5").exists()
    assert runtime_finalizer_calls == 0


def test_stage_mvroma_holds_one_lock_across_prepare_and_execute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []

    @contextmanager
    def lock(path: object):
        events.append("lock_enter")
        yield
        events.append("lock_exit")

    def prepare(cfg: object, paths: object) -> dict:
        events.append("prepare")
        return {"prepared": True}

    def execute(cfg: object, paths: object, prepared: dict) -> dict:
        assert prepared == {"prepared": True}
        events.append("execute")
        return {"groups": 0, "planned_sources": 0}

    monkeypatch.setattr(core, "mvroma_stage_lock", lock)
    monkeypatch.setattr(core, "prepare_mvroma_stage_runtime", prepare, raising=False)
    monkeypatch.setattr(core, "execute_prepared_mvroma_stage", execute, raising=False)
    cfg = SimpleNamespace(work_dir=str(tmp_path))

    core.stage_mvroma(cfg)

    assert events == ["lock_enter", "prepare", "execute", "lock_exit"]


def test_stage_mvroma_preserves_execute_error_when_final_close_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    @contextmanager
    def lock(path: object):
        yield

    def prepare(cfg: object, paths: object) -> object:
        def close() -> None:
            raise RuntimeError("final close failed")

        return SimpleNamespace(close=close)

    def execute(cfg: object, paths: object, prepared: object) -> None:
        raise KeyboardInterrupt("candidate interrupted")

    monkeypatch.setattr(core, "mvroma_stage_lock", lock)
    monkeypatch.setattr(core, "prepare_mvroma_stage_runtime", prepare)
    monkeypatch.setattr(core, "execute_prepared_mvroma_stage", execute)

    with pytest.raises(KeyboardInterrupt, match="candidate interrupted") as caught:
        core.stage_mvroma(SimpleNamespace(work_dir=str(tmp_path)))

    assert isinstance(caught.value.__cause__, RuntimeError)
    assert str(caught.value.__cause__) == "final close failed"


def test_execute_prepared_candidate_wires_snapshot_verifier_runner_and_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    post_model_ref = _post_model_expectation_ref()
    cfg = SimpleNamespace(
        device="cuda:0",
        mvroma_grid_h=560,
        mvroma_grid_w=840,
        mvroma_chunk=32,
        mvroma_sample_mode="score_grid",
        mvroma_sample_grid="8x12",
        roma_cert_thresh=0.35,
        agg_maxkp=4000,
        mvroma_resume=True,
        overwrite=False,
    )
    paths = SimpleNamespace(
        images=tmp_path / "images",
        dense_matches=tmp_path / "mvroma" / "matches.h5",
    )
    assets = {
        "schema": "mvroma-assets/v1",
        "files": {
            "mvroma_checkpoint": {"size": 1, "sha256": "a" * 64},
            "ufm_config": {"size": 1, "sha256": "b" * 64},
            "ufm_weights": {"size": 1, "sha256": "c" * 64},
            "dinov2_weights": {"size": 1, "sha256": "d" * 64},
        },
        "dinov2_source": {"schema": "tree"},
    }
    stage_contract = core.build_mvroma_stage_contract(
        implementation={},
        models={"runtime_assets": assets},
        inference=_candidate_inference_contract(),
        runtime=_candidate_runtime_contract("cuda:0"),
        post_model_expectation_ref=post_model_ref,
    )

    class UFMClass:
        pass

    runtime = SimpleNamespace(
        runner_ufm_class=UFMClass,
        torch=SimpleNamespace(
            cuda=SimpleNamespace(
                synchronize=lambda device: events.append("cuda_sync"),
                empty_cache=lambda: events.append("empty_cache"),
            )
        ),
    )
    mvroma_loaded = SimpleNamespace(model=object(), load_identity={})
    ufm_loaded = SimpleNamespace(prematch=[None, object()], load_identity={})
    raw_post_model = {"raw": "identity"}
    prepared = SimpleNamespace(
        cfg=cfg,
        paths=paths,
        runtime_objects=runtime,
        runtime_paths=SimpleNamespace(
            asset_paths={"mvroma_checkpoint": tmp_path / "mvroma.pth"},
            ufm_snapshot={"snapshot_path": tmp_path / "snapshot"},
            dinov2_weights=tmp_path / "dino.pth",
        ),
        private_dinov2_root=str(tmp_path / "private-dino"),
        initial_assets=assets,
        jobs=[{"source_index": 0}],
        image_tree={"schema": "mvroma-image-tree/v1", "by_name": {}},
        stage_contract=stage_contract,
        close=lambda: events.append("close"),
    )

    def load_mvroma(*args: object, **kwargs: object) -> object:
        events.append("load_mvroma")
        assert kwargs["expected_load_identity"] == core.MVROMA_STATE_LOAD_EXPECTED
        return mvroma_loaded

    def load_ufm(*args: object, **kwargs: object) -> object:
        events.append("load_ufm")
        assert kwargs["expected_load_identity"] == (
            core.MVROMA_UFM_STATE_LOAD_EXPECTED
        )
        return ufm_loaded

    def build_runner(*args: object, **kwargs: object) -> object:
        events.append("build_runner")
        assert kwargs["prematch"] is ufm_loaded.prematch

        def run(job: dict) -> dict:
            events.append("run")
            return {}

        return run

    def collect_identity(*args: object, **kwargs: object) -> object:
        events.append("collect")
        assert kwargs["post_model_assets"] == assets
        return SimpleNamespace(identity=raw_post_model)

    def verify_identity(
        post_model: dict, *, pre_model_runtime: dict, expected_ref: dict
    ) -> dict:
        events.append("verify")
        assert post_model is raw_post_model
        assert pre_model_runtime == _candidate_runtime_contract("cuda:0")
        assert expected_ref == post_model_ref
        return {"sha256": expected_ref["sha256"]}

    def execute_attested(*args: object, **kwargs: object) -> dict:
        events.append("execute")
        assert args[1] == paths.dense_matches.parent / "source-shards-v1"
        assert args[2] == paths.dense_matches
        assert args[3] == stage_contract["sha256"]
        runner = kwargs["runner_loader"]()
        prepared.stage_contract["payload"]["post_model_expectation_ref"][
            "sha256"
        ] = "f" * 64
        kwargs["post_model_verifier"](assets)
        runner({"source_index": 0})
        kwargs["pre_merge_runtime_finalizer"]()
        return {
            "groups": 0,
            "reused_sources": 0,
            "recomputed_sources": 1,
            "model_builds": 1,
        }

    monkeypatch.setattr(core, "load_pinned_mvroma_model", load_mvroma)
    monkeypatch.setattr(core, "load_pinned_ufm_model_with_identity", load_ufm)
    monkeypatch.setattr(core, "build_mvroma_candidate_source_runner", build_runner)
    monkeypatch.setattr(
        core, "collect_prepared_mvroma_post_model_identity", collect_identity
    )
    monkeypatch.setattr(core, "verify_mvroma_post_model_expectation", verify_identity)
    monkeypatch.setattr(core, "execute_attested_mvroma_resume", execute_attested)

    result = core.execute_prepared_mvroma_stage(cfg, paths, prepared)

    assert result == {
        "groups": 0,
        "reused_sources": 0,
        "recomputed_sources": 1,
        "model_builds": 1,
    }
    assert events == [
        "execute",
        "load_mvroma",
        "load_ufm",
        "build_runner",
        "collect",
        "verify",
        "run",
        "cuda_sync",
        "empty_cache",
        "close",
    ]


def test_execute_prepared_candidate_fake_cpu_end_to_end_and_full_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import gc
    import weakref

    jobs, image_root, image_tree, asset_paths, assets, values = (
        _attested_execution_fixture(tmp_path)
    )
    events: list[str] = []
    reference = _post_model_expectation_ref()
    stage_contract = core.build_mvroma_stage_contract(
        implementation={},
        models={"runtime_assets": assets},
        inference=_candidate_inference_contract(),
        runtime=_candidate_runtime_contract("cpu"),
        post_model_expectation_ref=reference,
    )
    cfg = SimpleNamespace(
        device="cpu",
        mvroma_grid_h=560,
        mvroma_grid_w=840,
        mvroma_chunk=32,
        mvroma_sample_mode="score_grid",
        mvroma_sample_grid="8x12",
        roma_cert_thresh=0.35,
        agg_maxkp=4000,
        mvroma_resume=True,
        overwrite=False,
    )
    paths = SimpleNamespace(
        images=image_root,
        dense_matches=tmp_path / "work" / "matches-mvroma-dense.h5",
    )

    class UFMClass:
        pass

    class ModelToken:
        pass

    runtime = SimpleNamespace(runner_ufm_class=UFMClass, torch=object())
    raw_post_model = {"raw": "identity"}
    model_refs: list[weakref.ReferenceType[object]] = []

    def prepared_runtime(
        close_event: str, contract: dict | None = None
    ) -> SimpleNamespace:
        def close() -> None:
            gc.collect()
            assert all(reference() is None for reference in model_refs)
            events.append(close_event)

        return SimpleNamespace(
            cfg=cfg,
            paths=paths,
            runtime_objects=runtime,
            runtime_paths=SimpleNamespace(
                asset_paths=asset_paths,
                ufm_snapshot={"snapshot_path": tmp_path / "snapshot"},
                dinov2_weights=asset_paths["dinov2_weights"],
            ),
            private_dinov2_root=str(asset_paths["dinov2_source"]),
            initial_assets=assets,
            jobs=jobs,
            image_tree=image_tree,
            stage_contract=json.loads(json.dumps(contract or stage_contract)),
            close=close,
        )

    def load_mvroma(*args: object, **kwargs: object) -> object:
        events.append("load_mvroma")
        model = ModelToken()
        model_refs.append(weakref.ref(model))
        return SimpleNamespace(model=model, load_identity={})

    def load_ufm(*args: object, **kwargs: object) -> object:
        events.append("load_ufm")
        model = ModelToken()
        model_refs.append(weakref.ref(model))
        return SimpleNamespace(prematch=[None, model], load_identity={})

    def build_runner(*args: object, **kwargs: object) -> object:
        events.append("build_runner")
        mvroma_model = kwargs["model"]
        prematch_model = kwargs["prematch"]

        def run(job: dict) -> dict:
            assert mvroma_model is not None
            assert prematch_model is not None
            events.append("inference")
            return {str(job["targets"][0]): values}

        return run

    def collect_identity(*args: object, **kwargs: object) -> object:
        events.append("collect")
        assert kwargs["post_model_assets"] == assets
        return SimpleNamespace(identity=raw_post_model)

    def verify_identity(
        post_model: dict, *, pre_model_runtime: dict, expected_ref: dict
    ) -> dict:
        events.append("verify")
        assert post_model is raw_post_model
        assert expected_ref == reference
        return {"sha256": expected_ref["sha256"]}

    monkeypatch.setattr(core, "load_pinned_mvroma_model", load_mvroma)
    monkeypatch.setattr(core, "load_pinned_ufm_model_with_identity", load_ufm)
    monkeypatch.setattr(core, "build_mvroma_candidate_source_runner", build_runner)
    monkeypatch.setattr(
        core, "collect_prepared_mvroma_post_model_identity", collect_identity
    )
    monkeypatch.setattr(core, "verify_mvroma_post_model_expectation", verify_identity)

    first = core.execute_prepared_mvroma_stage(
        cfg, paths, prepared_runtime("close_first")
    )

    assert first is not None
    assert first["model_builds"] == 1
    assert first["recomputed_sources"] == 1
    assert first["groups"] == 1
    assert paths.dense_matches.is_file()
    assert len(list((paths.dense_matches.parent / "source-shards-v1").glob("*.h5"))) == 1
    assert events == [
        "load_mvroma",
        "load_ufm",
        "build_runner",
        "collect",
        "verify",
        "inference",
        "close_first",
    ]

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("full cache loaded or verified models")

    monkeypatch.setattr(core, "load_pinned_mvroma_model", forbidden)
    monkeypatch.setattr(core, "load_pinned_ufm_model_with_identity", forbidden)
    monkeypatch.setattr(core, "build_mvroma_candidate_source_runner", forbidden)
    monkeypatch.setattr(
        core, "collect_prepared_mvroma_post_model_identity", forbidden
    )
    monkeypatch.setattr(core, "verify_mvroma_post_model_expectation", forbidden)
    events.clear()

    cached = core.execute_prepared_mvroma_stage(
        cfg, paths, prepared_runtime("close_cached")
    )

    assert cached is not None
    assert cached["model_builds"] == 0
    assert cached["reused_sources"] == 1
    assert cached["recomputed_sources"] == 0
    assert cached["groups"] == 1
    assert events == ["close_cached"]

    changed_reference = {**reference, "sha256": "c" * 64}
    changed_stage_contract = core.build_mvroma_stage_contract(
        implementation={},
        models={"runtime_assets": assets},
        inference=_candidate_inference_contract(),
        runtime=_candidate_runtime_contract("cpu"),
        post_model_expectation_ref=changed_reference,
    )

    def verify_changed(
        post_model: dict, *, pre_model_runtime: dict, expected_ref: dict
    ) -> dict:
        events.append("verify")
        assert expected_ref == changed_reference
        return {"sha256": expected_ref["sha256"]}

    monkeypatch.setattr(core, "load_pinned_mvroma_model", load_mvroma)
    monkeypatch.setattr(core, "load_pinned_ufm_model_with_identity", load_ufm)
    monkeypatch.setattr(core, "build_mvroma_candidate_source_runner", build_runner)
    monkeypatch.setattr(
        core, "collect_prepared_mvroma_post_model_identity", collect_identity
    )
    monkeypatch.setattr(core, "verify_mvroma_post_model_expectation", verify_changed)
    events.clear()

    changed = core.execute_prepared_mvroma_stage(
        cfg,
        paths,
        prepared_runtime("close_changed", changed_stage_contract),
    )

    assert changed is not None
    assert changed["model_builds"] == 1
    assert changed["reused_sources"] == 0
    assert changed["recomputed_sources"] == 1
    assert changed_stage_contract["sha256"] != stage_contract["sha256"]
    assert events == [
        "load_mvroma",
        "load_ufm",
        "build_runner",
        "collect",
        "verify",
        "inference",
        "close_changed",
    ]


def test_candidate_loader_failure_releases_partial_model_before_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import gc
    import weakref

    events: list[str] = []
    token_refs: list[weakref.ReferenceType[object]] = []
    alive_at_close: list[bool] = []
    assets = {"files": {"mvroma_checkpoint": {}, "ufm_config": {}, "ufm_weights": {}, "dinov2_weights": {}}}
    stage_contract = core.build_mvroma_stage_contract(
        implementation={},
        models={"runtime_assets": assets},
        inference=_candidate_inference_contract(),
        runtime=_candidate_runtime_contract("cpu"),
        post_model_expectation_ref=_post_model_expectation_ref(),
    )
    cfg = SimpleNamespace(
        device="cpu",
        mvroma_grid_h=560,
        mvroma_grid_w=840,
        mvroma_chunk=32,
        mvroma_sample_mode="score_grid",
        mvroma_sample_grid="8x12",
        roma_cert_thresh=0.35,
        agg_maxkp=4000,
        mvroma_resume=True,
        overwrite=False,
    )
    paths = SimpleNamespace(
        images=tmp_path / "images",
        dense_matches=tmp_path / "work" / "matches.h5",
    )

    class Token:
        pass

    class UFMClass:
        pass

    def close() -> None:
        gc.collect()
        events.append("close")
        alive_at_close.append(token_refs[0]() is not None)

    prepared = SimpleNamespace(
        cfg=cfg,
        paths=paths,
        runtime_objects=SimpleNamespace(
            runner_ufm_class=UFMClass, torch=object()
        ),
        runtime_paths=SimpleNamespace(
            asset_paths={"mvroma_checkpoint": tmp_path / "mvroma.pth"},
            ufm_snapshot={"snapshot_path": tmp_path / "snapshot"},
            dinov2_weights=tmp_path / "dino.pth",
        ),
        private_dinov2_root=str(tmp_path / "private-dino"),
        initial_assets=assets,
        jobs=[{"source_index": 0}],
        image_tree={"schema": "mvroma-image-tree/v1", "by_name": {}},
        stage_contract=stage_contract,
        close=close,
    )

    def load_mvroma(*args: object, **kwargs: object) -> object:
        events.append("load_mvroma")
        token = Token()
        token_refs.append(weakref.ref(token))
        return SimpleNamespace(model=token, load_identity={})

    def load_ufm(*args: object, **kwargs: object) -> object:
        events.append("load_ufm")
        raise KeyboardInterrupt("UFM loader interrupted")

    def execute_attested(*args: object, **kwargs: object) -> dict:
        kwargs["runner_loader"]()
        raise AssertionError("unreachable")

    monkeypatch.setattr(core, "load_pinned_mvroma_model", load_mvroma)
    monkeypatch.setattr(core, "load_pinned_ufm_model_with_identity", load_ufm)
    monkeypatch.setattr(core, "execute_attested_mvroma_resume", execute_attested)

    with pytest.raises(KeyboardInterrupt, match="UFM loader interrupted"):
        core.execute_prepared_mvroma_stage(cfg, paths, prepared)

    gc.collect()
    assert events == ["load_mvroma", "load_ufm", "close"]
    assert alive_at_close == [False]
    assert token_refs[0]() is None


def test_prepare_mvroma_stage_runtime_builds_contract_and_closes_private_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch

    source_mvroma, source_ufm, fixture_modules = _module_fixture(tmp_path / "source")
    subprocess.run(["git", "init", "-q"], cwd=source_mvroma, check=True)
    subprocess.run(["git", "add", "*.py", "src"], cwd=source_mvroma, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=O101",
            "-c",
            "user.email=o101@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=source_mvroma,
        check=True,
    )
    source_identity = core.attest_mvroma_python_source_roots(
        source_mvroma, source_ufm
    )
    source_expected = {
        "mvroma_git_head": source_identity["provenance"]["mvroma_git_head"],
        "mvroma_file_count": source_identity["identity"]["mvroma"]["file_count"],
        "mvroma_tree_sha256": source_identity["identity"]["mvroma"]["tree_sha256"],
        "ufm_file_count": source_identity["identity"]["ufm"]["file_count"],
        "ufm_tree_sha256": source_identity["identity"]["ufm"]["tree_sha256"],
    }
    monkeypatch.setattr(
        core, "MVROMA_PYTHON_SOURCE_EXPECTED", source_expected, raising=False
    )

    cache = tmp_path / "hf-cache"
    snapshot = (
        cache
        / "models--infinity1096--UFM-Refine"
        / "snapshots"
        / core.MVROMA_UFM_REVISION
    )
    snapshot.mkdir(parents=True)
    config = snapshot / "config.json"
    ufm_weights = snapshot / "model.safetensors"
    config.write_text("{}")
    ufm_weights.write_bytes(b"tiny-ufm")
    ufm_expected = {
        "config.json": core.mvroma_file_content_identity(config),
        "model.safetensors": core.mvroma_file_content_identity(ufm_weights),
    }
    monkeypatch.setattr(core, "MVROMA_UFM_EXPECTED_FILES", ufm_expected)

    dino_source = tmp_path / "dinov2"
    dino_source.mkdir()
    (dino_source / "hubconf.py").write_text("MODEL = 1\n")
    dino_tree = core.mvroma_tree_content_identity(dino_source)
    monkeypatch.setattr(
        core,
        "MVROMA_DINOV2_SOURCE_EXPECTED",
        {
            "file_count": dino_tree["file_count"],
            "sha256sum_sha256": dino_tree["sha256sum_sha256"],
        },
    )
    dino_weights = tmp_path / "dino.pth"
    dino_weights.write_bytes(b"tiny-dino")
    monkeypatch.setattr(
        core,
        "MVROMA_DINOV2_WEIGHTS_EXPECTED",
        core.mvroma_file_content_identity(dino_weights),
    )
    mvroma_weights = tmp_path / "mvroma.pth"
    mvroma_weights.write_bytes(b"tiny-mvroma")
    monkeypatch.setattr(
        core,
        "MVROMA_CHECKPOINT_EXPECTED",
        core.mvroma_file_content_identity(mvroma_weights),
        raising=False,
    )

    cfg = SimpleNamespace(
        work_dir=str(tmp_path / "work"),
        mvroma_root=str(source_mvroma),
        ufm_root=str(source_ufm),
        ufm_hf_hub_cache=str(cache),
        dinov2_source_root=str(dino_source),
        dinov2_weights=str(dino_weights),
        mvroma_weights=str(mvroma_weights),
        mvroma_chunk=1,
        limit_src=0,
        device="cpu",
        mvroma_grid_h=560,
        mvroma_grid_w=840,
        mvroma_sample_mode="score_grid",
        mvroma_sample_grid="8x12",
        roma_cert_thresh=0.35,
        agg_maxkp=4000,
        o101_mvroma_candidate=True,
    )
    paths = core.cfg_paths(cfg)
    paths.images.mkdir(parents=True)
    (paths.images / "source.jpg").write_bytes(b"source")
    (paths.images / "target.jpg").write_bytes(b"target")
    paths.pairs.parent.mkdir(parents=True, exist_ok=True)
    paths.pairs.write_text("source.jpg target.jpg\n")
    before_cwd = Path.cwd()
    before_path = list(sys.path)
    expected_post_model_ref = json.loads(
        json.dumps(core.MVROMA_POST_MODEL_EXPECTED_REF)
    )

    def fake_importer() -> SimpleNamespace:
        private_mvroma = Path(sys.path[0]).resolve()
        private_ufm = Path(sys.path[1]).resolve()
        imported: dict[str, object] = {}
        for name, original_module in fixture_modules.items():
            original = Path(str(original_module.__file__))
            root = source_mvroma if name.startswith("src.") else source_ufm
            private_root = private_mvroma if name.startswith("src.") else private_ufm
            relative = original.relative_to(root)
            private_path = private_root / relative
            module = SimpleNamespace(
                __file__=str(private_path),
                __loader__=SourceFileLoader(name, str(private_path)),
            )
            if name == "uniception.models.utils.config":
                module._HAS_FUSED_ATTN = True
                module._USE_FUSED_ATTN = 1
                module.use_fused_attn = lambda: True
            imported[name] = module
            sys.modules[name] = module
        return SimpleNamespace(torch=torch, modules=imported)

    monkeypatch.setattr(
        core, "import_mvroma_runtime_modules", fake_importer, raising=False
    )

    prepared = core.prepare_mvroma_stage_runtime(cfg, paths)
    private_roots = [Path(prepared.private_mvroma_root), Path(prepared.private_ufm_root)]
    try:
        assert prepared.stage_contract["payload"]["schema"] == (
            "mvroma-stage-contract/v2"
        )
        assert prepared.stage_contract["payload"]["implementation"][
            "loader_policy"
        ] == "a011-postmodel-v3-held-source-candidate/v2"
        assert prepared.stage_contract["payload"]["runtime"]["phase"] == (
            "post_import_pre_model"
        )
        assert prepared.stage_contract["payload"][
            "post_model_expectation_ref"
        ] == expected_post_model_ref
        monkeypatch.setitem(
            core.MVROMA_POST_MODEL_EXPECTED_REF, "sha256", "c" * 64
        )
        assert prepared.stage_contract["payload"][
            "post_model_expectation_ref"
        ] == expected_post_model_ref
        assert prepared.stage_contract["sha256"]
        assert len(prepared.jobs) == 1
        assert prepared.image_tree["file_count"] == 2
        assert all(root.exists() for root in private_roots)
        assert "src.build_model" in sys.modules
        assert str(tmp_path) not in json.dumps(prepared.stage_contract["payload"])
    finally:
        prepared.close()

    assert all(not root.exists() for root in private_roots)
    assert "src.build_model" not in sys.modules
    assert Path.cwd() == before_cwd
    assert list(sys.path) == before_path


def test_candidate_source_runner_preserves_legacy_geometry_and_bound_paths(
    tmp_path: Path,
) -> None:
    import numpy as np
    import torch
    from PIL import Image

    source = tmp_path / "held-source.jpg"
    target_a = tmp_path / "held-a.jpg"
    target_b = tmp_path / "held-b.jpg"
    Image.new("RGB", (8, 6)).save(source)
    Image.new("RGB", (10, 12)).save(target_a)
    Image.new("RGB", (20, 8)).save(target_b)

    height, width = 4, 4
    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )
    normalized = np.stack(
        [
            2.0 * xx / (width - 1) - 1.0,
            2.0 * yy / (height - 1) - 1.0,
        ],
        axis=0,
    )
    calls: list[dict[str, object]] = []

    def run_model_test(model: object, data: dict, **kwargs: object) -> dict:
        calls.append({"data": data, "kwargs": kwargs})
        references = len(data["ref_img_paths"])
        flow = torch.from_numpy(normalized).reshape(1, 1, 2, height, width)
        flow = flow.repeat(1, references, 1, 1, 1)
        certainty = torch.full((1, references, 1, height, width), 10.0)
        return {1: {"flow": flow, "certainty": certainty}}

    cfg = SimpleNamespace(
        mvroma_grid_h=height,
        mvroma_grid_w=width,
        mvroma_sample_mode="score_grid",
        mvroma_sample_grid="2x2",
        roma_cert_thresh=0.35,
        agg_maxkp=16,
        device="cpu",
    )
    runtime = SimpleNamespace(
        np=np,
        torch=torch,
        Image=Image,
        run_model_test=run_model_test,
    )
    job = {
        "source_index": 7,
        "source": "seq/source.jpg",
        "targets": ["seq/a.jpg", "seq/b.jpg"],
        "chunks": [["seq/a.jpg", "seq/b.jpg"]],
        "bound_source_path": str(source),
        "bound_target_paths": [str(target_a), str(target_b)],
    }

    prematch = [None, object()]
    runner = core.build_mvroma_candidate_source_runner(
        cfg, runtime, model=object(), prematch=prematch
    )
    result = runner(job)

    assert list(result) == ["seq/a.jpg", "seq/b.jpg"]
    assert [call["data"] for call in calls] == [{
        "query_img_path": str(source),
        "ref_img_paths": [str(target_a), str(target_b)],
    }]
    for call in calls:
        assert call["kwargs"] == {
            "coarse_res_hw": (560, 560),
            "target_res_hw": (height, width),
            "prematch_model": prematch,
            "prematch_model_name": "ufm",
            "upsample_preds": True,
            "num_cluster": 512,
            "device": "cpu",
        }

    legacy_y, legacy_x = np.meshgrid(
        np.arange(height), np.arange(width), indexing="ij"
    )
    source_expected = np.stack(
        [legacy_x / 3.0 * 7.0, legacy_y / 3.0 * 5.0], axis=-1
    ).reshape(-1, 2).astype(np.float32)
    for target, target_size in (("seq/a.jpg", (10, 12)), ("seq/b.jpg", (20, 8))):
        keypoints0, keypoints1, scores = result[target]
        target_grid_x = (normalized[0] + 1.0) / 2.0 * (width - 1)
        target_grid_y = (normalized[1] + 1.0) / 2.0 * (height - 1)
        target_expected = np.stack(
            [
                target_grid_x / (width - 1) * (target_size[0] - 1),
                target_grid_y / (height - 1) * (target_size[1] - 1),
            ],
            axis=-1,
        ).reshape(-1, 2).astype(np.float32)
        np.testing.assert_array_equal(keypoints0, source_expected)
        np.testing.assert_array_equal(keypoints1, target_expected)
        assert keypoints0.dtype == keypoints1.dtype == scores.dtype == np.float32
        assert scores.shape == (16,)
        np.testing.assert_allclose(scores, torch.sigmoid(torch.tensor(10.0)).item())


def test_candidate_source_runner_restarts_legacy_random_state_for_each_chunk(
    tmp_path: Path,
) -> None:
    import numpy as np
    import torch
    from PIL import Image

    paths = [tmp_path / name for name in ("source.jpg", "a.jpg", "b.jpg")]
    for path in paths:
        Image.new("RGB", (6, 6)).save(path)
    height = width = 5
    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )
    normalized = np.stack(
        [2.0 * xx / 4.0 - 1.0, 2.0 * yy / 4.0 - 1.0], axis=0
    )

    def run_model_test(model: object, data: dict, **kwargs: object) -> dict:
        flow = torch.from_numpy(normalized).reshape(1, 1, 2, height, width)
        certainty = torch.full((1, 1, 1, height, width), 10.0)
        return {1: {"flow": flow, "certainty": certainty}}

    cfg = SimpleNamespace(
        mvroma_grid_h=height,
        mvroma_grid_w=width,
        mvroma_sample_mode="random",
        mvroma_sample_grid="2x2",
        roma_cert_thresh=0.35,
        agg_maxkp=16,
        device="cpu",
    )
    runtime = SimpleNamespace(
        np=np,
        torch=torch,
        Image=Image,
        run_model_test=run_model_test,
    )
    job = {
        "source_index": 7,
        "source": "source.jpg",
        "targets": ["a.jpg", "b.jpg"],
        "chunks": [["a.jpg"], ["b.jpg"]],
        "bound_source_path": str(paths[0]),
        "bound_target_paths": [str(paths[1]), str(paths[2])],
    }

    result = core.build_mvroma_candidate_source_runner(
        cfg, runtime, model=object(), prematch=object()
    )(job)

    expected_indices = np.random.RandomState(7).choice(25, 16, replace=False)
    expected = np.stack(
        [xx.reshape(-1)[expected_indices] / 4.0 * 5.0,
         yy.reshape(-1)[expected_indices] / 4.0 * 5.0],
        axis=1,
    ).astype(np.float32)
    np.testing.assert_array_equal(result["a.jpg"][0], expected)
    np.testing.assert_array_equal(result["b.jpg"][0], expected)


def test_candidate_source_runner_caches_dimensions_by_logical_image_name(
    tmp_path: Path,
) -> None:
    import numpy as np
    import torch
    from PIL import Image

    image_paths = [tmp_path / f"image-{index}.jpg" for index in range(4)]
    for path in image_paths:
        Image.new("RGB", (8, 8)).save(path)
    opened: list[str] = []

    def counted_open(path: str):
        opened.append(path)
        return Image.open(path)

    height = width = 4
    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )
    normalized = np.stack(
        [2.0 * xx / 3.0 - 1.0, 2.0 * yy / 3.0 - 1.0], axis=0
    )

    def run_model_test(model: object, data: dict, **kwargs: object) -> dict:
        return {
            1: {
                "flow": torch.from_numpy(normalized).reshape(1, 1, 2, 4, 4),
                "certainty": torch.full((1, 1, 1, 4, 4), 10.0),
            }
        }

    cfg = SimpleNamespace(
        mvroma_grid_h=4,
        mvroma_grid_w=4,
        mvroma_sample_mode="score_grid",
        mvroma_sample_grid="2x2",
        roma_cert_thresh=0.35,
        agg_maxkp=16,
        device="cpu",
    )
    runtime = SimpleNamespace(
        np=np,
        torch=torch,
        Image=SimpleNamespace(open=counted_open),
        run_model_test=run_model_test,
    )
    runner = core.build_mvroma_candidate_source_runner(
        cfg, runtime, model=object(), prematch=object()
    )
    for source_index, (source_path, target_path) in enumerate(
        ((image_paths[0], image_paths[1]), (image_paths[2], image_paths[3]))
    ):
        runner(
            {
                "source_index": source_index,
                "source": f"source-{source_index}.jpg",
                "targets": ["shared-target.jpg"],
                "chunks": [["shared-target.jpg"]],
                "bound_source_path": str(source_path),
                "bound_target_paths": [str(target_path)],
            }
        )

    assert opened.count(str(image_paths[0])) == 1
    assert opened.count(str(image_paths[2])) == 1
    assert sum(opened.count(str(path)) for path in (image_paths[1], image_paths[3])) == 1
