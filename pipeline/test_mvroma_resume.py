from __future__ import annotations

import importlib.util
import json
import os
import re
import signal
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest


MODULE_PATH = Path(__file__).with_name("build_localizable_map_core.py")
SPEC = importlib.util.spec_from_file_location("build_localizable_map_core_o101", MODULE_PATH)
assert SPEC and SPEC.loader
core = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(core)


def test_source_plan_preserves_legacy_sort_dedup_indices_and_chunks() -> None:
    lines = [
        "z/000002.jpg b/000001.jpg",
        "a/000001.jpg a/000003.jpg",
        "a/000001.jpg a/000002.jpg",
        "a/000001.jpg a/000003.jpg",
        "same.jpg same.jpg",
        "malformed",
    ]

    jobs = core.build_mvroma_source_jobs(lines, limit_src=0, chunk_size=1)

    assert jobs == [
        {
            "source_index": 0,
            "source": "a/000001.jpg",
            "targets": ["a/000002.jpg", "a/000003.jpg"],
            "chunks": [["a/000002.jpg"], ["a/000003.jpg"]],
            "shard_name": jobs[0]["shard_name"],
        },
        {
            "source_index": 1,
            "source": "b/000001.jpg",
            "targets": ["z/000002.jpg"],
            "chunks": [["z/000002.jpg"]],
            "shard_name": jobs[1]["shard_name"],
        },
    ]
    assert re.fullmatch(r"000000-[0-9a-f]{16}\.h5", jobs[0]["shard_name"])
    assert re.fullmatch(r"000001-[0-9a-f]{16}\.h5", jobs[1]["shard_name"])


def test_source_plan_applies_limit_without_renumbering_drift() -> None:
    lines = ["c.jpg d.jpg", "a.jpg b.jpg", "e.jpg f.jpg"]

    jobs = core.build_mvroma_source_jobs(lines, limit_src=2, chunk_size=6)

    assert [(job["source_index"], job["source"]) for job in jobs] == [
        (0, "a.jpg"),
        (1, "c.jpg"),
    ]


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_source_plan_rejects_nonpositive_chunk_size(chunk_size: int) -> None:
    with pytest.raises(ValueError, match="chunk_size"):
        core.build_mvroma_source_jobs(["a.jpg b.jpg"], 0, chunk_size)


def test_source_plan_rejects_pair_name_collision() -> None:
    with pytest.raises(ValueError, match="collision"):
        core.build_mvroma_source_jobs(
            ["a/b.jpg z.jpg", "a-b.jpg z.jpg"],
            limit_src=0,
            chunk_size=6,
        )


def test_source_fingerprint_binds_plan_contract_and_image_contents() -> None:
    job = core.build_mvroma_source_jobs(
        ["a.jpg b.jpg", "a.jpg c.jpg"], limit_src=0, chunk_size=1
    )[0]
    images = {"a.jpg": "a" * 64, "b.jpg": "b" * 64, "c.jpg": "c" * 64}

    baseline = core.mvroma_source_fingerprint(job, "d" * 64, images)

    assert baseline == core.mvroma_source_fingerprint(job, "d" * 64, images)
    assert baseline != core.mvroma_source_fingerprint(job, "e" * 64, images)
    assert baseline != core.mvroma_source_fingerprint(
        {**job, "source_index": 7}, "d" * 64, images
    )
    assert baseline != core.mvroma_source_fingerprint(
        job, "d" * 64, {**images, "c.jpg": "f" * 64}
    )


def _write_raw_fixture(path: Path, dataset_order: tuple[str, ...]) -> None:
    values = {
        "keypoints0": np.arange(32, dtype=np.float32).reshape(16, 2),
        "keypoints1": np.arange(32, dtype=np.float32).reshape(16, 2) + 0.5,
        "scores": np.linspace(0.1, 0.9, 16, dtype=np.float32),
    }
    with h5py.File(path, "w") as h5:
        group = h5.create_group("seq-a.jpg/seq-b.jpg")
        for key in dataset_order:
            group.create_dataset(key, data=values[key])


def test_raw_semantic_digest_is_exact_and_layout_independent(tmp_path: Path) -> None:
    first = tmp_path / "first.h5"
    second = tmp_path / "second.h5"
    _write_raw_fixture(first, ("keypoints0", "keypoints1", "scores"))
    _write_raw_fixture(second, ("scores", "keypoints1", "keypoints0"))

    first_digest = core.mvroma_raw_semantic_digest(first)
    second_digest = core.mvroma_raw_semantic_digest(second)

    assert first_digest == "6c2199a278a3f6b5a9f1519588560df4b5242fb9125e1a7b6b901122bd7b046c"
    assert second_digest == first_digest


def _raw_values(offset: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keypoints0 = np.arange(32, dtype=np.float32).reshape(16, 2) + offset
    keypoints1 = keypoints0 + 0.5
    scores = np.linspace(0.1, 0.9, 16, dtype=np.float32)
    return keypoints0, keypoints1, scores


def _job_and_metadata() -> tuple[dict, dict]:
    job = core.build_mvroma_source_jobs(
        ["a/source.jpg b/one.jpg", "a/source.jpg c/two.jpg"],
        limit_src=0,
        chunk_size=6,
    )[0]
    images = {
        "a/source.jpg": "a" * 64,
        "b/one.jpg": "b" * 64,
        "c/two.jpg": "c" * 64,
    }
    return job, core.build_mvroma_shard_metadata(
        job, "d" * 64, images, max_correspondences=4000
    )


def test_valid_missing_and_zero_output_shards_are_complete(tmp_path: Path) -> None:
    job, metadata = _job_and_metadata()
    partial = tmp_path / job["shard_name"]

    published = core.publish_mvroma_source_shard_atomic(
        partial,
        metadata,
        {"b/one.jpg": _raw_values()},
    )

    assert published["valid"]
    assert published["processed_target_count"] == 2
    assert published["produced_groups"] == [core.pair_name("a/source.jpg", "b/one.jpg")]
    assert core.validate_mvroma_source_shard(partial, metadata)["valid"]

    zero = tmp_path / "zero.h5"
    zero_published = core.publish_mvroma_source_shard_atomic(zero, metadata, {})
    assert zero_published["valid"]
    assert zero_published["processed_target_count"] == 2
    assert zero_published["produced_groups"] == []
    assert core.validate_mvroma_source_shard(zero, metadata)["valid"]


def test_stale_source_fingerprint_rejects_complete_shard(tmp_path: Path) -> None:
    job, metadata = _job_and_metadata()
    path = tmp_path / job["shard_name"]
    core.publish_mvroma_source_shard_atomic(path, metadata, {"b/one.jpg": _raw_values()})
    stale = {**metadata, "source_fingerprint": "f" * 64}

    result = core.validate_mvroma_source_shard(path, stale)

    assert not result["valid"]
    assert "fingerprint" in result["reason"]


@pytest.mark.parametrize("mutation", ["dtype", "shape", "nonfinite", "extra_dataset"])
def test_corrupt_raw_schema_is_rejected(tmp_path: Path, mutation: str) -> None:
    job, metadata = _job_and_metadata()
    path = tmp_path / f"{mutation}.h5"
    core.publish_mvroma_source_shard_atomic(path, metadata, {"b/one.jpg": _raw_values()})
    leaf = core.pair_name("a/source.jpg", "b/one.jpg")
    with h5py.File(path, "a") as h5:
        group = h5[leaf]
        if mutation == "dtype":
            values = group["scores"][...].astype(np.float64)
            del group["scores"]
            group.create_dataset("scores", data=values)
        elif mutation == "shape":
            values = group["keypoints1"][...].reshape(8, 4)
            del group["keypoints1"]
            group.create_dataset("keypoints1", data=values)
        elif mutation == "nonfinite":
            group["scores"][0] = np.nan
        else:
            group.create_dataset("matches0", data=np.arange(16, dtype=np.int32))

    assert not core.validate_mvroma_source_shard(path, metadata)["valid"]


@pytest.mark.parametrize("location", ["root", "intermediate", "leaf", "dataset"])
def test_unexpected_attributes_are_rejected(tmp_path: Path, location: str) -> None:
    job, metadata = _job_and_metadata()
    path = tmp_path / f"attr-{location}.h5"
    core.publish_mvroma_source_shard_atomic(path, metadata, {"b/one.jpg": _raw_values()})
    leaf = core.pair_name("a/source.jpg", "b/one.jpg")
    with h5py.File(path, "a") as h5:
        objects = {
            "root": h5,
            "intermediate": h5[leaf.split("/", 1)[0]],
            "leaf": h5[leaf],
            "dataset": h5[leaf]["scores"],
        }
        objects[location].attrs["unexpected"] = "value"

    assert not core.validate_mvroma_source_shard(path, metadata)["valid"]


@pytest.mark.parametrize("link_kind", ["soft", "external"])
def test_nonhard_links_are_rejected_even_when_values_match(
    tmp_path: Path, link_kind: str
) -> None:
    job, metadata = _job_and_metadata()
    path = tmp_path / f"link-{link_kind}.h5"
    core.publish_mvroma_source_shard_atomic(path, metadata, {"b/one.jpg": _raw_values()})
    leaf = core.pair_name("a/source.jpg", "b/one.jpg")
    with h5py.File(path, "a") as h5:
        values = h5[leaf]["keypoints1"][...]
        del h5[leaf]["keypoints1"]
        if link_kind == "soft":
            h5[leaf]["keypoints1"] = h5py.SoftLink(f"/{leaf}/keypoints0")
        else:
            external = tmp_path / "external.h5"
            with h5py.File(external, "w") as other:
                other.create_dataset("keypoints1", data=values)
            h5[leaf]["keypoints1"] = h5py.ExternalLink(external.name, "/keypoints1")

    result = core.validate_mvroma_source_shard(path, metadata)
    assert not result["valid"]
    assert "link" in result["reason"]


def test_atomic_precommit_failure_preserves_previous_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job, metadata = _job_and_metadata()
    path = tmp_path / job["shard_name"]
    core.publish_mvroma_source_shard_atomic(path, metadata, {"b/one.jpg": _raw_values()})
    before = path.read_bytes()
    original_replace = core.os.replace

    def fail_replace(_source: object, _target: object) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr(core.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failure"):
        core.publish_mvroma_source_shard_atomic(
            path, metadata, {"b/one.jpg": _raw_values(100.0)}
        )
    monkeypatch.setattr(core.os, "replace", original_replace)

    assert path.read_bytes() == before
    assert not list(tmp_path.glob(f".{path.name}.tmp-*"))
    assert core.validate_mvroma_source_shard(path, metadata)["valid"]


def test_file_fsync_failure_preserves_previous_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job, metadata = _job_and_metadata()
    path = tmp_path / job["shard_name"]
    core.publish_mvroma_source_shard_atomic(path, metadata, {"b/one.jpg": _raw_values()})
    before = path.read_bytes()

    def fail_fsync(_path: Path) -> None:
        raise OSError("injected file fsync failure")

    monkeypatch.setattr(core, "_fsync_file", fail_fsync)
    with pytest.raises(OSError, match="file fsync failure"):
        core.publish_mvroma_source_shard_atomic(
            path, metadata, {"b/one.jpg": _raw_values(200.0)}
        )

    assert path.read_bytes() == before
    assert not list(tmp_path.glob(f".{path.name}.tmp-*"))


def test_directory_fsync_failure_reports_commit_uncertainty_with_complete_new_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job, old_metadata = _job_and_metadata()
    path = tmp_path / job["shard_name"]
    core.publish_mvroma_source_shard_atomic(path, old_metadata, {"b/one.jpg": _raw_values()})
    new_metadata = {**old_metadata, "stage_contract_sha256": "e" * 64}
    new_metadata["source_fingerprint"] = "f" * 64

    def fail_directory_fsync(_path: Path) -> None:
        raise OSError("injected directory fsync failure")

    monkeypatch.setattr(core, "_fsync_directory", fail_directory_fsync)
    with pytest.raises(OSError, match="directory fsync failure"):
        core.publish_mvroma_source_shard_atomic(
            path, new_metadata, {"b/one.jpg": _raw_values(300.0)}
        )

    assert core.validate_mvroma_source_shard(path, new_metadata)["valid"]
    assert not core.validate_mvroma_source_shard(path, old_metadata)["valid"]


def test_validation_uses_one_hdf5_open_for_schema_values_and_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job, metadata = _job_and_metadata()
    path = tmp_path / job["shard_name"]
    core.publish_mvroma_source_shard_atomic(path, metadata, {"b/one.jpg": _raw_values()})
    real_file = h5py.File
    opens: list[object] = []

    def observed_file(*args: object, **kwargs: object) -> h5py.File:
        opens.append(args[0])
        return real_file(*args, **kwargs)

    monkeypatch.setattr(h5py, "File", observed_file)
    result = core.validate_mvroma_source_shard(path, metadata)

    assert result["valid"]
    assert len(opens) == 1


def test_symlink_shard_is_rejected(tmp_path: Path) -> None:
    job, metadata = _job_and_metadata()
    real_path = tmp_path / "real.h5"
    link_path = tmp_path / job["shard_name"]
    core.publish_mvroma_source_shard_atomic(
        real_path, metadata, {"b/one.jpg": _raw_values()}
    )
    link_path.symlink_to(real_path.name)

    result = core.validate_mvroma_source_shard(link_path, metadata)

    assert not result["valid"]
    assert "symlink" in result["reason"] or "regular" in result["reason"]


@pytest.mark.parametrize("storage", ["external", "virtual"])
def test_non_self_contained_dataset_storage_is_rejected(
    tmp_path: Path, storage: str
) -> None:
    job, metadata = _job_and_metadata()
    path = tmp_path / f"{storage}.h5"
    core.publish_mvroma_source_shard_atomic(path, metadata, {"b/one.jpg": _raw_values()})
    leaf = core.pair_name("a/source.jpg", "b/one.jpg")
    scores = _raw_values()[2]

    if storage == "external":
        raw_path = tmp_path / "scores.raw"
        raw_path.write_bytes(scores.tobytes())
        with h5py.File(path, "a") as h5:
            del h5[leaf]["scores"]
            h5[leaf].create_dataset(
                "scores",
                shape=scores.shape,
                dtype=scores.dtype,
                external=[(raw_path.name, 0, scores.nbytes)],
            )
    else:
        backing_path = tmp_path / "scores-vds-source.h5"
        with h5py.File(backing_path, "w") as backing:
            backing.create_dataset("scores", data=scores)
        layout = h5py.VirtualLayout(shape=scores.shape, dtype=scores.dtype)
        layout[:] = h5py.VirtualSource(
            backing_path.name, "scores", shape=scores.shape
        )
        with h5py.File(path, "a") as h5:
            del h5[leaf]["scores"]
            h5[leaf].create_virtual_dataset("scores", layout)

    result = core.validate_mvroma_source_shard(path, metadata)

    assert not result["valid"]
    assert storage in result["reason"]


def test_correspondence_count_above_frozen_maximum_is_rejected(tmp_path: Path) -> None:
    job, metadata = _job_and_metadata()
    path = tmp_path / "too-many.h5"
    core.publish_mvroma_source_shard_atomic(path, metadata, {"b/one.jpg": _raw_values()})
    leaf = core.pair_name("a/source.jpg", "b/one.jpg")
    count = 4001
    keypoints0 = np.arange(count * 2, dtype=np.float32).reshape(count, 2)
    keypoints1 = keypoints0 + 0.5
    scores = np.linspace(0.1, 0.9, count, dtype=np.float32)
    with h5py.File(path, "a") as h5:
        group = h5[leaf]
        for key in ("keypoints0", "keypoints1", "scores"):
            del group[key]
        group.create_dataset("keypoints0", data=keypoints0)
        group.create_dataset("keypoints1", data=keypoints1)
        group.create_dataset("scores", data=scores)
    with h5py.File(path, "a") as h5:
        manifest = json.loads(h5.attrs[core._MVROMA_MANIFEST_ATTR])
        manifest["content_sha256"] = core.mvroma_raw_semantic_digest(path)
        h5.attrs.modify(
            core._MVROMA_MANIFEST_ATTR,
            core._canonical_json_bytes(manifest).decode("utf-8"),
        )

    result = core.validate_mvroma_source_shard(path, metadata)

    assert not result["valid"]
    assert "maximum" in result["reason"]


def test_successful_publication_reports_final_path(tmp_path: Path) -> None:
    job, metadata = _job_and_metadata()
    path = tmp_path / job["shard_name"]

    result = core.publish_mvroma_source_shard_atomic(
        path, metadata, {"b/one.jpg": _raw_values()}
    )

    assert result["path"] == str(path)
    assert Path(result["path"]).is_file()


@pytest.mark.parametrize("mutation", ["truncated", "missing_dataset", "bad_digest"])
def test_additional_corruptions_fail_closed(tmp_path: Path, mutation: str) -> None:
    job, metadata = _job_and_metadata()
    path = tmp_path / f"additional-{mutation}.h5"
    core.publish_mvroma_source_shard_atomic(path, metadata, {"b/one.jpg": _raw_values()})
    leaf = core.pair_name("a/source.jpg", "b/one.jpg")
    if mutation == "truncated":
        path.write_bytes(path.read_bytes()[:128])
    else:
        with h5py.File(path, "a") as h5:
            if mutation == "missing_dataset":
                del h5[leaf]["scores"]
            else:
                manifest = json.loads(h5.attrs[core._MVROMA_MANIFEST_ATTR])
                manifest["content_sha256"] = "0" * 64
                h5.attrs.modify(
                    core._MVROMA_MANIFEST_ATTR,
                    core._canonical_json_bytes(manifest).decode("utf-8"),
                )

    assert not core.validate_mvroma_source_shard(path, metadata)["valid"]


def test_shard_metadata_requires_explicit_runtime_maximum() -> None:
    job = core.build_mvroma_source_jobs(["a.jpg b.jpg"], 0, 6)[0]
    images = {"a.jpg": "a" * 64, "b.jpg": "b" * 64}

    with pytest.raises(TypeError):
        core.build_mvroma_shard_metadata(job, "d" * 64, images)


def test_unexpected_graph_is_rejected_before_unbounded_traversal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job, metadata = _job_and_metadata()
    path = tmp_path / "wide-graph.h5"
    core.publish_mvroma_source_shard_atomic(path, metadata, {"b/one.jpg": _raw_values()})
    with h5py.File(path, "a") as h5:
        for index in range(100):
            h5.create_group(f"unexpected-{index:03d}")
    calls = 0
    real_token = core._mvroma_h5_object_token

    def counted_token(obj: object) -> int:
        nonlocal calls
        calls += 1
        return real_token(obj)

    monkeypatch.setattr(core, "_mvroma_h5_object_token", counted_token)
    result = core.validate_mvroma_source_shard(path, metadata)

    assert not result["valid"]
    assert calls <= 7


def test_temp_inode_swap_after_validation_preserves_previous_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job, metadata = _job_and_metadata()
    path = tmp_path / job["shard_name"]
    core.publish_mvroma_source_shard_atomic(path, metadata, {"b/one.jpg": _raw_values()})
    before = path.read_bytes()
    real_validate = core.validate_mvroma_source_shard

    def swap_after_validation(candidate: Path, expected: dict) -> dict:
        result = real_validate(candidate, expected)
        displaced = candidate.with_name(candidate.name + ".validated")
        candidate.replace(displaced)
        candidate.write_bytes(b"unvalidated replacement")
        return result

    monkeypatch.setattr(core, "validate_mvroma_source_shard", swap_after_validation)
    with pytest.raises(ValueError, match="changed after validation"):
        core.publish_mvroma_source_shard_atomic(
            path, metadata, {"b/one.jpg": _raw_values(500.0)}
        )

    assert path.read_bytes() == before


def _merge_fixture(
    tmp_path: Path,
) -> tuple[list[tuple[Path, dict]], dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    jobs = core.build_mvroma_source_jobs(
        [
            "a/source.jpg b/one.jpg",
            "c/source.jpg d/three.jpg",
            "c/source.jpg e/missing.jpg",
            "f/source.jpg g/zero.jpg",
        ],
        limit_src=0,
        chunk_size=1,
    )
    images = {
        "a/source.jpg": "a" * 64,
        "b/one.jpg": "b" * 64,
        "c/source.jpg": "c" * 64,
        "d/three.jpg": "d" * 64,
        "e/missing.jpg": "e" * 64,
        "f/source.jpg": "f" * 64,
        "g/zero.jpg": "0" * 64,
    }
    by_source = {
        "a/source.jpg": {"b/one.jpg": _raw_values(10.0)},
        "c/source.jpg": {"d/three.jpg": _raw_values(20.0)},
        "f/source.jpg": {},
    }
    entries: list[tuple[Path, dict]] = []
    expected: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    shard_dir = tmp_path / "shards"
    for job in jobs:
        metadata = core.build_mvroma_shard_metadata(
            job,
            "d" * 64,
            images,
            max_correspondences=4000,
        )
        path = shard_dir / job["shard_name"]
        matches = by_source[job["source"]]
        core.publish_mvroma_source_shard_atomic(path, metadata, matches)
        entries.append((path, metadata))
        for target, values in matches.items():
            expected[core.pair_name(job["source"], target)] = values
    return entries, expected


def _raw_leaf_paths(path: Path) -> list[str]:
    leaves: list[str] = []
    with h5py.File(path, "r") as h5:
        h5.visititems(
            lambda name, obj: leaves.append(name)
            if isinstance(obj, h5py.Group)
            and set(obj.keys()) == {"keypoints0", "keypoints1", "scores"}
            else None
        )
    return sorted(leaves)


def test_merge_reads_only_explicit_planned_shards(tmp_path: Path) -> None:
    entries, expected = _merge_fixture(tmp_path)
    final = tmp_path / "matches.h5"
    unrelated = entries[0][0].parent / ".leftover.tmp.h5"
    _write_raw_fixture(unrelated, ("keypoints0", "keypoints1", "scores"))
    rogue = entries[0][0].parent / "unplanned.h5"
    _write_raw_fixture(rogue, ("keypoints0", "keypoints1", "scores"))

    result = core.merge_mvroma_shards_atomic(
        final, entries, expected_source_count=len(entries)
    )

    assert result["path"] == str(final)
    assert result["groups"] == len(expected)
    assert result["shards"] == len(entries)
    assert _raw_leaf_paths(final) == sorted(expected)
    with h5py.File(final, "r") as h5:
        assert not h5.attrs


def test_merge_equals_independent_legacy_monolithic_writer(tmp_path: Path) -> None:
    entries, expected = _merge_fixture(tmp_path)
    candidate = tmp_path / "candidate.h5"
    legacy = tmp_path / "legacy.h5"
    with h5py.File(legacy, "w") as h5:
        for leaf, values in expected.items():
            group = h5.create_group(leaf)
            for key, value in zip(("keypoints0", "keypoints1", "scores"), values):
                group.create_dataset(key, data=value)

    core.merge_mvroma_shards_atomic(
        candidate, entries, expected_source_count=len(entries)
    )

    assert core.mvroma_raw_semantic_digest(candidate) == core.mvroma_raw_semantic_digest(
        legacy
    )
    with h5py.File(candidate, "r") as actual, h5py.File(legacy, "r") as reference:
        assert _raw_leaf_paths(candidate) == _raw_leaf_paths(legacy)
        for leaf in expected:
            for key in ("keypoints0", "keypoints1", "scores"):
                assert np.array_equal(actual[leaf][key][...], reference[leaf][key][...])
                assert actual[leaf][key].dtype == reference[leaf][key].dtype


@pytest.mark.parametrize("duplicate_kind", ["path", "group"])
def test_merge_rejects_duplicate_shard_or_group_before_commit(
    tmp_path: Path, duplicate_kind: str
) -> None:
    entries, _expected = _merge_fixture(tmp_path)
    final = tmp_path / "matches.h5"
    final.write_bytes(b"previous final")
    before = final.read_bytes()
    if duplicate_kind == "path":
        bad_entries = [*entries, entries[0]]
    else:
        duplicate_path = entries[0][0].with_name("different-name.h5")
        duplicate_path.write_bytes(entries[0][0].read_bytes())
        bad_entries = [*entries, (duplicate_path, entries[0][1])]

    with pytest.raises(ValueError, match="duplicate"):
        core.merge_mvroma_shards_atomic(
            final, bad_entries, expected_source_count=len(bad_entries)
        )

    assert final.read_bytes() == before


def test_invalid_planned_shard_preserves_previous_final(tmp_path: Path) -> None:
    entries, _expected = _merge_fixture(tmp_path)
    final = tmp_path / "matches.h5"
    final.write_bytes(b"previous final")
    before = final.read_bytes()
    entries[1][0].write_bytes(b"truncated")

    with pytest.raises(ValueError, match="invalid planned shard"):
        core.merge_mvroma_shards_atomic(
            final, entries, expected_source_count=len(entries)
        )

    assert final.read_bytes() == before


def test_final_merge_replace_failure_preserves_previous_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entries, _expected = _merge_fixture(tmp_path)
    final = tmp_path / "matches.h5"
    final.write_bytes(b"previous final")
    before = final.read_bytes()

    def fail_replace(_source: object, _target: object) -> None:
        raise OSError("injected final replace failure")

    monkeypatch.setattr(core.os, "replace", fail_replace)
    with pytest.raises(OSError, match="final replace failure"):
        core.merge_mvroma_shards_atomic(
            final, entries, expected_source_count=len(entries)
        )

    assert final.read_bytes() == before
    assert not list(tmp_path.glob(f".{final.name}.tmp-*"))


def test_final_merge_directory_fsync_failure_exposes_complete_new_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entries, expected = _merge_fixture(tmp_path)
    final = tmp_path / "matches.h5"
    final.write_bytes(b"previous final")

    def fail_directory_fsync(_path: Path) -> None:
        raise OSError("injected final directory fsync failure")

    monkeypatch.setattr(core, "_fsync_directory", fail_directory_fsync)
    with pytest.raises(OSError, match="final directory fsync failure"):
        core.merge_mvroma_shards_atomic(
            final, entries, expected_source_count=len(entries)
        )

    assert _raw_leaf_paths(final) == sorted(expected)


def test_merge_validates_and_copies_each_shard_through_same_open_handle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entries, _expected = _merge_fixture(tmp_path)
    final = tmp_path / "matches.h5"
    real_consume = core._strict_mvroma_shard_inspection
    consumed: list[Path] = []

    def observed_consume(
        path: Path, metadata: dict, consumer: object = None
    ) -> dict:
        assert consumer is not None
        consumed.append(path)
        return real_consume(path, metadata, consumer)

    monkeypatch.setattr(core, "_strict_mvroma_shard_inspection", observed_consume)
    core.merge_mvroma_shards_atomic(
        final, entries, expected_source_count=len(entries)
    )

    assert consumed == [path for path, _metadata in entries]


@pytest.mark.parametrize("failure", ["file_fsync", "candidate_validation"])
def test_final_merge_precommit_failure_preserves_previous_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    entries, _expected = _merge_fixture(tmp_path)
    final = tmp_path / "matches.h5"
    final.write_bytes(b"previous final")
    before = final.read_bytes()
    if failure == "file_fsync":
        monkeypatch.setattr(
            core,
            "_fsync_file",
            lambda _path: (_ for _ in ()).throw(OSError("injected final file fsync")),
        )
        expected_error = "final file fsync"
    else:
        monkeypatch.setattr(
            core,
            "_strict_mvroma_final_inspection",
            lambda *_args: (_ for _ in ()).throw(
                ValueError("injected final candidate validation")
            ),
        )
        expected_error = "final candidate validation"

    with pytest.raises((OSError, ValueError), match=expected_error):
        core.merge_mvroma_shards_atomic(
            final, entries, expected_source_count=len(entries)
        )

    assert final.read_bytes() == before
    assert not list(tmp_path.glob(f".{final.name}.tmp-*"))


def test_distinct_valid_shards_with_colliding_group_names_are_rejected(
    tmp_path: Path,
) -> None:
    first_job = core.build_mvroma_source_jobs(["a/b.jpg z.jpg"], 0, 6)[0]
    second_job = core.build_mvroma_source_jobs(["a-b.jpg z.jpg"], 0, 6)[0]
    second_job = {**second_job, "source_index": 1, "shard_name": "000001-collision.h5"}
    images = {"a/b.jpg": "a" * 64, "a-b.jpg": "b" * 64, "z.jpg": "z" * 64}
    entries: list[tuple[Path, dict]] = []
    for job, offset in ((first_job, 1.0), (second_job, 2.0)):
        metadata = core.build_mvroma_shard_metadata(
            job,
            "d" * 64,
            images,
            max_correspondences=4000,
        )
        path = tmp_path / "shards" / job["shard_name"]
        core.publish_mvroma_source_shard_atomic(
            path, metadata, {"z.jpg": _raw_values(offset)}
        )
        entries.append((path, metadata))
    final = tmp_path / "matches.h5"
    final.write_bytes(b"previous final")
    before = final.read_bytes()

    with pytest.raises(ValueError, match="duplicate produced group"):
        core.merge_mvroma_shards_atomic(
            final, entries, expected_source_count=len(entries)
        )

    assert final.read_bytes() == before


def test_source_path_swap_during_same_handle_copy_aborts_before_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entries, _expected = _merge_fixture(tmp_path)
    final = tmp_path / "matches.h5"
    final.write_bytes(b"previous final")
    before = final.read_bytes()
    victim = entries[0][0]
    real_inspect = core._strict_mvroma_shard_inspection

    def swap_during_consume(
        path: Path, metadata: dict, consumer: object = None
    ) -> dict:
        if path != victim or consumer is None:
            return real_inspect(path, metadata, consumer)

        def wrapped_consumer(h5: h5py.File, inspection: dict) -> None:
            consumer(h5, inspection)
            displaced = path.with_name(path.name + ".displaced")
            path.replace(displaced)
            path.write_bytes(b"replacement")

        return real_inspect(path, metadata, wrapped_consumer)

    monkeypatch.setattr(core, "_strict_mvroma_shard_inspection", swap_during_consume)
    with pytest.raises(ValueError, match="pathname changed"):
        core.merge_mvroma_shards_atomic(
            final, entries, expected_source_count=len(entries)
        )

    assert final.read_bytes() == before


def test_finite_copy_mutation_is_rejected_by_source_derived_leaf_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entries, _expected = _merge_fixture(tmp_path)
    final = tmp_path / "matches.h5"
    final.write_bytes(b"previous final")
    before = final.read_bytes()
    real_validate = core._validate_mvroma_raw_arrays
    mutated = False

    def mutate_once(values: object, maximum: int | None = None) -> tuple:
        nonlocal mutated
        keypoints0, keypoints1, scores = real_validate(values, maximum)
        if not mutated:
            keypoints0 = keypoints0.copy()
            keypoints0[0, 0] += np.float32(0.25)
            mutated = True
        return keypoints0, keypoints1, scores

    monkeypatch.setattr(core, "_validate_mvroma_raw_arrays", mutate_once)
    with pytest.raises(ValueError, match="leaf digest mismatch"):
        core.merge_mvroma_shards_atomic(
            final, entries, expected_source_count=len(entries)
        )

    assert final.read_bytes() == before


def test_final_path_equal_to_planned_shard_is_rejected_without_mutation(
    tmp_path: Path,
) -> None:
    entries, _expected = _merge_fixture(tmp_path)
    final = entries[0][0]
    metadata = entries[0][1]
    before = final.read_bytes()
    assert core.validate_mvroma_source_shard(final, metadata)["valid"]

    with pytest.raises(ValueError, match="collides with planned shard"):
        core.merge_mvroma_shards_atomic(
            final, entries, expected_source_count=len(entries)
        )

    assert final.read_bytes() == before
    assert core.validate_mvroma_source_shard(final, metadata)["valid"]


@pytest.mark.parametrize("kept_indices", [(1, 2), (0, 2), (0, 1)])
def test_merge_rejects_missing_head_middle_or_tail_of_frozen_plan(
    tmp_path: Path, kept_indices: tuple[int, ...]
) -> None:
    entries, _expected = _merge_fixture(tmp_path)
    final = tmp_path / "matches.h5"
    final.write_bytes(b"previous final")
    before = final.read_bytes()
    incomplete = [entries[index] for index in kept_indices]

    with pytest.raises(ValueError, match="complete source plan"):
        core.merge_mvroma_shards_atomic(
            final, incomplete, expected_source_count=len(entries)
        )

    assert final.read_bytes() == before


@pytest.mark.parametrize("publisher", ["shard", "final"])
def test_dangling_temp_symlink_is_removed_after_precommit_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, publisher: str
) -> None:
    if publisher == "shard":
        job, metadata = _job_and_metadata()
        final = tmp_path / job["shard_name"]
        real_validate = core.validate_mvroma_source_shard

        def replace_temp_with_dangling(candidate: Path, expected: dict) -> dict:
            result = real_validate(candidate, expected)
            candidate.unlink()
            candidate.symlink_to("missing-target")
            return result

        monkeypatch.setattr(
            core, "validate_mvroma_source_shard", replace_temp_with_dangling
        )
        with pytest.raises(ValueError, match="changed after validation"):
            core.publish_mvroma_source_shard_atomic(
                final, metadata, {"b/one.jpg": _raw_values()}
            )
    else:
        entries, _expected = _merge_fixture(tmp_path)
        final = tmp_path / "matches.h5"
        real_validate_final = core._strict_mvroma_final_inspection

        def replace_final_temp_with_dangling(
            candidate: Path, expected: dict, maximum: int
        ) -> dict:
            result = real_validate_final(candidate, expected, maximum)
            candidate.unlink()
            candidate.symlink_to("missing-target")
            return result

        monkeypatch.setattr(
            core,
            "_strict_mvroma_final_inspection",
            replace_final_temp_with_dangling,
        )
        with pytest.raises(ValueError, match="changed after validation"):
            core.merge_mvroma_shards_atomic(
                final, entries, expected_source_count=len(entries)
            )

    assert not list(tmp_path.rglob(".*.tmp-*"))


@pytest.mark.parametrize("publisher", ["shard", "final"])
def test_replace_source_swap_never_reports_success_or_leaves_wrong_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, publisher: str
) -> None:
    previous = b"previous destination"
    if publisher == "shard":
        job, metadata = _job_and_metadata()
        destination = tmp_path / job["shard_name"]
        destination.write_bytes(previous)

        def publish() -> object:
            return core.publish_mvroma_source_shard_atomic(
                destination, metadata, {"b/one.jpg": _raw_values()}
            )

    else:
        entries, _expected = _merge_fixture(tmp_path / "fixture")
        destination = tmp_path / "matches.h5"
        destination.write_bytes(previous)

        def publish() -> object:
            return core.merge_mvroma_shards_atomic(
                destination, entries, expected_source_count=len(entries)
            )

    real_replace = core.os.replace

    def replace_swapped_source(source: object, target: object) -> None:
        candidate = Path(source)
        candidate.unlink()
        candidate.write_bytes(b"wrong unvalidated commit")
        real_replace(source, target)

    monkeypatch.setattr(core.os, "replace", replace_swapped_source)

    with pytest.raises(ValueError, match="committed candidate identity"):
        publish()

    assert not destination.exists() or destination.read_bytes() == previous


@pytest.mark.parametrize("publisher", ["shard", "final"])
def test_atomic_candidate_work_directory_is_private_0700(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, publisher: str
) -> None:
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o770)
    shared.chmod(0o770)
    if publisher == "shard":
        job, metadata = _job_and_metadata()
        destination = shared / job["shard_name"]

        def publish() -> object:
            return core.publish_mvroma_source_shard_atomic(
                destination, metadata, {"b/one.jpg": _raw_values()}
            )

    else:
        entries, _expected = _merge_fixture(tmp_path / "fixture")
        destination = shared / "matches.h5"

        def publish() -> object:
            return core.merge_mvroma_shards_atomic(
                destination, entries, expected_source_count=len(entries)
            )

    observed_modes: list[int] = []
    real_replace = core.os.replace

    def observe_candidate_directory(source: object, target: object) -> None:
        observed_modes.append(stat.S_IMODE(Path(source).parent.stat().st_mode))
        real_replace(source, target)

    monkeypatch.setattr(core.os, "replace", observe_candidate_directory)

    publish()

    assert observed_modes == [0o700]


def _execution_fixture(
    tmp_path: Path, source_count: int = 5
) -> tuple[list[dict], dict[str, str], Path, Path]:
    pair_lines = [f"src/{index:02d}.jpg tgt/{index:02d}.jpg" for index in range(source_count)]
    jobs = core.build_mvroma_source_jobs(pair_lines, limit_src=0, chunk_size=1)
    names = {name for line in pair_lines for name in line.split()}
    image_sha256 = {
        name: f"{position + 1:064x}" for position, name in enumerate(sorted(names))
    }
    shard_dir = tmp_path / "source-shards-v1"
    final = tmp_path / "matches.h5"
    return jobs, image_sha256, shard_dir, final


def _fake_runner(
    calls: list[int], *, fail_before_source: int | None = None, zero_source: int = 1
) -> object:
    def run(job: dict) -> dict:
        source_index = int(job["source_index"])
        if fail_before_source is not None and source_index == fail_before_source:
            raise RuntimeError(f"injected interruption before source {source_index}")
        calls.append(source_index)
        if source_index == zero_source:
            return {}
        return {
            target: _raw_values(float(source_index * 100 + target_index))
            for target_index, target in enumerate(job["targets"])
        }

    return run


def _execute_fake(
    jobs: list[dict],
    images: dict[str, str],
    shard_dir: Path,
    final: Path,
    runner_factory: object,
    *,
    mvroma_resume: bool = True,
    overwrite: bool = False,
    before_merge: object = None,
) -> dict:
    return core.execute_mvroma_resume(
        jobs,
        shard_dir,
        final,
        "d" * 64,
        images,
        max_correspondences=4000,
        mvroma_resume=mvroma_resume,
        overwrite=overwrite,
        runner_factory=runner_factory,
        before_merge=before_merge,
    )


def test_interruption_resumes_only_unpublished_sources_including_zero_output(
    tmp_path: Path,
) -> None:
    jobs, images, shard_dir, final = _execution_fixture(tmp_path)
    first_calls: list[int] = []
    with pytest.raises(RuntimeError, match="source 3"):
        _execute_fake(
            jobs,
            images,
            shard_dir,
            final,
            lambda: _fake_runner(first_calls, fail_before_source=3),
        )
    assert first_calls == [0, 1, 2]
    assert len(list(shard_dir.glob("*.h5"))) == 3

    resumed_calls: list[int] = []
    result = _execute_fake(
        jobs,
        images,
        shard_dir,
        final,
        lambda: _fake_runner(resumed_calls),
    )

    assert resumed_calls == [3, 4]
    assert result["reused_sources"] == 3
    assert result["recomputed_sources"] == 2
    assert result["model_builds"] == 1
    assert len(list(shard_dir.glob("*.h5"))) == 5


def test_full_cache_hit_rebuilds_final_without_runner_factory(tmp_path: Path) -> None:
    jobs, images, shard_dir, final = _execution_fixture(tmp_path)
    first_calls: list[int] = []
    _execute_fake(
        jobs, images, shard_dir, final, lambda: _fake_runner(first_calls)
    )
    expected_digest = core.mvroma_raw_semantic_digest(final)
    final.write_bytes(b"stale legacy final")
    before_merge_calls = 0

    def before_merge() -> None:
        nonlocal before_merge_calls
        before_merge_calls += 1

    def forbidden_factory() -> object:
        raise AssertionError("cache hit must not build a runner")

    result = _execute_fake(
        jobs,
        images,
        shard_dir,
        final,
        forbidden_factory,
        before_merge=before_merge,
    )

    assert result["reused_sources"] == len(jobs)
    assert result["recomputed_sources"] == 0
    assert result["model_builds"] == 0
    assert before_merge_calls == 1
    assert core.mvroma_raw_semantic_digest(final) == expected_digest


def test_legacy_final_without_shards_recomputes_every_source(tmp_path: Path) -> None:
    jobs, images, shard_dir, final = _execution_fixture(tmp_path)
    _write_raw_fixture(final, ("keypoints0", "keypoints1", "scores"))
    calls: list[int] = []
    factory_calls = 0

    def factory() -> object:
        nonlocal factory_calls
        factory_calls += 1
        return _fake_runner(calls)

    result = _execute_fake(jobs, images, shard_dir, final, factory)

    assert calls == list(range(len(jobs)))
    assert factory_calls == 1
    assert result["reused_sources"] == 0
    assert result["recomputed_sources"] == len(jobs)


def test_only_invalid_shard_is_recomputed(tmp_path: Path) -> None:
    jobs, images, shard_dir, final = _execution_fixture(tmp_path)
    seed_calls: list[int] = []
    _execute_fake(jobs, images, shard_dir, final, lambda: _fake_runner(seed_calls))
    invalid_index = 2
    (shard_dir / jobs[invalid_index]["shard_name"]).write_bytes(b"corrupt")
    calls: list[int] = []

    result = _execute_fake(
        jobs, images, shard_dir, final, lambda: _fake_runner(calls)
    )

    assert calls == [invalid_index]
    assert result["reused_sources"] == len(jobs) - 1
    assert result["recomputed_sources"] == 1


@pytest.mark.parametrize(
    ("mvroma_resume", "overwrite"), [(False, False), (True, True), (False, True)]
)
def test_resume_precedence_can_force_every_source(
    tmp_path: Path, mvroma_resume: bool, overwrite: bool
) -> None:
    jobs, images, shard_dir, final = _execution_fixture(tmp_path)
    seed_calls: list[int] = []
    _execute_fake(jobs, images, shard_dir, final, lambda: _fake_runner(seed_calls))
    calls: list[int] = []

    result = _execute_fake(
        jobs,
        images,
        shard_dir,
        final,
        lambda: _fake_runner(calls),
        mvroma_resume=mvroma_resume,
        overwrite=overwrite,
    )

    assert calls == list(range(len(jobs)))
    assert result["reused_sources"] == 0
    assert result["recomputed_sources"] == len(jobs)


def test_runner_factory_failure_publishes_no_new_shard(tmp_path: Path) -> None:
    jobs, images, shard_dir, final = _execution_fixture(tmp_path)

    def fail_factory() -> object:
        raise RuntimeError("post-load attestation failed")

    with pytest.raises(RuntimeError, match="attestation"):
        _execute_fake(jobs, images, shard_dir, final, fail_factory)

    assert not list(shard_dir.glob("*.h5"))
    assert not final.exists()


def test_execute_rejects_final_shard_collision_before_runner_or_mutation(
    tmp_path: Path,
) -> None:
    jobs, images, shard_dir, _unused_final = _execution_fixture(tmp_path)
    final = shard_dir / jobs[0]["shard_name"]
    shard_dir.mkdir(parents=True)
    final.write_bytes(b"preexisting bytes")
    before = final.read_bytes()
    factory_calls = 0

    def forbidden_factory() -> object:
        nonlocal factory_calls
        factory_calls += 1
        raise AssertionError("collision must fail before runner factory")

    with pytest.raises(ValueError, match="collides with planned shard"):
        _execute_fake(jobs, images, shard_dir, final, forbidden_factory)

    assert factory_calls == 0
    assert final.read_bytes() == before


def test_execute_rejects_symlinked_shard_root_before_runner(
    tmp_path: Path,
) -> None:
    jobs, images, shard_dir, final = _execution_fixture(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    shard_dir.symlink_to(outside, target_is_directory=True)
    factory_calls = 0

    def forbidden_factory() -> object:
        nonlocal factory_calls
        factory_calls += 1
        raise AssertionError("symlinked shard root reached runner")

    with pytest.raises(ValueError, match="shard root.*symlink"):
        _execute_fake(jobs, images, shard_dir, final, forbidden_factory)

    assert factory_calls == 0
    assert list(outside.iterdir()) == []


def test_execute_rejects_final_inside_shard_root_before_runner(
    tmp_path: Path,
) -> None:
    jobs, images, shard_dir, _final = _execution_fixture(
        tmp_path, source_count=1
    )
    final = shard_dir / "nested-final.h5"
    factory_calls = 0

    def forbidden_factory() -> object:
        nonlocal factory_calls
        factory_calls += 1
        raise AssertionError("nested final reached runner")

    with pytest.raises(ValueError, match="final path.*inside shard root"):
        _execute_fake(jobs, images, shard_dir, final, forbidden_factory)

    assert factory_calls == 0
    assert not final.exists()


def test_execute_holds_shard_root_inode_across_path_swap(
    tmp_path: Path,
) -> None:
    jobs, images, shard_dir, final = _execution_fixture(tmp_path, source_count=1)
    outside = tmp_path / "outside"
    backup = tmp_path / "held-original"
    outside.mkdir()

    def runner_factory() -> object:
        def run(job: dict) -> dict:
            shard_dir.rename(backup)
            shard_dir.symlink_to(outside, target_is_directory=True)
            return {
                str(job["targets"][0]): _raw_values(),
            }

        return run

    with pytest.raises(RuntimeError, match="shard root.*changed"):
        _execute_fake(jobs, images, shard_dir, final, runner_factory)

    assert list(outside.iterdir()) == []
    assert len(list(backup.glob("*.h5"))) == 1
    assert not final.exists()


def test_execute_rejects_shard_root_swap_after_before_merge_without_final(
    tmp_path: Path,
) -> None:
    jobs, images, shard_dir, final = _execution_fixture(tmp_path, source_count=1)
    outside = tmp_path / "outside"
    backup = tmp_path / "held-original"
    outside.mkdir()

    def runner_factory() -> object:
        def run(job: dict) -> dict:
            return {str(job["targets"][0]): _raw_values()}

        return run

    def before_merge() -> None:
        shard_dir.rename(backup)
        shard_dir.symlink_to(outside, target_is_directory=True)

    with pytest.raises(RuntimeError, match="shard root.*changed"):
        _execute_fake(
            jobs,
            images,
            shard_dir,
            final,
            runner_factory,
            before_merge=before_merge,
        )

    assert list(outside.iterdir()) == []
    assert len(list(backup.glob("*.h5"))) == 1
    assert not final.exists()


def test_execute_rechecks_shard_root_before_final_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jobs, images, shard_dir, final = _execution_fixture(tmp_path, source_count=1)
    outside = tmp_path / "outside"
    backup = tmp_path / "held-original"
    outside.mkdir()
    original_inspection = core._strict_mvroma_final_inspection

    def inspect_then_swap(*args: object, **kwargs: object) -> dict:
        result = original_inspection(*args, **kwargs)
        shard_dir.rename(backup)
        shard_dir.symlink_to(outside, target_is_directory=True)
        return result

    monkeypatch.setattr(core, "_strict_mvroma_final_inspection", inspect_then_swap)

    def runner_factory() -> object:
        def run(job: dict) -> dict:
            return {str(job["targets"][0]): _raw_values()}

        return run

    with pytest.raises(RuntimeError, match="shard root.*changed"):
        _execute_fake(jobs, images, shard_dir, final, runner_factory)

    assert list(outside.iterdir()) == []
    assert len(list(backup.glob("*.h5"))) == 1
    assert not final.exists()


def test_execute_removes_exact_final_after_postcommit_shard_root_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jobs, images, shard_dir, final = _execution_fixture(tmp_path, source_count=1)
    outside = tmp_path / "outside"
    backup = tmp_path / "held-original"
    outside.mkdir()
    original_committed_identity = core._mvroma_committed_identity

    def verify_then_swap(path: Path, expected: dict[str, int]) -> None:
        original_committed_identity(path, expected)
        if Path(path) == final:
            shard_dir.rename(backup)
            shard_dir.symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(core, "_mvroma_committed_identity", verify_then_swap)

    def runner_factory() -> object:
        def run(job: dict) -> dict:
            return {str(job["targets"][0]): _raw_values()}

        return run

    with pytest.raises(RuntimeError, match="shard root.*changed"):
        _execute_fake(jobs, images, shard_dir, final, runner_factory)

    assert list(outside.iterdir()) == []
    assert len(list(backup.glob("*.h5"))) == 1
    assert not final.exists()


def test_postcommit_drift_does_not_delete_attacker_final_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jobs, images, shard_dir, final = _execution_fixture(tmp_path, source_count=1)
    outside = tmp_path / "outside"
    backup = tmp_path / "held-original"
    attacker = tmp_path / "attacker-final"
    attacker_bytes = b"attacker-owned replacement"
    outside.mkdir()
    attacker.write_bytes(attacker_bytes)
    original_committed_identity = core._mvroma_committed_identity

    def verify_then_replace(path: Path, expected: dict[str, int]) -> None:
        original_committed_identity(path, expected)
        if Path(path) == final:
            attacker.replace(final)
            shard_dir.rename(backup)
            shard_dir.symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(core, "_mvroma_committed_identity", verify_then_replace)

    def runner_factory() -> object:
        def run(job: dict) -> dict:
            return {str(job["targets"][0]): _raw_values()}

        return run

    with pytest.raises(RuntimeError, match="shard root.*changed"):
        _execute_fake(jobs, images, shard_dir, final, runner_factory)

    assert final.read_bytes() == attacker_bytes
    assert list(outside.iterdir()) == []
    assert len(list(backup.glob("*.h5"))) == 1


def test_execute_removes_exact_final_after_post_fsync_shard_root_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jobs, images, shard_dir, final = _execution_fixture(tmp_path, source_count=1)
    outside = tmp_path / "outside"
    backup = tmp_path / "held-original"
    outside.mkdir()
    original_fsync_directory = core._fsync_directory
    swapped = False

    def fsync_then_swap(path: Path) -> None:
        nonlocal swapped
        original_fsync_directory(path)
        if not swapped and Path(path) == final.parent and final.exists():
            swapped = True
            shard_dir.rename(backup)
            shard_dir.symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(core, "_fsync_directory", fsync_then_swap)

    def runner_factory() -> object:
        def run(job: dict) -> dict:
            return {str(job["targets"][0]): _raw_values()}

        return run

    with pytest.raises(RuntimeError, match="shard root.*changed"):
        _execute_fake(jobs, images, shard_dir, final, runner_factory)

    assert swapped is True
    assert list(outside.iterdir()) == []
    assert len(list(backup.glob("*.h5"))) == 1
    assert not final.exists()


def test_execute_removes_exact_final_when_shard_root_swaps_at_merge_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jobs, images, shard_dir, final = _execution_fixture(tmp_path, source_count=1)
    outside = tmp_path / "outside"
    backup = tmp_path / "held-original"
    outside.mkdir()
    original_merge = core.merge_mvroma_shards_atomic

    def merge_then_swap(*args: object, **kwargs: object) -> dict:
        result = original_merge(*args, **kwargs)
        shard_dir.rename(backup)
        shard_dir.symlink_to(outside, target_is_directory=True)
        return result

    monkeypatch.setattr(core, "merge_mvroma_shards_atomic", merge_then_swap)

    def runner_factory() -> object:
        def run(job: dict) -> dict:
            return {str(job["targets"][0]): _raw_values()}

        return run

    with pytest.raises(RuntimeError, match="shard root.*changed"):
        _execute_fake(jobs, images, shard_dir, final, runner_factory)

    assert list(outside.iterdir()) == []
    assert len(list(backup.glob("*.h5"))) == 1
    assert not final.exists()


def test_execute_removes_exact_final_when_shard_root_swaps_at_context_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jobs, images, shard_dir, final = _execution_fixture(tmp_path, source_count=1)
    outside = tmp_path / "outside"
    backup = tmp_path / "held-original"
    outside.mkdir()
    original_execute = core._execute_mvroma_resume_held_root

    def execute_then_swap(*args: object, **kwargs: object) -> dict:
        result = original_execute(*args, **kwargs)
        shard_dir.rename(backup)
        shard_dir.symlink_to(outside, target_is_directory=True)
        return result

    monkeypatch.setattr(
        core, "_execute_mvroma_resume_held_root", execute_then_swap
    )

    def runner_factory() -> object:
        def run(job: dict) -> dict:
            return {str(job["targets"][0]): _raw_values()}

        return run

    with pytest.raises(RuntimeError, match="shard root.*changed"):
        _execute_fake(jobs, images, shard_dir, final, runner_factory)

    assert list(outside.iterdir()) == []
    assert len(list(backup.glob("*.h5"))) == 1
    assert not final.exists()


def test_merge_rejects_symlinked_parent_alias_to_planned_shard(tmp_path: Path) -> None:
    real_root = tmp_path / "real"
    entries, _expected = _merge_fixture(real_root)
    alias_root = tmp_path / "alias"
    alias_root.symlink_to(real_root, target_is_directory=True)
    final = alias_root / "shards" / entries[0][0].name
    metadata = entries[0][1]
    before = entries[0][0].read_bytes()

    with pytest.raises(ValueError, match="collides with planned shard"):
        core.merge_mvroma_shards_atomic(
            final, entries, expected_source_count=len(entries)
        )

    assert entries[0][0].read_bytes() == before
    assert core.validate_mvroma_source_shard(entries[0][0], metadata)["valid"]


@pytest.mark.parametrize("bad_plan", ["noncontiguous_index", "duplicate_shard_path"])
def test_execute_rejects_incomplete_or_duplicate_plan_before_runner(
    tmp_path: Path, bad_plan: str
) -> None:
    jobs, images, shard_dir, final = _execution_fixture(tmp_path, source_count=2)
    if bad_plan == "noncontiguous_index":
        jobs = [{**jobs[0], "source_index": 1}]
        expected_error = "complete source plan"
    else:
        jobs = [jobs[0], {**jobs[1], "shard_name": jobs[0]["shard_name"]}]
        expected_error = "duplicate planned shard path"
    factory_calls = 0

    def forbidden_factory() -> object:
        nonlocal factory_calls
        factory_calls += 1
        raise AssertionError("bad plan must fail before runner factory")

    with pytest.raises(ValueError, match=expected_error):
        _execute_fake(jobs, images, shard_dir, final, forbidden_factory)

    assert factory_calls == 0
    assert not list(shard_dir.glob("*.h5")) if shard_dir.exists() else True


def test_lock_path_is_fixed_beside_dense_h5(tmp_path: Path) -> None:
    dense = tmp_path / "matches-mvroma-dense.h5"

    assert core.mvroma_lock_path(dense) == tmp_path / "matches-mvroma-dense.h5.lock"


def test_flock_contention_fails_closed_and_sigkill_releases_lock(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "matches-mvroma-dense.h5.lock"
    child_code = """
import importlib.util
import sys
import time

spec = importlib.util.spec_from_file_location("o101_lock_child", sys.argv[1])
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)
with module.mvroma_stage_lock(sys.argv[2]):
    print("LOCKED", flush=True)
    time.sleep(60)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", child_code, str(MODULE_PATH), str(lock_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "LOCKED"
        with pytest.raises(RuntimeError, match="already locked"):
            with core.mvroma_stage_lock(lock_path):
                pass
        process.send_signal(signal.SIGKILL)
        assert process.wait(timeout=5) == -signal.SIGKILL
        with core.mvroma_stage_lock(lock_path):
            pass
        assert lock_path.is_file()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_fork_child_does_not_keep_stage_lock_after_owner_sigkill(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "matches-mvroma-dense.h5.lock"
    child_code = """
import importlib.util
import os
import sys
import time

spec = importlib.util.spec_from_file_location("o101_lock_fork_owner", sys.argv[1])
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)
with module.mvroma_stage_lock(sys.argv[2]):
    grandchild = os.fork()
    if grandchild == 0:
        time.sleep(60)
        os._exit(0)
    print(f"LOCKED {grandchild}", flush=True)
    time.sleep(60)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", child_code, str(MODULE_PATH), str(lock_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    grandchild_pid: int | None = None
    try:
        assert process.stdout is not None
        status = process.stdout.readline().strip().split()
        assert status[0] == "LOCKED"
        grandchild_pid = int(status[1])
        process.send_signal(signal.SIGKILL)
        assert process.wait(timeout=5) == -signal.SIGKILL
        with core.mvroma_stage_lock(lock_path):
            pass
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        if grandchild_pid is not None:
            try:
                os.kill(grandchild_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_lock_path_replacement_cannot_create_a_second_stage_lock(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "matches-mvroma-dense.h5.lock"
    original = tmp_path / "original-lock-inode"

    with pytest.raises(RuntimeError, match="lock path changed"):
        with core.mvroma_stage_lock(lock_path):
            lock_path.rename(original)
            lock_path.write_text("replacement")
            with pytest.raises(RuntimeError, match="already locked"):
                with core.mvroma_stage_lock(lock_path):
                    pass

    assert original.is_file()
    with core.mvroma_stage_lock(lock_path):
        pass


def test_guard_and_lock_replacement_cannot_create_a_second_stage_lock(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "matches-mvroma-dense.h5.lock"
    guard_path = core._mvroma_guard_lock_path(lock_path)
    original_lock = tmp_path / "original-lock-inode"
    original_guard = tmp_path / "original-guard-inode"

    with pytest.raises(RuntimeError, match="lock path changed"):
        with core.mvroma_stage_lock(lock_path):
            guard_path.rename(original_guard)
            lock_path.rename(original_lock)
            guard_path.write_text("replacement guard")
            lock_path.write_text("replacement lock")
            with pytest.raises(RuntimeError, match="already locked"):
                with core.mvroma_stage_lock(lock_path):
                    pass

    assert original_guard.is_file()
    assert original_lock.is_file()
    with core.mvroma_stage_lock(lock_path):
        pass


def _raise_lock_primary_marker(error: BaseException) -> None:
    raise error


def _lock_traceback_names(error: BaseException) -> list[str]:
    names: list[str] = []
    current = error.__traceback__
    while current is not None:
        names.append(current.tb_frame.f_code.co_name)
        current = current.tb_next
    return names


def _lock_exception_messages(error: BaseException) -> set[str]:
    messages: set[str] = set()
    pending = [error]
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        messages.add(str(current))
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
    return messages


@pytest.mark.parametrize("cleanup_site", ["unlock", "kernel_close"])
@pytest.mark.parametrize(
    ("error_type", "message"),
    [
        (SystemExit, "stage body requested exit"),
        (RuntimeError, "CUDA error: stage body failed"),
    ],
)
def test_stage_lock_cleanup_preserves_body_primary_and_closes_all_fds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cleanup_site: str,
    error_type: type[BaseException],
    message: str,
) -> None:
    import fcntl

    primary = error_type(message)
    cleanup = RuntimeError(f"{cleanup_site} cleanup failed")
    before_fds = set(core._MVROMA_ACTIVE_STAGE_LOCK_FDS)
    real_flock = fcntl.flock
    unlock_failed = False

    class FakeKernelLock:
        def close(self) -> None:
            if cleanup_site == "kernel_close":
                raise cleanup

    def flock(fd: int, operation: int) -> None:
        nonlocal unlock_failed
        if (
            cleanup_site == "unlock"
            and operation == fcntl.LOCK_UN
            and not unlock_failed
        ):
            unlock_failed = True
            raise cleanup
        real_flock(fd, operation)

    monkeypatch.setattr(
        core, "_acquire_mvroma_kernel_stage_lock", lambda _path: FakeKernelLock()
    )
    monkeypatch.setattr(fcntl, "flock", flock)

    try:
        with pytest.raises(BaseException) as caught:
            with core.mvroma_stage_lock(tmp_path / "stage.lock"):
                _raise_lock_primary_marker(primary)

        assert caught.value is primary
        assert type(caught.value) is error_type
        assert str(caught.value) == message
        assert "_raise_lock_primary_marker" in _lock_traceback_names(caught.value)
        assert caught.value.__cause__ is cleanup
        assert set(core._MVROMA_ACTIVE_STAGE_LOCK_FDS) == before_fds
    finally:
        leaked = set(core._MVROMA_ACTIVE_STAGE_LOCK_FDS) - before_fds
        core._MVROMA_ACTIVE_STAGE_LOCK_FDS.difference_update(leaked)
        for fd in leaked:
            try:
                os.close(fd)
            except OSError:
                pass


@pytest.mark.parametrize("cleanup_site", ["unlock", "kernel_close"])
def test_stage_lock_cleanup_without_body_error_remains_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cleanup_site: str,
) -> None:
    import fcntl

    cleanup = RuntimeError(f"{cleanup_site} cleanup failed")
    before_fds = set(core._MVROMA_ACTIVE_STAGE_LOCK_FDS)
    real_flock = fcntl.flock
    unlock_failed = False

    class FakeKernelLock:
        def close(self) -> None:
            if cleanup_site == "kernel_close":
                raise cleanup

    def flock(fd: int, operation: int) -> None:
        nonlocal unlock_failed
        if (
            cleanup_site == "unlock"
            and operation == fcntl.LOCK_UN
            and not unlock_failed
        ):
            unlock_failed = True
            raise cleanup
        real_flock(fd, operation)

    monkeypatch.setattr(
        core, "_acquire_mvroma_kernel_stage_lock", lambda _path: FakeKernelLock()
    )
    monkeypatch.setattr(fcntl, "flock", flock)

    try:
        with pytest.raises(RuntimeError) as caught:
            with core.mvroma_stage_lock(tmp_path / "stage.lock"):
                pass

        assert caught.value is cleanup
        assert set(core._MVROMA_ACTIVE_STAGE_LOCK_FDS) == before_fds
    finally:
        leaked = set(core._MVROMA_ACTIVE_STAGE_LOCK_FDS) - before_fds
        core._MVROMA_ACTIVE_STAGE_LOCK_FDS.difference_update(leaked)
        for fd in leaked:
            try:
                os.close(fd)
            except OSError:
                pass


def test_stage_lock_fd_close_error_is_not_registered_for_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleanup = RuntimeError("stage fd close failed")
    before_fds = set(core._MVROMA_ACTIVE_STAGE_LOCK_FDS)
    real_close = os.close
    target_fd: int | None = None
    failed = False
    reused_fds: list[int] = []

    def close(fd: int) -> None:
        nonlocal failed
        if target_fd is not None and fd == target_fd and not failed:
            failed = True
            real_close(fd)
            raise cleanup
        real_close(fd)

    monkeypatch.setattr(core.os, "close", close)
    try:
        with pytest.raises(RuntimeError) as caught:
            with core.mvroma_stage_lock(tmp_path / "stage.lock"):
                active = set(core._MVROMA_ACTIVE_STAGE_LOCK_FDS) - before_fds
                assert len(active) == 2
                target_fd = max(active)

        assert caught.value is cleanup
        assert target_fd is not None
        assert target_fd not in core._MVROMA_ACTIVE_STAGE_LOCK_FDS
        for _ in range(16):
            reused_fds.append(os.open("/dev/null", os.O_RDONLY))
            if reused_fds[-1] == target_fd:
                break
        assert target_fd in reused_fds
        core._close_inherited_mvroma_stage_locks()
        os.fstat(target_fd)
    finally:
        for fd in reused_fds:
            try:
                real_close(fd)
            except OSError:
                pass
        core._MVROMA_ACTIVE_STAGE_LOCK_FDS.difference_update(
            set(core._MVROMA_ACTIVE_STAGE_LOCK_FDS) - before_fds
        )


def test_stage_lock_kernel_close_error_is_not_registered_for_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleanup = RuntimeError("kernel stage close failed")

    class FakeKernelLock:
        def __init__(self) -> None:
            self.close_calls = 0
            self.closed = False

        def close(self) -> None:
            self.close_calls += 1
            self.closed = True
            if self.close_calls == 1:
                raise cleanup

    kernel_lock = FakeKernelLock()

    def acquire(_path: Path) -> FakeKernelLock:
        core._MVROMA_ACTIVE_KERNEL_STAGE_LOCKS.add(kernel_lock)
        return kernel_lock

    monkeypatch.setattr(core, "_acquire_mvroma_kernel_stage_lock", acquire)
    try:
        with pytest.raises(RuntimeError) as caught:
            with core.mvroma_stage_lock(tmp_path / "stage.lock"):
                pass

        assert caught.value is cleanup
        assert kernel_lock.closed is True
        assert kernel_lock.close_calls == 1
        assert kernel_lock not in core._MVROMA_ACTIVE_KERNEL_STAGE_LOCKS
        core._close_inherited_mvroma_stage_locks()
        assert kernel_lock.close_calls == 1
    finally:
        core._MVROMA_ACTIVE_KERNEL_STAGE_LOCKS.discard(kernel_lock)


def test_stage_lock_acquire_close_error_does_not_keep_fd_registered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fcntl

    primary = RuntimeError("flock acquisition failed")
    cleanup = RuntimeError("acquisition fd close failed")
    before_fds = set(core._MVROMA_ACTIVE_STAGE_LOCK_FDS)
    real_close = os.close
    target_fd: int | None = None
    close_failed = False

    def flock(fd: int, operation: int) -> None:
        nonlocal target_fd
        if operation & fcntl.LOCK_EX:
            target_fd = fd
            raise primary
        raise AssertionError(f"unexpected flock operation: {operation}")

    def close(fd: int) -> None:
        nonlocal close_failed
        if target_fd is not None and fd == target_fd and not close_failed:
            close_failed = True
            real_close(fd)
            raise cleanup
        real_close(fd)

    monkeypatch.setattr(fcntl, "flock", flock)
    monkeypatch.setattr(core.os, "close", close)
    try:
        with pytest.raises(RuntimeError) as caught:
            with core.mvroma_stage_lock(tmp_path / "stage.lock"):
                pass

        assert caught.value is primary
        assert caught.value.__cause__ is cleanup
        assert target_fd is not None
        assert target_fd not in core._MVROMA_ACTIVE_STAGE_LOCK_FDS
    finally:
        leaked = set(core._MVROMA_ACTIVE_STAGE_LOCK_FDS) - before_fds
        core._MVROMA_ACTIVE_STAGE_LOCK_FDS.difference_update(leaked)
        for fd in leaked:
            try:
                real_close(fd)
            except OSError:
                pass


def test_kernel_stage_lock_bind_close_error_does_not_register_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import errno
    import socket

    primary = OSError(errno.EIO, "kernel bind failed")
    cleanup = RuntimeError("kernel acquisition close failed")

    class FakeSocket:
        def __init__(self) -> None:
            self.close_calls = 0
            self.closed = False

        def set_inheritable(self, inheritable: bool) -> None:
            assert inheritable is False

        def bind(self, _name: bytes) -> None:
            raise primary

        def close(self) -> None:
            self.close_calls += 1
            self.closed = True
            if self.close_calls == 1:
                raise cleanup

    kernel_lock = FakeSocket()
    monkeypatch.setattr(socket, "socket", lambda *_args: kernel_lock)
    try:
        with pytest.raises(RuntimeError, match="cannot acquire") as caught:
            core._acquire_mvroma_kernel_stage_lock(tmp_path / "stage.lock")

        assert caught.value.__cause__ is primary
        assert primary.__cause__ is cleanup
        assert kernel_lock.closed is True
        assert kernel_lock.close_calls == 1
        assert kernel_lock not in core._MVROMA_ACTIVE_KERNEL_STAGE_LOCKS
        core._close_inherited_mvroma_stage_locks()
        assert kernel_lock.close_calls == 1
    finally:
        core._MVROMA_ACTIVE_KERNEL_STAGE_LOCKS.discard(kernel_lock)


def test_open_lock_inode_fstat_close_error_does_not_keep_fd_registered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = RuntimeError("lock fstat failed")
    cleanup = RuntimeError("lock fstat close failed")
    before_fds = set(core._MVROMA_ACTIVE_STAGE_LOCK_FDS)
    real_close = os.close
    target_fd: int | None = None
    close_failed = False

    def fstat(fd: int):
        nonlocal target_fd
        target_fd = fd
        raise primary

    def close(fd: int) -> None:
        nonlocal close_failed
        if target_fd is not None and fd == target_fd and not close_failed:
            close_failed = True
            real_close(fd)
            raise cleanup
        real_close(fd)

    monkeypatch.setattr(core.os, "fstat", fstat)
    monkeypatch.setattr(core.os, "close", close)
    try:
        with pytest.raises(RuntimeError) as caught:
            core._open_mvroma_lock_inode(tmp_path / "stage.lock")

        assert caught.value is primary
        assert caught.value.__cause__ is cleanup
        assert target_fd is not None
        assert target_fd not in core._MVROMA_ACTIVE_STAGE_LOCK_FDS
    finally:
        leaked = set(core._MVROMA_ACTIVE_STAGE_LOCK_FDS) - before_fds
        core._MVROMA_ACTIVE_STAGE_LOCK_FDS.difference_update(leaked)
        for fd in leaked:
            try:
                real_close(fd)
            except OSError:
                pass


def test_kernel_lock_allocation_transition_is_visible_to_fork_prepare(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import socket
    import threading

    allocation_entered = threading.Event()
    release_allocation = threading.Event()
    fork_prepare_started = threading.Event()
    fork_prepare_acquired = threading.Event()
    fork_transition_seen: list[bool] = []
    acquired: list[object] = []
    errors: list[BaseException] = []

    class FakeSocket:
        def set_inheritable(self, inheritable: bool) -> None:
            assert inheritable is False

        def bind(self, _name: bytes) -> None:
            pass

        def close(self) -> None:
            pass

    kernel_lock = FakeSocket()

    def socket_factory(*_args: object) -> FakeSocket:
        allocation_entered.set()
        if not release_allocation.wait(timeout=2.0):
            raise TimeoutError("test did not release socket allocation")
        return kernel_lock

    def acquire() -> None:
        try:
            acquired.append(
                core._acquire_mvroma_kernel_stage_lock(tmp_path / "stage.lock")
            )
        except BaseException as exc:
            errors.append(exc)

    def prepare_fork() -> None:
        fork_prepare_started.set()
        core._before_mvroma_stage_lock_fork()
        fork_transition_seen.append(
            bool(core._MVROMA_STAGE_LOCK_RESOURCE_TRANSITIONS)
        )
        fork_prepare_acquired.set()
        core._after_mvroma_stage_lock_fork_parent()

    monkeypatch.setattr(socket, "socket", socket_factory)
    acquire_thread = threading.Thread(target=acquire)
    fork_thread = threading.Thread(target=prepare_fork)
    try:
        acquire_thread.start()
        assert allocation_entered.wait(timeout=2.0)
        fork_thread.start()
        assert fork_prepare_started.wait(timeout=2.0)
        assert fork_prepare_acquired.wait(timeout=2.0)
        assert fork_transition_seen == [True]

        release_allocation.set()
        acquire_thread.join(timeout=2.0)
        fork_thread.join(timeout=2.0)
        assert acquire_thread.is_alive() is False
        assert fork_thread.is_alive() is False
        assert errors == []
        assert acquired == [kernel_lock]
        assert kernel_lock in core._MVROMA_ACTIVE_KERNEL_STAGE_LOCKS
    finally:
        release_allocation.set()
        acquire_thread.join(timeout=2.0)
        fork_thread.join(timeout=2.0)
        if kernel_lock in core._MVROMA_ACTIVE_KERNEL_STAGE_LOCKS:
            core._close_registered_mvroma_kernel_lock(kernel_lock)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork")
def test_fork_during_kernel_lock_allocation_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import socket

    real_socket = socket.socket
    child_pids: list[int] = []
    child_exit_codes: list[int] = []
    kernel_lock: object | None = None

    def fork_during_allocation(*args: object, **kwargs: object):
        allocated = real_socket(*args, **kwargs)
        child_pid = os.fork()
        if child_pid == 0:
            os._exit(91)
        child_pids.append(child_pid)
        return allocated

    monkeypatch.setattr(socket, "socket", fork_during_allocation)
    try:
        kernel_lock = core._acquire_mvroma_kernel_stage_lock(
            tmp_path / "stage.lock"
        )
    finally:
        if kernel_lock in core._MVROMA_ACTIVE_KERNEL_STAGE_LOCKS:
            core._close_registered_mvroma_kernel_lock(kernel_lock)
        for child_pid in child_pids:
            waited_pid, status = os.waitpid(child_pid, 0)
            assert waited_pid == child_pid
            child_exit_codes.append(os.waitstatus_to_exitcode(status))

    assert child_exit_codes == [70]


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork")
def test_fork_during_lock_inode_allocation_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_open = os.open
    child_pids: list[int] = []
    child_exit_codes: list[int] = []
    lock_fd: int | None = None

    def fork_during_allocation(*args: object, **kwargs: object) -> int:
        allocated = real_open(*args, **kwargs)
        child_pid = os.fork()
        if child_pid == 0:
            os._exit(91)
        child_pids.append(child_pid)
        return allocated

    monkeypatch.setattr(core.os, "open", fork_during_allocation)
    try:
        lock_fd, _identity = core._open_mvroma_lock_inode(tmp_path / "stage.lock")
    finally:
        if lock_fd in core._MVROMA_ACTIVE_STAGE_LOCK_FDS:
            core._close_registered_mvroma_stage_fd(lock_fd)
        for child_pid in child_pids:
            waited_pid, status = os.waitpid(child_pid, 0)
            assert waited_pid == child_pid
            child_exit_codes.append(os.waitstatus_to_exitcode(status))

    assert child_exit_codes == [70]


def test_inherited_lock_cleanup_attempts_every_handle_after_baseexception() -> None:
    real_close = os.close
    later_fd = real_open_fd = os.open("/dev/null", os.O_RDONLY)

    class FatalInheritedClose(BaseException):
        pass

    class BadKernelLock:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            raise FatalInheritedClose("inherited kernel close failed")

    bad_kernel_lock = BadKernelLock()
    core._MVROMA_ACTIVE_KERNEL_STAGE_LOCKS.add(bad_kernel_lock)
    core._MVROMA_ACTIVE_STAGE_LOCK_FDS.add(later_fd)
    try:
        cleanup_failed = core._close_inherited_mvroma_stage_locks()

        assert cleanup_failed is True
        assert bad_kernel_lock.close_calls == 1
        with pytest.raises(OSError):
            os.fstat(later_fd)
        assert core._MVROMA_ACTIVE_KERNEL_STAGE_LOCKS == set()
        assert core._MVROMA_ACTIVE_STAGE_LOCK_FDS == set()
    finally:
        core._MVROMA_ACTIVE_KERNEL_STAGE_LOCKS.discard(bad_kernel_lock)
        core._MVROMA_ACTIVE_STAGE_LOCK_FDS.discard(later_fd)
        try:
            real_close(real_open_fd)
        except OSError:
            pass


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork")
def test_fork_child_cleanup_error_fails_closed_before_python_resumes() -> None:
    class FatalInheritedClose(BaseException):
        pass

    class BadKernelLock:
        def close(self) -> None:
            raise FatalInheritedClose("inherited kernel close failed")

    bad_kernel_lock = BadKernelLock()
    core._MVROMA_ACTIVE_KERNEL_STAGE_LOCKS.add(bad_kernel_lock)
    try:
        child_pid = os.fork()
        if child_pid == 0:
            os._exit(91)
        waited_pid, status = os.waitpid(child_pid, 0)
        assert waited_pid == child_pid
        assert os.waitstatus_to_exitcode(status) == 70
    finally:
        core._MVROMA_ACTIVE_KERNEL_STAGE_LOCKS.discard(bad_kernel_lock)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork")
def test_fork_during_fd_close_discard_transition_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_open = os.open
    real_close = os.close
    target_fd = real_open("/dev/null", os.O_RDONLY)
    replacement_fds: list[int] = []
    child_pids: list[int] = []
    child_exit_codes: list[int] = []
    transitioned = False

    def close_and_fork_before_discard(fd: int) -> None:
        nonlocal transitioned
        if fd == target_fd and not transitioned:
            transitioned = True
            real_close(fd)
            replacement_fd = real_open("/dev/null", os.O_RDONLY)
            replacement_fds.append(replacement_fd)
            child_pid = os.fork()
            if child_pid == 0:
                os._exit(91)
            child_pids.append(child_pid)
            return
        real_close(fd)

    core._MVROMA_ACTIVE_STAGE_LOCK_FDS.add(target_fd)
    monkeypatch.setattr(core.os, "close", close_and_fork_before_discard)
    try:
        core._close_registered_mvroma_stage_fd(target_fd)
        for child_pid in child_pids:
            waited_pid, status = os.waitpid(child_pid, 0)
            assert waited_pid == child_pid
            child_exit_codes.append(os.waitstatus_to_exitcode(status))

        assert replacement_fds == [target_fd]
        assert child_exit_codes == [70]
    finally:
        core._MVROMA_ACTIVE_STAGE_LOCK_FDS.discard(target_fd)
        for replacement_fd in replacement_fds:
            try:
                real_close(replacement_fd)
            except OSError:
                pass


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork")
def test_nested_resource_transition_keeps_child_fail_closed() -> None:
    before = set(core._MVROMA_STAGE_LOCK_RESOURCE_TRANSITIONS)
    outer = core._begin_mvroma_stage_lock_resource_transition()
    inner = core._begin_mvroma_stage_lock_resource_transition()
    try:
        core._finish_mvroma_stage_lock_resource_transition(inner)
        assert outer in core._MVROMA_STAGE_LOCK_RESOURCE_TRANSITIONS

        child_pid = os.fork()
        if child_pid == 0:
            os._exit(91)
        waited_pid, status = os.waitpid(child_pid, 0)
        assert waited_pid == child_pid
        assert os.waitstatus_to_exitcode(status) == 70
    finally:
        core._finish_mvroma_stage_lock_resource_transition(inner)
        core._finish_mvroma_stage_lock_resource_transition(outer)

    assert core._MVROMA_STAGE_LOCK_RESOURCE_TRANSITIONS == before


def test_kernel_lock_allocation_baseexception_clears_transition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import socket

    before = set(core._MVROMA_STAGE_LOCK_RESOURCE_TRANSITIONS)

    def interrupt_allocation(*_args: object, **_kwargs: object) -> object:
        raise KeyboardInterrupt("socket allocation interrupted")

    monkeypatch.setattr(socket, "socket", interrupt_allocation)
    with pytest.raises(KeyboardInterrupt, match="allocation interrupted"):
        core._acquire_mvroma_kernel_stage_lock(tmp_path / "stage.lock")

    assert core._MVROMA_STAGE_LOCK_RESOURCE_TRANSITIONS == before


def test_partial_kernel_registry_add_failure_closes_once_and_clears_transition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import socket

    before = set(core._MVROMA_STAGE_LOCK_RESOURCE_TRANSITIONS)

    class PartialAddFailureSet(set[object]):
        def add(self, value: object) -> None:
            super().add(value)
            raise MemoryError("kernel registry add failed")

    class FakeSocket:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    registry = PartialAddFailureSet()
    kernel_lock = FakeSocket()
    monkeypatch.setattr(core, "_MVROMA_ACTIVE_KERNEL_STAGE_LOCKS", registry)
    monkeypatch.setattr(socket, "socket", lambda *_args: kernel_lock)

    with pytest.raises(MemoryError, match="registry add failed"):
        core._acquire_mvroma_kernel_stage_lock(tmp_path / "stage.lock")

    assert kernel_lock.close_calls == 1
    assert kernel_lock not in registry
    assert core._MVROMA_STAGE_LOCK_RESOURCE_TRANSITIONS == before


def test_partial_fd_registry_add_failure_closes_once_and_clears_transition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_open = os.open
    opened_fds: list[int] = []
    before = set(core._MVROMA_STAGE_LOCK_RESOURCE_TRANSITIONS)

    class PartialAddFailureSet(set[int]):
        def add(self, value: int) -> None:
            super().add(value)
            raise MemoryError("fd registry add failed")

    def open_and_record(*args: object, **kwargs: object) -> int:
        fd = real_open(*args, **kwargs)
        opened_fds.append(fd)
        return fd

    registry = PartialAddFailureSet()
    monkeypatch.setattr(core, "_MVROMA_ACTIVE_STAGE_LOCK_FDS", registry)
    monkeypatch.setattr(core.os, "open", open_and_record)

    with pytest.raises(MemoryError, match="registry add failed"):
        core._open_mvroma_lock_inode(tmp_path / "stage.lock")

    assert len(opened_fds) == 1
    with pytest.raises(OSError):
        os.fstat(opened_fds[0])
    assert opened_fds[0] not in registry
    assert core._MVROMA_STAGE_LOCK_RESOURCE_TRANSITIONS == before


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork")
def test_interrupted_fork_prepare_makes_child_fail_closed() -> None:
    worker = r"""
import importlib.util
import json
import os
import signal
import sys
import threading
import time

spec = importlib.util.spec_from_file_location("o101_interrupted_prepare", sys.argv[1])
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)

held = threading.Event()
release = threading.Event()

def hold_registry_guard():
    with module._MVROMA_STAGE_LOCK_REGISTRY_GUARD:
        held.set()
        release.wait(timeout=5.0)

holder = threading.Thread(target=hold_registry_guard)
holder.start()
assert held.wait(timeout=2.0)

def interrupt_prepare(_signum, _frame):
    raise RuntimeError("interrupt fork prepare")

signal.signal(signal.SIGALRM, interrupt_prepare)
signal.setitimer(signal.ITIMER_REAL, 0.05)
child_pid = None
try:
    child_pid = os.fork()
finally:
    signal.setitimer(signal.ITIMER_REAL, 0.0)
    if child_pid != 0:
        release.set()
        holder.join(timeout=2.0)

if child_pid == 0:
    os._exit(91)

deadline = time.monotonic() + 2.0
child_status = None
while time.monotonic() < deadline:
    waited_pid, status = os.waitpid(child_pid, os.WNOHANG)
    if waited_pid == child_pid:
        child_status = os.waitstatus_to_exitcode(status)
        break
    time.sleep(0.01)
if child_status is None:
    os.kill(child_pid, signal.SIGKILL)
    waited_pid, status = os.waitpid(child_pid, 0)
    assert waited_pid == child_pid
    child_status = os.waitstatus_to_exitcode(status)

print(json.dumps({"child_exit": child_status, "holder_alive": holder.is_alive()}))
"""
    result = subprocess.run(
        [sys.executable, "-c", worker, str(MODULE_PATH)],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    outcome = json.loads(result.stdout.strip().splitlines()[-1])

    assert outcome == {"child_exit": 70, "holder_alive": False}


def test_fork_prepare_acquire_then_interrupt_balances_parent_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class AcquireThenInterruptGuard:
        def __init__(self) -> None:
            self.owned = False
            self.release_calls = 0

        def acquire(self, *_args: object, **_kwargs: object) -> None:
            self.owned = True
            raise RuntimeError("signal arrived after acquire")

        def release(self) -> None:
            assert self.owned is True
            self.owned = False
            self.release_calls += 1

        def _is_owned(self) -> bool:
            return self.owned

    guard = AcquireThenInterruptGuard()
    monkeypatch.setattr(core, "_MVROMA_STAGE_LOCK_REGISTRY_GUARD", guard)

    core._before_mvroma_stage_lock_fork()
    core._after_mvroma_stage_lock_fork_parent()

    assert guard.release_calls == 1
    assert guard.owned is False


def test_nested_fork_prepare_balances_outer_acquisition() -> None:
    import threading

    acquired_while_outer_active: list[bool] = []
    acquired_after_outer_release: list[bool] = []

    def probe(target: list[bool]) -> None:
        acquired = core._MVROMA_STAGE_LOCK_REGISTRY_GUARD.acquire(timeout=0.05)
        target.append(acquired)
        if acquired:
            core._MVROMA_STAGE_LOCK_REGISTRY_GUARD.release()

    core._before_mvroma_stage_lock_fork()
    try:
        core._before_mvroma_stage_lock_fork()
        core._after_mvroma_stage_lock_fork_parent()

        blocked_probe = threading.Thread(
            target=probe, args=(acquired_while_outer_active,)
        )
        blocked_probe.start()
        blocked_probe.join(timeout=1.0)
        assert blocked_probe.is_alive() is False
        assert acquired_while_outer_active == [False]
    finally:
        core._after_mvroma_stage_lock_fork_parent()

    released_probe = threading.Thread(
        target=probe, args=(acquired_after_outer_release,)
    )
    released_probe.start()
    released_probe.join(timeout=1.0)
    assert released_probe.is_alive() is False
    assert acquired_after_outer_release == [True]


def test_stage_lock_body_primary_survives_late_fd_close_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = RuntimeError("stage body failed")
    cleanup = RuntimeError("late fd close failed")
    before_fds = set(core._MVROMA_ACTIVE_STAGE_LOCK_FDS)
    real_close = os.close
    target_fd: int | None = None
    failed = False

    def close(fd: int) -> None:
        nonlocal failed
        if target_fd is not None and fd == target_fd and not failed:
            failed = True
            real_close(fd)
            raise cleanup
        real_close(fd)

    monkeypatch.setattr(core.os, "close", close)
    with pytest.raises(RuntimeError) as caught:
        with core.mvroma_stage_lock(tmp_path / "stage.lock"):
            active = set(core._MVROMA_ACTIVE_STAGE_LOCK_FDS) - before_fds
            assert len(active) == 2
            target_fd = max(active)
            _raise_lock_primary_marker(primary)

    assert caught.value is primary
    assert caught.value.__cause__ is cleanup
    assert set(core._MVROMA_ACTIVE_STAGE_LOCK_FDS) == before_fds


def test_stage_lock_preserves_multiple_cleanup_errors_without_cycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fcntl

    primary = RuntimeError("stage body failed")
    unlock_cleanup = RuntimeError("unlock failed")
    fd_cleanup = RuntimeError("fd close failed")
    kernel_cleanup = RuntimeError("kernel close failed")
    before_fds = set(core._MVROMA_ACTIVE_STAGE_LOCK_FDS)
    real_flock = fcntl.flock
    real_close = os.close
    target_fd: int | None = None
    unlock_failed = False
    close_failed = False

    class FakeKernelLock:
        def close(self) -> None:
            raise kernel_cleanup

    kernel_lock = FakeKernelLock()

    def acquire(_path: Path) -> FakeKernelLock:
        core._MVROMA_ACTIVE_KERNEL_STAGE_LOCKS.add(kernel_lock)
        return kernel_lock

    def flock(fd: int, operation: int) -> None:
        nonlocal unlock_failed
        if operation == fcntl.LOCK_UN and not unlock_failed:
            unlock_failed = True
            raise unlock_cleanup
        real_flock(fd, operation)

    def close(fd: int) -> None:
        nonlocal close_failed
        if target_fd is not None and fd == target_fd and not close_failed:
            close_failed = True
            real_close(fd)
            raise fd_cleanup
        real_close(fd)

    monkeypatch.setattr(core, "_acquire_mvroma_kernel_stage_lock", acquire)
    monkeypatch.setattr(fcntl, "flock", flock)
    monkeypatch.setattr(core.os, "close", close)
    try:
        with pytest.raises(RuntimeError) as caught:
            with core.mvroma_stage_lock(tmp_path / "stage.lock"):
                active = set(core._MVROMA_ACTIVE_STAGE_LOCK_FDS) - before_fds
                target_fd = max(active)
                _raise_lock_primary_marker(primary)

        assert caught.value is primary
        assert _lock_exception_messages(caught.value) == {
            "stage body failed",
            "unlock failed",
            "fd close failed",
            "kernel close failed",
        }
        assert set(core._MVROMA_ACTIVE_STAGE_LOCK_FDS) == before_fds
        assert kernel_lock not in core._MVROMA_ACTIVE_KERNEL_STAGE_LOCKS
    finally:
        core._MVROMA_ACTIVE_KERNEL_STAGE_LOCKS.discard(kernel_lock)


def test_open_nonregular_lock_inode_late_close_error_keeps_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleanup = RuntimeError("nonregular close failed")
    before_fds = set(core._MVROMA_ACTIVE_STAGE_LOCK_FDS)
    real_fstat = os.fstat
    real_close = os.close
    target_fd: int | None = None
    close_failed = False

    def fstat(fd: int):
        nonlocal target_fd
        target_fd = fd
        values = list(real_fstat(fd))
        values[0] = stat.S_IFIFO | 0o600
        return os.stat_result(values)

    def close(fd: int) -> None:
        nonlocal close_failed
        if target_fd is not None and fd == target_fd and not close_failed:
            close_failed = True
            real_close(fd)
            raise cleanup
        real_close(fd)

    monkeypatch.setattr(core.os, "fstat", fstat)
    monkeypatch.setattr(core.os, "close", close)
    with pytest.raises(RuntimeError, match="not a regular file") as caught:
        core._open_mvroma_lock_inode(tmp_path / "stage.lock")

    assert caught.value.__cause__ is cleanup
    assert target_fd is not None
    assert target_fd not in core._MVROMA_ACTIVE_STAGE_LOCK_FDS
    assert set(core._MVROMA_ACTIVE_STAGE_LOCK_FDS) == before_fds


def test_old_config_defaults_mvroma_resume_true() -> None:
    assert core.apply_config_defaults({})["mvroma_resume"] is True


@pytest.mark.parametrize(
    ("saved_value", "cli_flag", "expected"),
    [
        (None, None, True),
        (True, "--no-mvroma-resume", False),
        (False, None, False),
        (False, "--mvroma-resume", True),
    ],
)
def test_config_reentry_preserves_or_explicitly_overrides_mvroma_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    saved_value: bool | None,
    cli_flag: str | None,
    expected: bool,
) -> None:
    config = tmp_path / "build_config.json"
    data: dict[str, object] = {
        "work_dir": str(tmp_path),
        "stages": "mvroma",
        "resume": False,
    }
    if saved_value is not None:
        data["mvroma_resume"] = saved_value
    config.write_text(json.dumps(data), encoding="utf-8")
    argv = [str(MODULE_PATH), "--config", str(config)]
    if cli_flag:
        argv.append(cli_flag)
    monkeypatch.setattr(sys, "argv", argv)

    parsed = core.parse_args()

    assert parsed.mvroma_resume is expected
    assert parsed.resume is False


def test_new_cli_explicit_no_mvroma_resume_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [str(MODULE_PATH), "--work-dir", str(tmp_path), "--no-mvroma-resume"],
    )

    parsed = core.parse_args()

    assert parsed.mvroma_resume is False


def test_stage_aggregate_holds_same_lock_for_entire_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = SimpleNamespace(work_dir=str(tmp_path))
    dense = core.cfg_paths(cfg).dense_matches
    lock = core.mvroma_lock_path(dense)
    calls: list[str] = []

    def fake_unlocked(_cfg: SimpleNamespace) -> None:
        calls.append("body")
        with pytest.raises(RuntimeError, match="already locked"):
            with core.mvroma_stage_lock(lock):
                pass

    monkeypatch.setattr(core, "_stage_aggregate_unlocked", fake_unlocked, raising=False)
    with core.mvroma_stage_lock(lock):
        with pytest.raises(RuntimeError, match="already locked"):
            core.stage_aggregate(cfg)
    assert calls == []

    core.stage_aggregate(cfg)
    assert calls == ["body"]
