from __future__ import annotations

import json
import multiprocessing
import os
import sys
from pathlib import Path

import pytest

import prepare_update_frames as target

np = pytest.importorskip("numpy")


def write_report_in_child(path: Path) -> None:
    try:
        target.write_json_report(path, [])
    except (OSError, ValueError):
        pass


@pytest.mark.parametrize(
    "seq",
    [
        "",
        ".",
        "..",
        "../victim",
        "nested/seq",
        r"..\victim",
        "/tmp/victim",
        "bad\x00name",
    ],
)
def test_validate_sequence_name_rejects_non_component_paths(seq: str):
    with pytest.raises(ValueError, match="invalid sequence"):
        target.validate_sequence_name(seq)


@pytest.mark.parametrize("seq", ["P2000200", "site-01.take_2", "新場域_01"])
def test_validate_sequence_name_accepts_single_components(seq: str):
    assert target.validate_sequence_name(seq) == seq


def test_parse_video_preserves_valid_absolute_video_path(tmp_path: Path):
    video = tmp_path / "source.mp4"

    assert target.parse_video([f"site-01.take_2={video}"]) == {
        "site-01.take_2": video
    }


@pytest.mark.parametrize("seq", ["", "..", "../victim", "nested/seq", r"..\victim"])
def test_parse_video_rejects_invalid_sequence_at_cli_boundary(seq: str):
    with pytest.raises(SystemExit, match="invalid sequence"):
        target.parse_video([f"{seq}=/tmp/source.mp4"])


@pytest.mark.parametrize("seq", ["../victim", "/tmp/victim", r"..\victim"])
def test_open_sequence_directory_rechecks_sequence_name(tmp_path: Path, seq: str):
    root_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(ValueError, match="invalid sequence"):
            target.open_sequence_directory_at(root_fd, seq, overwrite=False)
    finally:
        os.close(root_fd)


@pytest.mark.parametrize("overwrite", [False, True])
def test_open_sequence_directory_rejects_existing_child_symlink(
    tmp_path: Path, overwrite: bool
):
    output = tmp_path / "output"
    outside = tmp_path / "outside"
    output.mkdir()
    outside.mkdir()
    (output / "linked").symlink_to(outside, target_is_directory=True)

    root_fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(ValueError, match="unsafe (output directory|sequence output)"):
            target.open_sequence_directory_at(root_fd, "linked", overwrite=overwrite)
    finally:
        os.close(root_fd)


def test_secure_rmtree_at_stays_on_held_directory_fd(tmp_path: Path):
    original_root = tmp_path / "root"
    moved_root = tmp_path / "moved-root"
    outside = tmp_path / "outside"
    (original_root / "SAFESEQ").mkdir(parents=True)
    (original_root / "SAFESEQ" / "inside.txt").write_text("remove")
    (outside / "SAFESEQ").mkdir(parents=True)
    outside_sentinel = outside / "SAFESEQ" / "preserve.txt"
    outside_sentinel.write_text("preserve", encoding="utf-8")
    root_fd = os.open(original_root, os.O_RDONLY | os.O_DIRECTORY)
    original_root.rename(moved_root)
    original_root.symlink_to(outside, target_is_directory=True)

    try:
        target.secure_rmtree_at(root_fd, "SAFESEQ")
    finally:
        os.close(root_fd)

    assert not (moved_root / "SAFESEQ").exists()
    assert outside_sentinel.read_text(encoding="utf-8") == "preserve"


def test_secure_rmtree_at_unlinks_nested_symlink_without_following_target(
    tmp_path: Path,
):
    root = tmp_path / "root"
    sequence = root / "SAFESEQ"
    outside = tmp_path / "outside"
    (sequence / "nested").mkdir(parents=True)
    (sequence / "nested" / "inside.txt").write_text("remove")
    outside.mkdir()
    outside_sentinel = outside / "preserve.txt"
    outside_sentinel.write_text("preserve", encoding="utf-8")
    (sequence / "outside-link").symlink_to(outside, target_is_directory=True)
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)

    try:
        assert target.secure_rmtree_at(root_fd, "SAFESEQ") is True
    finally:
        os.close(root_fd)

    assert not sequence.exists()
    assert outside_sentinel.read_text(encoding="utf-8") == "preserve"


def test_parent_swap_before_delete_cannot_escape_held_root_fd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    output = tmp_path / "output"
    geometry = output / "geometry"
    moved_geometry = output / "geometry-held"
    outside = tmp_path / "outside"
    (geometry / "SAFESEQ").mkdir(parents=True)
    (geometry / "SAFESEQ" / "inside.txt").write_text("remove")
    (outside / "SAFESEQ").mkdir(parents=True)
    outside_sentinel = outside / "SAFESEQ" / "preserve.txt"
    outside_sentinel.write_text("preserve", encoding="utf-8")

    def swap_geometry_parent() -> None:
        geometry.rename(moved_geometry)
        geometry.symlink_to(outside, target_is_directory=True)

    if hasattr(target, "secure_rmtree_at"):
        original_rmtree = target.secure_rmtree_at
        swapped = False

        def swapping_rmtree(parent_fd: int, name: str) -> bool:
            nonlocal swapped
            if not swapped:
                swap_geometry_parent()
                swapped = True
            return original_rmtree(parent_fd, name)

        monkeypatch.setattr(target, "secure_rmtree_at", swapping_rmtree)
    else:
        original_resolve = target.sequence_output_dir
        resolve_calls = 0

        def swapping_resolve(root: Path, seq: str, *, anchor: Path | None = None):
            nonlocal resolve_calls
            resolved = original_resolve(root, seq, anchor=anchor)
            resolve_calls += 1
            if resolve_calls == 2:
                swap_geometry_parent()
            return resolved

        monkeypatch.setattr(target, "sequence_output_dir", swapping_resolve)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_update_frames.py",
            "--out-root",
            str(output),
            "--video",
            f"SAFESEQ={tmp_path / 'missing.mp4'}",
            "--split-classes",
            "--overwrite",
        ],
    )

    with pytest.raises(SystemExit):
        target.main()

    assert outside_sentinel.read_text(encoding="utf-8") == "preserve"


@pytest.mark.parametrize("derived_root", ["geometry", "connector"])
def test_split_classes_rejects_symlinked_derived_root_before_deletion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, derived_root: str
):
    output = tmp_path / "output"
    outside = tmp_path / "outside"
    protected = outside / "SAFESEQ"
    output.mkdir()
    protected.mkdir(parents=True)
    sentinel = protected / "do-not-delete.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    (output / derived_root).symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_update_frames.py",
            "--out-root",
            str(output),
            "--video",
            f"SAFESEQ={tmp_path / 'missing.mp4'}",
            "--split-classes",
            "--overwrite",
        ],
    )

    with pytest.raises(SystemExit):
        target.main()

    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_main_rejects_report_symlink_without_overwriting_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    output = tmp_path / "output"
    sequence = output / "SAFESEQ"
    sequence.mkdir(parents=True)
    (sequence / "existing.jpg").write_bytes(b"existing")
    protected_report = tmp_path / "protected-report.txt"
    protected_report.write_text("preserve", encoding="utf-8")
    (output / "frame_selection_report.json").symlink_to(protected_report)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_update_frames.py",
            "--out-root",
            str(output),
            "--video",
            f"SAFESEQ={tmp_path / 'unused.mp4'}",
        ],
    )

    with pytest.raises(SystemExit):
        target.main()

    assert protected_report.read_text(encoding="utf-8") == "preserve"


def test_main_writes_report_for_existing_valid_sequence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    output = tmp_path / "output"
    sequence = output / "SAFESEQ"
    sequence.mkdir(parents=True)
    (sequence / "existing.jpg").write_bytes(b"existing")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_update_frames.py",
            "--out-root",
            str(output),
            "--video",
            f"SAFESEQ={tmp_path / 'unused.mp4'}",
        ],
    )

    target.main()

    report = json.loads((output / "frame_selection_report.json").read_text())
    assert report == [{"seq": "SAFESEQ", "mode": "skip_existing", "saved": 1}]


def test_report_fifo_is_rejected_without_blocking(tmp_path: Path):
    report = tmp_path / "frame_selection_report.json"
    os.mkfifo(report)
    process = multiprocessing.get_context("fork").Process(
        target=write_report_in_child, args=(report,)
    )
    process.start()
    process.join(timeout=0.5)
    try:
        assert not process.is_alive()
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=1.0)
    assert report.is_fifo()


def test_report_serialization_failure_preserves_previous_report(tmp_path: Path):
    report = tmp_path / "frame_selection_report.json"
    report.write_text("previous-valid-report", encoding="utf-8")

    with pytest.raises(TypeError):
        target.write_json_report(report, [{"bad": object()}])

    assert report.read_text(encoding="utf-8") == "previous-valid-report"
    assert list(tmp_path.iterdir()) == [report]


def test_write_jpg_matches_opencv_encoder_bytes(tmp_path: Path):
    frame = np.arange(17 * 23 * 3, dtype=np.uint8).reshape(17, 23, 3)
    ok, expected = target.cv2.imencode(
        ".jpg", frame, [int(target.cv2.IMWRITE_JPEG_QUALITY), 95]
    )
    assert ok
    output = tmp_path / "frame.jpg"

    target.write_jpg(output, frame, 95)

    assert output.read_bytes() == expected.tobytes()


def test_write_jpg_through_held_directory_fd(tmp_path: Path):
    frame = np.zeros((16, 24, 3), dtype=np.uint8)
    directory_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        target.write_jpg(target.directory_fd_path(directory_fd) / "frame.jpg", frame, 95)
    finally:
        os.close(directory_fd)

    assert (tmp_path / "frame.jpg").stat().st_size > 0


def test_write_jpg_rejects_symlink_without_overwriting_target(tmp_path: Path):
    protected = tmp_path / "protected.jpg"
    protected.write_bytes(b"preserve")
    output = tmp_path / "frame.jpg"
    output.symlink_to(protected)
    frame = np.zeros((16, 24, 3), dtype=np.uint8)

    with pytest.raises(SystemExit, match="failed to open"):
        target.write_jpg(output, frame, 95)

    assert protected.read_bytes() == b"preserve"


def test_valid_overwrite_deletes_only_selected_sequence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    output = tmp_path / "output"
    selected = output / "SAFESEQ"
    sibling = output / "SIBLING"
    selected.mkdir(parents=True)
    sibling.mkdir()
    (selected / "old.jpg").write_bytes(b"old")
    sibling_sentinel = sibling / "preserve.txt"
    sibling_sentinel.write_text("preserve", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_update_frames.py",
            "--out-root",
            str(output),
            "--video",
            f"SAFESEQ={tmp_path / 'missing.mp4'}",
            "--overwrite",
        ],
    )

    with pytest.raises(SystemExit, match="cannot open"):
        target.main()

    assert selected.is_dir()
    assert not (selected / "old.jpg").exists()
    assert sibling_sentinel.read_text(encoding="utf-8") == "preserve"


@pytest.mark.parametrize("seq_kind", ["traversal", "absolute"])
def test_main_rejects_escape_before_overwrite_deletion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seq_kind: str
):
    output = tmp_path / "output"
    victim = tmp_path / "victim"
    victim.mkdir()
    sentinel = victim / "do-not-delete.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    seq = "../victim" if seq_kind == "traversal" else str(victim)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_update_frames.py",
            "--out-root",
            str(output),
            "--video",
            f"{seq}={tmp_path / 'missing.mp4'}",
            "--overwrite",
        ],
    )

    with pytest.raises(SystemExit):
        target.main()

    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert not output.exists()


def test_corpus_hash_hit_is_r0_noop_and_does_not_extract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    build_video = tmp_path / "build.mp4"
    test_video = tmp_path / "test.mp4"
    new_video = tmp_path / "new.mp4"
    build_video.write_bytes(b"build-bytes-aaa")
    test_video.write_bytes(b"test-bytes-bbb")
    new_video.write_bytes(b"fresh-bytes-ccc")
    build_digest = target.sha256_file(build_video)
    test_digest = target.sha256_file(test_video)
    new_digest = target.sha256_file(new_video)
    assert len({build_digest, test_digest, new_digest}) == 3

    manifest = tmp_path / "corpus_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "build": [{"seq": "S01", "rel": "build.mp4", "sha256": build_digest}],
                "test": [
                    {
                        "seq": "T01",
                        "rel": "test.mp4",
                        "source_sha256": test_digest,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    hashes = target.corpus_content_hashes(target.load_corpus_manifest(manifest))
    assert target.lookup_corpus_hit(build_video, hashes)["split"] == "build"
    assert target.lookup_corpus_hit(test_video, hashes)["split"] == "test"
    assert target.lookup_corpus_hit(new_video, hashes) is None

    output = tmp_path / "output"
    output.mkdir()
    extracted = []

    def boom(*_args, **_kwargs):
        extracted.append(True)
        raise AssertionError("corpus hit must not extract")

    monkeypatch.setattr(target, "extract_manifest", boom)
    monkeypatch.setattr(target, "extract_fps_flow", boom)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_update_frames.py",
            "--out-root",
            str(output),
            "--video",
            f"BUILDSEQ={build_video}",
            "--video",
            f"TESTSEQ={test_video}",
            "--corpus-manifest",
            str(manifest),
        ],
    )

    target.main()

    report = json.loads((output / "frame_selection_report.json").read_text())
    assert extracted == []
    assert [row["mode"] for row in report] == ["corpus_noop", "corpus_noop"]
    assert all(row["route"] == "R0" for row in report)
    assert all(row["last_seen_updated"] is False for row in report)
    assert all(row["saved"] == 0 for row in report)
    assert not (output / "BUILDSEQ").exists()
    assert not (output / "TESTSEQ").exists()


def test_omitted_corpus_manifest_does_not_invent_a_site(tmp_path: Path):
    video = tmp_path / "same.mp4"
    video.write_bytes(b"payload")
    assert target.lookup_corpus_hit(video, {}) is None

