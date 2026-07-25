from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import lmdb
import numpy as np
import pytest
from PIL import Image


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from build_file_list_lmdb import (  # noqa: E402
    build_lmdb_from_file_list,
    verify_existing_lmdb_from_file_list,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_jpeg(
    path: Path, *, size: tuple[int, int], color: tuple[int, int, int]
) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size=size, color=color).save(path, format="JPEG", quality=95)
    return path.read_bytes()


def _make_split(tmp_path: Path, names: list[str]) -> Path:
    split = tmp_path / "split"
    colors = [(240, 10, 10), (10, 240, 10), (10, 10, 240)]
    sizes = [(11, 7), (13, 9), (17, 5)]
    shapes = []
    for index, name in enumerate(names):
        width, height = sizes[index]
        _write_jpeg(split / "rgb" / name, size=(width, height), color=colors[index])
        shapes.append((height, width))
    (split / "file_list.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    np.save(split / "poses.npy", np.repeat(np.eye(4)[None], len(names), axis=0))
    np.save(split / "calibration.npy", np.repeat(np.eye(3)[None], len(names), axis=0))
    np.save(split / "image_shapes.npy", np.asarray(shapes, dtype=np.int32))
    return split


def test_build_lmdb_preserves_non_lexical_file_list_and_raw_bytes(
    tmp_path: Path,
) -> None:
    names = ["scene/z.jpg", "scene/a.jpg"]
    split = _make_split(tmp_path, names)

    report = build_lmdb_from_file_list(split)

    target = split / "rgb_lmdb"
    source_list = (split / "file_list.txt").read_bytes()
    assert (target / "file_list.txt").read_bytes() == source_list
    assert report["source_file_list_sha256"] == _sha256(source_list)
    assert report["generated_file_list_sha256"] == _sha256(source_list)
    assert report["rows"] == 2
    assert report["entries"] == 2
    assert report["bytes"] == sum(
        (split / "rgb" / name).stat().st_size for name in names
    )
    assert report["missing"] == 0
    assert report["duplicate"] == 0
    assert report["extra"] == 0
    assert report["lmdb_dir"] == str(target.resolve())
    assert report["reader_integration_tested"] is False
    assert report["training_ready"] is False
    assert [entry["path"] for entry in report["per_key"]] == names
    assert (target / "verification.json").is_file()
    assert (
        json.loads((target / "verification.json").read_text(encoding="utf-8")) == report
    )

    env = lmdb.open(str(target), readonly=True, lock=False, max_dbs=1)
    db = env.open_db(b"images", integerkey=True)
    with env.begin(db=db) as transaction:
        for index, name in enumerate(names):
            assert (
                transaction.get(index.to_bytes(4, sys.byteorder))
                == (split / "rgb" / name).read_bytes()
            )
    env.close()


def test_verify_existing_rewrites_only_receipt_not_lmdb_data(tmp_path: Path) -> None:
    split = _make_split(tmp_path, ["scene/z.jpg", "scene/a.jpg"])
    build_lmdb_from_file_list(split)
    target = split / "rgb_lmdb"
    data_mdb = target / "data.mdb"
    before = (data_mdb.stat().st_size, _sha256(data_mdb.read_bytes()))

    report = verify_existing_lmdb_from_file_list(split)

    after = (data_mdb.stat().st_size, _sha256(data_mdb.read_bytes()))
    assert after == before
    assert report["lmdb_dir"] == str(target.resolve())
    assert report["reader_integration_tested"] is False
    assert report["training_ready"] is False
    assert (
        json.loads((target / "verification.json").read_text(encoding="utf-8")) == report
    )


@pytest.mark.parametrize(
    ("names", "contents", "expected"),
    [
        (["scene/a.jpg", "scene/a.jpg"], None, "duplicate"),
        (["scene/a.jpg", ""], None, "empty"),
        (["scene/a.jpg", "../escape.jpg"], None, "unsafe"),
        (["scene/a.jpg", "scene/missing.jpg"], None, "missing"),
    ],
)
def test_build_lmdb_fails_closed_for_unsafe_or_invalid_file_lists(
    tmp_path: Path,
    names: list[str],
    contents: str | None,
    expected: str,
) -> None:
    split = _make_split(tmp_path, ["scene/a.jpg", "scene/b.jpg"])
    text = contents if contents is not None else "\n".join(names) + "\n"
    (split / "file_list.txt").write_text(text, encoding="utf-8")

    with pytest.raises((FileNotFoundError, ValueError), match=expected):
        build_lmdb_from_file_list(split)

    assert not (split / "rgb_lmdb").exists()
    assert not list(split.glob(".rgb_lmdb.tmp-*"))


def test_build_lmdb_refuses_existing_target_without_deleting_it(tmp_path: Path) -> None:
    split = _make_split(tmp_path, ["scene/a.jpg"])
    target = split / "rgb_lmdb"
    target.mkdir()
    sentinel = target / "do-not-delete"
    sentinel.write_text("preserved", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_lmdb_from_file_list(split)

    assert sentinel.read_text(encoding="utf-8") == "preserved"


def test_build_lmdb_rejects_array_rows_that_do_not_match_file_list(
    tmp_path: Path,
) -> None:
    split = _make_split(tmp_path, ["scene/a.jpg", "scene/b.jpg"])
    np.save(split / "poses.npy", np.eye(4)[None])

    with pytest.raises(ValueError, match="poses.npy.*row count"):
        build_lmdb_from_file_list(split)

    assert not (split / "rgb_lmdb").exists()


def test_build_lmdb_removes_staging_when_full_decode_verification_fails(
    tmp_path: Path,
) -> None:
    split = _make_split(tmp_path, ["scene/a.jpg"])
    (split / "rgb" / "scene/a.jpg").write_bytes(b"not-a-decodable-image")

    with pytest.raises(ValueError, match="failed to decode"):
        build_lmdb_from_file_list(split)

    assert not (split / "rgb_lmdb").exists()
    assert not list(split.glob(".rgb_lmdb.tmp-*"))
