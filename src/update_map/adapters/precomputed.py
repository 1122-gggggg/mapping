from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ..models import MatchSet, RetrievalResult
from .base import AdapterError


def pair_key(query_id: str, reference_id: str) -> str:
    return hashlib.sha1(f"{query_id}\n{reference_id}".encode("utf-8")).hexdigest()


class PrecomputedRetriever:
    def __init__(self, retrieval_file: str | Path):
        self.path = Path(retrieval_file)
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.entries = payload.get("queries", payload)

    def retrieve(self, query_id: str, query_path: Path, top_k: int) -> list[RetrievalResult]:
        raw = self.entries.get(query_id)
        if raw is None:
            raw = self.entries.get(str(query_path))
        if raw is None:
            raw = self.entries.get(query_path.as_posix(), [])
        output: list[RetrievalResult] = []
        for item in raw[:top_k]:
            if isinstance(item, str):
                output.append(RetrievalResult(item, 1.0))
            else:
                output.append(
                    RetrievalResult(
                        reference_id=str(item.get("reference", item.get("reference_id"))),
                        score=float(item.get("score", 1.0)),
                    )
                )
        return output


class PrecomputedMatcher:
    def __init__(self, matches_root: str | Path, index_file: str | Path | None = None):
        self.root = Path(matches_root)
        self.index: dict[str, str] = {}
        index_path = Path(index_file) if index_file else self.root / "index.json"
        if index_path.exists():
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            self.index = payload.get("pairs", payload)

    def resolve(self, query_id: str, reference_id: str) -> Path:
        key = f"{query_id}\n{reference_id}"
        mapped = self.index.get(key) or self.index.get(pair_key(query_id, reference_id))
        if mapped:
            return self.root / mapped
        return self.root / f"{pair_key(query_id, reference_id)}.npz"

    def match(
        self,
        query_id: str,
        query_path: Path,
        reference_id: str,
        reference_path: Path,
    ) -> MatchSet:
        path = self.resolve(query_id, reference_id)
        if not path.exists():
            raise AdapterError(f"Precomputed match file not found: {path}")
        payload = np.load(path, allow_pickle=False)
        required = {"query_xy", "reference_xy", "confidence"}
        missing = required - set(payload.files)
        if missing:
            raise AdapterError(f"Missing arrays in {path}: {sorted(missing)}")
        sigma = payload["sigma"] if "sigma" in payload.files else None
        return MatchSet(
            query_id=query_id,
            reference_id=reference_id,
            query_xy=payload["query_xy"],
            reference_xy=payload["reference_xy"],
            confidence=payload["confidence"],
            sigma=sigma,
            metadata={"source": str(path)},
        )
