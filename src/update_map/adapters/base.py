from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

from ..models import MatchSet, RetrievalResult


class Retriever(Protocol):
    def retrieve(self, query_id: str, query_path: Path, top_k: int) -> Sequence[RetrievalResult]: ...


class Matcher(Protocol):
    def match(
        self,
        query_id: str,
        query_path: Path,
        reference_id: str,
        reference_path: Path,
    ) -> MatchSet: ...


class AdapterError(RuntimeError):
    pass
