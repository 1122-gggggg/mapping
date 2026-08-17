from __future__ import annotations

import importlib
import json
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..models import MatchSet, RetrievalResult
from .base import AdapterError


def _load_callable(spec: str) -> Callable[..., Any]:
    if ":" not in spec:
        raise AdapterError("Callable spec must use module:function")
    module_name, function_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    function = getattr(module, function_name, None)
    if not callable(function):
        raise AdapterError(f"Not a callable: {spec}")
    return function


class CallableRetriever:
    def __init__(self, callable_spec: str):
        self.function = _load_callable(callable_spec)

    def retrieve(self, query_id: str, query_path: Path, top_k: int) -> list[RetrievalResult]:
        raw = self.function(query_id=query_id, query_path=str(query_path), top_k=top_k)
        output: list[RetrievalResult] = []
        for item in raw:
            if isinstance(item, RetrievalResult):
                output.append(item)
            elif isinstance(item, str):
                output.append(RetrievalResult(item, 1.0))
            else:
                output.append(RetrievalResult(str(item[0]), float(item[1])))
        return output[:top_k]


class CallableMatcher:
    def __init__(self, callable_spec: str):
        self.function = _load_callable(callable_spec)

    def match(
        self,
        query_id: str,
        query_path: Path,
        reference_id: str,
        reference_path: Path,
    ) -> MatchSet:
        raw = self.function(
            query_id=query_id,
            query_path=str(query_path),
            reference_id=reference_id,
            reference_path=str(reference_path),
        )
        if isinstance(raw, MatchSet):
            return raw
        if not isinstance(raw, dict):
            raise AdapterError("Matcher callable must return MatchSet or dict")
        return MatchSet(
            query_id=query_id,
            reference_id=reference_id,
            query_xy=raw["query_xy"],
            reference_xy=raw["reference_xy"],
            confidence=raw["confidence"],
            sigma=raw.get("sigma"),
            metadata=raw.get("metadata", {}),
        )


class CommandRetriever:
    """Execute a command template that writes retrieval JSON to ``{output}``."""

    def __init__(self, command_template: str):
        self.command_template = command_template

    def retrieve(self, query_id: str, query_path: Path, top_k: int) -> list[RetrievalResult]:
        with tempfile.TemporaryDirectory(prefix="update_map_retrieval_") as directory:
            output = Path(directory) / "retrieval.json"
            command = self.command_template.format(
                query_id=query_id,
                query=shlex.quote(str(query_path)),
                top_k=top_k,
                output=shlex.quote(str(output)),
            )
            result = subprocess.run(command, shell=True, text=True, capture_output=True)
            if result.returncode != 0:
                raise AdapterError(
                    f"Retriever command failed ({result.returncode}): {result.stderr.strip()}"
                )
            payload = json.loads(output.read_text(encoding="utf-8"))
            raw = payload.get("results", payload)
            return [
                RetrievalResult(
                    str(item.get("reference", item.get("reference_id"))),
                    float(item.get("score", 1.0)),
                )
                for item in raw[:top_k]
            ]


class CommandMatcher:
    """Execute a command template that writes the standard match ``.npz`` to ``{output}``."""

    def __init__(self, command_template: str):
        self.command_template = command_template

    def match(
        self,
        query_id: str,
        query_path: Path,
        reference_id: str,
        reference_path: Path,
    ) -> MatchSet:
        with tempfile.TemporaryDirectory(prefix="update_map_match_") as directory:
            output = Path(directory) / "matches.npz"
            command = self.command_template.format(
                query_id=query_id,
                query=shlex.quote(str(query_path)),
                reference_id=reference_id,
                reference=shlex.quote(str(reference_path)),
                output=shlex.quote(str(output)),
            )
            result = subprocess.run(command, shell=True, text=True, capture_output=True)
            if result.returncode != 0:
                raise AdapterError(
                    f"Matcher command failed ({result.returncode}): {result.stderr.strip()}"
                )
            payload = np.load(output, allow_pickle=False)
            return MatchSet(
                query_id=query_id,
                reference_id=reference_id,
                query_xy=payload["query_xy"],
                reference_xy=payload["reference_xy"],
                confidence=payload["confidence"],
                sigma=payload["sigma"] if "sigma" in payload.files else None,
                metadata={"command": self.command_template},
            )
