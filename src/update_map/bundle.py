from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io.hashing import create_map_snapshot, verify_map_snapshot


@dataclass
class BundlePointer:
    version: str
    path: str
    promoted_at: str
    previous_version: str | None = None


class CandidateBundleManager:
    """Versioned sidecar promotion and rollback without rewriting the current map."""

    def __init__(self, registry_root: str | Path, base_map_root: str | Path):
        self.registry_root = Path(registry_root)
        self.base_map_root = Path(base_map_root)
        self.versions_root = self.registry_root / "versions"
        self.pointer_path = self.registry_root / "active_bundle.json"
        self.history_path = self.registry_root / "promotion_history.jsonl"
        self.versions_root.mkdir(parents=True, exist_ok=True)

    def stage(
        self,
        candidate_dir: str | Path,
        version: str,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        source = Path(candidate_dir)
        if not source.exists() or not source.is_dir():
            raise FileNotFoundError(f"Candidate directory not found: {source}")
        forbidden_names = {
            "cameras.bin", "images.bin", "points3D.bin",
            "cameras.txt", "images.txt", "points3D.txt",
        }
        forbidden = sorted(
            str(path.relative_to(source))
            for path in source.rglob("*")
            if path.is_file() and path.name in forbidden_names
        )
        if forbidden:
            raise ValueError(
                "Production sidecar may not contain a reconstruction; keep old-view submaps quarantined: "
                + ", ".join(forbidden)
            )
        destination = self.versions_root / version
        if destination.exists():
            raise FileExistsError(f"Bundle version already exists: {version}")
        shutil.copytree(source, destination)
        manifest = destination / "manifest.json"
        payload = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
        payload.setdefault("base_map_snapshot", create_map_snapshot(self.base_map_root))
        payload.update(
            {
                "bundle_version": version,
                "staged_at": datetime.now(timezone.utc).isoformat(),
                "metadata": metadata or {},
            }
        )
        manifest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return destination

    def active(self) -> BundlePointer | None:
        if not self.pointer_path.exists():
            return None
        return BundlePointer(**json.loads(self.pointer_path.read_text(encoding="utf-8")))

    def _append_history(self, event: dict[str, Any]) -> None:
        self.registry_root.mkdir(parents=True, exist_ok=True)
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")

    def promote(
        self,
        version: str,
        regression_report: dict[str, Any],
    ) -> BundlePointer:
        if not bool(regression_report.get("passed", False)):
            raise ValueError("Cannot promote a bundle whose regression report did not pass")
        bundle = self.versions_root / version
        if not bundle.exists():
            raise FileNotFoundError(f"Unknown bundle version: {version}")
        manifest_path = bundle / "manifest.json"
        if not manifest_path.exists():
            raise ValueError("Candidate bundle has no manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        snapshot = manifest.get("base_map_snapshot")
        if snapshot:
            verification = verify_map_snapshot(self.base_map_root, snapshot)
            if not verification["ok"]:
                raise ValueError(f"Base map no longer matches candidate snapshot: {verification}")
        (bundle / "promotion_regression.json").write_text(
            json.dumps(regression_report, indent=2, sort_keys=True), encoding="utf-8"
        )
        previous = self.active()
        pointer = BundlePointer(
            version=version,
            path=str(bundle.resolve()),
            promoted_at=datetime.now(timezone.utc).isoformat(),
            previous_version=previous.version if previous else None,
        )
        self.registry_root.mkdir(parents=True, exist_ok=True)
        temporary = self.pointer_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(pointer.__dict__, indent=2), encoding="utf-8")
        temporary.replace(self.pointer_path)
        self._append_history({"event": "PROMOTE", **pointer.__dict__})
        return pointer

    def rollback(self, version: str | None = None) -> BundlePointer:
        current = self.active()
        if current is None:
            raise RuntimeError("No active bundle to roll back")
        target = version or current.previous_version
        if not target:
            raise RuntimeError("No previous bundle version is recorded")
        bundle = self.versions_root / target
        if not bundle.exists():
            raise FileNotFoundError(f"Rollback target does not exist: {target}")
        pointer = BundlePointer(
            version=target,
            path=str(bundle.resolve()),
            promoted_at=datetime.now(timezone.utc).isoformat(),
            previous_version=current.version,
        )
        temporary = self.pointer_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(pointer.__dict__, indent=2), encoding="utf-8")
        temporary.replace(self.pointer_path)
        self._append_history(
            {"event": "ROLLBACK", "from": current.version, "to": target, **pointer.__dict__}
        )
        return pointer
