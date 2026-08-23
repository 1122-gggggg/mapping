from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .adapters import (
    CallableMatcher,
    CallableRetriever,
    CommandMatcher,
    CommandRetriever,
    PrecomputedMatcher,
    PrecomputedRetriever,
)
from .bundle import CandidateBundleManager
from .config import UpdateMapConfig, load_config
from .io.hashing import create_map_snapshot, load_snapshot, save_snapshot, verify_map_snapshot
from .io.manifests import build_image_manifest, write_manifest
from .map_adapters import load_map
from .models import to_jsonable
from .pipeline import HistoricalAugmentationPipeline
from .quality import enrich_records_with_quality, select_historical_keyframes
from .reporting import (
    generate_data_audit_markdown,
    pose_estimate_row,
    write_json,
    write_records_table,
)
from .splits import audit_dataset_splits
from .states import ImageSource
from .synthetic import run_synthetic_demo

app = typer.Typer(
    name="update-map",
    help="Historical-view augmentation for a frozen current map.",
    no_args_is_help=True,
)
console = Console()


def _resolve_config(config_path: Path | None) -> UpdateMapConfig:
    config = load_config(config_path)
    errors = config.validate(require_paths=False)
    if errors:
        raise typer.BadParameter("; ".join(errors))
    return config


def _reference_paths(base_map, image_root: str | Path) -> dict[str, Path]:
    root = Path(image_root)
    if not root.exists():
        raise FileNotFoundError(f"Current image root not found: {root}")
    by_name: dict[str, Path] = {}
    by_basename: dict[str, list[Path]] = {}
    for path in root.rglob("*"):
        if path.is_file():
            by_name[path.relative_to(root).as_posix()] = path
            by_basename.setdefault(path.name, []).append(path)
    output: dict[str, Path] = {}
    missing: list[str] = []
    for image in base_map.images.values():
        candidate = by_name.get(Path(image.name).as_posix())
        if candidate is None:
            matches = by_basename.get(Path(image.name).name, [])
            if len(matches) == 1:
                candidate = matches[0]
        if candidate is None:
            missing.append(image.name)
        else:
            output[image.name] = candidate
            output[str(image.image_id)] = candidate
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"Could not resolve {len(missing)} current map images under {root}; examples: {preview}"
        )
    return output


def _build_adapters(config: UpdateMapConfig):
    adapter = config.adapters
    if adapter.retrieval_type == "precomputed":
        retrieval_file = adapter.retrieval_file or str(
            Path(config.paths.precomputed_localizer) / "retrieval.json"
        )
        retriever = PrecomputedRetriever(retrieval_file)
    elif adapter.retrieval_type == "callable":
        retriever = CallableRetriever(adapter.python_retriever)
    elif adapter.retrieval_type == "command":
        retriever = CommandRetriever(adapter.command_retriever)
    else:
        raise ValueError(f"Unsupported retrieval adapter: {adapter.retrieval_type}")
    if adapter.matcher_type == "precomputed":
        matches_root = adapter.matches_root or str(
            Path(config.paths.precomputed_localizer) / "matches"
        )
        matcher = PrecomputedMatcher(matches_root)
    elif adapter.matcher_type == "callable":
        matcher = CallableMatcher(adapter.python_matcher)
    elif adapter.matcher_type == "command":
        matcher = CommandMatcher(adapter.command_matcher)
    else:
        raise ValueError(f"Unsupported matcher adapter: {adapter.matcher_type}")
    return retriever, matcher


def _run_direct_ingestion(config: UpdateMapConfig, output: Path) -> dict[str, object]:
    required_errors = config.validate(require_paths=True)
    if required_errors:
        raise ValueError("; ".join(required_errors))
    if not config.paths.current_images:
        raise ValueError("paths.current_images is required to resolve map reference images")
    base_map = load_map(config.paths.base_map, config.adapters.map_loader)
    snapshot = create_map_snapshot(base_map.root or config.paths.base_map)
    save_snapshot(snapshot, output / "base_map_hashes.json")
    records = build_image_manifest(
        config.paths.historical_data,
        config.paths.current_validation or None,
        config.paths.current_images or None,
    )
    records = enrich_records_with_quality(records, config.quality)
    split_audit = audit_dataset_splits(records, check_content_hashes=False)
    write_json(split_audit, output / "manifests" / "split_audit.json")
    historical = [item for item in records if item.source == ImageSource.HISTORICAL_UPDATE]
    selected, rejected = select_historical_keyframes(historical, config.quality)
    write_manifest(records, output / "manifests" / "all_images.csv")
    write_manifest(selected, output / "manifests" / "historical_keyframes.csv")
    write_manifest(rejected, output / "manifests" / "historical_rejected.csv")
    retriever, matcher = _build_adapters(config)
    pipeline = HistoricalAugmentationPipeline(
        base_map,
        config,
        retriever,
        matcher,
        _reference_paths(base_map, config.paths.current_images),
    )
    if not base_map.cameras:
        raise ValueError("Base map has no camera model")
    query_camera = base_map.cameras[sorted(base_map.cameras)[0]]
    rows = []
    result_dir = output / "direct_registration"
    result_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for record in selected:
        try:
            result = pipeline.direct_register(record, query_camera)
            pipeline.export_direct_result(result, result_dir / f"{record.image_id.replace(':', '_')}.json")
            row = pose_estimate_row(result.pose_estimate)
            row.update(
                {
                    "session_id": record.session_id,
                    "path": str(record.path),
                    "failure_class": result.failure_class.value if result.failure_class else None,
                    "accepted_associations": len(result.accepted_associations),
                }
            )
        except Exception as exc:
            row = {
                "query_id": record.image_id,
                "session_id": record.session_id,
                "path": str(record.path),
                "status": "PIPELINE_ERROR",
                "error": str(exc),
            }
        counts[row["status"]] = counts.get(row["status"], 0) + 1
        rows.append(row)
    write_records_table(rows, output / "manifests" / "direct_registration_results.parquet")
    immutable = pipeline.verify_core_immutable()
    summary = {
        "base_map": {
            "cameras": len(base_map.cameras),
            "images": len(base_map.images),
            "points3d": len(base_map.points3d),
            "real_points": len(base_map.real_point_ids()),
            "virtual_points": len(base_map.virtual_point_ids()),
            "source_format": base_map.source_format,
            "adapter": config.adapters.map_loader,
        },
        "localizer": config.adapters.localizer,
        "historical_images": len(historical),
        "selected_keyframes": len(selected),
        "rejected_keyframes": len(rejected),
        "registration_counts": counts,
        "core_immutable": immutable,
        "validation_grade": split_audit.validation_grade.value,
        "critical_dataset_leakage": split_audit.critical_leakage,
    }
    write_json(summary, output / "direct_registration_summary.json")
    return summary


@app.command("validate-config")
def validate_config_command(
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config"),
    require_paths: bool = typer.Option(False, help="Also require data paths to exist"),
) -> None:
    value = load_config(config)
    errors = value.validate(require_paths=require_paths)
    if errors:
        for error in errors:
            console.print(f"[red]ERROR[/red] {error}")
        raise typer.Exit(code=2)
    console.print("[green]Configuration is valid.[/green]")


@app.command("inspect-map")
def inspect_map(
    base_map: Path = typer.Argument(..., exists=True),
    map_adapter: str = typer.Option(
        "colmap",
        "--map-adapter",
        help="Built-in map adapter or package.module:loader",
    ),
) -> None:
    model = load_map(base_map, map_adapter)
    table = Table(title="Current map")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Format", model.source_format)
    table.add_row("Cameras", str(len(model.cameras)))
    table.add_row("Images", str(len(model.images)))
    table.add_row("Points3D", str(len(model.points3d)))
    table.add_row("Real/verified points", str(len(model.real_point_ids())))
    table.add_row("Virtual BA-only points", str(len(model.virtual_point_ids())))
    table.add_row("Model root", str(model.root))
    console.print(table)


@app.command("hash-map")
def hash_map(
    base_map: Path = typer.Argument(..., exists=True, file_okay=False),
    output: Path = typer.Option(Path("base_map_hashes.json"), help="Snapshot JSON"),
) -> None:
    snapshot = create_map_snapshot(base_map)
    save_snapshot(snapshot, output)
    console.print(f"Saved snapshot: {output}")
    console.print(f"Aggregate SHA-256: {snapshot['aggregate_sha256']}")


@app.command("verify-map")
def verify_map(
    base_map: Path = typer.Argument(..., exists=True, file_okay=False),
    snapshot: Path = typer.Argument(..., exists=True, dir_okay=False),
) -> None:
    report = verify_map_snapshot(base_map, load_snapshot(snapshot))
    console.print_json(json.dumps(report))
    if not report["ok"]:
        raise typer.Exit(code=3)


@app.command("stage-bundle")
def stage_bundle(
    candidate: Path = typer.Argument(..., exists=True, file_okay=False),
    version: str = typer.Option(..., help="Immutable candidate version name."),
    registry: Path = typer.Option(Path("bundle_registry")),
    base_map: Path = typer.Option(..., exists=True, file_okay=False),
) -> None:
    manager = CandidateBundleManager(registry, base_map)
    staged = manager.stage(candidate, version)
    console.print(f"Staged candidate bundle: {staged}")


@app.command("promote-bundle")
def promote_bundle(
    version: str = typer.Option(...),
    regression_report: Path = typer.Option(..., exists=True, dir_okay=False),
    registry: Path = typer.Option(Path("bundle_registry")),
    base_map: Path = typer.Option(..., exists=True, file_okay=False),
) -> None:
    report = json.loads(regression_report.read_text(encoding="utf-8"))
    pointer = CandidateBundleManager(registry, base_map).promote(version, report)
    console.print_json(json.dumps(to_jsonable(pointer)))


@app.command("rollback-bundle")
def rollback_bundle(
    version: Optional[str] = typer.Option(None, help="Target version; defaults to previous."),
    registry: Path = typer.Option(Path("bundle_registry")),
    base_map: Path = typer.Option(..., exists=True, file_okay=False),
) -> None:
    pointer = CandidateBundleManager(registry, base_map).rollback(version)
    console.print_json(json.dumps(to_jsonable(pointer)))


@app.command("active-bundle")
def active_bundle(
    registry: Path = typer.Option(Path("bundle_registry")),
    base_map: Path = typer.Option(..., exists=True, file_okay=False),
) -> None:
    pointer = CandidateBundleManager(registry, base_map).active()
    console.print_json(json.dumps(to_jsonable(pointer)))


@app.command("audit")
def audit(
    config: Optional[Path] = typer.Option(None, exists=True, dir_okay=False),
    base_map: Optional[Path] = typer.Option(None, exists=True),
    map_adapter: Optional[str] = typer.Option(
        None,
        "--map-adapter",
        help="Built-in map adapter or package.module:loader",
    ),
    historical: Optional[Path] = typer.Option(None, exists=True, file_okay=False),
    validation: Optional[Path] = typer.Option(None, file_okay=False),
    current_images: Optional[Path] = typer.Option(None, file_okay=False),
    output: Path = typer.Option(Path("runs/audit")),
    check_content_hashes: bool = typer.Option(
        False, help="Hash current/validation images to detect copied-frame leakage."
    ),
) -> None:
    cfg = _resolve_config(config)
    if base_map:
        cfg.paths.base_map = str(base_map)
    if map_adapter:
        cfg.adapters.map_loader = map_adapter
    if historical:
        cfg.paths.historical_data = str(historical)
    if validation:
        cfg.paths.current_validation = str(validation)
    if current_images:
        cfg.paths.current_images = str(current_images)
    errors = cfg.validate(require_paths=True)
    if errors:
        raise typer.BadParameter("; ".join(errors))
    output.mkdir(parents=True, exist_ok=True)
    model = load_map(cfg.paths.base_map, cfg.adapters.map_loader)
    records = build_image_manifest(
        cfg.paths.historical_data,
        cfg.paths.current_validation or None,
        cfg.paths.current_images or None,
    )
    records = enrich_records_with_quality(records, cfg.quality)
    write_manifest(records, output / "manifests" / "images.csv")
    split_audit = audit_dataset_splits(records, check_content_hashes=check_content_hashes)
    write_json(split_audit, output / "manifests" / "split_audit.json")
    snapshot = create_map_snapshot(model.root or cfg.paths.base_map)
    save_snapshot(snapshot, output / "base_map_hashes.json")
    markdown = generate_data_audit_markdown(
        records,
        {"cameras": len(model.cameras), "images": len(model.images), "points3d": len(model.points3d)},
        split_audit.validation_grade.value,
    )
    markdown += "\n## Split audit\n\n"
    markdown += f"- Critical leakage: `{split_audit.critical_leakage}`\n"
    markdown += f"- Exact path overlaps: `{len(split_audit.exact_path_overlaps)}`\n"
    markdown += f"- Duplicate-content overlaps: `{len(split_audit.duplicate_content_overlaps)}`\n"
    for warning in split_audit.warnings:
        markdown += f"- Warning: {warning}\n"
    (output / "data_audit.md").write_text(markdown, encoding="utf-8")
    console.print(f"Audit written to {output}")


@app.command("run-direct")
def run_direct(
    config: Path = typer.Option(..., exists=True, dir_okay=False),
    output: Optional[Path] = typer.Option(None),
) -> None:
    cfg = _resolve_config(config)
    destination = output or Path(cfg.paths.output_root) / "direct_ingestion"
    destination.mkdir(parents=True, exist_ok=True)
    summary = _run_direct_ingestion(cfg, destination)
    console.print_json(json.dumps(to_jsonable(summary)))


@app.command("run-protocol")
def run_protocol(
    config: Path = typer.Option(..., exists=True, dir_okay=False),
    output: Optional[Path] = typer.Option(None),
) -> None:
    """Run all currently resolvable stages and emit explicit blockers for external stages."""

    cfg = _resolve_config(config)
    destination = output or Path(cfg.paths.output_root) / "protocol"
    destination.mkdir(parents=True, exist_ok=True)
    summary = _run_direct_ingestion(cfg, destination)
    blockers: list[str] = []
    if not cfg.paths.precomputed_change_masks:
        blockers.append(
            "No precomputed change masks configured. Direct candidates remain unpromoted until pose-aligned multi-view change detection is run."
        )
    if not cfg.paths.current_validation:
        blockers.append(
            "No independent current validation session configured. E0-E5 promotion evaluation is provisional only."
        )
    status = {
        "completed": ["base_map_snapshot", "data_manifest", "quality_filter", "direct_registration"],
        "blocked_or_pending": blockers,
        "summary": summary,
    }
    write_json(status, destination / "protocol_status.json")
    lines = ["# Protocol status", "", "## Completed", ""]
    lines.extend(f"- {item}" for item in status["completed"])
    lines.extend(["", "## Pending / blockers", ""])
    lines.extend(f"- {item}" for item in blockers or ["None"])
    (destination / "PROTOCOL_STATUS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    console.print(f"Protocol outputs: {destination}")


@app.command("synthetic-demo")
def synthetic_demo(
    output: Path = typer.Option(Path("runs/synthetic_demo")),
) -> None:
    report = run_synthetic_demo(output)
    console.print_json(json.dumps(to_jsonable(report)))
    console.print(f"Synthetic artifacts written to {output}")


if __name__ == "__main__":
    app()
