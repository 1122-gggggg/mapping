from __future__ import annotations

import argparse
from pathlib import Path

from mapdoctor.adapters import get_adapter, list_adapters
from mapdoctor.benchmark import load_localization_results, summarize_benchmark
from mapdoctor.comparison import compare_results
from mapdoctor.config import load_settings
from mapdoctor.metrics import analyze
from mapdoctor.report import (
    write_analysis_bundle,
    write_benchmark_bundle,
    write_comparison_bundle,
)
from mapdoctor.scoring import score


def _add_map_args(parser: argparse.ArgumentParser, include_backend: bool) -> None:
    parser.add_argument("model", type=Path, help="Sparse model directory, e.g. sparse/0 or sparse")
    if include_backend:
        parser.add_argument("--backend", choices=list_adapters(), required=True, help="Map producer interface")
    parser.add_argument("--config", type=Path, default=None, help="Optional MapDoctor JSON configuration")
    parser.add_argument("--output", type=Path, default=Path("mapdoctor-report"), help="Output directory")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mapdoctor",
        description="Diagnose, benchmark, and regression-test COLMAP/GLOMAP/GLUEMAP localization maps.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    analyze_parser = sub.add_parser("analyze", help="Analyze a map through an explicit backend interface")
    _add_map_args(analyze_parser, include_backend=True)
    for backend in list_adapters():
        backend_parser = sub.add_parser(backend, help=f"Analyze a map through the {backend.upper()} interface")
        _add_map_args(backend_parser, include_backend=False)
        backend_parser.set_defaults(backend=backend)

    benchmark_parser = sub.add_parser("benchmark", help="Evaluate held-out localization query results")
    benchmark_parser.add_argument("results", type=Path)
    benchmark_parser.add_argument("--config", type=Path, default=None)
    benchmark_parser.add_argument("--output", type=Path, default=Path("mapdoctor-benchmark"))

    compare_parser = sub.add_parser("compare", help="Regression-test a candidate map against a base")
    compare_parser.add_argument("base", type=Path)
    compare_parser.add_argument("candidate", type=Path)
    compare_parser.add_argument("--config", type=Path, default=None)
    compare_parser.add_argument(
        "--query-manifest",
        type=Path,
        default=None,
        help="Immutable JSON/text query universe; missing candidate rows fail closed",
    )
    compare_parser.add_argument("--output", type=Path, default=Path("mapdoctor-comparison"))

    regions_parser = sub.add_parser(
        "diagnose-regions",
        help="Classify weak/healthy regions with confidence intervals and shrinkage",
    )
    regions_parser.add_argument("results", type=Path)
    regions_parser.add_argument("--config", type=Path, default=None)
    regions_parser.add_argument(
        "--assignments",
        type=Path,
        default=None,
        help="Optional JSON mapping of query names to route/region IDs",
    )
    regions_parser.add_argument("--cell-size", type=float, default=None)
    regions_parser.add_argument("--min-samples", type=int, default=8)
    regions_parser.add_argument("--min-failures-for-weak", type=int, default=2)
    regions_parser.add_argument("--weak-failure-rate", type=float, default=0.30)
    regions_parser.add_argument("--healthy-failure-rate", type=float, default=0.10)
    regions_parser.add_argument("--confidence", type=float, default=0.95)
    regions_parser.add_argument("--prior-strength", type=float, default=8.0)
    regions_parser.add_argument(
        "--output",
        type=Path,
        default=Path("mapdoctor-region-diagnostics.json"),
    )

    risk_parser = sub.add_parser(
        "risk-coverage",
        help="Evaluate a predicted failure-risk score as a selective localizer",
    )
    risk_parser.add_argument("results", type=Path)
    risk_parser.add_argument("risk_scores", type=Path)
    risk_parser.add_argument("--config", type=Path, default=None)
    risk_parser.add_argument("--ece-bins", type=int, default=10)
    risk_parser.add_argument(
        "--target-failure-rate",
        type=float,
        nargs="+",
        default=[0.01, 0.02, 0.05],
    )
    risk_parser.add_argument(
        "--output",
        type=Path,
        default=Path("mapdoctor-risk-coverage.json"),
    )

    graph_parser = sub.add_parser(
        "graph-fragility",
        help="Find articulation images and bridge edges in the covisibility graph",
    )
    graph_parser.add_argument("model", type=Path)
    graph_parser.add_argument("--backend", choices=list_adapters(), required=True)
    graph_parser.add_argument("--minimum-shared-landmarks", type=int, default=15)
    graph_parser.add_argument(
        "--route-images",
        type=Path,
        default=None,
        help="Optional JSON/text manifest of route-relevant reference image names",
    )
    graph_parser.add_argument(
        "--output",
        type=Path,
        default=Path("mapdoctor-graph-fragility.json"),
    )

    hloc_parser = sub.add_parser(
        "export-hloc",
        help="Convert trusted hloc localization logs into the MapDoctor benchmark schema",
    )
    hloc_parser.add_argument("logs", type=Path, help="hloc *_logs.pkl file")
    hloc_parser.add_argument("reference_model", type=Path, help="Reference SfM model used by hloc")
    hloc_parser.add_argument("--queries", type=Path, default=None, help="Optional hloc query list; includes skipped queries")
    hloc_parser.add_argument("--output", type=Path, required=True, help="Output .csv or .json benchmark file")
    hloc_parser.add_argument(
        "--trust-pickle",
        action="store_true",
        help="Required acknowledgement that Python pickle can execute code; use only trusted hloc logs",
    )
    hloc_parser.add_argument(
        "--consensus-translation-fraction",
        type=float,
        default=0.01,
        help="Pose-consensus center-distance threshold as a fraction of reference scene extent",
    )
    hloc_parser.add_argument(
        "--consensus-max-rotation-deg",
        type=float,
        default=5.0,
        help="Pose-consensus rotation threshold in degrees",
    )

    sub.add_parser("adapters", help="List available map interfaces")
    return parser


def _analyze(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    adapter = get_adapter(args.backend)
    inspection = adapter.inspect(args.model)
    model = adapter.load(args.model)
    metrics = analyze(model, settings.health)
    readiness = score(metrics, settings.health)
    paths = write_analysis_bundle(metrics, readiness, args.output)
    print(f"Backend: {adapter.display_name} ({inspection.model_format})")
    print(f"MapDoctor score: {readiness.score}/100 ({readiness.grade})")
    print(f"HTML: {paths['html']}")
    print(f"JSON: {paths['json']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in {"analyze", *list_adapters()}:
        return _analyze(args)
    if args.command == "benchmark":
        settings = load_settings(args.config)
        results = load_localization_results(args.results)
        summary = summarize_benchmark(results, settings.localization, settings.region_cell_size)
        paths = write_benchmark_bundle(summary, results, args.output)
        print(f"Queries: {summary.total_queries}")
        print(f"Strict success rate: {summary.strict_success_rate:.1%}")
        print(f"HTML: {paths['html']}")
        return 0
    if args.command == "compare":
        from mapdoctor.diagnostics.io import load_query_manifest

        settings = load_settings(args.config)
        required_queries = (
            load_query_manifest(args.query_manifest)
            if args.query_manifest is not None
            else None
        )
        result = compare_results(
            load_localization_results(args.base),
            load_localization_results(args.candidate),
            settings.localization,
            settings.comparison,
            required_queries=required_queries,
        )
        paths = write_comparison_bundle(result, args.output)
        print(f"Regression gate: {result.status}")
        print(f"Query universe: {result.query_universe_source} ({result.compared_queries})")
        if result.missing_from_candidate:
            print(f"Missing candidate queries: {len(result.missing_from_candidate)}")
        print(f"HTML: {paths['html']}")
        return 0 if result.status == "PASS" else 1
    if args.command == "diagnose-regions":
        from mapdoctor.diagnostics.io import load_region_assignments, write_json
        from mapdoctor.diagnostics.regions import RegionDiagnosisConfig, diagnose_regions

        settings = load_settings(args.config)
        assignments = (
            load_region_assignments(args.assignments)
            if args.assignments is not None
            else None
        )
        report = diagnose_regions(
            load_localization_results(args.results),
            settings.localization,
            assignments=assignments,
            cell_size=(
                settings.region_cell_size
                if args.cell_size is None
                else args.cell_size
            ),
            config=RegionDiagnosisConfig(
                weak_failure_rate=args.weak_failure_rate,
                healthy_failure_rate=args.healthy_failure_rate,
                confidence=args.confidence,
                min_samples=args.min_samples,
                min_failures_for_weak=args.min_failures_for_weak,
                prior_strength=args.prior_strength,
            ),
        )
        path = write_json(report, args.output)
        weak = sum(region.status == "WEAK" for region in report.regions)
        uncertain = sum(
            region.status in {"UNCERTAIN", "INSUFFICIENT_EVIDENCE"}
            for region in report.regions
        )
        print(f"Regions: {len(report.regions)}; weak: {weak}; uncertain: {uncertain}")
        print(f"JSON: {path}")
        return 0
    if args.command == "risk-coverage":
        from mapdoctor.diagnostics.io import load_risk_scores, write_json
        from mapdoctor.diagnostics.risk_coverage import evaluate_risk_coverage

        settings = load_settings(args.config)
        report = evaluate_risk_coverage(
            load_localization_results(args.results),
            load_risk_scores(args.risk_scores),
            settings.localization,
            ece_bins=args.ece_bins,
            target_failure_rates=args.target_failure_rate,
        )
        path = write_json(report, args.output)
        print(f"AURC: {report.aurc:.6f}; excess AURC: {report.excess_aurc:.6f}")
        print(f"Brier: {report.brier_score:.6f}; ECE: {report.expected_calibration_error:.6f}")
        print(f"JSON: {path}")
        return 0
    if args.command == "graph-fragility":
        from mapdoctor.diagnostics.graph import analyze_covisibility_fragility
        from mapdoctor.diagnostics.io import load_query_manifest, write_json

        model = get_adapter(args.backend).load(args.model)
        route_images = (
            load_query_manifest(args.route_images)
            if args.route_images is not None
            else None
        )
        report = analyze_covisibility_fragility(
            model,
            minimum_shared_landmarks=args.minimum_shared_landmarks,
            route_image_names=route_images,
        )
        path = write_json(report, args.output)
        print(
            f"Components: {report.component_count}; "
            f"articulations: {len(report.articulation_images)}; "
            f"bridges: {len(report.bridge_edges)}"
        )
        print(f"JSON: {path}")
        return 0
    if args.command == "export-hloc":
        from mapdoctor.integrations.hloc import export_hloc_logs, write_hloc_results

        results = export_hloc_logs(
            args.logs,
            args.reference_model,
            query_list=args.queries,
            trust_pickle=args.trust_pickle,
            consensus_translation_fraction=args.consensus_translation_fraction,
            consensus_max_rotation_deg=args.consensus_max_rotation_deg,
        )
        path = write_hloc_results(results, args.output)
        localized = sum(result.success for result in results)
        print(f"Exported {len(results)} queries ({localized} localized) to {path}")
        return 0
    if args.command == "adapters":
        for backend in list_adapters():
            adapter = get_adapter(backend)
            print(f"{backend}\t{adapter.__class__.__name__}\t{adapter.display_name}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
