"""Statistical, selective-risk, and covisibility-graph diagnostics."""

from .graph import CovisibilityFragilityReport, analyze_covisibility_fragility
from .regions import (
    RegionDiagnosisConfig,
    RegionDiagnosticsReport,
    diagnose_regions,
)
from .risk_coverage import RiskCoverageReport, evaluate_risk_coverage
from .statistics import (
    ProportionInterval,
    beta_posterior_mean,
    empirical_bayes_prior,
    wilson_interval,
)

__all__ = [
    "CovisibilityFragilityReport",
    "ProportionInterval",
    "RegionDiagnosisConfig",
    "RegionDiagnosticsReport",
    "RiskCoverageReport",
    "analyze_covisibility_fragility",
    "beta_posterior_mean",
    "diagnose_regions",
    "empirical_bayes_prior",
    "evaluate_risk_coverage",
    "wilson_interval",
]
