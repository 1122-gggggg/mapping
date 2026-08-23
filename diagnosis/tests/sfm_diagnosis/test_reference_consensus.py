import numpy as np

from sfm_diagnosis.reference_consensus import (
    ReferenceConsensusStatus,
    ReferenceHypothesis,
    assess_reference_consensus,
    compute_reference_consensus,
)


def _hypothesis(reference_id, center, sigma=0.05):
    return ReferenceHypothesis(
        query_id="q0",
        reference_id=reference_id,
        center_w=np.asarray(center, dtype=float),
        covariance_w=np.eye(3) * sigma**2,
    )


def test_consistent_reference_hypotheses_are_healthy():
    hypotheses = [
        _hypothesis("r0", [0.00, 0.00, 0.00]),
        _hypothesis("r1", [0.03, -0.01, 0.01]),
        _hypothesis("r2", [-0.02, 0.01, -0.01]),
        _hypothesis("r3", [0.01, 0.02, 0.00]),
    ]
    metrics = compute_reference_consensus(hypotheses)
    assessment = assess_reference_consensus(
        metrics,
        max_dispersion_m=0.20,
        max_consensus_sigma_m=0.20,
    )
    assert metrics.covariance_eligible_ratio == 1.0
    assert metrics.sigma_disp_m < 0.05
    assert metrics.sigma_cons_m is not None
    assert assessment.status == ReferenceConsensusStatus.HEALTHY


def test_multimodal_reference_hypotheses_trigger_disagreement():
    hypotheses = [
        _hypothesis("a0", [0.00, 0.00, 0.00]),
        _hypothesis("a1", [0.02, 0.00, 0.00]),
        _hypothesis("b0", [2.00, 0.00, 0.00]),
        _hypothesis("b1", [2.02, 0.00, 0.00]),
    ]
    metrics = compute_reference_consensus(hypotheses)
    assessment = assess_reference_consensus(
        metrics,
        max_dispersion_m=0.30,
        max_consensus_sigma_m=1.0,
    )
    assert metrics.sigma_disp_m > 0.9
    assert assessment.status in {
        ReferenceConsensusStatus.REFERENCE_DISAGREEMENT,
        ReferenceConsensusStatus.CRITICAL,
    }


def test_agreeing_but_uncertain_hypotheses_trigger_observability():
    hypotheses = [
        _hypothesis("r0", [0.00, 0.00, 0.00], sigma=3.0),
        _hypothesis("r1", [0.02, 0.00, 0.00], sigma=3.0),
        _hypothesis("r2", [-0.01, 0.01, 0.00], sigma=3.0),
    ]
    metrics = compute_reference_consensus(hypotheses)
    assessment = assess_reference_consensus(
        metrics,
        max_dispersion_m=0.30,
        max_consensus_sigma_m=0.50,
    )
    assert metrics.sigma_disp_m < 0.1
    assert metrics.sigma_cons_m is not None
    assert metrics.sigma_cons_m > 0.5
    assert assessment.status == ReferenceConsensusStatus.OBSERVABILITY_WEAK


def test_center_only_hypotheses_keep_dispersion_but_not_covariance():
    hypotheses = [
        ReferenceHypothesis("q0", "r0", np.array([0.0, 0.0, 0.0])),
        ReferenceHypothesis("q0", "r1", np.array([0.1, 0.0, 0.0])),
    ]
    metrics = compute_reference_consensus(hypotheses)
    assert metrics.covariance_eligible_count == 0
    assert metrics.sigma_cons_m is None
    assert metrics.sigma_disp_m > 0.0
