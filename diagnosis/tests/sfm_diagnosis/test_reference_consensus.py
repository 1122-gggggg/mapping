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


def test_sigma_joint_is_max_of_held_out_median_normalized_scores():
    from sfm_diagnosis.reference_consensus import (
        compute_sigma_joint,
        fit_selective_normalization,
    )

    held_out = []
    for query, centers, sigma in (
        ("h0", ([0.0, 0.0, 0.0], [0.02, 0.0, 0.0]), 0.10),
        ("h1", ([0.0, 0.0, 0.0], [0.04, 0.0, 0.0]), 0.10),
    ):
        held_out.append(
            compute_reference_consensus(
                [
                    ReferenceHypothesis(query, "r0", np.array(centers[0]), np.eye(3) * sigma**2),
                    ReferenceHypothesis(query, "r1", np.array(centers[1]), np.eye(3) * sigma**2),
                ]
            )
        )
    norm = fit_selective_normalization(held_out)
    assert norm is not None

    dispersion_query = compute_reference_consensus(
        [
            ReferenceHypothesis("q", "a0", np.array([0.0, 0.0, 0.0]), np.eye(3) * 0.10**2),
            ReferenceHypothesis("q", "a1", np.array([1.0, 0.0, 0.0]), np.eye(3) * 0.10**2),
        ]
    )
    covariance_query = compute_reference_consensus(
        [
            ReferenceHypothesis("q", "r0", np.array([0.0, 0.0, 0.0]), np.eye(3) * 3.0**2),
            ReferenceHypothesis("q", "r1", np.array([0.02, 0.0, 0.0]), np.eye(3) * 3.0**2),
        ]
    )
    joint_disp = compute_sigma_joint(dispersion_query, norm)
    joint_cov = compute_sigma_joint(covariance_query, norm)
    assert joint_disp is not None and joint_cov is not None
    assert joint_disp == max(
        dispersion_query.sigma_cons_m / norm.median_sigma_cons_m,
        dispersion_query.sigma_disp_m / norm.median_sigma_disp_m,
    )
    assert joint_cov == max(
        covariance_query.sigma_cons_m / norm.median_sigma_cons_m,
        covariance_query.sigma_disp_m / norm.median_sigma_disp_m,
    )
    assert joint_disp > 1.0
    assert joint_cov > 1.0


def test_sigma_joint_rejects_covariance_ineligible_queries():
    from sfm_diagnosis.reference_consensus import compute_sigma_joint, fit_selective_normalization

    held = compute_reference_consensus(
        [
            ReferenceHypothesis("h", "r0", np.array([0.0, 0.0, 0.0]), np.eye(3) * 0.05**2),
            ReferenceHypothesis("h", "r1", np.array([0.01, 0.0, 0.0]), np.eye(3) * 0.05**2),
        ]
    )
    ineligible = compute_reference_consensus(
        [
            ReferenceHypothesis("q", "r0", np.array([0.0, 0.0, 0.0])),
            ReferenceHypothesis("q", "r1", np.array([0.4, 0.0, 0.0])),
        ]
    )
    assert compute_sigma_joint(ineligible, fit_selective_normalization([held])) is None


def test_analyze_reference_hypotheses_rejects_held_out_query_leak(tmp_path):
    from sfm_diagnosis.reference_consensus import analyze_reference_hypotheses

    path = tmp_path / "hyp.json"
    path.write_text(
        """
[
  {"query_id": "q0", "reference_id": "r0", "x": 0, "y": 0, "z": 0, "sigma_m": 0.05},
  {"query_id": "q0", "reference_id": "r1", "x": 0.02, "y": 0, "z": 0, "sigma_m": 0.05}
]
""",
        encoding="utf-8",
    )
    try:
        analyze_reference_hypotheses(path, held_out_path=path)
    except ValueError as exc:
        assert "leak" in str(exc)
    else:
        raise AssertionError("overlapping held-out query ids must fail closed")


def test_analyze_reference_hypotheses_leaves_sigma_joint_unset_without_held_out(tmp_path):
    from sfm_diagnosis.reference_consensus import analyze_reference_hypotheses

    path = tmp_path / "hyp.json"
    path.write_text(
        """
[
  {"query_id": "q0", "reference_id": "r0", "x": 0, "y": 0, "z": 0, "sigma_m": 0.05},
  {"query_id": "q0", "reference_id": "r1", "x": 0.02, "y": 0, "z": 0, "sigma_m": 0.05}
]
""",
        encoding="utf-8",
    )
    rows = analyze_reference_hypotheses(path)
    assert rows[0]["sigma_joint"] is None
    payload = str(rows[0])
    assert "score_gap" not in payload
    assert "top1" not in payload


def test_center_only_hypotheses_keep_dispersion_but_not_covariance():
    hypotheses = [
        ReferenceHypothesis("q0", "r0", np.array([0.0, 0.0, 0.0])),
        ReferenceHypothesis("q0", "r1", np.array([0.1, 0.0, 0.0])),
    ]
    metrics = compute_reference_consensus(hypotheses)
    assert metrics.covariance_eligible_count == 0
    assert metrics.sigma_cons_m is None
    assert metrics.sigma_disp_m > 0.0
