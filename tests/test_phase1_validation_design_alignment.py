import math

import numpy as np

from src.phase1.evaluators.phase1_validation_layers.layer3_oracle_profitability import (
    _fee_drag,
)
from src.phase1.metrics import (
    Phase1BehaviorQualityMetrics,
    Phase1BehaviorQualityThresholds,
    Phase1LabelPredictabilityMetrics,
    Phase1LabelPredictabilityThresholds,
    evaluate_behavior_quality_rules,
    evaluate_label_predictability_rules,
)


def _passing_behavior_metrics(**overrides) -> Phase1BehaviorQualityMetrics:
    values = {
        "weak_support_code_ratio": 0.0,
        "weak_morphology_code_ratio": 0.0,
        "weak_motif_code_ratio": 0.0,
        "weak_pair_code_ratio": 0.0,
        "weak_lift_nonprofitable_code_ratio": 0.0,
        "intra_code_action_similarity": 0.80,
        "inter_intra_separation": 1.50,
        "latent_silhouette_score": 0.20,
        "duplicate_code_pair_count": 0,
        "profitable_code_coverage": 0.80,
        "num_codes": 10,
    }
    values.update(overrides)
    return Phase1BehaviorQualityMetrics(**values)


def _passing_label_metrics(**overrides) -> Phase1LabelPredictabilityMetrics:
    values = {
        "probe_top1_accuracy": 0.40,
        "probe_top3_accuracy": 0.70,
        "probe_balanced_accuracy": 0.35,
        "label_entropy_given_morphology": 0.70,
        "mutual_information_lift": 2.5,
        "probe_return_retention": 0.50,
        "label_entropy": 1.0,
        "num_codes": 10,
    }
    values.update(overrides)
    return Phase1LabelPredictabilityMetrics(**values)


def test_behavior_rules_use_distinct_support_and_structure_weak_thresholds() -> None:
    thresholds = Phase1BehaviorQualityThresholds()

    support_failed = evaluate_behavior_quality_rules(
        _passing_behavior_metrics(weak_support_code_ratio=0.21),
        thresholds,
    )
    structure_passed = evaluate_behavior_quality_rules(
        _passing_behavior_metrics(weak_morphology_code_ratio=0.39),
        thresholds,
    )

    assert not support_failed.passed
    assert structure_passed.passed


def test_behavior_duplicate_code_pair_limit_defaults_to_codebook_size() -> None:
    thresholds = Phase1BehaviorQualityThresholds()

    at_k = evaluate_behavior_quality_rules(
        _passing_behavior_metrics(duplicate_code_pair_count=10, num_codes=10),
        thresholds,
    )
    above_k = evaluate_behavior_quality_rules(
        _passing_behavior_metrics(duplicate_code_pair_count=11, num_codes=10),
        thresholds,
    )

    assert at_k.passed
    assert not above_k.passed


def test_label_entropy_given_morphology_is_hard_gate() -> None:
    thresholds = Phase1LabelPredictabilityThresholds()

    passing = evaluate_label_predictability_rules(
        _passing_label_metrics(label_entropy_given_morphology=0.80, label_entropy=1.0),
        thresholds,
    )
    failing = evaluate_label_predictability_rules(
        _passing_label_metrics(label_entropy_given_morphology=0.81, label_entropy=1.0),
        thresholds,
    )

    assert passing.passed
    assert not failing.passed


def test_fee_drag_uses_total_fee_over_gross_profit() -> None:
    fees = np.asarray([0.10, 0.20, 0.30])
    gross_returns = np.asarray([1.0, -2.0, 3.0])

    assert math.isclose(
        _fee_drag(fees, gross_returns),
        (0.10 + 0.20 + 0.30) / (1.0 + 3.0),
        rel_tol=1e-12,
    )
    assert math.isinf(_fee_drag(fees, np.asarray([-1.0, 0.0, -2.0])))
