import math

import numpy as np

from src.phase1.evaluators.phase1_validation_layers.layer3_oracle_profitability import (
    _dominant_pair_positive_ratio,
    _fee_drag,
    _per_code_profitability,
    _top_contribution_ratio,
)
from src.phase1.evaluators.phase1_validation_layers.layer2_behavior_quality import (
    _entropy_purity,
)
from src.phase1.evaluators.phase1_validation_layers.layer4_label_predictability import (
    _fit_linear_probe,
    _predict_probe,
)
from src.phase1.metrics import (
    Phase1BehaviorQualityMetrics,
    Phase1BehaviorQualityThresholds,
    Phase1EvaluationSnapshot,
    Phase1LabelPredictabilityMetrics,
    Phase1LabelPredictabilityThresholds,
    Phase1OracleProfitabilityMetrics,
    Phase1OracleProfitabilityThresholds,
    Phase1VQInternalMetrics,
    Phase1VQInternalThresholds,
    Phase1ValidationRuntimeConfig,
    evaluate_behavior_quality_rules,
    evaluate_label_predictability_rules,
    evaluate_oracle_profitability_rules,
    evaluate_vq_internal_rules,
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


def _passing_oracle_metrics(**overrides) -> Phase1OracleProfitabilityMetrics:
    values = {
        "mean_decoded_advantage_vs_flat": 1.0,
        "decoded_win_rate_vs_flat": 0.60,
        "mean_advantage_vs_random_label": 0.50,
        "random_label_relative_lift": 0.30,
        "retention_ratio": 0.70,
        "downside_control": 1.0,
        "risk_adjusted_return": 0.50,
        "top_5_contribution": 0.20,
        "trimmed_decoded_advantage": 0.50,
        "fee_drag": 0.10,
        "turnover_return_correlation": 0.0,
        "bad_code_ratio": 0.0,
        "dominant_pair_positive_ratio": 0.80,
        "random_label_risk_adjusted_return": 0.10,
        "risk_adjusted_return_vs_random": 0.40,
    }
    values.update(overrides)
    return Phase1OracleProfitabilityMetrics(**values)


def _passing_vq_metrics(**overrides) -> Phase1VQInternalMetrics:
    values = {
        "validation_action_accuracy": 0.90,
        "reconstruction_loss_gap": 1.10,
        "active_code_ratio": 0.90,
        "max_code_occupancy": 0.30,
        "normalized_code_perplexity": 0.70,
        "dead_code_ratio": 0.10,
        "assignment_churn_recent_mean": 0.10,
        "code_lifetime_pass_ratio": 0.90,
        "quantization_distance": 0.20,
        "nearest_second_margin_median": 0.20,
        "decoder_turnover_error": 0.10,
        "entry_timing_error_median": 0.10,
        "direction_accuracy": 0.90,
        "quantization_distance_gap": 1.0,
    }
    values.update(overrides)
    return Phase1VQInternalMetrics(**values)


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


def test_behavior_entropy_purity_can_support_structure_gate() -> None:
    values = np.asarray(["a"] * 70 + ["b"] * 10 + ["c"] * 10 + ["d"] * 10)

    assert _entropy_purity(values) > 0.20


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


def test_top_contribution_uses_top_fraction_of_all_horizons() -> None:
    returns = np.asarray([100.0, 50.0] + [-1.0] * 18)

    assert math.isclose(_top_contribution_ratio(returns, 0.10), 1.0)


def test_per_code_profitability_filters_to_active_codes() -> None:
    code_ids = np.asarray([0, 0, 0, 1])
    decoded_advantage = np.asarray([1.0, 1.0, 1.0, -10.0])
    decoded_returns = decoded_advantage
    dp_advantage = np.asarray([1.0, 1.0, 1.0, 1.0])
    gross_returns = np.asarray([1.0, 1.0, 1.0, 1.0])
    fees = np.asarray([0.0, 0.0, 0.0, 0.0])

    per_code = _per_code_profitability(
        code_ids=code_ids,
        decoded_advantage=decoded_advantage,
        decoded_returns=decoded_returns,
        dp_advantage=dp_advantage,
        decoded_gross_returns=gross_returns,
        decoded_fees=fees,
        thresholds=Phase1OracleProfitabilityThresholds(),
        active_codes=(0,),
    )

    assert [item.code_id for item in per_code] == [0]


def test_oracle_rules_require_risk_adjusted_return_above_random() -> None:
    thresholds = Phase1OracleProfitabilityThresholds()

    passing = evaluate_oracle_profitability_rules(
        _passing_oracle_metrics(risk_adjusted_return_vs_random=0.01),
        thresholds,
    )
    failing = evaluate_oracle_profitability_rules(
        _passing_oracle_metrics(risk_adjusted_return_vs_random=-0.01),
        thresholds,
    )

    assert passing.passed
    assert not failing.passed


def test_vq_rules_treat_quantization_distance_gap_as_scoring_signal() -> None:
    thresholds = Phase1VQInternalThresholds()

    passing = evaluate_vq_internal_rules(
        _passing_vq_metrics(quantization_distance_gap=1.25),
        thresholds,
    )
    still_passing = evaluate_vq_internal_rules(
        _passing_vq_metrics(quantization_distance_gap=1.26),
        thresholds,
    )

    assert passing.passed
    assert still_passing.passed


def test_vq_rules_use_raw_entry_timing_timestep_threshold() -> None:
    thresholds = Phase1VQInternalThresholds(entry_timing_error_max=10.8)

    passing = evaluate_vq_internal_rules(
        _passing_vq_metrics(entry_timing_error_median=10.8),
        thresholds,
    )
    failing = evaluate_vq_internal_rules(
        _passing_vq_metrics(entry_timing_error_median=10.9),
        thresholds,
    )

    assert passing.passed
    assert not failing.passed


def test_vq_rules_warn_on_context_dependent_missing_stability_metrics() -> None:
    thresholds = Phase1VQInternalThresholds()

    result = evaluate_vq_internal_rules(
        _passing_vq_metrics(
            assignment_churn_recent_mean=math.nan,
            entry_timing_error_median=math.nan,
        ),
        thresholds,
    )

    by_name = {metric.name: metric for metric in result.metrics}
    assert result.passed
    assert by_name["assignment_churn_recent_mean"].severity == "warn"
    assert by_name["entry_timing_error_median"].severity == "warn"


def test_dominant_pair_positive_ratio_is_computed_per_active_code() -> None:
    snapshot = Phase1EvaluationSnapshot(
        split="val",
        epoch=1,
        sample_ids=np.arange(5),
        states=np.zeros((5, 4, 2), dtype=np.float32),
        prices=np.ones((5, 4), dtype=np.float32),
        demo_actions=np.ones((5, 4), dtype=np.int64),
        demo_rewards=np.zeros((5, 4), dtype=np.float32),
        decoded_actions=np.asarray(
            [
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [2, 2, 2, 2],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ],
            dtype=np.int64,
        ),
        decoded_logits=np.zeros((5, 4, 3), dtype=np.float32),
        code_ids=np.asarray([0, 0, 0, 1, 1], dtype=np.int64),
        z_e=np.zeros((5, 2), dtype=np.float32),
        z_q=np.zeros((5, 2), dtype=np.float32),
        distances=np.zeros((5, 2), dtype=np.float32),
        reconstruction_loss=0.0,
        action_accuracy=1.0,
    )
    decoded_advantage = np.asarray([-1.0, -1.0, 10.0, 1.0, 1.0])

    assert _dominant_pair_positive_ratio(snapshot, decoded_advantage) == 0.5


def test_dominant_pair_positive_ratio_ignores_inactive_codes() -> None:
    sample_count = 101
    snapshot = Phase1EvaluationSnapshot(
        split="val",
        epoch=1,
        sample_ids=np.arange(sample_count),
        states=np.zeros((sample_count, 4, 2), dtype=np.float32),
        prices=np.ones((sample_count, 4), dtype=np.float32),
        demo_actions=np.ones((sample_count, 4), dtype=np.int64),
        demo_rewards=np.zeros((sample_count, 4), dtype=np.float32),
        decoded_actions=np.ones((sample_count, 4), dtype=np.int64),
        decoded_logits=np.zeros((sample_count, 4, 3), dtype=np.float32),
        code_ids=np.asarray([0] * 100 + [1], dtype=np.int64),
        z_e=np.zeros((sample_count, 2), dtype=np.float32),
        z_q=np.zeros((sample_count, 2), dtype=np.float32),
        distances=np.zeros((sample_count, 2), dtype=np.float32),
        reconstruction_loss=0.0,
        action_accuracy=1.0,
    )
    decoded_advantage = np.asarray([1.0] * 100 + [-10.0])

    assert (
        _dominant_pair_positive_ratio(
            snapshot,
            decoded_advantage,
            active_code_min_occupancy=0.02,
        )
        == 1.0
    )


def test_linear_probe_is_trainable_and_deterministic() -> None:
    features = np.asarray(
        [[-2.0], [-1.0], [-1.5], [1.0], [2.0], [1.5]],
        dtype=np.float64,
    )
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    runtime_config = Phase1ValidationRuntimeConfig(
        probe_epochs=80,
        probe_learning_rate=0.05,
        probe_batch_size=2,
        random_seed=7,
    )

    first_probe = _fit_linear_probe(features, labels, runtime_config)
    second_probe = _fit_linear_probe(features, labels, runtime_config)
    first_ranked = _predict_probe(first_probe, features)
    second_ranked = _predict_probe(second_probe, features)

    assert np.mean(first_ranked[:, 0] == labels) >= 5 / 6
    np.testing.assert_array_equal(first_ranked, second_ranked)
