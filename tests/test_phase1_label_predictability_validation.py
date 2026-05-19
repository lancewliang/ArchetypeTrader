import numpy as np
import polars as pl

from src.data.state_normalizer import StateNormalizer
from src.phase1.evaluators.phase1_validation_layers.layer4_label_predictability import (
    build_probe_features,
)
from src.phase1.metrics import (
    Phase1BehaviorQualityMetrics,
    Phase1LabelPredictabilityMetrics,
    Phase1LabelPredictabilityThresholds,
    Phase1LayerResult,
    Phase1MetricResult,
    Phase1OracleProfitabilityMetrics,
    Phase1TeacherQualityMetrics,
    Phase1TieBreakerMetrics,
    Phase1VQInternalMetrics,
    Phase1ValidationMetrics,
    Phase1ValidationScore,
    Phase1ValidationScoreWeights,
    aggregate_validation_result,
    build_tie_breaker_metrics,
    compute_phase1_validation_score,
    evaluate_label_predictability_rules,
)


def test_build_probe_features_uses_previous_segment_and_current_t0() -> None:
    states = np.asarray(
        [
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
            [[4.0, 40.0], [5.0, 50.0], [6.0, 60.0]],
        ],
        dtype=np.float32,
    )

    features = build_probe_features(states)

    expected = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 10.0],
            [1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0],
        ]
    )
    assert features.shape == (2, 8)
    np.testing.assert_allclose(features, expected)


def test_state_normalizer_fits_train_and_reuses_same_stats_for_validation() -> None:
    train = pl.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0, 4.0],
            "feature_b": [10.0, 12.0, 14.0, 16.0],
        }
    )
    validation_values = np.asarray(
        [
            [2.0, 12.0],
            [4.0, 16.0],
        ],
        dtype=np.float32,
    )

    normalizer = StateNormalizer.fit(train, ["feature_a", "feature_b"])
    train_normalized = normalizer.transform(train.to_numpy())
    validation_normalized = normalizer.transform(validation_values)

    np.testing.assert_allclose(train_normalized.mean(axis=0), [0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(train_normalized.std(axis=0), [1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(
        validation_normalized,
        (validation_values - normalizer.mean) / normalizer.std,
    )


def test_label_predictability_reference_checks_do_not_fail_gate() -> None:
    metrics = Phase1LabelPredictabilityMetrics(
        probe_top1_accuracy=0.05,
        probe_top3_accuracy=0.10,
        probe_balanced_accuracy=0.04,
        label_entropy_given_morphology=1.8,
        mutual_information_lift=0.5,
        probe_return_retention=0.1,
        label_entropy=2.0,
        num_codes=10,
    )

    result = evaluate_label_predictability_rules(
        metrics,
        Phase1LabelPredictabilityThresholds(),
    )

    assert result.name == "label_predictability"
    assert result.passed is True
    assert all(metric.passed for metric in result.metrics)
    assert {metric.severity for metric in result.metrics} == {"warn"}


def test_label_predictability_is_tie_breaker_not_score_component() -> None:
    metrics = _sample_validation_metrics(
        label_predictability=Phase1LabelPredictabilityMetrics(
            probe_top1_accuracy=0.05,
            probe_top3_accuracy=0.67,
            probe_balanced_accuracy=0.04,
            label_entropy_given_morphology=1.8,
            mutual_information_lift=0.5,
            probe_return_retention=0.1,
            label_entropy=2.0,
            num_codes=10,
        )
    )

    score = compute_phase1_validation_score(
        metrics,
        Phase1ValidationScoreWeights(),
    )
    tie_breakers = build_tie_breaker_metrics(
        metrics,
        reconstruction_loss=0.123,
    )

    assert "label_predictability" not in {
        component.name for component in score.components
    }
    assert {
        component.name: component.weight for component in score.components
    }["oracle_profitability"] == 0.35
    assert tie_breakers.probe_top3_accuracy == 0.67


def test_aggregate_result_ignores_label_predictability_layer_failure() -> None:
    metrics = _sample_validation_metrics(
        label_predictability=Phase1LabelPredictabilityMetrics(
            probe_top1_accuracy=0.05,
            probe_top3_accuracy=0.10,
            probe_balanced_accuracy=0.04,
            label_entropy_given_morphology=1.8,
            mutual_information_lift=0.5,
            probe_return_retention=0.1,
            label_entropy=2.0,
            num_codes=10,
        )
    )

    result = aggregate_validation_result(
        checkpoint_id="ckpt",
        stage="vq",
        epoch=3,
        layers=(
            _layer("teacher_quality", True),
            _layer("label_predictability", False),
        ),
        metrics=metrics,
        code_diagnostics=(),
        drift_diagnostics={},
        score=Phase1ValidationScore(total_score=0.5, components=()),
        tie_breaker_metrics=Phase1TieBreakerMetrics(
            risk_adjusted_return=1.0,
            probe_top3_accuracy=0.10,
            retention_ratio=0.7,
            active_code_ratio=0.9,
            max_code_occupancy=0.2,
            reconstruction_loss=0.123,
        ),
    )

    assert result.passed is True
    assert result.failed_layers == ()
    assert result.score is not None


def _sample_validation_metrics(
    *,
    label_predictability: Phase1LabelPredictabilityMetrics,
) -> Phase1ValidationMetrics:
    return Phase1ValidationMetrics(
        teacher_quality=Phase1TeacherQualityMetrics(
            dp_advantage_vs_flat=1.0,
            dp_win_rate_vs_flat=0.7,
            near_zero_opportunity_ratio=0.1,
            fee_sensitivity=0.8,
            morphology_coverage=0.8,
            dp_return_concentration_after_top5_removed=0.5,
        ),
        vq_internal=Phase1VQInternalMetrics(
            validation_action_accuracy=0.95,
            reconstruction_loss_gap=1.05,
            active_code_ratio=0.9,
            max_code_occupancy=0.2,
            normalized_code_perplexity=0.7,
            dead_code_ratio=0.0,
            assignment_churn_recent_mean=0.05,
            code_lifetime_pass_ratio=0.95,
            quantization_distance=0.2,
            nearest_second_margin_median=0.2,
            decoder_turnover_error=0.05,
            entry_timing_error_median=0.03,
            direction_accuracy=0.95,
            quantization_distance_gap=1.05,
        ),
        behavior_quality=Phase1BehaviorQualityMetrics(
            weak_support_code_ratio=0.05,
            weak_morphology_code_ratio=0.1,
            weak_motif_code_ratio=0.1,
            weak_pair_code_ratio=0.1,
            weak_lift_nonprofitable_code_ratio=0.1,
            intra_code_action_similarity=0.8,
            inter_intra_separation=1.6,
            latent_silhouette_score=0.3,
            duplicate_code_pair_count=0,
            profitable_code_coverage=0.8,
            num_codes=10,
        ),
        oracle_profitability=Phase1OracleProfitabilityMetrics(
            mean_decoded_advantage_vs_flat=0.5,
            decoded_win_rate_vs_flat=0.65,
            mean_advantage_vs_random_label=0.3,
            random_label_relative_lift=0.4,
            retention_ratio=0.7,
            downside_control=0.8,
            risk_adjusted_return=1.2,
            top_5_contribution=0.3,
            trimmed_decoded_advantage=0.2,
            fee_drag=0.2,
            turnover_return_correlation=0.1,
            bad_code_ratio=0.05,
            dominant_pair_positive_ratio=0.8,
            random_label_risk_adjusted_return=0.2,
            risk_adjusted_return_vs_random=1.0,
        ),
        label_predictability=label_predictability,
    )


def _layer(name: str, passed: bool) -> Phase1LayerResult:
    return Phase1LayerResult(
        layer_id=4 if name == "label_predictability" else 0,
        name=name,
        passed=passed,
        metrics=(
            Phase1MetricResult(
                name=f"{name}_metric",
                value=1.0,
                threshold="reference",
                severity="pass" if passed else "fail",
                passed=passed,
                layer=name,
            ),
        ),
    )
