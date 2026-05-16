from __future__ import annotations

import pytest

from src.phase1.metrics import (
    Phase1MetricResult,
    Phase1LabelPredictabilityMetrics,
    Phase1LabelPredictabilityThresholds,
    Phase1VQInternalMetrics,
    Phase1VQInternalThresholds,
    evaluate_label_predictability_rules,
    evaluate_vq_internal_rules,
)


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


def test_rule_messages_include_direction_and_change_meaning() -> None:
    vq_result = evaluate_vq_internal_rules(
        _passing_vq_metrics(),
        Phase1VQInternalThresholds(),
    )
    vq_messages = {metric.name: metric.message for metric in vq_result.metrics}

    assert "指标方向：越大越好" in vq_messages["validation_action_accuracy"]
    assert "变大表示质量、收益或覆盖度提升" in vq_messages["validation_action_accuracy"]
    assert "指标方向：越小越好" in vq_messages["reconstruction_loss_gap"]
    assert "变大表示风险、误差或异常占比上升" in vq_messages["reconstruction_loss_gap"]
    assert "指标方向：落在目标区间内最好" in vq_messages["normalized_code_perplexity"]
    assert "低于下限或高于上限都可能失败" in vq_messages["normalized_code_perplexity"]

    vq_metrics = {metric.name: metric for metric in vq_result.metrics}
    assert vq_metrics["validation_action_accuracy"].threshold_value == 0.85
    assert vq_metrics["validation_action_accuracy"].direction == "greater_is_better"
    assert vq_metrics[
        "validation_action_accuracy"
    ].distance_to_threshold == pytest.approx(0.05)
    assert vq_metrics["reconstruction_loss_gap"].threshold_value == 1.25
    assert vq_metrics["reconstruction_loss_gap"].direction == "less_is_better"
    assert vq_metrics["reconstruction_loss_gap"].distance_to_threshold == pytest.approx(
        0.15
    )
    assert vq_metrics["normalized_code_perplexity"].threshold_value == (0.5, 0.9)
    assert vq_metrics["normalized_code_perplexity"].direction == "between"
    assert vq_metrics[
        "normalized_code_perplexity"
    ].distance_to_threshold == pytest.approx(0.2)

    label_result = evaluate_label_predictability_rules(
        _passing_label_metrics(),
        Phase1LabelPredictabilityThresholds(),
    )
    label_messages = {metric.name: metric.message for metric in label_result.metrics}

    assert "指标方向：越小越好" in label_messages["label_entropy_given_morphology"]
    assert (
        "变大表示 morphology 对 label 的解释力变弱"
        in label_messages["label_entropy_given_morphology"]
    )

    label_metrics = {metric.name: metric for metric in label_result.metrics}
    assert label_metrics["label_entropy_given_morphology"].threshold_value == 0.8
    assert label_metrics["label_entropy_given_morphology"].direction == "less_is_better"
    assert label_metrics[
        "label_entropy_given_morphology"
    ].distance_to_threshold == pytest.approx(0.1)


def test_metric_result_from_dict_is_backward_compatible() -> None:
    metric = Phase1MetricResult.from_dict(
        {
            "name": "legacy_metric",
            "value": 1.0,
            "threshold": ">= 0.5",
            "severity": "pass",
            "passed": True,
            "layer": "legacy",
            "message": "ok",
        }
    )

    assert metric.threshold_value is None
    assert metric.direction is None
    assert metric.distance_to_threshold is None
