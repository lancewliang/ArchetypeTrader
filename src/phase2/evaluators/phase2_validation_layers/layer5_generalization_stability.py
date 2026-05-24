"""Phase II validation Layer 5: generalization, stability and predictability."""

from __future__ import annotations

import math

import numpy as np

from ...metrics import (
    Phase2GeneralizationStabilityMetrics,
    Phase2GeneralizationStabilityPayload,
    Phase2LayerComputation,
    Phase2PredictabilityMetrics,
    Phase2PredictabilityPayload,
)
from ._numeric import as_float_array, nan_value, safe_mean, safe_std


def compute_predictability_metrics(
    payload: Phase2PredictabilityPayload,
    *,
    selected_code_entropy: float,
    selected_code_entropy_given_morphology: float,
    mutual_information_lift: float,
) -> Phase2PredictabilityMetrics:
    """Build predictability metrics from probe payload and aggregate statistics."""

    return Phase2PredictabilityMetrics(
        probe_top1_accuracy=float(payload.probe_validation_accuracy),
        probe_top3_accuracy=nan_value(),
        probe_balanced_accuracy=nan_value(),
        selected_code_entropy_given_morphology=float(
            selected_code_entropy_given_morphology
        ),
        selected_code_entropy=float(selected_code_entropy),
        mutual_information_lift=float(mutual_information_lift),
    )


def compute_generalization_stability_metrics(
    payload: Phase2GeneralizationStabilityPayload,
    *,
    validation_mean_return: float,
    train_mean_return: float | None = None,
    test_mean_return: float | None = None,
    train_usage_distribution: tuple[float, ...] | None = None,
    validation_usage_distribution: tuple[float, ...] | None = None,
    q_margins: tuple[float, ...] = (),
    low_confidence_margin_threshold: float = 0.10,
    td_loss_history: tuple[float, ...] = (),
    imitation_loss_history: tuple[float, ...] = (),
    reward_mean_history: tuple[float, ...] = (),
    predictability_metrics: Phase2PredictabilityMetrics | None = None,
) -> Phase2LayerComputation:
    """Compute Layer 5 stability diagnostics from aggregate histories."""

    validation_score_history = as_float_array(payload.validation_score_history)
    selected_action_churn_history = as_float_array(payload.selected_action_churn_history)
    q_value_scale_history = as_float_array(payload.q_value_scale_history)
    q_margin_values = as_float_array(q_margins)

    metrics = Phase2GeneralizationStabilityMetrics(
        train_val_return_gap=_abs_gap(train_mean_return, validation_mean_return),
        val_test_return_gap=_abs_gap(validation_mean_return, test_mean_return),
        train_val_usage_kl=_kl_optional(
            train_usage_distribution,
            validation_usage_distribution,
        ),
        validation_score_churn=_last_abs_diff(validation_score_history),
        selected_action_churn=safe_mean(selected_action_churn_history),
        q_value_scale_mean=safe_mean(q_value_scale_history),
        q_value_scale_std=safe_std(q_value_scale_history),
        q_margin_mean=safe_mean(q_margin_values),
        low_confidence_selection_rate=_low_confidence_rate(
            q_margin_values,
            low_confidence_margin_threshold,
        ),
        td_loss_trend=_last_minus_first(td_loss_history),
        imitation_loss_trend=_last_minus_first(imitation_loss_history),
        reward_mean_trend=_last_minus_first(reward_mean_history),
        predictability=predictability_metrics,
    )
    return Phase2LayerComputation(
        layer_id=5,
        layer_name="generalization_stability",
        metrics=metrics,
        extra_payload={"generalization_stability_payload": payload},
    )


def _abs_gap(left: float | None, right: float | None) -> float:
    """Absolute finite gap or NaN."""

    if left is None or right is None:
        return nan_value()
    if not math.isfinite(float(left)) or not math.isfinite(float(right)):
        return nan_value()
    return abs(float(left) - float(right))


def _last_abs_diff(values: np.ndarray) -> float:
    """Absolute difference between last two finite history points."""

    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return nan_value()
    return float(abs(finite[-1] - finite[-2]))


def _last_minus_first(values: tuple[float, ...]) -> float:
    """Trend proxy: last finite value minus first finite value."""

    array = as_float_array(values)
    finite = array[np.isfinite(array)]
    if finite.size < 2:
        return nan_value()
    return float(finite[-1] - finite[0])


def _low_confidence_rate(values: np.ndarray, threshold: float) -> float:
    """Rate of q margins below threshold."""

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return nan_value()
    return float(np.mean((finite < threshold).astype(np.float64)))


def _kl_optional(
    left: tuple[float, ...] | None,
    right: tuple[float, ...] | None,
) -> float:
    """KL(left || right) for optional distributions."""

    if left is None or right is None:
        return nan_value()
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    if left_values.size == 0 or left_values.shape != right_values.shape:
        return nan_value()
    eps = 1e-12
    p = left_values + eps
    q = right_values + eps
    p = p / np.sum(p)
    q = q / np.sum(q)
    return float(np.sum(p * np.log(p / q)))


__all__ = [
    "compute_generalization_stability_metrics",
    "compute_predictability_metrics",
]
