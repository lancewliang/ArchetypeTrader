"""Phase II validation Layer 3: demonstration consistency raw metrics."""

from __future__ import annotations

import numpy as np

from ...metrics import (
    Phase2DemonstrationConsistencyMetrics,
    Phase2DemonstrationConsistencyPayload,
    Phase2LayerComputation,
)
from ._numeric import as_float_array, nan_value, safe_mean


def compute_demonstration_consistency_metrics(
    payload: Phase2DemonstrationConsistencyPayload,
    *,
    cross_entropy_to_assigned: float = float("nan"),
    kl_to_assigned_onehot: float = float("nan"),
) -> Phase2LayerComputation:
    """Compute selector consistency with Phase I assigned labels."""

    selected = np.asarray(payload.selected_code_ids, dtype=np.int64)
    assigned = np.asarray(payload.assigned_code_labels, dtype=np.int64)
    selector_returns = as_float_array(payload.selector_returns)
    assigned_returns = as_float_array(payload.assigned_label_returns)
    selected_q = as_float_array(payload.selected_q_values)
    assigned_q = as_float_array(payload.assigned_label_q_values)

    size = min(selected.size, assigned.size)
    if size <= 0:
        label_match_rate = nan_value()
        deviation_mask = np.asarray([], dtype=np.bool_)
    else:
        selected = selected[:size]
        assigned = assigned[:size]
        label_match_rate = float(np.mean((selected == assigned).astype(np.float64)))
        deviation_mask = selected != assigned

    deviation_delta = _paired_delta(selector_returns, assigned_returns, deviation_mask)
    profitable_deviation_rate = _deviation_rate(
        selector_returns,
        assigned_returns,
        deviation_mask,
        greater=True,
    )
    unprofitable_deviation_rate = _deviation_rate(
        selector_returns,
        assigned_returns,
        deviation_mask,
        greater=False,
    )
    metrics = Phase2DemonstrationConsistencyMetrics(
        label_match_rate=label_match_rate,
        cross_entropy_to_assigned=float(cross_entropy_to_assigned),
        kl_to_assigned_onehot=float(kl_to_assigned_onehot),
        label_q_margin=_q_margin_mean(selected_q, assigned_q),
        profitable_deviation_rate=profitable_deviation_rate,
        unprofitable_deviation_rate=unprofitable_deviation_rate,
        deviation_return_delta=deviation_delta,
    )
    return Phase2LayerComputation(
        layer_id=3,
        layer_name="demonstration_consistency",
        metrics=metrics,
        extra_payload={"demonstration_consistency_payload": payload},
    )


def _paired_delta(
    left: np.ndarray,
    right: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Mean left-right on masked finite pairs."""

    size = min(left.shape[0], right.shape[0], mask.shape[0])
    if size <= 0:
        return nan_value()
    valid = mask[:size] & np.isfinite(left[:size]) & np.isfinite(right[:size])
    if not np.any(valid):
        return nan_value()
    return float(np.mean(left[:size][valid] - right[:size][valid]))


def _deviation_rate(
    left: np.ndarray,
    right: np.ndarray,
    mask: np.ndarray,
    *,
    greater: bool,
) -> float:
    """Rate among deviated finite pairs."""

    size = min(left.shape[0], right.shape[0], mask.shape[0])
    if size <= 0:
        return nan_value()
    valid = mask[:size] & np.isfinite(left[:size]) & np.isfinite(right[:size])
    if not np.any(valid):
        return nan_value()
    comparison = left[:size][valid] > right[:size][valid]
    if not greater:
        comparison = left[:size][valid] < right[:size][valid]
    return float(np.mean(comparison.astype(np.float64)))


def _q_margin_mean(selected_q: np.ndarray, assigned_q: np.ndarray) -> float:
    """Mean selected-assigned Q margin over finite paired values."""

    size = min(selected_q.shape[0], assigned_q.shape[0])
    if size <= 0:
        return nan_value()
    return safe_mean(selected_q[:size] - assigned_q[:size])


__all__ = ["compute_demonstration_consistency_metrics"]
