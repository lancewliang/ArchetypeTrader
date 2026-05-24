"""Phase II validation Layer 2: baseline uplift raw metrics."""

from __future__ import annotations

import numpy as np

from ...metrics import (
    Phase2BaselineUpliftMetrics,
    Phase2BaselineUpliftPayload,
    Phase2LayerComputation,
)
from ._numeric import as_float_array, finite_values, nan_value, safe_mean, safe_ratio


def compute_baseline_uplift_metrics(
    payload: Phase2BaselineUpliftPayload,
) -> Phase2LayerComputation:
    """Compute selector uplift versus assigned-label, random and oracle baselines."""

    selector = as_float_array(payload.selector_returns)
    assigned = as_float_array(payload.assigned_label_returns)
    random = as_float_array(payload.random_returns)
    oracle = as_float_array(payload.oracle_returns)

    selector_mean = safe_mean(selector)
    assigned_mean = safe_mean(assigned)
    random_mean = safe_mean(random)
    oracle_mean = safe_mean(oracle)
    uplift_vs_assigned = selector_mean - assigned_mean
    uplift_vs_random = selector_mean - random_mean

    metrics = Phase2BaselineUpliftMetrics(
        assigned_mean_return=assigned_mean,
        random_mean_return=random_mean,
        oracle_mean_return=oracle_mean,
        uplift_vs_assigned=uplift_vs_assigned,
        uplift_vs_random=uplift_vs_random,
        relative_uplift_vs_assigned=safe_ratio(
            uplift_vs_assigned,
            abs(assigned_mean),
        ),
        oracle_capture_ratio=safe_ratio(selector_mean, oracle_mean),
        regret_to_oracle=oracle_mean - selector_mean,
        beat_assigned_rate=_paired_beat_rate(selector, assigned),
        beat_random_rate=_paired_beat_rate(selector, random),
    )
    return Phase2LayerComputation(
        layer_id=2,
        layer_name="baseline_uplift",
        metrics=metrics,
        extra_payload={"baseline_uplift_payload": payload},
    )


def _paired_beat_rate(left: np.ndarray, right: np.ndarray) -> float:
    """Return finite paired rate of left > right."""

    size = min(left.shape[0], right.shape[0])
    if size <= 0:
        return nan_value()
    left_values = left[:size]
    right_values = right[:size]
    valid = np.isfinite(left_values) & np.isfinite(right_values)
    if not np.any(valid):
        return nan_value()
    return float(np.mean((left_values[valid] > right_values[valid]).astype(np.float64)))


__all__ = ["compute_baseline_uplift_metrics"]
