"""Phase II validation Layer 1: selector profitability raw metrics."""

from __future__ import annotations

import numpy as np

from ...metrics import (
    Phase2LayerComputation,
    Phase2SelectorProfitabilityMetrics,
    Phase2SelectorProfitabilityPayload,
)
from ._numeric import (
    as_float_array,
    finite_values,
    nan_value,
    safe_mean,
    safe_median,
    safe_percentile,
    safe_ratio,
    safe_std,
    safe_sum,
)


def compute_selector_profitability_metrics(
    payload: Phase2SelectorProfitabilityPayload,
) -> Phase2LayerComputation:
    """Compute selector return, risk, fee and turnover metrics."""

    returns = as_float_array(payload.selector_returns)
    gross_returns = as_float_array(payload.selector_gross_returns)
    fees = as_float_array(payload.selector_fees)
    turnover = as_float_array(payload.selector_turnover)

    finite_returns = finite_values(returns)
    mean_return = safe_mean(returns)
    return_std = safe_std(returns)
    negative_returns = finite_returns[finite_returns < 0.0]
    downside_std = safe_std(negative_returns)
    mean_gross_return = safe_mean(gross_returns)
    mean_fee = safe_mean(fees)

    metrics = Phase2SelectorProfitabilityMetrics(
        mean_return=mean_return,
        median_return=safe_median(returns),
        total_return=safe_sum(returns),
        win_rate=_rate(finite_returns > 0.0),
        sharpe_like=safe_ratio(mean_return, return_std),
        downside_sharpe_like=safe_ratio(mean_return, downside_std),
        p05_return=safe_percentile(returns, 5.0),
        loss_rate=_rate(finite_returns < 0.0),
        mean_gross_return=mean_gross_return,
        mean_fee=mean_fee,
        fee_drag_ratio=safe_ratio(mean_fee, abs(mean_gross_return)),
        mean_turnover=safe_mean(turnover),
    )
    return Phase2LayerComputation(
        layer_id=1,
        layer_name="selector_profitability",
        metrics=metrics,
        extra_payload={"selector_profitability_payload": payload},
    )


def _rate(mask: np.ndarray) -> float:
    """Return mean boolean rate, NaN for empty input."""

    if mask.size == 0:
        return nan_value()
    return float(np.mean(mask.astype(np.float64)))


__all__ = ["compute_selector_profitability_metrics"]
