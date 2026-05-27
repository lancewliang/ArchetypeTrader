"""Phase II validation Layer 1: selector profitability raw metrics."""

from __future__ import annotations

import numpy as np

from ...metrics import (
    Phase2LayerComputation,
    Phase2SelectorProfitabilityMetrics,
    Phase2SelectorProfitabilityPayload,
)
from src.utils._numeric import (
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
    """Compute selector return, risk, fee and turnover metrics.

    算法:
        1. 将 selector 净收益、gross return、fee、turnover 转成 float64 一维数组；
        2. 只在有限值上计算均值、分位数和标准差；
        3. 从净收益序列中派生收益质量指标；
        4. 从 gross return 和 fee 派生成本拖累；
        5. 返回 Layer 1 raw metrics，不在这里应用 hard gate。

    核心公式:
        - ``mean_return = mean(r)``
        - ``median_return = median(r)``
        - ``total_return = Σ r_i``
        - ``win_rate = mean(1[r_i > 0])``
        - ``loss_rate = mean(1[r_i < 0])``
        - ``sharpe_like = mean_return / std(r)``
        - ``downside_sharpe_like = mean_return / std({r_i | r_i < 0})``
        - ``p05_return = percentile(r, 5)``
        - ``fee_drag_ratio = mean_fee / abs(mean_gross_return)``

    说明:
        本层直接衡量 selector greedy policy 是否赚钱以及收益是否被波动、左尾
        或手续费吞噬。收益、Sharpe、win rate 越大越好；fee drag、turnover 和
        loss rate 越小越稳。
    """

    returns = as_float_array(payload.selector_returns)
    gross_returns = as_float_array(payload.selector_gross_returns)
    fees = as_float_array(payload.selector_fees)
    turnover = as_float_array(payload.selector_turnover)

    # finite_returns 是所有收益类统计的主体样本集合；NaN/inf 由 Layer 0 负责
    # 标记，这里只保证聚合公式不被非有限值污染。
    finite_returns = finite_values(returns)
    mean_return = safe_mean(returns)
    return_std = safe_std(returns)
    # downside Sharpe 只用亏损样本估计下行波动，避免盈利侧波动抬高风险分母。
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
        selector_profitability_payload=payload,
    )


def _rate(mask: np.ndarray) -> float:
    """Return mean boolean rate, NaN for empty input.

    公式:
        ``rate = mean(1[condition_i])``。

    用途:
        用同一个 helper 计算 win_rate、loss_rate 等布尔比例；空输入返回 NaN，
        避免把没有样本的比例误写成 0。
    """

    if mask.size == 0:
        return nan_value()
    return float(np.mean(mask.astype(np.float64)))


__all__ = ["compute_selector_profitability_metrics"]
