"""组合收益与风险指标: net_return / sharpe / sortino / MDD / calmar / turnover / equity curve。

设计文档锚点: Phase II 执行计划 §Step 6。

复用 Phase I 的 risk metrics 基础函数，扩展 Phase II 特有的 portfolio 指标。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_EPS = 1e-10


@dataclass
class EquityCurveSummary:
    """equity_curve_summary 最小结构。"""
    start_value: float = 1.0
    end_value: float = 0.0
    max_value: float = 0.0
    min_value: float = 0.0
    max_drawdown_start_step: int = 0
    max_drawdown_end_step: int = 0
    peak_step: int = 0
    valley_step: int = 0
    per_horizon_cumulative_pnl: List[float] = field(default_factory=list)


def compute_net_return(horizon_rewards: List[float]) -> float:
    """计算总净收益。"""
    return sum(horizon_rewards)


def compute_sharpe(
    step_returns: List[float],
    annualization_factor: int = 525600,
) -> float:
    """计算年化 Sharpe ratio。"""
    if len(step_returns) < 2:
        return 0.0
    mean = sum(step_returns) / len(step_returns)
    var = sum((r - mean) ** 2 for r in step_returns) / (len(step_returns) - 1)
    std = math.sqrt(max(var, 0.0))
    if std < _EPS:
        return 0.0
    return mean / std * math.sqrt(annualization_factor)


def compute_sortino(
    step_returns: List[float],
    annualization_factor: int = 525600,
) -> float:
    """计算年化 Sortino ratio。"""
    if len(step_returns) < 2:
        return 0.0
    mean = sum(step_returns) / len(step_returns)
    downside = [min(r, 0.0) for r in step_returns]
    down_var = sum(d ** 2 for d in downside) / (len(step_returns) - 1)
    down_std = math.sqrt(max(down_var, 0.0))
    if down_std < _EPS:
        return 0.0
    return mean / down_std * math.sqrt(annualization_factor)


def compute_max_drawdown(equity_curve: List[float]) -> float:
    """计算最大回撤。"""
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for v in equity_curve:
        if v > peak:
            peak = v
        dd = (peak - v) / max(abs(peak), _EPS)
        if dd > max_dd:
            max_dd = dd
    return max_dd


def compute_calmar(annual_return: float, max_drawdown: float) -> float:
    """计算 Calmar ratio。"""
    if max_drawdown < _EPS:
        return 0.0
    return annual_return / max_drawdown


def compute_turnover(actions: List[int]) -> float:
    """计算 archetype 切换频率（turnover）。"""
    if len(actions) < 2:
        return 0.0
    switches = sum(1 for i in range(1, len(actions)) if actions[i] != actions[i - 1])
    return switches / (len(actions) - 1)


def compute_boundary_cost(boundary_costs: List[float]) -> float:
    """计算总边界换仓成本。"""
    return sum(boundary_costs)


def build_equity_curve_summary(
    horizon_rewards: List[float],
) -> EquityCurveSummary:
    """构建 equity_curve_summary。"""
    if not horizon_rewards:
        return EquityCurveSummary()

    cumulative = []
    running = 1.0
    for r in horizon_rewards:
        running += r
        cumulative.append(running)

    max_val = max(cumulative)
    min_val = min(cumulative)
    peak_step = cumulative.index(max_val)
    valley_step = cumulative.index(min_val)

    # 找最大回撤区间
    peak = cumulative[0]
    max_dd = 0.0
    dd_start = 0
    dd_end = 0
    current_peak_idx = 0
    for i, v in enumerate(cumulative):
        if v > peak:
            peak = v
            current_peak_idx = i
        dd = (peak - v) / max(abs(peak), _EPS)
        if dd > max_dd:
            max_dd = dd
            dd_start = current_peak_idx
            dd_end = i

    return EquityCurveSummary(
        start_value=1.0,
        end_value=cumulative[-1] if cumulative else 1.0,
        max_value=max_val,
        min_value=min_val,
        max_drawdown_start_step=dd_start,
        max_drawdown_end_step=dd_end,
        peak_step=peak_step,
        valley_step=valley_step,
        per_horizon_cumulative_pnl=cumulative,
    )
