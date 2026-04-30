"""风险调整收益指标 (Sharpe / Sortino / MDD / Calmar).

设计文档锚点: §4.11 与 §6.8。
"""
from __future__ import annotations

import math
from typing import List, Sequence


DEFAULT_ANNUALIZATION_FACTOR = 525_600  # 365 * 24 * 60 (分钟级)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(max(var, 0.0))


def sharpe_ratio(step_returns, annualization_factor: int = DEFAULT_ANNUALIZATION_FACTOR) -> float:
    """``mean / std * sqrt(annualization)``。

    实现注意
    --------
    - ``std == 0`` 时返回 ``0.0`` 而非 ``inf``，避免被 ``selection_policy`` 误判为最优。
    - 默认年化因子 ``525_600 = 365 * 24 * 60``，对应分钟级数据。
    """
    vals = list(step_returns)
    s = _std(vals)
    if s < 1e-12:
        # std=0 时返回 0；避免 inf。
        return 0.0
    return _mean(vals) / s * math.sqrt(annualization_factor)


def sortino_ratio(step_returns, annualization_factor: int = DEFAULT_ANNUALIZATION_FACTOR) -> float:
    """只惩罚下行波动（``negative returns`` 的标准差）。

    全部 returns ≥ 0 时返回 ``0.0`` 而非 ``inf``，避免给 selection_policy 错误信号。
    """
    vals = list(step_returns)
    if not vals:
        return 0.0
    downsides = [v for v in vals if v < 0]
    if not downsides:
        # 全部 >= 0 → 下行波动为 0，按照惯例返回 0 而不是 inf。
        return 0.0
    s = _std(downsides)
    if s < 1e-12:
        return 0.0
    return _mean(vals) / s * math.sqrt(annualization_factor)


def equity_curve_from_step_returns(step_returns: Sequence[float]) -> List[float]:
    """累计净值序列；初值 = 1.0；每步累加（不复利）。

    分钟级 returns 量级很小，使用累加近似复利更稳定（避免 1.000001^N 累计偏差）。
    """
    curve: List[float] = []
    cum = 1.0
    for r in step_returns:
        cum += float(r)
        curve.append(cum)
    return curve


def max_drawdown(equity_curve: Sequence[float]) -> float:
    """``max((peak - trough) / peak)``，返回非负浮点数。

    单调上升的曲线返回 0；某段下行后再创新高时，drawdown 取整段最大值。
    """
    peak = float("-inf")
    mdd = 0.0
    for v in equity_curve:
        peak = max(peak, v)
        if peak <= 0:
            continue
        dd = (peak - v) / peak
        mdd = max(mdd, dd)
    return mdd


def calmar_ratio(annual_return: float, mdd: float) -> float:
    """``annual_return / max(mdd, eps)``；``mdd == 0`` 时返回 ``0.0`` 防 inf。"""
    if mdd <= 1e-12:
        return 0.0
    return annual_return / mdd
