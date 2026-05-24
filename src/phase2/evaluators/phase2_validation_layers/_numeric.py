"""Small numeric helpers for Phase II validation layer calculators."""

from __future__ import annotations

import math

import numpy as np


def nan_value() -> float:
    """Return a quiet NaN."""

    return float("nan")


def as_float_array(values: tuple[float, ...] | list[float] | np.ndarray) -> np.ndarray:
    """Convert a sequence to a 1D float64 array.

    算法:
        1. 用 ``np.asarray(values, dtype=np.float64)`` 统一数值类型；
        2. 如果输入不是 1D，则 flatten 成一维序列。

    用途:
        各 layer 的收益、Q value、history 序列统一按一维向量计算，避免调用方
        传入 shape=[N, 1] 或更高维数组时公式口径不一致。
    """

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        return array.reshape(-1)
    return array


def finite_values(values: np.ndarray) -> np.ndarray:
    """Return finite values only.

    公式:
        ``finite_values(x) = x[isfinite(x)]``。

    用途:
        validation 统计只对有限值求均值/分位数，避免 NaN/inf 污染 checkpoint
        selector 和 report。
    """

    return values[np.isfinite(values)]


def safe_mean(values: np.ndarray) -> float:
    """Mean over finite values, NaN for empty input.

    公式:
        ``mean = sum(x_i) / n``，其中 ``x_i`` 只包含有限值。

    NaN 规则:
        没有有限值时返回 NaN，让上层 rule/report 显式暴露缺失指标。
    """

    finite = finite_values(values)
    if finite.size == 0:
        return nan_value()
    return float(np.mean(finite))


def safe_median(values: np.ndarray) -> float:
    """Median over finite values, NaN for empty input.

    公式:
        ``median({x_i | isfinite(x_i)})``。

    用途:
        降低极端收益样本对主体收益判断的影响。
    """

    finite = finite_values(values)
    if finite.size == 0:
        return nan_value()
    return float(np.median(finite))


def safe_sum(values: np.ndarray) -> float:
    """Sum over finite values, NaN for empty input.

    公式:
        ``sum = Σ x_i``，其中 ``x_i`` 只包含有限值。

    用途:
        计算累计 horizon return 或总收益代理值。
    """

    finite = finite_values(values)
    if finite.size == 0:
        return nan_value()
    return float(np.sum(finite))


def safe_percentile(values: np.ndarray, percentile: float) -> float:
    """Percentile over finite values, NaN for empty input.

    公式:
        ``percentile({x_i | isfinite(x_i)}, p)``。

    用途:
        Layer 1 用 p=5 估计左尾风险，避免单个最小值过度敏感。
    """

    finite = finite_values(values)
    if finite.size == 0:
        return nan_value()
    return float(np.percentile(finite, percentile))


def safe_std(values: np.ndarray) -> float:
    """Population std over finite values, NaN for empty input.

    公式:
        ``std = sqrt(mean((x_i - mean(x))^2))``，使用 numpy population std
        ``ddof=0``。

    用途:
        作为 Sharpe-like、Q scale stability 等风险/波动指标的分母或诊断项。
    """

    finite = finite_values(values)
    if finite.size == 0:
        return nan_value()
    return float(np.std(finite))


def safe_ratio(numerator: float, denominator: float) -> float:
    """Return numerator / denominator with NaN for invalid denominator.

    公式:
        ``ratio = numerator / denominator``。

    防御:
        任一输入非有限，或 ``abs(denominator) <= 1e-12`` 时返回 NaN，避免把
        无意义的极大比值写入 validation result。
    """

    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return nan_value()
    if abs(denominator) <= 1e-12:
        return nan_value()
    return float(numerator / denominator)
