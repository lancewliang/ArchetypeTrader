"""Shared helpers for Phase I validation scoring."""

from __future__ import annotations

import math


def _clip01(value: float) -> float:
    """把数值截断到 [0, 1]，不可计算值按 0 处理。"""

    if math.isnan(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _positive_score(value: float, scale: float = 1.0) -> float:
    """把正向无界指标压缩到 [0, 1]。"""

    if math.isnan(value) or value <= 0:
        return 0.0
    return _clip01(value / (abs(value) + scale))


def _threshold_progress(value: float, threshold: float) -> float:
    """把“越高越好且有最低阈值”的比例指标归一化。"""

    if math.isnan(value):
        return 0.0
    if threshold <= 0:
        return _clip01(value)
    return _clip01(value / threshold)


def _inverse_ratio_score(value: float, maximum: float) -> float:
    """把“越低越好且有上限”的比例指标归一化。"""

    if math.isnan(value):
        return 0.0
    if maximum <= 0:
        return 1.0 if value <= 0 else 0.0
    return _clip01(1.0 - value / maximum)


def _accuracy_window_score(value: float, lower: float, upper: float = 1.0) -> float:
    """把准确率按指定合格窗口映射到 [0, 1]。"""

    if math.isnan(value) or value <= lower:
        return 0.0
    if upper <= lower:
        return 1.0
    return _clip01((value - lower) / (upper - lower))
