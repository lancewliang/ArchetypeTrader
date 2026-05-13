"""Shared helpers for Phase I validation hard gate rules."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from .phase1_metric_results import Phase1LayerResult, Phase1MetricResult


def _is_missing(value: Any) -> bool:
    """判断 hard gate 指标是否缺失或不可计算。"""

    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def _metric_result(
    *,
    name: str,
    value: int | float | str | bool | None,
    threshold: str,
    passed: bool,
    layer: str,
    message: str,
) -> Phase1MetricResult:
    """创建 metric result，并把缺失 hard gate 统一标记为 skip-as-fail。"""

    if _is_missing(value):
        return Phase1MetricResult(
            name=name,
            value=None,
            threshold=threshold,
            severity="skip",
            passed=False,
            layer=layer,
            message=f"{message}；指标缺失或不可计算，按 hard gate 失败处理",
        )
    return Phase1MetricResult(
        name=name,
        value=value,
        threshold=threshold,
        severity="pass" if passed else "fail",
        passed=passed,
        layer=layer,
        message=message,
    )


def _gt(
    *,
    name: str,
    value: float,
    threshold_value: float,
    layer: str,
    message: str,
) -> Phase1MetricResult:
    """构造“必须严格大于阈值”的 hard gate 结果。"""

    return _metric_result(
        name=name,
        value=value,
        threshold=f"> {threshold_value:g}",
        passed=value > threshold_value if not _is_missing(value) else False,
        layer=layer,
        message=message,
    )


def _ge(
    *,
    name: str,
    value: float,
    threshold_value: float,
    layer: str,
    message: str,
) -> Phase1MetricResult:
    """构造“必须大于或等于阈值”的 hard gate 结果。"""

    return _metric_result(
        name=name,
        value=value,
        threshold=f">= {threshold_value:g}",
        passed=value >= threshold_value if not _is_missing(value) else False,
        layer=layer,
        message=message,
    )


def _le(
    *,
    name: str,
    value: float | int,
    threshold_value: float | int,
    layer: str,
    message: str,
) -> Phase1MetricResult:
    """构造“必须小于或等于上限”的 hard gate 结果。"""

    return _metric_result(
        name=name,
        value=value,
        threshold=f"<= {threshold_value:g}",
        passed=value <= threshold_value if not _is_missing(value) else False,
        layer=layer,
        message=message,
    )


def _between(
    *,
    name: str,
    value: float,
    lower: float,
    upper: float,
    layer: str,
    message: str,
) -> Phase1MetricResult:
    """构造“必须落在闭区间内”的 hard gate 结果。"""

    return _metric_result(
        name=name,
        value=value,
        threshold=f"[{lower:g}, {upper:g}]",
        passed=lower <= value <= upper if not _is_missing(value) else False,
        layer=layer,
        message=message,
    )


def _build_layer_result(
    *,
    layer_id: int,
    name: str,
    metrics: Sequence[Phase1MetricResult],
) -> Phase1LayerResult:
    """根据本层所有 metric result 聚合 layer result。"""

    metric_tuple = tuple(metrics)
    return Phase1LayerResult(
        layer_id=layer_id,
        name=name,
        passed=all(metric.passed for metric in metric_tuple),
        metrics=metric_tuple,
    )
