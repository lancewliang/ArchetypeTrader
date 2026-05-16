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
    threshold_value: float | tuple[float, float] | None = None,
    direction: str | None = None,
    distance_to_threshold: float | None = None,
    passed: bool,
    layer: str,
    message: str,
    direction_message: str | None = None,
) -> Phase1MetricResult:
    """创建 metric result，并把缺失 hard gate 统一标记为 skip-as-fail。"""

    full_message = _append_direction(message, direction_message)
    if _is_missing(value):
        return Phase1MetricResult(
            name=name,
            value=None,
            threshold=threshold,
            severity="skip",
            passed=False,
            layer=layer,
            message=f"{full_message}；指标缺失或不可计算，按 hard gate 失败处理",
            threshold_value=threshold_value,
            direction=direction,  # type: ignore[arg-type]
            distance_to_threshold=None,
        )
    return Phase1MetricResult(
        name=name,
        value=value,
        threshold=threshold,
        severity="pass" if passed else "fail",
        passed=passed,
        layer=layer,
        message=full_message,
        threshold_value=threshold_value,
        direction=direction,  # type: ignore[arg-type]
        distance_to_threshold=distance_to_threshold,
    )


def _append_direction(message: str, direction_message: str | None) -> str:
    """把指标方向说明追加到 report message。"""

    if direction_message is None:
        return message
    return f"{message}；{direction_message}"


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
        threshold_value=float(threshold_value),
        direction="greater_is_better",
        distance_to_threshold=(
            float(value - threshold_value) if not _is_missing(value) else None
        ),
        passed=value > threshold_value if not _is_missing(value) else False,
        layer=layer,
        message=message,
        direction_message=(
            "指标方向：越大越好；变大表示质量、收益或覆盖度提升，"
            "变小表示更接近失败阈值"
        ),
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
        threshold_value=float(threshold_value),
        direction="greater_is_better",
        distance_to_threshold=(
            float(value - threshold_value) if not _is_missing(value) else None
        ),
        passed=value >= threshold_value if not _is_missing(value) else False,
        layer=layer,
        message=message,
        direction_message=(
            "指标方向：越大越好；变大表示质量、收益或覆盖度提升，"
            "变小表示更接近失败阈值"
        ),
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
        threshold_value=float(threshold_value),
        direction="less_is_better",
        distance_to_threshold=(
            float(threshold_value - value) if not _is_missing(value) else None
        ),
        passed=value <= threshold_value if not _is_missing(value) else False,
        layer=layer,
        message=message,
        direction_message=(
            "指标方向：越小越好；变大表示风险、误差或异常占比上升，"
            "变小表示更健康"
        ),
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
        threshold_value=(float(lower), float(upper)),
        direction="between",
        distance_to_threshold=(
            float(min(value - lower, upper - value))
            if not _is_missing(value)
            else None
        ),
        passed=lower <= value <= upper if not _is_missing(value) else False,
        layer=layer,
        message=message,
        direction_message=(
            "指标方向：落在目标区间内最好；变大表示更靠近上限，"
            "变小表示更靠近下限，低于下限或高于上限都可能失败"
        ),
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
