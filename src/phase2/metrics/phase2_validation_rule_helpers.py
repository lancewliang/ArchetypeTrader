"""Shared helpers for Phase II validation rules."""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .phase2_metric_results import (
        MetricDirection,
        MetricSeverity,
        MetricThresholdValue,
        Phase2LayerResult,
        Phase2MetricResult,
    )


def _is_missing(value: Any) -> bool:
    """判断指标是否缺失或不可计算。"""

    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def _finite_distance(value: float) -> float | None:
    """只保留有限阈值距离。"""

    return float(value) if math.isfinite(value) else None


def _build_layer_result(
    *,
    layer_id: int,
    name: str,
    metrics: Sequence[Phase2MetricResult],
    force_passed: bool | None = None,
) -> Phase2LayerResult:
    """根据本层 metric result 聚合 layer result。"""

    from .phase2_metric_results import Phase2LayerResult

    metric_tuple = tuple(metrics)
    passed = all(metric.passed for metric in metric_tuple)
    if force_passed is not None:
        passed = bool(force_passed)
    return Phase2LayerResult(
        layer_id=layer_id,
        name=name,
        passed=passed,
        metrics=metric_tuple,
    )


def _metric_result(
    *,
    name: str,
    value: int | float | str | bool | None,
    threshold: str,
    passed: bool,
    layer: str,
    message: str,
    severity_when_failed: MetricSeverity = "fail",
    threshold_value: MetricThresholdValue = None,
    direction: MetricDirection | None = None,
    distance_to_threshold: float | None = None,
) -> Phase2MetricResult:
    """创建 hard gate 或 warning metric result。"""

    from .phase2_metric_results import Phase2MetricResult

    if _is_missing(value):
        return Phase2MetricResult(
            name=name,
            value=None,
            threshold=threshold,
            severity="skip" if severity_when_failed == "fail" else "warn",
            passed=severity_when_failed != "fail",
            layer=layer,
            message=f"{message}；指标缺失或不可计算",
            threshold_value=threshold_value,
            direction=direction,
            distance_to_threshold=None,
        )
    return Phase2MetricResult(
        name=name,
        value=value,
        threshold=threshold,
        severity="pass" if passed else severity_when_failed,
        passed=passed if severity_when_failed == "fail" else True,
        layer=layer,
        message=message,
        threshold_value=threshold_value,
        direction=direction,
        distance_to_threshold=distance_to_threshold,
    )


def _ge(
    *,
    name: str,
    value: float,
    threshold_value: float,
    layer: str,
    message: str,
    severity_when_failed: MetricSeverity = "fail",
) -> Phase2MetricResult:
    """构造“越大越好且需大于等于阈值”的结果。"""

    return _metric_result(
        name=name,
        value=value,
        threshold=f">= {threshold_value:g}",
        passed=value >= threshold_value if not _is_missing(value) else False,
        layer=layer,
        message=message,
        severity_when_failed=severity_when_failed,
        threshold_value=float(threshold_value),
        direction="greater_is_better",
        distance_to_threshold=(
            _finite_distance(value - threshold_value)
            if not _is_missing(value)
            else None
        ),
    )


def _gt(
    *,
    name: str,
    value: float,
    threshold_value: float,
    layer: str,
    message: str,
    severity_when_failed: MetricSeverity = "fail",
) -> Phase2MetricResult:
    """构造“越大越好且需严格大于阈值”的结果。"""

    return _metric_result(
        name=name,
        value=value,
        threshold=f"> {threshold_value:g}",
        passed=value > threshold_value if not _is_missing(value) else False,
        layer=layer,
        message=message,
        severity_when_failed=severity_when_failed,
        threshold_value=float(threshold_value),
        direction="greater_is_better",
        distance_to_threshold=(
            _finite_distance(value - threshold_value)
            if not _is_missing(value)
            else None
        ),
    )


def _le(
    *,
    name: str,
    value: float | int,
    threshold_value: float | int,
    layer: str,
    message: str,
    severity_when_failed: MetricSeverity = "fail",
) -> Phase2MetricResult:
    """构造“越小越好且需小于等于阈值”的结果。"""

    return _metric_result(
        name=name,
        value=value,
        threshold=f"<= {threshold_value:g}",
        passed=value <= threshold_value if not _is_missing(value) else False,
        layer=layer,
        message=message,
        severity_when_failed=severity_when_failed,
        threshold_value=float(threshold_value),
        direction="less_is_better",
        distance_to_threshold=(
            _finite_distance(float(threshold_value) - float(value))
            if not _is_missing(value)
            else None
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
    severity_when_failed: MetricSeverity = "fail",
) -> Phase2MetricResult:
    """构造“落在闭区间内最好”的结果。"""

    return _metric_result(
        name=name,
        value=value,
        threshold=f"[{lower:g}, {upper:g}]",
        passed=lower <= value <= upper if not _is_missing(value) else False,
        layer=layer,
        message=message,
        severity_when_failed=severity_when_failed,
        threshold_value=(float(lower), float(upper)),
        direction="between",
        distance_to_threshold=(
            _finite_distance(min(value - lower, upper - value))
            if not _is_missing(value)
            else None
        ),
    )


def _eq_bool(
    *,
    name: str,
    value: bool,
    expected: bool,
    layer: str,
    message: str,
    severity_when_failed: MetricSeverity = "fail",
) -> Phase2MetricResult:
    """构造 bool 等值检查结果。"""

    return _metric_result(
        name=name,
        value=bool(value),
        threshold=str(expected).lower(),
        passed=bool(value) is bool(expected),
        layer=layer,
        message=message,
        severity_when_failed=severity_when_failed,
        threshold_value=bool(expected),
        direction="equal",
        distance_to_threshold=None,
    )


__all__ = [
    "_between",
    "_build_layer_result",
    "_eq_bool",
    "_ge",
    "_gt",
    "_is_missing",
    "_le",
]
