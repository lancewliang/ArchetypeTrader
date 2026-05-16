"""Phase I codebook validation report template context builder."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from ..metrics import (
    Phase1CodeDiagnostic,
    Phase1LayerResult,
    Phase1MetricResult,
)


JsonObject = dict[str, Any]


def _format_value(value: Any) -> str:
    """格式化 HTML 表格中的指标值。"""

    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return f"{value:.6g}"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, Mapping) and "total_score" in value:
        return _format_value(value.get("total_score"))
    return str(value)


def _badge_class(severity: str | bool) -> str:
    """将状态映射为 HTML badge class。"""

    if isinstance(severity, bool):
        return "pass" if severity else "fail"
    if severity in {"pass", "fail", "warn", "skip"}:
        return severity
    return "warn"


@dataclass(frozen=True)
class Phase1CodebookReportContextBuilder:
    """Phase I codebook validation report 模板上下文构建器。"""

    title: str = "Phase I Codebook Validation Report"

    def build(self, payload: Mapping[str, Any]) -> JsonObject:
        """把 report payload 转成模板使用的展示模型。"""

        validation = payload["validation"]
        summary = payload["summary"]
        report = payload["report"]
        config = payload.get("config", {})
        layers = tuple(
            Phase1LayerResult.from_dict(layer)
            for layer in validation.get("layers", ())
        )
        code_diagnostics = tuple(
            Phase1CodeDiagnostic.from_dict(item)
            for item in validation.get("code_diagnostics", ())
        )
        drift_diagnostics = {
            str(name): Phase1MetricResult.from_dict(item)
            for name, item in validation.get("drift_diagnostics", {}).items()
        }

        passed = bool(summary.get("passed", False))
        failed_layers = summary.get("failed_layers", [])
        failed_text = ", ".join(str(layer) for layer in failed_layers) or "-"
        return {
            "page_title": str(report.get("title", self.title)),
            "header_title": self.title,
            "report": {
                "generated_at": str(report.get("generated_at", "-")),
                "schema": str(report.get("schema", "-")),
            },
            "summary": {
                "checkpoint_id": str(summary.get("checkpoint_id", "-")),
                "stage": str(summary.get("stage", "-")),
                "epoch": str(summary.get("epoch", "-")),
                "score": _format_value(summary.get("score")),
                "failed_layers": failed_text,
                "code_diagnostic_count": str(summary.get("code_diagnostic_count", 0)),
                "badge_class": _badge_class(passed),
                "status_label": "PASS" if passed else "FAIL",
            },
            "layers": [self._build_layer_context(layer) for layer in layers],
            "code_diagnostics": [
                self._build_code_diagnostic_context(item)
                for item in code_diagnostics
            ],
            "code_distribution": self._build_code_distribution_context(
                validation.get("metrics", {})
            ),
            "tie_breaker_rows": self._build_mapping_rows(
                validation.get("tie_breaker_metrics", {})
            ),
            "score_breakdown_rows": self._build_score_breakdown_rows(
                validation.get("score")
            ),
            "drift_diagnostics": [
                self._build_metric_context(metric)
                for metric in drift_diagnostics.values()
            ],
            "config_rows": self._build_mapping_rows(config),
        }

    def _build_layer_context(self, layer: Phase1LayerResult) -> JsonObject:
        """构建单个 validation layer 的模板上下文。"""

        failed = sum(1 for metric in layer.metrics if not metric.passed)
        return {
            "layer_id": str(layer.layer_id),
            "name": layer.name,
            "badge_class": _badge_class(layer.passed),
            "status_label": "PASS" if layer.passed else "FAIL",
            "metric_count": str(len(layer.metrics)),
            "failed_count": str(failed),
            "metrics": [
                self._build_metric_context(metric)
                for metric in layer.metrics
            ],
        }

    def _build_metric_context(self, metric: Phase1MetricResult) -> JsonObject:
        """构建单个 metric result 的模板上下文。"""

        return {
            "name": metric.name,
            "value": _format_value(metric.value),
            "threshold": metric.threshold,
            "badge_class": _badge_class(metric.severity),
            "severity_label": metric.severity.upper(),
            "message": metric.message,
        }

    def _build_code_diagnostic_context(
        self,
        item: Phase1CodeDiagnostic,
    ) -> JsonObject:
        """构建单个 code diagnostic 的模板上下文。"""

        return {
            "code_id": str(item.code_id),
            "support": str(item.support),
            "occupancy": _format_value(item.occupancy),
            "dominant_morphology": str(item.dominant_morphology or "-"),
            "dominant_morphology_ratio": _format_value(
                item.dominant_morphology_ratio
            ),
            "dominant_motif": str(item.dominant_motif or "-"),
            "dominant_motif_ratio": _format_value(item.dominant_motif_ratio),
            "dominant_pair": str(item.dominant_pair or "-"),
            "decoded_mean_advantage": _format_value(item.decoded_mean_advantage),
            "retention_ratio": _format_value(item.retention_ratio),
            "status": item.status,
        }

    def _build_code_distribution_context(
        self,
        metrics: Mapping[str, Any],
    ) -> list[JsonObject]:
        """构建 codebook 使用分布的模板上下文。"""

        vq_metrics = metrics.get("vq_internal", {})
        if not isinstance(vq_metrics, Mapping):
            return []

        raw_distribution = vq_metrics.get("code_distribution", ())
        raw_active_codes = vq_metrics.get("active_codes", ())
        try:
            distribution = [float(value) for value in raw_distribution]
        except (TypeError, ValueError):
            return []
        try:
            active_codes = {int(code_id) for code_id in raw_active_codes}
        except (TypeError, ValueError):
            active_codes = set()

        return [
            {
                "code_id": str(code_id),
                "occupancy": _format_value(occupancy),
                "occupancy_percent": (
                    f"{occupancy * 100:.3g}%"
                    if math.isfinite(occupancy)
                    else _format_value(occupancy)
                ),
                "bar_width": (
                    f"{max(0.0, min(100.0, occupancy * 100.0)):.3g}%"
                    if math.isfinite(occupancy)
                    else "0%"
                ),
                "active": code_id in active_codes,
                "badge_class": "pass" if code_id in active_codes else "warn",
                "status_label": "ACTIVE" if code_id in active_codes else "INACTIVE",
            }
            for code_id, occupancy in enumerate(distribution)
        ]

    def _build_mapping_rows(self, payload: Mapping[str, Any]) -> list[JsonObject]:
        """构建普通 key-value 表格上下文。"""

        return [
            {"key": str(key), "value": _format_value(value)}
            for key, value in payload.items()
        ]

    def _build_score_breakdown_rows(self, payload: Any) -> list[JsonObject]:
        """构建综合 score 子项拆解表格上下文。"""

        if not isinstance(payload, Mapping):
            return []
        components = payload.get("components", ())
        if not isinstance(components, list | tuple):
            return []
        rows: list[JsonObject] = []
        for component in components:
            if not isinstance(component, Mapping):
                continue
            rows.append(
                {
                    "name": str(component.get("name", "-")),
                    "value": _format_value(component.get("value")),
                    "weight": _format_value(component.get("weight")),
                    "weighted_value": _format_value(
                        component.get("weighted_value")
                    ),
                }
            )
        return rows


__all__ = ["Phase1CodebookReportContextBuilder"]
