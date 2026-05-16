"""Phase I codebook validation report template context builder."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from ..metrics import (
    Phase1CodeDiagnostic,
    Phase1LayerResult,
    Phase1MetricResult,
    Phase1OracleProfitabilityPayload,
    Phase1RiskFinding,
    Phase1ValidationScore,
    Phase1VQInternalPayload,
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
        artifacts = payload.get("artifacts", {})
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
        risk_findings = tuple(
            Phase1RiskFinding.from_dict(item)
            for item in validation.get("risk_findings", ())
        )
        vq_internal_payload = self._vq_internal_payload(
            validation.get("vq_internal_payload")
        )
        oracle_profitability_payload = self._oracle_profitability_payload(
            validation.get("oracle_profitability_payload")
        )
        validation_score = self._validation_score(validation.get("score"))

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
                "risk_finding_count": str(summary.get("risk_finding_count", 0)),
                "badge_class": _badge_class(passed),
                "status_label": "PASS" if passed else "FAIL",
            },
            "layers": [self._build_layer_context(layer) for layer in layers],
            "code_diagnostics": [
                self._build_code_diagnostic_context(item)
                for item in code_diagnostics
            ],
            "oracle_profitability_kpis": self._build_oracle_profitability_kpis(
                validation.get("metrics", {})
            ),
            "per_code_profit_series": self._build_per_code_profit_series(
                code_diagnostics,
                oracle_profitability_payload,
            ),
            "code_distribution": self._build_code_distribution_context(
                vq_internal_payload
            )
            if vq_internal_payload is not None
            else [],
            "tie_breaker_rows": self._build_mapping_rows(
                validation.get("tie_breaker_metrics", {})
            ),
            "score_breakdown_rows": self._build_score_breakdown_rows(
                validation_score
            )
            if validation_score is not None
            else [],
            "drift_diagnostics": [
                self._build_metric_context(metric)
                for metric in drift_diagnostics.values()
            ],
            "risk_findings": [
                self._build_risk_finding_context(finding)
                for finding in risk_findings
            ],
            "config_rows": self._build_mapping_rows(config),
            "artifact_rows": self._build_mapping_rows(artifacts),
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
            "threshold_value": _format_value(metric.threshold_value),
            "direction": str(metric.direction or "-"),
            "distance_to_threshold": _format_value(metric.distance_to_threshold),
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
            "morphology_lift": _format_value(item.morphology_lift),
            "dominant_motif": str(item.dominant_motif or "-"),
            "dominant_motif_ratio": _format_value(item.dominant_motif_ratio),
            "dominant_pair": str(item.dominant_pair or "-"),
            "dominant_pair_ratio": _format_value(item.dominant_pair_ratio),
            "decoded_mean_advantage": _format_value(item.decoded_mean_advantage),
            "decoded_win_rate": _format_value(item.decoded_win_rate),
            "retention_ratio": _format_value(item.retention_ratio),
            "fee_drag": _format_value(item.fee_drag),
            "status": item.status,
            "badge_class": _badge_class(item.status),
        }

    def _build_code_distribution_context(
        self,
        payload: Phase1VQInternalPayload,
    ) -> list[JsonObject]:
        """构建 codebook 使用分布的模板上下文。"""

        distribution = [float(value) for value in payload.code_distribution]
        active_codes = {int(code_id) for code_id in payload.active_codes}

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

    def _build_risk_finding_context(
        self,
        finding: Phase1RiskFinding,
    ) -> JsonObject:
        """构建单个 risk finding 的模板上下文。"""

        return {
            "severity": finding.severity,
            "badge_class": "fail" if finding.severity == "fail" else "warn",
            "title": finding.title,
            "reason": finding.reason,
            "related_metrics": ", ".join(finding.related_metrics) or "-",
            "related_codes": (
                ", ".join(str(code_id) for code_id in finding.related_codes) or "-"
            ),
            "related_pairs": ", ".join(finding.related_pairs) or "-",
            "recommended_action": finding.recommended_action,
        }

    def _build_oracle_profitability_kpis(
        self,
        metrics_payload: Any,
    ) -> list[JsonObject]:
        """构建 oracle-label 收益卡 KPI 展示字段。"""

        oracle_metrics = self._nested_value(metrics_payload, "oracle_profitability")
        if oracle_metrics in (None, ""):
            return []

        definitions = (
            (
                "mean_decoded_advantage_vs_flat",
                "mean decoded advantage",
            ),
            (
                "random_label_relative_lift",
                "vs random uplift",
            ),
            (
                "top_5_contribution",
                "top 5% contribution",
            ),
            (
                "trimmed_decoded_advantage",
                "trimmed advantage",
            ),
        )
        rows: list[JsonObject] = []
        for key, label in definitions:
            value = self._nested_value(oracle_metrics, key)
            if value in (None, ""):
                continue
            rows.append(
                {
                    "key": key,
                    "label": label,
                    "value": _format_value(value),
                }
            )
        return rows

    def _build_per_code_profit_series(
        self,
        code_diagnostics: tuple[Phase1CodeDiagnostic, ...],
        oracle_profitability_payload: Phase1OracleProfitabilityPayload | None,
    ) -> list[JsonObject]:
        """构建 per-code 盈利图表序列。"""

        if code_diagnostics:
            return [
                {
                    "code_id": str(item.code_id),
                    "label": f"code {item.code_id}",
                    "value": _format_value(item.decoded_mean_advantage),
                    "badge_class": self._profit_badge_class(
                        item.decoded_mean_advantage
                    ),
                }
                for item in code_diagnostics
                if item.decoded_mean_advantage is not None
            ]
        if oracle_profitability_payload is None:
            return []
        return [
            {
                "code_id": str(item.code_id),
                "label": f"code {item.code_id}",
                "value": _format_value(item.mean_advantage),
                "badge_class": self._profit_badge_class(item.mean_advantage),
            }
            for item in oracle_profitability_payload.per_code_profitability
        ]

    def _vq_internal_payload(self, payload: Any) -> Phase1VQInternalPayload | None:
        """从 report payload 恢复第一层 VQ internal payload。"""

        if isinstance(payload, Phase1VQInternalPayload):
            return payload
        if isinstance(payload, Mapping):
            return Phase1VQInternalPayload.from_dict(payload)
        return None

    def _oracle_profitability_payload(
        self,
        payload: Any,
    ) -> Phase1OracleProfitabilityPayload | None:
        """从 report payload 恢复第三层 oracle profitability payload。"""

        if isinstance(payload, Phase1OracleProfitabilityPayload):
            return payload
        if isinstance(payload, Mapping):
            return Phase1OracleProfitabilityPayload.from_dict(payload)
        return None

    def _nested_value(self, payload: Any, key: str) -> Any:
        """兼容 Mapping 和 dataclass/object 的字段读取。"""

        if isinstance(payload, Mapping):
            return payload.get(key)
        return getattr(payload, key, None)

    def _profit_badge_class(self, value: float | None) -> str:
        """将 per-code 盈利值映射为展示状态。"""

        if value is None or not math.isfinite(value):
            return "warn"
        return "pass" if value >= 0.0 else "fail"

    def _build_mapping_rows(self, payload: Mapping[str, Any]) -> list[JsonObject]:
        """构建普通 key-value 表格上下文。"""

        return [
            {"key": str(key), "value": _format_value(value)}
            for key, value in payload.items()
        ]

    def _build_score_breakdown_rows(
        self,
        score: Phase1ValidationScore,
    ) -> list[JsonObject]:
        """构建综合 score 子项拆解表格上下文。"""

        return [
            {
                "name": component.name,
                "value": _format_value(component.value),
                "weight": _format_value(component.weight),
                "weighted_value": _format_value(component.weighted_value),
            }
            for component in score.components
        ]

    def _validation_score(self, payload: Any) -> Phase1ValidationScore | None:
        """从 report payload 恢复 Phase I validation score。"""

        if payload is None:
            return None
        if isinstance(payload, Phase1ValidationScore):
            return payload
        if isinstance(payload, Mapping):
            return Phase1ValidationScore.from_dict(payload)
        if isinstance(payload, int | float):
            return Phase1ValidationScore.from_float(float(payload))
        return None


__all__ = ["Phase1CodebookReportContextBuilder"]
