"""Phase I codebook validation report payload/HTML 呈现入口。

文件功能说明:
    本文件实现 ``Phase1CodebookReport``，负责把已经计算好的
    ``Phase1ValidationResult`` 转换为 report payload 和静态 HTML report。

设计边界:
    - 只消费 validation result、config 和 artifacts；
    - 不重新计算 raw metrics；
    - 不重新执行 hard gate rules；
    - 不参与 checkpoint selection；
    - HTML 是静态审计视图，不依赖外部 JS/CSS 文件。

使用场景:
    训练主流程在拿到 ``Phase1ValidationResult`` 后，调用本类渲染 HTML，
    再交给 datastore 保存，供人工审计和实验复盘。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import math
from pathlib import Path
from typing import Any, Mapping

from ._template import render_template_file
from ..metrics import (
    Phase1CodeDiagnostic,
    Phase1LayerResult,
    Phase1MetricResult,
    Phase1ValidationResult,
)


JsonObject = dict[str, Any]
_TEMPLATE_PATH = Path(__file__).with_name("templates") / "phase1_codebook_report.html"


def _json_safe(value: Any) -> Any:
    """把常见非 JSON 原生对象转换为可序列化值。

    输入参数:
        value: 任意 report payload 值，可包含 Path、tuple、list、mapping 或具备
            ``to_dict()`` 方法的对象。

    输出:
        JSON 可序列化的 Python 值。

    使用场景:
        写 JSON 前统一处理 artifacts、config 和 validation result 中的路径或对象。
    """

    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _json_safe(value.to_dict())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_safe(item) for item in value]
    return value


def _format_value(value: Any) -> str:
    """格式化 HTML 表格中的指标值。

    输入参数:
        value: 原始指标值。

    输出:
        面向 HTML 展示的字符串。

    使用场景:
        metric、tie-breaker、config 和 artifact 表格渲染。
    """

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
    return str(value)


def _badge_class(severity: str | bool) -> str:
    """将状态映射为 HTML badge class。

    输入参数:
        severity: metric severity 字符串，或 layer/checkpoint passed 布尔值。

    输出:
        ``pass``、``fail``、``warn`` 或 ``skip``。

    使用场景:
        report 中统一渲染 checkpoint、layer 和 metric 状态。
    """

    if isinstance(severity, bool):
        return "pass" if severity else "fail"
    if severity in {"pass", "fail", "warn", "skip"}:
        return severity
    return "warn"


@dataclass(frozen=True)
class Phase1CodebookReport:
    """Phase I codebook validation 报告渲染器。

    功能说明:
        提供 ``build_payload()``、``build_html()`` 和 ``render_html()`` 入口。
        payload 保留完整机器可读结构，HTML 提供人工审计视图；文件写入由
        datastore 统一负责。

    使用场景:
        full validation 完成后，把 ``Phase1ValidationResult`` 转成 payload 并
        渲染为 ``phase1_codebook_validation.html`` 内容。
    """

    title: str = "Phase I Codebook Validation Report"

    def build_payload(
        self,
        *,
        validation_result: Phase1ValidationResult,
        config: Mapping[str, object] | None = None,
        artifacts: Mapping[str, str | Path] | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> JsonObject:
        """构建 JSON/HTML 共用 report payload。

        输入参数:
            validation_result: 五层 validation 完整结果。
            config: Phase I 配置快照，可为空。
            artifacts: 产物路径索引，可为空。
            metadata: 额外 report 元数据，例如 run id、git sha 或数据批次。

        输出:
            普通 dict，包含 report 元信息、summary、validation、config 和 artifacts。

        使用场景:
            ``render_html()`` 的前置步骤；外部系统也可直接调用本方法拿到
            机器可读 payload。
        """

        generated_at = datetime.now(UTC).isoformat()
        validation_payload = validation_result.to_dict()
        summary = {
            "checkpoint_id": validation_result.checkpoint_id,
            "stage": validation_result.stage,
            "epoch": validation_result.epoch,
            "passed": validation_result.passed,
            "score": validation_result.score,
            "failed_layers": list(validation_result.failed_layers),
            "failed_layer_count": len(validation_result.failed_layers),
            "layer_count": len(validation_result.layers),
            "code_diagnostic_count": len(validation_result.code_diagnostics),
        }
        payload = {
            "report": {
                "title": self.title,
                "generated_at": generated_at,
                "schema": "phase1_codebook_validation_report.v1",
                **dict(metadata or {}),
            },
            "summary": summary,
            "validation": validation_payload,
            "config": dict(config or {}),
            "artifacts": dict(artifacts or {}),
        }
        return _json_safe(payload)

    def build_html(
        self,
        *,
        validation_result: Phase1ValidationResult,
        config: Mapping[str, object] | None = None,
        artifacts: Mapping[str, str | Path] | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> str:
        """从 validation result 直接构建静态 HTML 报告内容。"""

        payload = self.build_payload(
            validation_result=validation_result,
            config=config,
            artifacts=artifacts,
            metadata=metadata,
        )
        return self.render_html(payload)

    def render_html(self, payload: Mapping[str, Any]) -> str:
        """将 report payload 渲染为静态 HTML 字符串。

        输入参数:
            payload: ``build_payload()`` 返回的 report payload。

        输出:
            完整 HTML 文本。

        使用场景:
            主流程完成 validation 后渲染人工审计 HTML，再交给 datastore 保存；
            测试中也可直接校验 HTML 片段。
        """

        return render_template_file(_TEMPLATE_PATH, self._build_html_context(payload))

    def _build_html_context(self, payload: Mapping[str, Any]) -> JsonObject:
        """把 report payload 转成模板使用的展示模型。

        输入参数:
            payload: ``build_payload()`` 返回的 report payload。

        输出:
            只包含字符串、数字、bool、list 和 dict 的模板上下文。

        使用场景:
            ``render_html()`` 调用模板前完成强类型恢复、值格式化和状态映射，
            HTML 结构本身留在独立模板文件中。
        """

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
            "tie_breaker_rows": self._build_mapping_rows(
                validation.get("tie_breaker_metrics", {})
            ),
            "drift_diagnostics": [
                self._build_metric_context(metric)
                for metric in drift_diagnostics.values()
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

    def _build_mapping_rows(self, payload: Mapping[str, Any]) -> list[JsonObject]:
        """构建普通 key-value 表格上下文。"""

        return [
            {"key": str(key), "value": _format_value(value)}
            for key, value in payload.items()
        ]


__all__ = ["Phase1CodebookReport"]
