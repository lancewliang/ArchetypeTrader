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
from pathlib import Path
from typing import Any, Mapping

from ._template import render_template_file
from .phase1_codebook_report_context import Phase1CodebookReportContextBuilder
from ..metrics import Phase1ValidationResult


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
        score = validation_result.score
        summary = {
            "checkpoint_id": validation_result.checkpoint_id,
            "stage": validation_result.stage,
            "epoch": validation_result.epoch,
            "passed": validation_result.passed,
            "score": score.total_score if hasattr(score, "total_score") else score,
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

        context = Phase1CodebookReportContextBuilder(title=self.title).build(payload)
        return render_template_file(_TEMPLATE_PATH, context)


__all__ = ["Phase1CodebookReport"]
