"""Phase II selector validation report payload/HTML 呈现入口。

文件功能说明:
    本文件实现 ``Phase2SelectorReport``，负责把已经计算好的
    ``Phase2ValidationResult`` 或 ``Phase2CheckpointSelectionResult`` 转换为
    report payload 和静态 HTML report。

设计边界:
    - 只消费 validation result、selection result、config 和 artifacts；
    - 不重新计算 raw metrics；
    - 不重新执行 hard gate rules；
    - 不重新选择 checkpoint；
    - HTML 是静态审计视图，不依赖外部 JS/CSS 文件。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping

from ...phase1.report._template import render_template_file 
from ..metrics import Phase2ValidationResult
from .phase2_selector_report_context import Phase2SelectorReportContextBuilder
from .phase2_selector_report_schema import (
    DEFAULT_PHASE2_REPORT_TITLE,
    Phase2ReportMeta,
    Phase2ReportDocument 
)


_TEMPLATE_PATH = Path(__file__).with_name("templates") / "phase2_selector_report.html"


@dataclass(frozen=True)
class Phase2SelectorReport:
    """Phase II selector validation 报告渲染器。

    功能说明:
        提供 ``build_document()``、``build_payload()``、``build_html()`` 和
        ``render_html()`` 入口。payload 保留完整机器可读结构，HTML 提供人工
        审计视图；文件写入由 artifact store 统一负责。
    """

    title: str = DEFAULT_PHASE2_REPORT_TITLE

    def build_document(
        self,
        *,
        validation_result: Phase2ValidationResult,
        config: Mapping[str, object] | None = None,
        artifacts: Mapping[str, str | Path] | None = None,
        metadata: Mapping[str, object] | None = None,
        selection: Mapping[str, object] | None = None,
    ) -> Phase2ReportDocument:
        """从 validation result 构建强类型 report payload document。"""

        return Phase2ReportDocument(
            report=Phase2ReportMeta(
                title=self.title,
                generated_at=datetime.now(UTC).isoformat(),
                metadata=metadata,
            ),
            selection=selection or {},
            summary=_build_validation_summary(validation_result),
            validation=validation_result,
            config=config or {},
            artifacts=artifacts or {},
        )

    def build_html(
        self,
        *,
        validation_result: Phase2ValidationResult,
        config: Mapping[str, object] | None = None,
        artifacts: Mapping[str, str | Path] | None = None,
        metadata: Mapping[str, object] | None = None,
        selection: Mapping[str, object] | None = None,
    ) -> str:
        """从 validation result 直接构建静态 HTML 报告内容。"""

        document = self.build_document(
            validation_result=validation_result,
            config=config,
            artifacts=artifacts,
            metadata=metadata,
            selection=selection,
        )
        return self.render_html(document)
 
    def render_html(
        self,
        payload: Phase2ReportDocument ,
    ) -> str:
        """将 report payload 渲染为静态 HTML 字符串。"""

        context = Phase2SelectorReportContextBuilder(title=self.title).build(payload)
        return render_template_file(_TEMPLATE_PATH, context.to_dict())


def _build_validation_summary(
    validation_result: Phase2ValidationResult,
) -> dict:
    """从 validation result 生成 report 首页摘要。"""

    metrics = validation_result.metrics
    failed_layers = tuple(
        layer.name for layer in validation_result.layers if not layer.passed
    )
    passed = len(failed_layers) == 0
    return {
        "passed": passed,
        "status": "pass" if passed else "fail",
        "mean_return": metrics.mean_return,
        "median_return": metrics.median_return,
        "sharpe_like": metrics.sharpe_like,
        "win_rate": metrics.win_rate,
        "mean_turnover": metrics.mean_turnover,
        "layer_count": len(validation_result.layers),
        "failed_layers": list(failed_layers),
    }
    

__all__ = ["Phase2SelectorReport"]
