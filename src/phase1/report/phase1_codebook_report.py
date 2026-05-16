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
from pathlib import Path
from typing import Mapping

from ._template import render_template_file
from .phase1_codebook_report_context import Phase1CodebookReportContextBuilder
from .phase1_codebook_report_schema import (
    JsonObject,
    Phase1CodebookReportDocument,
)
from ..metrics import Phase1ValidationResult


_TEMPLATE_PATH = Path(__file__).with_name("templates") / "phase1_codebook_report.html"


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

    def build_document(
        self,
        *,
        validation_result: Phase1ValidationResult,
        config: Mapping[str, object] | None = None,
        artifacts: Mapping[str, str | Path] | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> Phase1CodebookReportDocument:
        """构建强类型 report payload document。

        输入参数:
            validation_result: 五层 validation 完整结果。
            config: Phase I 配置快照，可为空。
            artifacts: 产物路径索引，可为空。
            metadata: 额外 report 元数据，例如 run id、git sha 或数据批次。

        输出:
            ``Phase1CodebookReportDocument`` 强类型对象。
        """

        return Phase1CodebookReportDocument.from_validation_result(
            validation_result=validation_result,
            title=self.title,
            config=config,
            artifacts=artifacts,
            metadata=metadata,
        )
 
    def build_html(
        self,
        *,
        validation_result: Phase1ValidationResult,
        config: Mapping[str, object] | None = None,
        artifacts: Mapping[str, str | Path] | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> str:
        """从 validation result 直接构建静态 HTML 报告内容。"""

        document = self.build_document(
            validation_result=validation_result,
            config=config,
            artifacts=artifacts,
            metadata=metadata,
        )
        return self.render_html(document)

    def render_html(
        self,
        payload: Phase1CodebookReportDocument,
    ) -> str:
        """将 report payload 渲染为静态 HTML 字符串。

        输入参数:
            payload: ``build_document()`` 返回的强类型 document。

        输出:
            完整 HTML 文本。

        使用场景:
            主流程完成 validation 后渲染人工审计 HTML，再交给 datastore 保存；
            测试中也可直接校验 HTML 片段。
        """

        context = Phase1CodebookReportContextBuilder(title=self.title).build(payload)
        return render_template_file(_TEMPLATE_PATH, context.to_dict())


__all__ = ["Phase1CodebookReport"]
