"""Phase I training report skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..phase1_artifact_store import Phase1ArtifactStore
from ..checkpoint import Phase1CheckpointSelectionResult


@dataclass
class Phase1Report:
    """Write Phase I training reports.

    功能描述:
        独立承接 Phase I 训练报告生成与保存入口。主流程只负责在所有产物生成后
        调用 ``write_report``，报告内容组装、schema 校验和持久化细节后续都在
        本类中补齐。

    论文描述:
        Phase I report 应记录 archetype discovery 的训练配置、best checkpoint、
        split-level metrics、诊断信息和产物索引，供 Phase II/III 复现实验和审计
        使用。
    """

    data_store: Phase1ArtifactStore

    def write_report(
        self,
        *,
        config: dict[str, Any] | None = None,
        best_checkpoint_selection: Phase1CheckpointSelectionResult | None = None,
        metrics: dict[str, Any] | None = None,
        diagnostics: dict[str, Any] | None = None,
        output_path: str | Path | None = None,
    ) -> None:
        """Write the Phase I training report.

        参数:
            config: Phase I 主流程配置快照。
            best_checkpoint_selection: selector 输出的 best checkpoint 选择结果。
            metrics: 训练、验证和测试指标摘要。
            diagnostics: collapse、边界样本、action distribution 等诊断摘要。
            output_path: 可选报告输出路径；默认由 ``Phase1ArtifactStore`` 管理。

        方法作用:
            定义 Phase I report 的独立写出入口。当前方法只保留骨架，正式实现
            后续再补齐 report payload 组装、schema 校验和
            ``Phase1ArtifactStore.save_phase1_report`` 调用。
        """

        ...
