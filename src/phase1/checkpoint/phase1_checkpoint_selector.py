"""Phase I checkpoint selection entrypoint.

文件功能说明:
    本文件定义 Phase I checkpoint selection 的输入摘要、选择结果和选择器入口。
    选择器从多个 ``Phase1ValidationCheckpoint`` 中挑选最适合进入 Phase II 的
    checkpoint。

    当前选择所需的信息已经由 codebook evaluator 和 metrics 子模块提前计算好，
    主要包括:

    - ``codebook_validation.passed``: 五层 hard gate 是否全部通过；
    - ``codebook_validation.score``: 通过 hard gate 后的综合评分；
    - ``codebook_validation.tie_breaker_metrics``: score 接近时的决胜指标；
    - ``codebook_validation.failed_layers``: 失败候选的审计摘要。

设计边界:
    - 本文件只负责候选 checkpoint 的过滤、排序、选择和失败摘要组织；
    - 不重新计算 evaluator、validation layer、metrics rules 或 score；
    - 不访问模型、DataLoader、训练数据或文件系统；
    - 不保存 best checkpoint，保存动作由 ``Phase1ArtifactStore`` 或主流程负责；
    - 不静默回退到最低 loss checkpoint。没有通过 hard gate 的候选时，应返回明确
      失败结果，供主流程阻断 Phase II artifact export。

使用场景:
    Phase I 训练期间会按 validation interval 生成多个
    ``Phase1ValidationCheckpoint``。训练结束后，主流程调用
    ``Phase1CheckpointSelector.select_best()`` 获取选择结果，再决定是否:

    1. 保存 best checkpoint；
    2. 导出 Phase II horizon archetype labels；
    3. 在 report 或日志中记录失败候选摘要。
"""

from __future__ import annotations

from dataclasses import dataclass

from .phase1_checkpoint import Phase1ValidationCheckpoint


@dataclass(frozen=True)
class Phase1RejectedCheckpointSummary:
    """被 checkpoint selector 排除的候选摘要。

    功能说明:
        保存一个未进入最终候选集合或未被选中的 checkpoint 的关键审计信息。
        该对象只承载 selector 层需要展示的摘要，不复制完整
        ``Phase1ValidationResult``。

    设计边界:
        本类不判断 checkpoint 是否合格，也不计算失败原因；调用方应在 selector
        过滤候选时根据 ``codebook_validation`` 字段填充这些信息。

    使用场景:
        当没有任何 checkpoint 通过 hard gate，或需要在 report 中解释为什么某些
        checkpoint 没有被选为 best checkpoint 时，使用该摘要生成可读 payload。
    """

    # checkpoint 所属训练阶段，例如 "vq"。
    stage: str

    # checkpoint 对应 epoch。
    epoch: int

    # checkpoint 稳定 ID，通常来自 codebook_validation.checkpoint_id。
    checkpoint_id: str

    # 五层 hard gate 是否全部通过。
    passed: bool

    # validation 综合评分。失败 checkpoint 通常为 None。
    score: float | None

    # 失败 layer 名称列表，例如 ("vq_internal", "oracle_profitability")。
    failed_layers: tuple[str, ...]

    # selector 层排除该候选的原因，例如 "validation_failed" 或 "missing_score"。
    reason: str

    def to_dict(self) -> dict[str, object]:
        """序列化为普通 dict。

        输入:
            无，读取当前 dataclass 字段。

        输出:
            JSON 友好的 dict，供 report、日志或 artifact metadata 保存。
        """

        return {
            "stage": self.stage,
            "epoch": self.epoch,
            "checkpoint_id": self.checkpoint_id,
            "passed": self.passed,
            "score": self.score,
            "failed_layers": list(self.failed_layers),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class Phase1CheckpointSelectionResult:
    """Phase I best checkpoint 选择结果。

    功能说明:
        表达 selector 的最终输出。它既能承载成功选择的 checkpoint，也能在没有
        合格候选时承载失败原因和失败候选摘要。

    设计边界:
        本类不持久化模型权重，也不替代 ``Phase1ValidationResult``。完整五层指标
        仍保存在 ``selected.codebook_validation`` 或各 checkpoint 自身 payload 中。
        本类只提供 selector 视角下的结果摘要。

    使用场景:
        ``Phase1CheckpointSelector.select_best()`` 返回该对象；主流程根据
        ``has_selection`` 判断是否继续保存 best checkpoint 和导出 Phase II labels。
    """

    # 被选中的 validation checkpoint；没有合格候选时为 None。
    selected: Phase1ValidationCheckpoint | None

    # 被选中 checkpoint 的稳定 ID；没有合格候选时为 None。
    selected_checkpoint_id: str | None

    # 被选中 checkpoint 的 epoch；没有合格候选时为 None。
    selected_epoch: int | None

    # 被选中 checkpoint 的 validation score；没有合格候选时为 None。
    selected_score: float | None

    # 输入候选总数。
    candidate_count: int

    # 通过 hard gate 且具备 score 的候选数量。
    eligible_count: int

    # 被排除或未选中候选的摘要。
    rejected: tuple[Phase1RejectedCheckpointSummary, ...]

    # 选择结果原因，例如 "selected_highest_score" 或 "no_passed_checkpoint"。
    reason: str

    @property
    def has_selection(self) -> bool:
        """判断 selector 是否成功选出 checkpoint。

        输入:
            无，读取 ``selected`` 字段。

        输出:
            ``True`` 表示可以继续保存 best checkpoint 和导出 Phase II artifacts；
            ``False`` 表示 Phase I validation 没有合格候选，主流程应停止后续导出。
        """

        return self.selected is not None

    def to_dict(self) -> dict[str, object]:
        """序列化 selection result 摘要。

        输入:
            无，读取当前选择结果字段。

        输出:
            JSON 友好的 dict。为避免重复保存完整 checkpoint payload，输出中只包含
            selected checkpoint 的 ID、epoch、score 和失败候选摘要。
        """

        return {
            "selected_checkpoint_id": self.selected_checkpoint_id,
            "selected_epoch": self.selected_epoch,
            "selected_score": self.selected_score,
            "candidate_count": self.candidate_count,
            "eligible_count": self.eligible_count,
            "rejected": [item.to_dict() for item in self.rejected],
            "reason": self.reason,
        }


class Phase1CheckpointSelector:
    """选择最优 Phase I validation checkpoint。

    功能说明:
        从调用方传入的 ``Phase1ValidationCheckpoint`` 列表中筛选并排序候选。
        预期选择策略为:

        1. 过滤未通过 ``codebook_validation.passed`` 的 checkpoint；
        2. 过滤 ``codebook_validation.score is None`` 的 checkpoint；
        3. 按 ``score`` 从高到低排序；
        4. score 差距小于 tie tolerance 时，使用 ``tie_breaker_metrics`` 决胜；
        5. 若仍无法区分，优先选择更早 epoch，降低训练后期 codebook 漂移风险。

    设计边界:
        本类不理解五层 validation 的内部公式，只消费
        ``Phase1ValidationResult`` 暴露出的稳定字段。这样 selector 与 evaluator、
        metrics rules 和 report 解耦。

    使用场景:
        ``Phase1MainFlow.select_and_save_best_checkpoint()`` 在 Phase I 训练结束后
        调用本类，得到最优 checkpoint 或失败摘要。
    """

    def select_best(
        self,
        validation_checkpoints: list[Phase1ValidationCheckpoint],
    ) -> Phase1CheckpointSelectionResult:
        """从显式传入的 validation checkpoints 中选择最优项。

        输入:
            validation_checkpoints:
                候选 checkpoint 列表。每个元素必须包含:

                - ``stage`` 和 ``epoch``，用于过滤、排序和审计；
                - ``train`` / ``val`` 基础训练指标，供 report 展示；
                - ``codebook_validation``，即完整五层 validation result。

        输出:
            ``Phase1CheckpointSelectionResult``。

            成功时:
                - ``selected`` 指向最优 ``Phase1ValidationCheckpoint``；
                - ``selected_checkpoint_id`` / ``selected_epoch`` / ``selected_score``
                  描述被选中的 checkpoint；
                - ``reason`` 说明选择来自最高 score 或 tie-breaker。

            失败时:
                - ``selected`` 为 None；
                - ``eligible_count`` 为 0；
                - ``rejected`` 包含失败候选摘要；
                - ``reason`` 说明没有通过 hard gate 或没有候选输入。

        失败语义:
            如果没有候选通过五层 hard gate，本函数不应返回任意失败 checkpoint
            作为 best checkpoint。Phase II label export 必须依赖
            ``result.has_selection`` 显式判断。
        """

        ...
