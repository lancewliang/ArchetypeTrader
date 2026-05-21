"""Phase II selector 统一评估器骨架。

文件功能说明:
    本文件定义 Phase II archetype selection 的统一 evaluator 入口。Phase II
    当前只有 selector 一个评估主题，因此不再拆分
    ``phase2_selection_evaluator.py``；收益评估、assigned-label baseline、
    random baseline、code usage 和 label consistency 诊断都收敛到本文件。

设计边界:
    - 只定义 evaluator 的依赖、方法签名、输入输出契约和职责说明；
    - 不实现真实模型推理、decoder action 生成、交易收益计算或指标聚合；
    - 不保存 checkpoint、metrics JSON 或 report；
    - 不修改训练状态，不执行 optimizer step；
    - 不把 Phase I assigned label 拼入 selector observation。

使用场景:
    ``Phase2MainFlow`` 在 validation/test 阶段创建本 evaluator，并调用
    ``evaluate()`` 生成 ``Phase2ValidationResult``。trainer 和 checkpoint
    selector 后续只消费该结果中的 metrics 和 diagnostics，不重新运行评估逻辑。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..metrics.phase2_metric_results import Phase2ValidationResult
from ..phase2_config import Phase2RewardConfig
from ..phase2_selection_data_schema import Phase2SelectionDataset

class Phase2Evaluator:
    """Phase II archetype selector 的统一评估入口。

    功能说明:
        评估一个已训练或训练中的 selector 在某个 dataset split 上的表现。
        预期评估内容包括:

        - 使用 greedy Q action 选择 archetype 后的 horizon-level return；
        - 使用 Phase I assigned label 作为 oracle/imitation baseline 的 return；
        - 随机 archetype baseline 的 return；
        - selector code usage entropy，诊断是否 code collapse；
        - selected code 与 assigned label 的一致性和混淆矩阵。

    设计边界:
        本类只消费 Q-network、冻结 decoder policy、reward 配置和
        ``Phase2SelectionDataset``。它不训练模型、不读取 checkpoint 文件、不写报告。

    使用场景:
        ``Phase2DoubleDqnTrainer`` 在 validation interval 调用本类；
        ``Phase2MainFlow`` 在训练结束后用同一类评估 validation/test split；
        ``Phase2CheckpointSelector`` 只读取本类产出的 validation result，不直接依赖本类。
    """

    def __init__(
        self,
        reward_config: Phase2RewardConfig,
        device: "torch.device | str",
    ) -> None:
        """初始化 Phase II evaluator 依赖。

        功能说明:
            保存评估所需的 selector Q-network、冻结 Phase I decoder policy、
            reward 口径配置和运行设备。

        使用场景:
            在 ``Phase2MainFlow`` 构造训练组件后创建一次；每次 validation/test
            复用同一个 evaluator，传入不同 split 的 ``Phase2SelectionDataset``。

        参数:
            q_network: 已训练或训练中的 selector Q-network。
            decoder_policy: 冻结的 Phase I decoder policy，用 selected code 生成动作。
            reward_config: Phase II reward 和 baseline 计算配置。
            device: 评估运行设备，例如 ``"cuda"`` 或 ``"cpu"``。
        """

        self.reward_config = reward_config
        self.device = device

    def evaluate_checkpoint(
        self,
        dataset: Phase2SelectionDataset,
        deterministic: bool = True,
        split_name: str = "validation",
        epoch: int | None = None,
    ) -> Phase2ValidationResult:
        """评估 selector 并返回统一 validation result。

        功能说明:
            使用 selector 对 ``dataset.visible_states`` 做 archetype 选择，再通过
            frozen decoder policy 和统一 reward 口径计算 horizon return。评估结果
            应写入 ``Phase2ValidationResult.metrics``，解释性诊断写入
            ``Phase2ValidationResult.diagnostics``。

        使用场景:
            validation checkpoint 保存前调用；训练结束生成 report 前调用；
            test split 离线评估时也复用该入口。

        参数:
            dataset: Phase II selection dataset，包含 selector 可见状态、完整 horizon
                环境数据和 Phase I assigned label。
            deterministic: 为 True 时使用 greedy Q action；正式 validation/test
                checkpoint 选择应使用 True。
            split_name: 当前评估 split 名称，例如 ``"validation"`` 或 ``"test"``。
            epoch: 当前训练 epoch；离线评估或未知 epoch 时可以为 None。

        返回:
            ``Phase2ValidationResult``，供 checkpoint、report 和 selector 复用。
        """

        raise NotImplementedError("Phase2 evaluator main evaluation is not implemented yet.")
