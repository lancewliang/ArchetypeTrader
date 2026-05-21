"""Phase II checkpoint payload 骨架。

文件功能说明:
    本文件定义 Phase II archetype selector 训练过程中需要落盘或传递的
    checkpoint 相关 payload 类型。模型参数 checkpoint 和 validation checkpoint
    分开建模，避免把大体积权重、优化器状态和可审计的评估结果混在同一个
    payload 中。

设计边界:
    - 只定义强类型数据容器，不执行 ``torch.save``、JSON 序列化或文件 I/O；
    - 不重新计算 validation metrics，也不判断某个 checkpoint 是否为 best；
    - 不加载 Phase I checkpoint，只记录恢复 Phase II selector 所需的训练配置和
      state dict；
    - 选择 best checkpoint 的逻辑放在 ``phase2_checkpoint_selector.py``。

使用场景:
    trainer 在每个 validation interval 生成 ``Phase2Checkpoint`` 和
    ``Phase2ValidationCheckpoint``；artifact store 负责保存它们；report 和
    checkpoint selector 只消费 validation checkpoint 中的校验结果。
"""

from dataclasses import dataclass
from typing import Any, Mapping

from ..metrics.phase2_metric_results import Phase2ValidationResult
from ..phase2_config import Phase2TrainConfig




@dataclass(frozen=True)
class Phase2Checkpoint:
    """Phase II selector 模型 checkpoint payload。

    功能说明:
        保存恢复 Double DQN selector 训练或推理所需的最小状态，包括 epoch、
        训练配置、Q-network 参数和 optimizer 参数。

    设计边界:
        本类不保存 validation metrics，也不表达 best/selected 语义。模型参数
        文件是否被选中，由独立的 ``Phase2ValidationCheckpoint`` 和
        ``Phase2CheckpointSelector`` 决定。

    使用场景:
        trainer 在训练过程中周期性创建该对象；artifact store 使用 torch 友好的
        payload 保存它；后续恢复训练或加载 selector 推理时读取它。
    """

    # checkpoint 对应训练 epoch。
    epoch: int

    # 恢复训练和构造 selector 所需的 Phase II 训练配置。
    config: Phase2TrainConfig

    # Q-network 的 state_dict，不包含 validation metrics。
    q_network_state_dict: Mapping[str, Any]

    # optimizer 的 state_dict，用于恢复训练。
    optimizer_state_dict: Mapping[str, Any]


@dataclass(frozen=True)
class Phase2ValidationCheckpoint:
    """Phase II validation checkpoint payload。

    功能说明:
        将某个 epoch 的 validation 结果和模型 checkpoint 关联起来。第一版骨架只
        保存 epoch 与 ``Phase2ValidationResult``；完整实现会继续补充
        model checkpoint 路径、hash、Phase I lineage 等审计字段。

    设计边界:
        本类不保存模型权重、不读取模型文件，也不重新运行 evaluator。它是
        checkpoint selector 和 report 的轻量输入。

    使用场景:
        每次 validation 后创建该对象并落盘为 JSON 友好的 payload；训练结束后
        ``Phase2CheckpointSelector`` 从多个 validation checkpoint 中选择最佳项。
    """

    # validation checkpoint 对应训练 epoch。
    epoch: int

    # evaluator 产出的指标与诊断信息。
    validation_result: Phase2ValidationResult
