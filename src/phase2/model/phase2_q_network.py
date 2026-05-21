"""Phase II selector Q-network 骨架。

文件功能说明:
    本文件定义 Phase II horizon-level archetype selector 的 Q-network 接口。
    selector 输入在线可见状态 ``(previous_t_states, current_t_states)``，输出每个
    archetype 的 Q value，供 Double DQN trainer、evaluator 和 checkpoint 保存复用。

设计边界:
    - 只定义 Q-network 输出 schema、模型入口和动作选择接口；
    - 不实现具体 temporal encoder、MLP head 或参数初始化策略；
    - 不计算 Double DQN TD target、不访问 replay buffer；
    - 不调用 frozen decoder，也不计算交易 reward；
    - 不读取 Phase I assigned label，label regularization 属于 loss/trainer 层。

使用场景:
    ``Phase2DoubleDqnTrainer`` 使用本网络计算 online/target Q value；
    ``ArchetypeSelectionEnv`` 的 action 来自本网络的 ``select_action()``；
    ``Phase2Evaluator`` 使用 ``greedy_action()`` 做 deterministic validation/test。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ...model.data_types import VisibleStatesDataset
from ...model.tensor_data_types import ArchetypeLabelTensor
from ..phase2_config import Phase2ModelConfig


@dataclass(frozen=True)
class Phase2QNetworkOutput:
    """Phase II Q-network 前向输出。

    功能说明:
        保存 selector 对每个 archetype 的 Q value。该对象让 trainer、evaluator 和
        后续 diagnostics 使用同一个稳定输出结构。

    设计边界:
        本类只承载 forward 结果，不计算 action、不应用 softmax、不参与 loss。

    使用场景:
        ``Phase2QNetwork.forward()`` 返回该对象；Double DQN loss 从
        ``q_values`` 中 gather selected action 的 Q value。
    """

    # 每个 archetype 的 Q value，形状为 [batch, num_archetypes]。
    q_values: torch.Tensor


class Phase2QNetwork(nn.Module):
    """Phase II horizon-level archetype selector Q-network 骨架。

    功能说明:
        接收 selector 在线可见状态，并输出每个 archetype 的 Q value。第一版接口
        显式接收 ``previous_t_states`` 和 ``current_t_states`` 两列，和
        ``Phase2SelectionDatasetBuilder.to_tensor_dataset()`` 的输出保持一致。

    设计边界:
        本类只负责 Q-network 的模型接口。具体结构可以是 flatten + MLP、temporal
        encoder 或其他实现，但必须保持输入输出契约不变。

    使用场景:
        trainer 创建 online network 和 target network；checkpoint 只保存本网络的
        ``state_dict``；evaluator 用 greedy action 评估 selector。
    """

    def __init__(self, config: Phase2ModelConfig) -> None:
        """构建 horizon-level archetype selector Q-network。

        功能说明:
            保存 Q-network 配置。后续实现应在这里创建状态编码层和
            ``num_archetypes`` 维 Q-value head。

        使用场景:
            ``Phase2MainFlow._create_q_network()`` 根据 ``Phase2ModelConfig`` 创建
            online/target 两套网络。

        参数:
            config: Phase II selector 模型配置，包含 state_dim、num_archetypes、
                hidden_dim、num_layers 和 dropout。
        """

        super().__init__()
        self.config = config

    def forward(
        self,
        visible_states: VisibleStatesDataset,
    ) -> Phase2QNetworkOutput:
        """输入 selector 可见状态，输出每个 archetype 的 Q value。

        功能说明:
            后续实现应编码上一分片完整状态序列和当前分片前 ``TSize`` 个状态，
            并输出 ``[batch, num_archetypes]`` 的 Q value。

        使用场景:
            Double DQN loss 计算 online/target Q value；evaluator 计算 greedy action；
            diagnostics 可读取全部 Q value 分布。

        参数:
            previous_t_states: 上一分片完整状态序列，预期形状为
                ``[batch, horizon, state_dim]``。
            current_t_states: 当前分片可见状态窗口，预期形状为
                ``[batch, TSize, state_dim]``。

        返回:
            ``Phase2QNetworkOutput``，其中 ``q_values`` 形状为
            ``[batch, num_archetypes]``。
        """

        raise NotImplementedError("Phase2 Q-network forward is not implemented yet.")
