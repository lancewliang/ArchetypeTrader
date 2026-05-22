"""Phase II Double DQN loss 骨架。

文件功能说明:
    本文件定义 Phase II archetype selector 的 Double DQN loss 接口。输入来自
    ``Phase2ReplayBuffer.sample()`` 的 horizon-level transition batch，输出用于
    trainer 反向传播和日志记录的 loss payload。

论文算法对应:
    Phase II selector 的强化学习目标为:

    ``J = E[sum_t gamma^t r_t_sel - alpha KL(a_hat_sel || pi_sel)]``

    工程上使用 Double DQN 近似优化该目标:

    ``a_next = argmax_a Q_online(s_next, a)``

    ``y = r_t_sel + gamma * (1 - done) * Q_target(s_next, a_next)``

    ``td_loss = loss(Q_online(s, a_selected), y)``

    同时加入 Phase I assigned-label imitation regularization:

    ``total_loss = td_loss_beta * td_loss + imitation_loss_beta * imitation_loss``

设计边界:
    - 只计算 Double DQN target、TD loss 和 imitation loss 的组合；
    - 不采样 replay buffer，不收集环境 transition；
    - 不执行 ``backward()``、梯度裁剪或 ``optimizer.step()``，这些属于 trainer；
    - 不计算交易 reward，``r_t_sel`` 应已由 env/reward 模块写入 batch；
    - 不保存 checkpoint，不调用 evaluator。

使用场景:
    ``Phase2DoubleDqnTrainer.update_q_network()`` 从 replay buffer 得到 batch 后，
    调用 ``compute_double_dqn_loss()``，再对返回的 ``total_loss`` 执行反向传播。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ...model.tensor_data_types import ArchetypeLabelTensor
from .phase2_selection_reward import compute_imitation_kl_loss

from ..model.phase2_q_network import Phase2QNetwork
from ..phase2_config import Phase2RewardConfig, Phase2TrainConfig

@dataclass(frozen=True)
class Phase2DoubleDqnLossOutput:
    """Phase II Double DQN loss 输出 payload。

    功能说明:
        承载一次 replay batch 更新所需的总 loss、分项 loss 和关键诊断指标。trainer
        只消费该对象，不需要理解 TD target 或 imitation regularization 的内部公式。

    字段:
        total_loss: 用于 ``backward()`` 的总 loss，标量 tensor。
        td_loss: Double DQN TD error loss，标量 tensor。
        imitation_loss: assigned-label imitation regularization loss，标量 tensor。
        mean_q_selected: 当前 batch 被选 action 的 Q value 均值，用于日志。
        mean_td_target: 当前 batch TD target 均值，用于日志。

    使用场景:
        ``compute_double_dqn_loss()`` 返回本对象；trainer 写 TensorBoard 或训练日志时
        分别记录各字段。
    """

    total_loss: torch.Tensor
    td_loss: torch.Tensor
    imitation_loss: torch.Tensor
    mean_q_selected: torch.Tensor
    mean_td_target: torch.Tensor


def compute_double_dqn_targets(
    online_q_network: Phase2QNetwork,
    target_q_network: Phase2QNetwork,
    batch: Phase2SelectionTransitionBatch,
    gamma: float,
) -> torch.Tensor:
    """计算 Double DQN bootstrap target。

    功能说明:
        使用 online network 在 ``next_visible_states`` 上选择下一步 greedy action，
        再使用 target network 对同一个 action 估计 Q target。done 样本不追加
        bootstrap 项。

    输入参数:
        online_q_network: 当前训练中的 Q-network，用于
            ``argmax_a Q_online(s_next, a)`` 选动作。
        target_q_network: target Q-network，用于
            ``Q_target(s_next, a_next)`` 估计目标值。
        batch: replay buffer 采样得到的 transition batch，包含 reward、done 和
            next visible states。
        gamma: 折扣因子，对应 ``Phase2RewardConfig.gamma``。

    输出:
        ``torch.Tensor``，形状为 ``[batch]``，即
        ``r + gamma * (1 - done) * Q_target(s_next, a_next)``。

    使用场景:
        ``compute_double_dqn_loss()`` 调用本函数生成 TD target。

    论文算法:
        对应 Double DQN 对 ``sum_t gamma^t r_t_sel`` 的 bootstrap 估计。
    """

    raise NotImplementedError("Phase2 Double DQN target is not implemented yet.")


def compute_td_loss(
    online_q_values: torch.Tensor,
    actions: ArchetypeLabelTensor,
    td_targets: torch.Tensor,
) -> torch.Tensor:
    """计算 selected action 的 TD error loss。

    功能说明:
        从 ``online_q_values`` 中 gather 当前实际执行 action 的 Q value，与
        ``td_targets`` 计算 Huber 或 MSE TD loss。具体 loss 类型由完整实现决定，
        但输入输出契约保持不变。

    输入参数:
        online_q_values: online network 对当前状态输出的 Q values，形状为
            ``[batch, num_archetypes]``。
        actions: 当前 replay batch 中实际选择的 archetype id，形状为 ``[batch]``。
        td_targets: ``compute_double_dqn_targets()`` 生成的 target，形状为 ``[batch]``。

    输出:
        ``torch.Tensor``，标量 TD loss。

    使用场景:
        ``compute_double_dqn_loss()`` 中计算 TD 分项，并用于最终 ``total_loss``。

    论文算法:
        对应 ``Q_online(s, a_selected)`` 向 Double DQN target ``y`` 回归。
    """

    raise NotImplementedError("Phase2 TD loss is not implemented yet.")


def compute_double_dqn_loss(
    online_q_network: Phase2QNetwork,
    target_q_network: Phase2QNetwork,
    batch: Phase2SelectionTransitionBatch,
    reward_config: Phase2RewardConfig,
    train_config: Phase2TrainConfig,
) -> Phase2DoubleDqnLossOutput:
    """组合 TD loss 和 assigned-label imitation loss。

    功能说明:
        计算当前 batch 的 online Q values、Double DQN TD target、TD loss 和
        imitation loss，并按训练配置中的权重组合为 ``total_loss``。本函数只返回
        loss payload，不执行参数更新。

    输入参数:
        online_q_network: 当前训练中的 Q-network。
        target_q_network: target Q-network。
        batch: replay buffer 采样得到的 ``Phase2SelectionTransitionBatch``。
        reward_config: reward/loss 相关配置，例如 ``gamma``、``normalize_rewards``
            和 ``imitation_alpha``。
        train_config: 训练配置，例如 ``td_loss_beta`` 和 ``imitation_loss_beta``。

    输出:
        ``Phase2DoubleDqnLossOutput``，包含 ``total_loss``、``td_loss``、
        ``imitation_loss``、``mean_q_selected`` 和 ``mean_td_target``。

    使用场景:
        ``Phase2DoubleDqnTrainer.update_q_network()`` 调用本函数，然后对
        ``total_loss`` 执行 ``backward()``、梯度裁剪和 optimizer step。

    论文算法:
        对应 Phase II 目标 ``J`` 的工程化优化：TD loss 学习交易收益项
        ``sum gamma^t r_t_sel``，imitation loss 对应 ``alpha KL`` 约束。
    """

    raise NotImplementedError("Phase2 Double DQN loss is not implemented yet.")


__all__ = [
    "Phase2DoubleDqnLossOutput",
    "compute_double_dqn_loss",
    "compute_double_dqn_targets",
    "compute_imitation_kl_loss",
    "compute_td_loss",
]
