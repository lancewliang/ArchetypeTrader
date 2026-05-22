"""Phase II selector reward 与 imitation 基础计算骨架。

文件功能说明:
    本文件定义 Phase II archetype selection 中 reward 相关的函数接口。Phase II
    的 selector 每次选择一个 archetype id，冻结的 Phase I decoder 会把该
    archetype 解码为一个 horizon 内的基础动作序列，环境再通过统一交易执行口径
    得到该 horizon 的收益。本文件负责把交易执行结果转换为 selector 使用的
    horizon-level reward，并提供 imitation regularization 和 epsilon 调度的基础
    工具接口。

论文算法对应:
    论文中的 Phase II 目标可以写作:

    ``J = E[sum_t gamma^t r_t_sel - alpha KL(a_hat_sel || pi_sel)]``

    其中 ``r_t_sel`` 是 selector 选择某个 archetype 后，由冻结 decoder 生成基础
    动作并执行得到的 horizon-level trading return；``KL`` / imitation 项约束
    selector 不要过度偏离 Phase I assigned label。这里的
    ``compute_selection_reward()`` 只处理 ``r_t_sel``，不把 KL 或 imitation loss
    混入 reward，便于 trainer/loss 分别记录收益项和监督约束项。

设计边界:
    - 不调用 Q-network，不访问 replay buffer，不执行 optimizer；
    - 不生成 decoder 动作，动作生成属于 ``phase2_decoder_policy.py``；
    - 不直接运行交易执行，执行细节属于 ``ActionExecutionCalculator`` 和 env；
    - 不计算 Double DQN TD target，该逻辑属于 ``phase2_double_dqn_loss.py``；
    - 不保存日志、checkpoint 或 report。

使用场景:
    ``ArchetypeSelectionEnv`` 可调用 ``compute_selection_reward()`` 将
    ``ActionExecutionResult`` 转成 scalar reward；Double DQN loss 可调用
    ``compute_imitation_kl_loss()`` 计算 assigned-label imitation regularization；
    trainer 可调用 ``build_epsilon_by_epoch()`` 获得当前 epoch 的探索率。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from ...utils import ActionExecutionResult
    from ..phase2_config import Phase2RewardConfig, Phase2TrainConfig


def compute_selection_reward(
    execution: ActionExecutionResult,
    reward_config: Phase2RewardConfig,
) -> np.ndarray:
    """把交易执行结果转换为 Phase II horizon-level reward。

    功能说明:
        从 ``ActionExecutionResult`` 中读取净收益、手续费、换手等执行结果，生成
        selector 的 ``r_t_sel``。完整实现可以使用 ``execution.returns`` 作为默认
        reward，并根据 ``Phase2RewardConfig.reward_clip`` 做裁剪。

    输入参数:
        execution: 统一交易执行器输出，包含净收益 ``returns``、未扣成本收益
            ``gross_returns``、手续费 ``fees`` 和换手 ``turnover``。
        reward_config: Phase II reward 配置，包含手续费、reward 裁剪和 reward
            标准化等策略参数。

    输出:
        ``np.ndarray``，形状通常为 ``[sample]``。每个元素是一个 horizon 对应的
        selector reward ``r_t_sel``。

    使用场景:
        ``ArchetypeSelectionEnv.step()`` 或 ``run_horizon()`` 在执行 decoder 动作后
        调用本函数，得到写入 env step result 和 replay transition 的 reward。

    论文算法:
        对应 ``J`` 中的 ``r_t_sel`` 项，只表示交易收益，不包含 imitation KL。
    """

    raise NotImplementedError("Phase2 selection reward is not implemented yet.")


def compute_imitation_kl_loss(
    q_values: torch.Tensor,
    assigned_labels: torch.Tensor,
) -> torch.Tensor:
    """计算 selector 与 Phase I assigned label 的 imitation regularization。

    功能说明:
        将 selector 输出的 Q values 视作 archetype logits，和 Phase I 离线导出的
        ``assigned_labels`` 计算 imitation loss。第一版可以用 cross entropy 等价
        实现 one-hot label KL；后续可扩展为 label posterior 或 smoothed target。

    输入参数:
        q_values: selector Q-network 输出，形状为 ``[batch, num_archetypes]``。
        assigned_labels: Phase I assigned archetype label，形状为 ``[batch]``，
            dtype 应为 ``torch.long``。

    输出:
        ``torch.Tensor``，标量 imitation loss，可参与
        ``total_loss = td_loss_beta * td_loss + imitation_loss_beta * imitation_loss``。

    使用场景:
        ``compute_double_dqn_loss()`` 在计算 TD loss 后调用本函数，作为辅助监督项。

    论文算法:
        对应 ``- alpha KL(a_hat_sel || pi_sel)`` 中的 KL / imitation 约束项。
    """

    raise NotImplementedError("Phase2 imitation KL loss is not implemented yet.")





def normalize_horizon_rewards(
    rewards: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """标准化 replay batch 中的 horizon-level reward。

    功能说明:
        对 batch 内 ``r_t_sel`` 做均值方差标准化，降低不同市场阶段收益尺度差异对
        TD target 的影响。是否启用由 ``Phase2RewardConfig.normalize_rewards`` 控制。

    输入参数:
        rewards: replay batch 中的 horizon reward，形状为 ``[batch]``。
        eps: 防止除零的数值稳定项。

    输出:
        ``torch.Tensor``，形状与 ``rewards`` 相同，表示标准化后的 reward。

    使用场景:
        ``compute_double_dqn_loss()`` 计算 TD target 前，根据 reward config 决定是否调用。

    论文算法:
        不改变理论目标函数，只是训练数值稳定处理；对应 ``r_t_sel`` 的 batch
        尺度归一化。
    """

    raise NotImplementedError("Phase2 reward normalization is not implemented yet.")


__all__ = [
    "build_epsilon_by_epoch",
    "compute_imitation_kl_loss",
    "compute_selection_reward",
    "normalize_horizon_rewards",
]
