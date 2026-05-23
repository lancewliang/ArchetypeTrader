"""Phase II Double DQN loss.

本模块实现论文 Phase II archetype selection 的训练目标：Double DQN 的
TD 目标负责最大化 horizon-level return，assigned archetype label 的 KL
regularization 负责把 selector 约束在 demonstration archetype 附近。
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F

from ..model.phase2_q_network import Phase2QNetwork
from ..phase2_config import Phase2RewardConfig, Phase2TrainConfig
from .phase2_replay_buffer import Phase2SelectionTransitionTensorBatch


@dataclass(frozen=True)
class Phase2DoubleDqnLossOutput:
    """Phase II selector 一次 Q-network update 的 loss 和诊断输出。"""

    total_loss: torch.Tensor
    td_loss: torch.Tensor
    imitation_loss: torch.Tensor
    selected_q_mean: float
    td_target_mean: float
    reward_mean: float
    greedy_next_action_mean: float
    grad_norm: float | None = None

    def with_grad_norm(self, grad_norm: float) -> "Phase2DoubleDqnLossOutput":
        """返回带梯度范数诊断值的新输出。"""

        return replace(self, grad_norm=float(grad_norm))


def compute_double_dqn_loss(
    *,
    online_q_network: Phase2QNetwork,
    target_q_network: Phase2QNetwork,
    batch: Phase2SelectionTransitionTensorBatch,
    reward_config: Phase2RewardConfig,
    train_config: Phase2TrainConfig,
) -> Phase2DoubleDqnLossOutput:
    """计算论文 Phase II selector 的 Double DQN + imitation KL loss.

    Double DQN 部分使用 online network 在 next state 上选动作，再用 target
    network 评估该动作：

    ``y = r + gamma * (1 - done) * Q_target(s', argmax_a Q_online(s', a))``。

    论文公式 (5) 中的 ``KL(hat_a || pi(a|s))`` 在 one-hot assigned label 下
    等价于对 ``softmax(Q(s, .))`` 做 cross entropy。
    """

    actions = _as_flat_long(batch.actions, name="actions")
    dones = _as_flat_float(batch.dones, name="dones")
    rewards = _prepare_rewards(batch.rewards, reward_config)

    online_q_values = online_q_network(batch.visible_states)
    _validate_q_values(online_q_values, actions, "online_q_values")
    selected_q_values = online_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    td_targets, greedy_next_actions = compute_double_dqn_targets(
        online_q_network=online_q_network,
        target_q_network=target_q_network,
        next_visible_states=batch.next_visible_states,
        rewards=rewards,
        dones=dones,
        gamma=reward_config.gamma,
    )
    td_loss = compute_td_loss(selected_q_values, td_targets)
    imitation_loss = compute_imitation_kl_loss(
        q_values=online_q_values,
        demonstration_horizon_label_batch=batch.demonstration_horizon_label_batch,
    )
    total_loss = (
        train_config.td_loss_beta * td_loss
        + reward_config.imitation_alpha
        * train_config.imitation_loss_beta
        * imitation_loss
    )

    return Phase2DoubleDqnLossOutput(
        total_loss=total_loss,
        td_loss=td_loss,
        imitation_loss=imitation_loss,
        selected_q_mean=_tensor_mean(selected_q_values),
        td_target_mean=_tensor_mean(td_targets),
        reward_mean=_tensor_mean(rewards),
        greedy_next_action_mean=_tensor_mean(greedy_next_actions.float()),
    )


def compute_double_dqn_targets(
    *,
    online_q_network: Phase2QNetwork,
    target_q_network: Phase2QNetwork,
    next_visible_states: tuple[torch.Tensor, ...],
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """计算 Double DQN bootstrap target 和 online greedy next action。"""

    if gamma < 0.0:
        raise ValueError(f"gamma must be non-negative, got {gamma}")

    rewards = _as_flat_float(rewards, name="rewards")
    dones = _as_flat_float(dones, name="dones")
    if rewards.shape != dones.shape:
        raise ValueError(
            "rewards and dones must have the same shape, "
            f"got {tuple(rewards.shape)} and {tuple(dones.shape)}"
        )

    with torch.no_grad():
        next_online_q_values = online_q_network(next_visible_states)
        greedy_next_actions = torch.argmax(next_online_q_values, dim=1)
        next_target_q_values = target_q_network(next_visible_states)
        _validate_q_values(next_target_q_values, greedy_next_actions, "next_target_q")
        next_q_values = next_target_q_values.gather(
            1,
            greedy_next_actions.unsqueeze(1),
        ).squeeze(1)
        td_targets = rewards + float(gamma) * (1.0 - dones) * next_q_values
    return td_targets, greedy_next_actions


def compute_td_loss(
    selected_q_values: torch.Tensor,
    td_targets: torch.Tensor,
) -> torch.Tensor:
    """计算 DQN TD loss，使用 Huber loss 提升 outlier 下的稳定性。"""

    selected_q_values = _as_flat_float(selected_q_values, name="selected_q_values")
    td_targets = _as_flat_float(td_targets, name="td_targets")
    if selected_q_values.shape != td_targets.shape:
        raise ValueError(
            "selected_q_values and td_targets must have the same shape, "
            f"got {tuple(selected_q_values.shape)} and {tuple(td_targets.shape)}"
        )
    return F.smooth_l1_loss(selected_q_values, td_targets.detach())


def compute_imitation_kl_loss(
    *,
    q_values: torch.Tensor,
    demonstration_horizon_label_batch: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """计算论文公式 (5) 的 assigned-label KL regularization.

    ``demonstration_horizon_label_batch`` 的第二列是 Phase I VQ encoder 赋给
    当前 horizon 的 ``code_label``。由于 label 是 one-hot 分布，
    ``KL(one_hot || softmax(Q))`` 与 cross entropy 等价。
    """

    if q_values.ndim != 2:
        raise ValueError(
            "q_values must have shape [batch, num_archetypes], "
            f"got {tuple(q_values.shape)}"
        )
    _, code_labels = demonstration_horizon_label_batch
    code_labels = _as_flat_long(code_labels, name="code_labels")
    if code_labels.shape[0] != q_values.shape[0]:
        raise ValueError(
            "code_labels batch size must match q_values batch size, "
            f"got {code_labels.shape[0]} and {q_values.shape[0]}"
        )
    _validate_action_range(code_labels, q_values.shape[1], name="code_labels")
    return F.cross_entropy(q_values, code_labels)


def _prepare_rewards(
    rewards: torch.Tensor,
    reward_config: Phase2RewardConfig,
) -> torch.Tensor:
    rewards = _as_flat_float(rewards, name="rewards")
    if reward_config.reward_clip is not None:
        clip_value = float(reward_config.reward_clip)
        if clip_value <= 0.0:
            raise ValueError(f"reward_clip must be positive, got {clip_value}")
        rewards = rewards.clamp(min=-clip_value, max=clip_value)
    if reward_config.normalize_rewards and rewards.numel() > 1:
        std = rewards.std(unbiased=False)
        rewards = (rewards - rewards.mean()) / std.clamp_min(1e-8)
    return rewards


def _validate_q_values(
    q_values: torch.Tensor,
    actions: torch.Tensor,
    name: str,
) -> None:
    if q_values.ndim != 2:
        raise ValueError(
            f"{name} must have shape [batch, num_archetypes], "
            f"got {tuple(q_values.shape)}"
        )
    if actions.shape != (q_values.shape[0],):
        raise ValueError(
            "actions must have shape [batch] matching q_values, "
            f"got {tuple(actions.shape)} and {tuple(q_values.shape)}"
        )
    _validate_action_range(actions, q_values.shape[1], name="actions")


def _validate_action_range(
    actions: torch.Tensor,
    num_actions: int,
    *,
    name: str,
) -> None:
    if actions.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    if torch.any(actions < 0) or torch.any(actions >= num_actions):
        min_action = int(actions.min().detach().cpu().item())
        max_action = int(actions.max().detach().cpu().item())
        raise ValueError(
            f"{name} must be in [0, {num_actions}), "
            f"got min={min_action}, max={max_action}"
        )


def _as_flat_float(values: torch.Tensor, *, name: str) -> torch.Tensor:
    if values.ndim != 1:
        values = values.reshape(-1)
    return values.to(dtype=torch.float32)


def _as_flat_long(values: torch.Tensor, *, name: str) -> torch.Tensor:
    if values.ndim != 1:
        values = values.reshape(-1)
    return values.to(dtype=torch.long)


def _tensor_mean(values: torch.Tensor) -> float:
    return float(values.detach().float().mean().cpu().item())


__all__ = [
    "Phase2DoubleDqnLossOutput",
    "compute_double_dqn_loss",
    "compute_double_dqn_targets",
    "compute_imitation_kl_loss",
    "compute_td_loss",
]
