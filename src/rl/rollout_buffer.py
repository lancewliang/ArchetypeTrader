"""Rollout buffer: 保存 PPO rollout 数据，支持按 env_id 分组的 GAE 计算。

设计文档锚点: Phase II 执行计划 §Step 5。

职责:
- 保存 obs / env_id / action / log_prob / value / reward / done / truncated。
- 保存 kl_label / is_labeled / dead_code_mask。
- 保存 info_cost_paid / info_boundary_cost / info_chosen_code。
- 计算 GAE（必须按 env_id 分组，不跨 env 混算）。
- done=True 才切断 bootstrap；truncated=True 只表示 buffer 截断。
- 支持 flatten 为 minibatch。

关键约束:
- GAE 严格按 env_id 分组计算。
- raw/scaled reward 同步记录。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch


@dataclass
class RolloutSample:
    """单步 rollout 记录。"""
    obs: np.ndarray
    env_id: int
    action: int
    log_prob: float
    value: float
    reward: float
    reward_raw: float
    done: bool
    truncated: bool
    reward_was_clipped: bool = False
    kl_label: Optional[int] = None
    is_labeled: bool = False
    dead_code_mask: Optional[Any] = None
    info_cost_paid: float = 0.0
    info_boundary_cost: float = 0.0
    info_chosen_code: int = 0


class RolloutBuffer:
    """PPO rollout buffer。

    使用方式::

        buffer = RolloutBuffer(num_envs=4, rollout_length=128, gamma=0.99, gae_lambda=0.95)
        for step in range(rollout_length):
            buffer.add(samples_from_all_envs)
        buffer.compute_gae(last_values)
        for minibatch in buffer.iterate_minibatches(minibatch_size):
            ...
        buffer.reset()
    """

    def __init__(
        self,
        num_envs: int,
        rollout_length: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        self.num_envs = num_envs
        self.rollout_length = rollout_length
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        # 按 [step, env] 组织存储
        self._obs: List[List[np.ndarray]] = []
        self._actions: List[List[int]] = []
        self._log_probs: List[List[float]] = []
        self._values: List[List[float]] = []
        self._rewards: List[List[float]] = []
        self._rewards_raw: List[List[float]] = []
        self._reward_was_clipped: List[List[bool]] = []
        self._dones: List[List[bool]] = []
        self._truncateds: List[List[bool]] = []
        self._kl_labels: List[List[Optional[int]]] = []
        self._is_labeled: List[List[bool]] = []
        self._info_cost_paid: List[List[float]] = []
        self._info_boundary_cost: List[List[float]] = []
        self._info_chosen_code: List[List[int]] = []
        self._advantages: Optional[np.ndarray] = None
        self._returns: Optional[np.ndarray] = None
        self._step_count = 0

    def add(self, samples: List[RolloutSample]) -> None:
        """添加一步的所有 env 样本。

        Parameters
        ----------
        samples : 长度为 num_envs 的样本列表。
        """
        assert len(samples) == self.num_envs
        self._obs.append([s.obs for s in samples])
        self._actions.append([s.action for s in samples])
        self._log_probs.append([s.log_prob for s in samples])
        self._values.append([s.value for s in samples])
        self._rewards.append([s.reward for s in samples])
        self._rewards_raw.append([s.reward_raw for s in samples])
        self._reward_was_clipped.append([s.reward_was_clipped for s in samples])
        self._dones.append([s.done for s in samples])
        self._truncateds.append([s.truncated for s in samples])
        self._kl_labels.append([s.kl_label for s in samples])
        self._is_labeled.append([s.is_labeled for s in samples])
        self._info_cost_paid.append([s.info_cost_paid for s in samples])
        self._info_boundary_cost.append([s.info_boundary_cost for s in samples])
        self._info_chosen_code.append([s.info_chosen_code for s in samples])
        self._step_count += 1

    def compute_gae(self, last_values: List[float]) -> None:
        """计算 GAE advantages 和 returns。

        必须按 env_id 分组计算，不跨 env 混算。
        done=True 才切断 bootstrap；truncated=True 只表示 buffer 截断。

        Parameters
        ----------
        last_values : 每个 env 的最后一步 value（用于 bootstrap）。
        """
        T = self._step_count
        E = self.num_envs
        advantages = np.zeros((T, E), dtype=np.float32)
        returns = np.zeros((T, E), dtype=np.float32)

        # 按 env 分组计算 GAE
        # 设计约束: done=True 才切断 bootstrap；truncated=True 只表示 buffer 截断，
        # 此时 done=False，需要用 bootstrap value 继续估算。
        for env_id in range(E):
            last_gae = 0.0
            # next_value / next_non_terminal 初始化为 rollout 末尾的 bootstrap
            next_value = last_values[env_id]
            next_non_terminal = 1.0

            for t in reversed(range(T)):
                reward = self._rewards[t][env_id]
                value = self._values[t][env_id]
                done = self._dones[t][env_id]
                truncated = self._truncateds[t][env_id]

                # 对于 t+1 步（即上一轮迭代处理的步骤）的终止信号:
                # - done=True: episode 真正结束，next_value=0，切断 bootstrap
                # - truncated=True (done=False): buffer 截断，用 bootstrap value
                #   （已在 next_value 中，来自 last_values 或上一步的 value）
                # 注意: done 和 truncated 描述的是 step t 执行后的状态，
                # 影响的是从 t 到 t+1 的 bootstrap。
                if done:
                    # episode 结束，不 bootstrap
                    delta = reward - value
                    last_gae = delta
                else:
                    delta = reward + self.gamma * next_value * next_non_terminal - value
                    last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae

                advantages[t, env_id] = last_gae
                returns[t, env_id] = last_gae + value

                # 为下一轮迭代（处理 t-1）准备 next_value / next_non_terminal
                if done:
                    next_value = 0.0
                    next_non_terminal = 0.0
                elif truncated:
                    # truncated 时 done=False，下一轮迭代的 next_value 应该是
                    # 当前步的 value（因为 t-1 到 t 之间 episode 没有结束）
                    next_value = value
                    next_non_terminal = 1.0
                else:
                    next_value = value
                    next_non_terminal = 1.0

        self._advantages = advantages
        self._returns = returns

    def iterate_minibatches(self, minibatch_size: int, device: str = "cpu"):
        """生成 minibatch 迭代器。

        Yields
        ------
        dict : 包含 obs / action / log_prob / value / advantage / return_ /
               kl_label / is_labeled 的 tensor batch。
        """
        assert self._advantages is not None, "必须先调用 compute_gae()"
        T = self._step_count
        E = self.num_envs
        total = T * E

        # Flatten 所有数据
        all_obs = []
        all_actions = []
        all_log_probs = []
        all_values = []
        all_advantages = []
        all_returns = []
        all_kl_labels = []
        all_is_labeled = []

        for t in range(T):
            for e in range(E):
                all_obs.append(self._obs[t][e])
                all_actions.append(self._actions[t][e])
                all_log_probs.append(self._log_probs[t][e])
                all_values.append(self._values[t][e])
                all_advantages.append(self._advantages[t, e])
                all_returns.append(self._returns[t, e])
                kl = self._kl_labels[t][e]
                all_kl_labels.append(kl if kl is not None else -1)
                all_is_labeled.append(self._is_labeled[t][e])

        obs_tensor = torch.tensor(np.array(all_obs), dtype=torch.float32, device=device)
        actions_tensor = torch.tensor(all_actions, dtype=torch.long, device=device)
        log_probs_tensor = torch.tensor(all_log_probs, dtype=torch.float32, device=device)
        values_tensor = torch.tensor(all_values, dtype=torch.float32, device=device)
        advantages_tensor = torch.tensor(all_advantages, dtype=torch.float32, device=device)
        returns_tensor = torch.tensor(all_returns, dtype=torch.float32, device=device)
        kl_labels_tensor = torch.tensor(all_kl_labels, dtype=torch.long, device=device)
        is_labeled_tensor = torch.tensor(all_is_labeled, dtype=torch.bool, device=device)

        # 随机打乱
        indices = torch.randperm(total, device=device)
        for start in range(0, total, minibatch_size):
            end = min(start + minibatch_size, total)
            mb_idx = indices[start:end]
            yield {
                "obs": obs_tensor[mb_idx],
                "action": actions_tensor[mb_idx],
                "old_log_prob": log_probs_tensor[mb_idx],
                "old_value": values_tensor[mb_idx],
                "advantage": advantages_tensor[mb_idx],
                "return_": returns_tensor[mb_idx],
                "kl_label": kl_labels_tensor[mb_idx],
                "is_labeled": is_labeled_tensor[mb_idx],
            }

    def reset(self) -> None:
        """清空 buffer，准备下一次 rollout。"""
        self._obs.clear()
        self._actions.clear()
        self._log_probs.clear()
        self._values.clear()
        self._rewards.clear()
        self._rewards_raw.clear()
        self._reward_was_clipped.clear()
        self._dones.clear()
        self._truncateds.clear()
        self._kl_labels.clear()
        self._is_labeled.clear()
        self._info_cost_paid.clear()
        self._info_boundary_cost.clear()
        self._info_chosen_code.clear()
        self._advantages = None
        self._returns = None
        self._step_count = 0

    def get_stats(self) -> Dict[str, float]:
        """返回 buffer 统计（reward mean/std、clipped ratio 等）。"""
        all_rewards = [r for step in self._rewards for r in step]
        all_raw = [r for step in self._rewards_raw for r in step]
        all_clipped = [c for step in self._reward_was_clipped for c in step]
        all_dones = [d for step in self._dones for d in step]
        all_truncated = [t for step in self._truncateds for t in step]
        if not all_rewards:
            return {"reward_mean": 0.0, "reward_std": 0.0, "reward_raw_mean": 0.0}
        return {
            "reward_mean": float(np.mean(all_rewards)),
            "reward_std": float(np.std(all_rewards)),
            "reward_raw_mean": float(np.mean(all_raw)),
            "reward_raw_std": float(np.std(all_raw)),
            "reward_clipped_ratio": float(np.mean(all_clipped)) if all_clipped else 0.0,
            "reward_unclipped_mean": float(np.mean(all_raw)),
            "reward_unclipped_std": float(np.std(all_raw)),
            "rollout_done_count": float(sum(1 for d in all_dones if d)),
            "rollout_truncated_count": float(sum(1 for t in all_truncated if t)),
            "rollout_bootstrap_count": float(sum(1 for t, d in zip(all_truncated, all_dones) if t and not d)),
        }

    def flat_values_returns(self) -> Dict[str, List[float]]:
        """返回 GAE 后 flatten 的 values/returns，供 PPO 健康指标使用。"""
        assert self._returns is not None, "必须先调用 compute_gae()"
        values: List[float] = []
        returns: List[float] = []
        for t in range(self._step_count):
            for e in range(self.num_envs):
                values.append(float(self._values[t][e]))
                returns.append(float(self._returns[t, e]))
        return {"values": values, "returns": returns}
