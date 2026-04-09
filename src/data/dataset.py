"""PyTorch Dataset 封装 — DP 轨迹数据用于 VQ 训练

封装 (s_demo, a_demo, r_demo) 三元组，支持从 .npz 文件加载
DP Planner 生成的示范轨迹。

支持 per-feature z-score 归一化，解决不同品种（如 AL vs ETH）
因价格尺度差异导致的 encoder z_e 分布坍缩问题。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class TrajectoryDataset(Dataset):
    """DP 示范轨迹数据集，用于 VQ Encoder-Decoder 训练。

    每个样本为一条 horizon 长度的示范轨迹 (s_demo, a_demo, r_demo)。

    支持 per-feature z-score 归一化:
    - states: 按特征维度 (state_dim) 做 z-score
    - rewards: 全局 z-score
    归一化统计量保存在 norm_stats 中，供 env_validation 等下游模块复用。
    """

    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        normalize: bool = False,
        norm_stats: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        """
        Args:
            states: 状态序列，shape (N, h, state_dim)
            actions: 动作序列，shape (N, h)，值域 {0, 1, 2}
            rewards: 奖励序列，shape (N, h)
            normalize: 是否对 states 和 rewards 做 z-score 归一化
            norm_stats: 外部提供的归一化统计量。若为 None 且 normalize=True，则从当前数据计算。
        """
        # --- 维度校验 ---
        if states.ndim != 3:
            raise ValueError(f"states 应为 3D (N, h, state_dim)，实际为 {states.ndim}D")
        if actions.ndim != 2:
            raise ValueError(f"actions 应为 2D (N, h)，实际为 {actions.ndim}D")
        if rewards.ndim != 2:
            raise ValueError(f"rewards 应为 2D (N, h)，实际为 {rewards.ndim}D")

        n_states, h_states = states.shape[0], states.shape[1]
        n_actions, h_actions = actions.shape
        n_rewards, h_rewards = rewards.shape

        if not (n_states == n_actions == n_rewards):
            raise ValueError(f"样本数不一致: states={n_states}, actions={n_actions}, rewards={n_rewards}")
        if h_actions != h_states or h_rewards != h_states:
            raise ValueError(
                f"horizon 长度不一致: states h={h_states}, actions h={h_actions}, rewards h={h_rewards}"
            )

        # 归一化处理
        self.norm_stats: Optional[Dict[str, np.ndarray]] = None
        if normalize:
            if norm_stats is not None:
                self.norm_stats = norm_stats
            else:
                flat_states = states.reshape(-1, states.shape[-1])
                flat_rewards = rewards.reshape(-1)
                self.norm_stats = {
                    "state_mean": flat_states.mean(axis=0).astype(np.float32),
                    "state_std": flat_states.std(axis=0).astype(np.float32),
                    "reward_mean": np.float32(flat_rewards.mean()),
                    "reward_std": np.float32(flat_rewards.std()),
                }
            s_mean = self.norm_stats["state_mean"]
            s_std = np.maximum(self.norm_stats["state_std"], 1e-8)
            r_mean = self.norm_stats["reward_mean"]
            r_std = max(float(self.norm_stats["reward_std"]), 1e-8)

            states = (states - s_mean) / s_std
            rewards = (rewards - r_mean) / r_std

            logger.info(
                "轨迹数据已归一化: state_std range=[%.4f, %.4f], reward_std=%.4f",
                float(s_std.min()), float(s_std.max()), r_std,
            )

        # 转换为 Tensor 并保存
        self.states = torch.as_tensor(states, dtype=torch.float32)
        self.actions = torch.as_tensor(actions, dtype=torch.long)
        self.rewards = torch.as_tensor(rewards, dtype=torch.float32)

    def normalize_states(self, raw_states: np.ndarray) -> np.ndarray:
        """用本数据集的统计量归一化外部 states（如 env.states）。"""
        if self.norm_stats is None:
            return raw_states
        s_mean = self.norm_stats["state_mean"]
        s_std = np.maximum(self.norm_stats["state_std"], 1e-8)
        return (raw_states - s_mean) / s_std

    def __len__(self) -> int:
        return self.states.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.states[idx], self.actions[idx], self.rewards[idx]

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        normalize: bool = False,
        norm_stats: Optional[Dict[str, np.ndarray]] = None,
    ) -> "TrajectoryDataset":
        """从 .npz 文件加载 DP Planner 保存的轨迹数据。"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"轨迹文件不存在: {path}")

        data = np.load(str(path))

        required_keys = {"states", "actions", "rewards"}
        missing = required_keys - set(data.keys())
        if missing:
            raise KeyError(f".npz 文件缺少必要的键: {missing}，可用的键: {list(data.keys())}")

        return cls(
            states=data["states"],
            actions=data["actions"],
            rewards=data["rewards"],
            normalize=normalize,
            norm_stats=norm_stats,
        )
