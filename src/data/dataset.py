"""PyTorch Dataset 封装 — DP 轨迹数据用于 VQ 训练

封装 (s_demo, a_demo, r_demo) 三元组，支持从 .npz 文件加载
DP Planner 生成的示范轨迹。
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    """DP 示范轨迹数据集，用于 VQ Encoder-Decoder 训练。

    每个样本为一条 horizon 长度的示范轨迹 (s_demo, a_demo, r_demo)。

    # Section 4.1: VQ 训练数据
    # 输入: DP Planner 生成的 30k 条示范轨迹
    # 每条轨迹包含 (s_demo, a_demo, r_demo)
    """

    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
    ) -> None:
        """
        Args:
            states: 状态序列，shape (N, h, state_dim)
            actions: 动作序列，shape (N, h)，值域 {0, 1, 2}
            rewards: 奖励序列，shape (N, h)

        Raises:
            ValueError: 输入数组的第一维（样本数 N）不一致，
                        或 states 不是 3D / actions 不是 2D / rewards 不是 2D，
                        或 actions 和 rewards 的 horizon 维度与 states 不匹配。
        """
        # --- 维度校验 ---
        if states.ndim != 3:
            raise ValueError(
                f"states 应为 3D (N, h, state_dim)，实际为 {states.ndim}D"
            )
        if actions.ndim != 2:
            raise ValueError(
                f"actions 应为 2D (N, h)，实际为 {actions.ndim}D"
            )
        if rewards.ndim != 2:
            raise ValueError(
                f"rewards 应为 2D (N, h)，实际为 {rewards.ndim}D"
            )

        n_states, h_states = states.shape[0], states.shape[1]
        n_actions, h_actions = actions.shape
        n_rewards, h_rewards = rewards.shape

        if not (n_states == n_actions == n_rewards):
            raise ValueError(
                f"样本数不一致: states={n_states}, actions={n_actions}, rewards={n_rewards}"
            )
        if h_actions != h_states or h_rewards != h_states:
            raise ValueError(
                f"horizon 长度不一致: states h={h_states}, "
                f"actions h={h_actions}, rewards h={h_rewards}"
            )

        # 转换为 Tensor 并保存
        self.states = torch.as_tensor(states, dtype=torch.float32)
        self.actions = torch.as_tensor(actions, dtype=torch.long)
        self.rewards = torch.as_tensor(rewards, dtype=torch.float32)

    def __len__(self) -> int:
        return self.states.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回第 idx 条轨迹的 (s_demo, a_demo, r_demo)。

        Returns:
            s_demo: (h, state_dim) float32
            a_demo: (h,) long
            r_demo: (h,) float32
        """
        return self.states[idx], self.actions[idx], self.rewards[idx]

    @classmethod
    def from_npz(cls, path: str | Path) -> "TrajectoryDataset":
        """从 .npz 文件加载 DP Planner 保存的轨迹数据。

        预期 .npz 文件包含键: 'states', 'actions', 'rewards'
        （与 DPPlanner.generate_trajectories() 的输出格式一致）。

        Args:
            path: .npz 文件路径

        Returns:
            TrajectoryDataset 实例

        Raises:
            FileNotFoundError: 文件不存在
            KeyError: .npz 文件缺少必要的键
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"轨迹文件不存在: {path}")

        data = np.load(str(path))

        required_keys = {"states", "actions", "rewards"}
        missing = required_keys - set(data.keys())
        if missing:
            raise KeyError(
                f".npz 文件缺少必要的键: {missing}，"
                f"可用的键: {list(data.keys())}"
            )

        return cls(
            states=data["states"],
            actions=data["actions"],
            rewards=data["rewards"],
        )
