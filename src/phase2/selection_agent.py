"""Selection Agent — Horizon 级别原型选择 RL Agent

# Section 4.2: Horizon-level RL agent
# M_sel = ⟨S_sel, A_sel, R_sel, γ⟩
# 在每个 horizon 开始时，根据当前市场状态选择原型索引 a_sel ∈ {0, ..., K-1}
# 使用 Actor-Critic 架构：共享特征提取 + 策略头 + 价值头
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple


class SelectionAgent(nn.Module):
    """原型选择 RL Agent（Actor-Critic 架构）。

    # Section 4.2: Horizon-level RL agent
    # 输入: horizon 第一个 bar 的状态向量 s_sel ∈ R^{state_dim}
    # 输出: 原型选择概率分布 π(a_sel | s_sel) 和状态价值 V(s_sel)
    #
    # 网络结构:
    #   共享层: Linear(state_dim, 128) → ReLU → Linear(128, 64) → ReLU
    #   策略头: Linear(64, K) → Softmax → action_probs
    #   价值头: Linear(64, 1) → value

    Args:
        state_dim: 状态向量维度 (默认 45)
        num_archetypes: 原型数量 K (默认 10)
    """

    def __init__(self, state_dim: int, num_archetypes: int = 10) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.num_archetypes = num_archetypes

        # Section 4.2: 共享特征提取层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # Section 4.2: 策略头 — 输出 K 个原型的选择概率
        self.policy_head = nn.Linear(64, num_archetypes)

        # Section 4.2: 价值头 — 输出状态价值估计
        self.value_head = nn.Linear(64, 1)

    def forward(self, state: Tensor) -> Tuple[Tensor, Tensor]:
        """根据当前状态输出原型选择的策略分布和状态价值。

        # Section 4.2: Horizon-level RL agent
        # 1. 共享特征提取
        # 2. 策略头: Softmax 输出概率分布
        # 3. 价值头: 输出标量价值

        Args:
            state: (batch, state_dim) — horizon 第一个 bar 的状态

        Returns:
            action_probs: (batch, K) 原型选择概率，非负且和为 1
            value: (batch, 1) 状态价值估计
        """
        # Step 1: 共享特征提取
        features = self.shared(state)  # (batch, 64)

        # Step 2: 策略头 — Softmax 输出概率分布
        logits = self.policy_head(features)  # (batch, K)
        action_probs = torch.softmax(logits, dim=-1)  # (batch, K)

        # Step 3: 价值头 — 状态价值估计
        value = self.value_head(features)  # (batch, 1)

        return action_probs, value

    def select_archetype(self, state: Tensor) -> int:
        """推理时选择原型索引。

        # Section 4.2: 从策略分布中采样原型索引
        # 使用 Categorical 分布采样，支持探索

        Args:
            state: (1, state_dim) 或 (state_dim,) — 单个状态向量

        Returns:
            选中的原型索引 ∈ {0, ..., K-1}
        """
        # 确保输入为 2D: (1, state_dim)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        with torch.no_grad():
            action_probs, _ = self.forward(state)
            # 从概率分布中采样
            dist = torch.distributions.Categorical(action_probs)
            archetype_idx = dist.sample()

        return archetype_idx.item()
