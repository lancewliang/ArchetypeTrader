"""Refinement Agent — Step 级别策略适配 RL Agent

# Section 4.3: Step-level policy adapter
# M_ref = {S_ref, A_ref, R_ref}
# s_ref = [s_ref1, s_ref2]
#   s_ref1 = 市场观测 (market_dim)
#   s_ref2 = 上下文向量 [e_a_sel, a_base, R_arche, τ_remain]
#            (code_dim + 1 + 1 + 1 = 19)
# a_ref ∈ {-1, 0, 1}（减仓、不变、加仓）
#
# 使用 AdaLN 将 s_ref2 条件化到 s_ref1 的处理中，
# 然后通过 MLP 输出策略分布和状态价值。
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

from src.phase3.adaln import AdaptiveLayerNorm


class RefinementAgent(nn.Module):
    """原型精炼 RL Agent（Actor-Critic + AdaLN 架构）。

    # Section 4.3: Step-level policy adapter
    # 输入:
    #   s_ref1: (batch, market_dim) 市场观测
    #   s_ref2: (batch, context_dim) 上下文 [e_a_sel, a_base, R_arche, τ_remain]
    # 输出:
    #   action_probs: (batch, 3) 调整信号概率 {-1, 0, 1}
    #   value: (batch, 1) 状态价值估计
    #
    # 网络结构:
    #   1. 市场编码: Linear(market_dim, 64) → ReLU
    #   2. AdaLN 条件化: AdaLN(encoded, s_ref2)
    #   3. 共享 MLP: Linear(64, 64) → ReLU
    #   4. 策略头: Linear(64, 3) → Softmax → action_probs
    #   5. 价值头: Linear(64, 1) → value

    Args:
        market_dim: 市场观测维度 (s_ref1), 默认 45
        context_dim: 上下文维度 (s_ref2 = code_dim + 1 + 1 + 1 = 19)
    """

    def __init__(self, market_dim: int, context_dim: int) -> None:
        super().__init__()
        self.market_dim = market_dim
        self.context_dim = context_dim

        hidden_dim = 64

        # Section 4.3: 市场观测编码层
        self.market_encoder = nn.Sequential(
            nn.Linear(market_dim, hidden_dim),
            nn.ReLU(),
        )

        # Section 4.3: AdaLN 条件化 — 用 s_ref2 调制编码后的市场特征
        self.adaln = AdaptiveLayerNorm(
            feature_dim=hidden_dim,
            condition_dim=context_dim,
        )

        # Section 4.3: 共享 MLP 层
        self.shared_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Section 4.3: 策略头 — 输出 3 个调整信号 {-1, 0, 1} 的概率
        self.policy_head = nn.Linear(hidden_dim, 3)

        # Section 4.3: 价值头 — 输出状态价值估计
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, s_ref1: Tensor, s_ref2: Tensor) -> Tuple[Tensor, Tensor]:
        """输出调整信号的策略分布和状态价值。

        # Section 4.3: Step-level policy adapter
        # 1. 编码市场观测 s_ref1
        # 2. 使用 AdaLN 将 s_ref2 条件化到编码特征
        # 3. 共享 MLP 提取高层特征
        # 4. 策略头: Softmax 输出 {-1, 0, 1} 概率分布
        # 5. 价值头: 输出标量价值

        Args:
            s_ref1: (batch, market_dim) 市场观测
            s_ref2: (batch, context_dim) 上下文 [e_a_sel, a_base, R_arche, τ_remain]

        Returns:
            action_probs: (batch, 3) 调整信号概率 {-1, 0, 1}
            value: (batch, 1) 状态价值估计
        """
        # Step 1: 编码市场观测
        encoded = self.market_encoder(s_ref1)  # (batch, 64)

        # Step 2: AdaLN 条件化 — 用上下文向量调制市场特征
        # AdaLN(x, c) = γ(c) × LayerNorm(x) + β(c)
        conditioned = self.adaln(encoded, s_ref2)  # (batch, 64)

        # Step 3: 共享 MLP
        features = self.shared_mlp(conditioned)  # (batch, 64)

        # Step 4: 策略头 — Softmax 输出概率分布
        logits = self.policy_head(features)  # (batch, 3)
        action_probs = torch.softmax(logits, dim=-1)  # (batch, 3)

        # Step 5: 价值头 — 状态价值估计
        value = self.value_head(features)  # (batch, 1)

        return action_probs, value
