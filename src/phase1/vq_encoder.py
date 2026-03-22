"""VQ Encoder — LSTM-based 示范轨迹编码器

# Section 4.1: LSTM-based encoder
# q_θe(z_e | s_demo, a_demo, r_demo)
# 将示范轨迹编码为连续嵌入 z_e ∈ R^{latent_dim}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class VQEncoder(nn.Module):
    """LSTM 编码器，将示范轨迹 (s_demo, a_demo, r_demo) 编码为连续嵌入 z_e。

    # Section 4.1: LSTM-based encoder
    # 输入: 拼接 [s_demo, one_hot(a_demo), r_demo] 沿特征维度
    # LSTM hidden_dim=128, 取最后隐藏状态
    # 线性投影到 latent_dim=16

    Args:
        state_dim: 状态向量维度 (默认 45)
        action_dim: 动作空间大小 (默认 3)
        hidden_dim: LSTM 隐藏层维度 (默认 128)
        latent_dim: 潜在嵌入维度 (默认 16)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 3,
        hidden_dim: int = 128,
        latent_dim: int = 16,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Section 4.1: LSTM encoder
        # 输入维度 = state_dim + action_dim (one-hot) + 1 (reward)
        input_dim = state_dim + action_dim + 1
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        # 线性投影: 最后隐藏状态 → z_e
        self.projection = nn.Linear(hidden_dim, latent_dim)

    def forward(self, s_demo: Tensor, a_demo: Tensor, r_demo: Tensor) -> Tensor:
        """编码示范轨迹为连续嵌入 z_e。

        # Section 4.1: LSTM-based encoder
        # 1. One-hot 编码 a_demo
        # 2. 拼接 [s_demo, one_hot(a_demo), r_demo] 沿特征维度
        # 3. 送入 LSTM，取最后隐藏状态
        # 4. 线性投影到 latent_dim

        Args:
            s_demo: 状态序列 (batch, h, state_dim)
            a_demo: 动作序列 (batch, h)，值域 {0, 1, 2}
            r_demo: 奖励序列 (batch, h)

        Returns:
            z_e: 连续嵌入 (batch, latent_dim)
        """
        # Step 1: One-hot 编码动作
        a_onehot = F.one_hot(a_demo.long(), num_classes=self.action_dim).float()
        # a_onehot: (batch, h, action_dim)

        # Step 2: 拼接输入特征 [s_demo, a_onehot, r_demo]
        r_expanded = r_demo.unsqueeze(-1)  # (batch, h, 1)
        lstm_input = torch.cat([s_demo, a_onehot, r_expanded], dim=-1)
        # lstm_input: (batch, h, state_dim + action_dim + 1)

        # Step 3: LSTM 编码，取最后隐藏状态
        _, (h_n, _) = self.lstm(lstm_input)
        # h_n: (1, batch, hidden_dim) — 单层单向 LSTM
        last_hidden = h_n.squeeze(0)  # (batch, hidden_dim)

        # Step 4: 线性投影到潜在空间
        z_e = self.projection(last_hidden)  # (batch, latent_dim)

        return z_e
