"""VQ Decoder — 根据状态和量化嵌入重建动作序列

# Section 4.1: Decoder
# p_θd(â_demo | s_demo, z_q)
# 根据状态序列和量化嵌入 z_q 生成动作 logits
"""

import torch
import torch.nn as nn
from torch import Tensor


class VQDecoder(nn.Module):
    """VQ 解码器，根据状态和量化嵌入生成动作序列。

    # Section 4.1: Decoder
    # 输入: states (batch, h, state_dim) 和 z_q (batch, code_dim)
    # 1. 将 z_q 沿时间维度扩展: z_q_expanded (batch, h, code_dim)
    # 2. 拼接 [states, z_q_expanded] → (batch, h, state_dim + code_dim)
    # 3. MLP: Linear → ReLU → Linear → action_logits (batch, h, action_dim)

    Args:
        state_dim: 状态向量维度 (默认 45)
        code_dim: 码本向量维度 (默认 16)
        hidden_dim: 隐藏层维度 (默认 128)
        action_dim: 动作空间大小 (默认 3)
    """

    def __init__(
        self,
        state_dim: int,
        code_dim: int = 16,
        hidden_dim: int = 128,
        action_dim: int = 3,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.code_dim = code_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        # Section 4.1: MLP decoder
        # 输入维度 = state_dim + code_dim
        self.mlp = nn.Sequential(
            nn.Linear(state_dim + code_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, states: Tensor, z_q: Tensor) -> Tensor:
        """根据状态和量化嵌入生成动作 logits。

        # Section 4.1: Decoder forward
        # 1. 扩展 z_q 到时间维度: (batch, code_dim) → (batch, h, code_dim)
        # 2. 拼接 [states, z_q_expanded] → (batch, h, state_dim + code_dim)
        # 3. MLP 生成 action_logits (batch, h, action_dim)

        Args:
            states: 状态序列 (batch, h, state_dim)
            z_q: 量化嵌入 (batch, code_dim)

        Returns:
            action_logits: 动作 logits (batch, h, action_dim)
        """
        batch, h, _ = states.shape

        # Step 1: 扩展 z_q 到时间维度
        z_q_expanded = z_q.unsqueeze(1).expand(batch, h, self.code_dim)
        # z_q_expanded: (batch, h, code_dim)

        # Step 2: 拼接状态和量化嵌入
        decoder_input = torch.cat([states, z_q_expanded], dim=-1)
        # decoder_input: (batch, h, state_dim + code_dim)

        # Step 3: MLP 生成 action logits
        action_logits = self.mlp(decoder_input)
        # action_logits: (batch, h, action_dim)

        return action_logits
