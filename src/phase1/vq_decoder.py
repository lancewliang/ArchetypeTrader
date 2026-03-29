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

    实现细节:
        论文只规定 decoder 条件分布 `p_theta_d(a_hat_demo | s_demo, z_q)`，
        并未写死必须使用“裸 concat 后两层线性层”的具体参数化方式。
        本分支将 decoder 改为 state 分支 + latent 调制分支：
        1. 先将 states 投影为 state_hidden。
        2. 再由 z_q 生成按通道的 gain / bias，对 state_hidden 做条件调制。
        3. 最后经 MLP 输出 action logits。

        这样仍然是 `p(a | s, z_q)`，但可以显著削弱“只靠 state shortcut” 的实现偏置。

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

        # State branch: s_demo -> hidden
        self.state_projection = nn.Linear(state_dim, hidden_dim)

        # Latent branch: z_q -> per-channel modulation
        self.latent_gain = nn.Linear(code_dim, hidden_dim)
        self.latent_bias = nn.Linear(code_dim, hidden_dim)

        # 输出头仍然是 MLP decoder，只是前面的融合方式改为条件调制
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.output_mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """初始化 decoder 参数。

        只固定论文未写明的初始化细节，不改变 decoder 的输入输出定义。
        """
        nn.init.xavier_uniform_(self.state_projection.weight)
        nn.init.zeros_(self.state_projection.bias)

        nn.init.xavier_uniform_(self.latent_gain.weight)
        nn.init.zeros_(self.latent_gain.bias)

        nn.init.xavier_uniform_(self.latent_bias.weight)
        nn.init.zeros_(self.latent_bias.bias)

        for module in self.output_mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, states: Tensor, z_q: Tensor) -> Tensor:
        """根据状态和量化嵌入生成动作 logits。

        # Section 4.1: Decoder forward
        # 1. 扩展 z_q 到时间维度: (batch, code_dim) → (batch, h, code_dim)
        # 2. states 经线性层映射为 state_hidden
        # 3. z_q 生成 gain / bias，对 state_hidden 做条件调制
        # 4. 调制后的 hidden 再经 MLP 输出 action logits

        Args:
            states: 状态序列 (batch, h, state_dim)
            z_q: 量化嵌入 (batch, code_dim)

        Returns:
            action_logits: 动作 logits (batch, h, action_dim)
        """
        if states.ndim != 3:
            raise ValueError(f"states 应为 3D 张量 (batch, h, state_dim)，实际为 {tuple(states.shape)}")
        if z_q.ndim != 2:
            raise ValueError(f"z_q 应为 2D 张量 (batch, code_dim)，实际为 {tuple(z_q.shape)}")
        batch, horizon, state_dim = states.shape
        if state_dim != self.state_dim:
            raise ValueError(f"state_dim 不匹配: actual={state_dim}, expected={self.state_dim}")
        if z_q.shape != (batch, self.code_dim):
            raise ValueError(f"z_q shape 不匹配: actual={tuple(z_q.shape)}, expected=({batch}, {self.code_dim})")

        states = states.to(dtype=torch.float32)
        z_q = z_q.to(dtype=torch.float32)

        # Step 1: 扩展 z_q 到时间维度
        z_q_expanded = z_q.unsqueeze(1).expand(batch, horizon, self.code_dim)

        # Step 2: 状态分支
        state_hidden = self.state_projection(states)

        # Step 3: latent 条件调制
        gain = 1.0 + torch.tanh(self.latent_gain(z_q_expanded))
        bias = self.latent_bias(z_q_expanded)
        fused_hidden = self.fusion_norm(state_hidden * gain + bias)

        # Step 4: MLP 生成 action logits
        action_logits = self.output_mlp(fused_hidden)

        if not torch.isfinite(action_logits).all():
            raise ValueError("decoder 输出 action_logits 中存在 NaN/Inf")

        return action_logits
