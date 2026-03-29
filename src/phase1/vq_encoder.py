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
    # LSTM hidden_dim=128
    # 通过时间维聚合得到轨迹级 summary，再投影到 latent_dim=16

    说明:
        本分支保持论文接口 `q_θe(z_e | s_demo, a_demo, r_demo)` 不变，
        只调整 encoder 的内部 summary 方式：
        1. reward 仍作为输入通道，但先按训练集统计量做标准化；
        2. 不再只取最后隐藏状态，而是对整段 LSTM 输出做 mean-pooling 与
           max-pooling，再通过 MLP projection head 得到 z_e。

        这样做的目标是增强不同示范轨迹之间的可分性，避免“最后一步 summary"
        过强地把整条 horizon 压成近常数向量。

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

        self.input_dim = state_dim + action_dim + 1
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )

        # 时间维聚合后的 summary = [mean_pool(outputs), max_pool(outputs)]
        self.summary_dim = hidden_dim * 2
        self.projection_head = nn.Sequential(
            nn.Linear(self.summary_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.register_buffer("reward_mean", torch.zeros(1, dtype=torch.float32))
        self.register_buffer("reward_std", torch.ones(1, dtype=torch.float32))
        self.reward_norm_eps = 1e-6

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """初始化 LSTM 与 projection head 参数。"""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        for module in self.projection_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def set_reward_normalization(
        self,
        reward_mean: float | Tensor,
        reward_std: float | Tensor,
        eps: float = 1e-6,
    ) -> None:
        """设置 reward 标准化统计量。"""
        mean_tensor = torch.as_tensor(reward_mean, dtype=self.reward_mean.dtype, device=self.reward_mean.device).reshape(1)
        std_tensor = torch.as_tensor(reward_std, dtype=self.reward_std.dtype, device=self.reward_std.device).reshape(1)
        safe_std = torch.clamp(std_tensor, min=max(float(eps), 1e-12))
        with torch.no_grad():
            self.reward_mean.copy_(mean_tensor)
            self.reward_std.copy_(safe_std)
        self.reward_norm_eps = float(max(eps, 1e-12))

    def get_reward_normalization(self) -> dict:
        """返回当前 encoder 内保存的 reward 标准化统计量。"""
        return {
            "reward_mean": float(self.reward_mean.item()),
            "reward_std": float(self.reward_std.item()),
            "reward_norm_eps": float(self.reward_norm_eps),
        }

    def get_output_parameterization(self) -> dict:
        """返回 encoder 输出侧的实现细节配置。"""
        return {
            "temporal_pooling": "mean_and_max",
            "latent_head": "two_layer_mlp_projection",
            "latent_dim": int(self.latent_dim),
        }

    def _normalize_rewards(self, r_demo: Tensor, dtype: torch.dtype) -> Tensor:
        reward_mean = self.reward_mean.to(device=r_demo.device, dtype=dtype)
        reward_std = self.reward_std.to(device=r_demo.device, dtype=dtype)
        reward_std = torch.clamp(reward_std, min=max(self.reward_norm_eps, 1e-12))
        return (r_demo.to(dtype=dtype) - reward_mean) / reward_std

    def build_lstm_input(self, s_demo: Tensor, a_demo: Tensor, r_demo: Tensor) -> Tensor:
        """构造论文定义的 encoder 输入张量。"""
        if s_demo.ndim != 3:
            raise ValueError(f"s_demo 应为 3D 张量 (batch, h, state_dim)，实际为 {tuple(s_demo.shape)}")
        if a_demo.ndim != 2:
            raise ValueError(f"a_demo 应为 2D 张量 (batch, h)，实际为 {tuple(a_demo.shape)}")
        if r_demo.ndim != 2:
            raise ValueError(f"r_demo 应为 2D 张量 (batch, h)，实际为 {tuple(r_demo.shape)}")

        batch_size, horizon, state_dim = s_demo.shape
        if state_dim != self.state_dim:
            raise ValueError(f"state_dim 不匹配: actual={state_dim}, expected={self.state_dim}")
        if a_demo.shape != (batch_size, horizon):
            raise ValueError(
                f"a_demo shape 不匹配: actual={tuple(a_demo.shape)}, expected=({batch_size}, {horizon})"
            )
        if r_demo.shape != (batch_size, horizon):
            raise ValueError(
                f"r_demo shape 不匹配: actual={tuple(r_demo.shape)}, expected=({batch_size}, {horizon})"
            )
        if not torch.isfinite(s_demo).all():
            raise ValueError("s_demo 中存在 NaN/Inf")
        if not torch.isfinite(r_demo).all():
            raise ValueError("r_demo 中存在 NaN/Inf")

        feature_dtype = torch.float32
        a_onehot = F.one_hot(a_demo.long(), num_classes=self.action_dim).to(dtype=feature_dtype)
        normalized_rewards = self._normalize_rewards(r_demo, dtype=feature_dtype).unsqueeze(-1)
        lstm_input = torch.cat(
            [
                s_demo.to(dtype=feature_dtype),
                a_onehot.to(dtype=feature_dtype),
                normalized_rewards.to(dtype=feature_dtype),
            ],
            dim=-1,
        )

        if not torch.isfinite(lstm_input).all():
            raise ValueError("build_lstm_input 输出中存在 NaN/Inf")
        return lstm_input.contiguous()

    def summarize_temporal_outputs(self, lstm_outputs: Tensor) -> Tensor:
        """对整段 LSTM 输出做时间维聚合。"""
        if lstm_outputs.ndim != 3:
            raise ValueError(f"lstm_outputs 应为 3D 张量，实际为 {tuple(lstm_outputs.shape)}")
        mean_pool = torch.mean(lstm_outputs, dim=1)
        max_pool = torch.amax(lstm_outputs, dim=1)
        return torch.cat([mean_pool, max_pool], dim=-1)

    def forward(self, s_demo: Tensor, a_demo: Tensor, r_demo: Tensor) -> Tensor:
        """编码示范轨迹为连续嵌入 z_e。"""
        lstm_input = self.build_lstm_input(s_demo, a_demo, r_demo)
        lstm_outputs, _ = self.lstm(lstm_input)
        trajectory_summary = self.summarize_temporal_outputs(lstm_outputs)
        z_e = self.projection_head(trajectory_summary)

        if not torch.isfinite(z_e).all():
            raise ValueError("encoder 输出 z_e 中存在 NaN/Inf")
        return z_e
