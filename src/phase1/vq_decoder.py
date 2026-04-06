"""VQ Decoder — BiLSTM 动作序列解码器 + Positional Encoding + Single-Trade 推理约束

# Section 4.1: Decoder
# p_θd(â_demo | s_demo, z_q)
#
# BiLSTM decoder: 双向 LSTM 让 decoder 能同时看到过去和未来的 state 信息，
# 改善方向性预测（解决单向 LSTM 的 long 坍缩问题）。
#
# 增强: 加入 learnable positional encoding，帮助 decoder 精确定位 change point。
# BiLSTM 虽然有隐式的位置感知，但显式 PE 能让模型更容易学到
# "change point 通常出现在 horizon 的哪个位置"。
#
# 训练时: BiLSTM 逐步预测 action logits，用加权 CrossEntropyLoss 训练。
# 推理时: 在 BiLSTM 输出的 logits 上施加 single-trade 约束后处理。
"""

import math

import torch
import torch.nn as nn
from torch import Tensor


class VQDecoder(nn.Module):
    """BiLSTM 解码器 + positional encoding + single-trade 推理约束。

    Args:
        state_dim: 状态向量维度 (默认 45)
        code_dim: 码本向量维度 (默认 16)
        hidden_dim: LSTM 单方向隐藏层维度 (默认 128)
        action_dim: 动作空间大小 (默认 3)
        max_horizon: 最大 horizon 长度，用于 positional encoding (默认 128)
    """

    def __init__(
        self,
        state_dim: int,
        code_dim: int = 16,
        hidden_dim: int = 128,
        action_dim: int = 3,
        max_horizon: int = 128,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.code_dim = code_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        # Learnable positional encoding
        pe_dim = 16
        self.pe_dim = pe_dim
        self.pos_embedding = nn.Embedding(max_horizon, pe_dim)

        # BiLSTM: 输入 = state + code + positional encoding
        self.lstm = nn.LSTM(
            input_size=state_dim + code_dim + pe_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        # 输出投影: 2 * hidden_dim → action_dim
        self.output_proj = nn.Linear(2 * hidden_dim, action_dim)

    def forward(self, states: Tensor, z_q: Tensor) -> Tensor:
        """生成 action logits（训练和推理共用）。

        Args:
            states: (batch, h, state_dim)
            z_q: (batch, code_dim)

        Returns:
            action_logits: (batch, h, action_dim)
        """
        batch, h, _ = states.shape
        device = states.device

        z_q_expanded = z_q.unsqueeze(1).expand(batch, h, self.code_dim)

        # Positional encoding: (h,) → (1, h, pe_dim) → (batch, h, pe_dim)
        positions = torch.arange(h, device=device)
        pe = self.pos_embedding(positions).unsqueeze(0).expand(batch, h, self.pe_dim)

        lstm_input = torch.cat([states, z_q_expanded, pe], dim=-1)

        lstm_out, _ = self.lstm(lstm_input)  # (batch, h, 2 * hidden_dim)
        action_logits = self.output_proj(lstm_out)  # (batch, h, action_dim)
        return action_logits

    @torch.no_grad()
    def decode_with_single_trade_constraint(
        self, states: Tensor, z_q: Tensor,
    ) -> Tensor:
        """推理时使用: 在 logits 上施加 single-trade 约束（全向量化实现）。

        搜索最优的 single-change 分割点，使得动作序列为
        "action_a × t + action_b × (h-t)" 的形式，且总 log-probability 最大。

        Args:
            states: (batch, h, state_dim)
            z_q: (batch, code_dim)

        Returns:
            actions: (batch, h) 符合 single-trade 约束的动作序列
        """
        logits = self.forward(states, z_q)
        log_probs = torch.log_softmax(logits, dim=-1)

        batch, h, num_actions = log_probs.shape
        device = logits.device

        prefix_sum = torch.cumsum(log_probs, dim=1)
        suffix_sum = prefix_sum[:, -1:, :] - prefix_sum + log_probs

        # 情况 1: 无变化
        no_change_scores = prefix_sum[:, -1, :]
        no_change_best_score, no_change_best_a = no_change_scores.max(dim=1)

        # 情况 2: 在第 t 步变化
        if h > 1:
            prefix_part = prefix_sum[:, :h-1, :]
            suffix_part = suffix_sum[:, 1:, :]
            combined = prefix_part.unsqueeze(-1) + suffix_part.unsqueeze(-2)

            diag_mask = torch.eye(num_actions, dtype=torch.bool, device=device)
            combined.masked_fill_(diag_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

            flat = combined.reshape(batch, -1)
            flat_best_score, flat_best_idx = flat.max(dim=1)

            a2_size = num_actions
            a1_size = num_actions
            best_a2_change = flat_best_idx % a2_size
            best_a1_change = (flat_best_idx // a2_size) % a1_size
            best_t_change = flat_best_idx // (a1_size * a2_size) + 1

            use_change = flat_best_score > no_change_best_score
        else:
            use_change = torch.zeros(batch, dtype=torch.bool, device=device)
            best_a1_change = no_change_best_a
            best_a2_change = no_change_best_a
            best_t_change = torch.full_like(no_change_best_a, h)

        best_a1 = torch.where(use_change, best_a1_change, no_change_best_a)
        best_a2 = torch.where(use_change, best_a2_change, no_change_best_a)
        best_t = torch.where(use_change, best_t_change, torch.full_like(best_t_change, h))

        time_indices = torch.arange(h, device=device).unsqueeze(0)
        best_t_expanded = best_t.unsqueeze(1)
        best_actions = torch.where(
            time_indices < best_t_expanded,
            best_a1.unsqueeze(1).expand(batch, h),
            best_a2.unsqueeze(1).expand(batch, h),
        )

        return best_actions
