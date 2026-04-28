"""VQ Decoder — 可配置动作序列解码器 + single-trade 推理约束。

# Section 4.1: Decoder
# p_θd(a_base | s, z_q)
#
# 支持三种骨干网络：
# - lstm_causal: 单向 LSTM（生产默认，严格时间因果）
# - bilstm: 双向 LSTM（兼容旧 checkpoint）
# - causal_transformer: 因果掩码 Transformer（可选）
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


VALID_DECODER_ARCHS = {"lstm_causal", "bilstm", "causal_transformer"}


def _build_sinusoidal_position_encoding(
    seq_len: int,
    dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """构建标准正弦位置编码，shape=(1, seq_len, dim)。"""
    positions = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)
    pe = torch.zeros(seq_len, dim, device=device, dtype=dtype)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=dtype)
        * (-math.log(10000.0) / max(dim, 1)),
    )
    pe[:, 0::2] = torch.sin(positions * div_term)
    pe[:, 1::2] = torch.cos(positions * div_term[: pe[:, 1::2].shape[1]])
    return pe.unsqueeze(0)


class VQDecoder(nn.Module):
    """VQ 解码器（可切换 LSTM / causal Transformer）。"""

    def __init__(
        self,
        state_dim: int,
        code_dim: int = 16,
        hidden_dim: int = 128,
        action_dim: int = 3,
        decoder_arch: str = "lstm_causal",
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        transformer_ffn_dim: int | None = None,
        transformer_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if decoder_arch not in VALID_DECODER_ARCHS:
            raise ValueError(
                f"不支持的 decoder_arch={decoder_arch!r}，可选: {sorted(VALID_DECODER_ARCHS)}",
            )

        self.state_dim = state_dim
        self.code_dim = code_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.decoder_arch = decoder_arch

        self.transformer_layers = int(transformer_layers)
        self.transformer_heads = int(transformer_heads)
        self.transformer_ffn_dim = (
            int(transformer_ffn_dim)
            if transformer_ffn_dim is not None
            else int(hidden_dim * 4)
        )
        self.transformer_dropout = float(transformer_dropout)

        input_dim = state_dim + code_dim
        output_input_dim: int

        self.lstm: nn.LSTM | None
        self.input_proj: nn.Linear | None
        self.transformer: nn.TransformerEncoder | None

        if decoder_arch in {"lstm_causal", "bilstm"}:
            is_bidirectional = decoder_arch == "bilstm"
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                batch_first=True,
                bidirectional=is_bidirectional,
            )
            self.input_proj = None
            self.transformer = None
            output_input_dim = hidden_dim * (2 if is_bidirectional else 1)
        else:
            if self.transformer_heads < 1:
                raise ValueError("transformer_heads 必须 >= 1")
            if hidden_dim % self.transformer_heads != 0:
                raise ValueError(
                    "hidden_dim 必须能被 transformer_heads 整除，"
                    f"当前 hidden_dim={hidden_dim}, transformer_heads={self.transformer_heads}",
                )
            if self.transformer_layers < 1:
                raise ValueError("transformer_layers 必须 >= 1")
            if self.transformer_ffn_dim < hidden_dim:
                raise ValueError("transformer_ffn_dim 建议 >= hidden_dim")

            self.lstm = None
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=self.transformer_heads,
                dim_feedforward=self.transformer_ffn_dim,
                dropout=self.transformer_dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=self.transformer_layers,
            )
            output_input_dim = hidden_dim

        self.output_proj = nn.Linear(output_input_dim, action_dim)

    def _forward_backbone(self, decoder_inputs: Tensor) -> Tensor:
        """骨干网络前向，输出时序 hidden states。"""
        if self.decoder_arch in {"lstm_causal", "bilstm"}:
            if self.lstm is None:
                raise RuntimeError("decoder 架构与内部模块不一致: lstm 为空")
            seq_out, _ = self.lstm(decoder_inputs)
            return seq_out

        if self.input_proj is None or self.transformer is None:
            raise RuntimeError("decoder 架构与内部模块不一致: transformer 组件缺失")

        _, h, _ = decoder_inputs.shape
        hidden = self.input_proj(decoder_inputs)
        hidden = hidden + _build_sinusoidal_position_encoding(
            h, hidden.shape[-1], device=hidden.device, dtype=hidden.dtype,
        )
        causal_mask = torch.triu(
            torch.ones(h, h, device=hidden.device, dtype=torch.bool),
            diagonal=1,
        )
        return self.transformer(hidden, mask=causal_mask)

    def forward(self, states: Tensor, z_q: Tensor) -> Tensor:
        """生成 action logits。

        Args:
            states: (batch, h, state_dim)
            z_q: (batch, code_dim)

        Returns:
            action_logits: (batch, h, action_dim)
        """
        batch, h, _ = states.shape
        z_q_expanded = z_q.unsqueeze(1).expand(batch, h, self.code_dim)
        decoder_inputs = torch.cat([states, z_q_expanded], dim=-1)

        hidden_states = self._forward_backbone(decoder_inputs)
        action_logits = self.output_proj(hidden_states)
        return action_logits

    def _decode_logits_with_single_trade_constraint(self, logits: Tensor) -> Tensor:
        """在给定 logits 上搜索最优 single-trade 动作序列。"""
        log_probs = torch.log_softmax(logits, dim=-1)  # (batch, h, action_dim)

        batch, h, num_actions = log_probs.shape
        device = logits.device

        # 前缀和: prefix_sum[:, t, a] = Σ_{i=0}^{t} log_prob[:, i, a]
        prefix_sum = torch.cumsum(log_probs, dim=1)  # (batch, h, action_dim)

        # suffix_sum[:, t, a] = Σ_{i=t}^{h-1} log_prob[:, i, a]
        suffix_sum = prefix_sum[:, -1:, :] - prefix_sum + log_probs  # (batch, h, action_dim)

        # ---- 情况 1: 无变化，全程同一个动作 ----
        no_change_scores = prefix_sum[:, -1, :]  # (batch, num_actions)
        no_change_best_score, no_change_best_a = no_change_scores.max(dim=1)

        # ---- 情况 2: 在第 t 步变化 (前 t 步 a1, 后 h-t 步 a2, a1 ≠ a2) ----
        if h > 1:
            prefix_part = prefix_sum[:, : h - 1, :]  # (batch, h-1, num_actions)
            suffix_part = suffix_sum[:, 1:, :]  # (batch, h-1, num_actions)

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

    @torch.no_grad()
    def decode_with_single_trade_constraint(
        self, states: Tensor, z_q: Tensor,
    ) -> Tensor:
        """推理时使用: 在 logits 上施加 single-trade 约束（全向量化实现）。"""
        logits = self.forward(states, z_q)  # (batch, h, action_dim)
        return self._decode_logits_with_single_trade_constraint(logits)

    @torch.no_grad()
    def decode_causally_with_single_trade_constraint(
        self, states: Tensor, z_q: Tensor,
    ) -> Tensor:
        """按前缀逐步解码，避免当前动作看到 horizon 后续状态。

        说明:
        - 第 τ 步动作只允许依赖 states[:, :τ+1, :]；
        - 每一步先在当前前缀上做 single-trade 搜索，再取最后一个动作；
        - 为保持在线执行语义，一旦动作相对历史首次切换，后续将锁定该动作，
          防止随着前缀变长又“改回去”而产生多次换向。
        """
        batch, h, _ = states.shape
        device = states.device

        actions = torch.empty((batch, h), dtype=torch.long, device=device)
        current_actions: Tensor | None = None
        change_used = torch.zeros(batch, dtype=torch.bool, device=device)

        for step in range(h):
            prefix_states = states[:, : step + 1, :]
            prefix_logits = self.forward(prefix_states, z_q)
            proposed_actions = self._decode_logits_with_single_trade_constraint(prefix_logits)[:, -1]

            if step == 0:
                current_actions = proposed_actions
            else:
                assert current_actions is not None
                switch_mask = (~change_used) & (proposed_actions != current_actions)
                change_used = change_used | switch_mask
                current_actions = torch.where(switch_mask, proposed_actions, current_actions)

            actions[:, step] = current_actions

        return actions
