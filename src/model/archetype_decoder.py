"""Archetype action decoder."""

from __future__ import annotations

import torch
from torch import nn

from .market_state_input import MarketStateInputEncoder, validate_market_state_inputs
from .tensor_data_types import ActionLogitTensor, LatentTensor


class ArchetypeActionDecoder(nn.Module):
    """根据市场状态序列和 archetype latent 重构逐步动作 logits。

    设计原因:
        decoder 是 Phase II/III 使用 archetype 的执行入口。它必须是因果的：
        第 ``tau`` 步动作只能依赖 ``s_0...s_tau`` 和选中的 archetype，
        不能偷看 horizon 后面的未来状态。

    网络层说明:
        1. ``market_input_encoder``:
           把每个时间步的三路市场状态分别投影并融合到 ``hidden_dim``。
           decoder 不接收未来动作或 reward，因此推理阶段可以直接使用。
        2. ``z_q_seq`` 扩展:
           ``z_q`` 是每条 trajectory 一个 archetype 向量。forward 中会把它
           从 ``[batch, latent_dim]`` 扩展为 ``[batch, horizon, latent_dim]``，
           让每个时间步都知道当前要执行哪类交易原型。
        3. ``decoder_input`` 拼接:
           每个时间步输入为 ``[market_emb_t, z_q]``。状态提供当前市场上下文，
           archetype 提供全局交易风格/计划。
        4. ``lstm``:
           单向 LSTM 逐步处理 ``decoder_input``，第 ``tau`` 步 hidden 只包含
           到 ``tau`` 为止的信息，因此满足因果解码要求。
        5. ``action_head``:
           把每个时间步的 LSTM hidden 映射成 ``action_dim`` logits。训练时
           logits 用于 ``cross_entropy`` 重构 teacher action；推理时用
           ``argmax`` 得到基础动作序列。
    """

    def __init__(
        self,
        state_dim: int,
        relative_state_dim: int,
        trend_state_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 16,
        action_dim: int = 3,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.relative_state_dim = relative_state_dim
        self.trend_state_dim = trend_state_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        self.market_input_encoder = MarketStateInputEncoder(
            state_dim=state_dim,
            relative_state_dim=relative_state_dim,
            trend_state_dim=trend_state_dim,
            hidden_dim=hidden_dim,
        )
        self.lstm = nn.LSTM(
            input_size=hidden_dim + latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(
        self,
        states: torch.Tensor,
        relative_states: torch.Tensor,
        trend_states: torch.Tensor,
        z_q: LatentTensor,
    ) -> ActionLogitTensor:
        """返回动作 logits。

        输入形状:
            ``states``: ``[batch, horizon, state_dim]``
                一批市场状态序列。``batch`` 是样本数，``horizon`` 是每条
                trajectory 的时间步长度，``state_dim`` 是单步状态特征数。

            ``z_q``: ``[batch, latent_dim]``
                每条 trajectory 对应的 archetype latent/codebook 向量。
                ``z_q.shape[0]`` 必须和 ``states.shape[0]`` 一致。

        内部形状:
            ``market_emb``: ``[batch, horizon, hidden_dim]``
            ``z_q_seq``: ``[batch, horizon, latent_dim]``
            ``decoder_input``: ``[batch, horizon, hidden_dim + latent_dim]``

        输出形状:
            ``action_logits``: ``[batch, horizon, action_dim]``
        """

        validate_market_state_inputs(
            states=states,
            relative_states=relative_states,
            trend_states=trend_states,
            state_dim=self.state_dim,
            relative_state_dim=self.relative_state_dim,
            trend_state_dim=self.trend_state_dim,
        )
        if z_q.ndim != 2:
            raise ValueError("z_q must have shape [batch, latent_dim]")
        if states.shape[0] != z_q.shape[0]:
            raise ValueError("states and z_q must have the same batch size")

        batch_size, horizon, _ = states.shape
        market_emb = self.market_input_encoder(
            states,
            relative_states,
            trend_states,
        )
        z_q_seq = z_q.float().unsqueeze(1).expand(batch_size, horizon, self.latent_dim)
        decoder_input = torch.cat([market_emb, z_q_seq], dim=-1)
        hidden_seq, _ = self.lstm(decoder_input)
        return self.action_head(hidden_seq)


__all__ = ["ArchetypeActionDecoder"]
