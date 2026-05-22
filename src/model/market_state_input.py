"""Shared market state input layers for archetype models."""

from __future__ import annotations

import torch
from torch import nn


class MarketStateInputEncoder(nn.Module):
    """三路市场状态输入层。

    ``states``、``relative_states`` 和 ``trend_states`` 分别先进入独立 adapter，
    再在 embedding 层融合。这样避免三类分布和语义不同的原始特征被一个
    Linear 直接混合，也让后续 ablation 可以清楚拆分每一路输入的贡献。
    """

    def __init__(
        self,
        state_dim: int,
        relative_state_dim: int,
        trend_state_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.relative_state_dim = relative_state_dim
        self.trend_state_dim = trend_state_dim
        self.hidden_dim = hidden_dim

        self.state_adapter = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.relative_state_adapter = nn.Sequential(
            nn.Linear(relative_state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.trend_state_adapter = nn.Sequential(
            nn.Linear(trend_state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(
        self,
        states: torch.Tensor,
        relative_states: torch.Tensor,
        trend_states: torch.Tensor,
    ) -> torch.Tensor:
        """返回融合后的逐时间步市场 embedding，形状 ``[batch, horizon, hidden_dim]``。"""

        validate_market_state_inputs(
            states=states,
            relative_states=relative_states,
            trend_states=trend_states,
            state_dim=self.state_dim,
            relative_state_dim=self.relative_state_dim,
            trend_state_dim=self.trend_state_dim,
        )
        state_emb = self.state_adapter(states.float())
        relative_emb = self.relative_state_adapter(relative_states.float())
        trend_emb = self.trend_state_adapter(trend_states.float())
        return self.fusion(torch.cat([state_emb, relative_emb, trend_emb], dim=-1))


def validate_market_state_inputs(
    *,
    states: torch.Tensor,
    relative_states: torch.Tensor,
    trend_states: torch.Tensor,
    state_dim: int,
    relative_state_dim: int,
    trend_state_dim: int,
) -> None:
    """校验三路市场状态输入的 batch、horizon 和特征维度。"""

    if states.ndim != 3:
        raise ValueError("states must have shape [batch, horizon, state_dim]")
    if relative_states.ndim != 3:
        raise ValueError(
            "relative_states must have shape [batch, horizon, relative_state_dim]"
        )
    if trend_states.ndim != 3:
        raise ValueError(
            "trend_states must have shape [batch, horizon, trend_state_dim]"
        )
    if relative_states.shape[:2] != states.shape[:2]:
        raise ValueError("relative_states and states must share [batch, horizon]")
    if trend_states.shape[:2] != states.shape[:2]:
        raise ValueError("trend_states and states must share [batch, horizon]")
    if states.shape[-1] != state_dim:
        raise ValueError(f"states last dim must be {state_dim}, got {states.shape[-1]}")
    if relative_states.shape[-1] != relative_state_dim:
        raise ValueError(
            "relative_states last dim must be "
            f"{relative_state_dim}, got {relative_states.shape[-1]}"
        )
    if trend_states.shape[-1] != trend_state_dim:
        raise ValueError(
            "trend_states last dim must be "
            f"{trend_state_dim}, got {trend_states.shape[-1]}"
        )


__all__ = ["MarketStateInputEncoder", "validate_market_state_inputs"]
