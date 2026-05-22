"""Archetype trajectory encoder."""

from __future__ import annotations

import torch
from torch import nn

from .market_state_input import MarketStateInputEncoder
from .tensor_data_types import LatentTensor, TrajectoryTensorBatch
from .trajectory_batch import normalize_trajectory_batch


class ArchetypeTrajectoryEncoder(nn.Module):
    """把一条 demonstration trajectory 编码成连续 latent ``z_e``。

    设计原因:
        archetype 不是单个时点的市场状态，而是一整段 horizon 内
        ``状态-动作-reward`` 的联合行为模式。encoder 同时读取三路市场
        状态、动作和 reward，让 latent 能表达 DP teacher 在该 horizon
        中为什么这样交易。

    网络层说明:
        1. ``market_input_encoder``:
           把 ``states``、``relative_states`` 和 ``trend_states`` 分别投影
           到 ``hidden_dim``，再融合成逐时间步市场 embedding。
        2. ``action_embedding`` + ``action_norm``:
           把离散 teacher action id 映射成 ``hidden_dim`` 向量。这样
           short/flat/long 不再只是整数类别，而是可学习的动作语义向量；
           ``LayerNorm`` 让动作分支的数值尺度和状态、reward 分支更接近。
        3. ``reward_adapter``:
           把每个时间步的一维 reward 投影到 ``hidden_dim``。reward 分支
           让 encoder 看到 teacher 动作带来的即时收益/代价，帮助区分
           表面动作相似但盈亏路径不同的 trajectory。
        4. ``fusion``:
           在每个时间步拼接 ``market/action/reward`` 三路 embedding 后，
           压回 ``hidden_dim``。这一层学习“在该市场状态下采取该动作并
           获得该 reward”的局部行为含义。
        5. ``lstm``:
           沿 horizon 聚合局部行为 embedding，建模持仓延续、反转、止盈
           等跨时间步模式。这里使用单向 LSTM，encoder 编码整条 teacher
           trajectory；decoder 的因果性由 decoder 自身保证。
        6. ``projection``:
           把 LSTM 的最后隐状态压缩成 ``latent_dim`` 连续 latent ``z_e``。
           这个低维瓶颈迫使模型保留整条 trajectory 的核心交易风格，而
           不是保存每个时间步的细节。
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
        self.action_embedding = nn.Embedding(action_dim, hidden_dim)
        self.action_norm = nn.LayerNorm(hidden_dim)
        self.reward_adapter = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, batch: TrajectoryTensorBatch) -> LatentTensor:
        """返回每条 trajectory 的连续 latent ``z_e``，形状为 ``[batch, latent_dim]``。"""

        states, relative_states, trend_states, actions, rewards, _ = (
            normalize_trajectory_batch(batch)
        )
        market_emb = self.market_input_encoder(
            states,
            relative_states,
            trend_states,
        )
        action_emb = self.action_norm(self.action_embedding(actions.long()))
        reward_emb = self.reward_adapter(rewards.float())

        fused = self.fusion(torch.cat([market_emb, action_emb, reward_emb], dim=-1))
        _, (hidden, _) = self.lstm(fused)
        return self.projection(hidden[-1])


__all__ = ["ArchetypeTrajectoryEncoder"]
