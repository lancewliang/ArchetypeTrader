"""Archetype selector 网络: 输入 s^sel，输出 K 类离散 logits 与 critic value。

设计文档锚点: Phase II 执行计划 §Step 4。

职责:
- 输入 s^sel，输出 K 类离散 logits 与 critic value。
- 支持 deterministic=False/True 两种推理模式。
- 默认 LayerNorm；RunningMeanStd 只作为 ablation。
- position_continuity=true 时强制要求状态包含 prev_terminal_position 编码。
- dead code mask 基于 Phase I global usage（不是 Phase II subset）。
- mask 作用于 logits = -inf。

关键约束:
- state_dim_breakdown 必须与配置和 report 一致。
- 多档仓位时禁止 one_hot_3。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from src.config.phase2_config import SelectorNetworkConfig


class ArchetypeSelector(nn.Module):
    """Archetype selector 网络。

    架构: MLP with LayerNorm，共享 trunk + actor head + critic head。

    使用方式::

        selector = ArchetypeSelector(state_dim=20, num_codes=10, config=config)
        logits, value = selector(obs)
        logits_masked = selector.apply_dead_code_mask(logits, dead_mask)
    """

    def __init__(
        self,
        state_dim: int,
        num_codes: int,
        config: SelectorNetworkConfig,
    ) -> None:
        """初始化 selector 网络。

        Parameters
        ----------
        state_dim : selector 状态维度（feature_dim + position_dim）。
        num_codes : archetype 数量 K。
        config : 网络超参配置。
        """
        super().__init__()
        self.state_dim = state_dim
        self.num_codes = num_codes
        self.config = config

        # 构建共享 trunk
        layers: List[nn.Module] = []
        in_dim = state_dim
        activation_fn = nn.ReLU if config.activation == "relu" else nn.GELU
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if config.use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(activation_fn())
            in_dim = hidden_dim

        self.trunk = nn.Sequential(*layers)
        trunk_out_dim = config.hidden_dims[-1] if config.hidden_dims else state_dim

        # Actor head: 输出 K 类 logits
        self.actor_head = nn.Linear(trunk_out_dim, num_codes)

        # Critic head: 输出 scalar value
        self.critic_head = nn.Linear(trunk_out_dim, 1)

        # 初始化权重
        self._init_weights()

    def _init_weights(self) -> None:
        """正交初始化: trunk 用 sqrt(2) gain (ReLU)，actor head 用小 gain，critic 用 1.0。"""
        for module in self.trunk.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=2**0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # actor head 用更小的 gain，让初始策略接近均匀
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        if self.actor_head.bias is not None:
            nn.init.zeros_(self.actor_head.bias)
        # critic head 用标准 gain
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        if self.critic_head.bias is not None:
            nn.init.zeros_(self.critic_head.bias)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播。

        Parameters
        ----------
        obs : [batch, state_dim] selector 状态。

        Returns
        -------
        logits : [batch, K] 离散 action logits。
        value : [batch] critic value。
        """
        features = self.trunk(obs)
        logits = self.actor_head(features)
        value = self.critic_head(features).squeeze(-1)
        return logits, value

    @staticmethod
    def apply_dead_code_mask(
        logits: torch.Tensor,
        dead_code_mask: torch.Tensor,
    ) -> torch.Tensor:
        """将 dead code 的 logits 设为 -inf。

        Parameters
        ----------
        logits : [batch, K]。
        dead_code_mask : [K] bool tensor，True 表示 dead code。

        Returns
        -------
        masked_logits : [batch, K]。
        """
        masked = logits.clone()
        masked[:, dead_code_mask] = float("-inf")
        return masked

    def state_dim_breakdown(self) -> Dict[str, int]:
        """返回 state 维度分解，用于校验。"""
        return {
            "total_state_dim": self.state_dim,
            "num_codes": self.num_codes,
        }
