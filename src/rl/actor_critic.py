"""Actor-Critic 封装: 统一 act / evaluate_actions 接口。

设计文档锚点: Phase II 执行计划 §Step 4。

职责:
- act(obs, deterministic=False) 返回 action / log_prob / value。
- evaluate_actions(obs, action) 返回 log_prob / entropy / value。
- 封装 ArchetypeSelector，提供 PPO 训练所需的标准接口。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch.distributions import Categorical

from src.models.archetype_selector import ArchetypeSelector


@dataclass
class ActOutput:
    """act() 的返回值。"""
    action: torch.Tensor      # [batch] int tensor
    log_prob: torch.Tensor    # [batch] float tensor
    value: torch.Tensor       # [batch] float tensor


@dataclass
class EvalOutput:
    """evaluate_actions() 的返回值。"""
    log_prob: torch.Tensor    # [batch] float tensor
    entropy: torch.Tensor     # [batch] float tensor
    value: torch.Tensor       # [batch] float tensor


class ActorCritic:
    """Actor-Critic 封装。

    使用方式::

        ac = ActorCritic(selector, dead_code_mask=mask)
        out = ac.act(obs, deterministic=False)
        eval_out = ac.evaluate_actions(obs, out.action)
    """

    def __init__(
        self,
        selector: ArchetypeSelector,
        dead_code_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """初始化。

        Parameters
        ----------
        selector : ArchetypeSelector 网络实例。
        dead_code_mask : [K] bool tensor，True 表示 dead code。
        """
        self.selector = selector
        self.dead_code_mask = dead_code_mask

    def _mask_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """应用 dead code mask。"""
        if self.dead_code_mask is not None:
            mask = self.dead_code_mask.to(device=logits.device)
            return ArchetypeSelector.apply_dead_code_mask(logits, mask)
        return logits

    def act(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> ActOutput:
        """选择动作。

        Parameters
        ----------
        obs : [batch, state_dim]。
        deterministic : True 时返回 argmax；False 时从 Categorical 采样。

        Returns
        -------
        ActOutput : action / log_prob / value。
        """
        logits, value = self.selector(obs)
        logits = self._mask_logits(logits)
        dist = Categorical(logits=logits)

        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        return ActOutput(action=action, log_prob=log_prob, value=value)

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
    ) -> EvalOutput:
        """评估已有动作的 log_prob / entropy / value。

        Parameters
        ----------
        obs : [batch, state_dim]。
        action : [batch] int tensor。

        Returns
        -------
        EvalOutput : log_prob / entropy / value。
        """
        logits, value = self.selector(obs)
        logits = self._mask_logits(logits)
        dist = Categorical(logits=logits)

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return EvalOutput(log_prob=log_prob, entropy=entropy, value=value)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """仅获取 critic value（用于 GAE bootstrap）。

        Parameters
        ----------
        obs : [batch, state_dim]。

        Returns
        -------
        value : [batch] float tensor。
        """
        _, value = self.selector(obs)
        return value
