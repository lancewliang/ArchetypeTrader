"""PPO loss 计算: policy clip / value loss / entropy bonus / kl_demo_loss。

设计文档锚点: Phase II 执行计划 §Step 5。

职责:
- policy_clip: clipped surrogate objective。
- value_loss: clipped value loss。
- entropy_bonus: 鼓励探索。
- kl_demo_loss: 只在 is_labeled=true 上生效；masked dead code 样本 loss=0。
- approx_kl: 用于 early stop 判断。

关键约束:
- kl_demo_loss 只在 is_labeled=true 上生效。
- masked KL label 样本 loss=0。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F


@dataclass
class PPOLossOutput:
    """PPO loss 各项分解。"""
    total: torch.Tensor          # scalar tensor
    policy_loss: torch.Tensor    # scalar tensor
    value_loss: torch.Tensor     # scalar tensor
    entropy_loss: torch.Tensor   # scalar tensor
    kl_demo_loss: torch.Tensor   # scalar tensor
    approx_kl: float             # float，用于 early stop 判断
    clip_fraction: float


class PPOLoss:
    """PPO loss 计算器。

    使用方式::

        loss_fn = PPOLoss(clip_ratio=0.2, value_coef=0.5, entropy_coef=0.01, kl_demo_coef=0.1)
        output = loss_fn.compute(
            log_prob=new_log_prob, old_log_prob=old_log_prob,
            advantage=advantage, value=new_value, return_=return_,
            entropy=entropy, kl_label=kl_label, is_labeled=is_labeled,
            dead_code_mask=dead_code_mask, action=action,
        )
    """

    def __init__(
        self,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        kl_demo_coef: float = 0.1,
        num_codes: int = 10,
    ) -> None:
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.kl_demo_coef = kl_demo_coef
        self.num_codes = num_codes

    def compute(
        self,
        log_prob: torch.Tensor,
        old_log_prob: torch.Tensor,
        advantage: torch.Tensor,
        value: torch.Tensor,
        return_: torch.Tensor,
        entropy: torch.Tensor,
        kl_label: Optional[torch.Tensor] = None,
        is_labeled: Optional[torch.Tensor] = None,
        dead_code_mask: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
    ) -> PPOLossOutput:
        """计算 PPO loss 各项。

        Parameters
        ----------
        log_prob : [batch] 新策略的 log probability。
        old_log_prob : [batch] 旧策略的 log probability。
        advantage : [batch] GAE advantage。
        value : [batch] 新 critic value。
        return_ : [batch] GAE return。
        entropy : [batch] 策略熵。
        kl_label : [batch] Phase I code_label（可选，-1 表示无 label）。
        is_labeled : [batch] bool，是否有 label。
        dead_code_mask : [K] bool，dead code mask。
        action : [batch] 选择的 action。
        logits : [batch, K] 新策略的 logits（用于 KL demo loss）。

        Returns
        -------
        PPOLossOutput : 各项 loss 分解。
        """
        # 1. Policy loss (clipped surrogate)
        ratio = torch.exp(log_prob - old_log_prob)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantage
        policy_loss = -torch.min(surr1, surr2).mean()

        # 2. Value loss
        value_loss = F.mse_loss(value, return_)

        # 3. Entropy bonus
        entropy_loss = -entropy.mean()

        # 4. KL demo loss (cross-entropy with Phase I code_label)
        kl_demo_loss = torch.tensor(0.0, device=log_prob.device)
        if (
            self.kl_demo_coef > 0
            and kl_label is not None
            and is_labeled is not None
            and logits is not None
        ):
            labeled_mask = is_labeled.bool().clone()
            # 排除 dead code 指向的 label: 若 kl_label 指向 masked code，该样本 KL 置零
            if dead_code_mask is not None:
                for i in range(labeled_mask.shape[0]):
                    if labeled_mask[i]:
                        label_val = kl_label[i].item()
                        if 0 <= label_val < dead_code_mask.shape[0] and dead_code_mask[label_val]:
                            labeled_mask[i] = False

            if labeled_mask.any():
                labeled_logits = logits[labeled_mask]
                labeled_targets = kl_label[labeled_mask]
                kl_demo_loss = F.cross_entropy(labeled_logits, labeled_targets)

        # 5. Approx KL
        with torch.no_grad():
            log_ratio = log_prob - old_log_prob
            approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()

        # 6. Clip fraction
        with torch.no_grad():
            clip_fraction = (
                (torch.abs(ratio - 1.0) > self.clip_ratio).float().mean().item()
            )

        # Total loss
        total = (
            policy_loss
            + self.value_coef * value_loss
            + self.entropy_coef * entropy_loss
            + self.kl_demo_coef * kl_demo_loss
        )

        return PPOLossOutput(
            total=total,
            policy_loss=policy_loss,
            value_loss=value_loss,
            entropy_loss=entropy_loss,
            kl_demo_loss=kl_demo_loss,
            approx_kl=approx_kl,
            clip_fraction=clip_fraction,
        )
