"""VQ encoder-decoder loss.

设计文档锚点: §6.4 与 §6.5。

总损失:
``L = L_rec + L_codebook + β0 * L_commit + λ_usage * L_usage + λ_tc * L_tc + λ_align * L_align``
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


@dataclass
class LossOutputs:
    total: "torch.Tensor"
    reconstruction: "torch.Tensor"
    codebook: "torch.Tensor"
    commitment: "torch.Tensor"
    usage: Optional["torch.Tensor"]
    contrastive: Optional["torch.Tensor"]
    alignment: Optional["torch.Tensor"] = None


def compute_soft_code_assignments(z_e, codebook, temperature: float):
    """基于 encoder latent 与 codebook 距离计算 soft assignment。"""
    temp = max(float(temperature), 1.0e-6)
    distances = (
        (z_e ** 2).sum(dim=1, keepdim=True)
        - 2 * z_e @ codebook.t()
        + (codebook ** 2).sum(dim=1)
    )
    return torch.softmax(-distances / temp, dim=1)


class Phase1Loss(nn.Module if nn is not None else object):  # type: ignore[misc]
    """Phase I 总损失。

    Parameters
    ----------
    beta0 : commitment 权重，论文 = 0.25。
    usage_weight : ``L_usage`` 权重；``paper_strict_reproduction=True`` 时设 0
                   以严格对齐论文公式 (4)。
    contrastive_weight : ``L_tc`` 权重；默认 0.05。仅在 batch 含 ``contrastive_pair_id``
                         时计算。
    contrastive_temperature : 0.1（InfoNCE 模式生效；当前实现仅支持 cosine）。
    use_infonce : 预留；当前实现走 cosine 距离，避免负样本不足时 InfoNCE 不稳。

    边界
    ----
    - contrastive 项只约束 encoder ``z_e``，不约束 ``z_q`` 或 decoder logits；
      这点由 ``_contrastive_loss`` 的输入只接 ``z_e`` 保证。
    - usage 项不破坏论文公式形态: ``KL(U(K) || p_code)``，与重构 / VQ / commitment
      作为加权项叠加。
    """

    def __init__(
        self,
        beta0: float = 0.25,
        usage_weight: float = 0.01,
        contrastive_weight: float = 0.0,
        contrastive_temperature: float = 0.1,
        use_infonce: bool = False,
        num_codes: Optional[int] = None,
        usage_profit_alignment_weight: float = 0.0,
        usage_profit_alignment_target_corr: float = 0.2,
    ) -> None:
        if nn is None:  # pragma: no cover
            raise ImportError("Phase1Loss 需要 torch")
        super().__init__()
        self.beta0 = beta0
        self.usage_weight = usage_weight
        self.contrastive_weight = contrastive_weight
        self.contrastive_temperature = contrastive_temperature
        self.use_infonce = use_infonce
        self.usage_profit_alignment_weight = usage_profit_alignment_weight
        self.usage_profit_alignment_target_corr = usage_profit_alignment_target_corr
        # ``num_codes`` 必须由 trainer 显式传入（``ModelConfig.num_codes``），
        # 否则 KL(uniform || p_code) 会在 codebook collapse 时低估真实 K
        # （从 code_id 推断会漏掉未使用的 code，造成 usage_loss 偏小）。
        # 留 None 仅用于单元测试的简化 fallback 场景。
        self.num_codes = num_codes

    def forward(
        self,
        *,
        action_logits,
        target_actions,
        z_e,
        z_q_no_grad,
        code_id,
        contrastive_pair_ids: Optional[List[str]] = None,
        trajectory_returns=None,
        codebook=None,
        soft_assignment_temperature: float = 2.0,
    ) -> LossOutputs:
        """计算总损失。

        组合
        ----
        - ``reconstruction``: ``F.cross_entropy(logits.view(-1, 3), targets.view(-1))``。
          每步 weight=1（``val_weighted_reconstruction_accuracy`` 是评估侧概念，
          训练侧使用均匀 CE，避免训练分布与评估分布不一致）。
        - ``codebook``: ``((sg[z_e] - z_q)^2).mean()`` → 推动 codebook 向 ``z_e`` 移动。
          EMA 模式下 codebook 不再走梯度，该项仅作审计 / 记录之用。
        - ``commitment``: ``((z_e - sg[z_q])^2).mean()`` → 推动 ``z_e`` 向选中的 code 收敛。
        - ``usage``: 若 ``usage_weight > 0``，按 ``KL(U(K) || p_code)`` 计算。
          K 从 ``code_id`` 的最大值推断（兼容性写法；理想情况下应由 quantizer 提供）。
        - ``contrastive``: 若 ``contrastive_pair_ids`` 提供并能凑成对，计算 cosine
          距离均值；否则返回 None。

        Returns
        -------
        LossOutputs : ``total + reconstruction + codebook + commitment + usage + contrastive``。
                      其中后两个项可能为 ``None``。
        """
        # reconstruction CE: [B, h, 3] vs [B, h]
        b, h, c = action_logits.shape
        rec = F.cross_entropy(action_logits.reshape(b * h, c), target_actions.reshape(b * h))

        # codebook loss: ||sg[z_e] - z_q||^2  → 推动 codebook 向 z_e 移动
        codebook_loss = ((z_e.detach() - z_q_no_grad) ** 2).mean()
        # commitment loss: ||z_e - sg[z_q]||^2 → 推动 z_e 向选中的 code 收敛
        commitment = ((z_e - z_q_no_grad.detach()) ** 2).mean()

        usage = None
        if self.usage_weight > 0:
            if code_id is None:
                raise ValueError("usage loss requires code_id")
            # 优先使用 init 时显式传入的 num_codes；缺失时回退到从 code_id 推断
            # （仅用于单测；trainer 路径必须传入避免 collapse 低估 K）。
            if self.num_codes is not None:
                num_codes = int(self.num_codes)
            else:
                num_codes = int(code_id.max().item()) + 1
            usage = self._kl_uniform(code_id, num_codes=num_codes)

        contrastive = None
        if self.contrastive_weight > 0 and contrastive_pair_ids is not None:
            contrastive = self._contrastive_loss(z_e, contrastive_pair_ids)

        alignment = None
        if (
            self.usage_profit_alignment_weight > 0
            and trajectory_returns is not None
            and codebook is not None
        ):
            soft_assignments = compute_soft_code_assignments(
                z_e, codebook, soft_assignment_temperature
            )
            alignment = self._usage_profit_alignment(
                soft_assignments,
                trajectory_returns,
                self.usage_profit_alignment_target_corr,
            )

        total = rec + codebook_loss + self.beta0 * commitment
        if usage is not None:
            total = total + self.usage_weight * usage
        if contrastive is not None:
            total = total + self.contrastive_weight * contrastive
        if alignment is not None:
            total = total + self.usage_profit_alignment_weight * alignment
        return LossOutputs(
            total=total,
            reconstruction=rec,
            codebook=codebook_loss,
            commitment=commitment,
            usage=usage,
            contrastive=contrastive,
            alignment=alignment,
        )

    def forward_pretrain(self, *, action_logits, target_actions) -> LossOutputs:
        """Phase A 损失：只计算 action reconstruction CE。"""
        b, h, c = action_logits.shape
        rec = F.cross_entropy(action_logits.reshape(b * h, c), target_actions.reshape(b * h))
        zero = rec.new_zeros(())
        return LossOutputs(
            total=rec,
            reconstruction=rec,
            codebook=zero,
            commitment=zero,
            usage=None,
            contrastive=None,
            alignment=None,
        )

    # ---------- usage / contrastive ----------

    def _kl_uniform(self, code_id, num_codes: int):
        """KL(U(K) || p_code): 鼓励 batch 内 code 分布趋近均匀。"""
        counts = torch.bincount(code_id.flatten(), minlength=num_codes).float()
        total = counts.sum().clamp_min(1.0)
        p = (counts + 1e-6) / (total + num_codes * 1e-6)
        u = torch.full_like(p, 1.0 / num_codes)
        # KL(u || p) = Σ u (log u - log p)
        return (u * (u.log() - p.log())).sum()

    def _contrastive_loss(self, z_e, pair_ids: List[str]):
        """简单 cosine 相似 contrastive: 同 pair_id 的两个 z_e 距离越近 loss 越低。"""
        # 找出 pair_id 不为空的样本，按 pair_id 分组（每组应正好 2）。
        groups: dict[str, list[int]] = {}
        for i, pid in enumerate(pair_ids):
            if not pid:
                continue
            groups.setdefault(pid, []).append(i)
        pairs = [v for v in groups.values() if len(v) == 2]
        if not pairs:
            return torch.tensor(0.0, device=z_e.device)
        a_idx = torch.tensor([p[0] for p in pairs], device=z_e.device)
        b_idx = torch.tensor([p[1] for p in pairs], device=z_e.device)
        a = F.normalize(z_e[a_idx], dim=-1)
        b = F.normalize(z_e[b_idx], dim=-1)
        cos = (a * b).sum(dim=-1)
        return (1 - cos).mean()

    def _usage_profit_alignment(
        self,
        soft_assignments,
        trajectory_returns,
        target_corr: float,
        eps: float = 1.0e-6,
    ):
        """鼓励高收益 code 获得更高使用率，惩罚 usage-return 相关性不足。"""
        if soft_assignments.shape[0] < 2 or soft_assignments.shape[1] < 2:
            return soft_assignments.new_zeros(())

        returns = trajectory_returns.to(
            device=soft_assignments.device,
            dtype=soft_assignments.dtype,
        ).reshape(-1, 1)
        if returns.shape[0] != soft_assignments.shape[0]:
            raise ValueError(
                "trajectory_returns batch size must match soft_assignments"
            )

        code_mass = soft_assignments.sum(dim=0).clamp_min(eps)
        usage = code_mass / soft_assignments.shape[0]
        code_returns = (soft_assignments * returns).sum(dim=0) / code_mass

        usage_centered = usage - usage.mean()
        return_centered = code_returns - code_returns.mean()
        covariance = torch.mean(usage_centered * return_centered)
        denom = torch.sqrt(
            torch.mean(usage_centered ** 2) * torch.mean(return_centered ** 2) + eps
        )
        corr = covariance / denom
        target = soft_assignments.new_tensor(
            max(-1.0, min(1.0, float(target_corr)))
        )
        return torch.relu(target - corr)
