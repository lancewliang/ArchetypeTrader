"""VQ Codebook — 向量量化码本模块

# Phase I, Section 4.1: 可学习码本
# ε = {e_0, ..., e_{K-1}}, K=10, 维度 16
# 量化: k = argmin_j ||z_e - e_j||², z_q = e_k
#
# 本版本新增:
# 1. code usage 统计与 perplexity 计算，用于定位 code collapse。
# 2. differentiable usage balance regularization，用于缓解单一 code 吞掉全部样本。
# 3. dead-code reinitialization，用于把长期未使用的 code 重新拉回训练轨道。
#
# 本次修正新增:
# 1. code spread regularization：直接把“不同 archetype 要分开”写入损失。
# 2. reinitialize_codes 保留，但会与训练脚本中的“underused code”检测配套使用。
#
# 论文关联:
# - Section 4.1: 离散 archetype codebook
# - 这里的新增逻辑不改变原 VQ 量化定义，只是增强训练稳定性。
"""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class VQCodebook(nn.Module):
    """向量量化码本，维护 K 个可学习原型向量。

    # 论文 Section 4.1: 可学习码本
    # ε = {e_0, ..., e_{K-1}}, K=10, 维度 16
    # 量化: k = argmin_j ||z_e - e_j||², z_q = e_k

    Args:
        num_codes: 原型数量 K (默认 10)
        code_dim: 码本向量维度 (默认 16)
    """

    def __init__(self, num_codes: int = 10, code_dim: int = 16) -> None:
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.embeddings = nn.Embedding(num_codes, code_dim)

        # 新增: 显式初始化，减小训练早期所有 code 过近导致的塌缩风险。
        nn.init.uniform_(self.embeddings.weight, a=-1.0 / code_dim, b=1.0 / code_dim)

    def _pairwise_distances(self, z_e: Tensor) -> Tensor:
        """计算 z_e 到全部 code 的平方欧氏距离。"""
        codebook = self.embeddings.weight
        distances = (
            torch.sum(z_e ** 2, dim=1, keepdim=True)
            - 2 * z_e @ codebook.t()
            + torch.sum(codebook ** 2, dim=1, keepdim=False)
        )
        return distances

    def quantize(self, z_e: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """向量量化：最近邻查找 + straight-through estimator。

        # Phase I, Step 3: 向量量化最近邻查找
        # 1. 计算距离 ||z_e - e_j||² for all j
        # 2. 找最近邻 k = argmin_j distances
        # 3. z_q = embeddings[k]
        # 4. straight-through: z_q_st = z_e + (z_q - z_e).detach()
        # 5. commitment_loss = ||z_e.detach() - z_q||²

        Args:
            z_e: 编码器输出的连续嵌入 (batch, code_dim)

        Returns:
            z_q_st: 量化后的嵌入，带 straight-through 梯度 (batch, code_dim)
            indices: 选中的码本索引 (batch,)
            commitment_loss: 码本承诺损失标量
        """
        # Step 1: 计算 ||z_e - e_j||² = ||z_e||² - 2 * z_e · e_j + ||e_j||²
        distances = self._pairwise_distances(z_e)

        # Step 2: 最近邻查找
        indices = torch.argmin(distances, dim=1)  # (batch,)

        # Step 3: 取出量化向量
        z_q = self.embeddings(indices)  # (batch, code_dim)

        # Step 4: Straight-through estimator
        # 前向传播使用 z_q，反向传播梯度流过 z_e
        z_q_st = z_e + (z_q - z_e).detach()

        # Step 5: Commitment loss = ||z_e.detach() - z_q||²
        # [NOTE: 完整 VQ 损失还包含 β₀ × ||z_e - sg[z_q]||²，在训练循环中计算]
        commitment_loss = torch.mean((z_e.detach() - z_q) ** 2)

        return z_q_st, indices, commitment_loss

    # ------------------------------------------------------------------
    # 训练稳定性辅助函数
    # ------------------------------------------------------------------

    def usage_histogram(self, indices: Tensor) -> Tensor:
        """统计当前 batch 的 code 使用频次。"""
        return torch.bincount(indices.detach(), minlength=self.num_codes).to(torch.float32)

    def perplexity(self, histogram: Tensor) -> Tensor:
        """基于 code 使用频率计算 perplexity。

        perplexity 越接近 1，越说明所有样本都挤在极少数 code 上。
        """
        histogram = histogram.to(torch.float32)
        total = torch.clamp(histogram.sum(), min=1.0)
        probs = histogram / total
        probs = probs[probs > 0]
        if probs.numel() == 0:
            return torch.tensor(0.0, device=histogram.device)
        entropy = -(probs * torch.log(probs)).sum()
        return torch.exp(entropy)

    def soft_assign(self, z_e: Tensor, temperature: float = 1.0) -> Tuple[Tensor, Tensor]:
        """对每个 z_e 计算 differentiable soft assignment。

        设计意图:
        - 上一版的 balance loss 只依赖 hard indices，几乎不向 encoder / codebook 回传有用梯度。
        - 这里改成基于 softmax(-distance / temperature) 的软分配，
          从而真正对“批量平均 code 使用分布”施加可微约束。
        """
        if temperature <= 0:
            raise ValueError(f"temperature 必须 > 0，收到 {temperature}")
        distances = self._pairwise_distances(z_e)
        probs = F.softmax(-distances / temperature, dim=-1)
        return probs, distances

    def balance_loss(self, z_e: Tensor, temperature: float = 0.5) -> Tuple[Tensor, Tensor]:
        """鼓励 batch 内 code 使用更均衡，缓解 code collapse。

        返回:
            balance_loss: KL(mean_probs || uniform)
            mean_probs: 当前 batch 的平均软分配分布
        """
        probs, _ = self.soft_assign(z_e, temperature=temperature)
        mean_probs = probs.mean(dim=0)
        uniform = torch.full_like(mean_probs, fill_value=1.0 / self.num_codes)
        balance_loss = torch.sum(mean_probs * (torch.log(mean_probs + 1e-8) - torch.log(uniform + 1e-8)))
        return balance_loss, mean_probs

    def entropy_regularization(self, z_e: Tensor, temperature: float = 0.5) -> Tensor:
        """鼓励 batch 级 code 使用熵更高。

        与 balance_loss 目标一致，但以“负熵”的形式提供一个更直观的监控量。
        训练时通常只需用 balance_loss；这里额外暴露接口，便于后续实验。
        """
        probs, _ = self.soft_assign(z_e, temperature=temperature)
        mean_probs = probs.mean(dim=0)
        entropy = -(mean_probs * torch.log(mean_probs + 1e-8)).sum()
        return -entropy

    def spread_loss(self) -> Tensor:
        """鼓励不同 code 在表示空间中分开，缓解“多个 code 挤成一团”。

        这里对归一化后的 code 向量做 pairwise cosine similarity 惩罚：
        - 对角线天然为 1，不计入损失
        - 非对角项越接近 0，说明不同 archetype 越独立
        """
        normalized = F.normalize(self.embeddings.weight, dim=-1)
        similarity = normalized @ normalized.t()
        eye = torch.eye(self.num_codes, device=similarity.device, dtype=torch.bool)
        off_diag = similarity.masked_select(~eye)
        if off_diag.numel() == 0:
            return torch.tensor(0.0, device=similarity.device)
        return torch.mean(off_diag ** 2)

    @torch.no_grad()
    def reinitialize_codes(
        self,
        dead_code_ids: Iterable[int],
        candidate_vectors: Tensor,
        noise_std: float = 1e-3,
    ) -> None:
        """使用当前 epoch 的 encoder 输出重置长期死亡的 code。

        设计意图:
        - 如果某些 code 连续多个 epoch 完全没被选中，说明它们已经掉出训练轨道。
        - 直接用当前 z_e 分布中的样本向量重置，比随机初始化更容易恢复使用。
        """
        if candidate_vectors.numel() == 0:
            return

        if candidate_vectors.ndim != 2 or candidate_vectors.shape[1] != self.code_dim:
            raise ValueError(
                f"candidate_vectors 形状应为 (N, {self.code_dim})，收到 {tuple(candidate_vectors.shape)}"
            )

        device = self.embeddings.weight.device
        candidate_vectors = candidate_vectors.to(device)

        for code_id in dead_code_ids:
            sample_idx = torch.randint(
                low=0,
                high=candidate_vectors.shape[0],
                size=(1,),
                device=device,
            ).item()
            new_value = candidate_vectors[sample_idx]
            if noise_std > 0:
                new_value = new_value + noise_std * torch.randn_like(new_value)
            self.embeddings.weight.data[int(code_id)].copy_(new_value)
