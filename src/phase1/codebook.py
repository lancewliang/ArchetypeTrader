"""VQ Codebook — 向量量化码本模块

# Phase I, Section 4.1: 可学习码本
# ε = {e_0, ..., e_{K-1}}, K=10, 维度 16
# 量化: k = argmin_j ||z_e - e_j||², z_q = e_k
"""

from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
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

    def compute_distances(self, z_e: Tensor) -> Tensor:
        """计算每个连续嵌入到全部 codebook 向量的平方距离。

        新增该方法是为了复用同一套距离计算逻辑，便于：
        1. quantize() 内部最近邻查找；
        2. warm-start / dead-code reinit 后的调试验证；
        3. 外部验证脚本手动核对 argmin 最近邻。

        Args:
            z_e: 编码器输出的连续嵌入 (batch, code_dim)

        Returns:
            距离矩阵 (batch, num_codes)
        """
        codebook = self.embeddings.weight
        return (
            torch.sum(z_e ** 2, dim=1, keepdim=True)
            - 2 * z_e @ codebook.t()
            + torch.sum(codebook ** 2, dim=1, keepdim=False)
        )

    @torch.no_grad()
    def initialize_from_samples(self, latent_samples: Tensor, seed: int = 42) -> Dict[str, int]:
        """使用一批 latent 样本对 codebook 做 warm-start 初始化。

        这是 debug branch 的稳定化初始化，不改变训练目标函数，仅改变初始码本值。
        为了让不同 code 尽量分散，这里采用简单的 farthest-point 选择：
        - 先随机选一个样本作为起点；
        - 之后每次选择“到当前已选集合最远”的样本。

        Args:
            latent_samples: 编码器输出样本，shape (N, code_dim)
            seed: 随机种子

        Returns:
            初始化摘要信息。
        """
        samples = latent_samples.detach().reshape(-1, self.code_dim).cpu()
        num_samples = int(samples.shape[0])
        if num_samples <= 0:
            raise ValueError("latent_samples 为空，无法执行 codebook warm-start 初始化")

        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

        selected_indices = []
        first_idx = int(torch.randint(low=0, high=num_samples, size=(1,), generator=generator).item())
        selected_indices.append(first_idx)
        min_dist = torch.sum((samples - samples[first_idx]) ** 2, dim=1)

        while len(selected_indices) < min(self.num_codes, num_samples):
            next_idx = int(torch.argmax(min_dist).item())
            selected_indices.append(next_idx)
            next_dist = torch.sum((samples - samples[next_idx]) ** 2, dim=1)
            min_dist = torch.minimum(min_dist, next_dist)

        # 当样本数少于 code 数时，循环补齐，保证 shape 匹配。
        while len(selected_indices) < self.num_codes:
            refill_idx = selected_indices[len(selected_indices) % max(len(selected_indices), 1)]
            selected_indices.append(refill_idx)

        selected = samples[selected_indices[: self.num_codes]].to(
            device=self.embeddings.weight.device,
            dtype=self.embeddings.weight.dtype,
        )
        self.embeddings.weight.copy_(selected)
        return {
            "num_source_vectors": num_samples,
            "num_initialized_codes": self.num_codes,
        }

    @torch.no_grad()
    def reinitialize_codes_from_samples(
        self,
        code_indices: Sequence[int],
        latent_samples: Tensor,
        seed: int = 42,
    ) -> Dict[str, int]:
        """用当前 epoch 收集到的 latent 样本重置长期未使用的 code。

        该方法用于 debug branch 中的 dead-code reinit。它不改变训练目标，
        仅在发现 code 长期无样本分配时，重新为对应 embedding 赋一个更贴近数据分布的值。

        Args:
            code_indices: 需要重置的 code 索引列表。
            latent_samples: 用于采样的 latent 样本，shape (N, code_dim)
            seed: 随机种子。

        Returns:
            重置摘要信息。
        """
        indices = [int(idx) for idx in code_indices]
        if not indices:
            return {"num_reinitialized_codes": 0, "num_source_vectors": 0}

        samples = latent_samples.detach().reshape(-1, self.code_dim).cpu()
        num_samples = int(samples.shape[0])
        if num_samples <= 0:
            raise ValueError("latent_samples 为空，无法执行 dead-code reinit")

        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        perm = torch.randperm(num_samples, generator=generator)
        chosen = samples[perm[: len(indices)]].to(
            device=self.embeddings.weight.device,
            dtype=self.embeddings.weight.dtype,
        )

        if chosen.shape[0] < len(indices):
            repeat = chosen[torch.arange(len(indices)) % max(chosen.shape[0], 1)]
            chosen = repeat

        for row, code_idx in enumerate(indices):
            self.embeddings.weight[code_idx].copy_(chosen[row])

        return {
            "num_reinitialized_codes": len(indices),
            "num_source_vectors": num_samples,
        }

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
        distances = self.compute_distances(z_e)

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
