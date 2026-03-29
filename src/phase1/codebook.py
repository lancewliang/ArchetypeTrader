"""VQ Codebook — 向量量化码本模块

# Phase I, Section 4.1: 可学习码本
# ε = {e_0, ..., e_{K-1}}, K=10, 维度 16
# 量化: k = argmin_j ||z_e - e_j||², z_q = e_k
"""

from typing import Tuple

import math

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
        self.reset_embedding_parameters()

    def reset_embedding_parameters(self) -> None:
        """重置码本参数。

        该方法只影响实现层初始化策略，不改变论文中的量化公式
        `k = argmin_j ||z_e - e_j||^2, z_q = e_k`。
        默认采用与 embedding 维度相关的均匀分布，避免初始向量尺度过大。
        """
        bound = 1.0 / math.sqrt(float(max(self.code_dim, 1)))
        nn.init.uniform_(self.embeddings.weight, -bound, bound)

    def set_codebook_vectors(self, vectors: Tensor) -> None:
        """用外部给定向量覆盖码本初值。

        该接口用于 data-driven codebook initialization。它只改变码本的
        初始位置，不改变 Phase I 的主流程、最近邻量化规则或损失函数。

        Args:
            vectors: shape = (num_codes, code_dim) 的初始化向量矩阵。

        Raises:
            ValueError: 当 shape 与当前码本不一致时抛出。
        """
        if vectors.ndim != 2:
            raise ValueError(f"vectors 应为 2D 张量，实际 ndim={vectors.ndim}")
        expected_shape = (self.num_codes, self.code_dim)
        if tuple(vectors.shape) != expected_shape:
            raise ValueError(
                f"vectors shape 不匹配: actual={tuple(vectors.shape)}, expected={expected_shape}"
            )

        with torch.no_grad():
            prepared = vectors.to(
                device=self.embeddings.weight.device,
                dtype=self.embeddings.weight.dtype,
            )
            self.embeddings.weight.copy_(prepared)

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
        # 码本权重 (num_codes, code_dim)
        codebook = self.embeddings.weight

        # Step 1: 计算 ||z_e - e_j||² = ||z_e||² - 2 * z_e · e_j + ||e_j||²
        # 展开平方距离以避免显式广播
        distances = (
            torch.sum(z_e ** 2, dim=1, keepdim=True)
            - 2 * z_e @ codebook.t()
            + torch.sum(codebook ** 2, dim=1, keepdim=False)
        )  # (batch, num_codes)

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
