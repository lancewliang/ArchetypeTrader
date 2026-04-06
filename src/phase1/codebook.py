"""VQ Codebook — 向量量化码本模块 (EMA + SGD 双模式)

# Phase I, Section 4.1: 可学习码本
# ε = {e_0, ..., e_{K-1}}, K=10, 维度 16
# 量化: k = argmin_j ||z_e - e_j||², z_q = e_k
#
# 增强: 支持 EMA (Exponential Moving Average) 更新码本向量 (VQ-VAE-2 风格)。
# EMA 模式下码本不通过梯度更新，而是用指数滑动平均跟踪分配到每个 code 的
# z_e 均值，收敛更稳定、codebook 利用率更高。
"""

import logging
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class VQCodebook(nn.Module):
    """向量量化码本，维护 K 个可学习原型向量。

    支持两种更新模式:
    - SGD 模式 (use_ema=False): 原始 VQ-VAE，码本通过 commitment loss 梯度更新
    - EMA 模式 (use_ema=True): VQ-VAE-2 风格，码本通过指数滑动平均更新，
      收敛更稳定，codebook 利用率更高

    Args:
        num_codes: 原型数量 K (默认 10)
        code_dim: 码本向量维度 (默认 16)
        use_ema: 是否使用 EMA 更新码本 (默认 True)
        ema_decay: EMA 衰减率 γ (默认 0.99)
        ema_epsilon: Laplace 平滑防止除零 (默认 1e-5)
    """

    def __init__(
        self,
        num_codes: int = 10,
        code_dim: int = 16,
        use_ema: bool = True,
        ema_decay: float = 0.99,
        ema_epsilon: float = 1e-5,
    ) -> None:
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema_epsilon = ema_epsilon

        self.embeddings = nn.Embedding(num_codes, code_dim)

        if use_ema:
            # EMA 模式: 码本权重不参与梯度更新
            self.embeddings.weight.requires_grad = False
            # 注册 EMA 缓冲区 (不参与 state_dict 的梯度，但会被保存/加载)
            self.register_buffer("_ema_cluster_size", torch.zeros(num_codes))
            self.register_buffer("_ema_embed_sum", self.embeddings.weight.clone())

    def quantize(self, z_e: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """向量量化：最近邻查找 + straight-through estimator。

        EMA 模式下，在训练时自动更新码本向量（无需梯度）。
        SGD 模式下行为与原始实现完全一致。

        Args:
            z_e: 编码器输出的连续嵌入 (batch, code_dim)

        Returns:
            z_q_st: 量化后的嵌入，带 straight-through 梯度 (batch, code_dim)
            indices: 选中的码本索引 (batch,)
            commitment_loss: 码本承诺损失标量
                EMA 模式: 仅 encoder commitment (β₀ × ||z_e - sg[z_q]||²)
                SGD 模式: codebook commitment (||sg[z_e] - z_q||²)
        """
        codebook = self.embeddings.weight

        # 计算距离 ||z_e - e_j||²
        distances = (
            torch.sum(z_e ** 2, dim=1, keepdim=True)
            - 2 * z_e @ codebook.t()
            + torch.sum(codebook ** 2, dim=1, keepdim=False)
        )

        # 最近邻查找
        indices = torch.argmin(distances, dim=1)
        z_q = self.embeddings(indices)

        # EMA 更新码本 (仅训练时)
        if self.use_ema and self.training:
            self._ema_update(z_e, indices)

        # Straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()

        # Commitment loss
        if self.use_ema:
            # EMA 模式: 只需 encoder commitment，码本自己通过 EMA 更新
            # 返回 0 作为 codebook loss，encoder commitment 在训练循环中计算
            commitment_loss = torch.tensor(0.0, device=z_e.device)
        else:
            # SGD 模式: ||sg[z_e] - z_q||²
            commitment_loss = torch.mean((z_e.detach() - z_q) ** 2)

        return z_q_st, indices, commitment_loss

    @torch.no_grad()
    def _ema_update(self, z_e: Tensor, indices: Tensor) -> None:
        """EMA 更新码本向量。

        对每个 code k:
          N_k ← γ × N_k + (1-γ) × n_k        (n_k = 分配到 k 的样本数)
          m_k ← γ × m_k + (1-γ) × Σ z_e[i]   (对所有分配到 k 的 z_e 求和)
          e_k ← m_k / (N_k + ε)               (Laplace 平滑)
        """
        # one-hot 编码: (batch, num_codes)
        onehot = torch.zeros(
            indices.shape[0], self.num_codes,
            device=z_e.device, dtype=z_e.dtype,
        )
        onehot.scatter_(1, indices.unsqueeze(1), 1.0)

        # 当前 batch 的统计量
        batch_cluster_size = onehot.sum(dim=0)          # (num_codes,)
        batch_embed_sum = onehot.t() @ z_e              # (num_codes, code_dim)

        # EMA 更新
        self._ema_cluster_size.mul_(self.ema_decay).add_(
            batch_cluster_size, alpha=1.0 - self.ema_decay,
        )
        self._ema_embed_sum.mul_(self.ema_decay).add_(
            batch_embed_sum, alpha=1.0 - self.ema_decay,
        )

        # Laplace 平滑后更新码本
        n = self._ema_cluster_size.sum()
        cluster_size_smoothed = (
            (self._ema_cluster_size + self.ema_epsilon)
            / (n + self.num_codes * self.ema_epsilon)
            * n
        )
        self.embeddings.weight.data.copy_(
            self._ema_embed_sum / cluster_size_smoothed.unsqueeze(1)
        )

    # ------------------------------------------------------------------
    # Codebook collapse 对策
    # ------------------------------------------------------------------

    @torch.no_grad()
    def init_from_data(self, z_e_samples: Tensor, n_iter: int = 10) -> None:
        """用 k-means 从 z_e 样本初始化码本向量。

        在 Phase A 结束后、Phase B 开始前调用，使码本向量位于 z_e 分布的
        实际区域内，避免随机初始化导致的死码问题。

        Args:
            z_e_samples: 从 Phase A encoder 收集的 z_e 样本 (N, code_dim)
            n_iter: k-means 迭代次数 (默认 10)
        """
        N = z_e_samples.shape[0]
        K = self.num_codes
        device = self.embeddings.weight.device

        if N < K:
            logger.warning(
                "z_e 样本数 (%d) 少于码本大小 (%d)，跳过 k-means 初始化", N, K,
            )
            return

        z = z_e_samples.to(device).float()
        centroids = self._kmeans(z, K, n_iter)
        self.embeddings.weight.copy_(centroids)
        logger.info("码本已通过 k-means 从 %d 个 z_e 样本初始化 (%d 次迭代)", N, n_iter)

    @torch.no_grad()
    def init_from_data_direction_aware(
        self,
        z_e_samples: Tensor,
        actions: Tensor,
        n_iter: int = 10,
    ) -> None:
        """方向感知的码本初始化：先按交易方向分组，再在组内 k-means。

        DP 轨迹的 single-trade 结构意味着每条轨迹有一个主要方向：
        - long (action 2 占主导)
        - short (action 0 占主导)
        - flat (action 1 占主导，或无变化)

        将 K 个码本条目按方向分配（如 4 long + 4 short + 2 flat），
        在每个方向子集内做 k-means，确保码本覆盖所有方向。
        这直接解决 LSTM decoder 的方向性坍缩问题。

        Args:
            z_e_samples: encoder 输出的 z_e 样本 (N, code_dim)
            actions: 对应的 DP 动作序列 (N, h)，值域 {0, 1, 2}
            n_iter: k-means 迭代次数
        """
        N = z_e_samples.shape[0]
        K = self.num_codes
        device = self.embeddings.weight.device

        if N < K:
            logger.warning(
                "z_e 样本数 (%d) 少于码本大小 (%d)，退化为普通 k-means", N, K,
            )
            self.init_from_data(z_e_samples, n_iter)
            return

        z = z_e_samples.to(device).float()
        a = actions.to(device).long()  # (N, h)

        # 判断每条轨迹的主要方向
        # 统计每条轨迹中 short(0) / flat(1) / long(2) 的占比
        action_counts = torch.zeros(N, 3, device=device)
        for c in range(3):
            action_counts[:, c] = (a == c).float().sum(dim=1)

        dominant_action = action_counts.argmax(dim=1)  # (N,) 0=short, 1=flat, 2=long

        long_mask = dominant_action == 2
        short_mask = dominant_action == 0
        flat_mask = dominant_action == 1

        long_count = long_mask.sum().item()
        short_count = short_mask.sum().item()
        flat_count = flat_mask.sum().item()

        logger.info(
            "方向分布: long=%d, short=%d, flat=%d (total=%d)",
            long_count, short_count, flat_count, N,
        )

        # 按方向分配码本条目: 4 long + 4 short + 2 flat
        # 如果某个方向样本不足，将配额转给其他方向
        k_long = 4
        k_short = 4
        k_flat = K - k_long - k_short  # 2

        # 调整：如果某方向样本为 0，将配额转给最大方向
        if long_count == 0:
            if short_count >= flat_count:
                k_short += k_long
            else:
                k_flat += k_long
            k_long = 0
        if short_count == 0:
            if long_count >= flat_count:
                k_long += k_short
            else:
                k_flat += k_short
            k_short = 0
        if flat_count == 0:
            if long_count >= short_count:
                k_long += k_flat
            else:
                k_short += k_flat
            k_flat = 0

        centroids = torch.zeros(K, self.code_dim, device=device)
        idx = 0

        for mask, k_alloc, label in [
            (long_mask, k_long, "long"),
            (short_mask, k_short, "short"),
            (flat_mask, k_flat, "flat"),
        ]:
            if k_alloc == 0:
                continue
            z_sub = z[mask]
            if z_sub.shape[0] == 0:
                continue
            # 如果子集样本少于分配数，用全部样本 + 噪声填充
            if z_sub.shape[0] < k_alloc:
                for j in range(k_alloc):
                    src = z_sub[j % z_sub.shape[0]]
                    centroids[idx] = src + torch.randn_like(src) * 1e-3
                    idx += 1
                continue
            # 组内 k-means
            sub_centroids = self._kmeans(z_sub, k_alloc, n_iter)
            centroids[idx:idx + k_alloc] = sub_centroids
            idx += k_alloc

        self.embeddings.weight.copy_(centroids)
        logger.info(
            "码本已通过方向感知 k-means 初始化: %d long + %d short + %d flat = %d codes",
            k_long, k_short, k_flat, K,
        )

    @staticmethod
    def _kmeans(z: Tensor, k: int, n_iter: int) -> Tensor:
        """在给定样本上执行 k-means，返回 k 个中心。"""
        N = z.shape[0]
        device = z.device

        # k-means++ 初始化
        indices = torch.zeros(k, dtype=torch.long, device=device)
        indices[0] = torch.randint(N, (1,), device=device)
        for i in range(1, k):
            dists = torch.cdist(z, z[indices[:i]]).min(dim=1).values
            probs = dists / dists.sum()
            indices[i] = torch.multinomial(probs, 1)

        centroids = z[indices].clone()

        for _ in range(n_iter):
            dists = torch.cdist(z, centroids)
            assignments = dists.argmin(dim=1)
            new_centroids = torch.zeros_like(centroids)
            for c in range(k):
                mask = assignments == c
                if mask.any():
                    new_centroids[c] = z[mask].mean(dim=0)
                else:
                    counts = torch.bincount(assignments, minlength=k)
                    largest = counts.argmax()
                    pool = z[assignments == largest]
                    new_centroids[c] = pool[torch.randint(len(pool), (1,))]
            centroids = new_centroids

        return centroids

    @torch.no_grad()
    def reset_dead_codes(self, z_e_batch: Tensor, code_counts: np.ndarray) -> int:
        """将死码重置为当前 z_e 分布中的采样点（加少量噪声）。

        在每个 Phase B epoch 结束后调用。死码定义为 code_counts == 0 的条目。

        Args:
            z_e_batch: 当前 epoch 最后一个 batch 的 z_e (batch, code_dim)
            code_counts: 当前 epoch 各码本条目的使用计数 (num_codes,)

        Returns:
            重置的死码数量
        """
        dead_mask = code_counts == 0
        dead_indices = np.where(dead_mask)[0]
        if len(dead_indices) == 0:
            return 0

        device = self.embeddings.weight.device
        z = z_e_batch.to(device).float()
        N = z.shape[0]

        for idx in dead_indices:
            # 从 z_e 中随机采样一个点，加少量噪声
            sampled = z[torch.randint(N, (1,), device=device)].squeeze(0)
            noise = torch.randn_like(sampled) * 1e-3
            self.embeddings.weight.data[idx] = sampled + noise

        logger.info("重置 %d 个死码 (indices=%s)", len(dead_indices), dead_indices.tolist())
        return len(dead_indices)
