"""Codebook initialization and health helper functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .tensor_data_types import (
    ArchetypeLabelTensor,
    LatentTensor,
)


@dataclass(frozen=True)
class QuantizeOutput:
    """VectorQuantizer 的输出。"""

    quantized: LatentTensor
    code_indices: ArchetypeLabelTensor
    vq_loss: torch.Tensor
    codebook_loss: torch.Tensor
    commitment_loss: torch.Tensor
    distances: torch.Tensor
    z_q_no_grad: LatentTensor


@dataclass(frozen=True)
class CodebookInitResult:
    """Codebook 初始化结果摘要。"""

    method: str
    num_samples: int
    num_centers: int
    direction_counts: Mapping[int, int]
    direction_quotas: Mapping[int, int]


class VectorQuantizer(nn.Module):
    """VQ-VAE 风格的最近邻 codebook。"""

    def __init__(
        self,
        num_archetypes: int = 10,
        latent_dim: int = 16,
        commitment_cost: float = 0.25,
    ) -> None:
        super().__init__()
        if num_archetypes <= 0:
            raise ValueError("num_archetypes must be positive")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")

        self.num_archetypes = num_archetypes
        self.latent_dim = latent_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_archetypes, latent_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=1.0 / latent_dim)

    def forward(self, z_e: LatentTensor) -> QuantizeOutput:
        return self.quantize(z_e)

    def quantize(self, z_e: LatentTensor) -> QuantizeOutput:
        """最近邻量化，并返回 STE 后的 ``quantized``。"""

        if z_e.ndim != 2:
            raise ValueError("z_e must have shape [batch, latent_dim]")
        if z_e.shape[-1] != self.latent_dim:
            raise ValueError(
                f"z_e last dim must be {self.latent_dim}, got {z_e.shape[-1]}"
            )

        codebook = self.embedding.weight
        distances = torch.cdist(z_e, codebook, p=2).pow(2)
        code_indices = distances.argmin(dim=-1)
        z_q = self.embedding(code_indices)

        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        quantized = z_e + (z_q - z_e).detach()
        return QuantizeOutput(
            quantized=quantized,
            code_indices=code_indices,
            vq_loss=vq_loss,
            codebook_loss=codebook_loss,
            commitment_loss=commitment_loss,
            distances=distances,
            z_q_no_grad=z_q.detach(),
        )

    def embedding_from_code(self, code_id: ArchetypeLabelTensor) -> LatentTensor:
        """根据 archetype id 取出 codebook 向量。"""

        return self.embedding(code_id.long())

    @torch.no_grad()
    def initialize_from_directional_kmeans(
        self,
        latents: LatentTensor,
        directions: torch.Tensor | None = None,
        *,
        random_state: int = 0,
        n_init: int = 10,
        max_iter: int = 100,
    ) -> CodebookInitResult:
        """使用方向感知 k-means 初始化 codebook。"""

        return initialize_codebook_from_directional_kmeans(
            self.embedding,
            latents,
            directions,
            random_state=random_state,
            n_init=n_init,
            max_iter=max_iter,
        )


@torch.no_grad()
def initialize_codebook_from_directional_kmeans(
    embedding: nn.Embedding,
    latents: torch.Tensor,
    directions: torch.Tensor | None = None,
    *,
    random_state: int = 0,
    n_init: int = 10,
    max_iter: int = 100,
) -> CodebookInitResult:
    """使用方向感知 k-means 初始化 codebook embedding。

    初始化逻辑:
        1. 先按 trajectory 主方向把 encoder latent 分成 short/flat/long/mixed。
        2. 对出现过的方向至少分配一个 code，再按样本数比例分配剩余 code。
        3. 每个方向内部独立跑 k-means，得到方向内的 archetype 初始中心。
        4. 若某方向样本太少导致中心不足，用全局 k-means 中最远的中心补齐。

    这样做的目的不是固定 code 的语义，而是让正式 VQ 训练开始时 codebook
    已覆盖主要交易方向，减少随机初始化造成的早期 label collapse。
    """

    num_archetypes = int(embedding.num_embeddings)
    latent_dim = int(embedding.embedding_dim)
    latents_np = _latents_to_numpy(latents, latent_dim=latent_dim)
    num_samples = int(latents_np.shape[0])
    if num_samples == 0:
        raise ValueError("latents must contain at least one sample")
    if n_init < 1:
        raise ValueError("n_init must be >= 1")
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")

    if directions is None:
        direction_np = np.zeros(num_samples, dtype=np.int64)
    else:
        direction_np = _directions_to_numpy(directions, num_samples=num_samples)

    direction_counts = {
        int(label): int(np.sum(direction_np == label))
        for label in np.unique(direction_np)
    }
    direction_quotas = _allocate_direction_quotas(
        direction_counts=direction_counts,
        num_centers=num_archetypes,
    )

    centers: list[np.ndarray] = []
    for offset, (direction_label, quota) in enumerate(direction_quotas.items()):
        group_latents = latents_np[direction_np == direction_label]
        group_centers = _kmeans_centers(
            group_latents,
            num_centers=min(quota, group_latents.shape[0]),
            random_state=random_state + offset + 1,
            n_init=n_init,
            max_iter=max_iter,
        )
        centers.extend(group_centers)

    global_centers = _kmeans_centers(
        latents_np,
        num_centers=min(num_archetypes, num_samples),
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
    )
    centers = _fill_missing_centers(
        centers=centers,
        fallback_centers=global_centers,
        latents=latents_np,
        num_centers=num_archetypes,
        random_state=random_state,
    )

    center_tensor = torch.as_tensor(
        np.stack(centers, axis=0),
        dtype=embedding.weight.dtype,
        device=embedding.weight.device,
    )
    embedding.weight.copy_(center_tensor)
    return CodebookInitResult(
        method="directional_kmeans",
        num_samples=num_samples,
        num_centers=num_archetypes,
        direction_counts=direction_counts,
        direction_quotas=direction_quotas,
    )


def classify_trajectory_directions(actions: torch.Tensor) -> torch.Tensor:
    """把 action sequence 映射为主方向标签。

    标签约定:
        ``0`` = short，``1`` = flat，``2`` = long，``3`` = mixed。

    输入:
        ``actions``: ``[batch, horizon]``，动作编码遵循项目约定
        ``0=short, 1=flat, 2=long``。
    """

    if actions.ndim != 2:
        raise ValueError("actions must have shape [batch, horizon]")
    positions = actions.long() - 1
    long_count = torch.sum(positions > 0, dim=1)
    short_count = torch.sum(positions < 0, dim=1)

    labels = torch.full(
        (actions.shape[0],),
        1,
        dtype=torch.long,
        device=actions.device,
    )
    labels = torch.where(long_count > short_count, torch.full_like(labels, 2), labels)
    labels = torch.where(short_count > long_count, torch.full_like(labels, 0), labels)
    mixed = (long_count == short_count) & (long_count > 0)
    labels = torch.where(mixed, torch.full_like(labels, 3), labels)
    return labels


def _latents_to_numpy(latents: torch.Tensor, *, latent_dim: int) -> np.ndarray:
    if latents.ndim != 2:
        raise ValueError("latents must have shape [num_samples, latent_dim]")
    if latents.shape[-1] != latent_dim:
        raise ValueError(
            f"latents last dim must be {latent_dim}, got {latents.shape[-1]}"
        )
    return latents.detach().float().cpu().numpy()


def _directions_to_numpy(
    directions: torch.Tensor,
    *,
    num_samples: int,
) -> np.ndarray:
    if directions.ndim != 1:
        raise ValueError("directions must have shape [num_samples]")
    if directions.shape[0] != num_samples:
        raise ValueError(
            f"directions length must be {num_samples}, got {directions.shape[0]}"
        )
    return directions.detach().long().cpu().numpy()


def _allocate_direction_quotas(
    *,
    direction_counts: Mapping[int, int],
    num_centers: int,
) -> dict[int, int]:
    if num_centers <= 0:
        raise ValueError("num_centers must be positive")

    positive_counts = {
        int(label): int(count)
        for label, count in direction_counts.items()
        if int(count) > 0
    }
    if not positive_counts:
        return {}

    selected_labels = sorted(
        positive_counts,
        key=lambda label: (-positive_counts[label], label),
    )[:num_centers]
    quotas = {label: 1 for label in selected_labels}
    remaining = num_centers - len(quotas)
    if remaining <= 0:
        return dict(sorted(quotas.items()))

    selected_total = sum(positive_counts[label] for label in selected_labels)
    fractional: list[tuple[int, float]] = []
    for label in selected_labels:
        raw_extra = remaining * positive_counts[label] / selected_total
        extra = int(np.floor(raw_extra))
        quotas[label] += extra
        fractional.append((label, raw_extra - extra))

    assigned = sum(quotas.values())
    for label, _ in sorted(fractional, key=lambda item: (-item[1], item[0])):
        if assigned >= num_centers:
            break
        quotas[label] += 1
        assigned += 1
    return dict(sorted(quotas.items()))


def _kmeans_centers(
    latents: np.ndarray,
    *,
    num_centers: int,
    random_state: int,
    n_init: int,
    max_iter: int,
) -> list[np.ndarray]:
    if num_centers <= 0 or latents.shape[0] == 0:
        return []
    if num_centers == 1:
        return [np.mean(latents, axis=0)]
    if latents.shape[0] <= num_centers:
        return [latents[index].copy() for index in range(latents.shape[0])]

    from sklearn.cluster import KMeans

    kmeans = KMeans(
        n_clusters=num_centers,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
    )
    kmeans.fit(latents)
    return [
        center.astype(latents.dtype, copy=False)
        for center in kmeans.cluster_centers_
    ]


def _fill_missing_centers(
    *,
    centers: list[np.ndarray],
    fallback_centers: list[np.ndarray],
    latents: np.ndarray,
    num_centers: int,
    random_state: int,
) -> list[np.ndarray]:
    if len(centers) >= num_centers:
        return centers[:num_centers]

    filled = [center.copy() for center in centers]
    while len(filled) < num_centers and fallback_centers:
        selected = _pop_farthest_center(filled, fallback_centers)
        filled.append(selected)

    if len(filled) >= num_centers:
        return filled[:num_centers]

    rng = np.random.default_rng(random_state)
    scale = float(np.std(latents)) * 1e-4
    noise_scale = scale if np.isfinite(scale) and scale > 0 else 1e-4
    while len(filled) < num_centers:
        sample = latents[int(rng.integers(0, latents.shape[0]))].copy()
        sample += rng.normal(0.0, noise_scale, size=sample.shape).astype(
            sample.dtype
        )
        filled.append(sample)
    return filled


def _pop_farthest_center(
    selected_centers: list[np.ndarray],
    candidate_centers: list[np.ndarray],
) -> np.ndarray:
    if not selected_centers:
        return candidate_centers.pop(0).copy()

    selected = np.stack(selected_centers, axis=0)
    candidates = np.stack(candidate_centers, axis=0)
    distances = np.linalg.norm(
        candidates[:, None, :] - selected[None, :, :],
        axis=-1,
    )
    farthest_index = int(np.argmax(np.min(distances, axis=1)))
    return candidate_centers.pop(farthest_index).copy()


__all__ = [
    "CodebookInitResult",
    "QuantizeOutput",
    "VectorQuantizer",
    "classify_trajectory_directions",
    "initialize_codebook_from_directional_kmeans",
]
