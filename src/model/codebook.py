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


@dataclass(frozen=True)
class CodebookDeadCodeResetResult:
    """Dead-code reset 结果摘要。"""

    method: str
    num_samples: int
    num_centers: int
    min_occupancy: float
    dead_code_indices: tuple[int, ...]
    reset_code_indices: tuple[int, ...]
    source_sample_indices: tuple[int, ...]
    occupancy: tuple[float, ...]

    @property
    def reset_count(self) -> int:
        """实际被重置的 code 数量。"""

        return len(self.reset_code_indices)


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
    def reset_dead_codes(
        self,
        latents: LatentTensor,
        code_indices: ArchetypeLabelTensor | None = None,
        *,
        min_occupancy: float = 0.001,
        max_resets: int | None = None,
        random_state: int = 0,
        jitter_scale: float = 1e-4,
    ) -> CodebookDeadCodeResetResult:
        """把低占用 code 重置到当前 batch/epoch 中量化误差较高的 latent。

        该方法是训练期防坍缩机制：先按 ``code_indices`` 统计 codebook
        occupancy，找出 ``occupancy < min_occupancy`` 的 dead codes，再用
        当前 latent 中离最近 codebook 最远的样本重新初始化这些 code。相比
        随机噪声重置，高误差样本通常代表当前 codebook 覆盖不足的区域。
        """

        return reset_dead_codes_from_latents(
            self.embedding,
            latents,
            code_indices,
            min_occupancy=min_occupancy,
            max_resets=max_resets,
            random_state=random_state,
            jitter_scale=jitter_scale,
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


@torch.no_grad()
def reset_dead_codes_from_latents(
    embedding: nn.Embedding,
    latents: torch.Tensor,
    code_indices: torch.Tensor | None = None,
    *,
    min_occupancy: float = 0.001,
    max_resets: int | None = None,
    random_state: int = 0,
    jitter_scale: float = 1e-4,
) -> CodebookDeadCodeResetResult:
    """根据当前 latent 分布重置 dead codebook entries。"""

    num_archetypes = int(embedding.num_embeddings)
    latent_dim = int(embedding.embedding_dim)
    if min_occupancy < 0.0:
        raise ValueError("min_occupancy must be non-negative")
    if max_resets is not None and max_resets < 0:
        raise ValueError("max_resets must be non-negative")
    if jitter_scale < 0.0:
        raise ValueError("jitter_scale must be non-negative")

    if latents.ndim != 2:
        raise ValueError("latents must have shape [num_samples, latent_dim]")
    if latents.shape[-1] != latent_dim:
        raise ValueError(
            f"latents last dim must be {latent_dim}, got {latents.shape[-1]}"
        )
    num_samples = int(latents.shape[0])
    if num_samples == 0:
        raise ValueError("latents must contain at least one sample")

    latents_device = latents.detach().to(
        device=embedding.weight.device,
        dtype=embedding.weight.dtype,
    )
    distances = torch.cdist(latents_device, embedding.weight, p=2).pow(2)
    nearest_distances, inferred_indices = distances.min(dim=-1)

    if code_indices is None:
        assignment_indices = inferred_indices
    else:
        if code_indices.ndim != 1:
            raise ValueError("code_indices must have shape [num_samples]")
        if code_indices.shape[0] != num_samples:
            raise ValueError(
                f"code_indices length must be {num_samples}, got {code_indices.shape[0]}"
            )
        assignment_indices = code_indices.detach().to(
            device=embedding.weight.device,
            dtype=torch.long,
        )
        invalid_assignments = (assignment_indices < 0) | (
            assignment_indices >= num_archetypes
        )
        if bool(torch.any(invalid_assignments)):
            raise ValueError("code_indices must be in [0, num_archetypes)")

    counts = torch.bincount(assignment_indices, minlength=num_archetypes)
    occupancy_tensor = counts.to(dtype=torch.float64) / float(num_samples)
    occupancy = tuple(float(value) for value in occupancy_tensor.cpu().tolist())
    dead_code_indices = tuple(
        int(index)
        for index, value in enumerate(occupancy)
        if value < min_occupancy
    )
    if not dead_code_indices or max_resets == 0:
        return CodebookDeadCodeResetResult(
            method="dead_code_reset",
            num_samples=num_samples,
            num_centers=num_archetypes,
            min_occupancy=float(min_occupancy),
            dead_code_indices=dead_code_indices,
            reset_code_indices=(),
            source_sample_indices=(),
            occupancy=occupancy,
        )

    reset_code_indices = dead_code_indices
    if max_resets is not None:
        reset_code_indices = reset_code_indices[:max_resets]
    reset_count = len(reset_code_indices)
    if reset_count == 0:
        return CodebookDeadCodeResetResult(
            method="dead_code_reset",
            num_samples=num_samples,
            num_centers=num_archetypes,
            min_occupancy=float(min_occupancy),
            dead_code_indices=dead_code_indices,
            reset_code_indices=(),
            source_sample_indices=(),
            occupancy=occupancy,
        )

    source_indices = _select_dead_code_reset_sources(
        nearest_distances,
        reset_count=reset_count,
        random_state=random_state,
    )
    replacement = latents_device[source_indices].clone()

    latent_std = float(torch.std(latents_device).detach().cpu().item())
    noise_std = latent_std * jitter_scale
    if not np.isfinite(noise_std) or noise_std <= 0.0:
        noise_std = jitter_scale
    if noise_std > 0.0:
        generator = torch.Generator(device=embedding.weight.device)
        generator.manual_seed(int(random_state))
        replacement = replacement + torch.randn(
            replacement.shape,
            generator=generator,
            device=replacement.device,
            dtype=replacement.dtype,
        ) * noise_std

    reset_index_tensor = torch.as_tensor(
        reset_code_indices,
        dtype=torch.long,
        device=embedding.weight.device,
    )
    embedding.weight.index_copy_(0, reset_index_tensor, replacement)

    return CodebookDeadCodeResetResult(
        method="dead_code_reset",
        num_samples=num_samples,
        num_centers=num_archetypes,
        min_occupancy=float(min_occupancy),
        dead_code_indices=dead_code_indices,
        reset_code_indices=tuple(int(index) for index in reset_code_indices),
        source_sample_indices=tuple(int(index) for index in source_indices.cpu().tolist()),
        occupancy=occupancy,
    )


def _select_dead_code_reset_sources(
    nearest_distances: torch.Tensor,
    *,
    reset_count: int,
    random_state: int,
) -> torch.Tensor:
    if reset_count <= 0:
        return torch.empty(0, dtype=torch.long, device=nearest_distances.device)

    num_samples = int(nearest_distances.shape[0])
    top_count = min(reset_count, num_samples)
    source_indices = torch.topk(nearest_distances, k=top_count, largest=True).indices
    if top_count >= reset_count:
        return source_indices

    rng = np.random.default_rng(random_state)
    extra = rng.choice(num_samples, size=reset_count - top_count, replace=True)
    extra_indices = torch.as_tensor(
        extra,
        dtype=torch.long,
        device=nearest_distances.device,
    )
    return torch.cat([source_indices, extra_indices], dim=0)


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
    "CodebookDeadCodeResetResult",
    "CodebookInitResult",
    "QuantizeOutput",
    "VectorQuantizer",
    "classify_trajectory_directions",
    "initialize_codebook_from_directional_kmeans",
    "reset_dead_codes_from_latents",
]
