"""Phase II distribution-shift / OOD monitoring."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from src.config.phase2_config import DistributionShiftConfig


@dataclass
class DistributionShiftStats:
    """Frozen train-state statistics."""
    mean: List[float]
    std: List[float]
    dims: List[int]


@dataclass
class DistributionShiftResult:
    """OOD scoring result."""
    score: float
    triggered: bool
    fallback_action: Optional[str] = None
    per_dim_scores: Dict[int, float] = field(default_factory=dict)


class Phase2DistributionShiftMonitor:
    """Simple z-score based OOD monitor for selector states."""

    def __init__(
        self,
        config: DistributionShiftConfig,
        dims: Optional[Sequence[int]] = None,
    ) -> None:
        self.config = config
        self.dims = list(dims) if dims is not None else None
        self.stats: Optional[DistributionShiftStats] = None

    def fit(self, states: Iterable[Sequence[float]]) -> DistributionShiftStats:
        arr = np.asarray(list(states), dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError("states 必须是二维数组")
        dims = self.dims if self.dims is not None else list(range(arr.shape[1]))
        selected = arr[:, dims]
        mean = selected.mean(axis=0)
        std = selected.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        self.stats = DistributionShiftStats(
            mean=mean.astype(float).tolist(),
            std=std.astype(float).tolist(),
            dims=list(dims),
        )
        return self.stats

    def score(self, state: Sequence[float]) -> DistributionShiftResult:
        if self.stats is None:
            raise RuntimeError("必须先调用 fit()")
        arr = np.asarray(state, dtype=np.float32)
        selected = arr[self.stats.dims]
        mean = np.asarray(self.stats.mean, dtype=np.float32)
        std = np.asarray(self.stats.std, dtype=np.float32)
        z = np.abs((selected - mean) / std)
        per_dim = {
            int(dim): float(score)
            for dim, score in zip(self.stats.dims, z)
        }
        score = float(z.max()) if z.size else 0.0
        triggered = score > self.config.threshold
        return DistributionShiftResult(
            score=score,
            triggered=triggered,
            fallback_action=self.config.fallback_action if triggered else None,
            per_dim_scores=per_dim,
        )

