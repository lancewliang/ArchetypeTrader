"""Running mean/std utilities for Phase II ablation tests."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Literal

import numpy as np


@dataclass
class RunningMeanStd:
    """Numerically stable running mean/std for small feature vectors."""

    shape: tuple[int, ...]
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        self.mean = np.zeros(self.shape, dtype=np.float64)
        self.var = np.ones(self.shape, dtype=np.float64)
        self.count = float(self.epsilon)

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == len(self.shape):
            values = values.reshape((1,) + self.shape)
        batch_mean = values.mean(axis=0)
        batch_var = values.var(axis=0)
        batch_count = values.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def normalize(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float64) - self.mean) / np.sqrt(
            self.var + self.epsilon
        )

    def _update_from_moments(self, batch_mean, batch_var, batch_count: int) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        self.mean = new_mean
        self.var = m_2 / total_count
        self.count = total_count


class RunningMeanStdAblationManager:
    """Ablation manager with per-env and delayed-merge protocols."""

    def __init__(
        self,
        num_envs: int,
        shape: tuple[int, ...],
        mode: Literal["per_env_only", "delayed_merge_next_rollout"],
    ) -> None:
        self.num_envs = int(num_envs)
        self.mode = mode
        self.env_stats: List[RunningMeanStd] = [
            RunningMeanStd(shape) for _ in range(self.num_envs)
        ]
        self.active_stats = RunningMeanStd(shape)
        self.pending_stats = RunningMeanStd(shape)

    def observe(self, env_id: int, values: np.ndarray) -> None:
        if self.mode == "per_env_only":
            self.env_stats[env_id].update(values)
        elif self.mode == "delayed_merge_next_rollout":
            self.pending_stats.update(values)
        else:
            raise ValueError(f"unknown running mean/std mode: {self.mode!r}")

    def normalize(self, env_id: int, values: np.ndarray) -> np.ndarray:
        if self.mode == "per_env_only":
            return self.env_stats[env_id].normalize(values)
        return self.active_stats.normalize(values)

    def finalize_rollout(self) -> None:
        """Publish pending stats for use by the next rollout."""
        if self.mode == "delayed_merge_next_rollout":
            self.active_stats = self.pending_stats
            self.pending_stats = RunningMeanStd(self.active_stats.shape)
