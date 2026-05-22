"""Trajectory tensor batch validation helpers."""

from __future__ import annotations

from .tensor_data_types import TrajectoryTensorBatch


def normalize_trajectory_batch(batch: TrajectoryTensorBatch) -> TrajectoryTensorBatch:
    """统一模型输入形状，减少训练代码里的样板转换。"""

    if len(batch) >= 6:
        states, relative_states, trend_states, actions, rewards, sample_ids = batch[:6]
    else:
        raise ValueError(
            "trajectory batch must be "
            "(states, relative_states, trend_states, actions, rewards, sample_ids)"
        )
    if states.ndim != 3:
        raise ValueError("states must have shape [batch, horizon, state_dim]")
    if relative_states.ndim != 3:
        raise ValueError(
            "relative_states must have shape [batch, horizon, relative_feature_dim]"
        )
    if trend_states.ndim != 3:
        raise ValueError(
            "trend_states must have shape [batch, horizon, trend_feature_dim]"
        )
    if actions.ndim != 2:
        raise ValueError("actions must have shape [batch, horizon]")
    if rewards.ndim == 2:
        rewards = rewards.unsqueeze(-1)
    if rewards.ndim != 3 or rewards.shape[-1] != 1:
        raise ValueError(
            "rewards must have shape [batch, horizon] or [batch, horizon, 1]"
        )
    if sample_ids.ndim != 1:
        raise ValueError("sample_ids must have shape [batch]")
    if states.shape[:2] != actions.shape:
        raise ValueError("states and actions must share [batch, horizon]")
    if relative_states.shape[:2] != states.shape[:2]:
        raise ValueError("relative_states and states must share [batch, horizon]")
    if trend_states.shape[:2] != states.shape[:2]:
        raise ValueError("trend_states and states must share [batch, horizon]")
    if states.shape[:2] != rewards.shape[:2]:
        raise ValueError("states and rewards must share [batch, horizon]")
    if sample_ids.shape != (states.shape[0],):
        raise ValueError("sample_ids must match batch size")
    return states, relative_states, trend_states, actions, rewards, sample_ids


__all__ = ["normalize_trajectory_batch"]
