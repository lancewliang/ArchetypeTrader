"""Shared Phase II reward scaling helpers."""
from __future__ import annotations

import numpy as np

from src.config.phase2_config import Phase2Config


def scale_phase2_reward(config: Phase2Config, reward: float) -> tuple[float, bool]:
    """Apply Phase II reward scaling and optional clipping.

    Returns the scaled reward and whether clipping changed the scaled value.
    This is intentionally shared by PPO rollout and replay/reporting paths so
    ``reward_scaled`` has one meaning across artifacts.
    """
    method = config.reward_scaling.method
    if method == "divide_by_horizon":
        scaled = reward / max(config.horizon, 1)
    elif method == "raw":
        scaled = reward
    else:
        raise ValueError(f"unknown reward_scaling.method: {method!r}")

    clip_range = config.reward_scaling.clip_range
    if clip_range is None:
        return float(scaled), False

    clipped = float(np.clip(scaled, -float(clip_range), float(clip_range)))
    return clipped, bool(clipped != scaled)
