"""状态归一化工具 — 从 Phase 1 checkpoint 加载 norm_stats 并提供归一化方法。

Phase 1 训练时对 states 和 rewards 做了 per-feature z-score 归一化，
Phase 2/3/Evaluation 中所有喂给 Phase 1 模型（encoder/decoder）以及
Phase 2 selector / Phase 3 refinement agent 的 states 都必须使用
相同的归一化参数。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class StateNormalizer:
    """从 Phase 1 checkpoint 加载归一化统计量，提供 states/rewards 归一化。"""

    def __init__(self, norm_stats: Dict[str, Any]) -> None:
        self.state_mean = np.asarray(norm_stats["state_mean"], dtype=np.float32)
        self.state_std = np.maximum(
            np.asarray(norm_stats["state_std"], dtype=np.float32), 1e-8,
        )
        self.reward_mean = float(norm_stats["reward_mean"])
        self.reward_std = max(float(norm_stats["reward_std"]), 1e-8)

    def normalize_states(self, raw: np.ndarray) -> np.ndarray:
        """归一化 states，支持任意前导维度 (..., state_dim)。"""
        return (raw - self.state_mean) / self.state_std

    def normalize_rewards(self, raw: np.ndarray) -> np.ndarray:
        return (raw - self.reward_mean) / self.reward_std

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: str = "cpu") -> "StateNormalizer":
        """从 Phase 1 checkpoint 文件加载。"""
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if "norm_stats" not in ckpt:
            raise KeyError(
                f"Phase 1 checkpoint 中缺少 norm_stats，请重新运行 Phase 1 训练: {checkpoint_path}"
            )
        logger.info("从 checkpoint 加载归一化统计量")
        return cls(ckpt["norm_stats"])

    @classmethod
    def from_checkpoint_dict(cls, checkpoint: dict) -> Optional["StateNormalizer"]:
        """从已加载的 checkpoint dict 中提取，若无 norm_stats 返回 None。"""
        if "norm_stats" not in checkpoint:
            return None
        return cls(checkpoint["norm_stats"])
