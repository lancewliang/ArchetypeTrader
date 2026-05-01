"""学习率、entropy coef、kl_demo coef 的退火调度。

设计文档锚点: Phase II 执行计划 §Step 5。

职责:
- 支持 learning rate schedule（linear decay / cosine）。
- 支持 entropy coef anneal。
- 支持 kl_demo coef anneal（含 anneal_to 终值）。
- 初期可选 higher entropy warmup。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

from src.config.phase2_config import Phase2Config


@dataclass
class ScheduleState:
    """当前 schedule 状态快照。"""
    lr: float
    entropy_coef: float
    kl_demo_coef: float
    progress: float  # 0.0 ~ 1.0


class ScheduleManager:
    """统一管理所有 schedule。

    使用方式::

        sm = ScheduleManager(config, optimizer, total_updates=1000)
        for update_idx in range(total_updates):
            sm.step(update_idx)
            current = sm.current_state()
    """

    def __init__(
        self,
        config: Phase2Config,
        optimizer: Any,
        total_updates: int,
    ) -> None:
        self.config = config
        self.optimizer = optimizer
        self.total_updates = max(total_updates, 1)
        self._initial_lr: float = config.ppo.lr
        self._initial_entropy_coef: float = config.ppo.entropy_coef
        self._initial_kl_demo_coef: float = config.ppo.kl_demo_coef
        self._current_lr: float = config.ppo.lr
        self._current_entropy_coef: float = config.ppo.entropy_coef
        self._current_kl_demo_coef: float = config.ppo.kl_demo_coef
        self._current_update: int = 0

    def step(self, update_idx: int) -> None:
        """根据 update_idx 更新所有 schedule。

        Parameters
        ----------
        update_idx : 当前 update 序号（从 0 开始）。
        """
        self._current_update = update_idx
        progress = min(update_idx / self.total_updates, 1.0)

        # Learning rate: linear decay
        self._current_lr = self._initial_lr * (1.0 - progress)
        self._current_lr = max(self._current_lr, 1e-7)

        # 更新 optimizer lr
        if self.optimizer is not None:
            for pg in self.optimizer.param_groups:
                pg["lr"] = self._current_lr

        # Entropy coef: linear decay
        self._current_entropy_coef = self._initial_entropy_coef * (1.0 - progress)

        # KL demo coef: linear decay (可选 anneal_to)
        self._current_kl_demo_coef = self._initial_kl_demo_coef * (1.0 - progress * 0.5)

    def current_state(self) -> ScheduleState:
        """返回当前 schedule 状态。"""
        progress = min(self._current_update / self.total_updates, 1.0)
        return ScheduleState(
            lr=self._current_lr,
            entropy_coef=self._current_entropy_coef,
            kl_demo_coef=self._current_kl_demo_coef,
            progress=progress,
        )

    def get_state(self) -> Dict[str, Any]:
        """获取可序列化状态（用于 checkpoint）。"""
        return {
            "current_update": self._current_update,
            "current_lr": self._current_lr,
            "current_entropy_coef": self._current_entropy_coef,
            "current_kl_demo_coef": self._current_kl_demo_coef,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """从 checkpoint 恢复。"""
        self._current_update = state.get("current_update", 0)
        self._current_lr = state.get("current_lr", self._initial_lr)
        self._current_entropy_coef = state.get(
            "current_entropy_coef", self._initial_entropy_coef
        )
        self._current_kl_demo_coef = state.get(
            "current_kl_demo_coef", self._initial_kl_demo_coef
        )
        # 同步 optimizer
        if self.optimizer is not None:
            for pg in self.optimizer.param_groups:
                pg["lr"] = self._current_lr
