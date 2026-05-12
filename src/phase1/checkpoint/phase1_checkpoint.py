"""Phase I checkpoint payload types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping


Phase1CheckpointStage = Literal["pretrain", "vq"]
"""Phase I checkpoint 所属训练阶段。"""

Phase1CheckpointConfig = Mapping[str, object]
"""Phase I checkpoint 中保存的配置快照。"""

Phase1StateDict = Mapping[str, object]
"""Phase I checkpoint 中保存的模型或优化器 state dict。"""

Phase1CheckpointMetrics = Mapping[str, Mapping[str, object]]
"""Phase I checkpoint 中保存的 split-level 指标。"""


@dataclass(frozen=True)
class Phase1Checkpoint:
    """Phase I checkpoint 的强类型 payload。

    该类型集中定义 store 层写 checkpoint 时必须提供的字段，避免训练循环用
    裸 ``dict[str, Any]`` 传参时出现 key 拼写错误、漏字段或阶段名不合法。
    """

    stage: Phase1CheckpointStage
    epoch: int
    is_best: bool
    config: Phase1CheckpointConfig
    model_state_dict: Phase1StateDict
    optimizer_state_dict: Phase1StateDict
 

    def to_dict(self) -> dict[str, object]:
        """转换为 torch/json 友好的普通字典。"""

        return {
            "stage": self.stage,
            "epoch": self.epoch,
            "is_best": self.is_best,
            "config": dict(self.config),
            "model_state_dict": dict(self.model_state_dict),
            "optimizer_state_dict": dict(self.optimizer_state_dict)           
        }
