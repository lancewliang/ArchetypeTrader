"""Phase I checkpoint payload types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from ..metrics import Phase1Metrics, Phase1ValidationResult



@dataclass(frozen=True)
class Phase1ValidationCheckpoint:
    """Phase I 单个 epoch 的强类型验证检验点 payload。"""

    train: Phase1Metrics
    val: Phase1Metrics
    codebook_validation: Phase1ValidationResult

    def to_dict(self) -> dict[str, object]:
        """转换为 JSON 友好的普通字典。"""

        return {
            "train": self.train.to_dict(include_context=True),
            "val": self.val.to_dict(include_context=True),
            "codebook_validation": self.codebook_validation.to_dict(),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "Phase1ValidationCheckpoint":
        """从 JSON payload 恢复强类型验证检验点。"""

        return cls(
            train=Phase1Metrics.from_dict(payload["train"]),
            val=Phase1Metrics.from_dict(payload["val"]),
            codebook_validation=Phase1ValidationResult.from_dict(
                payload["codebook_validation"]
            ),
        )
