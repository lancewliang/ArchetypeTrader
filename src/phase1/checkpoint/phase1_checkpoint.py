"""Phase I checkpoint payload types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

from ..metrics import Phase1Metrics, Phase1ValidationResult

Phase1CheckpointStage = Literal["pretrain", "vq"]
Phase1CheckpointConfig = Mapping[str, Any]
Phase1StateDict = Mapping[str, Any]
Phase1CheckpointMetrics = Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class Phase1Checkpoint:
    """Phase I model checkpoint payload.

    This payload stores the training state needed to resume/export a model.
    Codebook validation metrics are stored separately as
    ``Phase1ValidationCheckpoint`` JSON payloads.
    """

    stage: Phase1CheckpointStage
    epoch: int
    is_best: bool
    config: Phase1CheckpointConfig
    model_state_dict: Phase1StateDict
    optimizer_state_dict: Phase1StateDict


    def to_dict(self) -> dict[str, object]:
        """Convert to a torch.save-friendly mapping."""

        return {
            "stage": self.stage,
            "epoch": self.epoch,
            "is_best": self.is_best,
            "config": dict(self.config),
            "model_state_dict": dict(self.model_state_dict),
            "optimizer_state_dict": dict(self.optimizer_state_dict)
         
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase1Checkpoint":
        """Restore a checkpoint payload from a mapping."""

        stage = payload["stage"]
        if stage not in {"pretrain", "vq"}:
            raise ValueError(f"unsupported phase1 checkpoint stage: {stage!r}")

        return cls(
            stage=stage,
            epoch=int(payload["epoch"]),
            is_best=bool(payload["is_best"]),
            config=dict(payload["config"]),
            model_state_dict=dict(payload["model_state_dict"]),
            optimizer_state_dict=dict(payload["optimizer_state_dict"]),
        )


@dataclass(frozen=True)
class Phase1ValidationCheckpoint:
    """Phase I 单个 epoch 的强类型验证检验点 payload。"""
    stage: str
    epoch: int        
    train: Phase1Metrics
    val: Phase1Metrics
    codebook_validation: Phase1ValidationResult

    def to_dict(self) -> dict[str, object]:
        """转换为 JSON 友好的普通字典。"""

        return {
            "stage": self.stage,
            "epoch": self.epoch,
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
            stage=payload["stage"],
            epoch=payload["epoch"],
            train=Phase1Metrics.from_dict(payload["train"]),
            val=Phase1Metrics.from_dict(payload["val"]),
            codebook_validation=Phase1ValidationResult.from_dict(
                payload["codebook_validation"]
            ),
        )
