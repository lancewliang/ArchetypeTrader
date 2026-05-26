"""Phase I checkpoint payload types."""

from __future__ import annotations

from typing import Any, Literal, Mapping

from src.utils import PydanticMappingModel

from ..metrics import Phase1Metrics, Phase1ValidationResult

Phase1CheckpointStage = Literal["pretrain", "vq"]
Phase1CheckpointConfig = Mapping[str, Any]
Phase1StateDict = Mapping[str, Any]
Phase1CheckpointMetrics = Mapping[str, Mapping[str, Any]]


class Phase1Checkpoint(PydanticMappingModel):
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


class Phase1ValidationCheckpoint(PydanticMappingModel):
    """Phase I 单个 epoch 的强类型验证检验点 payload。"""

    stage: str
    epoch: int
    train: Phase1Metrics
    val: Phase1Metrics
    codebook_validation: Phase1ValidationResult
