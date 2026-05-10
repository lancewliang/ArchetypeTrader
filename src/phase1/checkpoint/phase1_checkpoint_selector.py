"""Phase I best-checkpoint selector skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

from .phase1_checkpoint import Phase1Checkpoint, Phase1CheckpointStage


Phase1CheckpointMetricMode = Literal["min", "max"]
"""Best checkpoint metric comparison direction."""


@dataclass(frozen=True)
class Phase1CheckpointSelectionConfig:
    """Phase I checkpoint selection policy.

    功能描述:
        描述如何从一组 Phase I checkpoint 中选出 best checkpoint。selector
        读取所有候选 checkpoint 后，应从指定 split 的评估指标中取出
        ``metric_name``，再按 ``metric_mode`` 判断越小越好或越大越好。

    论文描述:
        Phase I 的 best checkpoint 应由离线评估者输出的指标决定，而不是由
        最后一个 epoch 或训练 loss 直接决定。这样可以固定最能代表稳定
        archetype discovery 结果的 encoder、decoder 和 VQ codebook。
    """

    stage: Phase1CheckpointStage = "vq"
    split: str = "val"
    metric_name: str = "total_loss"
    metric_mode: Phase1CheckpointMetricMode = "min"


@dataclass(frozen=True)
class Phase1CheckpointSelectionResult:
    """Phase I best-checkpoint selection result."""

    checkpoint_path: Path
    checkpoint: Phase1Checkpoint
    metric_name: str
    metric_value: float
    metric_mode: Phase1CheckpointMetricMode


class Phase1CheckpointSelector:
    """Select the best Phase I checkpoint from evaluator metrics.

    功能描述:
        负责扫描一个 Phase I checkpoint 集合，读取每个 checkpoint 中由
        evaluator 写入的 split-level metrics，并按
        ``Phase1CheckpointSelectionConfig`` 选出 best checkpoint。当前类只定义
        骨架和调用契约，后续实现再补齐文件扫描、checkpoint 反序列化、指标
        缺失处理、tie-break 规则和 best checkpoint 固化策略。

    论文描述:
        Phase I 训练会在多个 epoch 产生 VQ checkpoint。selector 应用验证集或
        评估集指标挑选最好的 archetype discovery 结果，后续 Phase II/III 只
        复用该 checkpoint 导出的 encoder、decoder 和 codebook，避免训练末期
        波动污染离线 archetypes。
    """

    def __init__(self, config: Phase1CheckpointSelectionConfig | None = None) -> None:
        self.config = config or Phase1CheckpointSelectionConfig()

    def select_best(
        self,
        checkpoint_paths: Sequence[str | Path],
    ) -> Phase1CheckpointSelectionResult:
        """Select the best checkpoint from explicit checkpoint paths."""

        ...

    def select_best_from_dir(
        self,
        checkpoint_dir: str | Path,
    ) -> Phase1CheckpointSelectionResult:
        """Scan a checkpoint directory and select the best checkpoint."""

        ...

    def list_candidate_checkpoints(
        self,
        checkpoint_dir: str | Path,
    ) -> list[Path]:
        """Return candidate checkpoint paths for the configured stage."""

        ...

    def load_checkpoint(self, checkpoint_path: str | Path) -> Phase1Checkpoint:
        """Load one Phase I checkpoint payload."""

        ...

    def metric_value(self, checkpoint: Phase1Checkpoint) -> float:
        """Read the configured evaluator metric from one checkpoint."""

        ...

    def is_better(self, candidate_value: float, current_best_value: float) -> bool:
        """Compare two metric values according to the configured metric mode."""

        ...

    def mark_best_checkpoint(
        self,
        selection: Phase1CheckpointSelectionResult,
        output_path: str | Path | None = None,
    ) -> None:
        """Persist or mark the selected checkpoint as the Phase I best checkpoint."""

        ...
