from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from ...model.tensor_data_types import (
    TrajectoryTensorBatch,
    move_trajectory_batch_to_device,
)
from ...model.vq_archetype import ArchetypeVQModel
from ..metrics import Phase1Metrics



class Phase1Evaluator:
    """Phase I 评估与导出骨架。

    功能描述:
        承接 Phase I 训练后的评估和 horizon label 导出逻辑。当前只保留流程
        骨架，后续实现会在这里补齐离线评估和 horizon label 生成。
    """

    def __init__(
        self,
        model: ArchetypeVQModel,
        device: torch.device | str,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader[TrajectoryTensorBatch],
        *,
        stage: str | None = None,
        split: str | None = None,
        epoch: int | None = None,
    ) -> Phase1Metrics:
        self.model.eval()
        totals = Phase1Metrics(stage=stage, split=split, epoch=epoch)
        for batch in dataloader:
            batch = move_trajectory_batch_to_device(batch, self.device)
            outputs = self.model(batch)
            totals.add_batch(batch_size=batch[0].shape[0], outputs=outputs, actions=batch[3])
        return totals.averaged()
