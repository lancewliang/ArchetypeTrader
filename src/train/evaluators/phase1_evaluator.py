from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from ...model.vq_archetype import ArchetypeVQModel
from ..metrics import Phase1Metrics


class Phase1Evaluator:
    """Phase I 评估与导出骨架。

    功能描述:
        承接 Phase I 训练后的评估、导出和报告生成逻辑。当前只保留流程骨架，
        后续实现会在这里补齐 checkpoint 选择、Phase II 产物导出、horizon label
        生成和报告写出。
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
        dataloader: DataLoader[tuple[torch.Tensor, ...]],
        *,
        use_vq: bool,
        stage: str | None = None,
        split: str | None = None,
        epoch: int | None = None,
    ) -> Phase1Metrics:
        self.model.eval()
        totals = Phase1Metrics(stage=stage, split=split, epoch=epoch)
        for batch in dataloader:
            batch = self._move_batch(batch)
            outputs = (
                self.model(batch)
                if use_vq
                else self.model.forward_pretrain(batch)
            )
            totals.add_batch(batch_size=batch[0].shape[0], outputs=outputs)
        return totals.averaged()

    def _move_batch(
        self,
        batch: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        states, actions, rewards = batch
        return (
            states.to(self.device),
            actions.to(self.device),
            rewards.to(self.device),
        )
