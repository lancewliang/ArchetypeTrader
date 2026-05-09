from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import DataLoader

from ...model.vq_archetype import ArchetypeVQModel

if TYPE_CHECKING:
    from ..phase1_main import Phase1MainFlow


class Phase1Evaluator:
    """Phase I 评估与导出骨架。

    功能描述:
        承接 Phase I 训练后的评估、导出和报告生成逻辑。当前只保留流程骨架，
        后续实现会在这里补齐 checkpoint 选择、Phase II 产物导出、horizon label
        生成和报告写出。
    """

    def __init__(self, flow: Phase1MainFlow) -> None:
        self.flow = flow
        self.model: ArchetypeVQModel | None = None

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader[tuple[torch.Tensor, ...]],
        *,
        use_vq: bool,
    ) -> dict[str, float]:
        if self.model is None:
            raise RuntimeError("model must be initialized")

        self.model.eval()
        totals = self._empty_metric_totals()
        for batch in dataloader:
            batch = self._move_batch(batch)
            outputs = (
                self.model(batch)
                if use_vq
                else self.model.forward_pretrain(batch)
            )
            self._accumulate_metrics(totals, outputs, batch_size=batch[0].shape[0])
        return self._finalize_metrics(totals)

    def _move_batch(
        self,
        batch: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        states, actions, rewards = batch
        return (
            states.to(self.flow.device),
            actions.to(self.flow.device),
            rewards.to(self.flow.device),
        )

    def _empty_metric_totals(self) -> dict[str, float]:
        return {
            "samples": 0.0,
            "total_loss": 0.0,
            "reconstruction_loss": 0.0,
            "vq_loss": 0.0,
            "codebook_loss": 0.0,
            "commitment_loss": 0.0,
        }

    def _accumulate_metrics(
        self,
        totals: dict[str, float],
        outputs: Any,
        *,
        batch_size: int,
    ) -> None:
        totals["samples"] += batch_size
        totals["total_loss"] += float(outputs.total_loss.detach().cpu()) * batch_size
        totals["reconstruction_loss"] += (
            float(outputs.reconstruction_loss.detach().cpu()) * batch_size
        )
        totals["vq_loss"] += float(outputs.vq_loss.detach().cpu()) * batch_size
        totals["codebook_loss"] += (
            float(outputs.codebook_loss.detach().cpu()) * batch_size
        )
        totals["commitment_loss"] += (
            float(outputs.commitment_loss.detach().cpu()) * batch_size
        )

    def _finalize_metrics(self, totals: dict[str, float]) -> dict[str, float]:
        samples = max(totals.pop("samples"), 1.0)
        return {name: value / samples for name, value in totals.items()}
 
