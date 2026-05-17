"""Phase I 训练期基础指标聚合器。

本文件只负责训练/验证循环中的轻量指标统计，不承载五层 codebook validation
结果。五层 validation 的中间数据、规则判定和评分分别由
``phase1_validation_data_schema.py``、``phase1_validation_rules.py`` 和
``phase1_validation_score.py`` 负责。

使用场景:
    1. ``Phase1MainFlow._run_epoch()`` 在训练或预训练时逐 batch 调用
       ``add_batch()``，最后调用 ``averaged()`` 得到 epoch 指标；
    2. ``Phase1Evaluator.evaluate()`` 在 validation/test split 上逐 batch
       统计基础 loss 和 action reconstruction accuracy；
    3. report、datastore JSON 或日志通过 ``to_dict()`` 保存这些训练期基础指标。

设计约束:
    - loss 按 sample 加权累加，避免最后一个小 batch 影响平均值；
    - action accuracy 按所有 timestep 逐点统计，不能简单对 batch accuracy 求均值；
    - 本类只依赖模型输出 ``VqModelOutputs``，不访问 model、DataLoader 或文件系统。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

from ...model.vq_archetype import VqModelOutputs


def _tensor_to_float(value: torch.Tensor | float | int) -> float:
    """把标量 Tensor 或 Python number 转为 float。

    使用场景:
        模型输出中的 loss 通常是带梯度的标量 Tensor。指标聚合时只需要数值，
        因此需要 ``detach`` 后搬到 CPU 并转为 ``float``。
    """

    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("metric tensor must be scalar")
        return float(value.detach().cpu().item())
    return float(value)


@dataclass
class Phase1Metrics:
    """Phase I 训练期基础指标。

    功能说明:
        聚合一个 epoch 或一个 split 的基础训练指标，包括总 loss、重构 loss、
        VQ loss、codebook loss、commitment loss 和动作重构准确率。

    使用场景:
        训练循环创建空对象后反复调用 ``add_batch()``，最后调用 ``averaged()``；
        report/checkpoint 保存时调用 ``to_dict()``。本类不用于五层 codebook
        validation hard gate。
    """

    # 训练阶段名称，例如 "pretrain"、"train" 或 "eval"。用于日志和 report 分组。
    stage: str | None = None

    # 数据 split 名称，例如 "train"、"val" 或 "test"。用于区分指标来源。
    split: str | None = None

    # 当前 epoch 编号。用于把指标和 checkpoint 对齐。
    epoch: int | None = None

    # 已累计样本数。训练中用于加权平均，序列化时用于审计指标覆盖范围。
    num_samples: int = 0

    # 总 loss。累加器状态下是按 sample 加权总和；averaged() 后是样本平均值。
    total_loss: float = 0.0

    # 动作重构 cross entropy loss。累加器状态下是加权总和；averaged() 后是样本平均值。
    reconstruction_loss: float = 0.0

    # VQ 总损失。累加器状态下是加权总和；averaged() 后是样本平均值。
    vq_loss: float = 0.0

    # codebook loss。累加器状态下是加权总和；averaged() 后是样本平均值。
    codebook_loss: float = 0.0

    # commitment loss。累加器状态下是加权总和；averaged() 后是样本平均值。
    commitment_loss: float = 0.0

    # morphology 辅助分类 loss。未启用时为 0。
    morphology_loss: float = 0.0

    # morphology-motif pair 辅助分类 loss。未启用时为 0。
    pair_loss: float = 0.0

    # codebook embedding diversity regularization loss。未启用时为 0。
    codebook_diversity_loss: float = 0.0

    # decoder action prototype diversity regularization loss。未启用时为 0。
    prototype_diversity_loss: float = 0.0

    # 动作重构准确率。训练中会随 batch 更新为当前累计准确率；averaged() 后为最终准确率。
    action_accuracy: float = 0.0

    # 已正确重构的 action timestep 数。用于按 timestep 计算 accuracy，不直接写入默认 dict。
    correct_actions: int = field(default=0, repr=False)

    # 已统计的 action timestep 总数。用于按 timestep 计算 accuracy，不直接写入默认 dict。
    total_actions: int = field(default=0, repr=False)

    def add_batch(
        self,
        batch_size: int,
        outputs: VqModelOutputs,
        actions: torch.Tensor | None = None,
        total_loss: torch.Tensor | float | int | None = None,
        morphology_loss: torch.Tensor | float | int | None = None,
        pair_loss: torch.Tensor | float | int | None = None,
        codebook_diversity_loss: torch.Tensor | float | int | None = None,
        prototype_diversity_loss: torch.Tensor | float | int | None = None,
    ) -> None:
        """累加一个 batch 的基础指标。

        参数:
            batch_size:
                当前 batch 的样本数，用于对 batch-level loss 做 sample 加权。
            outputs:
                ``ArchetypeVQModel.forward()`` 或 ``forward_pretrain()`` 的输出。
            actions:
                DP teacher target actions，形状通常为 ``[batch, horizon]``。

        使用场景:
            训练和评估循环中每处理完一个 batch 调用一次。若未传 ``actions``，
            本方法仍会累计 loss，但无法更新 action accuracy。
        """

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.num_samples += int(batch_size)
        self.total_loss += _tensor_to_float(
            outputs.total_loss if total_loss is None else total_loss
        ) * batch_size
        self.reconstruction_loss += _tensor_to_float(outputs.reconstruction_loss) * batch_size
        self.vq_loss += _tensor_to_float(outputs.vq_loss) * batch_size
        self.codebook_loss += _tensor_to_float(outputs.codebook_loss) * batch_size
        self.commitment_loss += _tensor_to_float(outputs.commitment_loss) * batch_size
        if morphology_loss is not None:
            self.morphology_loss += _tensor_to_float(morphology_loss) * batch_size
        if pair_loss is not None:
            self.pair_loss += _tensor_to_float(pair_loss) * batch_size
        if codebook_diversity_loss is not None:
            self.codebook_diversity_loss += (
                _tensor_to_float(codebook_diversity_loss) * batch_size
            )
        if prototype_diversity_loss is not None:
            self.prototype_diversity_loss += (
                _tensor_to_float(prototype_diversity_loss) * batch_size
            )

        if actions is not None:
            self._add_action_accuracy(outputs=outputs, actions=actions)

    def _add_action_accuracy(self, *, outputs: VqModelOutputs, actions: torch.Tensor) -> None:
        """按 timestep 累加动作重构准确率计数。"""

        targets = actions.detach()
        if targets.ndim == 3 and targets.shape[-1] == 1:
            targets = targets.squeeze(-1)
        if targets.ndim != 2:
            raise ValueError("actions must have shape [batch, horizon]")

        predicted_actions = outputs.action_logits.detach().argmax(dim=-1)
        if predicted_actions.shape != targets.shape:
            raise ValueError(
                "predicted actions and target actions must have the same shape, "
                f"got {tuple(predicted_actions.shape)} and {tuple(targets.shape)}"
            )

        correct = (predicted_actions.cpu() == targets.long().cpu()).sum().item()
        total = targets.numel()
        self.correct_actions += int(correct)
        self.total_actions += int(total)
        self.action_accuracy = (
            self.correct_actions / self.total_actions if self.total_actions > 0 else 0.0
        )

    def averaged(self) -> "Phase1Metrics":
        """返回样本平均后的指标对象。

        使用场景:
            epoch 或 split 结束后调用。原对象保持累加器状态不变，返回的新对象
            可以安全写入 checkpoint/report。
        """

        if self.num_samples <= 0:
            return Phase1Metrics(
                stage=self.stage,
                split=self.split,
                epoch=self.epoch,
                num_samples=0,
                correct_actions=self.correct_actions,
                total_actions=self.total_actions,
            )

        return Phase1Metrics(
            stage=self.stage,
            split=self.split,
            epoch=self.epoch,
            num_samples=self.num_samples,
            total_loss=self.total_loss / self.num_samples,
            reconstruction_loss=self.reconstruction_loss / self.num_samples,
            vq_loss=self.vq_loss / self.num_samples,
            codebook_loss=self.codebook_loss / self.num_samples,
            commitment_loss=self.commitment_loss / self.num_samples,
            morphology_loss=self.morphology_loss / self.num_samples,
            pair_loss=self.pair_loss / self.num_samples,
            codebook_diversity_loss=self.codebook_diversity_loss / self.num_samples,
            prototype_diversity_loss=(
                self.prototype_diversity_loss / self.num_samples
            ),
            action_accuracy=(
                self.correct_actions / self.total_actions if self.total_actions > 0 else 0.0
            ),
            correct_actions=self.correct_actions,
            total_actions=self.total_actions,
        )

    def to_dict(self, *, include_context: bool = True) -> dict[str, Any]:
        """序列化为普通 dict。

        参数:
            include_context:
                为 True 时包含 ``stage/split/epoch/num_samples``，适合直接写入
                datastore JSON/report；为 False 时只输出指标字段，适合嵌入其他 payload。
        """

        payload: dict[str, Any] = {
            "total_loss": self.total_loss,
            "reconstruction_loss": self.reconstruction_loss,
            "vq_loss": self.vq_loss,
            "codebook_loss": self.codebook_loss,
            "commitment_loss": self.commitment_loss,
            "morphology_loss": self.morphology_loss,
            "pair_loss": self.pair_loss,
            "codebook_diversity_loss": self.codebook_diversity_loss,
            "prototype_diversity_loss": self.prototype_diversity_loss,
            "action_accuracy": self.action_accuracy,
        }
        if include_context:
            payload = {
                "stage": self.stage,
                "split": self.split,
                "epoch": self.epoch,
                "num_samples": self.num_samples,
                **payload,
            }
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase1Metrics":
        """从 dict 恢复训练期指标。

        使用场景:
            report 或 datastore 读取历史 metrics 时使用。历史 payload
            通常没有 ``correct_actions`` / ``total_actions``，因此默认恢复为 0。
        """

        return cls(
            stage=payload.get("stage"),
            split=payload.get("split"),
            epoch=int(payload["epoch"]) if payload.get("epoch") is not None else None,
            num_samples=int(payload.get("num_samples", 0)),
            total_loss=float(payload.get("total_loss", 0.0)),
            reconstruction_loss=float(payload.get("reconstruction_loss", 0.0)),
            vq_loss=float(payload.get("vq_loss", 0.0)),
            codebook_loss=float(payload.get("codebook_loss", 0.0)),
            commitment_loss=float(payload.get("commitment_loss", 0.0)),
            morphology_loss=float(payload.get("morphology_loss", 0.0)),
            pair_loss=float(payload.get("pair_loss", 0.0)),
            codebook_diversity_loss=float(
                payload.get("codebook_diversity_loss", 0.0)
            ),
            prototype_diversity_loss=float(
                payload.get("prototype_diversity_loss", 0.0)
            ),
            action_accuracy=float(payload.get("action_accuracy", 0.0)),
            correct_actions=int(payload.get("correct_actions", 0)),
            total_actions=int(payload.get("total_actions", 0)),
        )


__all__ = ["Phase1Metrics"]
