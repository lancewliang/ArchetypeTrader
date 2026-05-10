"""Phase I training and evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .vq_archetype import VqModelOutputs


@dataclass
class Phase1Metrics:
    """Phase I 训练或评估过程中产生的指标数据。

    功能描述:
        保存一次 Phase I 训练/验证/测试指标。训练循环可以先用
        ``add_batch`` 直接接收模型返回的 ``VqModelOutputs`` 并按 batch size
        累积加权 loss，再调用 ``averaged`` 得到 epoch-level 均值指标；写
        checkpoint 或 report 时使用 ``to_dict`` 转为普通字典，避免存储层
        依赖 dataclass 实例。

    论文描述:
        Phase I 的核心训练目标由动作重构损失和 VQ 损失组成:
        ``L = L_rec + L_codebook + beta_0 L_commitment``。这些字段记录每个
        split/epoch 上该目标的分解结果，便于判断 reconstruction 是否稳定、
        codebook 是否收敛，以及不同 archetype 训练批次是否可复现。
    """

    stage: str | None = None
    split: str | None = None
    epoch: int | None = None
    sample_count: int = 0
    total_loss: float = 0.0
    reconstruction_loss: float = 0.0
    vq_loss: float = 0.0
    codebook_loss: float = 0.0
    commitment_loss: float = 0.0
    extra_metrics: dict[str, float] = field(default_factory=dict)

    def add_batch(
        self,
        *,
        batch_size: int,
        outputs: VqModelOutputs,
        extra_metrics: dict[str, float] | None = None,
    ) -> None:
        """按 batch size 累积一批模型输出指标。"""

        if batch_size < 0:
            raise ValueError("batch_size must be non-negative")

        self.sample_count += batch_size
        self.total_loss += self._loss_value(outputs.total_loss) * batch_size
        self.reconstruction_loss += (
            self._loss_value(outputs.reconstruction_loss) * batch_size
        )
        self.vq_loss += self._loss_value(outputs.vq_loss) * batch_size
        self.codebook_loss += self._loss_value(outputs.codebook_loss) * batch_size
        self.commitment_loss += (
            self._loss_value(outputs.commitment_loss) * batch_size
        )

        if extra_metrics:
            for name, value in extra_metrics.items():
                self.extra_metrics[name] = (
                    self.extra_metrics.get(name, 0.0) + value * batch_size
                )

    @staticmethod
    def _loss_value(loss: Any) -> float:
        return float(loss.detach().cpu())

    def averaged(self) -> Phase1Metrics:
        """返回 sample mean 形式的指标副本。"""

        denominator = max(float(self.sample_count), 1.0)
        return Phase1Metrics(
            stage=self.stage,
            split=self.split,
            epoch=self.epoch,
            sample_count=self.sample_count,
            total_loss=self.total_loss / denominator,
            reconstruction_loss=self.reconstruction_loss / denominator,
            vq_loss=self.vq_loss / denominator,
            codebook_loss=self.codebook_loss / denominator,
            commitment_loss=self.commitment_loss / denominator,
            extra_metrics={
                name: value / denominator
                for name, value in self.extra_metrics.items()
            },
        )

    def to_dict(self, *, include_context: bool = False) -> dict[str, Any]:
        """转换为 checkpoint/report 友好的普通字典。"""

        metrics: dict[str, Any] = {
            "sample_count": self.sample_count,
            "total_loss": self.total_loss,
            "reconstruction_loss": self.reconstruction_loss,
            "vq_loss": self.vq_loss,
            "codebook_loss": self.codebook_loss,
            "commitment_loss": self.commitment_loss,
        }
        metrics.update(self.extra_metrics)

        if include_context:
            if self.stage is not None:
                metrics["stage"] = self.stage
            if self.split is not None:
                metrics["split"] = self.split
            if self.epoch is not None:
                metrics["epoch"] = self.epoch
        return metrics

    @classmethod
    def from_dict(cls, metrics: dict[str, Any]) -> Phase1Metrics:
        """从普通字典恢复 ``Phase1Metrics``。"""

        known_fields = {
            "stage",
            "split",
            "epoch",
            "sample_count",
            "total_loss",
            "reconstruction_loss",
            "vq_loss",
            "codebook_loss",
            "commitment_loss",
        }
        extra_metrics = {
            name: float(value)
            for name, value in metrics.items()
            if name not in known_fields
        }
        return cls(
            stage=metrics.get("stage"),
            split=metrics.get("split"),
            epoch=metrics.get("epoch"),
            sample_count=int(metrics.get("sample_count", 0)),
            total_loss=float(metrics.get("total_loss", 0.0)),
            reconstruction_loss=float(metrics.get("reconstruction_loss", 0.0)),
            vq_loss=float(metrics.get("vq_loss", 0.0)),
            codebook_loss=float(metrics.get("codebook_loss", 0.0)),
            commitment_loss=float(metrics.get("commitment_loss", 0.0)),
            extra_metrics=extra_metrics,
        )
