"""Phase II 训练期基础指标聚合器。

本文件只负责 Double DQN 训练循环中的轻量指标统计，不承载 validation/test
selection metrics。validation/test 的可排序结果由 ``phase2_metric_results.py``
定义。

使用场景:
    ``Phase2DoubleDqnTrainer.train_one_epoch()`` 在每次 Q-network update 后调用
    ``add_batch()``，最后调用 ``averaged()`` 得到 epoch 级训练指标。

设计约束:
    - loss 按 replay sample 加权累加，避免不同 batch 大小影响平均值；
    - Q value、TD target、reward、grad norm 等诊断按 update batch 加权统计；
    - 本类只依赖 ``Phase2DoubleDqnLossOutput``，不访问 model、DataLoader 或文件系统。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

import torch

if TYPE_CHECKING:
    from ..rl.phase2_double_dqn_loss import Phase2DoubleDqnLossOutput


def _tensor_to_float(value: torch.Tensor | float | int | None) -> float:
    """把标量 Tensor 或 Python number 转为 float。"""

    if value is None:
        return 0.0
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("metric tensor must be scalar")
        return float(value.detach().cpu().item())
    return float(value)


@dataclass
class Phase2Metrics:
    """Phase II Double DQN 训练期基础指标。

    功能说明:
        聚合一个 epoch 内的 Q-network update 指标，包括总 loss、TD loss、
        imitation loss、Q/target/reward 诊断和梯度范数。

    使用场景:
        trainer 创建空对象后反复调用 ``add_batch()``，epoch 结束后调用
        ``averaged()``；日志、checkpoint metadata 或 artifact store 保存时调用
        ``to_dict()``。
    """

    # 训练阶段名称，例如 "train" 或 "eval"。用于日志和 report 分组。
    stage: str | None = None

    # 数据 split 名称，例如 "train"、"val" 或 "test"。用于区分指标来源。
    split: str | None = None

    # 当前 epoch 编号。用于把训练指标和 checkpoint 对齐。
    epoch: int | None = None

    # 已累计 replay sample 数。用途：加权平均和审计训练覆盖范围；方向：
    # 诊断字段，覆盖越充分越可靠，但不直接作为好坏排序。
    num_samples: int = 0

    # 已执行 Q-network update 次数。用途：确认训练进度和日志完整性；方向：
    # 诊断字段，不直接作为好坏排序。
    num_updates: int = 0

    # Double DQN + imitation regularization 总 loss。含义：TD loss 与 imitation
    # loss 的加权和；用途：观察训练是否收敛；方向：通常越小越好，但需结合
    # validation return，不能单独选择 checkpoint。
    total_loss: float = 0.0

    # TD Huber loss。含义：Q value 对 Double DQN bootstrap target 的拟合误差；
    # 用途：诊断 value learning 是否收敛；方向：通常越小越好，异常升高表示
    # Q 估计不稳或 reward 分布变化。
    td_loss: float = 0.0

    # assigned-label imitation cross entropy/KL loss。含义：selector policy 与
    # Phase I assigned label 先验的距离；用途：诊断 imitation regularization 是否
    # 生效；方向：越小表示越贴近 assigned label，但过小可能退化为只复制 label。
    imitation_loss: float = 0.0

    # 当前 action 对应的 Q value 均值。用途：监控 Q 尺度和 overestimation；
    # 方向：诊断字段，不是越大越好；异常过大通常表示估值发散风险。
    selected_q_mean: float = 0.0

    # Double DQN bootstrap target 均值。用途：对照 selected_q_mean 检查 target
    # 尺度；方向：诊断字段，不直接作为好坏排序。
    td_target_mean: float = 0.0

    # replay batch reward 均值。用途：观察训练采样到的即时 reward 水平；方向：
    # 越大通常越好，但受 replay 分布和 reward clipping 影响，只作训练诊断。
    reward_mean: float = 0.0

    # next state 上 online network greedy action 的均值。用途：粗略观察 action/code
    # 选择是否偏向高编号或低编号；方向：无好坏方向，仅作为行为诊断。
    greedy_next_action_mean: float = 0.0

    # 梯度裁剪前的梯度范数均值。用途：诊断训练稳定性；方向：越小不一定越好，
    # 但异常过大表示梯度爆炸风险，长期接近 0 表示学习停滞风险。
    grad_norm: float = 0.0

    def add_batch(
        self,
        batch_size: int,
        outputs: "Phase2DoubleDqnLossOutput",
        actions: torch.Tensor | None = None,
    ) -> None:
        """累加一个 replay update batch 的基础指标。

        参数:
            batch_size:
                当前 replay batch 的样本数，用于对 batch-level 指标做 sample 加权。
            outputs:
                ``compute_double_dqn_loss()`` 产生的 loss 和诊断输出。
            actions:
                当前 update batch 的 selector action。保留该参数是为了和 Phase I
                metrics 的调用风格一致；Phase II 当前不从 action 额外计算指标。
        """

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if actions is not None and actions.numel() != batch_size:
            raise ValueError(
                "actions must contain exactly batch_size elements, "
                f"got {actions.numel()} and {batch_size}"
            )

        self.num_samples += int(batch_size)
        self.num_updates += 1
        self.total_loss += _tensor_to_float(outputs.total_loss) * batch_size
        self.td_loss += _tensor_to_float(outputs.td_loss) * batch_size
        self.imitation_loss += _tensor_to_float(outputs.imitation_loss) * batch_size
        self.selected_q_mean += float(outputs.selected_q_mean) * batch_size
        self.td_target_mean += float(outputs.td_target_mean) * batch_size
        self.reward_mean += float(outputs.reward_mean) * batch_size
        self.greedy_next_action_mean += float(outputs.greedy_next_action_mean) * batch_size
        self.grad_norm += _tensor_to_float(outputs.grad_norm) * batch_size

    def averaged(self) -> "Phase2Metrics":
        """返回样本平均后的指标对象，原累加器保持不变。"""

        if self.num_samples <= 0:
            return Phase2Metrics(
                stage=self.stage,
                split=self.split,
                epoch=self.epoch,
                num_samples=0,
                num_updates=self.num_updates,
            )

        return Phase2Metrics(
            stage=self.stage,
            split=self.split,
            epoch=self.epoch,
            num_samples=self.num_samples,
            num_updates=self.num_updates,
            total_loss=self.total_loss / self.num_samples,
            td_loss=self.td_loss / self.num_samples,
            imitation_loss=self.imitation_loss / self.num_samples,
            selected_q_mean=self.selected_q_mean / self.num_samples,
            td_target_mean=self.td_target_mean / self.num_samples,
            reward_mean=self.reward_mean / self.num_samples,
            greedy_next_action_mean=self.greedy_next_action_mean / self.num_samples,
            grad_norm=self.grad_norm / self.num_samples,
        )

    def to_dict(self, *, include_context: bool = True) -> dict[str, Any]:
        """序列化为普通 dict。"""

        payload: dict[str, Any] = {
            "total_loss": self.total_loss,
            "td_loss": self.td_loss,
            "imitation_loss": self.imitation_loss,
            "selected_q_mean": self.selected_q_mean,
            "td_target_mean": self.td_target_mean,
            "reward_mean": self.reward_mean,
            "greedy_next_action_mean": self.greedy_next_action_mean,
            "grad_norm": self.grad_norm,
        }
        if include_context:
            payload = {
                "stage": self.stage,
                "split": self.split,
                "epoch": self.epoch,
                "num_samples": self.num_samples,
                "num_updates": self.num_updates,
                **payload,
            }
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase2Metrics":
        """从 dict 恢复训练期基础指标。"""

        return cls(
            stage=payload.get("stage"),
            split=payload.get("split"),
            epoch=int(payload["epoch"]) if payload.get("epoch") is not None else None,
            num_samples=int(payload.get("num_samples", 0)),
            num_updates=int(payload.get("num_updates", 0)),
            total_loss=float(payload.get("total_loss", 0.0)),
            td_loss=float(payload.get("td_loss", 0.0)),
            imitation_loss=float(payload.get("imitation_loss", 0.0)),
            selected_q_mean=float(payload.get("selected_q_mean", 0.0)),
            td_target_mean=float(payload.get("td_target_mean", 0.0)),
            reward_mean=float(payload.get("reward_mean", 0.0)),
            greedy_next_action_mean=float(
                payload.get("greedy_next_action_mean", 0.0)
            ),
            grad_norm=float(payload.get("grad_norm", 0.0)),
        )


__all__ = ["Phase2Metrics"]
