"""Phase I codebook validation 五层统一编排入口。

文件功能说明:
    本文件实现 ``Phase1CodebookEvaluator``，负责把 evaluator 收集的 train/val
    snapshot 交给五个独立 layer calculator，随后调用 metrics rules、score 和
    result schema 组装完整 ``Phase1ValidationResult``。

设计边界:
    - 本文件负责编排，不直接实现五层 raw metric 公式；
    - raw metric 计算分别委托给 ``phase1_validation_layers/layer*.py``；
    - pass/fail 判定统一委托给 ``phase1_validation_rules.py``；
    - 综合评分统一委托给 ``phase1_validation_score.py``；
    - 本文件不负责 checkpoint 保存、report 渲染或 checkpoint selection。

使用场景:
    Phase I 每个候选 checkpoint 训练完成后，主流程调用
    ``evaluate_checkpoint()`` 得到可序列化、可审计的完整五层 validation result。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
import torch
from torch.utils.data import DataLoader

from ...model.data_types import HorizonDataset
from ...model.tensor_data_types import (
    TrajectoryTensorBatch,
    move_trajectory_batch_to_device,
)
from ...model.vq_archetype import ArchetypeVQModel
from ..metrics import (
    CodeAssignmentSnapshot,
    Phase1BehaviorQualityMetrics,
    Phase1BehaviorQualityThresholds,
    Phase1EvaluationSnapshot,
    Phase1LabelPredictabilityMetrics,
    Phase1LabelPredictabilityThresholds,
    Phase1OracleProfitabilityMetrics,
    Phase1OracleProfitabilityThresholds,
    Phase1PerCodeProfitability,
    Phase1TeacherQualityMetrics,
    Phase1TeacherQualityThresholds,
    Phase1VQInternalMetrics,
    Phase1ValidationMetrics,
    Phase1ValidationResult,
    Phase1ValidationRuntimeConfig,
    Phase1ValidationScoreWeights,
    Phase1VQInternalThresholds,
    aggregate_validation_result,
    build_tie_breaker_metrics,
    compute_phase1_validation_score,
    evaluate_behavior_quality_rules,
    evaluate_label_predictability_rules,
    evaluate_oracle_profitability_rules,
    evaluate_teacher_quality_rules,
    evaluate_vq_internal_rules,
)
from .phase1_validation_layers import (
    compute_behavior_quality_metrics,
    compute_label_predictability_metrics,
    compute_oracle_profitability_metrics,
    compute_teacher_quality_metrics,
    compute_vq_internal_metrics,
)


class Phase1CodebookEvaluator:
    """完整 Phase I codebook validation 编排器。

    功能说明:
        收集模型在 train/validation split 上的中间数组，调用五层 raw metric
        calculator，执行 hard gate rules，计算综合 score 和 tie-breaker，并返回
        checkpoint 级 ``Phase1ValidationResult``。

    设计边界:
        - 只做数据收集和编排；
        - 不在本类中重新实现任何 layer 的公式；
        - 不做落盘或 HTML/JSON report 渲染；
        - 不替代 ``Phase1CheckpointSelector`` 的候选排序职责。

    使用场景:
        Phase I 训练循环保存 checkpoint 前后调用本类，对当前模型状态做五层
        validation，并把结果写入 checkpoint metrics payload。
    """

    def __init__(
        self,
        model: ArchetypeVQModel,
        *,
        teacher_thresholds: Phase1TeacherQualityThresholds | None = None,
        vq_internal_thresholds: Phase1VQInternalThresholds | None = None,
        behavior_thresholds: Phase1BehaviorQualityThresholds | None = None,
        oracle_profitability_thresholds: Phase1OracleProfitabilityThresholds | None = None,
        label_predictability_thresholds: Phase1LabelPredictabilityThresholds | None = None,
        score_weights: Phase1ValidationScoreWeights | None = None,
        runtime_config: Phase1ValidationRuntimeConfig | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        """初始化 codebook validation evaluator。

        输入参数:
            model: 当前待验证的 ``ArchetypeVQModel``。
            teacher_thresholds: Layer 0 hard gate 阈值配置；不传则使用默认值。
            vq_internal_thresholds: Layer 1 hard gate 阈值配置；不传则使用默认值。
            behavior_thresholds: Layer 2 hard gate 阈值配置；不传则使用默认值。
            oracle_profitability_thresholds: Layer 3 hard gate 阈值配置；不传则使用默认值。
            label_predictability_thresholds: Layer 4 hard gate 阈值配置；不传则使用默认值。
            score_weights: 五层综合评分权重；不传则使用默认值。
            runtime_config: 五层 raw metric 计算运行参数；不传则使用默认值。
            device: 模型推理设备。

        输出:
            无。初始化后可调用 ``collect_snapshot()`` 或 ``evaluate_checkpoint()``。

        使用场景:
            在 Phase I 主流程构建模型后创建一次，后续每个 checkpoint 复用同一个
            evaluator。
        """

        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.teacher_thresholds = teacher_thresholds or Phase1TeacherQualityThresholds()
        self.vq_internal_thresholds = (
            vq_internal_thresholds or Phase1VQInternalThresholds()
        )
        self.behavior_thresholds = behavior_thresholds or Phase1BehaviorQualityThresholds()
        self.oracle_profitability_thresholds = (
            oracle_profitability_thresholds or Phase1OracleProfitabilityThresholds()
        )
        self.label_predictability_thresholds = (
            label_predictability_thresholds or Phase1LabelPredictabilityThresholds()
        )
        self.score_weights = score_weights or Phase1ValidationScoreWeights()
        self.runtime_config = runtime_config or Phase1ValidationRuntimeConfig()

    @torch.no_grad()
    def collect_snapshot(
        self,
        dataloader: DataLoader[TrajectoryTensorBatch],
        *,
        split: str,
        epoch: int,
        horizon_dataset: HorizonDataset | None = None,
    ) -> Phase1EvaluationSnapshot:
        """收集单个 split 的五层 validation 中间数据。

        功能说明:
            遍历 dataloader，调用模型 forward，收集 states、actions、rewards、
            decoded logits/actions、code ids、latent、quantized latent、codebook
            distances、reconstruction loss 和 action accuracy。

        输入参数:
            dataloader: Phase I trajectory dataloader。用于 full validation 时应使用
                ``shuffle=False``，否则 sample_ids 和 horizon_dataset prices 无法稳定对齐。
            split: split 名称，例如 ``train`` 或 ``val``。
            epoch: 当前 checkpoint epoch。
            horizon_dataset: 可选 ``(states, prices)``。传入且长度匹配时读取 prices；
                不传时 snapshot.prices 为 ``None``。

        输出:
            ``Phase1EvaluationSnapshot``，供五个 layer calculator 读取。

        使用场景:
            ``evaluate_checkpoint()`` 内部分别收集 train 和 validation snapshot；
            调试时也可以单独调用本方法检查某个 split 的中间数组。
        """

        self.model.eval()
        state_parts: list[np.ndarray] = []
        action_parts: list[np.ndarray] = []
        reward_parts: list[np.ndarray] = []
        decoded_action_parts: list[np.ndarray] = []
        decoded_logit_parts: list[np.ndarray] = []
        code_id_parts: list[np.ndarray] = []
        z_e_parts: list[np.ndarray] = []
        z_q_parts: list[np.ndarray] = []
        distance_parts: list[np.ndarray] = []
        loss_weighted_sum = 0.0
        correct_actions = 0
        total_actions = 0
        total_samples = 0

        for batch in dataloader:
            batch = move_trajectory_batch_to_device(batch, self.device)
            states, actions, rewards = batch
            outputs = self.model(batch)
            quantize_output = self.model.quantizer.quantize(outputs.z_e)
            decoded_actions = outputs.action_logits.argmax(dim=-1)

            batch_size = int(states.shape[0])
            total_samples += batch_size
            loss_weighted_sum += (
                float(outputs.reconstruction_loss.detach().cpu().item()) * batch_size
            )
            correct_actions += int((decoded_actions == actions.long()).sum().item())
            total_actions += int(actions.numel())

            state_parts.append(states.detach().cpu().numpy())
            action_parts.append(actions.detach().cpu().numpy())
            reward_parts.append(rewards.detach().cpu().numpy())
            decoded_action_parts.append(decoded_actions.detach().cpu().numpy())
            decoded_logit_parts.append(outputs.action_logits.detach().cpu().numpy())
            code_id_parts.append(outputs.code_id.detach().cpu().numpy())
            z_e_parts.append(outputs.z_e.detach().cpu().numpy())
            z_q_parts.append(quantize_output.z_q_no_grad.detach().cpu().numpy())
            distance_parts.append(quantize_output.distances.detach().cpu().numpy())

        if total_samples <= 0:
            raise ValueError("validation dataloader produced no samples")

        prices = self._prices_from_horizon_dataset(
            horizon_dataset=horizon_dataset,
            expected_samples=total_samples,
        )
        sample_ids = np.arange(total_samples, dtype=np.int64)
        reconstruction_loss = loss_weighted_sum / total_samples
        action_accuracy = correct_actions / total_actions if total_actions > 0 else 0.0

        # Shape legend after concatenating all dataloader batches on axis=0:
        # N=total horizon samples, H=horizon length, F=state feature dim,
        # A=action classes, K=codebook size, D=latent/code embedding dim.
        return Phase1EvaluationSnapshot(
            split=split,
            epoch=epoch,
            sample_ids=sample_ids,
            # [N, H, F]
            states=np.concatenate(state_parts, axis=0),
            # None or [N, H]. HorizonDataset [N, H, 1] prices are normalized below.
            prices=prices,
            # [N, H]
            demo_actions=np.concatenate(action_parts, axis=0),
            # [N, H]
            demo_rewards=np.concatenate(reward_parts, axis=0),
            # [N, H]
            decoded_actions=np.concatenate(decoded_action_parts, axis=0),
            # [N, H, A]
            decoded_logits=np.concatenate(decoded_logit_parts, axis=0),
            # [N]
            code_ids=np.concatenate(code_id_parts, axis=0),
            # [N, D]
            z_e=np.concatenate(z_e_parts, axis=0),
            # [N, D]
            z_q=np.concatenate(z_q_parts, axis=0),
            # [N, K]
            distances=np.concatenate(distance_parts, axis=0),
            reconstruction_loss=reconstruction_loss,
            action_accuracy=action_accuracy,
        )

    def evaluate_checkpoint(
        self,
        *,
        train_loader: DataLoader[TrajectoryTensorBatch],
        val_loader: DataLoader[TrajectoryTensorBatch],
        epoch: int,
        checkpoint_id: str,
        stage: str = "vq",
        train_horizon_dataset: HorizonDataset | None = None,
        val_horizon_dataset: HorizonDataset | None = None,
        assignment_history: Sequence[CodeAssignmentSnapshot] = (),
    ) -> Phase1ValidationResult:
        """执行完整五层 checkpoint validation。

        功能说明:
            先收集 train/val snapshot，再按 Layer 0、1、3、2、4 的计算顺序执行
            raw metric calculator，随后按 0 到 4 的展示顺序执行 rules 判定，
            最后计算 score、tie-breaker 并组装 ``Phase1ValidationResult``。

        输入参数:
            train_loader: train split evaluation dataloader，建议 ``shuffle=False``。
            val_loader: validation split dataloader，必须 ``shuffle=False`` 才能和
                horizon prices 对齐。
            epoch: checkpoint 对应 epoch。
            checkpoint_id: checkpoint 稳定 ID 或文件名。
            stage: validation 所属训练阶段，默认 ``vq``。
            train_horizon_dataset: 可选 train horizon dataset，用于提供 prices。
            val_horizon_dataset: 可选 validation horizon dataset，用于提供 prices。
            assignment_history: 历史 assignment snapshots，用于 Layer 1 churn/lifetime。

        输出:
            ``Phase1ValidationResult``，包含 passed、score、failed layers、五层 raw
            metrics、layer rule results、code diagnostics 和 tie-breaker metrics。

        使用场景:
            每个 checkpoint 保存前后调用，将结果序列化进 checkpoint metrics；
            checkpoint selector 和 report 后续只消费该结果，不重新计算五层逻辑。
        """

        train_snapshot = self.collect_snapshot(
            train_loader,
            split="train",
            epoch=epoch,
            horizon_dataset=train_horizon_dataset,
        )
        val_snapshot = self.collect_snapshot(
            val_loader,
            split="val",
            epoch=epoch,
            horizon_dataset=val_horizon_dataset,
        )

        teacher_computation = compute_teacher_quality_metrics(
            train_snapshot=train_snapshot,
            val_snapshot=val_snapshot,
            runtime_config=self.runtime_config,
        )
        vq_computation = compute_vq_internal_metrics(
            train_snapshot=train_snapshot,
            val_snapshot=val_snapshot,
            assignment_history=assignment_history,
            runtime_config=self.runtime_config,
        )
        oracle_computation = compute_oracle_profitability_metrics(
            model=self.model,
            val_snapshot=val_snapshot,
            runtime_config=self.runtime_config,
            device=self.device,
        )
        per_code_profitability = oracle_computation.extra_payload.get(
            "per_code_profitability",
            (),
        )
        behavior_computation = compute_behavior_quality_metrics(
            train_snapshot=train_snapshot,
            val_snapshot=val_snapshot,
            runtime_config=self.runtime_config,
            per_code_profitability=cast(
                Sequence[Phase1PerCodeProfitability],
                per_code_profitability,
            ),
            thresholds=self.behavior_thresholds,
        )
        label_computation = compute_label_predictability_metrics(
            model=self.model,
            train_snapshot=train_snapshot,
            val_snapshot=val_snapshot,
            runtime_config=self.runtime_config,
            device=self.device,
        )

        metrics = Phase1ValidationMetrics(
            teacher_quality=cast(
                Phase1TeacherQualityMetrics,
                teacher_computation.metrics,
            ),
            vq_internal=cast(Phase1VQInternalMetrics, vq_computation.metrics),
            behavior_quality=cast(
                Phase1BehaviorQualityMetrics,
                behavior_computation.metrics,
            ),
            oracle_profitability=cast(
                Phase1OracleProfitabilityMetrics,
                oracle_computation.metrics,
            ),
            label_predictability=cast(
                Phase1LabelPredictabilityMetrics,
                label_computation.metrics,
            ),
        )
        layers = (
            evaluate_teacher_quality_rules(
                metrics.teacher_quality,
                self.teacher_thresholds,
            ),
            evaluate_vq_internal_rules(
                metrics.vq_internal,
                self.vq_internal_thresholds,
            ),
            evaluate_behavior_quality_rules(
                metrics.behavior_quality,
                self.behavior_thresholds,
            ),
            evaluate_oracle_profitability_rules(
                metrics.oracle_profitability,
                self.oracle_profitability_thresholds,
            ),
            evaluate_label_predictability_rules(
                metrics.label_predictability,
                self.label_predictability_thresholds,
            ),
        )
        passed = all(layer.passed for layer in layers)
        score = (
            compute_phase1_validation_score(metrics, self.score_weights)
            if passed
            else None
        )
        tie_breaker_metrics = build_tie_breaker_metrics(
            metrics,
            reconstruction_loss=val_snapshot.reconstruction_loss,
        )

        return aggregate_validation_result(
            checkpoint_id=checkpoint_id,
            stage=stage,
            epoch=epoch,
            layers=layers,
            metrics=metrics,
            code_diagnostics=behavior_computation.code_diagnostics,
            drift_diagnostics={},
            score=score,
            tie_breaker_metrics=tie_breaker_metrics,
        )

    @staticmethod
    def _prices_from_horizon_dataset(
        *,
        horizon_dataset: HorizonDataset | None,
        expected_samples: int,
    ) -> np.ndarray | None:
        """从可选 horizon dataset 中提取 prices。

        输入参数:
            horizon_dataset: ``(states, prices)`` 或 ``None``。
            expected_samples: 当前 dataloader 实际收集到的样本数。

        输出:
            prices 数组，shape=[N, H]；未提供 horizon dataset 或样本数不匹配时返回 ``None``。

        使用场景:
            ``collect_snapshot()`` 将 dataloader 中的 trajectory 与外部 horizon prices
            对齐。调用方应保证 dataloader 未 shuffle，且 horizon dataset 与 trajectory
            dataset 顺序一致。
        """

        if horizon_dataset is None:
            return None
        _, prices = horizon_dataset
        price_values = np.asarray(prices)
        if price_values.shape[0] != expected_samples:
            return None
        if price_values.ndim == 3 and price_values.shape[-1] == 1:
            return price_values[..., 0]
        if price_values.ndim != 2:
            raise ValueError(
                "horizon_dataset prices must have shape [N, H] or [N, H, 1], "
                f"got {price_values.shape}"
            )
        return price_values


__all__ = ["Phase1CodebookEvaluator"]
