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
    Phase1LayerComputation,
    Phase1OracleProfitabilityMetrics,
    Phase1OracleProfitabilityThresholds,
    Phase1PerCodeProfitability,
    Phase1MetricResult,
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
from .phase1_validation_layers.layer2_behavior_quality import (
    classify_action_motif,
    classify_market_morphology,
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
        self.last_assignment_snapshot: CodeAssignmentSnapshot | None = None

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
        sample_id_parts: list[np.ndarray] = []
        loss_weighted_sum = 0.0
        correct_actions = 0
        total_actions = 0
        total_samples = 0

        for batch in dataloader:
            raw_batch = batch
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
            if len(raw_batch) >= 4:
                sample_id_parts.append(raw_batch[3].detach().cpu().numpy())

        if total_samples <= 0:
            raise ValueError("validation dataloader produced no samples")

        states_array = np.concatenate(state_parts, axis=0)
        if sample_id_parts:
            sample_ids = np.concatenate(sample_id_parts, axis=0).astype(np.int64)
            if sample_ids.shape != (total_samples,):
                raise ValueError(
                    "sample_ids must have shape [N], "
                    f"got {sample_ids.shape} for {total_samples} samples"
                )
        else:
            sample_ids = np.arange(total_samples, dtype=np.int64)
        prices = self._prices_from_horizon_dataset(
            horizon_dataset=horizon_dataset,
            expected_samples=total_samples,
            sample_ids=sample_ids,
            collected_states=states_array,
        )
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
            states=states_array,
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
        current_assignment = vq_computation.extra_payload.get("current_assignment")
        self.last_assignment_snapshot = (
            current_assignment
            if isinstance(current_assignment, CodeAssignmentSnapshot)
            else None
        )
        oracle_computation = compute_oracle_profitability_metrics(
            model=self.model,
            val_snapshot=val_snapshot,
            runtime_config=self.runtime_config,
            device=self.device,
            thresholds=self.oracle_profitability_thresholds,
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
        drift_diagnostics = self._build_drift_diagnostics(
            train_snapshot=train_snapshot,
            val_snapshot=val_snapshot,
            val_oracle_computation=oracle_computation,
            label_computation=label_computation,
        )

        return aggregate_validation_result(
            checkpoint_id=checkpoint_id,
            stage=stage,
            epoch=epoch,
            layers=layers,
            metrics=metrics,
            code_diagnostics=behavior_computation.code_diagnostics,
            drift_diagnostics=drift_diagnostics,
            score=score,
            tie_breaker_metrics=tie_breaker_metrics,
        )

    def _build_drift_diagnostics(
        self,
        *,
        train_snapshot: Phase1EvaluationSnapshot,
        val_snapshot: Phase1EvaluationSnapshot,
        val_oracle_computation: Phase1LayerComputation,
        label_computation: Phase1LayerComputation,
    ) -> dict[str, Phase1MetricResult]:
        """计算 train/validation 横向 drift diagnostics。

        功能说明:
            汇总 train 与 validation 之间的分布漂移诊断，包括 market morphology
            分布 KL、code usage KL、action motif 分布 KL、reconstruction
            generalization gap、label predictability gap 以及 per-code return gap。
            返回值统一使用 ``Phase1MetricResult``，便于 report 和审计页面展示。

        设计边界:
            - 只产生解释性 warning，不参与五层 hard gate；
            - 触发阈值时 ``severity="warn"``，但 ``passed`` 始终保持 True；
            - 不重新定义五层 raw metric 或 selector 规则；
            - 依赖已有 snapshot 和 layer extra payload，不访问文件系统。

        使用场景:
            ``evaluate_checkpoint()`` 在五层 validation 完成后调用本方法，为
            checkpoint report 补充 train/validation 分布是否一致的解释信息。
        """

        diagnostics: dict[str, Phase1MetricResult] = {}

        train_morphology = classify_market_morphology(
            train_snapshot.prices,
            fee_rate=self.runtime_config.fee_rate,
        )
        val_morphology = classify_market_morphology(
            val_snapshot.prices,
            fee_rate=self.runtime_config.fee_rate,
        )
        diagnostics["morphology_distribution_kl"] = self._drift_kl_metric(
            name="morphology_distribution_kl",
            train_values=train_morphology,
            val_values=val_morphology,
            threshold_value=0.20,
            message="validation 市场形态分布不应明显偏离 train",
        )

        diagnostics["code_usage_kl"] = self._drift_kl_metric(
            name="code_usage_kl",
            train_values=train_snapshot.code_ids,
            val_values=val_snapshot.code_ids,
            threshold_value=0.20,
            message="validation code usage 不应明显偏离 train",
        )

        train_motifs = classify_action_motif(
            train_snapshot.decoded_actions,
            train_snapshot.prices,
        )
        val_motifs = classify_action_motif(
            val_snapshot.decoded_actions,
            val_snapshot.prices,
        )
        diagnostics["motif_distribution_kl"] = self._drift_kl_metric(
            name="motif_distribution_kl",
            train_values=train_motifs,
            val_values=val_motifs,
            threshold_value=0.20,
            message="validation motif 分布不应明显偏离 train",
        )

        reconstruction_gap = val_snapshot.reconstruction_loss / (
            train_snapshot.reconstruction_loss + 1e-12
        )
        diagnostics["reconstruction_generalization_gap"] = self._drift_upper_metric(
            name="reconstruction_generalization_gap",
            value=float(reconstruction_gap),
            threshold_value=1.25,
            message="validation/train reconstruction loss gap 过大时提示重构泛化风险",
        )

        predictability_gap = label_computation.extra_payload.get(
            "probe_predictability_gap"
        )
        diagnostics["label_predictability_gap"] = self._drift_upper_metric(
            name="label_predictability_gap",
            value=(
                float(predictability_gap)
                if isinstance(predictability_gap, (int, float))
                else None
            ),
            threshold_value=0.15,
            message="probe train/validation accuracy gap 过大时提示 selector 可学习性过拟合",
        )

        diagnostics["per_code_return_gap"] = self._per_code_return_gap_metric(
            train_snapshot=train_snapshot,
            val_oracle_computation=val_oracle_computation,
        )
        return diagnostics

    @staticmethod
    def _drift_result(
        *,
        name: str,
        value: float | None,
        threshold: str,
        triggered: bool,
        message: str,
    ) -> Phase1MetricResult:
        """构造单个 drift diagnostic metric result。

        功能说明:
            把 drift 诊断的数值、阈值、触发状态和说明文本封装成
            ``Phase1MetricResult``。输入缺失或非有限值时，统一转成
            ``severity="skip"`` 的可展示结果。

        设计边界:
            - 只负责结果对象标准化，不计算具体 drift 数值；
            - drift diagnostic 不作为 hard gate，返回对象的 ``passed`` 固定为 True；
            - 不修改调用方传入的 metric 名称、阈值文本和业务说明。

        使用场景:
            被 ``_drift_upper_metric()`` 和 ``_per_code_return_gap_metric()`` 复用，
            保证 drift 类诊断在 report 中有一致的 pass/warn/skip 表达。
        """

        if value is None or not np.isfinite(value):
            return Phase1MetricResult(
                name=name,
                value=None,
                threshold=threshold,
                severity="skip",
                passed=True,
                layer="drift",
                message=f"{message}；诊断输入缺失，跳过 drift 判定",
            )
        return Phase1MetricResult(
            name=name,
            value=float(value),
            threshold=threshold,
            severity="warn" if triggered else "pass",
            passed=True,
            layer="drift",
            message=message,
        )

    @classmethod
    def _drift_upper_metric(
        cls,
        *,
        name: str,
        value: float | None,
        threshold_value: float,
        message: str,
    ) -> Phase1MetricResult:
        """构造越低越好的 drift warning metric。

        功能说明:
            判断 ``value`` 是否超过 ``threshold_value``，并把超过阈值的情况标记为
            drift warning。适用于 KL、generalization gap、predictability gap 等
            数值越小越稳定的诊断项。

        设计边界:
            - 只处理 ``value > threshold`` 这一类上界阈值；
            - 不负责计算 ``value`` 本身；
            - 不改变 drift diagnostic 不参与 hard gate 的语义。

        使用场景:
            ``_build_drift_diagnostics()`` 中构造 reconstruction gap、
            label predictability gap 等上界型 warning 指标。
        """

        triggered = (
            value is not None and np.isfinite(value) and value > threshold_value
        )
        return cls._drift_result(
            name=name,
            value=value,
            threshold=f"warn if > {threshold_value:g}",
            triggered=bool(triggered),
            message=message,
        )

    @classmethod
    def _drift_kl_metric(
        cls,
        *,
        name: str,
        train_values: np.ndarray,
        val_values: np.ndarray,
        threshold_value: float,
        message: str,
    ) -> Phase1MetricResult:
        """构造 KL(P_val || P_train) drift warning metric。

        功能说明:
            将 train/validation 的离散标签数组转换为 KL 散度诊断项，度量
            validation 分布相对 train 分布的偏移程度。KL 值超过阈值时写入
            warning。

        设计边界:
            - 只适用于离散类别值，例如 morphology、code id、motif；
            - 空输入不强行计算，交给 ``_drift_upper_metric()`` 生成 skip 结果；
            - 只比较整体分布，不解释具体哪个类别贡献了漂移。

        使用场景:
            ``_build_drift_diagnostics()`` 中用于 morphology distribution、
            code usage 和 motif distribution 的 train/validation 分布对比。
        """

        if train_values.size == 0 or val_values.size == 0:
            value = None
        else:
            value = cls._categorical_kl(train_values, val_values)
        return cls._drift_upper_metric(
            name=name,
            value=value,
            threshold_value=threshold_value,
            message=message,
        )

    @staticmethod
    def _categorical_kl(train_values: np.ndarray, val_values: np.ndarray) -> float:
        """计算离散类别分布的 KL(P_val || P_train)。

        功能说明:
            将任意可转字符串的类别值展开为一维标签，统计 train 和 validation
            的类别频率分布，然后计算 ``sum(P_val * log(P_val / P_train))``。
            计算时加入极小 epsilon，避免类别只出现在一侧时产生除零。

        设计边界:
            - 只处理 categorical/discrete 分布，不适合连续数值特征；
            - 不做阈值判定，也不返回 ``Phase1MetricResult``；
            - 类别统一转为字符串后比较，保证 int、str 等标签可以进入同一流程。

        使用场景:
            作为 ``_drift_kl_metric()`` 的底层数值函数，供 drift report 展示
            train/validation 类别分布偏移程度。
        """

        train_flat = np.asarray(train_values).reshape(-1)
        val_flat = np.asarray(val_values).reshape(-1)
        labels = sorted(
            set(str(value) for value in train_flat)
            | set(str(value) for value in val_flat)
        )
        if not labels:
            return float("nan")
        train_counts = np.asarray(
            [np.sum(train_flat.astype(str) == label) for label in labels],
            dtype=np.float64,
        )
        val_counts = np.asarray(
            [np.sum(val_flat.astype(str) == label) for label in labels],
            dtype=np.float64,
        )
        train_prob = train_counts / max(1.0, float(np.sum(train_counts)))
        val_prob = val_counts / max(1.0, float(np.sum(val_counts)))
        eps = 1e-12
        return float(np.sum(val_prob * np.log((val_prob + eps) / (train_prob + eps))))

    def _per_code_return_gap_metric(
        self,
        *,
        train_snapshot: Phase1EvaluationSnapshot,
        val_oracle_computation: Phase1LayerComputation,
    ) -> Phase1MetricResult:
        """计算 per-code train/validation return gap drift warning。

        功能说明:
            对 train snapshot 重新执行 oracle profitability 计算，并与 validation
            oracle extra payload 中的 per-code mean advantage 对齐，统计共同 code
            的平均收益差距。差距超过 train per-code return 标准差时标记 warning。

        设计边界:
            - 只作为局部 code 稳定性的 drift 诊断，不参与 checkpoint 淘汰；
            - 需要 train prices 才能重算 train oracle profitability，缺失时返回 skip；
            - 只比较 train/validation 都存在且 mean advantage 有效的 code；
            - 不替代 Layer 3 的 oracle profitability hard gate。

        使用场景:
            ``_build_drift_diagnostics()`` 在 Layer 3 已完成后调用，用于 report
            提示某些 code 可能在 validation 上出现局部收益失效。
        """

        if train_snapshot.prices is None:
            return self._drift_result(
                name="per_code_return_gap",
                value=None,
                threshold="warn if > train per-code return std",
                triggered=False,
                message="per-code return train/validation gap 需要 prices 才能诊断",
            )

        train_oracle = compute_oracle_profitability_metrics(
            model=self.model,
            val_snapshot=train_snapshot,
            runtime_config=self.runtime_config,
            device=self.device,
            thresholds=self.oracle_profitability_thresholds,
        )
        train_items = train_oracle.extra_payload.get("per_code_profitability", ())
        val_items = val_oracle_computation.extra_payload.get(
            "per_code_profitability",
            (),
        )
        train_map = {
            item.code_id: item.mean_advantage
            for item in train_items
            if isinstance(item, Phase1PerCodeProfitability)
            and np.isfinite(item.mean_advantage)
        }
        val_map = {
            item.code_id: item.mean_advantage
            for item in val_items
            if isinstance(item, Phase1PerCodeProfitability)
            and np.isfinite(item.mean_advantage)
        }
        common_codes = sorted(set(train_map) & set(val_map))
        if not common_codes:
            return self._drift_result(
                name="per_code_return_gap",
                value=None,
                threshold="warn if > train per-code return std",
                triggered=False,
                message="没有可对齐的 train/validation per-code return",
            )

        gaps = np.asarray(
            [abs(val_map[code_id] - train_map[code_id]) for code_id in common_codes],
            dtype=np.float64,
        )
        train_returns = np.asarray(list(train_map.values()), dtype=np.float64)
        threshold_value = float(np.std(train_returns))
        value = float(np.mean(gaps))
        return self._drift_result(
            name="per_code_return_gap",
            value=value,
            threshold=f"warn if > {threshold_value:g}",
            triggered=value > threshold_value,
            message="per-code return train/validation gap 过大时提示 code 局部失效",
        )

    @staticmethod
    def _prices_from_horizon_dataset(
        *,
        horizon_dataset: HorizonDataset | None,
        expected_samples: int,
        sample_ids: np.ndarray,
        collected_states: np.ndarray,
    ) -> np.ndarray | None:
        """从可选 horizon dataset 中提取 prices。

        功能说明:
            根据 dataloader 收集到的 ``sample_ids`` 从 ``horizon_dataset`` 中取回
            对应 prices，并用 states 做严格对齐校验，防止 shuffled dataloader 或
            dataset 顺序不一致导致 price/action/reward 错配。

        设计边界:
            - 只负责 prices 提取和对齐校验，不修正错位数据；
            - 未提供 ``horizon_dataset`` 时返回 ``None``，由下游 layer 按缺失价格处理；
            - 价格只接受 ``[N, H]`` 或 ``[N, H, 1]``，输出统一为 ``[N, H]``；
            - 对齐失败直接抛出 ``ValueError``，避免静默生成不可审计的 metrics。

        输入参数:
            horizon_dataset: ``(states, prices)`` 或 ``None``。
            expected_samples: 当前 dataloader 实际收集到的样本数。
            sample_ids: 当前 dataloader batch 携带的稳定样本 ID。
            collected_states: dataloader 实际遍历得到的 states，用于校验 sample_id
                与 horizon dataset 的顺序是否一致。

        输出:
            prices 数组，shape=[N, H]；未提供 horizon dataset 或样本数不匹配时返回 ``None``。

        使用场景:
            ``collect_snapshot()`` 将 dataloader 中的 trajectory 与外部 horizon prices
            对齐。调用方应保证 dataloader 未 shuffle，且 horizon dataset 与 trajectory
            dataset 顺序一致。
        """

        if horizon_dataset is None:
            return None
        horizon_states, prices = horizon_dataset
        horizon_state_values = np.asarray(horizon_states)
        price_values = np.asarray(prices)
        if sample_ids.shape != (expected_samples,):
            raise ValueError(
                "sample_ids must match collected sample count, "
                f"got {sample_ids.shape} and {expected_samples}"
            )
        if np.any(sample_ids < 0) or np.any(sample_ids >= price_values.shape[0]):
            raise ValueError("sample_ids are outside horizon_dataset bounds")
        if horizon_state_values.shape[0] <= int(np.max(sample_ids, initial=-1)):
            raise ValueError("sample_ids are outside horizon_dataset state bounds")

        selected_states = horizon_state_values[sample_ids]
        if selected_states.shape != collected_states.shape:
            raise ValueError(
                "horizon_dataset states and collected states are not shape-aligned, "
                f"got {selected_states.shape} and {collected_states.shape}"
            )
        if not np.allclose(selected_states, collected_states, rtol=1e-5, atol=1e-6):
            raise ValueError(
                "horizon_dataset states do not align with dataloader sample_ids; "
                "use an eval dataloader with stable sample ids and shuffle=False"
            )

        price_values = price_values[sample_ids]
        if price_values.ndim == 3 and price_values.shape[-1] == 1:
            return price_values[..., 0]
        if price_values.ndim != 2:
            raise ValueError(
                "horizon_dataset prices must have shape [N, H] or [N, H, 1], "
                f"got {price_values.shape}"
            )
        return price_values


__all__ = ["Phase1CodebookEvaluator"]
