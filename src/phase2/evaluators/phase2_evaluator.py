"""Phase II selector 统一评估器。

本文件负责把 selector checkpoint 在 validation/test split 上评估成统一的
``Phase2ValidationResult``。Evaluator 只编排推理、rollout、raw metrics 计算和
rule 判定；不训练模型、不保存产物、不选择 best checkpoint。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch

from ..model.phase2_decoder_policy import FrozenArchetypeDecoderPolicy
from ..model.phase2_q_network import Phase2QNetwork
from ..phase2_batch_env import ArchetypeSelectionBatchEnv
from ..phase2_config import Phase2RewardConfig
from ..phase2_selection_dataset import Phase2SelectionDataset
from ...utils import ActionExecutionCalculator
from ..metrics import (
    Phase2BaselineUpliftPayload,
    Phase2BaselineUpliftThresholds,
    Phase2CodeUsageCollapsePayload,
    Phase2CodeUsageCollapseThresholds,
    Phase2DemonstrationConsistencyPayload,
    Phase2DemonstrationConsistencyThresholds,
    Phase2EvaluationValidityPayload,
    Phase2EvaluationValidityThresholds,
    Phase2GeneralizationStabilityPayload,
    Phase2GeneralizationStabilityThresholds,
    Phase2LayerComputation,
    Phase2LayerResult,
    Phase2SelectorProfitabilityPayload,
    Phase2SelectorProfitabilityThresholds,
    Phase2ValidationMetrics,
    Phase2ValidationPayloads,
    Phase2ValidationResult,
    evaluate_baseline_uplift_rules,
    evaluate_code_usage_collapse_rules,
    evaluate_demonstration_consistency_rules,
    evaluate_evaluation_validity_rules,
    evaluate_generalization_stability_rules,
    evaluate_selector_profitability_rules,
)
from .phase2_validation_layers import (
    build_phase2_code_diagnostics,
    build_selector_pair_profitability_matrix,
    compute_baseline_uplift_metrics,
    compute_code_usage_collapse_metrics,
    compute_demonstration_consistency_metrics,
    compute_evaluation_validity_metrics,
    compute_generalization_stability_metrics,
    compute_selector_profitability_metrics,
)
from .phase2_validation_layers.layer4_code_usage_collapse import (
    build_per_code_usage_diagnostics,
)


@dataclass(frozen=True)
class _RolloutMatrices:
    """每个样本、每个 code 的 rollout 结果矩阵。"""

    returns: np.ndarray
    gross_returns: np.ndarray
    fees: np.ndarray
    turnover: np.ndarray
    actions: np.ndarray
    failed_mask: np.ndarray


class Phase2Evaluator:
    """Phase II archetype selector 的统一评估入口。"""

    def __init__(
        self,
        reward_config: Phase2RewardConfig,
        device: torch.device | str,
        *,
        q_network: Phase2QNetwork | None = None,
        decoder_policy: FrozenArchetypeDecoderPolicy | None = None,
        rollout_batch_size: int = 512,
        random_seed: int = 42,
        evaluation_validity_thresholds: Phase2EvaluationValidityThresholds | None = None,
        selector_profitability_thresholds: (
            Phase2SelectorProfitabilityThresholds | None
        ) = None,
        baseline_uplift_thresholds: Phase2BaselineUpliftThresholds | None = None,
        demonstration_consistency_thresholds: (
            Phase2DemonstrationConsistencyThresholds | None
        ) = None,
        code_usage_collapse_thresholds: Phase2CodeUsageCollapseThresholds | None = None,
        generalization_stability_thresholds: (
            Phase2GeneralizationStabilityThresholds | None
        ) = None,
        validation_score_history: tuple[float, ...] = (),
        selected_action_churn_history: tuple[float, ...] = (),
        td_loss_history: tuple[float, ...] = (),
        imitation_loss_history: tuple[float, ...] = (),
        reward_mean_history: tuple[float, ...] = (),
        train_mean_return: float | None = None,
        test_mean_return: float | None = None,
        train_usage_distribution: tuple[float, ...] | None = None,
    ) -> None:
        """初始化 evaluator 依赖和阈值配置。"""

        self.reward_config = reward_config
        self.device = torch.device(device)
        self.q_network = q_network
        self.decoder_policy = decoder_policy
        self.rollout_batch_size = self._validate_batch_size(rollout_batch_size)
        self.random_seed = int(random_seed)

        self.evaluation_validity_thresholds = (
            evaluation_validity_thresholds or Phase2EvaluationValidityThresholds()
        )
        self.selector_profitability_thresholds = (
            selector_profitability_thresholds or Phase2SelectorProfitabilityThresholds()
        )
        self.baseline_uplift_thresholds = (
            baseline_uplift_thresholds or Phase2BaselineUpliftThresholds()
        )
        self.demonstration_consistency_thresholds = (
            demonstration_consistency_thresholds
            or Phase2DemonstrationConsistencyThresholds()
        )
        self.code_usage_collapse_thresholds = (
            code_usage_collapse_thresholds or Phase2CodeUsageCollapseThresholds()
        )
        self.generalization_stability_thresholds = (
            generalization_stability_thresholds
            or Phase2GeneralizationStabilityThresholds()
        )

        self.validation_score_history = tuple(float(v) for v in validation_score_history)
        self.selected_action_churn_history = tuple(
            float(v) for v in selected_action_churn_history
        )
        self.td_loss_history = tuple(float(v) for v in td_loss_history)
        self.imitation_loss_history = tuple(float(v) for v in imitation_loss_history)
        self.reward_mean_history = tuple(float(v) for v in reward_mean_history)
        self.train_mean_return = (
            float(train_mean_return) if train_mean_return is not None else None
        )
        self.test_mean_return = (
            float(test_mean_return) if test_mean_return is not None else None
        )
        self.train_usage_distribution = train_usage_distribution

    def evaluate_checkpoint(
        self,
        dataset: Phase2SelectionDataset,
        deterministic: bool = True,
        split_name: str = "validation",
        epoch: int | None = None,
        *,
        q_network: Phase2QNetwork | None = None,
        decoder_policy: FrozenArchetypeDecoderPolicy | None = None,
    ) -> Phase2ValidationResult:
        """评估 selector 并返回统一 validation result。"""

        resolved_q_network = q_network or self.q_network
        resolved_decoder_policy = decoder_policy or self.decoder_policy
        if resolved_q_network is None:
            raise ValueError("q_network is required to evaluate a Phase II checkpoint")
        if resolved_decoder_policy is None:
            raise ValueError(
                "decoder_policy is required to evaluate a Phase II checkpoint"
            )

        num_samples = self._sample_count(dataset)
        num_archetypes = self._num_archetypes(resolved_q_network, resolved_decoder_policy)
        sample_ids, assigned_code_labels = dataset.demonstration_horizon_label_dataset
        assigned_code_labels = np.asarray(assigned_code_labels, dtype=np.int64)

        q_values = self._compute_q_values(
            resolved_q_network,
            dataset.visible_states,
        )
        if q_values.shape != (num_samples, num_archetypes):
            raise ValueError(
                "q_network output must have shape [sample, num_archetypes], "
                f"got {tuple(q_values.shape)} != {(num_samples, num_archetypes)}"
            )
        selected_code_ids = self._select_code_ids(
            q_values,
            deterministic=deterministic,
            epoch=epoch,
        )
        random_code_ids = self._random_code_ids(
            num_samples=num_samples,
            num_archetypes=num_archetypes,
            epoch=epoch,
        )

        rollout_matrices = self._run_all_code_rollouts(
            dataset=dataset,
            decoder_policy=resolved_decoder_policy,
            num_archetypes=num_archetypes,
        )
        selector_returns = self._take_by_code(rollout_matrices.returns, selected_code_ids)
        selector_gross_returns = self._take_by_code(
            rollout_matrices.gross_returns,
            selected_code_ids,
        )
        selector_fees = self._take_by_code(rollout_matrices.fees, selected_code_ids)
        selector_turnover = self._take_by_code(
            rollout_matrices.turnover,
            selected_code_ids,
        )
        selector_actions = self._take_actions_by_code(
            rollout_matrices.actions,
            selected_code_ids,
        )
        assigned_label_returns = self._take_by_code(
            rollout_matrices.returns,
            assigned_code_labels,
        )
        random_returns = self._take_by_code(rollout_matrices.returns, random_code_ids)
        oracle_returns = self._oracle_returns(rollout_matrices.returns)
        hold_returns = self._hold_returns(dataset)

        selected_q_values = self._take_by_code(q_values, selected_code_ids)
        assigned_label_q_values = self._take_by_code(q_values, assigned_code_labels)
        q_margins = self._top1_top2_margins(q_values)

        evaluation_payload = Phase2EvaluationValidityPayload(
            split_name=split_name,
            epoch=epoch,
            num_samples=num_samples,
            failed_rollout_count=self._selected_failure_count(
                rollout_matrices.failed_mask,
                selected_code_ids,
            ),
            non_finite_reward_count=self._non_finite_selector_result_count(
                selector_returns,
                selector_gross_returns,
                selector_fees,
                selector_turnover,
            ),
            invalid_selected_code_count=self._invalid_code_count(
                selected_code_ids,
                num_archetypes,
            ),
            num_archetypes=num_archetypes,
        )
        layer0 = compute_evaluation_validity_metrics(
            evaluation_payload,
            deterministic_eval=deterministic,
            label_alignment_valid=self._label_alignment_valid(
                sample_ids,
                assigned_code_labels,
                num_samples=num_samples,
                num_archetypes=num_archetypes,
            ),
            visible_state_contract_valid=self._visible_state_contract_valid(dataset),
        )

        selector_payload = Phase2SelectorProfitabilityPayload(
            selector_returns=tuple(selector_returns),
            selector_gross_returns=tuple(selector_gross_returns),
            selector_fees=tuple(selector_fees),
            selector_turnover=tuple(selector_turnover),
        )
        layer1 = compute_selector_profitability_metrics(selector_payload)

        baseline_payload = Phase2BaselineUpliftPayload(
            selector_returns=tuple(selector_returns),
            assigned_label_returns=tuple(assigned_label_returns),
            random_returns=tuple(random_returns),
            oracle_returns=tuple(oracle_returns),
            random_seed=self._epoch_seed(epoch),
        )
        layer2 = compute_baseline_uplift_metrics(baseline_payload)

        demonstration_payload = Phase2DemonstrationConsistencyPayload(
            selected_code_ids=tuple(selected_code_ids),
            assigned_code_labels=tuple(assigned_code_labels),
            selector_returns=tuple(selector_returns),
            assigned_label_returns=tuple(assigned_label_returns),
            selected_q_values=tuple(selected_q_values),
            assigned_label_q_values=tuple(assigned_label_q_values),
        )
        layer3 = compute_demonstration_consistency_metrics(
            demonstration_payload,
            cross_entropy_to_assigned=self._cross_entropy_to_assigned(
                q_values,
                assigned_code_labels,
            ),
            kl_to_assigned_onehot=self._cross_entropy_to_assigned(
                q_values,
                assigned_code_labels,
            ),
        )

        per_code_diagnostics = build_per_code_usage_diagnostics(
            selected_code_ids=selected_code_ids,
            assigned_code_labels=assigned_code_labels,
            selector_returns=selector_returns,
            kl_returns=assigned_label_returns,
            num_archetypes=num_archetypes,
        )
        code_usage_payload = Phase2CodeUsageCollapsePayload(
            selected_code_ids=tuple(selected_code_ids),
            assigned_code_labels=tuple(assigned_code_labels),
            per_code_diagnostics=per_code_diagnostics,
        )
        layer4 = compute_code_usage_collapse_metrics(
            code_usage_payload,
            num_archetypes=num_archetypes,
            train_label_distribution=self.train_usage_distribution,
        )

        validation_mean_return = float(layer1.metrics.mean_return)
        validation_score_history = (
            self.validation_score_history + (validation_mean_return,)
        )
        validation_usage_distribution = self._code_distribution(
            selected_code_ids,
            num_archetypes,
        )
        stability_payload = Phase2GeneralizationStabilityPayload(
            train_score=self.train_mean_return,
            validation_score_history=validation_score_history,
            selected_action_churn_history=self.selected_action_churn_history,
            q_value_scale_history=(self._q_value_scale(q_values),),
        )
        layer5 = compute_generalization_stability_metrics(
            stability_payload,
            validation_mean_return=validation_mean_return,
            train_mean_return=self.train_mean_return,
            test_mean_return=self.test_mean_return,
            train_usage_distribution=self.train_usage_distribution,
            validation_usage_distribution=validation_usage_distribution,
            q_margins=tuple(q_margins),
            td_loss_history=self.td_loss_history,
            imitation_loss_history=self.imitation_loss_history,
            reward_mean_history=self.reward_mean_history,
        )

        layer_computations = (layer0, layer1, layer2, layer3, layer4, layer5)
        layers = self._evaluate_layer_rules(
            layer_computations,
            num_archetypes=num_archetypes,
        )
        metrics = Phase2ValidationMetrics(
            mean_return=float(layer1.metrics.mean_return),
            median_return=float(layer1.metrics.median_return),
            sharpe_like=float(layer1.metrics.sharpe_like),
            win_rate=float(layer1.metrics.win_rate),
            mean_turnover=float(layer1.metrics.mean_turnover),
        )
        payloads = Phase2ValidationPayloads(
            evaluation_validity_payload=evaluation_payload,
            selector_profitability_payload=selector_payload,
            baseline_uplift_payload=baseline_payload,
            demonstration_consistency_payload=demonstration_payload,
            code_usage_collapse_payload=code_usage_payload,
            generalization_stability_payload=stability_payload,
            report_payload=self._build_report_payload(
                selector_returns=selector_returns,
                assigned_label_returns=assigned_label_returns,
                random_returns=random_returns,
                oracle_returns=oracle_returns,
                hold_returns=hold_returns,
                selector_fees=selector_fees,
                selector_turnover=selector_turnover,
                selector_actions=selector_actions,
                q_margins=q_margins,
                selected_code_ids=selected_code_ids,
                assigned_code_labels=assigned_code_labels,
                per_code_diagnostics=per_code_diagnostics,
                dataset=dataset,
                num_archetypes=num_archetypes,
            ),
        )
        return Phase2ValidationResult(
            metrics=metrics,
            layers=layers,
            layer_computations=layer_computations,
            payloads=payloads,
        )

    def _compute_q_values(
        self,
        q_network: Phase2QNetwork,
        visible_states: tuple[np.ndarray, ...],
    ) -> np.ndarray:
        """批量计算全部样本的 Q values。"""

        num_samples = int(visible_states[0].shape[0])
        q_batches: list[np.ndarray] = []
        was_training = q_network.training
        q_network.to(self.device)
        q_network.eval()
        with torch.no_grad():
            for start in range(0, num_samples, self.rollout_batch_size):
                end = min(start + self.rollout_batch_size, num_samples)
                tensor_batch = self._visible_states_to_tensor_batch(
                    tuple(state[start:end] for state in visible_states),
                )
                q_batch = q_network(tensor_batch)
                q_batches.append(q_batch.detach().cpu().numpy().astype(np.float64))
        q_network.train(was_training)
        if not q_batches:
            return np.empty((0, self._q_network_num_archetypes(q_network)))
        return np.concatenate(q_batches, axis=0)

    def _select_code_ids(
        self,
        q_values: np.ndarray,
        *,
        deterministic: bool,
        epoch: int | None,
    ) -> np.ndarray:
        """根据 Q values 选择 code。"""

        if q_values.ndim != 2:
            raise ValueError("q_values must have shape [sample, num_archetypes]")
        if deterministic:
            return np.argmax(q_values, axis=1).astype(np.int64)

        rng = np.random.default_rng(self._epoch_seed(epoch))
        probabilities = self._softmax(q_values)
        return np.asarray(
            [
                rng.choice(q_values.shape[1], p=probabilities[index])
                for index in range(q_values.shape[0])
            ],
            dtype=np.int64,
        )

    def _run_all_code_rollouts(
        self,
        *,
        dataset: Phase2SelectionDataset,
        decoder_policy: FrozenArchetypeDecoderPolicy,
        num_archetypes: int,
    ) -> _RolloutMatrices:
        """对每个样本的每个 code 执行 rollout，返回 [sample, code] 矩阵。"""

        num_samples = self._sample_count(dataset)
        returns = np.full((num_samples, num_archetypes), np.nan, dtype=np.float64)
        gross_returns = np.full_like(returns, np.nan)
        fees = np.full_like(returns, np.nan)
        turnover = np.full_like(returns, np.nan)
        horizon = int(dataset.horizon_dataset[0].shape[1])
        actions = np.full(
            (num_samples, num_archetypes, horizon),
            -1,
            dtype=np.int64,
        )
        failed_mask = np.zeros((num_samples, num_archetypes), dtype=np.bool_)
        env = ArchetypeSelectionBatchEnv(
            dataset=dataset,
            decoder_policy=decoder_policy,
            reward_config=self.reward_config,
        )
        all_indices = np.arange(num_samples, dtype=np.int64)
        for code_id in range(num_archetypes):
            selected_codes = np.full(num_samples, code_id, dtype=np.int64)
            for start in range(0, num_samples, self.rollout_batch_size):
                end = min(start + self.rollout_batch_size, num_samples)
                indices = all_indices[start:end]
                try:
                    result = env.run_horizons(indices, selected_codes[start:end])
                except Exception:
                    failed_mask[start:end, code_id] = True
                    continue
                returns[start:end, code_id] = result.rewards
                gross_returns[start:end, code_id] = result.gross_returns
                fees[start:end, code_id] = result.fees
                turnover[start:end, code_id] = result.turnover
                actions[start:end, code_id, :] = result.actions
        return _RolloutMatrices(
            returns=returns,
            gross_returns=gross_returns,
            fees=fees,
            turnover=turnover,
            actions=actions,
            failed_mask=failed_mask,
        )

    def _evaluate_layer_rules(
        self,
        layer_computations: tuple[Phase2LayerComputation, ...],
        *,
        num_archetypes: int,
    ) -> tuple[Phase2LayerResult, ...]:
        """对 Layer 0-5 raw metrics 应用阈值规则。"""

        by_id = {computation.layer_id: computation for computation in layer_computations}
        return (
            evaluate_evaluation_validity_rules(
                by_id[0].metrics,
                self.evaluation_validity_thresholds,
            ),
            evaluate_selector_profitability_rules(
                by_id[1].metrics,
                self.selector_profitability_thresholds,
            ),
            evaluate_baseline_uplift_rules(
                by_id[2].metrics,
                self.baseline_uplift_thresholds,
            ),
            evaluate_demonstration_consistency_rules(
                by_id[3].metrics,
                self.demonstration_consistency_thresholds,
            ),
            evaluate_code_usage_collapse_rules(
                by_id[4].metrics,
                self.code_usage_collapse_thresholds,
                num_archetypes=num_archetypes,
            ),
            evaluate_generalization_stability_rules(
                by_id[5].metrics,
                self.generalization_stability_thresholds,
                num_archetypes=num_archetypes,
            ),
        )

    def _build_report_payload(
        self,
        *,
        selector_returns: np.ndarray,
        assigned_label_returns: np.ndarray,
        random_returns: np.ndarray,
        oracle_returns: np.ndarray,
        hold_returns: np.ndarray,
        selector_fees: np.ndarray,
        selector_turnover: np.ndarray,
        selector_actions: np.ndarray,
        q_margins: np.ndarray,
        selected_code_ids: np.ndarray,
        assigned_code_labels: np.ndarray,
        per_code_diagnostics: tuple[Any, ...],
        dataset: Phase2SelectionDataset,
        num_archetypes: int,
    ) -> Mapping[str, object]:
        """构造报表卡片复用的聚合 payload，不保存逐样本 trace。"""

        _, _, _, prices, _ = dataset.horizon_dataset
        return {
            "per_code_profitability_comparison": [
                item.to_dict() if hasattr(item, "to_dict") else item
                for item in per_code_diagnostics
            ],
            "selector_pair_profitability_matrix": (
                build_selector_pair_profitability_matrix(
                    selected_code_ids=selected_code_ids,
                    selector_returns=selector_returns,
                    kl_returns=assigned_label_returns,
                    random_returns=random_returns,
                    selector_fees=selector_fees,
                    selector_turnover=selector_turnover,
                    prices=prices,
                    selector_actions=selector_actions,
                    fee_rate=self.reward_config.fee_rate,
                )
            ),
            "code_diagnostics": (
                build_phase2_code_diagnostics(
                    selected_code_ids=selected_code_ids,
                    assigned_code_labels=assigned_code_labels,
                    selector_returns=selector_returns,
                    kl_returns=assigned_label_returns,
                    selector_fees=selector_fees,
                    selector_turnover=selector_turnover,
                    q_margins=q_margins,
                    num_archetypes=num_archetypes,
                    prices=prices,
                    selector_actions=selector_actions,
                    fee_rate=self.reward_config.fee_rate,
                )
            ),
            "codebook_usage_distribution": {
                "selector": self._count_distribution(selected_code_ids),
                "kl": self._count_distribution(assigned_code_labels),
            },
            "oracle_label_cumulative_returns": {
                "selector": self._cumulative_returns(selector_returns),
                "kl": self._cumulative_returns(assigned_label_returns),
                "random": self._cumulative_returns(random_returns),
                "oracle": self._cumulative_returns(oracle_returns),
                "hold": self._cumulative_returns(hold_returns),
            },
        }

    @staticmethod
    def _take_by_code(matrix: np.ndarray, code_ids: np.ndarray) -> np.ndarray:
        """按每行 code id 从 [sample, code] 矩阵取值，非法 code 返回 NaN。"""

        values = np.full(matrix.shape[0], np.nan, dtype=np.float64)
        codes = np.asarray(code_ids, dtype=np.int64)
        valid = (codes >= 0) & (codes < matrix.shape[1])
        if np.any(valid):
            row_indices = np.flatnonzero(valid)
            values[row_indices] = matrix[row_indices, codes[valid]]
        return values

    @staticmethod
    def _take_actions_by_code(matrix: np.ndarray, code_ids: np.ndarray) -> np.ndarray:
        """按每行 code id 从 [sample, code, horizon] 动作矩阵取动作序列。"""

        if matrix.ndim != 3:
            raise ValueError("action matrix must have shape [sample, code, horizon]")
        values = np.full((matrix.shape[0], matrix.shape[2]), -1, dtype=np.int64)
        codes = np.asarray(code_ids, dtype=np.int64)
        valid = (codes >= 0) & (codes < matrix.shape[1])
        if np.any(valid):
            row_indices = np.flatnonzero(valid)
            values[row_indices] = matrix[row_indices, codes[valid], :]
        return values

    @staticmethod
    def _oracle_returns(return_matrix: np.ndarray) -> np.ndarray:
        """每个样本 hindsight 选择收益最高的 code。"""

        if return_matrix.size == 0:
            return np.asarray([], dtype=np.float64)
        valid = np.any(np.isfinite(return_matrix), axis=1)
        values = np.full(return_matrix.shape[0], np.nan, dtype=np.float64)
        if np.any(valid):
            values[valid] = np.nanmax(return_matrix[valid], axis=1)
        return values

    def _hold_returns(self, dataset: Phase2SelectionDataset) -> np.ndarray:
        """计算一直持有 long position 的 baseline 收益。"""

        _, _, _, prices, depthprices = dataset.horizon_dataset
        sample_count = self._sample_count(dataset)
        horizon = int(prices.shape[1])
        actions = np.full((sample_count, horizon), 2, dtype=np.int64)
        execution = ActionExecutionCalculator.execute_actions(
            prices=prices,
            actions=actions,
            fee_rate=self.reward_config.fee_rate,
            depthprices=depthprices,
        )
        return self._postprocess_rewards(np.asarray(execution.returns, dtype=np.float64))

    @staticmethod
    def _top1_top2_margins(q_values: np.ndarray) -> np.ndarray:
        """返回每个样本 top1-top2 Q margin。"""

        if q_values.shape[1] < 2:
            return np.full(q_values.shape[0], np.nan, dtype=np.float64)
        sorted_q = np.sort(q_values, axis=1)
        return sorted_q[:, -1] - sorted_q[:, -2]

    @staticmethod
    def _softmax(values: np.ndarray) -> np.ndarray:
        """稳定 softmax。"""

        shifted = values - np.max(values, axis=1, keepdims=True)
        exp_values = np.exp(shifted)
        denominator = np.sum(exp_values, axis=1, keepdims=True)
        return exp_values / denominator

    def _random_code_ids(
        self,
        *,
        num_samples: int,
        num_archetypes: int,
        epoch: int | None,
    ) -> np.ndarray:
        """生成稳定 random baseline code ids。"""

        rng = np.random.default_rng(self._epoch_seed(epoch))
        return rng.integers(0, num_archetypes, size=num_samples, dtype=np.int64)

    def _postprocess_rewards(self, rewards: np.ndarray) -> np.ndarray:
        """应用 Phase II reward clip，保持 baseline 与 env reward 口径一致。"""

        reward_values = np.asarray(rewards, dtype=np.float64)
        if self.reward_config.reward_clip is None:
            return reward_values
        clip_value = float(self.reward_config.reward_clip)
        if clip_value <= 0.0:
            raise ValueError(f"reward_clip must be positive, got {clip_value}")
        return np.clip(reward_values, -clip_value, clip_value)

    def _visible_states_to_tensor_batch(
        self,
        visible_states: tuple[np.ndarray, ...],
    ) -> tuple[torch.Tensor, ...]:
        """把六路 visible states 转成 Q-network batch tensor。"""

        if len(visible_states) != 6:
            raise ValueError("visible_states must contain six arrays")
        tensors: list[torch.Tensor] = []
        batch_size: int | None = None
        for state in visible_states:
            array = np.asarray(state, dtype=np.float32)
            if array.ndim == 2:
                array = array[np.newaxis, ...]
            if array.ndim != 3:
                raise ValueError("visible state arrays must be 2D or 3D")
            if batch_size is None:
                batch_size = int(array.shape[0])
            elif int(array.shape[0]) != batch_size:
                raise ValueError("all visible state arrays must share batch size")
            tensors.append(torch.as_tensor(array, dtype=torch.float32, device=self.device))
        return tuple(tensors)

    def _cross_entropy_to_assigned(
        self,
        q_values: np.ndarray,
        assigned_code_labels: np.ndarray,
    ) -> float:
        """把 Q values softmax 后计算 assigned one-hot cross entropy。"""

        probabilities = self._softmax(q_values)
        labels = np.asarray(assigned_code_labels, dtype=np.int64)
        valid = (labels >= 0) & (labels < probabilities.shape[1])
        if not np.any(valid):
            return float("nan")
        row_indices = np.flatnonzero(valid)
        selected_probabilities = probabilities[row_indices, labels[valid]]
        return float(-np.mean(np.log(np.clip(selected_probabilities, 1e-12, 1.0))))

    @staticmethod
    def _non_finite_selector_result_count(*arrays: np.ndarray) -> int:
        """统计 selector 关键结果中任一字段非有限的样本数。"""

        if not arrays:
            return 0
        size = arrays[0].shape[0]
        invalid = np.zeros(size, dtype=np.bool_)
        for array in arrays:
            invalid |= ~np.isfinite(array[:size])
        return int(np.sum(invalid))

    @staticmethod
    def _invalid_code_count(code_ids: np.ndarray, num_archetypes: int) -> int:
        """统计非法 code id 数。"""

        codes = np.asarray(code_ids, dtype=np.int64)
        return int(np.sum((codes < 0) | (codes >= num_archetypes)))

    @staticmethod
    def _selected_failure_count(failed_mask: np.ndarray, code_ids: np.ndarray) -> int:
        """统计 selector 实际选中 code 对应的 rollout 失败数。"""

        codes = np.asarray(code_ids, dtype=np.int64)
        valid = (codes >= 0) & (codes < failed_mask.shape[1])
        if not np.any(valid):
            return int(codes.shape[0])
        row_indices = np.flatnonzero(valid)
        failed_count = int(np.sum(failed_mask[row_indices, codes[valid]]))
        failed_count += int(np.sum(~valid))
        return failed_count

    @staticmethod
    def _label_alignment_valid(
        sample_ids: np.ndarray,
        code_labels: np.ndarray,
        *,
        num_samples: int,
        num_archetypes: int,
    ) -> bool:
        """检查 label 数量和值域是否与当前 split 对齐。"""

        if sample_ids.shape != (num_samples,) or code_labels.shape != (num_samples,):
            return False
        if np.any(sample_ids < 0):
            return False
        if np.any(code_labels < 0) or np.any(code_labels >= num_archetypes):
            return False
        return True

    @staticmethod
    def _visible_state_contract_valid(dataset: Phase2SelectionDataset) -> bool:
        """检查 selector visible state 的基础形状契约。"""

        try:
            visible_states = dataset.visible_states
            horizon_states, relative_states, trend_states, prices, depthprices = (
                dataset.horizon_dataset
            )
            if len(visible_states) != 6:
                return False
            sample_count = int(visible_states[0].shape[0])
            for state in visible_states:
                if state.ndim != 3 or int(state.shape[0]) != sample_count:
                    return False
            for state in (horizon_states, relative_states, trend_states, prices):
                if state.ndim != 3 or int(state.shape[0]) != sample_count:
                    return False
            if depthprices is not None:
                if depthprices.ndim != 3 or int(depthprices.shape[0]) != sample_count:
                    return False
        except (TypeError, ValueError):
            return False
        return True

    @staticmethod
    def _sample_count(dataset: Phase2SelectionDataset) -> int:
        """返回 split 样本数。"""

        return int(dataset.visible_states[0].shape[0])

    @staticmethod
    def _num_archetypes(
        q_network: Phase2QNetwork,
        decoder_policy: FrozenArchetypeDecoderPolicy,
    ) -> int:
        """从 Q-network 或 decoder policy 读取 codebook size。"""

        num_archetypes = Phase2Evaluator._q_network_num_archetypes(q_network)
        if num_archetypes > 0:
            return num_archetypes
        if hasattr(decoder_policy, "num_archetypes"):
            return int(getattr(decoder_policy, "num_archetypes"))
        raise ValueError("unable to infer num_archetypes")

    @staticmethod
    def _q_network_num_archetypes(q_network: Phase2QNetwork) -> int:
        """从 Q-network config 读取 archetype 数。"""

        config = getattr(q_network, "config", None)
        if config is None or not hasattr(config, "num_archetypes"):
            return 0
        return int(getattr(config, "num_archetypes"))

    @staticmethod
    def _q_value_scale(q_values: np.ndarray) -> float:
        """Q value 尺度参考。"""

        finite = q_values[np.isfinite(q_values)]
        if finite.size == 0:
            return float("nan")
        return float(np.mean(np.abs(finite)))

    @staticmethod
    def _code_distribution(code_ids: np.ndarray, num_archetypes: int) -> tuple[float, ...]:
        """返回 code id 的归一化分布。"""

        if num_archetypes <= 0:
            return ()
        valid = code_ids[(code_ids >= 0) & (code_ids < num_archetypes)]
        counts = np.bincount(valid, minlength=num_archetypes).astype(np.float64)
        total = float(np.sum(counts))
        if total <= 0.0:
            return tuple(float(0.0) for _ in range(num_archetypes))
        return tuple(float(value) for value in counts / total)

    @staticmethod
    def _count_distribution(code_ids: np.ndarray) -> list[dict[str, int]]:
        """返回 report 友好的 code count 分布。"""

        if code_ids.size == 0:
            return []
        valid = code_ids[code_ids >= 0]
        if valid.size == 0:
            return []
        counts = np.bincount(valid.astype(np.int64))
        return [
            {"code_id": int(code_id), "count": int(count)}
            for code_id, count in enumerate(counts)
            if int(count) > 0
        ]

    @staticmethod
    def _cumulative_returns(values: np.ndarray) -> list[float]:
        """返回 NaN 安全的累计收益序列。"""

        finite_values = np.where(np.isfinite(values), values, 0.0)
        return [float(value) for value in np.cumsum(finite_values)]

    def _epoch_seed(self, epoch: int | None) -> int:
        """构造随 epoch 稳定变化的随机种子。"""

        return self.random_seed if epoch is None else self.random_seed + int(epoch)

    @staticmethod
    def _validate_batch_size(batch_size: int) -> int:
        """校验 batch size。"""

        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("rollout_batch_size must be positive")
        return batch_size


__all__ = ["Phase2Evaluator"]
