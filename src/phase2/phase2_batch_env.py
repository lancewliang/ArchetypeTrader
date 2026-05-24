"""Phase II batched archetype selection environment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..model.data_types import VisibleStates
from ..utils import ActionExecutionResult
from .phase2_env import ArchetypeSelectionEnv


@dataclass(frozen=True)
class Phase2SelectionBatchResult:
    """Phase II 批量 horizon-level env 执行结果。"""

    observations: VisibleStates
    rewards: np.ndarray
    next_observations: VisibleStates
    dones: np.ndarray
    sample_ids: np.ndarray
    selected_code_ids: np.ndarray
    assigned_code_labels: np.ndarray
    gross_returns: np.ndarray
    fees: np.ndarray
    turnover: np.ndarray


class ArchetypeSelectionBatchEnv(ArchetypeSelectionEnv):
    """Batched variant of ``ArchetypeSelectionEnv`` for high-throughput rollout."""

    def run_horizons(
        self,
        indices: np.ndarray,
        selected_code_ids: np.ndarray,
    ) -> Phase2SelectionBatchResult:
        """批量执行多个 horizon-level archetype action。"""

        normalized_indices = self._validate_indices(indices)
        selected_code_ids = self._validate_selected_code_ids(
            selected_code_ids,
            expected_size=normalized_indices.shape[0],
        )
        actions = self._decode_actions_batch(normalized_indices, selected_code_ids)
        execution = self._execute_actions_batch(normalized_indices, actions)
        rewards = self._postprocess_rewards(execution.returns)
        sample_ids, assigned_code_labels = self._sample_labels_at(normalized_indices)

        next_indices = normalized_indices + 1
        dones = next_indices >= self.sample_count
        next_indices = np.where(dones, normalized_indices, next_indices)

        return Phase2SelectionBatchResult(
            observations=self.visible_states_at(normalized_indices),
            rewards=rewards,
            next_observations=self.visible_states_at(next_indices),
            dones=dones.astype(np.bool_, copy=False),
            sample_ids=sample_ids,
            selected_code_ids=selected_code_ids,
            assigned_code_labels=assigned_code_labels,
            gross_returns=np.asarray(execution.gross_returns, dtype=np.float64),
            fees=np.asarray(execution.fees, dtype=np.float64),
            turnover=np.asarray(execution.turnover, dtype=np.float64),
        )

    def visible_states_at(self, indices: np.ndarray) -> VisibleStates:
        """批量读取 selector 可见状态，并复制以避免外部原地修改 dataset。"""

        normalized_indices = self._validate_indices(indices)
        return tuple(
            np.asarray(visible_state[normalized_indices], dtype=np.float32).copy()
            for visible_state in self.dataset.visible_states
        )

    def _decode_actions_batch(
        self,
        indices: np.ndarray,
        selected_code_ids: np.ndarray,
    ) -> np.ndarray:
        """批量使用 frozen decoder 将 selected code 转成 ``[batch, horizon]`` 动作序列。"""

        horizon_states, relative_states, trend_states, _, _ = self.dataset.horizon_dataset
        horizon_states_tensor = torch.as_tensor(
            horizon_states[indices],
            dtype=torch.float32,
        )
        relative_states_tensor = torch.as_tensor(
            relative_states[indices],
            dtype=torch.float32,
        )
        trend_states_tensor = torch.as_tensor(
            trend_states[indices],
            dtype=torch.float32,
        )
        selected_code_ids_tensor = torch.as_tensor(selected_code_ids, dtype=torch.long)

        decoded_actions = self.decoder_policy.decode_actions(
            horizon_states=horizon_states_tensor,
            horizon_relative_states=relative_states_tensor,
            horizon_trend_states=trend_states_tensor,
            selected_code_ids=selected_code_ids_tensor,
        )
        if isinstance(decoded_actions, torch.Tensor):
            action_values = decoded_actions.detach().cpu().numpy()
        else:
            action_values = np.asarray(decoded_actions)
        expected_shape = (indices.shape[0], horizon_states.shape[1])
        if action_values.shape != expected_shape:
            raise ValueError(
                "decoder actions shape must match batch horizon shape: "
                f"{tuple(action_values.shape)} != {expected_shape}"
            )
        action_values = np.asarray(action_values, dtype=np.int64)
        if np.any(action_values < 0) or np.any(action_values > 2):
            min_action = int(np.min(action_values))
            max_action = int(np.max(action_values))
            raise ValueError(
                "decoder actions must use ids 0=short, 1=flat, 2=long; "
                f"got min={min_action}, max={max_action}"
            )
        return action_values

    def _execute_actions_batch(
        self,
        indices: np.ndarray,
        actions: np.ndarray,
    ) -> ActionExecutionResult:
        """按统一交易执行口径批量计算 horizon 收益。"""

        _, _, _, prices, depthprices = self.dataset.horizon_dataset
        depthprice_slice = None
        if depthprices is not None:
            depthprice_slice = depthprices[indices]
        return self.execution_calculator.execute(
            prices=prices[indices],
            actions=actions,
            depthprices=depthprice_slice,
        )

    def _postprocess_rewards(self, rewards: np.ndarray) -> np.ndarray:
        """批量应用 Phase II reward 后处理。"""

        reward_values = np.asarray(rewards, dtype=np.float64)
        if self.reward_config.reward_clip is None:
            return reward_values
        clip_value = float(self.reward_config.reward_clip)
        if clip_value <= 0.0:
            raise ValueError(f"reward_clip must be positive, got {clip_value}")
        return np.clip(reward_values, -clip_value, clip_value)

    def _sample_labels_at(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """批量读取 dataset 行号对应的原始 sample_id 和 assigned code label。"""

        sample_ids, code_labels = self.dataset.demonstration_horizon_label_dataset
        return (
            np.asarray(sample_ids[indices], dtype=np.int64).copy(),
            np.asarray(code_labels[indices], dtype=np.int64).copy(),
        )

    def _validate_indices(self, indices: np.ndarray) -> np.ndarray:
        """校验一组 dataset 行号。"""

        index_values = np.asarray(indices, dtype=np.int64)
        if index_values.ndim != 1:
            raise ValueError("indices must have shape [batch]")
        if index_values.shape[0] == 0:
            raise ValueError("indices must not be empty")
        min_index = int(np.min(index_values))
        max_index = int(np.max(index_values))
        if min_index < 0 or max_index >= self.sample_count:
            raise IndexError(
                f"indices must be in [0, {self.sample_count}), "
                f"got min={min_index}, max={max_index}"
            )
        return index_values

    def _validate_selected_code_ids(
        self,
        selected_code_ids: np.ndarray,
        *,
        expected_size: int,
    ) -> np.ndarray:
        """校验一组 selected archetype ids。"""

        code_values = np.asarray(selected_code_ids, dtype=np.int64)
        if code_values.ndim != 1:
            raise ValueError("selected_code_ids must have shape [batch]")
        if code_values.shape[0] != expected_size:
            raise ValueError(
                "selected_code_ids batch size must match indices: "
                f"{code_values.shape[0]} != {expected_size}"
            )
        if code_values.shape[0] == 0:
            raise ValueError("selected_code_ids must not be empty")
        min_code = int(np.min(code_values))
        if min_code < 0:
            raise ValueError("selected_code_ids must be non-negative")
        num_archetypes = self._num_archetypes()
        if num_archetypes is not None:
            max_code = int(np.max(code_values))
            if max_code >= num_archetypes:
                raise ValueError(
                    f"selected_code_ids must be in [0, {num_archetypes}), "
                    f"got max={max_code}"
                )
        return code_values


__all__ = ["ArchetypeSelectionBatchEnv", "Phase2SelectionBatchResult"]
