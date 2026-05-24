"""Phase II Double DQN batch rollout trainer."""

from __future__ import annotations

import time

import numpy as np
import torch

from ...model.data_types import VisibleStates
from ..metrics.phase2_metrics import Phase2Metrics
from .phase2_double_dqn_trainer import (
    Phase2DoubleDqnTrainer,
    build_epsilon_by_epoch,
    logger,
)


class Phase2DoubleDqnBatchTrainer(Phase2DoubleDqnTrainer):
    """Phase II Double DQN trainer with batched environment rollout.

    This subclass only changes transition collection. Optimizer updates,
    target-network synchronization, checkpoint payloads and metric aggregation stay in
    ``Phase2DoubleDqnTrainer``.
    """

    def train_one_epoch(
        self,
        epoch: int,
    ) -> Phase2Metrics:
        """执行一个批量 rollout 的 Double DQN 训练 epoch。"""

        self.online_q_network.train()
        self.target_q_network.eval()

        epoch_started_at = time.perf_counter()
        epsilon = build_epsilon_by_epoch(epoch, self.train_config)
        totals = Phase2Metrics(stage="selector", split="train", epoch=epoch)
        total_steps = len(self.env)
        rollout_batch_size = self._rollout_batch_size()
        progress_interval = self._progress_log_interval(total_steps)
        steps_collected = 0
        updates_run = 0
        update_start_logged = False

        logger.info(
            "Phase II Double DQN epoch started: epoch=%d samples=%d epsilon=%.6f "
            "replay_size=%d batch_size=%d rollout_batch_size=%d "
            "updates_per_epoch=%d rollout_mode=batch",
            epoch,
            total_steps,
            epsilon,
            len(self.replay_buffer),
            self.train_config.batch_size,
            rollout_batch_size,
            self.train_config.updates_per_epoch,
        )

        for batch_start in range(0, total_steps, rollout_batch_size):
            batch_end = min(batch_start + rollout_batch_size, total_steps)
            sample_indices = np.arange(batch_start, batch_end, dtype=np.int64)
            visible_states = self.env.visible_states_at(sample_indices)
            actions = self.select_actions(
                visible_states=visible_states,
                epsilon=epsilon,
                deterministic=False,
            )
            batch_result = self.env.run_horizons(
                indices=sample_indices,
                selected_code_ids=actions,
            )
            self.replay_buffer.add_batch(
                visible_states=batch_result.observations,
                actions=batch_result.selected_code_ids,
                rewards=batch_result.rewards,
                next_visible_states=batch_result.next_observations,
                dones=batch_result.dones,
                demonstration_horizon_label_batch=(
                    batch_result.sample_ids,
                    batch_result.assigned_code_labels,
                ),
            )
            steps_collected += sample_indices.shape[0]

            updates_run, update_start_logged = self._maybe_run_epoch_updates(
                epoch=epoch,
                totals=totals,
                updates_run=updates_run,
                update_start_logged=update_start_logged,
                steps_collected=steps_collected,
                total_steps=total_steps,
                progress_interval=progress_interval,
            )

        return self._finish_epoch(
            epoch=epoch,
            totals=totals,
            steps_collected=steps_collected,
            total_steps=total_steps,
            updates_run=updates_run,
            epoch_started_at=epoch_started_at,
        )

    def select_actions(
        self,
        visible_states: VisibleStates,
        epsilon: float,
        deterministic: bool = False,
    ) -> np.ndarray:
        """批量使用 greedy 或 epsilon-greedy 策略选择 archetype ids。"""

        visible_state_batch = self._visible_states_to_numpy_batch(visible_states)
        batch_size = visible_state_batch[0].shape[0]
        num_archetypes = int(self.online_q_network.config.num_archetypes)
        actions = np.empty(batch_size, dtype=np.int64)

        if deterministic or epsilon <= 0.0:
            random_mask = np.zeros(batch_size, dtype=np.bool_)
        elif epsilon >= 1.0:
            random_mask = np.ones(batch_size, dtype=np.bool_)
        else:
            random_mask = self.rng.random(batch_size) < float(epsilon)

        random_count = int(np.sum(random_mask))
        if random_count > 0:
            actions[random_mask] = self.rng.integers(
                num_archetypes,
                size=random_count,
                dtype=np.int64,
            )

        greedy_indices = np.flatnonzero(~random_mask)
        if greedy_indices.shape[0] > 0:
            greedy_visible_states = tuple(
                state[greedy_indices] for state in visible_state_batch
            )
            greedy_state_batch = self._visible_states_to_tensor_batch(
                greedy_visible_states,
                device=self.device,
            )
            was_training = self.online_q_network.training
            with torch.no_grad():
                self.online_q_network.eval()
                q_values = self.online_q_network(greedy_state_batch)
            actions[greedy_indices] = (
                torch.argmax(q_values, dim=-1).detach().cpu().numpy().astype(np.int64)
            )
            self.online_q_network.train(was_training)

        return actions

    def _rollout_batch_size(self) -> int:
        """返回批量 env transition 采样大小。"""

        batch_size = int(self.train_config.rollout_batch_size)
        if batch_size <= 1:
            raise ValueError(
                "Phase2DoubleDqnBatchTrainer requires rollout_batch_size > 1, "
                f"got {self.train_config.rollout_batch_size}"
            )
        return batch_size


__all__ = ["Phase2DoubleDqnBatchTrainer"]
