import pytest
import torch
import torch.nn.functional as F

from src.phase2.phase2_config import Phase2RewardConfig, Phase2TrainConfig
from src.phase2.rl.phase2_double_dqn_loss import (
    compute_double_dqn_loss,
    compute_double_dqn_targets,
    compute_imitation_kl_loss,
)
from src.phase2.rl.phase2_double_dqn_trainer import build_epsilon_by_epoch
from src.phase2.rl.phase2_replay_buffer import Phase2SelectionTransitionTensorBatch


class _SequentialQNetwork(torch.nn.Module):
    def __init__(self, outputs: list[torch.Tensor]) -> None:
        super().__init__()
        self.outputs = outputs
        self.call_index = 0

    def forward(self, visible_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
        output = self.outputs[self.call_index]
        self.call_index += 1
        return output


def test_double_dqn_target_uses_online_argmax_and_target_value() -> None:
    online_next_q = torch.tensor([[1.0, 5.0, 2.0], [4.0, 3.0, 1.0]])
    target_next_q = torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    online = _SequentialQNetwork([online_next_q])
    target = _SequentialQNetwork([target_next_q])

    targets, greedy_next_actions = compute_double_dqn_targets(
        online_q_network=online,
        target_q_network=target,
        next_visible_states=_visible_states(),
        rewards=torch.tensor([1.0, 2.0]),
        dones=torch.tensor([0.0, 1.0]),
        gamma=0.5,
    )

    torch.testing.assert_close(greedy_next_actions, torch.tensor([1, 0]))
    torch.testing.assert_close(targets, torch.tensor([11.0, 2.0]))


def test_imitation_kl_loss_matches_one_hot_cross_entropy() -> None:
    q_values = torch.tensor([[0.0, 2.0, 4.0], [3.0, 1.0, 0.0]])
    labels = (torch.tensor([7, 8]), torch.tensor([1, 2]))

    loss = compute_imitation_kl_loss(
        q_values=q_values,
        demonstration_horizon_label_batch=labels,
    )

    torch.testing.assert_close(loss, F.cross_entropy(q_values, labels[1]))


def test_double_dqn_loss_combines_td_and_weighted_imitation_terms() -> None:
    current_q = torch.tensor(
        [[0.0, 2.0, 4.0], [3.0, 1.0, 0.0]],
        requires_grad=True,
    )
    online_next_q = torch.tensor([[1.0, 5.0, 2.0], [4.0, 3.0, 1.0]])
    target_next_q = torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    online = _SequentialQNetwork([current_q, online_next_q])
    target = _SequentialQNetwork([target_next_q])
    batch = Phase2SelectionTransitionTensorBatch(
        visible_states=_visible_states(),
        actions=torch.tensor([2, 0]),
        rewards=torch.tensor([1.0, 2.0]),
        next_visible_states=_visible_states(),
        dones=torch.tensor([0.0, 1.0]),
        demonstration_horizon_label_batch=(
            torch.tensor([7, 8]),
            torch.tensor([1, 2]),
        ),
    )

    output = compute_double_dqn_loss(
        online_q_network=online,
        target_q_network=target,
        batch=batch,
        reward_config=Phase2RewardConfig(
            gamma=0.5,
            imitation_alpha=0.5,
            normalize_rewards=False,
        ),
        train_config=Phase2TrainConfig(
            td_loss_beta=2.0,
            imitation_loss_beta=3.0,
        ),
    )

    expected_td_loss = F.smooth_l1_loss(
        torch.tensor([4.0, 3.0]),
        torch.tensor([11.0, 2.0]),
    )
    expected_imitation_loss = F.cross_entropy(
        current_q,
        torch.tensor([1, 2]),
    )
    expected_total = 2.0 * expected_td_loss + 0.5 * 3.0 * expected_imitation_loss
    torch.testing.assert_close(output.td_loss, expected_td_loss)
    torch.testing.assert_close(output.imitation_loss, expected_imitation_loss)
    torch.testing.assert_close(output.total_loss, expected_total)


def test_epsilon_schedule_starts_at_initial_value_then_linearly_decays() -> None:
    config = Phase2TrainConfig(
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay_epochs=10,
    )

    assert build_epsilon_by_epoch(1, config) == 1.0
    assert build_epsilon_by_epoch(6, config) == pytest.approx(0.55)
    assert build_epsilon_by_epoch(11, config) == pytest.approx(0.1)
    assert build_epsilon_by_epoch(20, config) == pytest.approx(0.1)


def _visible_states() -> tuple[torch.Tensor, ...]:
    return tuple(torch.zeros(2, 1, 1) for _ in range(6))
