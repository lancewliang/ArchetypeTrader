import numpy as np
import torch

from src.phase2.rl.phase2_replay_buffer import (
    Phase2ReplayBuffer,
    Phase2ReplayTransition,
)


VISIBLE_STATE_SHAPES = tuple((1,) for _ in range(6))


def _visible_states(value: float) -> tuple[np.ndarray, ...]:
    return tuple(
        np.full(shape, value, dtype=np.float32)
        for shape in VISIBLE_STATE_SHAPES
    )


def _visible_state_batch(values: list[float]) -> tuple[np.ndarray, ...]:
    return tuple(
        np.stack(
            [
                np.full(shape, value, dtype=np.float32)
                for value in values
            ],
            axis=0,
        )
        for shape in VISIBLE_STATE_SHAPES
    )


def _transition(
    *,
    sample_id: int,
    action: int,
    reward: float,
    code_label: int = 0,
) -> Phase2ReplayTransition:
    return Phase2ReplayTransition(
        visible_states=_visible_states(float(sample_id)),
        action=action,
        reward=reward,
        next_visible_states=_visible_states(float(sample_id) + 1.0),
        done=False,
        demonstration_horizon_label=(sample_id, code_label),
    )


def _sample_pairs(buffer: Phase2ReplayBuffer) -> set[tuple[int, int]]:
    batch = buffer.sample(batch_size=len(buffer), device=torch.device("cpu"))
    sample_ids, _ = batch.demonstration_horizon_label_batch
    return set(
        zip(
            sample_ids.cpu().numpy().tolist(),
            batch.actions.cpu().numpy().tolist(),
            strict=True,
        )
    )


def test_add_skips_duplicate_sample_id_action_pair() -> None:
    buffer = Phase2ReplayBuffer(
        capacity=4,
        visible_state_shapes=VISIBLE_STATE_SHAPES,
        seed=0,
    )

    buffer.add(_transition(sample_id=7, action=2, reward=1.0))
    buffer.add(_transition(sample_id=7, action=2, reward=9.0))

    assert len(buffer) == 1
    batch = buffer.sample(batch_size=1, device=torch.device("cpu"))
    assert batch.rewards.item() == 1.0


def test_add_allows_same_sample_id_with_different_action() -> None:
    buffer = Phase2ReplayBuffer(
        capacity=4,
        visible_state_shapes=VISIBLE_STATE_SHAPES,
        seed=0,
    )

    buffer.add(_transition(sample_id=7, action=2, reward=1.0))
    buffer.add(_transition(sample_id=7, action=3, reward=2.0))

    assert len(buffer) == 2
    assert _sample_pairs(buffer) == {(7, 2), (7, 3)}


def test_add_batch_skips_existing_and_in_batch_duplicate_pairs() -> None:
    buffer = Phase2ReplayBuffer(
        capacity=8,
        visible_state_shapes=VISIBLE_STATE_SHAPES,
        seed=0,
    )
    buffer.add(_transition(sample_id=1, action=0, reward=1.0))

    buffer.add_batch(
        visible_states=_visible_state_batch([10.0, 20.0, 21.0, 22.0]),
        actions=np.asarray([0, 0, 0, 1], dtype=np.int64),
        rewards=np.asarray([10.0, 20.0, 21.0, 22.0], dtype=np.float32),
        next_visible_states=_visible_state_batch([11.0, 21.0, 22.0, 23.0]),
        dones=np.asarray([False, False, False, True], dtype=np.bool_),
        demonstration_horizon_label_batch=(
            np.asarray([1, 2, 2, 2], dtype=np.int64),
            np.asarray([0, 0, 0, 0], dtype=np.int64),
        ),
    )

    assert len(buffer) == 3
    assert _sample_pairs(buffer) == {(1, 0), (2, 0), (2, 1)}


def test_evicted_pair_can_be_added_again() -> None:
    buffer = Phase2ReplayBuffer(
        capacity=2,
        visible_state_shapes=VISIBLE_STATE_SHAPES,
        seed=0,
    )

    buffer.add(_transition(sample_id=1, action=0, reward=1.0))
    buffer.add(_transition(sample_id=2, action=0, reward=2.0))
    buffer.add(_transition(sample_id=3, action=0, reward=3.0))
    buffer.add(_transition(sample_id=1, action=0, reward=4.0))

    assert len(buffer) == 2
    assert _sample_pairs(buffer) == {(1, 0), (3, 0)}
