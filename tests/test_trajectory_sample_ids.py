import numpy as np
import torch

from src.model.tensor_data_types import (
    build_trajectory_tensor_dataset,
    move_trajectory_batch_to_device,
)
from src.store.artifact_store import DataStore
from src.tool.SingleTrade_DP_Planner import SingleTrade_DP_Planner
from src.utils.trade_execution import LOB_DEPTH_WIDTH


def test_build_trajectory_dataset_appends_sample_ids() -> None:
    horizon = 3
    horizon_dataset = _horizon_dataset(sample_count=2, horizon=horizon)
    planner = SingleTrade_DP_Planner(horizon=horizon, fee_rate=0.0)

    trajectory_dataset = planner.build_trajectory_dataset(horizon_dataset)

    assert [trajectory[5] for trajectory in trajectory_dataset] == [0, 1]


def test_save_load_trajectory_dataset_preserves_sample_ids(tmp_path) -> None:
    trajectory_dataset = [
        _trajectory(sample_id=10),
        _trajectory(sample_id=20),
    ]
    path = tmp_path / "trajectory.npz"
    store = DataStore()

    store.save_trajectory_dataset(trajectory_dataset, path)
    loaded = store.load_trajectory_dataset(path)

    assert [trajectory[5] for trajectory in loaded] == [10, 20]
    np.testing.assert_allclose(loaded[0][0], trajectory_dataset[0][0])


def test_build_trajectory_tensor_dataset_uses_stored_sample_ids() -> None:
    trajectory_dataset = [
        _trajectory(sample_id=42),
        _trajectory(sample_id=7),
    ]

    tensor_dataset = build_trajectory_tensor_dataset(trajectory_dataset)

    assert tensor_dataset.tensors[5].tolist() == [42, 7]


def test_move_trajectory_batch_to_device_preserves_sample_ids() -> None:
    batch = (
        torch.zeros((2, 3, 2)),
        torch.zeros((2, 3, 1)),
        torch.zeros((2, 3, 1)),
        torch.ones((2, 3), dtype=torch.long),
        torch.zeros((2, 3)),
        torch.asarray([42, 7], dtype=torch.long),
    )

    moved = move_trajectory_batch_to_device(batch, "cpu")

    assert len(moved) == 6
    assert moved[5].tolist() == [42, 7]


def _trajectory(sample_id: int) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
]:
    horizon = 3
    return (
        np.ones((horizon, 2), dtype=np.float32),
        np.ones((horizon, 1), dtype=np.float32),
        np.ones((horizon, 1), dtype=np.float32),
        np.ones(horizon, dtype=np.int64),
        np.zeros(horizon, dtype=np.float32),
        sample_id,
    )


def _horizon_dataset(
    *,
    sample_count: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    states = np.arange(sample_count * horizon * 2, dtype=np.float32).reshape(
        sample_count,
        horizon,
        2,
    )
    relative_states = np.zeros((sample_count, horizon, 1), dtype=np.float32)
    trend_states = np.zeros((sample_count, horizon, 1), dtype=np.float32)
    prices = np.tile(
        np.asarray([[100.0], [101.0], [102.0]], dtype=np.float32),
        (sample_count, 1, 1),
    )
    depthprices = np.zeros(
        (sample_count, horizon, LOB_DEPTH_WIDTH),
        dtype=np.float32,
    )
    for sample_index in range(sample_count):
        for step_index in range(horizon):
            price = prices[sample_index, step_index, 0]
            depthprices[sample_index, step_index, :5] = price
            depthprices[sample_index, step_index, 5:10] = 10.0
            depthprices[sample_index, step_index, 10:15] = price
            depthprices[sample_index, step_index, 15:20] = 10.0
    return states, relative_states, trend_states, prices, depthprices
