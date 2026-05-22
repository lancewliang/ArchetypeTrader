import numpy as np
import polars as pl

from src.phase2.phase2_selection_dataset import Phase2SelectionDatasetBuilder


def test_phase2_visible_states_include_three_previous_and_current_streams() -> None:
    horizon_dataset = _horizon_dataset(sample_count=4, horizon=5)
    label_table = pl.DataFrame(
        {
            "sample_id": [0, 1, 2, 3],
            "code_label": [2, 1, 0, 2],
        }
    )
    builder = Phase2SelectionDatasetBuilder(tsize=2)

    dataset = builder.build_from_horizon_and_labels(horizon_dataset, label_table)

    (
        previous_states,
        previous_relative_states,
        previous_trend_states,
        current_states,
        current_relative_states,
        current_trend_states,
    ) = dataset.visible_states
    states, relative_states, trend_states, *_ = horizon_dataset
    np.testing.assert_allclose(previous_states, states[:-1])
    np.testing.assert_allclose(previous_relative_states, relative_states[:-1])
    np.testing.assert_allclose(previous_trend_states, trend_states[:-1])
    np.testing.assert_allclose(current_states, states[1:, :2])
    np.testing.assert_allclose(current_relative_states, relative_states[1:, :2])
    np.testing.assert_allclose(current_trend_states, trend_states[1:, :2])


def test_phase2_tensor_dataset_starts_with_six_visible_state_columns() -> None:
    horizon_dataset = _horizon_dataset(sample_count=3, horizon=4)
    label_table = pl.DataFrame(
        {
            "sample_id": [0, 1, 2],
            "code_label": [1, 0, 1],
        }
    )
    builder = Phase2SelectionDatasetBuilder(tsize=2)
    dataset = builder.build_from_horizon_and_labels(horizon_dataset, label_table)

    tensor_dataset = builder.to_tensor_dataset(dataset)

    assert len(tensor_dataset.tensors) == 13
    assert tuple(tensor_dataset.tensors[0].shape) == (2, 4, 3)
    assert tuple(tensor_dataset.tensors[1].shape) == (2, 4, 2)
    assert tuple(tensor_dataset.tensors[2].shape) == (2, 4, 1)
    assert tuple(tensor_dataset.tensors[3].shape) == (2, 2, 3)
    assert tuple(tensor_dataset.tensors[4].shape) == (2, 2, 2)
    assert tuple(tensor_dataset.tensors[5].shape) == (2, 2, 1)


def _horizon_dataset(
    *,
    sample_count: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    states = np.arange(sample_count * horizon * 3, dtype=np.float32).reshape(
        sample_count,
        horizon,
        3,
    )
    relative_states = np.arange(
        sample_count * horizon * 2,
        dtype=np.float32,
    ).reshape(sample_count, horizon, 2)
    trend_states = np.arange(sample_count * horizon, dtype=np.float32).reshape(
        sample_count,
        horizon,
        1,
    )
    prices = np.ones((sample_count, horizon, 1), dtype=np.float32)
    depthprices = np.zeros((sample_count, horizon, 20), dtype=np.float32)
    return states, relative_states, trend_states, prices, depthprices
