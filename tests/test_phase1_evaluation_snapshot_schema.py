from __future__ import annotations

import numpy as np
import pytest

from src.phase1.metrics.phase1_validation_data_schema import Phase1EvaluationSnapshot


def _snapshot_kwargs() -> dict[str, object]:
    return {
        "split": "val",
        "epoch": 1,
        "sample_ids": np.arange(2),
        "states": np.zeros((2, 3, 4), dtype=np.float32),
        "prices": np.zeros((2, 3), dtype=np.float32),
        "demo_actions": np.zeros((2, 3), dtype=np.int64),
        "demo_rewards": np.zeros((2, 3), dtype=np.float32),
        "decoded_actions": np.zeros((2, 3), dtype=np.int64),
        "decoded_logits": np.zeros((2, 3, 3), dtype=np.float32),
        "code_ids": np.zeros(2, dtype=np.int64),
        "z_e": np.zeros((2, 5), dtype=np.float32),
        "z_q": np.zeros((2, 5), dtype=np.float32),
        "distances": np.zeros((2, 7), dtype=np.float32),
        "reconstruction_loss": 0.0,
        "action_accuracy": 1.0,
    }


def test_snapshot_accepts_fixed_prices_and_rewards_shapes() -> None:
    snapshot = Phase1EvaluationSnapshot(**_snapshot_kwargs())

    assert snapshot.prices is not None
    assert snapshot.prices.shape == (2, 3)
    assert snapshot.demo_rewards.shape == (2, 3)


@pytest.mark.parametrize("field_name", ["prices", "demo_rewards"])
def test_snapshot_rejects_trailing_singleton_horizon_fields(field_name: str) -> None:
    kwargs = _snapshot_kwargs()
    kwargs[field_name] = np.zeros((2, 3, 1), dtype=np.float32)

    with pytest.raises(ValueError, match=field_name):
        Phase1EvaluationSnapshot(**kwargs)
