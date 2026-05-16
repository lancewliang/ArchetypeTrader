from __future__ import annotations

import math

import numpy as np

from src.phase1.evaluators.phase1_validation_layers.layer0_teacher_quality import (
    compute_demo_returns,
    compute_fee_sensitivity,
    compute_flat_returns,
    compute_teacher_quality_metrics,
    compute_top_removed_total_advantage,
)
from src.phase1.metrics import (
    Phase1EvaluationSnapshot,
    Phase1TeacherQualityPayload,
    Phase1ValidationRuntimeConfig,
)


def _snapshot(
    *,
    prices: np.ndarray | None,
    demo_rewards: np.ndarray,
    demo_actions: np.ndarray | None = None,
) -> Phase1EvaluationSnapshot:
    sample_count, horizon = demo_rewards.shape
    actions = (
        demo_actions
        if demo_actions is not None
        else np.ones((sample_count, horizon), dtype=np.int64)
    )
    return Phase1EvaluationSnapshot(
        split="val",
        epoch=1,
        sample_ids=np.arange(sample_count),
        states=np.zeros((sample_count, horizon, 2), dtype=np.float32),
        prices=prices,
        demo_actions=actions,
        demo_rewards=demo_rewards,
        decoded_actions=np.ones((sample_count, horizon), dtype=np.int64),
        decoded_logits=np.zeros((sample_count, horizon, 3), dtype=np.float32),
        code_ids=np.zeros(sample_count, dtype=np.int64),
        z_e=np.zeros((sample_count, 2), dtype=np.float32),
        z_q=np.zeros((sample_count, 2), dtype=np.float32),
        distances=np.zeros((sample_count, 2), dtype=np.float32),
        reconstruction_loss=0.0,
        action_accuracy=1.0,
    )


def test_layer0_returns_documented_extra_payload_and_missing_reason() -> None:
    rewards = np.asarray(
        [
            [0.10, 0.20, 0.30],
            [-0.10, 0.05, 0.00],
        ],
        dtype=np.float64,
    )
    snapshot = _snapshot(prices=None, demo_rewards=rewards)

    computation = compute_teacher_quality_metrics(
        train_snapshot=snapshot,
        val_snapshot=snapshot,
        runtime_config=Phase1ValidationRuntimeConfig(
            fee_rate=0.01,
            top_contribution_ratio=0.05,
        ),
    )

    assert isinstance(computation.extra_payload, Phase1TeacherQualityPayload)
    np.testing.assert_allclose(computation.extra_payload["dp_returns"], [0.60, -0.05])
    np.testing.assert_allclose(computation.extra_payload["flat_returns"], [0.0, 0.0])
    np.testing.assert_allclose(computation.extra_payload["advantages"], [0.60, -0.05])
    assert computation.extra_payload["missing_reason"] == "missing_prices"
    restored = Phase1TeacherQualityPayload.from_dict(
        computation.extra_payload.to_dict()
    )
    assert restored == computation.extra_payload
    assert math.isnan(computation.metrics.fee_sensitivity)
    assert math.isnan(computation.metrics.morphology_coverage)


def test_layer0_public_helpers_match_documented_basics() -> None:
    prices = np.asarray(
        [
            [100.0, 101.0, 102.0],
            [100.0, 99.0, 98.0],
        ],
        dtype=np.float64,
    )
    rewards = np.asarray(
        [
            [0.01, 0.02, 0.03],
            [-0.01, 0.01, 0.00],
        ],
        dtype=np.float64,
    )
    actions = np.asarray(
        [
            [2, 2, 2],
            [0, 0, 0],
        ],
        dtype=np.int64,
    )
    snapshot = _snapshot(prices=prices, demo_rewards=rewards, demo_actions=actions)

    np.testing.assert_allclose(compute_flat_returns(prices), [0.0, 0.0])
    np.testing.assert_allclose(compute_demo_returns(snapshot), [0.06, 0.0])
    assert math.isfinite(
        compute_fee_sensitivity(
            prices,
            actions,
            fee_rate=0.001,
            original_advantages=np.asarray([0.06, 0.01], dtype=np.float64),
        )
    )


def test_top_removed_total_advantage_keeps_small_sample_non_empty() -> None:
    assert compute_top_removed_total_advantage(np.asarray([1.5]), 0.05) == 1.5
    assert compute_top_removed_total_advantage(np.asarray([1.0, 10.0]), 1.0) == 1.0
