import math

import numpy as np
import pytest

from src.phase1.evaluators.phase1_validation_layers.layer1_vq_internal import (
    classify_main_direction,
    compute_action_accuracy,
    compute_assignment_churn,
    compute_code_distribution,
    compute_code_lifetime_pass_ratio,
    compute_first_trade_t,
    compute_nearest_second_margin,
    compute_normalized_perplexity,
    compute_vq_internal_metrics,
)
from src.phase1.metrics import (
    CodeAssignmentSnapshot,
    Phase1EvaluationSnapshot,
    Phase1VQInternalMetrics,
    Phase1ValidationRuntimeConfig,
)


def _assignment(
    *,
    epoch: int,
    code_ids: list[int],
    active_codes: tuple[int, ...] = (),
    split: str = "val",
) -> CodeAssignmentSnapshot:
    return CodeAssignmentSnapshot(
        epoch=epoch,
        split=split,
        sample_ids=np.arange(len(code_ids)),
        code_ids=np.asarray(code_ids, dtype=np.int64),
        active_codes=active_codes,
    )


def test_assignment_snapshot_validates_sample_code_alignment() -> None:
    with pytest.raises(ValueError):
        CodeAssignmentSnapshot(
            epoch=1,
            split="val",
            sample_ids=np.asarray([0, 1]),
            code_ids=np.asarray([0]),
            active_codes=(),
        )

    with pytest.raises(ValueError):
        CodeAssignmentSnapshot(
            epoch=1,
            split="val",
            sample_ids=np.asarray([0, 0]),
            code_ids=np.asarray([0, 1]),
            active_codes=(),
        )


def test_action_accuracy_validates_shape_and_counts_all_timesteps() -> None:
    assert compute_action_accuracy(
        np.asarray([[1, 2], [0, 1]]),
        np.asarray([[1, 0], [0, 1]]),
    ) == 0.75

    with pytest.raises(ValueError):
        compute_action_accuracy(np.ones((2, 2)), np.ones((2, 3)))


def test_code_distribution_and_perplexity_follow_documented_api() -> None:
    distribution = compute_code_distribution(np.asarray([0, 0, 1, 3]), 4)

    np.testing.assert_allclose(distribution, [0.5, 0.25, 0.0, 0.25])
    assert 0.0 < compute_normalized_perplexity(distribution) < 1.0
    assert math.isnan(compute_normalized_perplexity(np.zeros(4)))

    with pytest.raises(ValueError):
        compute_code_distribution(np.asarray([0, 4]), 4)


def test_assignment_churn_uses_adjacent_snapshots_not_current_vs_all_history() -> None:
    history = [
        _assignment(epoch=1, code_ids=[0, 0]),
        _assignment(epoch=2, code_ids=[1, 1]),
    ]
    current = _assignment(epoch=3, code_ids=[0, 0])

    assert compute_assignment_churn(current, history, window=2) == 1.0


def test_assignment_churn_requires_complete_recent_window() -> None:
    history = [
        _assignment(epoch=1, code_ids=[0, 0]),
    ]
    current = _assignment(epoch=2, code_ids=[1, 1])

    assert math.isnan(compute_assignment_churn(current, history, window=2))


def test_code_lifetime_counts_current_epoch_plus_consecutive_history() -> None:
    history = [
        _assignment(epoch=1, code_ids=[0], active_codes=(0, 1)),
        _assignment(epoch=2, code_ids=[0], active_codes=(0,)),
    ]

    assert compute_code_lifetime_pass_ratio((0, 1), history, 3) == 0.5


def test_code_lifetime_rejects_mixed_split_history() -> None:
    history = [
        _assignment(epoch=1, code_ids=[0], active_codes=(0,), split="train"),
        _assignment(epoch=2, code_ids=[0], active_codes=(0,), split="val"),
    ]

    with pytest.raises(ValueError):
        compute_code_lifetime_pass_ratio((0,), history, 2)


def test_nearest_second_margin_returns_per_sample_values() -> None:
    margins = compute_nearest_second_margin(
        np.asarray(
            [
                [1.0, 1.5, 3.0],
                [2.0, 5.0, 4.0],
            ]
        )
    )

    np.testing.assert_allclose(margins, [0.5, 1.0])


def test_direction_and_first_trade_follow_documented_position_semantics() -> None:
    actions = np.asarray(
        [
            [1, 2, 2, 0],  # long has majority over short
            [1, 0, 2, 1],  # equal long/short -> mixed
            [1, 1, 1, 1],  # flat
        ]
    )

    np.testing.assert_array_equal(
        classify_main_direction(actions),
        np.asarray(["long", "mixed", "flat"], dtype=object),
    )
    np.testing.assert_array_equal(compute_first_trade_t(actions), [1, 1, -1])


def test_vq_internal_metrics_use_raw_timestep_and_turnover_errors() -> None:
    train_snapshot = Phase1EvaluationSnapshot(
        split="train",
        epoch=3,
        sample_ids=np.arange(2),
        states=np.zeros((2, 4, 2), dtype=np.float32),
        prices=np.ones((2, 4), dtype=np.float32),
        demo_actions=np.ones((2, 4), dtype=np.int64),
        demo_rewards=np.zeros((2, 4), dtype=np.float32),
        decoded_actions=np.ones((2, 4), dtype=np.int64),
        decoded_logits=np.zeros((2, 4, 3), dtype=np.float32),
        code_ids=np.asarray([0, 1], dtype=np.int64),
        z_e=np.zeros((2, 2), dtype=np.float32),
        z_q=np.zeros((2, 2), dtype=np.float32),
        distances=np.asarray([[1.0, 2.0], [2.0, 1.0]], dtype=np.float32),
        reconstruction_loss=1.0,
        action_accuracy=1.0,
    )
    val_snapshot = Phase1EvaluationSnapshot(
        split="val",
        epoch=3,
        sample_ids=np.arange(2),
        states=np.zeros((2, 4, 2), dtype=np.float32),
        prices=np.ones((2, 4), dtype=np.float32),
        demo_actions=np.asarray([[1, 1, 2, 2], [1, 1, 1, 1]], dtype=np.int64),
        demo_rewards=np.zeros((2, 4), dtype=np.float32),
        decoded_actions=np.asarray([[1, 2, 2, 2], [1, 0, 0, 0]], dtype=np.int64),
        decoded_logits=np.zeros((2, 4, 3), dtype=np.float32),
        code_ids=np.asarray([0, 1], dtype=np.int64),
        z_e=np.zeros((2, 2), dtype=np.float32),
        z_q=np.zeros((2, 2), dtype=np.float32),
        distances=np.asarray([[1.0, 2.0], [2.0, 1.0]], dtype=np.float32),
        reconstruction_loss=1.0,
        action_accuracy=0.5,
    )

    computation = compute_vq_internal_metrics(
        train_snapshot=train_snapshot,
        val_snapshot=val_snapshot,
        assignment_history=(),
        runtime_config=Phase1ValidationRuntimeConfig(),
    )

    assert computation.metrics.entry_timing_error_median == 1.0
    assert computation.metrics.decoder_turnover_error == 0.5
    assert computation.extra_payload["assignment_churn_by_epoch"] == {}


def test_vq_internal_metrics_do_not_infer_codebook_size_from_used_codes() -> None:
    train_snapshot = Phase1EvaluationSnapshot(
        split="train",
        epoch=1,
        sample_ids=np.arange(2),
        states=np.zeros((2, 3, 2), dtype=np.float32),
        prices=np.ones((2, 3), dtype=np.float32),
        demo_actions=np.ones((2, 3), dtype=np.int64),
        demo_rewards=np.zeros((2, 3), dtype=np.float32),
        decoded_actions=np.ones((2, 3), dtype=np.int64),
        decoded_logits=np.zeros((2, 3, 3), dtype=np.float32),
        code_ids=np.asarray([0, 1], dtype=np.int64),
        z_e=np.zeros((2, 2), dtype=np.float32),
        z_q=np.zeros((2, 2), dtype=np.float32),
        distances=np.zeros((2, 0), dtype=np.float32),
        reconstruction_loss=1.0,
        action_accuracy=1.0,
    )
    val_snapshot = Phase1EvaluationSnapshot(
        split="val",
        epoch=1,
        sample_ids=np.arange(2),
        states=np.zeros((2, 3, 2), dtype=np.float32),
        prices=np.ones((2, 3), dtype=np.float32),
        demo_actions=np.ones((2, 3), dtype=np.int64),
        demo_rewards=np.zeros((2, 3), dtype=np.float32),
        decoded_actions=np.ones((2, 3), dtype=np.int64),
        decoded_logits=np.zeros((2, 3, 3), dtype=np.float32),
        code_ids=np.asarray([0, 1], dtype=np.int64),
        z_e=np.zeros((2, 2), dtype=np.float32),
        z_q=np.zeros((2, 2), dtype=np.float32),
        distances=np.zeros((2, 0), dtype=np.float32),
        reconstruction_loss=1.0,
        action_accuracy=1.0,
    )

    computation = compute_vq_internal_metrics(
        train_snapshot=train_snapshot,
        val_snapshot=val_snapshot,
        assignment_history=(),
        runtime_config=Phase1ValidationRuntimeConfig(),
    )

    assert computation.metrics.code_distribution == ()
    assert computation.metrics.active_codes == ()
    assert computation.extra_payload["codebook_size_available"] is False
    assert math.isnan(computation.metrics.active_code_ratio)
    assert math.isnan(computation.metrics.max_code_occupancy)
    assert math.isnan(computation.metrics.normalized_code_perplexity)
    assert math.isnan(computation.metrics.dead_code_ratio)


def test_vq_internal_metrics_mark_occupancy_unavailable_without_samples() -> None:
    train_snapshot = Phase1EvaluationSnapshot(
        split="train",
        epoch=1,
        sample_ids=np.arange(1),
        states=np.zeros((1, 3, 2), dtype=np.float32),
        prices=np.ones((1, 3), dtype=np.float32),
        demo_actions=np.ones((1, 3), dtype=np.int64),
        demo_rewards=np.zeros((1, 3), dtype=np.float32),
        decoded_actions=np.ones((1, 3), dtype=np.int64),
        decoded_logits=np.zeros((1, 3, 3), dtype=np.float32),
        code_ids=np.asarray([0], dtype=np.int64),
        z_e=np.zeros((1, 2), dtype=np.float32),
        z_q=np.zeros((1, 2), dtype=np.float32),
        distances=np.zeros((1, 4), dtype=np.float32),
        reconstruction_loss=1.0,
        action_accuracy=1.0,
    )
    val_snapshot = Phase1EvaluationSnapshot(
        split="val",
        epoch=1,
        sample_ids=np.arange(0),
        states=np.zeros((0, 3, 2), dtype=np.float32),
        prices=np.ones((0, 3), dtype=np.float32),
        demo_actions=np.ones((0, 3), dtype=np.int64),
        demo_rewards=np.zeros((0, 3), dtype=np.float32),
        decoded_actions=np.ones((0, 3), dtype=np.int64),
        decoded_logits=np.zeros((0, 3, 3), dtype=np.float32),
        code_ids=np.asarray([], dtype=np.int64),
        z_e=np.zeros((0, 2), dtype=np.float32),
        z_q=np.zeros((0, 2), dtype=np.float32),
        distances=np.zeros((0, 4), dtype=np.float32),
        reconstruction_loss=1.0,
        action_accuracy=math.nan,
    )

    computation = compute_vq_internal_metrics(
        train_snapshot=train_snapshot,
        val_snapshot=val_snapshot,
        assignment_history=(),
        runtime_config=Phase1ValidationRuntimeConfig(),
    )

    np.testing.assert_allclose(
        computation.metrics.code_distribution,
        [0.0, 0.0, 0.0, 0.0],
    )
    assert computation.metrics.active_codes == ()
    assert computation.extra_payload["code_distribution_sample_count"] == 0
    assert math.isnan(computation.metrics.active_code_ratio)
    assert math.isnan(computation.metrics.max_code_occupancy)
    assert math.isnan(computation.metrics.normalized_code_perplexity)
    assert math.isnan(computation.metrics.dead_code_ratio)


def test_vq_internal_metrics_prefer_configured_codebook_size() -> None:
    train_snapshot = Phase1EvaluationSnapshot(
        split="train",
        epoch=1,
        sample_ids=np.arange(2),
        states=np.zeros((2, 3, 2), dtype=np.float32),
        prices=np.ones((2, 3), dtype=np.float32),
        demo_actions=np.ones((2, 3), dtype=np.int64),
        demo_rewards=np.zeros((2, 3), dtype=np.float32),
        decoded_actions=np.ones((2, 3), dtype=np.int64),
        decoded_logits=np.zeros((2, 3, 3), dtype=np.float32),
        code_ids=np.asarray([0, 1], dtype=np.int64),
        z_e=np.zeros((2, 2), dtype=np.float32),
        z_q=np.zeros((2, 2), dtype=np.float32),
        distances=np.zeros((2, 0), dtype=np.float32),
        reconstruction_loss=1.0,
        action_accuracy=1.0,
    )
    val_snapshot = Phase1EvaluationSnapshot(
        split="val",
        epoch=1,
        sample_ids=np.arange(2),
        states=np.zeros((2, 3, 2), dtype=np.float32),
        prices=np.ones((2, 3), dtype=np.float32),
        demo_actions=np.ones((2, 3), dtype=np.int64),
        demo_rewards=np.zeros((2, 3), dtype=np.float32),
        decoded_actions=np.ones((2, 3), dtype=np.int64),
        decoded_logits=np.zeros((2, 3, 3), dtype=np.float32),
        code_ids=np.asarray([0, 1], dtype=np.int64),
        z_e=np.zeros((2, 2), dtype=np.float32),
        z_q=np.zeros((2, 2), dtype=np.float32),
        distances=np.zeros((2, 0), dtype=np.float32),
        reconstruction_loss=1.0,
        action_accuracy=1.0,
    )

    computation = compute_vq_internal_metrics(
        train_snapshot=train_snapshot,
        val_snapshot=val_snapshot,
        assignment_history=(),
        runtime_config=Phase1ValidationRuntimeConfig(codebook_size=4),
    )

    np.testing.assert_allclose(
        computation.metrics.code_distribution,
        [0.5, 0.5, 0.0, 0.0],
    )
    assert computation.metrics.active_codes == (0, 1)
    assert computation.metrics.active_code_ratio == 0.5


def test_vq_internal_metrics_round_trip_code_usage_fields() -> None:
    metrics = Phase1VQInternalMetrics(
        validation_action_accuracy=0.9,
        reconstruction_loss_gap=1.1,
        active_code_ratio=0.5,
        max_code_occupancy=0.5,
        normalized_code_perplexity=0.7,
        dead_code_ratio=0.0,
        assignment_churn_recent_mean=0.1,
        code_lifetime_pass_ratio=1.0,
        quantization_distance=0.2,
        nearest_second_margin_median=0.3,
        decoder_turnover_error=0.4,
        entry_timing_error_median=1.0,
        direction_accuracy=0.95,
        code_distribution=(0.5, 0.5, 0.0),
        active_codes=(0, 1),
    )

    restored = Phase1VQInternalMetrics.from_dict(metrics.to_dict())

    assert restored.code_distribution == (0.5, 0.5, 0.0)
    assert restored.active_codes == (0, 1)
