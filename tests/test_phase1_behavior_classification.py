from __future__ import annotations

import math

import numpy as np

from src.phase1.evaluators.phase1_validation_layers.layer2_behavior_quality import (
    _code_diagnostic_status,
    classify_action_motif,
    classify_market_morphology,
    compute_behavior_quality_metrics,
    compute_distribution_by_code,
    compute_duplicate_code_pair_count,
    compute_intra_code_action_similarity,
    compute_inter_intra_separation,
    compute_lift,
)
from src.phase1.metrics import (
    Phase1BehaviorQualityPayload,
    Phase1EvaluationSnapshot,
    Phase1ValidationRuntimeConfig,
)


def _snapshot(
    *,
    prices: np.ndarray | None,
    decoded_actions: np.ndarray,
    code_ids: np.ndarray,
) -> Phase1EvaluationSnapshot:
    sample_count, horizon = decoded_actions.shape
    return Phase1EvaluationSnapshot(
        split="val",
        epoch=1,
        sample_ids=np.arange(sample_count),
        states=np.zeros((sample_count, horizon, 2), dtype=np.float32),
        prices=prices,
        demo_actions=decoded_actions,
        demo_rewards=np.zeros((sample_count, horizon), dtype=np.float64),
        decoded_actions=decoded_actions,
        decoded_logits=np.zeros((sample_count, horizon, 3), dtype=np.float32),
        code_ids=code_ids,
        z_e=np.column_stack(
            [code_ids.astype(np.float64), np.arange(sample_count, dtype=np.float64)]
        ),
        z_q=np.zeros((sample_count, 2), dtype=np.float32),
        distances=np.zeros((sample_count, 2), dtype=np.float32),
        reconstruction_loss=0.0,
        action_accuracy=1.0,
    )


def test_classify_market_morphology_uses_documented_labels() -> None:
    quiet = [[100.0, 100.01, 99.99, 100.0, 100.01, 100.0] for _ in range(20)]
    examples = [
        [100.0, 102.0, 104.0, 106.0, 108.0, 110.0],
        [100.0, 98.0, 96.0, 94.0, 92.0, 90.0],
        [100.0, 95.0, 90.0, 98.0, 106.0, 112.0],
        [100.0, 105.0, 110.0, 102.0, 94.0, 88.0],
        [100.0, 110.0, 90.0, 100.0, 110.0, 100.0],
        [100.0, 100.01, 99.99, 100.0, 100.01, 100.0],
    ]

    labels = classify_market_morphology(
        np.asarray(quiet + examples, dtype=np.float64),
        fee_rate=0.0002,
    )

    assert labels[-6:].tolist() == [
        "uptrend",
        "downtrend",
        "reversal-up",
        "reversal-down",
        "range-high-vol",
        "range-low-vol",
    ]


def test_classify_action_motif_builds_direction_entry_style_and_reversal() -> None:
    prices = np.asarray(
        [
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            [100.0, 98.0, 96.0, 97.0, 99.0, 101.0],
            [100.0, 101.0, 102.0, 101.0, 99.0, 97.0],
            [100.0, 101.0, 99.0, 102.0, 98.0, 101.0],
        ],
        dtype=np.float64,
    )
    actions = np.asarray(
        [
            [1, 1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 2],
            [1, 1, 2, 2, 2, 2],
            [1, 2, 2, 0, 0, 0],
            [1, 2, 1, 2, 1, 2],
        ],
        dtype=np.int64,
    )

    motifs = classify_action_motif(actions, prices)

    assert motifs[0] == "flat + none + mostly-flat"
    assert motifs[1] == "long + early + hold"
    assert motifs[2] == "long + middle + delayed-hold + against-recent-move"
    assert motifs[3] == "short + early + hold + long-to-short"
    assert motifs[4] == "long + early + switching + with-recent-move"


def test_distribution_and_lift_helpers_match_documented_semantics() -> None:
    values = np.asarray(["up", "up", "down", "down", "down"])
    code_ids = np.asarray([0, 0, 0, 1, 1])

    by_code = compute_distribution_by_code(values, code_ids)
    global_distribution = {"up": 2 / 5, "down": 3 / 5}
    lift = compute_lift(by_code[0], global_distribution)

    assert by_code == {
        0: {"up": 2 / 3, "down": 1 / 3},
        1: {"down": 1.0},
    }
    assert math.isclose(lift["up"], (2 / 3) / (2 / 5), rel_tol=1e-10)
    assert math.isclose(lift["down"], (1 / 3) / (3 / 5), rel_tol=1e-10)


def test_action_structure_helpers_use_code_prototypes() -> None:
    actions = np.asarray(
        [
            [2, 2],
            [2, 1],
            [0, 0],
            [0, 1],
        ],
        dtype=np.int64,
    )
    code_ids = np.asarray([0, 0, 1, 1])

    similarity = compute_intra_code_action_similarity(actions, code_ids)
    separation = compute_inter_intra_separation(actions, code_ids)

    assert math.isclose(similarity, 0.875, rel_tol=1e-12)
    assert separation > 4.0


def test_duplicate_pair_helper_counts_each_code_pair_once() -> None:
    actions = np.asarray(
        [
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
            [0, 0, 0],
        ],
        dtype=np.int64,
    )
    code_ids = np.asarray([0, 0, 1, 2])

    assert (
        compute_duplicate_code_pair_count(
            actions,
            code_ids,
            threshold=0.85,
        )
        == 1
    )


def test_code_diagnostic_status_aggregates_support_structure_and_profitability() -> None:
    assert (
        _code_diagnostic_status(
            weak_support=False,
            weak_morphology=False,
            weak_motif=False,
            weak_pair=False,
            weak_lift_nonprofitable=False,
        )
        == "pass"
    )
    assert (
        _code_diagnostic_status(
            weak_support=False,
            weak_morphology=True,
            weak_motif=False,
            weak_pair=False,
            weak_lift_nonprofitable=True,
        )
        == "weak"
    )
    assert (
        _code_diagnostic_status(
            weak_support=True,
            weak_morphology=False,
            weak_motif=False,
            weak_pair=False,
            weak_lift_nonprofitable=False,
        )
        == "bad"
    )
    assert (
        _code_diagnostic_status(
            weak_support=False,
            weak_morphology=True,
            weak_motif=True,
            weak_pair=True,
            weak_lift_nonprofitable=False,
        )
        == "bad"
    )


def test_behavior_quality_returns_typed_extra_payload() -> None:
    decoded_actions = np.asarray(
        [
            [2, 2, 2],
            [2, 2, 1],
            [0, 0, 0],
            [0, 0, 1],
        ],
        dtype=np.int64,
    )
    code_ids = np.asarray([0, 0, 1, 1], dtype=np.int64)
    prices = np.asarray(
        [
            [100.0, 101.0, 102.0],
            [100.0, 100.5, 101.0],
            [100.0, 99.0, 98.0],
            [100.0, 99.5, 99.0],
        ],
        dtype=np.float64,
    )
    snapshot = _snapshot(
        prices=prices,
        decoded_actions=decoded_actions,
        code_ids=code_ids,
    )

    computation = compute_behavior_quality_metrics(
        train_snapshot=snapshot,
        val_snapshot=snapshot,
        runtime_config=Phase1ValidationRuntimeConfig(active_code_min_occupancy=0.01),
    )

    assert isinstance(computation.extra_payload, Phase1BehaviorQualityPayload)
    assert len(computation.extra_payload["morphology_labels"]) == code_ids.size
    assert len(computation.extra_payload["motif_labels"]) == code_ids.size
    assert computation.extra_payload["active_codes"] == (0, 1)
    restored = Phase1BehaviorQualityPayload.from_dict(
        computation.extra_payload.to_dict()
    )
    assert restored == computation.extra_payload
