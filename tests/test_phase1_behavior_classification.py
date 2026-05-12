from __future__ import annotations

import numpy as np

from src.phase1.evaluators.phase1_validation_layers.layer2_behavior_quality import (
    classify_action_motif,
    classify_market_morphology,
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
