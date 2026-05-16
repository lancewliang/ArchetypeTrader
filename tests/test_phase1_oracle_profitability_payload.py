from __future__ import annotations

import numpy as np
import pytest
import torch

from src.phase1.evaluators.phase1_validation_layers import (
    layer3_oracle_profitability as layer3_module,
)
from src.phase1.evaluators.phase1_validation_layers.layer3_oracle_profitability import (
    compute_oracle_profitability_metrics,
    compute_pair_profitability_matrix,
)
from src.phase1.metrics import (
    Phase1EvaluationSnapshot,
    Phase1OracleProfitabilityPayload,
    Phase1PairProfitabilityCell,
    Phase1PerCodeProfitability,
    Phase1ValidationRuntimeConfig,
)


def _snapshot() -> Phase1EvaluationSnapshot:
    sample_count = 2
    horizon = 3
    return Phase1EvaluationSnapshot(
        split="val",
        epoch=1,
        sample_ids=np.arange(sample_count),
        states=np.zeros((sample_count, horizon, 2), dtype=np.float32),
        prices=None,
        demo_actions=np.ones((sample_count, horizon), dtype=np.int64),
        demo_rewards=np.zeros((sample_count, horizon), dtype=np.float64),
        decoded_actions=np.ones((sample_count, horizon), dtype=np.int64),
        decoded_logits=np.zeros((sample_count, horizon, 3), dtype=np.float32),
        code_ids=np.asarray([0, 1], dtype=np.int64),
        z_e=np.zeros((sample_count, 2), dtype=np.float32),
        z_q=np.zeros((sample_count, 2), dtype=np.float32),
        distances=np.zeros((sample_count, 2), dtype=np.float32),
        reconstruction_loss=0.0,
        action_accuracy=1.0,
    )


def test_oracle_profitability_missing_prices_returns_typed_payload() -> None:
    computation = compute_oracle_profitability_metrics(
        model=object(),
        val_snapshot=_snapshot(),
        runtime_config=Phase1ValidationRuntimeConfig(random_seed=7),
        device=torch.device("cpu"),
    )

    assert isinstance(computation.extra_payload, Phase1OracleProfitabilityPayload)
    assert computation.extra_payload["per_code_profitability"] == ()
    assert len(computation.extra_payload["decoded_returns"]) == 2
    assert len(computation.extra_payload["dp_returns"]) == 2
    assert len(computation.extra_payload["flat_returns"]) == 2
    assert len(computation.extra_payload["random_label_returns"]) == 2
    assert computation.extra_payload["random_seed"] == 7


def test_oracle_profitability_payload_serializes_per_code_profitability() -> None:
    per_code = (
        Phase1PerCodeProfitability(
            code_id=3,
            mean_advantage=0.2,
            win_rate=0.75,
            retention_ratio=0.8,
            fee_drag=0.1,
            passed=True,
        ),
    )
    payload = Phase1OracleProfitabilityPayload(
        per_code_profitability=per_code,
        decoded_returns=(0.1, 0.2),
        dp_returns=(0.2, 0.3),
        flat_returns=(0.0, 0.0),
        random_label_returns=(-0.1, 0.0),
        random_seed=11,
        pair_profitability_matrix=(
            Phase1PairProfitabilityCell(
                morphology="uptrend",
                motif="long + early + hold",
                support=2,
                mean_decoded_advantage=0.15,
                decoded_win_rate=1.0,
                retention_ratio=0.6,
                fee_drag=0.1,
            ),
        ),
    )

    restored = Phase1OracleProfitabilityPayload.from_dict(payload.to_dict())

    assert restored == payload
    assert restored["per_code_profitability"][0].code_id == 3
    assert restored["pair_profitability_matrix"][0].morphology == "uptrend"


def test_oracle_profitability_metrics_outputs_pair_matrix(monkeypatch) -> None:
    sample_count = 3
    horizon = 4
    snapshot = Phase1EvaluationSnapshot(
        split="val",
        epoch=1,
        sample_ids=np.arange(sample_count),
        states=np.zeros((sample_count, horizon, 2), dtype=np.float32),
        prices=np.asarray(
            [
                [100.0, 101.0, 102.0, 103.0],
                [100.0, 101.0, 103.0, 104.0],
                [100.0, 99.0, 98.0, 97.0],
            ],
            dtype=np.float64,
        ),
        demo_actions=np.ones((sample_count, horizon), dtype=np.int64),
        demo_rewards=np.zeros((sample_count, horizon), dtype=np.float64),
        decoded_actions=np.asarray(
            [
                [2, 2, 2, 2],
                [2, 2, 2, 2],
                [0, 0, 0, 0],
            ],
            dtype=np.int64,
        ),
        decoded_logits=np.zeros((sample_count, horizon, 3), dtype=np.float32),
        code_ids=np.asarray([0, 0, 1], dtype=np.int64),
        z_e=np.zeros((sample_count, 2), dtype=np.float32),
        z_q=np.zeros((sample_count, 2), dtype=np.float32),
        distances=np.zeros((sample_count, 2), dtype=np.float32),
        reconstruction_loss=0.0,
        action_accuracy=1.0,
    )
    monkeypatch.setattr(
        layer3_module,
        "compute_random_label_returns",
        lambda **_: np.zeros(sample_count, dtype=np.float64),
    )

    computation = compute_oracle_profitability_metrics(
        model=object(),
        val_snapshot=snapshot,
        runtime_config=Phase1ValidationRuntimeConfig(
            fee_rate=0.0,
            random_seed=7,
        ),
        device=torch.device("cpu"),
    )

    assert isinstance(computation.extra_payload, Phase1OracleProfitabilityPayload)
    assert computation.extra_payload.pair_profitability_matrix


def test_pair_profitability_matrix_aggregates_morphology_motif_cells() -> None:
    cells = compute_pair_profitability_matrix(
        morphologies=np.asarray(["uptrend", "uptrend", "downtrend"], dtype=object),
        motifs=np.asarray(["long", "long", "short"], dtype=object),
        decoded_advantage=np.asarray([0.10, 0.20, -0.05], dtype=np.float64),
        decoded_returns=np.asarray([0.10, 0.20, -0.05], dtype=np.float64),
        flat_returns=np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
        dp_advantage=np.asarray([0.20, 0.30, 0.10], dtype=np.float64),
        decoded_gross_returns=np.asarray([0.12, 0.22, -0.03], dtype=np.float64),
        decoded_fees=np.asarray([0.02, 0.02, 0.02], dtype=np.float64),
    )

    assert [(cell.morphology, cell.motif, cell.support) for cell in cells] == [
        ("downtrend", "short", 1),
        ("uptrend", "long", 2),
    ]
    assert cells[0].mean_decoded_advantage == pytest.approx(-0.05)
    assert cells[0].decoded_win_rate == pytest.approx(0.0)
    assert cells[0].retention_ratio == pytest.approx(-0.5)
    assert cells[0].fee_drag == float("inf")
    assert cells[1].mean_decoded_advantage == pytest.approx(0.15)
    assert cells[1].decoded_win_rate == pytest.approx(1.0)
    assert cells[1].retention_ratio == pytest.approx(0.6)
    assert cells[1].fee_drag == pytest.approx(0.04 / 0.34)
