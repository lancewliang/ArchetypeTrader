from __future__ import annotations

import numpy as np
import torch

from src.phase1.evaluators.phase1_validation_layers.layer3_oracle_profitability import (
    compute_oracle_profitability_metrics,
)
from src.phase1.metrics import (
    Phase1EvaluationSnapshot,
    Phase1OracleProfitabilityPayload,
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
    )

    restored = Phase1OracleProfitabilityPayload.from_dict(payload.to_dict())

    assert restored == payload
    assert restored["per_code_profitability"][0].code_id == 3
