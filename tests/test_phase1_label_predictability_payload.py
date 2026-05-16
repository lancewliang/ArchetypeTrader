from __future__ import annotations

import numpy as np
import torch

from src.phase1.evaluators.phase1_validation_layers.layer4_label_predictability import (
    compute_label_predictability_metrics,
)
from src.phase1.metrics import (
    Phase1EvaluationSnapshot,
    Phase1LabelPredictabilityPayload,
    Phase1ValidationRuntimeConfig,
)


def _snapshot(*, labels: np.ndarray) -> Phase1EvaluationSnapshot:
    sample_count = labels.size
    horizon = 2
    states = np.zeros((sample_count, horizon, 2), dtype=np.float32)
    states[:, 0, 0] = labels.astype(np.float32)
    states[:, 0, 1] = np.arange(sample_count, dtype=np.float32) / sample_count
    return Phase1EvaluationSnapshot(
        split="val",
        epoch=1,
        sample_ids=np.arange(sample_count),
        states=states,
        prices=None,
        demo_actions=np.ones((sample_count, horizon), dtype=np.int64),
        demo_rewards=np.zeros((sample_count, horizon), dtype=np.float64),
        decoded_actions=np.ones((sample_count, horizon), dtype=np.int64),
        decoded_logits=np.zeros((sample_count, horizon, 3), dtype=np.float32),
        code_ids=labels,
        z_e=np.zeros((sample_count, 2), dtype=np.float32),
        z_q=np.zeros((sample_count, 2), dtype=np.float32),
        distances=np.zeros((sample_count, 2), dtype=np.float32),
        reconstruction_loss=0.0,
        action_accuracy=1.0,
    )


def test_label_predictability_returns_typed_payload() -> None:
    labels = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)
    snapshot = _snapshot(labels=labels)

    computation = compute_label_predictability_metrics(
        model=object(),
        train_snapshot=snapshot,
        val_snapshot=snapshot,
        runtime_config=Phase1ValidationRuntimeConfig(
            probe_epochs=2,
            random_seed=13,
        ),
        device=torch.device("cpu"),
    )

    assert isinstance(computation.extra_payload, Phase1LabelPredictabilityPayload)
    assert computation.extra_payload["probe_seed"] == 13
    assert "probe_train_accuracy" in computation.extra_payload
    assert "probe_validation_accuracy" in computation.extra_payload
    assert "probe_predictability_gap" in computation.extra_payload
    assert len(computation.extra_payload["probe_confusion_matrix"]) == 2
    restored = Phase1LabelPredictabilityPayload.from_dict(
        computation.extra_payload.to_dict()
    )
    assert restored == computation.extra_payload


def test_label_predictability_payload_serializes_confusion_matrix() -> None:
    payload = Phase1LabelPredictabilityPayload(
        probe_train_accuracy=0.8,
        probe_validation_accuracy=0.6,
        probe_predictability_gap=0.2,
        probe_confusion_matrix=((2, 1), (0, 3)),
        probe_seed=17,
    )

    restored = Phase1LabelPredictabilityPayload.from_dict(payload.to_dict())

    assert restored == payload
    assert restored["probe_confusion_matrix"] == ((2, 1), (0, 3))
