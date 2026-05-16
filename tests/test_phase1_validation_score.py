from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from src.phase1.metrics import (
    CodeAssignmentSnapshot,
    Phase1BehaviorQualityPayload,
    Phase1LabelPredictabilityPayload,
    Phase1OracleProfitabilityPayload,
    Phase1PerCodeProfitability,
    Phase1TeacherQualityPayload,
    Phase1ValidationScore,
    Phase1ValidationScoreWeights,
    Phase1VQInternalPayload,
    compute_phase1_validation_score,
    get_phase1_validation_score_value,
)
from src.phase1.metrics.phase1_metric_results import Phase1ValidationResult
from tests.test_phase1_artifact_store_checkpoints import _validation_result


def test_compute_phase1_validation_score_returns_auditable_object(monkeypatch) -> None:
    import src.phase1.metrics.phase1_validation_score as score_module

    monkeypatch.setattr(score_module, "compute_teacher_quality_score", lambda _: 0.5)
    monkeypatch.setattr(score_module, "compute_reconstruction_score", lambda _: 0.6)
    monkeypatch.setattr(score_module, "compute_codebook_health_score", lambda _: 0.7)
    monkeypatch.setattr(
        score_module,
        "compute_behavior_structure_score",
        lambda _: 0.8,
    )
    monkeypatch.setattr(
        score_module,
        "compute_oracle_profitability_score",
        lambda _: 0.9,
    )
    monkeypatch.setattr(
        score_module,
        "compute_label_predictability_score",
        lambda _: 1.0,
    )

    weights = Phase1ValidationScoreWeights(
        teacher_quality=0.10,
        reconstruction=0.20,
        codebook_health=0.15,
        behavior_structure=0.20,
        oracle_profitability=0.25,
        label_predictability=0.10,
    )

    score = compute_phase1_validation_score(SimpleNamespace(), weights)

    assert isinstance(score, Phase1ValidationScore)
    assert score.total_score == 0.76
    assert score.components[0].name == "teacher_quality"
    assert score.components[0].value == 0.5
    assert score.components[0].weight == 0.10
    assert score.components[0].weighted_value == 0.05
    assert [component.name for component in score.components] == [
        "teacher_quality",
        "reconstruction",
        "codebook_health",
        "behavior_structure",
        "oracle_profitability",
        "label_predictability",
    ]


def test_phase1_validation_score_round_trips_and_exposes_total() -> None:
    payload = {
        "total_score": 0.76,
        "components": [
            {
                "name": "teacher_quality",
                "value": 0.5,
                "weight": 0.1,
                "weighted_value": 0.05,
            }
        ],
    }

    score = Phase1ValidationScore.from_dict(payload)

    assert score.to_dict() == payload
    assert get_phase1_validation_score_value(score) == 0.76
    assert get_phase1_validation_score_value(0.75) == 0.75


def test_phase1_validation_result_reads_legacy_float_score() -> None:
    payload = _validation_result().to_dict()
    payload["score"] = 0.75

    validation = Phase1ValidationResult.from_dict(payload)

    assert isinstance(validation.score, Phase1ValidationScore)
    assert validation.score.total_score == 0.75


def test_phase1_validation_result_round_trips_layer_payloads() -> None:
    current_assignment = CodeAssignmentSnapshot(
        epoch=3,
        split="val",
        sample_ids=np.asarray([1, 2, 3]),
        code_ids=np.asarray([0, 1, 0]),
        active_codes=(0, 1),
    )
    validation = replace(
        _validation_result(),
        teacher_quality_payload=Phase1TeacherQualityPayload(
            dp_returns=(0.3, -0.1),
            flat_returns=(0.0, 0.0),
            advantages=(0.3, -0.1),
            missing_reason=None,
        ),
        vq_internal_payload=Phase1VQInternalPayload(
            code_distribution=(2 / 3, 1 / 3),
            active_codes=(0, 1),
            current_assignment=current_assignment,
            assignment_churn_by_epoch={2: 0.1},
            codebook_size=2,
            codebook_size_available=True,
            code_distribution_sample_count=3,
        ),
        behavior_quality_payload=Phase1BehaviorQualityPayload(
            morphology_labels=("uptrend", "range", "uptrend"),
            motif_labels=("long_hold", "flat", "long_hold"),
            active_codes=(0, 1),
        ),
        oracle_profitability_payload=Phase1OracleProfitabilityPayload(
            per_code_profitability=(
                Phase1PerCodeProfitability(
                    code_id=0,
                    mean_advantage=0.2,
                    win_rate=0.6,
                    retention_ratio=0.7,
                    fee_drag=0.1,
                    passed=True,
                ),
            ),
            decoded_returns=(0.2, -0.1),
            dp_returns=(0.3, -0.05),
            flat_returns=(0.0, 0.0),
            random_label_returns=(0.05, -0.02),
            random_seed=7,
        ),
        label_predictability_payload=Phase1LabelPredictabilityPayload(
            probe_train_accuracy=0.7,
            probe_validation_accuracy=0.5,
            probe_predictability_gap=0.2,
            probe_confusion_matrix=((2, 1), (0, 3)),
            probe_seed=13,
        ),
    )

    payload = validation.to_dict()
    restored = Phase1ValidationResult.from_dict(payload)

    assert payload["teacher_quality_payload"]["advantages"] == [0.3, -0.1]
    assert payload["vq_internal_payload"]["active_codes"] == [0, 1]
    assert payload["behavior_quality_payload"]["motif_labels"] == [
        "long_hold",
        "flat",
        "long_hold",
    ]
    assert payload["oracle_profitability_payload"]["random_seed"] == 7
    assert payload["label_predictability_payload"]["probe_seed"] == 13
    assert isinstance(restored.teacher_quality_payload, Phase1TeacherQualityPayload)
    assert isinstance(restored.vq_internal_payload, Phase1VQInternalPayload)
    assert isinstance(restored.behavior_quality_payload, Phase1BehaviorQualityPayload)
    assert isinstance(
        restored.oracle_profitability_payload,
        Phase1OracleProfitabilityPayload,
    )
    assert isinstance(
        restored.label_predictability_payload,
        Phase1LabelPredictabilityPayload,
    )
