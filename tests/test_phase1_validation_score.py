from __future__ import annotations

from types import SimpleNamespace

from src.phase1.metrics import (
    Phase1ValidationScore,
    Phase1ValidationScoreWeights,
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
