from __future__ import annotations

from types import SimpleNamespace

from src.phase1.checkpoint import Phase1CheckpointSelector
from src.phase1.metrics import Phase1TieBreakerMetrics


def _candidate(
    *,
    epoch: int,
    score: float | None,
    passed: bool = True,
    failed_layers: tuple[str, ...] = (),
    risk_adjusted_return: float = 1.0,
):
    validation = SimpleNamespace(
        checkpoint_id=f"vq_epoch_{epoch:04d}",
        passed=passed,
        score=score,
        failed_layers=failed_layers,
        tie_breaker_metrics=Phase1TieBreakerMetrics(
            risk_adjusted_return=risk_adjusted_return,
            probe_top3_accuracy=0.8,
            retention_ratio=0.75,
            active_code_ratio=0.7,
            max_code_occupancy=0.2,
            reconstruction_loss=0.1,
        ),
    )
    return SimpleNamespace(
        stage="vq",
        epoch=epoch,
        codebook_validation=validation,
    )


def test_select_best_filters_failed_and_chooses_highest_score() -> None:
    selector = Phase1CheckpointSelector()
    result = selector.select_best(
        [
            _candidate(epoch=1, score=None, passed=False, failed_layers=("vq",)),
            _candidate(epoch=2, score=0.72),
            _candidate(epoch=3, score=0.81),
        ]
    )

    assert result.has_selection
    assert result.selected_epoch == 3
    assert result.selected_score == 0.81
    assert result.candidate_count == 3
    assert result.eligible_count == 2
    assert [item.reason for item in result.rejected] == [
        "validation_failed",
        "lower_score",
    ]


def test_select_best_uses_tie_breaker_when_scores_are_close() -> None:
    selector = Phase1CheckpointSelector()
    result = selector.select_best(
        [
            _candidate(epoch=4, score=0.80, risk_adjusted_return=1.0),
            _candidate(epoch=5, score=0.81, risk_adjusted_return=1.5),
        ]
    )

    assert result.selected_epoch == 5
    assert result.reason == "selected_tie_breaker"
    assert result.rejected[0].reason == "tie_breaker_loser"
