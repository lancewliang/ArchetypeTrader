from types import SimpleNamespace

from src.phase1.checkpoint.phase1_checkpoint import Phase1Checkpoint
from src.phase1.checkpoint.phase1_checkpoint_selector import Phase1CheckpointSelector
from src.phase1.metrics.phase1_validation_score import Phase1ValidationScore
from src.phase1.phase1_artifact_store import Phase1ArtifactStore


def _tie_breaker(
    *,
    risk_adjusted_return: float,
    probe_top3_accuracy: float = 0.0,
    retention_ratio: float = 0.0,
    active_code_ratio: float = 1.0,
    max_code_occupancy: float = 0.1,
    reconstruction_loss: float = 0.1,
) -> SimpleNamespace:
    return SimpleNamespace(
        risk_adjusted_return=risk_adjusted_return,
        probe_top3_accuracy=probe_top3_accuracy,
        retention_ratio=retention_ratio,
        active_code_ratio=active_code_ratio,
        max_code_occupancy=max_code_occupancy,
        reconstruction_loss=reconstruction_loss,
    )


def _validation_checkpoint(
    *,
    epoch: int,
    score: float,
    risk_adjusted_return: float,
) -> SimpleNamespace:
    return SimpleNamespace(
        stage="vq",
        epoch=epoch,
        codebook_validation=SimpleNamespace(
            checkpoint_id=f"vq_epoch_{epoch:04d}",
            passed=True,
            score=Phase1ValidationScore.from_float(score),
            failed_layers=(),
            tie_breaker_metrics=_tie_breaker(
                risk_adjusted_return=risk_adjusted_return
            ),
        ),
    )


def test_phase1_checkpoint_selector_compares_score_objects() -> None:
    selector = Phase1CheckpointSelector()
    lower_risk = _validation_checkpoint(
        epoch=200,
        score=0.7847,
        risk_adjusted_return=0.48,
    )
    higher_risk = _validation_checkpoint(
        epoch=220,
        score=0.7834,
        risk_adjusted_return=0.52,
    )

    result = selector.select_best([lower_risk, higher_risk])

    assert result.has_selection
    assert result.selected_epoch == 220
    assert result.reason == "selected_tie_breaker"


def test_phase1_checkpoint_payload_round_trip_without_metrics(tmp_path) -> None:
    store = Phase1ArtifactStore(
        pair="FU",
        batchid="unit_test",
        artifacts_root=tmp_path,
    )
    store.initialize_phase1_artifact_dirs()
    checkpoint = Phase1Checkpoint(
        stage="vq",
        epoch=200,
        is_best=False,
        config={"pair": "FU"},
        model_state_dict={"weight": 1},
        optimizer_state_dict={"state": 2},
    )

    store.save_best_checkpoint(checkpoint)
    loaded = store.load_best_checkpoint()

    assert loaded is not None
    assert loaded.stage == "vq"
    assert loaded.epoch == 200
    assert loaded.is_best
    assert loaded.config == {"pair": "FU"}
