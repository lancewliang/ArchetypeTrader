from __future__ import annotations

import hashlib

import torch

from src.phase1.checkpoint import Phase1Checkpoint
from src.phase1.metrics import (
    Phase1BehaviorQualityMetrics,
    Phase1LabelPredictabilityMetrics,
    Phase1LayerResult,
    Phase1MetricResult,
    Phase1OracleProfitabilityMetrics,
    Phase1TeacherQualityMetrics,
    Phase1TieBreakerMetrics,
    Phase1ValidationMetrics,
    Phase1ValidationResult,
    Phase1VQInternalMetrics,
)
from src.phase1.phase1_artifact_store import Phase1ArtifactStore


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as payload_file:
        for chunk in iter(lambda: payload_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint(is_best: bool = False) -> Phase1Checkpoint:
    return Phase1Checkpoint(
        stage="vq",
        epoch=3,
        is_best=is_best,
        config={"learning_rate": 0.001},
        model_state_dict={"weight": torch.tensor([1.0, 2.0])},
        optimizer_state_dict={"step": 3},
        metrics={},
    )


def _validation_result() -> Phase1ValidationResult:
    return Phase1ValidationResult(
        checkpoint_id="vq_epoch_0003",
        stage="vq",
        epoch=3,
        passed=True,
        score=0.75,
        failed_layers=(),
        layers=(
            Phase1LayerResult(
                layer_id=1,
                name="vq_internal",
                passed=True,
                metrics=(
                    Phase1MetricResult(
                        name="validation_action_accuracy",
                        value=0.9,
                        threshold=">= 0.85",
                        severity="pass",
                        passed=True,
                        layer="vq_internal",
                    ),
                ),
            ),
        ),
        metrics=Phase1ValidationMetrics(
            teacher_quality=Phase1TeacherQualityMetrics(
                dp_advantage_vs_flat=1.0,
                dp_win_rate_vs_flat=0.6,
                near_zero_opportunity_ratio=0.1,
                fee_sensitivity=0.9,
                morphology_coverage=0.8,
                dp_return_concentration_after_top5_removed=0.7,
            ),
            vq_internal=Phase1VQInternalMetrics(
                validation_action_accuracy=0.9,
                reconstruction_loss_gap=1.1,
                active_code_ratio=0.8,
                max_code_occupancy=0.2,
                normalized_code_perplexity=0.7,
                dead_code_ratio=0.1,
                assignment_churn_recent_mean=0.05,
                code_lifetime_pass_ratio=1.0,
                quantization_distance=0.2,
                nearest_second_margin_median=0.3,
                decoder_turnover_error=0.05,
                entry_timing_error_median=1.0,
                direction_accuracy=0.9,
            ),
            behavior_quality=Phase1BehaviorQualityMetrics(
                weak_support_code_ratio=0.0,
                weak_morphology_code_ratio=0.0,
                weak_motif_code_ratio=0.0,
                weak_pair_code_ratio=0.0,
                weak_lift_nonprofitable_code_ratio=0.0,
                intra_code_action_similarity=0.8,
                inter_intra_separation=1.2,
                latent_silhouette_score=0.3,
                duplicate_code_pair_count=0,
                profitable_code_coverage=1.0,
            ),
            oracle_profitability=Phase1OracleProfitabilityMetrics(
                mean_decoded_advantage_vs_flat=1.0,
                decoded_win_rate_vs_flat=0.6,
                mean_advantage_vs_random_label=0.5,
                random_label_relative_lift=0.2,
                retention_ratio=0.8,
                downside_control=0.7,
                risk_adjusted_return=1.1,
                top_5_contribution=0.2,
                trimmed_decoded_advantage=0.8,
                fee_drag=0.1,
                turnover_return_correlation=0.2,
                bad_code_ratio=0.0,
                dominant_pair_positive_ratio=0.8,
            ),
            label_predictability=Phase1LabelPredictabilityMetrics(
                probe_top1_accuracy=0.5,
                probe_top3_accuracy=0.8,
                probe_balanced_accuracy=0.45,
                label_entropy_given_morphology=0.7,
                mutual_information_lift=1.5,
                probe_return_retention=0.7,
            ),
        ),
        code_diagnostics=(),
        drift_diagnostics={},
        tie_breaker_metrics=Phase1TieBreakerMetrics(
            risk_adjusted_return=1.1,
            probe_top3_accuracy=0.8,
            retention_ratio=0.8,
            active_code_ratio=0.8,
            max_code_occupancy=0.2,
            reconstruction_loss=0.1,
        ),
    )


def test_save_and_load_phase1_checkpoint_by_epoch(tmp_path) -> None:
    store = Phase1ArtifactStore(
        pair="BTC",
        batchid="batch_001",
        artifacts_root=tmp_path,
    )
    store.initialize_phase1_artifact_dirs()

    store.save_phase1_checkpoint(
        stage="vq",
        epoch=3,
        config={"learning_rate": 0.001},
        model_state_dict={"weight": torch.tensor([1.0, 2.0])},
        optimizer_state_dict={"step": 3},
        metrics={},
    )

    checkpoint_path = store.artifact_paths["checkpoints"] / "vq_epoch_0003.pt"
    loaded = store.load_phase1_checkpoint(stage="vq", epoch=3)

    assert checkpoint_path.exists()
    assert "last_checkpoint" not in store.artifact_paths
    assert loaded.stage == "vq"
    assert loaded.epoch == 3
    assert loaded.is_best is False
    assert loaded.metrics == {}
    assert torch.equal(loaded.model_state_dict["weight"], torch.tensor([1.0, 2.0]))

    sidecar_path = checkpoint_path.with_suffix(".pt.sha256")
    assert sidecar_path.read_text(encoding="utf-8").startswith(_sha256(checkpoint_path))


def test_save_best_checkpoint_marks_payload_as_best(tmp_path) -> None:
    store = Phase1ArtifactStore(
        pair="BTC",
        batchid="batch_001",
        artifacts_root=tmp_path,
    )
    store.initialize_phase1_artifact_dirs()

    checkpoint = _checkpoint(is_best=False)
    store.save_best_checkpoint(checkpoint)

    loaded = store.load_phase1_checkpoint(best=True)
    best_checkpoint_path = store.artifact_paths["best_checkpoint"]

    assert checkpoint.is_best is False
    assert loaded.is_best is True
    assert loaded.stage == "vq"
    assert loaded.epoch == 3
    assert best_checkpoint_path.exists()
    assert best_checkpoint_path.with_suffix(".pt.sha256").exists()


def test_save_and_load_phase1_validation_result_by_epoch(tmp_path) -> None:
    store = Phase1ArtifactStore(
        pair="BTC",
        batchid="batch_001",
        artifacts_root=tmp_path,
    )
    store.initialize_phase1_artifact_dirs()

    validation_result = _validation_result()
    path = store.save_phase1_validation_result(validation_result)

    loaded = store.load_phase1_validation_result(stage="vq", epoch=3)

    assert path == store.artifact_paths["validation_results"] / (
        "vq_epoch_0003_validation.json"
    )
    assert path.exists()
    assert path.with_suffix(".json.sha256").exists()
    assert "latest_validation_result" not in store.artifact_paths
    assert loaded.checkpoint_id == "vq_epoch_0003"
    assert loaded.metrics.vq_internal.validation_action_accuracy == 0.9


def test_save_and_load_phase1_epoch_metrics_by_epoch(tmp_path) -> None:
    store = Phase1ArtifactStore(
        pair="BTC",
        batchid="batch_001",
        artifacts_root=tmp_path,
    )
    store.initialize_phase1_artifact_dirs()

    path = store.save_phase1_epoch_metrics(
        stage="vq",
        epoch=3,
        metrics={
            "train": {"total_loss": 0.5},
            "val": {"total_loss": 0.25},
        },
    )

    loaded = store.load_phase1_epoch_metrics(stage="vq", epoch=3)

    assert path == store.artifact_paths["metrics"] / "vq_epoch_0003_metrics.json"
    assert path.exists()
    assert path.with_suffix(".json.sha256").exists()
    assert loaded["stage"] == "vq"
    assert loaded["epoch"] == 3
    assert loaded["train"]["total_loss"] == 0.5
    assert loaded["val"]["total_loss"] == 0.25
