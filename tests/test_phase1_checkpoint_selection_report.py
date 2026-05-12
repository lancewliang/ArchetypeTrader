from __future__ import annotations

from types import SimpleNamespace

from src.phase1.checkpoint import (
    Phase1CheckpointSelectionResult,
    Phase1RejectedCheckpointSummary,
)
from src.phase1.phase1_artifact_store import Phase1ArtifactStore
from src.phase1.report import Phase1CheckpointSelectionReport


def _selection_result() -> Phase1CheckpointSelectionResult:
    selected = SimpleNamespace(
        stage="vq",
        epoch=7,
        train={"total_loss": 0.2, "action_accuracy": 0.91},
        val={"total_loss": 0.25, "action_accuracy": 0.88},
        codebook_validation=SimpleNamespace(
            checkpoint_id="vq_epoch_0007",
            passed=True,
            score=0.86,
            failed_layers=(),
            tie_breaker_metrics={"risk_adjusted_return": 1.25},
        ),
    )
    return Phase1CheckpointSelectionResult(
        selected=selected,
        selected_checkpoint_id="vq_epoch_0007",
        selected_epoch=7,
        selected_score=0.86,
        candidate_count=2,
        eligible_count=1,
        rejected=(
            Phase1RejectedCheckpointSummary(
                stage="vq",
                epoch=5,
                checkpoint_id="vq_epoch_<0005>",
                passed=False,
                score=None,
                failed_layers=("vq_internal",),
                reason="validation_failed",
            ),
        ),
        reason="selected_highest_score",
    )


def test_checkpoint_selection_report_renders_html_and_escapes_values() -> None:
    html = Phase1CheckpointSelectionReport().build_html(
        selection_result=_selection_result(),
        config={"validation_interval": 5},
        artifacts={"best_checkpoint": "/tmp/best.pt"},
    )

    assert "<!doctype html>" in html
    assert "第一阶段检验点选择报告" in html
    assert "vq_epoch_0007" in html
    assert "vq_epoch_&lt;0005&gt;" in html
    assert "selected_highest_score" in html
    assert "risk_adjusted_return" in html
    assert 'data-tip="selector 最终选出的 Phase I 检验点' in html
    assert 'data-tip="检验点稳定 ID' in html
    assert 'data-tip="训练集基础指标名称' in html
    assert "/tmp/best.pt" in html
    assert "{%" not in html
    assert "{{" not in html


def test_save_checkpoint_selection_html_uses_phase1_report_path(tmp_path) -> None:
    store = Phase1ArtifactStore(
        pair="BTC",
        batchid="batch_001",
        artifacts_root=tmp_path,
    )
    store.initialize_phase1_artifact_dirs()

    path = store.save_phase1_checkpoint_selection_html(html="<html>ok</html>")

    assert path == store.artifact_paths["reports"] / "phase1_checkpoint_selection.html"
    assert path.read_text(encoding="utf-8") == "<html>ok</html>"
    assert path.with_suffix(".html.sha256").exists()
