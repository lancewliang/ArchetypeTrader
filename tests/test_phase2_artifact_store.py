from pathlib import Path

import pytest

from src.phase2.metrics.phase2_metric_results import (
    Phase2ValidationMetrics,
    Phase2ValidationResult,
)
from src.phase2.phase2_artifact_store import Phase2ArtifactStore


def _validation_result() -> Phase2ValidationResult:
    return Phase2ValidationResult(
        metrics=Phase2ValidationMetrics(
            mean_return=1.0,
            median_return=0.5,
            sharpe_like=1.2,
            win_rate=0.6,
            mean_turnover=0.1,
        )
    )


def test_default_selector_validation_html_uses_standard_artifact_path(
    tmp_path: Path,
) -> None:
    store = Phase2ArtifactStore("BTCUSDT", "batch_001", tmp_path)
    store.initialize_phase2_artifact_dirs()

    path = store.save_phase2_selector_validation_html(
        validation_result=_validation_result(),
        html="<html></html>",
    )

    assert path == store.artifact_paths["phase2_selector_validation_html"]
    assert path.name == "phase2_selector_validation.html"
    assert path.exists()
    assert not (path.parent / "None_selector_validation.html").exists()


def test_checkpoint_payload_requires_config() -> None:
    with pytest.raises(ValueError, match="missing config"):
        Phase2ArtifactStore._checkpoint_from_dict(
            {
                "epoch": 1,
                "q_network_state_dict": {},
                "optimizer_state_dict": {},
            }
        )
