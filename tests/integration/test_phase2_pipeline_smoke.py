"""Phase II 完整 smoke pipeline 集成测试。"""
from __future__ import annotations

import json

import pytest

from tests.phase2_test_utils import run_smoke_phase2_training


@pytest.fixture(scope="module")
def smoke_artifacts(tmp_path_factory):
    return run_smoke_phase2_training(tmp_path_factory.mktemp("phase2_smoke"))


class TestPhase2PipelineSmoke:

    @pytest.mark.integration
    def test_full_pipeline_smoke(self, smoke_artifacts):
        """完整 Phase II smoke pipeline。"""
        assert smoke_artifacts.phase2_report.exists()
        assert smoke_artifacts.best_selector.exists()

    @pytest.mark.integration
    def test_required_artifacts_exist(self, smoke_artifacts):
        """所有必要产物文件存在。"""
        for path in [
            smoke_artifacts.phase2_config_yaml,
            smoke_artifacts.horizon_index_train,
            smoke_artifacts.horizon_index_val,
            smoke_artifacts.env_shards,
            smoke_artifacts.best_selector,
            smoke_artifacts.last_selector,
            smoke_artifacts.checkpoint_manifest,
            smoke_artifacts.rollout_stats,
            smoke_artifacts.per_horizon_records_train,
            smoke_artifacts.per_horizon_records_val,
            smoke_artifacts.replay_log,
        ]:
            assert path.exists(), f"missing artifact: {path}"

    @pytest.mark.integration
    def test_report_test_used_for_selection_false(self, smoke_artifacts):
        """phase2_report.json 中 test_used_for_selection=false。"""
        payload = json.loads(smoke_artifacts.phase2_report.read_text(encoding="utf-8"))
        assert payload["test_used_for_selection"] is False
        assert payload["test_loaded_in_training"] is False
        assert payload["test_metrics"] == {}
        assert payload["phase1_batch_id"] == "smoke_phase1"
        assert "equity_curve_summary" in payload
        assert "execution_stress_summary" in payload

    @pytest.mark.integration
    def test_train_pipeline_does_not_emit_test_artifacts(self, smoke_artifacts):
        """训练入口不生成 test horizon index / test per-horizon records。"""
        assert not (
            smoke_artifacts.artifacts_dir / "phase2_horizon_index_test.feather"
        ).exists()
        assert not (
            smoke_artifacts.artifacts_dir / "phase2_per_horizon_records_test.feather"
        ).exists()
