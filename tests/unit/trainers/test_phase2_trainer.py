"""Phase2Trainer 单元测试。"""
from __future__ import annotations

import json

import polars as pl

from scripts.train_phase2 import run_kl_demo_ablation
from tests.phase2_test_utils import make_config, run_smoke_phase2_training, write_market_splits


class TestPhase2Trainer:

    def test_trainer_runs_full_orchestrator(self, tmp_path):
        """trainer 可以跑通完整 orchestrator。"""
        artifacts = run_smoke_phase2_training(tmp_path)
        assert artifacts.phase2_report.exists()
        assert artifacts.best_selector.exists()
        assert artifacts.last_selector.exists()

    def test_per_horizon_records_exported(self, tmp_path):
        """训练结束后导出 per-horizon records。"""
        artifacts = run_smoke_phase2_training(tmp_path, phase2_batch_id="records")
        for path in [
            artifacts.per_horizon_records_train,
            artifacts.per_horizon_records_val,
            artifacts.per_horizon_records_test,
        ]:
            assert path.exists()
            assert pl.read_ipc(path).height > 0

    def test_sensitivity_result_written(self, tmp_path):
        """sensitivity 结果写入 JSON。"""
        artifacts = run_smoke_phase2_training(tmp_path, phase2_batch_id="sensitivity")
        path = artifacts.artifacts_dir / "composite_score_sensitivity_phase2.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert "base_best" in payload

    def test_kl_demo_ablation_matrix_outputs_json_and_csv(self, tmp_path):
        """KL/demo ablation matrix 能生成 JSON 与 summary CSV。"""
        train, val, test = write_market_splits(tmp_path / "market")
        config = make_config(
            tmp_path,
            horizon=4,
            num_envs=1,
            rollout_length=2,
            total_timesteps=4,
            phase2_batch_id="ablation",
        )
        config = config.__class__.from_dict(
            {
                **config.to_dict(),
                "train_file": str(train),
                "val_file": str(val),
                "test_file": str(test),
            }
        )
        rc = run_kl_demo_ablation(config, [0.0, 0.1])
        assert rc == 0
        assert (config.artifacts_dir() / "phase2_ablation_kl_demo.json").exists()
        assert (config.artifacts_dir() / "phase2_ablation_summary.csv").exists()
