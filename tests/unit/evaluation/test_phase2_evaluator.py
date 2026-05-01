"""Phase II evaluator 单元测试。"""
from __future__ import annotations

from dataclasses import replace

from src.evaluation.phase2_evaluator import Phase2Evaluator
from tests.unit.evaluation.test_phase2_replay import _runner


class TestPhase2Evaluator:

    def test_val_fast_uses_subset(self, tmp_path):
        """val 快速评估使用 deterministic 子集。"""
        runner = _runner(tmp_path)
        config = replace(runner.config, fast_eval_max_horizons=2)
        runner.config = config
        evaluator = Phase2Evaluator(config, runner, num_codes=3)
        result = evaluator.evaluate_val_fast(update_idx=7)
        assert len(result.per_horizon_records) == 2
        assert result.metrics["update_idx"] == 7

    def test_walk_forward_returns_metrics_and_baselines(self, tmp_path):
        """完整 walk-forward 返回 metrics 和 baseline。"""
        runner = _runner(tmp_path, labeled=True)
        evaluator = Phase2Evaluator(runner.config, runner, num_codes=3)
        result = evaluator.evaluate_walk_forward("val")
        assert result.metrics["num_horizons"] == 4
        assert "random_selector" in result.baseline_results

    def test_rolling_validation_real_folds(self, tmp_path):
        """rolling validation 真实按 fold subset 执行并记录初始仓位。"""
        runner = _runner(tmp_path)
        config = replace(
            runner.config,
            rolling_validation=replace(runner.config.rolling_validation, num_folds=2),
        )
        runner.config = config
        evaluator = Phase2Evaluator(config, runner, num_codes=3)
        result = evaluator.evaluate_rolling_validation()
        assert result.fold_sizes == [2, 2]
        assert result.fold_initial_position_policy == "inherit_previous_fold"
        assert result.fold_initial_positions[0] == 0
        assert result.fold_initial_positions[1] == 1
