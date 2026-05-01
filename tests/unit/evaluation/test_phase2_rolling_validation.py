"""Phase II rolling validation 单元测试。

测试用例:
- 固定 fold 切法在同 seed 下结果一致。
- fold 均值、最差分位、波动聚合正确。
"""
import pytest
from types import SimpleNamespace

from src.config.phase2_config import (
    HorizonScheduleConfig,
    Phase2Config,
    RollingValidationConfig,
)
from src.evaluation.phase2_evaluator import Phase2Evaluator
from src.evaluation.phase2_replay import Phase2HorizonReplayRecord


class _FakeRollingRunner:
    def __init__(self, num_entries=6):
        self.dataset = SimpleNamespace(
            horizon_entries=[
                SimpleNamespace(split="val", sample_id=f"v{i}")
                for i in range(num_entries)
            ]
        )
        self.calls = []

    def run_walk_forward(
        self,
        split,
        deterministic=True,
        entry_indices=None,
        initial_position=0,
        fold_id=None,
    ):
        self.calls.append({
            "entry_indices": list(entry_indices or []),
            "initial_position": initial_position,
            "fold_id": fold_id,
        })
        final_position = int(fold_id or 0) + 1
        return [
            Phase2HorizonReplayRecord(
                sample_id=f"fold{fold_id}_{idx}",
                env_id=0,
                chosen_code=fold_id or 0,
                final_position=final_position,
                reward_raw=float((fold_id or 0) + 1),
                reward_scaled=float((fold_id or 0) + 1),
                boundary_cost=0.0,
                cost_paid=0.0,
                fold_id=fold_id,
                fold_initial_position=initial_position,
                step_returns=[float((fold_id or 0) + 1)],
            )
            for idx in (entry_indices or [])
        ]


class TestPhase2RollingValidation:

    def test_fixed_fold_deterministic(self):
        """固定 fold 切法在同 seed 下结果一致。"""
        config = Phase2Config(
            rolling_validation=RollingValidationConfig(enabled=True, num_folds=3),
            horizon_schedule=HorizonScheduleConfig(position_continuity=True),
        )
        runner = _FakeRollingRunner(num_entries=6)
        evaluator = Phase2Evaluator(config, runner, num_codes=4)
        result = evaluator.evaluate_rolling_validation()
        assert [c["entry_indices"] for c in runner.calls] == [[0, 1], [2, 3], [4, 5]]
        assert result.fold_sizes == [2, 2, 2]

    def test_fold_mean_aggregation(self):
        """fold 均值聚合正确。"""
        config = Phase2Config(
            rolling_validation=RollingValidationConfig(enabled=True, num_folds=2),
        )
        runner = _FakeRollingRunner(num_entries=4)
        result = Phase2Evaluator(config, runner, num_codes=4).evaluate_rolling_validation()
        assert result.fold_mean["net_return"] == pytest.approx(3.0)

    def test_worst_fold_quantile(self):
        """最差 fold 分位正确。"""
        config = Phase2Config(
            rolling_validation=RollingValidationConfig(enabled=True, num_folds=2),
        )
        runner = _FakeRollingRunner(num_entries=4)
        result = Phase2Evaluator(config, runner, num_codes=4).evaluate_rolling_validation()
        assert result.worst_fold_quantile["net_return"] == pytest.approx(2.0)

    def test_fold_volatility(self):
        """fold 间波动聚合正确。"""
        config = Phase2Config(
            rolling_validation=RollingValidationConfig(enabled=True, num_folds=2),
        )
        runner = _FakeRollingRunner(num_entries=4)
        result = Phase2Evaluator(config, runner, num_codes=4).evaluate_rolling_validation()
        assert result.fold_volatility["net_return"] > 0

    def test_fold_initial_position_inherits_previous_fold(self):
        """后一个 fold 继承前一个 fold 的 final_position。"""
        config = Phase2Config(
            rolling_validation=RollingValidationConfig(enabled=True, num_folds=3),
            horizon_schedule=HorizonScheduleConfig(position_continuity=True),
        )
        runner = _FakeRollingRunner(num_entries=6)
        result = Phase2Evaluator(config, runner, num_codes=4).evaluate_rolling_validation()
        assert [c["initial_position"] for c in runner.calls] == [0, 1, 2]
        assert result.fold_initial_position_policy == "inherit_previous_fold"
        assert result.fold_initial_positions == [0, 1, 2]
