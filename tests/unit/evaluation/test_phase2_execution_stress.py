"""Phase II execution stress 单元测试。

测试用例:
- commission/slippage/execution_lag stress 场景可运行。
- execution_lag +2 bars 结果写入 stress summary。
- selector latency p50/p95/p99 被记录。
"""
import pytest

from src.config.phase2_config import ExecutionStressConfig
from src.evaluation.phase2_execution_stress import (
    Phase2ExecutionStressRunner,
)


class TestPhase2ExecutionStress:

    def test_stress_scenarios_run(self):
        config = ExecutionStressConfig(
            commission_multipliers=[1.0, 2.0],
            slippage_multipliers=[1.0],
            execution_lag_offsets=[0, 2],
        )
        runner = Phase2ExecutionStressRunner(
            config,
            lambda scenario: [
                {
                    "chosen_code": 0,
                    "reward_raw": -scenario.commission_multiplier,
                    "boundary_cost": 0.0,
                    "cost_paid": 0.0,
                    "step_returns": [-scenario.commission_multiplier],
                }
            ],
            num_codes=2,
        )
        result = runner.run()
        assert len(result.scenarios) == 4

    def test_execution_lag_results_recorded(self):
        config = ExecutionStressConfig(
            commission_multipliers=[1.0],
            slippage_multipliers=[1.0],
            execution_lag_offsets=[2],
        )
        runner = Phase2ExecutionStressRunner(
            config,
            lambda _scenario: [{"chosen_code": 0, "reward_raw": 1.0}],
            num_codes=2,
        )
        result = runner.run()
        assert result.scenarios[0]["execution_lag_offset"] == 2

    def test_selector_latency_recorded(self):
        config = ExecutionStressConfig(
            commission_multipliers=[1.0],
            slippage_multipliers=[1.0],
            execution_lag_offsets=[0],
        )
        runner = Phase2ExecutionStressRunner(
            config,
            lambda _scenario: [{"chosen_code": 0, "reward_raw": 1.0}],
            num_codes=2,
        )
        result = runner.run()
        assert set(result.selector_latency) == {"p50_ms", "p95_ms", "p99_ms"}
