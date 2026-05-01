"""Phase II walk-forward backtest 集成测试。"""
from __future__ import annotations

import pytest

from tests.unit.evaluation.test_phase2_replay import _runner


class TestPhase2BacktestWalkForward:

    @pytest.mark.integration
    def test_position_continuity(self, tmp_path):
        """prev_terminal_position 在相邻 horizons 间正确传递。"""
        runner = _runner(tmp_path)
        records = runner.run_walk_forward("val", deterministic=True)
        assert records[0].boundary_cost > 0.0
        assert records[1].boundary_cost == 0.0
        assert all(r.final_position == 1 for r in records)

    @pytest.mark.integration
    def test_deterministic_argmax(self, tmp_path):
        """主结果使用 argmax。"""
        runner = _runner(tmp_path)
        records = runner.run_walk_forward("val", deterministic=True)
        assert {r.chosen_code for r in records} == {2}

    @pytest.mark.integration
    def test_stochastic_diagnostic_only(self, tmp_path):
        """stochastic seed pack 只写诊断，不覆盖主结果。"""
        runner = _runner(tmp_path)
        main = runner.run_walk_forward("val", deterministic=True)
        diag = runner.run_walk_forward("val", deterministic=True, stochastic_seeds=[1, 2])
        assert [r.chosen_code for r in main] == [r.chosen_code for r in diag]
