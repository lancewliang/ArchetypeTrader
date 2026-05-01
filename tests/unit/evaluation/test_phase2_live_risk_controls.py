"""Phase II live risk controls 单元测试。"""
from __future__ import annotations

from dataclasses import replace

from src.config.phase2_config import LiveRiskControlsConfig
from src.evaluation.phase2_replay import Phase2BacktestRunner
from tests.phase2_test_utils import (
    bias_actor_to_code,
    make_actor_critic,
    make_config,
    make_dataset,
    make_trading_env,
)
from tests.unit.evaluation.test_phase2_replay import ConstantPolicy


class TestPhase2LiveRiskControls:

    def test_daily_loss_limit_triggers_flatten(self, tmp_path):
        """达到 daily_loss_limit 触发 flatten。"""
        config = make_config(tmp_path, horizon=4)
        config = replace(
            config,
            live_risk_controls=LiveRiskControlsConfig(
                daily_loss_limit=-1.0,
                flatten_on_trigger=True,
                mid_horizon_emergency_flatten=True,
            ),
        )
        dataset = make_dataset(config, split="test", count=2)
        actor = make_actor_critic(state_dim=dataset.state_spec().total_dim)
        bias_actor_to_code(actor, 2)
        runner = Phase2BacktestRunner(config, actor, ConstantPolicy(), dataset, make_trading_env)
        records = runner.run_walk_forward("test")
        assert records[0].risk_triggered is True
        assert records[0].risk_reason == "daily_loss_limit"

    def test_mid_horizon_emergency_flatten_records_step(self, tmp_path):
        """mid_horizon_emergency_flatten=true 时记录触发 step。"""
        config = make_config(tmp_path, horizon=4)
        config = replace(
            config,
            live_risk_controls=LiveRiskControlsConfig(
                consecutive_loss_limit=0,
                flatten_on_trigger=True,
                mid_horizon_emergency_flatten=True,
            ),
        )
        dataset = make_dataset(config, split="test", count=1)
        actor = make_actor_critic(state_dim=dataset.state_spec().total_dim)
        bias_actor_to_code(actor, 2)
        runner = Phase2BacktestRunner(config, actor, ConstantPolicy(), dataset, make_trading_env)
        record = runner.run_walk_forward("test")[0]
        assert record.risk_triggered is True
        assert record.risk_trigger_step == 0

    def test_no_risk_trigger_when_controls_disabled(self, tmp_path):
        """未触发风控时 risk 标志保持 false。"""
        config = make_config(tmp_path, horizon=4)
        config = replace(
            config,
            live_risk_controls=LiveRiskControlsConfig(
                daily_loss_limit=None,
                consecutive_loss_limit=None,
                flatten_on_trigger=True,
                mid_horizon_emergency_flatten=True,
            ),
        )
        dataset = make_dataset(config, split="test", count=1)
        actor = make_actor_critic(state_dim=dataset.state_spec().total_dim)
        bias_actor_to_code(actor, 2)
        runner = Phase2BacktestRunner(config, actor, ConstantPolicy(), dataset, make_trading_env)
        assert runner.run_walk_forward("test")[0].risk_triggered is False
