"""Phase II replay 单元测试。"""
from __future__ import annotations

from src.evaluation.phase2_replay import (
    Phase2BacktestRunner,
    Phase2TestLabelLeakageError,
)
from src.models.phase1_frozen_policy import DecodeStepOutput
from tests.phase2_test_utils import (
    bias_actor_to_code,
    make_actor_critic,
    make_config,
    make_dataset,
    make_trading_env,
)


class ConstantPolicy:
    num_codes = 3

    def reset(self, code_id: int) -> None:
        self.code_id = code_id

    def decode_step(self, state_t):
        return DecodeStepOutput(action_logits=None, action=2, recurrent_state=None)


def _runner(tmp_path, *, split: str = "val", labeled: bool = False) -> Phase2BacktestRunner:
    config = make_config(tmp_path, horizon=4)
    dataset = make_dataset(config, split=split, count=4, labeled=labeled)
    actor_critic = make_actor_critic(state_dim=dataset.state_spec().total_dim)
    bias_actor_to_code(actor_critic, 2)
    return Phase2BacktestRunner(
        config,
        actor_critic,
        ConstantPolicy(),
        dataset,
        make_trading_env,
    )


class TestPhase2BacktestRunner:

    def test_walk_forward_time_order(self, tmp_path):
        """walk-forward 按时间正序执行。"""
        runner = _runner(tmp_path)
        records = runner.run_walk_forward("val", deterministic=True)
        assert [r.sample_id for r in records] == ["val_0", "val_1", "val_2", "val_3"]

    def test_position_continuity(self, tmp_path):
        """仓位在相邻 horizons 间正确传递。"""
        runner = _runner(tmp_path)
        records = runner.run_walk_forward("val", deterministic=True)
        assert all(r.final_position == 1 for r in records)
        assert records[0].boundary_cost > 0.0
        assert records[1].boundary_cost == 0.0

    def test_deterministic_argmax(self, tmp_path):
        """主结果使用 argmax。"""
        runner = _runner(tmp_path)
        records = runner.run_walk_forward("val", deterministic=True)
        assert {r.chosen_code for r in records} == {2}

    def test_baselines_run(self, tmp_path):
        """所有 baseline 可运行。"""
        runner = _runner(tmp_path, labeled=True)
        baselines = runner.run_baselines("val")
        assert {"random_selector", "buy_and_hold_long", "buy_and_hold_short", "always_flat", "phase1_demo_label"}.issubset(baselines)
        assert "single_archetype_0" in baselines

    def test_test_label_guard_raises(self, tmp_path):
        """test label 进入主决策路径时抛错。"""
        runner = _runner(tmp_path, split="test", labeled=True)
        try:
            runner.run_walk_forward("test", deterministic=True)
        except Phase2TestLabelLeakageError:
            return
        raise AssertionError("expected Phase2TestLabelLeakageError")

    def test_test_posthoc_demo_label_baseline_requires_explicit_flag(self, tmp_path):
        """test demo-label baseline 只能显式 posthoc 启用。"""
        runner = _runner(tmp_path, split="test", labeled=True)
        assert "phase1_demo_label" not in runner.run_baselines("test")
        assert "phase1_demo_label" in runner.run_baselines(
            "test",
            include_posthoc_demo_label=True,
        )

    def test_execution_lag_offset_changes_replay_inputs(self, tmp_path):
        """execution_lag_offset 改变实际 replay 输入，而不是只写 report。"""
        config = make_config(tmp_path, horizon=4)
        dataset = make_dataset(config, split="test", count=3)
        actor = make_actor_critic(state_dim=dataset.state_spec().total_dim)
        bias_actor_to_code(actor, 2)
        runner = Phase2BacktestRunner(
            config,
            actor,
            ConstantPolicy(),
            dataset,
            lambda: make_trading_env(commission_rate=0.01),
        )
        no_lag = runner.run_walk_forward("test", execution_lag_offset=0)
        lagged = runner.run_walk_forward("test", execution_lag_offset=2)
        assert no_lag
        assert lagged
        assert no_lag[0].cost_paid != lagged[0].cost_paid
