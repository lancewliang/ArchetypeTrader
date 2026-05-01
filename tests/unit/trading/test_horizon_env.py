"""HorizonEnv 单元测试。"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from src.config.phase2_config import HorizonScheduleConfig, LiveRiskControlsConfig
from src.models.phase1_frozen_policy import DecodeStepOutput
from src.trading.horizon_env import HorizonEnv
from tests.phase2_test_utils import make_config, make_dataset, make_trading_env


class ConstantPolicy:
    def __init__(self, action: int = 2) -> None:
        self.action = action
        self.reset_calls = 0
        self.decode_step_calls = 0

    def reset(self, code_id: int) -> None:
        self.reset_calls += 1

    def decode_step(self, state_t):
        self.decode_step_calls += 1
        return DecodeStepOutput(
            action_logits=np.array([0.0, 0.0, 1.0]),
            action=self.action,
            recurrent_state=None,
        )


def _env(tmp_path, *, action: int = 2, gap_bars: int = 0, position_continuity: bool = True):
    config = make_config(tmp_path, horizon=4, num_envs=1, rollout_length=2)
    config = replace(
        config,
        horizon_schedule=replace(
            config.horizon_schedule,
            position_continuity=position_continuity,
            gap_threshold_bars=1,
            gap_mode="force_flatten",
        ),
    )
    dataset = make_dataset(config, count=3, gap_bars=gap_bars)
    policy = ConstantPolicy(action)
    env = HorizonEnv(
        env_id=0,
        dataset=dataset,
        frozen_policy=policy,
        trading_env=make_trading_env(),
        config=config,
        horizon_indices=[0, 1, 2],
    )
    return env, policy


class TestHorizonEnv:

    def test_reset_returns_first_selector_state(self, tmp_path):
        env, _policy = _env(tmp_path)
        obs = env.reset(prev_terminal_position=1)
        assert obs.shape == (4,)
        assert obs[-1] == 1.0

    def test_step_executes_full_horizon(self, tmp_path):
        env, policy = _env(tmp_path, action=2)
        env.reset()
        next_obs, reward, done, truncated, info = env.step(0)
        assert policy.decode_step_calls == 4
        assert info.horizon_steps == 4
        assert info.final_position == 1
        assert reward != 0.0
        assert done is False
        assert truncated is False
        assert next_obs[-1] == 1.0

    def test_prev_terminal_position_carries_to_next_horizon(self, tmp_path):
        env, _policy = _env(tmp_path, action=2)
        env.reset()
        env.step(0)
        _next_obs, _reward, _done, _truncated, info = env.step(0)
        assert info.prev_terminal_position == 1

    def test_decode_step_used_once_per_timestep(self, tmp_path):
        env, policy = _env(tmp_path, action=1)
        env.reset()
        env.step(2)
        assert policy.reset_calls == 1
        assert policy.decode_step_calls == env.config.horizon

    def test_position_continuity_false_resets_each_horizon_flat(self, tmp_path):
        env, _policy = _env(tmp_path, action=2, position_continuity=False)
        env.reset(prev_terminal_position=1)
        _next_obs, _reward, _done, _truncated, info = env.step(0)
        assert info.prev_terminal_position == 0

    def test_gap_within_threshold_carries_position(self, tmp_path):
        env, _policy = _env(tmp_path, action=2, gap_bars=1)
        env.reset(prev_terminal_position=1)
        _next_obs, _reward, _done, _truncated, info = env.step(0)
        assert info.gap_mode_applied == "carry"
        assert info.prev_terminal_position == 1

    def test_gap_above_threshold_force_flattens(self, tmp_path):
        env, _policy = _env(tmp_path, action=2, gap_bars=3)
        env.reset(prev_terminal_position=1)
        _next_obs, _reward, _done, _truncated, info = env.step(0)
        assert info.gap_mode_applied == "force_flatten"
        assert info.prev_terminal_position == 0

    def test_risk_trigger_flattens_actions(self, tmp_path):
        config = make_config(tmp_path, horizon=4)
        config = replace(
            config,
            live_risk_controls=LiveRiskControlsConfig(
                daily_loss_limit=0.1,
                flatten_on_trigger=True,
                mid_horizon_emergency_flatten=True,
            ),
        )
        dataset = make_dataset(config, count=1)
        env = HorizonEnv(
            env_id=0,
            dataset=dataset,
            frozen_policy=ConstantPolicy(action=2),
            trading_env=make_trading_env(),
            config=config,
            horizon_indices=[0],
        )
        env.reset()
        env._cumulative_loss = -1.0
        _next_obs, _reward, _done, _truncated, info = env.step(0)
        assert info.risk_triggered is True
        assert info.final_position == 0

    def test_done_after_last_horizon(self, tmp_path):
        env, _policy = _env(tmp_path)
        env.reset()
        env.step(0)
        env.step(0)
        _next_obs, _reward, done, _truncated, _info = env.step(0)
        assert done is True
