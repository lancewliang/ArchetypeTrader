"""Rollout sampler unit tests."""
from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Iterable, Optional

import numpy as np
import pytest
import torch

from src.config.phase2_config import RolloutCollectionConfig
from src.rl.actor_critic import ActOutput
from src.rl.rollout_buffer import RolloutBuffer
from src.rl.rollout_sampler import (
    ProcessRolloutSampler,
    RolloutCollectionError,
    SerialRolloutSampler,
    ThreadedRolloutSampler,
)
from src.trading.horizon_factory import HorizonFactory
from tests.phase2_test_utils import (
    make_config,
    make_dataset,
    make_frozen_policy,
    make_trading_env,
)


@dataclass
class _FakeInfo:
    cost_paid: float = 0.0
    boundary_cost: float = 0.0


class _FakeEnv:
    def __init__(
        self,
        env_id: int,
        *,
        done_on_calls: Optional[Iterable[int]] = None,
        fail_on_call: Optional[int] = None,
        delay_seconds: float = 0.0,
    ) -> None:
        self.env_id = env_id
        self.cursor = 0
        self.step_calls = 0
        self.reset_calls = 0
        self.done_on_calls = set(done_on_calls or [])
        self.fail_on_call = fail_on_call
        self.delay_seconds = delay_seconds

    def reset(self) -> np.ndarray:
        self.cursor = 0
        self.reset_calls += 1
        return np.array([float(self.env_id), -float(self.reset_calls)], dtype=np.float32)

    def current_label_info(self):
        return self.env_id, True

    def step(self, action: int):
        call_idx = self.step_calls
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        if self.fail_on_call == call_idx:
            raise RuntimeError(f"boom env {self.env_id}")
        self.step_calls += 1
        self.cursor += 1
        done = call_idx in self.done_on_calls
        next_obs = np.array([float(self.env_id), float(self.cursor)], dtype=np.float32)
        reward = float(self.env_id * 10 + call_idx + action / 100.0)
        return next_obs, reward, done, False, _FakeInfo(
            cost_paid=float(action),
            boundary_cost=float(call_idx),
        )


class _FakeActorCritic:
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> ActOutput:
        action = obs[:, 0].long() % 3
        return ActOutput(
            action=action,
            log_prob=action.float() * 0.1,
            value=obs[:, 1].float(),
        )


class _BadActionActorCritic:
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> ActOutput:
        action = torch.full((obs.shape[0],), 999, dtype=torch.long, device=obs.device)
        return ActOutput(
            action=action,
            log_prob=torch.zeros(obs.shape[0], dtype=torch.float32, device=obs.device),
            value=torch.zeros(obs.shape[0], dtype=torch.float32, device=obs.device),
        )


def _config(tmp_path, *, mode: str = "serial"):
    config = make_config(
        tmp_path,
        horizon=4,
        num_envs=3,
        rollout_length=2,
        total_timesteps=6,
    )
    return replace(
        config,
        rollout_collection=RolloutCollectionConfig(
            mode=mode,
            max_workers=3,
            fail_fast=True,
        ),
    )


def _scale_reward(reward: float):
    return float(reward), False


def _buffer(config) -> RolloutBuffer:
    return RolloutBuffer(
        num_envs=config.num_envs,
        rollout_length=config.rollout_length,
        gamma=config.ppo.gamma,
        gae_lambda=config.ppo.gae_lambda,
    )


def _snapshot(buffer: RolloutBuffer) -> dict:
    return {
        "obs": [[obs.tolist() for obs in step] for step in buffer._obs],
        "actions": buffer._actions,
        "log_probs": buffer._log_probs,
        "values": buffer._values,
        "rewards": buffer._rewards,
        "dones": buffer._dones,
        "truncateds": buffer._truncateds,
        "kl_labels": buffer._kl_labels,
        "is_labeled": buffer._is_labeled,
    }


def _initial_obs(envs):
    return [env.reset() for env in envs]


def test_serial_and_threaded_samplers_match_for_deterministic_envs(tmp_path):
    serial_config = _config(tmp_path, mode="serial")
    thread_config = replace(
        serial_config,
        rollout_collection=RolloutCollectionConfig(
            mode="thread",
            max_workers=3,
            fail_fast=True,
        ),
    )
    serial_envs = [_FakeEnv(i) for i in range(3)]
    threaded_envs = [_FakeEnv(i) for i in range(3)]

    serial_buffer = _buffer(serial_config)
    threaded_buffer = _buffer(thread_config)
    SerialRolloutSampler(
        serial_config,
        _FakeActorCritic(),
        serial_envs,
        "cpu",
        _scale_reward,
    ).collect(serial_buffer, _initial_obs(serial_envs))
    ThreadedRolloutSampler(
        thread_config,
        _FakeActorCritic(),
        threaded_envs,
        "cpu",
        _scale_reward,
    ).collect(threaded_buffer, _initial_obs(threaded_envs))

    assert _snapshot(threaded_buffer) == _snapshot(serial_buffer)


def test_threaded_sampler_sorts_results_by_env_id_before_buffer_add(tmp_path):
    config = _config(tmp_path, mode="thread")
    envs = [
        _FakeEnv(0, delay_seconds=0.02),
        _FakeEnv(1, delay_seconds=0.01),
        _FakeEnv(2, delay_seconds=0.0),
    ]
    buffer = _buffer(config)

    ThreadedRolloutSampler(
        config,
        _FakeActorCritic(),
        envs,
        "cpu",
        _scale_reward,
    ).collect(buffer, _initial_obs(envs))

    assert buffer._actions[0] == [0, 1, 2]
    assert [obs[0] for obs in buffer._obs[0]] == [0.0, 1.0, 2.0]


def test_done_resets_env_before_next_logical_step(tmp_path):
    config = _config(tmp_path, mode="thread")
    envs = [_FakeEnv(0, done_on_calls={0}), _FakeEnv(1), _FakeEnv(2)]
    buffer = _buffer(config)

    ThreadedRolloutSampler(
        config,
        _FakeActorCritic(),
        envs,
        "cpu",
        _scale_reward,
    ).collect(buffer, _initial_obs(envs))

    assert buffer._dones[0][0] is True
    assert buffer._obs[1][0].tolist() == [0.0, -2.0]
    assert envs[0].reset_calls == 2


def test_rollout_terminal_truncation_only_when_not_done(tmp_path):
    config = _config(tmp_path, mode="thread")
    envs = [_FakeEnv(0), _FakeEnv(1, done_on_calls={1}), _FakeEnv(2)]
    buffer = _buffer(config)

    ThreadedRolloutSampler(
        config,
        _FakeActorCritic(),
        envs,
        "cpu",
        _scale_reward,
    ).collect(buffer, _initial_obs(envs))

    assert buffer._truncateds[1][0] is True
    assert buffer._truncateds[1][1] is False
    assert buffer._dones[1][1] is True


def test_worker_exception_does_not_commit_partial_buffer_step(tmp_path):
    config = _config(tmp_path, mode="thread")
    envs = [_FakeEnv(0, delay_seconds=0.01), _FakeEnv(1, fail_on_call=0), _FakeEnv(2)]
    buffer = _buffer(config)
    current_obs = _initial_obs(envs)
    before = [obs.copy() for obs in current_obs]

    with pytest.raises(RolloutCollectionError, match="env_id=1"):
        ThreadedRolloutSampler(
            config,
            _FakeActorCritic(),
            envs,
            "cpu",
            _scale_reward,
        ).collect(buffer, current_obs)

    assert buffer._step_count == 0
    for actual, expected in zip(current_obs, before):
        assert np.array_equal(actual, expected)


def _process_config_and_specs(tmp_path):
    config = make_config(
        tmp_path,
        horizon=4,
        num_envs=2,
        rollout_length=2,
        total_timesteps=4,
    )
    config = replace(
        config,
        rollout_collection=RolloutCollectionConfig(
            mode="process",
            max_workers=2,
            fail_fast=True,
            worker_startup_timeout_seconds=20.0,
            worker_step_timeout_seconds=20.0,
        ),
    )
    dataset = make_dataset(config, count=4, labeled=True)
    factory = HorizonFactory(config, dataset, make_frozen_policy(), make_trading_env)
    specs, _shards = factory.create_worker_specs(
        phase1_decoder_path=config.phase1_dir() / "decoder.pt",
        phase1_codebook_path=config.phase1_dir() / "codebook.pt",
        cost_config={"commission_rate": 0.0, "book_levels": 5},
        reward_alignment_name="paper_formula",
    )
    return config, specs


def test_process_sampler_collects_full_rollout_and_state_roundtrip(tmp_path):
    """process sampler 使用常驻 worker 收集完整 rollout 并支持 state roundtrip。"""
    config, specs = _process_config_and_specs(tmp_path)
    buffer = _buffer(config)
    sampler = ProcessRolloutSampler(
        config,
        _FakeActorCritic(),
        specs,
        "cpu",
        _scale_reward,
    )
    current_obs = [None] * len(specs)
    try:
        sampler.reset_all(current_obs)
        assert all(obs is not None for obs in current_obs)
        timing = sampler.collect(buffer, current_obs)
        assert buffer._step_count == config.rollout_length
        assert len(buffer._actions[0]) == len(specs)
        assert timing.samples_per_second > 0.0
        assert timing.worker_startup_seconds >= 0.0

        states = sampler.get_env_states()
        assert len(states) == len(specs)
        sampler.restore_env_states(states, current_obs)
        assert all(obs is not None for obs in current_obs)
    finally:
        sampler.close()


def test_process_sampler_worker_error_does_not_commit_partial_buffer_step(tmp_path):
    """process worker 异常时不提交当前 logical rollout step。"""
    config, specs = _process_config_and_specs(tmp_path)
    buffer = _buffer(config)
    sampler = ProcessRolloutSampler(
        config,
        _BadActionActorCritic(),
        specs,
        "cpu",
        _scale_reward,
    )
    current_obs = [None] * len(specs)
    try:
        sampler.reset_all(current_obs)
        before = [obs.copy() for obs in current_obs]
        with pytest.raises(RolloutCollectionError, match="rollout worker failed"):
            sampler.collect(buffer, current_obs)
        assert buffer._step_count == 0
        for actual, expected in zip(current_obs, before):
            assert np.array_equal(actual, expected)
    finally:
        sampler.close()
