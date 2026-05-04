"""Process rollout worker protocol tests."""
from __future__ import annotations

import multiprocessing as mp

import numpy as np

from src.rl.rollout_worker import (
    CloseCommand,
    ClosedResult,
    GetStateCommand,
    ResetCommand,
    ResetResult,
    RestoreStateCommand,
    RestoreStateResult,
    StateResult,
    StepCommand,
    StepResult,
    run_horizon_env_worker,
)
from src.trading.horizon_factory import HorizonFactory
from tests.phase2_test_utils import (
    make_config,
    make_dataset,
    make_frozen_policy,
    make_trading_env,
)


def _worker_spec(tmp_path):
    config = make_config(tmp_path, horizon=4, num_envs=1, rollout_length=2)
    dataset = make_dataset(config, count=3, labeled=True)
    factory = HorizonFactory(config, dataset, make_frozen_policy(), make_trading_env)
    specs, _shards = factory.create_worker_specs(
        phase1_decoder_path=config.phase1_dir() / "decoder.pt",
        phase1_codebook_path=config.phase1_dir() / "codebook.pt",
        cost_config={"commission_rate": 0.0, "book_levels": 5},
        reward_alignment_name="paper_formula",
    )
    return specs[0]


def test_rollout_worker_reset_step_state_restore_close(tmp_path):
    """worker 子进程可执行完整基础协议。"""
    spec = _worker_spec(tmp_path)
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    process = ctx.Process(
        target=run_horizon_env_worker,
        args=(spec, child_conn),
        daemon=True,
    )
    process.start()
    child_conn.close()
    try:
        parent_conn.send(ResetCommand())
        reset_result = parent_conn.recv()
        assert isinstance(reset_result, ResetResult)
        assert reset_result.env_id == 0

        obs = reset_result.obs
        parent_conn.send(StepCommand(
            rollout_step=0,
            rollout_length=2,
            action=0,
            obs=obs,
            log_prob=0.0,
            value=0.0,
        ))
        step_result = parent_conn.recv()
        assert isinstance(step_result, StepResult)
        assert step_result.sample.env_id == 0
        assert step_result.next_current_obs.shape == obs.shape

        parent_conn.send(GetStateCommand())
        state_result = parent_conn.recv()
        assert isinstance(state_result, StateResult)
        assert state_result.state["env_id"] == 0
        assert state_result.state["cursor"] >= 1

        parent_conn.send(RestoreStateCommand(
            cursor=0,
            prev_terminal_position=0,
            cumulative_loss=0.0,
            consecutive_losses=0,
        ))
        restore_result = parent_conn.recv()
        assert isinstance(restore_result, RestoreStateResult)
        assert np.asarray(restore_result.obs).shape == obs.shape

        parent_conn.send(CloseCommand())
        close_result = parent_conn.recv()
        assert isinstance(close_result, ClosedResult)
    finally:
        parent_conn.close()
        process.join(timeout=3.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=3.0)
    assert process.exitcode == 0
