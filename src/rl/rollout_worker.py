"""Persistent process worker protocol for Phase II PPO rollout collection."""
from __future__ import annotations

import random
import traceback as _traceback
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Any, Dict, Optional

import numpy as np
import torch

from src.config.phase2_config import Phase2Config
from src.models.phase1_frozen_policy import Phase1FrozenPolicy
from src.rl.reward_scaling import scale_phase2_reward
from src.rl.rollout_buffer import RolloutSample
from src.trading.cost_model import LobDepthCostModel
from src.trading.env import TradingEnv
from src.trading.horizon_env import HorizonEnv
from src.trading.horizon_factory import HorizonEnvWorkerSpec
from src.trading.reward_alignment import RewardAlignment


@dataclass(frozen=True)
class ResetCommand:
    prev_terminal_position: int = 0
    cursor: int = 0
    reset_risk_state: bool = True


@dataclass(frozen=True)
class StepCommand:
    rollout_step: int
    rollout_length: int
    action: int
    obs: np.ndarray
    log_prob: float
    value: float


@dataclass(frozen=True)
class GetStateCommand:
    pass


@dataclass(frozen=True)
class RestoreStateCommand:
    cursor: int
    prev_terminal_position: int
    cumulative_loss: float = 0.0
    consecutive_losses: int = 0


@dataclass(frozen=True)
class CloseCommand:
    pass


@dataclass(frozen=True)
class ResetResult:
    env_id: int
    obs: np.ndarray


@dataclass(frozen=True)
class StepResult:
    env_id: int
    sample: RolloutSample
    next_current_obs: np.ndarray


@dataclass(frozen=True)
class StateResult:
    env_id: int
    state: Dict[str, Any]


@dataclass(frozen=True)
class RestoreStateResult:
    env_id: int
    obs: np.ndarray


@dataclass(frozen=True)
class ClosedResult:
    env_id: int


@dataclass(frozen=True)
class WorkerError:
    env_id: int
    command: str
    cursor: Optional[int]
    message: str
    traceback: str


def run_horizon_env_worker(
    spec: HorizonEnvWorkerSpec,
    conn: Connection,
) -> None:
    """Child-process command loop for one HorizonEnv."""
    env: Optional[HorizonEnv] = None
    try:
        _seed_worker(spec.config, spec.env_id)
        env = _build_env(spec)
        while True:
            command = conn.recv()
            try:
                if isinstance(command, ResetCommand):
                    obs = env.reset(
                        prev_terminal_position=command.prev_terminal_position,
                        cursor=command.cursor,
                        reset_risk_state=command.reset_risk_state,
                    )
                    conn.send(ResetResult(spec.env_id, obs))
                elif isinstance(command, StepCommand):
                    conn.send(_step_env(spec.config, env, command))
                elif isinstance(command, GetStateCommand):
                    conn.send(StateResult(spec.env_id, _env_state(env)))
                elif isinstance(command, RestoreStateCommand):
                    obs = env.restore_state(
                        cursor=command.cursor,
                        prev_terminal_position=command.prev_terminal_position,
                        cumulative_loss=command.cumulative_loss,
                        consecutive_losses=command.consecutive_losses,
                    )
                    conn.send(RestoreStateResult(spec.env_id, obs))
                elif isinstance(command, CloseCommand):
                    conn.send(ClosedResult(spec.env_id))
                    break
                else:
                    raise ValueError(f"unknown rollout worker command: {type(command)!r}")
            except Exception as exc:
                conn.send(_worker_error(spec.env_id, env, command, exc))
    except BaseException as exc:
        conn.send(_worker_error(spec.env_id, env, "startup", exc))
    finally:
        conn.close()


def _seed_worker(config: Phase2Config, env_id: int) -> None:
    seed = int(config.seed) + int(env_id)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build_env(spec: HorizonEnvWorkerSpec) -> HorizonEnv:
    policy = Phase1FrozenPolicy.load(
        spec.phase1_decoder_path,
        spec.phase1_codebook_path,
        device=spec.config.rollout_collection.worker_device,
    )
    cost_model = LobDepthCostModel(
        commission_rate=spec.cost_config.get("commission_rate", 0.0002),
        book_levels=spec.cost_config.get("book_levels", 5),
        insufficient_depth_policy=spec.cost_config.get(
            "insufficient_depth_policy", "reject_transition"
        ),
        slippage_multiplier=spec.cost_config.get("slippage_multiplier", 1.0),
    )
    trading_env = TradingEnv(
        cost_model=cost_model,
        reward_alignment=RewardAlignment(spec.reward_alignment_name),
        max_position=spec.config.max_position,
    )
    return HorizonEnv(
        env_id=spec.env_id,
        dataset=spec.dataset,
        frozen_policy=policy,
        trading_env=trading_env,
        config=spec.config,
        horizon_indices=spec.horizon_indices,
    )


def _step_env(
    config: Phase2Config,
    env: HorizonEnv,
    command: StepCommand,
) -> StepResult:
    kl_label, is_labeled = env.current_label_info()
    next_obs, reward, done, env_truncated, info = env.step(command.action)
    truncated = bool(
        env_truncated
        or (command.rollout_step == command.rollout_length - 1 and not done)
    )

    reward_raw = reward
    reward_scaled, reward_was_clipped = _scale_reward(config, reward)
    sample = RolloutSample(
        obs=command.obs,
        env_id=env.env_id,
        action=command.action,
        log_prob=command.log_prob,
        value=command.value,
        reward=reward_scaled,
        reward_raw=reward_raw,
        done=done,
        truncated=truncated,
        reward_was_clipped=reward_was_clipped,
        kl_label=kl_label,
        is_labeled=is_labeled,
        info_cost_paid=info.cost_paid if info else 0.0,
        info_boundary_cost=info.boundary_cost if info else 0.0,
        info_chosen_code=command.action,
    )
    next_current_obs = env.reset() if done else next_obs
    return StepResult(env.env_id, sample, next_current_obs)


def _scale_reward(config: Phase2Config, reward: float) -> tuple[float, bool]:
    return scale_phase2_reward(config, reward)


def _env_state(env: HorizonEnv) -> Dict[str, Any]:
    return {
        "env_id": env.env_id,
        "cursor": env.cursor,
        "prev_terminal_position": env.prev_terminal_position,
        "cumulative_loss": env.cumulative_loss,
        "consecutive_losses": env.consecutive_losses,
    }


def _worker_error(
    env_id: int,
    env: Optional[HorizonEnv],
    command: Any,
    exc: BaseException,
) -> WorkerError:
    cursor = getattr(env, "cursor", None) if env is not None else None
    command_name = command if isinstance(command, str) else type(command).__name__
    return WorkerError(
        env_id=env_id,
        command=str(command_name),
        cursor=cursor,
        message=str(exc),
        traceback=_traceback.format_exc(),
    )
