"""Rollout collection backends for Phase II PPO.

The sampler keeps policy inference on the learner thread and optionally
parallelizes only the environment transition work. This preserves PPO's
on-policy rollout/update boundary while allowing expensive ``env.step()`` calls
to overlap.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import multiprocessing as mp
from multiprocessing.connection import Connection
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from src.config.phase2_config import Phase2Config
from src.rl.actor_critic import ActorCritic
from src.rl.rollout_buffer import RolloutBuffer, RolloutSample
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
    StepResult as WorkerStepResult,
    WorkerError,
    run_horizon_env_worker,
)
from src.trading.horizon_env import HorizonEnv
from src.trading.horizon_factory import HorizonEnvWorkerSpec


RewardScaler = Callable[[float], Tuple[float, bool]]


class RolloutCollectionError(RuntimeError):
    """Raised when rollout collection fails before a full buffer step commits."""


@dataclass(frozen=True)
class EnvStepRequest:
    """Inputs needed to execute one env transition after action sampling."""

    env_idx: int
    action: int
    obs: np.ndarray
    log_prob: float
    value: float
    rollout_step: int
    rollout_length: int


@dataclass(frozen=True)
class EnvStepResult:
    """Result of one env transition."""

    env_idx: int
    sample: RolloutSample
    next_current_obs: np.ndarray


@dataclass
class RolloutTimingStats:
    """Wall-clock timing for one rollout collection."""

    collect_seconds: float = 0.0
    policy_forward_seconds: float = 0.0
    env_step_seconds: float = 0.0
    ipc_wait_seconds: float = 0.0
    worker_startup_seconds: float = 0.0
    samples_per_second: float = 0.0


@dataclass
class _ProcessWorkerHandle:
    env_id: int
    process: mp.Process
    conn: Connection


class BaseRolloutSampler:
    """Base class shared by serial and threaded rollout collectors."""

    def __init__(
        self,
        config: Phase2Config,
        actor_critic: ActorCritic,
        envs: Sequence[HorizonEnv],
        device: str,
        scale_reward: RewardScaler,
    ) -> None:
        self.config = config
        self.actor_critic = actor_critic
        self.envs = list(envs)
        self.device = device
        self.scale_reward = scale_reward
        self.num_envs = len(self.envs)

    def collect(
        self,
        buffer: RolloutBuffer,
        current_obs: List[Optional[np.ndarray]],
    ) -> RolloutTimingStats:
        raise NotImplementedError

    def reset_all(self, current_obs: List[Optional[np.ndarray]]) -> None:
        for env_idx, env in enumerate(self.envs):
            current_obs[env_idx] = env.reset()

    def get_env_states(self) -> List[dict]:
        return [
            {
                "env_id": env.env_id,
                "cursor": env.cursor,
                "prev_terminal_position": env.prev_terminal_position,
                "cumulative_loss": env.cumulative_loss,
                "consecutive_losses": env.consecutive_losses,
            }
            for env in self.envs
        ]

    def restore_env_states(
        self,
        env_states: Sequence[dict],
        current_obs: List[Optional[np.ndarray]],
    ) -> None:
        for env_idx, (env, es) in enumerate(zip(self.envs, env_states)):
            obs = env.restore_state(
                cursor=es.get("cursor", 0),
                prev_terminal_position=es.get("prev_terminal_position", 0),
                cumulative_loss=es.get("cumulative_loss", 0.0),
                consecutive_losses=es.get("consecutive_losses", 0),
            )
            current_obs[env_idx] = obs

    def close(self) -> None:
        """Release sampler resources."""

    def _sample_actions(
        self,
        current_obs: List[Optional[np.ndarray]],
    ) -> tuple[List[np.ndarray], List[int], List[float], List[float], float]:
        obs_before = [self._require_obs(obs, idx) for idx, obs in enumerate(current_obs)]
        obs_batch = np.array(obs_before, dtype=np.float32)
        obs_tensor = torch.tensor(obs_batch, dtype=torch.float32, device=self.device)

        started = time.perf_counter()
        with torch.no_grad():
            act_out = self.actor_critic.act(obs_tensor, deterministic=False)
        elapsed = time.perf_counter() - started

        return (
            obs_before,
            act_out.action.cpu().tolist(),
            act_out.log_prob.cpu().tolist(),
            act_out.value.cpu().tolist(),
            elapsed,
        )

    @staticmethod
    def _require_obs(obs: Optional[np.ndarray], env_idx: int) -> np.ndarray:
        if obs is None:
            raise RolloutCollectionError(
                f"rollout current_obs missing for env_id={env_idx}"
            )
        return obs

    def _build_requests(
        self,
        obs_before: Sequence[np.ndarray],
        actions: Sequence[int],
        log_probs: Sequence[float],
        values: Sequence[float],
        rollout_step: int,
    ) -> List[EnvStepRequest]:
        return [
            EnvStepRequest(
                env_idx=env_idx,
                action=int(actions[env_idx]),
                obs=obs_before[env_idx],
                log_prob=float(log_probs[env_idx]),
                value=float(values[env_idx]),
                rollout_step=rollout_step,
                rollout_length=self.config.rollout_length,
            )
            for env_idx in range(len(self.envs))
        ]

    def _step_env(self, request: EnvStepRequest) -> EnvStepResult:
        env = self.envs[request.env_idx]
        cursor_before = getattr(env, "cursor", None)
        try:
            kl_label, is_labeled = env.current_label_info()
            next_obs, reward, done, env_truncated, info = env.step(request.action)
            truncated = bool(
                env_truncated
                or (
                    request.rollout_step == request.rollout_length - 1
                    and not done
                )
            )

            reward_raw = reward
            reward_scaled, reward_was_clipped = self.scale_reward(reward)
            sample = RolloutSample(
                obs=request.obs,
                env_id=request.env_idx,
                action=request.action,
                log_prob=request.log_prob,
                value=request.value,
                reward=reward_scaled,
                reward_raw=reward_raw,
                done=done,
                truncated=truncated,
                reward_was_clipped=reward_was_clipped,
                kl_label=kl_label,
                is_labeled=is_labeled,
                info_cost_paid=info.cost_paid if info else 0.0,
                info_boundary_cost=info.boundary_cost if info else 0.0,
                info_chosen_code=request.action,
            )
            next_current_obs = env.reset() if done else next_obs
            return EnvStepResult(
                env_idx=request.env_idx,
                sample=sample,
                next_current_obs=next_current_obs,
            )
        except Exception as exc:
            raise RolloutCollectionError(
                "rollout env step failed: "
                f"step={request.rollout_step}, env_id={request.env_idx}, "
                f"cursor={cursor_before}"
            ) from exc

    @staticmethod
    def _finish_timing(
        started: float,
        policy_forward_seconds: float,
        env_step_seconds: float,
        num_samples: int,
        ipc_wait_seconds: float = 0.0,
        worker_startup_seconds: float = 0.0,
    ) -> RolloutTimingStats:
        collect_seconds = time.perf_counter() - started
        samples_per_second = (
            float(num_samples) / collect_seconds if collect_seconds > 0 else 0.0
        )
        return RolloutTimingStats(
            collect_seconds=collect_seconds,
            policy_forward_seconds=policy_forward_seconds,
            env_step_seconds=env_step_seconds,
            ipc_wait_seconds=ipc_wait_seconds,
            worker_startup_seconds=worker_startup_seconds,
            samples_per_second=samples_per_second,
        )


class SerialRolloutSampler(BaseRolloutSampler):
    """Current rollout behavior, routed through the sampler interface."""

    def collect(
        self,
        buffer: RolloutBuffer,
        current_obs: List[Optional[np.ndarray]],
    ) -> RolloutTimingStats:
        started = time.perf_counter()
        policy_forward_seconds = 0.0
        env_step_seconds = 0.0

        for rollout_step in range(self.config.rollout_length):
            obs_before, actions, log_probs, values, policy_elapsed = (
                self._sample_actions(current_obs)
            )
            policy_forward_seconds += policy_elapsed
            requests = self._build_requests(
                obs_before, actions, log_probs, values, rollout_step
            )

            step_started = time.perf_counter()
            results = [self._step_env(request) for request in requests]
            env_step_seconds += time.perf_counter() - step_started

            results.sort(key=lambda result: result.env_idx)
            for result in results:
                current_obs[result.env_idx] = result.next_current_obs
            buffer.add([result.sample for result in results])

        return self._finish_timing(
            started,
            policy_forward_seconds,
            env_step_seconds,
            self.config.rollout_length * len(self.envs),
        )


class ThreadedRolloutSampler(BaseRolloutSampler):
    """Threaded env-step rollout collector."""

    def __init__(
        self,
        config: Phase2Config,
        actor_critic: ActorCritic,
        envs: Sequence[HorizonEnv],
        device: str,
        scale_reward: RewardScaler,
    ) -> None:
        super().__init__(config, actor_critic, envs, device, scale_reward)
        configured = config.rollout_collection.max_workers
        if configured is None:
            configured = min(len(self.envs), os.cpu_count() or len(self.envs) or 1)
        self.max_workers = max(1, int(configured))
        self.fail_fast = bool(config.rollout_collection.fail_fast)

    def collect(
        self,
        buffer: RolloutBuffer,
        current_obs: List[Optional[np.ndarray]],
    ) -> RolloutTimingStats:
        started = time.perf_counter()
        policy_forward_seconds = 0.0
        env_step_seconds = 0.0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for rollout_step in range(self.config.rollout_length):
                obs_before, actions, log_probs, values, policy_elapsed = (
                    self._sample_actions(current_obs)
                )
                policy_forward_seconds += policy_elapsed
                requests = self._build_requests(
                    obs_before, actions, log_probs, values, rollout_step
                )

                step_started = time.perf_counter()
                results = self._collect_threaded_step(executor, requests)
                env_step_seconds += time.perf_counter() - step_started

                results.sort(key=lambda result: result.env_idx)
                for result in results:
                    current_obs[result.env_idx] = result.next_current_obs
                buffer.add([result.sample for result in results])

        return self._finish_timing(
            started,
            policy_forward_seconds,
            env_step_seconds,
            self.config.rollout_length * len(self.envs),
        )

    def _collect_threaded_step(
        self,
        executor: ThreadPoolExecutor,
        requests: Sequence[EnvStepRequest],
    ) -> List[EnvStepResult]:
        future_to_request: dict[Future[EnvStepResult], EnvStepRequest] = {
            executor.submit(self._step_env, request): request
            for request in requests
        }
        results: List[EnvStepResult] = []
        errors: List[BaseException] = []

        for future in as_completed(future_to_request):
            try:
                results.append(future.result())
            except BaseException as exc:
                errors.append(exc)
                if self.fail_fast:
                    for pending in future_to_request:
                        if pending is not future:
                            pending.cancel()
                    break

        if errors:
            if len(errors) == 1:
                raise errors[0]
            raise RolloutCollectionError(
                f"{len(errors)} rollout env steps failed; first error: {errors[0]}"
            ) from errors[0]

        return results


class ProcessRolloutSampler(BaseRolloutSampler):
    """Persistent process-worker rollout collector."""

    def __init__(
        self,
        config: Phase2Config,
        actor_critic: ActorCritic,
        worker_specs: Sequence[HorizonEnvWorkerSpec],
        device: str,
        scale_reward: RewardScaler,
    ) -> None:
        super().__init__(config, actor_critic, [], device, scale_reward)
        self.worker_specs = list(worker_specs)
        if not self.worker_specs:
            raise ValueError("process rollout mode requires worker specs")
        configured_workers = config.rollout_collection.max_workers
        if configured_workers is not None and int(configured_workers) != len(self.worker_specs):
            raise ValueError(
                "process rollout mode requires max_workers to equal the number "
                f"of worker specs; got max_workers={configured_workers}, "
                f"worker_specs={len(self.worker_specs)}"
            )
        self.max_workers = len(self.worker_specs)
        self.num_envs = len(self.worker_specs)
        self.fail_fast = bool(config.rollout_collection.fail_fast)
        self._handles: List[_ProcessWorkerHandle] = []
        self._started = False
        self._worker_startup_seconds = 0.0

    def reset_all(self, current_obs: List[Optional[np.ndarray]]) -> None:
        self._ensure_started()
        commands = {
            spec.env_id: ResetCommand()
            for spec in self.worker_specs
        }
        results = self._roundtrip_all(
            commands,
            (ResetResult,),
            timeout=self.config.rollout_collection.worker_startup_timeout_seconds,
        )
        results.sort(key=lambda result: result.env_id)
        for result in results:
            current_obs[result.env_id] = result.obs

    def collect(
        self,
        buffer: RolloutBuffer,
        current_obs: List[Optional[np.ndarray]],
    ) -> RolloutTimingStats:
        self._ensure_started()
        started = time.perf_counter()
        policy_forward_seconds = 0.0
        env_step_seconds = 0.0
        ipc_wait_seconds = 0.0

        for rollout_step in range(self.config.rollout_length):
            obs_before, actions, log_probs, values, policy_elapsed = (
                self._sample_actions(current_obs)
            )
            policy_forward_seconds += policy_elapsed

            commands = {
                env_idx: StepCommand(
                    rollout_step=rollout_step,
                    rollout_length=self.config.rollout_length,
                    action=int(actions[env_idx]),
                    obs=obs_before[env_idx],
                    log_prob=float(log_probs[env_idx]),
                    value=float(values[env_idx]),
                )
                for env_idx in range(self.num_envs)
            }

            step_started = time.perf_counter()
            results = self._roundtrip_all(
                commands,
                (WorkerStepResult,),
                timeout=self.config.rollout_collection.worker_step_timeout_seconds,
            )
            elapsed = time.perf_counter() - step_started
            env_step_seconds += elapsed
            ipc_wait_seconds += elapsed

            results.sort(key=lambda result: result.env_id)
            for result in results:
                current_obs[result.env_id] = result.next_current_obs
            buffer.add([result.sample for result in results])

        return self._finish_timing(
            started,
            policy_forward_seconds,
            env_step_seconds,
            self.config.rollout_length * self.num_envs,
            ipc_wait_seconds=ipc_wait_seconds,
            worker_startup_seconds=self._worker_startup_seconds,
        )

    def get_env_states(self) -> List[dict]:
        self._ensure_started()
        commands = {
            spec.env_id: GetStateCommand()
            for spec in self.worker_specs
        }
        results = self._roundtrip_all(
            commands,
            (StateResult,),
            timeout=self.config.rollout_collection.worker_step_timeout_seconds,
        )
        results.sort(key=lambda result: result.env_id)
        return [result.state for result in results]

    def restore_env_states(
        self,
        env_states: Sequence[dict],
        current_obs: List[Optional[np.ndarray]],
    ) -> None:
        self._ensure_started()
        if len(env_states) != self.num_envs:
            raise RolloutCollectionError(
                f"env state count {len(env_states)} does not match workers {self.num_envs}"
            )
        commands = {}
        for env_idx, state in enumerate(env_states):
            env_id = int(state.get("env_id", env_idx))
            commands[env_id] = RestoreStateCommand(
                cursor=state.get("cursor", 0),
                prev_terminal_position=state.get("prev_terminal_position", 0),
                cumulative_loss=state.get("cumulative_loss", 0.0),
                consecutive_losses=state.get("consecutive_losses", 0),
            )
        results = self._roundtrip_all(
            commands,
            (RestoreStateResult,),
            timeout=self.config.rollout_collection.worker_step_timeout_seconds,
        )
        results.sort(key=lambda result: result.env_id)
        for result in results:
            current_obs[result.env_id] = result.obs

    def close(self) -> None:
        if not self._handles:
            return
        handles = list(self._handles)
        self._handles = []
        self._started = False
        for handle in handles:
            try:
                if handle.process.is_alive():
                    handle.conn.send(CloseCommand())
            except (BrokenPipeError, EOFError, OSError):
                pass
        for handle in handles:
            try:
                if handle.conn.poll(1.0):
                    _result = handle.conn.recv()
                handle.conn.close()
            except (EOFError, OSError):
                pass
            handle.process.join(timeout=2.0)
            if handle.process.is_alive():
                handle.process.terminate()
                handle.process.join(timeout=2.0)

    def _ensure_started(self) -> None:
        if self._started:
            return
        started = time.perf_counter()
        ctx = mp.get_context(self.config.rollout_collection.process_start_method)
        handles: List[_ProcessWorkerHandle] = []
        try:
            for spec in self.worker_specs:
                parent_conn, child_conn = ctx.Pipe()
                process = ctx.Process(
                    target=run_horizon_env_worker,
                    args=(spec, child_conn),
                    daemon=True,
                )
                process.start()
                child_conn.close()
                handles.append(_ProcessWorkerHandle(
                    env_id=spec.env_id,
                    process=process,
                    conn=parent_conn,
                ))
            self._handles = handles
            self._started = True
            self._worker_startup_seconds += time.perf_counter() - started
        except BaseException:
            for handle in handles:
                if handle.process.is_alive():
                    handle.process.terminate()
                handle.conn.close()
            raise

    def _roundtrip_all(
        self,
        commands: dict[int, object],
        expected_types: tuple[type, ...],
        timeout: Optional[float],
    ) -> list:
        handles_by_env = {handle.env_id: handle for handle in self._handles}
        try:
            for env_id, command in commands.items():
                handles_by_env[env_id].conn.send(command)

            deadline = time.monotonic() + timeout if timeout is not None else None
            results = []
            for env_id in sorted(commands):
                handle = handles_by_env[env_id]
                if deadline is not None:
                    remaining = max(deadline - time.monotonic(), 0.0)
                    if not handle.conn.poll(remaining):
                        raise RolloutCollectionError(
                            f"rollout worker timed out: env_id={env_id}, timeout={timeout}"
                        )
                result = handle.conn.recv()
                if isinstance(result, WorkerError):
                    raise self._worker_error(result)
                if not isinstance(result, expected_types):
                    raise RolloutCollectionError(
                        f"unexpected rollout worker result from env_id={env_id}: "
                        f"{type(result).__name__}"
                    )
                results.append(result)
            return results
        except BaseException:
            if self.fail_fast:
                self.close()
            raise

    @staticmethod
    def _worker_error(error: WorkerError) -> RolloutCollectionError:
        return RolloutCollectionError(
            "rollout worker failed: "
            f"env_id={error.env_id}, command={error.command}, "
            f"cursor={error.cursor}, message={error.message}\n{error.traceback}"
        )

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def make_rollout_sampler(
    config: Phase2Config,
    actor_critic: ActorCritic,
    envs: Sequence[HorizonEnv],
    device: str,
    scale_reward: RewardScaler,
    worker_specs: Optional[Sequence[HorizonEnvWorkerSpec]] = None,
) -> BaseRolloutSampler:
    """Factory for configured rollout collection backend."""
    mode = config.rollout_collection.mode
    if mode == "serial":
        return SerialRolloutSampler(config, actor_critic, envs, device, scale_reward)
    if mode == "thread":
        return ThreadedRolloutSampler(config, actor_critic, envs, device, scale_reward)
    if mode == "process":
        return ProcessRolloutSampler(
            config,
            actor_critic,
            worker_specs or [],
            device,
            scale_reward,
        )
    raise ValueError(f"unsupported rollout_collection.mode={mode!r}")
