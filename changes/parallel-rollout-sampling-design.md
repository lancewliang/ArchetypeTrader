# Phase II PPO Process-Based Parallel Rollout Sampling Design

## Status

Revised design. The required target is multi-process rollout collection. The process backend has been implemented; representative serial-vs-process benchmarking remains pending.

The already implemented thread backend is retained only as an optional diagnostic backend. It is not the target solution for training throughput, because CPU-bound `HorizonEnv.step()` work can remain limited by the Python GIL.

## Background

`PPOTrainer.collect_rollout()` originally batched selector inference, then stepped each `HorizonEnv` serially:

1. Build `obs_batch` from current env observations.
2. Run one batched `actor_critic.act(obs_tensor)` in the learner process.
3. Loop through envs and call `env.step(action)` one by one.
4. Append one `[env]` row into `RolloutBuffer`.

The current codebase now also has a `ThreadedRolloutSampler`, but that still runs inside one Python process. It can overlap I/O or C-extension work, yet it is not enough when Phase I streaming decode / trading replay is CPU-bound Python work. The next implementation must use worker processes.

## Goals

- Use multiple persistent worker processes to collect env transitions in parallel.
- Keep one central learner process and one PPO update stream.
- Keep selector inference batched in the learner process.
- Keep selector weights fixed during each rollout.
- Preserve `RolloutBuffer` shape and GAE semantics: samples remain organized as `[step, env_id]`.
- Preserve checkpoint/resume behavior for `cursor`, `prev_terminal_position`, cumulative loss, and consecutive loss state.
- Avoid future information leakage and cross-env mutable state leakage.
- Keep `serial` mode as the deterministic fallback.

## Non-Goals

- Do not implement distributed PPO or multiple learner processes.
- Do not update selector weights asynchronously while workers are still collecting a rollout.
- Do not move PPO loss or optimizer work into worker processes.
- Do not allow workers to access validation/test data during training.
- Do not change reward, label, GAE, or evaluation semantics.
- Do not introduce reward normalization.

## Proposed Architecture

Use persistent actor-style worker processes:

```text
Learner process
  - owns ArchetypeSelector / ActorCritic
  - owns optimizer, PPOLoss, RolloutBuffer, ScheduleManager
  - batches current obs and samples actions
  - sends one action command to each env worker
  - receives one transition result from each env worker
  - sorts results by env_id and commits one buffer row
  - runs PPO update only after full rollout is collected

Env worker process N
  - owns exactly one HorizonEnv
  - owns exactly one Phase1FrozenPolicy instance
  - owns exactly one TradingEnv instance
  - owns only its train shard horizon_indices
  - receives actions from learner
  - executes current_label_info(), env.step(), done reset
  - returns RolloutSample-compatible transition payload and next obs
```

The process backend must be implemented as persistent workers, not as `ProcessPoolExecutor.submit(env.step)` per rollout step. Per-step process-pool submission would repeatedly serialize state and destroy most of the throughput gain.

## Worker Construction

The process backend needs a worker spec that is pickleable and sufficient to build an env inside the child process.

Recommended new dataclass:

```python
@dataclass(frozen=True)
class HorizonEnvWorkerSpec:
    env_id: int
    horizon_indices: list[int]
    config: Phase2Config
    input_schema: dict
    reward_alignment_name: str
    cost_config: dict
    phase1_decoder_path: str
    phase1_codebook_path: str
    market_data_payload: Any
```

Implementation notes:

- First implementation may pass the train market frame or dataset payload by pickle at worker startup.
- If memory becomes too high, replace `market_data_payload` with memory-mapped numpy arrays or shared-memory blocks.
- Worker startup cost is acceptable because workers persist for the entire trainer lifetime.
- Workers must load `Phase1FrozenPolicy` inside the child process.
- Worker Phase I decode should default to CPU (`rollout_collection.worker_device="cpu"`) to avoid CUDA multiprocessing hazards. The learner may still train the selector on CUDA.

`HorizonFactory` should grow a worker-spec creation path:

```text
create_envs()
  - existing serial/thread path, returns in-process HorizonEnv instances

create_worker_specs()
  - process path, returns HorizonEnvWorkerSpec objects
```

The process backend can use `create_worker_specs()` and does not need live `HorizonEnv` instances in the learner process.

## Process Start Method

Use `multiprocessing.get_context(config.rollout_collection.process_start_method)`.

Default:

```python
process_start_method = "spawn"
```

Rationale:

- `spawn` is safer with PyTorch and CUDA than `fork`.
- Worker state is explicit and easier to audit.
- The cost is paid once at worker startup.

`forkserver` can be allowed on Linux as an advanced option. Plain `fork` should not be the default.

## Worker Protocol

Use one command pipe/queue and one result pipe/queue per worker, or a single result queue tagged by `env_id`.

Commands:

```python
ResetCommand(prev_terminal_position=0, cursor=0, reset_risk_state=True)
StepCommand(
    rollout_step=int,
    rollout_length=int,
    action=int,
    obs=np.ndarray,
    log_prob=float,
    value=float,
)
GetStateCommand()
RestoreStateCommand(cursor=int, prev_terminal_position=int, cumulative_loss=float, consecutive_losses=int)
CloseCommand()
```

Results:

```python
ResetResult(env_id=int, obs=np.ndarray)
StepResult(env_id=int, sample=RolloutSample, next_current_obs=np.ndarray)
StateResult(env_id=int, state=dict)
WorkerError(env_id=int, command=str, cursor=int | None, message=str, traceback=str)
ClosedResult(env_id=int)
```

The worker should execute `current_label_info()` before `env.step(action)`, exactly like the current serial path.

## Rollout Flow

For each `rollout_step`:

1. Learner snapshots current observations:

   ```python
   obs_before = list(current_obs)
   obs_batch = np.array(obs_before, dtype=np.float32)
   ```

2. Learner runs one batched selector forward:

   ```python
   with torch.no_grad():
       act_out = actor_critic.act(obs_tensor, deterministic=False)
   ```

3. Learner sends one `StepCommand` to each worker with:

   - action
   - old obs
   - old log prob
   - old value
   - rollout step index
   - rollout length

4. Each worker executes its own env transition:

   - `kl_label, is_labeled = env.current_label_info()`
   - `next_obs, reward, done, env_truncated, info = env.step(action)`
   - compute rollout truncation flag
   - apply reward scaling
   - build `RolloutSample`
   - if `done`, call `env.reset()` and return reset obs as `next_current_obs`

5. Learner waits for all worker results for that logical step.

6. Learner sorts results by `env_id`, updates `current_obs`, and commits:

   ```python
   buffer.add(samples_sorted_by_env_id)
   ```

7. If any worker fails, learner must not commit that logical step.

This preserves PPO on-policy semantics: actions come from the same pre-update policy snapshot, and PPO update starts only after the full rollout finishes.

## Future-Leakage Controls

Multi-process sampling does not itself cause future leakage. The implementation must still enforce:

- Selector weights are fixed for the full rollout.
- Workers never receive selector parameters or optimizer state.
- Workers receive only the sampled action and their own env state.
- A later horizon in one worker cannot update the learner policy before another worker's earlier horizon action is chosen.
- `current_label_info()` remains loss metadata only; it must not enter selector observations.
- `Phase2Dataset.get_selector_state(idx, prev_terminal_position)` remains horizon-start causal.
- Phase II training entry still loads train/val only; test remains in the independent backtest entry.
- Each worker owns only one env shard and cannot mutate another worker's cursor, position, or risk state.

The key rule is logical-time isolation: wall-clock parallelism is allowed; policy updates during rollout collection are not.

## Config Changes

Extend `RolloutCollectionConfig`:

```python
@dataclass(frozen=True)
class RolloutCollectionConfig:
    mode: Literal["serial", "thread", "process"] = "serial"
    max_workers: Optional[int] = None
    fail_fast: bool = True
    process_start_method: Literal["spawn", "forkserver"] = "spawn"
    worker_device: str = "cpu"
    worker_startup_timeout_seconds: float = 60.0
    worker_step_timeout_seconds: Optional[float] = None
    restart_failed_workers: bool = False
    shared_dataset_mode: Literal["pickle", "memmap"] = "pickle"
```

Fields:

- `mode="serial"` keeps deterministic in-process behavior.
- `mode="thread"` retains the existing diagnostic backend.
- `mode="process"` enables persistent env worker processes and is the throughput target.
- `max_workers=None` means one worker per env.
- `process_start_method="spawn"` avoids unsafe CUDA/fork interactions.
- `worker_device="cpu"` keeps Phase I decoder inference inside workers off the learner GPU.
- `worker_step_timeout_seconds` can fail-fast on wedged env workers.
- `restart_failed_workers=False` keeps failure semantics simple for the first implementation.
- `shared_dataset_mode="pickle"` is simplest; `memmap` is a later memory optimization.

## File-Level Implementation Plan

### `src/config/phase2_config.py`

- Extend `RolloutCollectionConfig.mode` to include `"process"`.
- Add process-specific fields and config docs.
- Keep default `mode="serial"`.
- Keep `thread` documented as non-target fallback.

### `src/trading/horizon_factory.py`

- Add `HorizonEnvWorkerSpec`.
- Add `create_worker_specs()`.
- Ensure worker specs use the same shard assignment as `create_envs()`.
- Keep `phase2_env_shards.feather` unchanged.

### `src/rl/rollout_worker.py`

New process worker module:

- Define command/result dataclasses.
- Define `run_horizon_env_worker(spec, command_conn, result_conn)`.
- Build `Phase2Dataset`, `Phase1FrozenPolicy`, `TradingEnv`, and `HorizonEnv` inside the child process.
- Execute reset/step/state/restore/close commands.
- Return structured `WorkerError` with traceback.

### `src/rl/rollout_sampler.py`

- Add `ProcessRolloutSampler`.
- Start persistent workers during setup.
- Initialize worker observations through `ResetCommand`.
- Send `StepCommand` per logical rollout step.
- Wait for all `StepResult`s, sort by `env_id`, then commit buffer row.
- Expose `get_env_states()` and `restore_env_states()` for checkpointing.
- Expose `close()` and ensure workers terminate at trainer shutdown.

### `src/rl/ppo_trainer.py`

- Construct process sampler when `mode="process"`.
- Delegate initial obs reset to sampler, because process mode owns worker envs.
- Update `get_state()` to read env states from sampler rather than directly from `self.envs`.
- Update `load_state()` to restore env state through sampler.
- Close process workers when training finishes or on fatal error.

### `src/trainers/phase2_trainer.py`

- Pass process worker specs into PPO trainer or sampler factory.
- Log rollout mode, worker count, start method, worker device, and dataset sharing mode.
- Ensure workers are closed in normal completion and exception paths.

### Tests

Add unit tests:

- Worker can build env from `HorizonEnvWorkerSpec`.
- Worker reset/step/state/restore commands work.
- `ProcessRolloutSampler` returns samples sorted by env id.
- Failed worker step does not commit a partial buffer row.
- Checkpoint get/load round trip uses worker states.
- `mode="process"` config loads and docs cover every field.

Add integration smoke tests:

- Tiny Phase II train run with `rollout_collection.mode="process"`.
- Compare serial vs process shape-level stats:
  - sample count
  - done/truncated/bootstrap count
  - finite PPO losses
  - valid rollout timing fields

Add benchmark:

- Run a representative short train job in `serial` and `process`.
- Compare `rollout_env_step_seconds` and `rollout_samples_per_second`.

## Error Handling

Process sampler must:

- fail if any worker returns `WorkerError`;
- include env id, cursor, command type, and traceback;
- avoid committing `current_obs` or buffer samples until all workers return for the logical step;
- close all workers on fail-fast;
- never leave worker processes alive after trainer shutdown.

For the first implementation, failed workers should terminate the rollout. Restarting failed workers can be added later.

## Checkpoint And Resume

Current checkpoint state stores env state from in-process env objects. Process mode must route this through the sampler:

```python
env_states = sampler.get_env_states()
sampler.restore_env_states(env_states)
```

State per worker:

- `env_id`
- `cursor`
- `prev_terminal_position`
- `cumulative_loss`
- `consecutive_losses`

After restore, workers must return their current obs so the learner's `current_obs` cache is consistent.

## Determinism

To keep process mode close to serial semantics:

- learner process owns torch action sampling;
- worker results are sorted by `env_id`;
- each worker is seeded deterministically from `config.seed + env_id`;
- workers should not use global RNG in `env.step()` unless env-local seeded RNG is used;
- selector and optimizer are never touched by workers.

Exact floating-point parity is not required for process mode. Shape, safety, and causal semantics are required.

## Performance Expectations

Expected improvement:

- If `HorizonEnv.step()` is CPU-bound, process mode should improve rollout throughput roughly up to `num_envs` / CPU-core limits.
- If selector forward or GPU learner update dominates, process mode will help less.
- If worker startup dominates only short smoke tests, use longer rollout benchmarks for timing decisions.
- If memory copying of dataset dominates startup or RAM, switch `shared_dataset_mode` from `pickle` to `memmap` in a follow-up.

Timing stats to keep:

- `rollout_collect_seconds`
- `rollout_policy_forward_seconds`
- `rollout_env_step_seconds`
- `rollout_ipc_wait_seconds`
- `rollout_samples_per_second`
- `rollout_worker_startup_seconds`

## Acceptance Criteria

- `mode="serial"` remains behaviorally compatible.
- `mode="process"` starts one persistent worker per env by default.
- Full rollout sample count remains `rollout_length * num_envs`.
- GAE remains grouped by `env_id`.
- Worker failure cannot commit a partial buffer step.
- Checkpoint save/load works in process mode.
- Process workers terminate cleanly after train completion and after exceptions.
- Tiny Phase II process smoke test passes.
- Representative benchmark shows whether process mode improves throughput enough to use in training.

## Open Questions

- Should process mode become the recommended default after benchmark, or remain opt-in?
- Should `shared_dataset_mode="memmap"` be implemented immediately if train frames are large?
- Should worker Phase I decoder always run on CPU, or should a single-GPU worker mode be allowed for special cases?
