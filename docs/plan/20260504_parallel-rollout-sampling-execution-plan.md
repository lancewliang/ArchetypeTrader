# Phase II PPO Process Rollout Execution Plan

## Status Summary

- Created: 2026-05-04
- Revised for process backend: 2026-05-04
- Source design: `changes/parallel-rollout-sampling-design.md`
- Overall status: Process backend implemented; representative benchmark pending
- Current code status: `serial`, `thread`, and `process` backends exist
- Target backend for throughput: `process`
- Default mode until benchmark: `serial`

## Status Legend

| Status | Meaning |
|---|---|
| Done | Implemented, reviewed, and verified. |
| In Progress | Actively being implemented or reviewed. |
| Pending | Ready or waiting for dependencies. |
| Blocked | Cannot proceed until a decision, fix, or external input lands. |
| Deferred | Explicitly postponed outside the process backend pass. |
| Superseded | Completed earlier, but not sufficient for the current process-backend goal. |

## Current Reality

The existing implementation added `ThreadedRolloutSampler`. That work improved the structure by extracting sampler logic and adding timing diagnostics, but it does not solve the user's throughput problem when env stepping is CPU-bound.

The implementation now includes a persistent multi-process backend:

```text
learner process: selector inference + PPO update
worker process per env: HorizonEnv.step + Phase1FrozenPolicy.decode_step + TradingEnv.replay
```

The process backend avoids per-step process-pool serialization. Each worker process keeps its own `HorizonEnv` alive for the full trainer lifetime.

## Execution Phases

### Phase MP0: Process Baseline And Constraints

Goal: define the exact process boundary and preserve current serial/thread behavior.

Deliverables:

- Confirm current `serial` and `thread` tests remain green before process work.
- Confirm worker processes should run Phase I frozen policy on CPU by default.
- Decide initial dataset transport: `pickle` first, `memmap` deferred unless memory pressure appears.

Exit criteria:

- Existing tests pass before process changes.
- Process implementation will target persistent workers, not per-step process pool calls.

### Phase MP1: Config Extension

Goal: expose process mode without changing default behavior.

Deliverables:

- Extend `RolloutCollectionConfig.mode` to include `"process"`.
- Add:
  - `process_start_method`
  - `worker_device`
  - `worker_startup_timeout_seconds`
  - `worker_step_timeout_seconds`
  - `restart_failed_workers`
  - `shared_dataset_mode`
- Update `Phase2Config.from_dict()` compatibility.
- Update config docs and tests.

Exit criteria:

- Old configs still load.
- `mode="process"` config loads.
- Default remains `serial`.

### Phase MP2: Worker Spec And Env Construction

Goal: give child processes enough information to construct their own envs.

Deliverables:

- Add `HorizonEnvWorkerSpec`.
- Add `HorizonFactory.create_worker_specs()`.
- Ensure worker spec shard assignment matches `create_envs()`.
- Include Phase I decoder/codebook paths, cost config, reward alignment, input schema, and train market data payload.

Exit criteria:

- Unit test verifies worker specs match env shard boundaries.
- Worker spec is pickleable.

### Phase MP3: Worker Process Protocol

Goal: implement the child-process command/result loop.

Deliverables:

- Add `src/rl/rollout_worker.py`.
- Define command/result dataclasses:
  - reset
  - step
  - get state
  - restore state
  - close
  - worker error
- Build `Phase2Dataset`, `Phase1FrozenPolicy`, `TradingEnv`, and `HorizonEnv` inside worker process.
- Return structured errors with traceback and cursor context.

Exit criteria:

- Unit test can start one worker, reset it, step it, get state, restore state, and close it.
- Worker exits cleanly.

### Phase MP4: ProcessRolloutSampler

Goal: add persistent process workers to rollout collection.

Deliverables:

- Add `ProcessRolloutSampler`.
- Start one worker per env by default.
- Initialize observations through worker reset commands.
- Send action commands per logical rollout step.
- Wait for all workers, sort by `env_id`, update current obs, then commit buffer row.
- Add timing fields:
  - `rollout_ipc_wait_seconds`
  - `rollout_worker_startup_seconds`
- Fail-fast without partial buffer commits.
- Close workers on normal shutdown and exceptions.

Exit criteria:

- Unit test verifies full process rollout sample count.
- Worker error test verifies no partial buffer step is committed.
- No worker processes remain after test completion.

### Phase MP5: PPOTrainer And Phase2Trainer Integration

Goal: route setup, checkpoint, resume, and cleanup through the process sampler.

Deliverables:

- `make_rollout_sampler()` supports `mode="process"`.
- `PPOTrainer.setup()` delegates initial obs reset to sampler where needed.
- `PPOTrainer.get_state()` uses sampler env states in process mode.
- `PPOTrainer.load_state()` restores worker env states in process mode.
- `PPOTrainer.close()` or equivalent closes workers.
- `Phase2Trainer.run()` closes workers in normal and error paths.
- Trainer logs mode, worker count, start method, worker device, and dataset sharing mode.

Exit criteria:

- Checkpoint state includes process worker env state.
- Resume test passes in process mode.
- No process leak on success or failure.

### Phase MP6: Tests And Smoke

Goal: verify process mode behavior, not just importability.

Deliverables:

- Unit tests for config docs and `mode="process"` parsing.
- Unit tests for worker spec construction.
- Unit tests for worker protocol.
- Unit tests for `ProcessRolloutSampler`.
- Integration smoke test for tiny Phase II training with `mode="process"`.
- Shape-level comparison between serial and process runs:
  - sample count
  - done/truncated/bootstrap counts
  - finite PPO loss
  - timing fields present

Exit criteria:

- Focused process tests pass.
- Existing serial/thread tests still pass.

### Phase MP7: Benchmark

Goal: prove process mode improves the slow training path.

Deliverables:

- Run representative short jobs in `serial` and `process`.
- Compare:
  - `rollout_collect_seconds`
  - `rollout_env_step_seconds`
  - `rollout_ipc_wait_seconds`
  - `rollout_samples_per_second`
  - PPO finite-loss checks
- Record benchmark result in this document or a sibling benchmark note.

Exit criteria:

- Decision recorded: keep process opt-in, recommend process, or optimize dataset sharing first.

## Execution Status Table

| ID | Phase | Task | Primary Files | Dependencies | Status | Verification |
|---|---|---|---|---|---|---|
| T1 | Existing Thread Work | Sampler abstraction and serial/thread backends | `src/rl/rollout_sampler.py`, `src/rl/ppo_trainer.py` | None | Superseded | Useful structure exists, but thread is not the throughput target |
| T2 | Existing Thread Work | Frozen policy wrapper isolation | `src/models/phase1_frozen_policy.py`, `src/trading/horizon_factory.py` | None | Done | Still required for correctness |
| T3 | Existing Thread Work | Rollout timing fields | `src/rl/ppo_trainer.py`, `src/trainers/phase2_trainer.py` | None | Done | Reuse and extend for process mode |
| MP0.1 | Baseline | Re-run current serial/thread focused tests before process changes | tests | None | Done | Existing green baseline recorded |
| MP0.2 | Baseline | Confirm persistent worker process boundary | design doc | None | Done | Design explicitly rejects per-step process pool |
| MP1.1 | Config | Extend `RolloutCollectionConfig.mode` with `"process"` | `src/config/phase2_config.py` | MP0.1 | Done | Config unit test covers process mode |
| MP1.2 | Config | Add process-specific rollout config fields | `src/config/phase2_config.py` | MP1.1 | Done | Defaults and from_dict compatibility tested |
| MP1.3 | Config | Add config docs for process fields | `src/config/phase2_config.py`, `tests/unit/config/test_phase2_config_docs.py` | MP1.2 | Done | Config docs coverage passes |
| MP2.1 | Worker Spec | Add `HorizonEnvWorkerSpec` | `src/trading/horizon_factory.py` | MP1.1 | Done | Spec is pickleable |
| MP2.2 | Worker Spec | Add `HorizonFactory.create_worker_specs()` | `src/trading/horizon_factory.py` | MP2.1 | Done | Shards match `create_envs()` |
| MP2.3 | Worker Spec | Include Phase I paths, cost config, schema, reward alignment, market payload | `src/trading/horizon_factory.py`, `src/trainers/phase2_trainer.py` | MP2.2 | Done | Worker can build env from spec |
| MP3.1 | Worker Protocol | Add worker command/result dataclasses | `src/rl/rollout_worker.py` | MP2.1 | Done | Module imports cleanly |
| MP3.2 | Worker Protocol | Implement child-process env construction | `src/rl/rollout_worker.py` | MP2.3 | Done | Worker reset returns valid obs |
| MP3.3 | Worker Protocol | Implement reset/step/get-state/restore/close commands | `src/rl/rollout_worker.py` | MP3.2 | Done | Worker protocol unit test passes |
| MP3.4 | Worker Protocol | Return structured `WorkerError` with traceback/cursor | `src/rl/rollout_worker.py` | MP3.3 | Done | Process sampler worker failure test covers error path |
| MP4.1 | Process Sampler | Add `ProcessRolloutSampler` | `src/rl/rollout_sampler.py` | MP3.3 | Done | Starts persistent workers |
| MP4.2 | Process Sampler | Initialize observations through worker reset | `src/rl/rollout_sampler.py` | MP4.1 | Done | Current obs populated from workers |
| MP4.3 | Process Sampler | Send step commands and commit full buffer rows only after all results arrive | `src/rl/rollout_sampler.py` | MP4.2 | Done | Full rollout sample count test passes |
| MP4.4 | Process Sampler | Add fail-fast cleanup with no partial commit | `src/rl/rollout_sampler.py` | MP4.3 | Done | Worker failure test passes |
| MP4.5 | Process Sampler | Add worker close lifecycle | `src/rl/rollout_sampler.py` | MP4.1 | Done | Tests close sampler/trainer workers |
| MP4.6 | Diagnostics | Add IPC wait and worker startup timing | `src/rl/rollout_sampler.py`, `src/rl/ppo_trainer.py` | MP4.1 | Done | Timing fields present in stats |
| MP5.1 | PPO Integration | Sampler factory supports `mode="process"` | `src/rl/rollout_sampler.py` | MP4.1 | Done | PPO trainer setup works |
| MP5.2 | PPO Integration | Route `get_state()` through sampler env-state API | `src/rl/ppo_trainer.py` | MP4.3 | Done | Process checkpoint state contains env states |
| MP5.3 | PPO Integration | Route `load_state()` through sampler restore API | `src/rl/ppo_trainer.py` | MP5.2 | Done | Process trainer state roundtrip test passes |
| MP5.4 | PPO Integration | Add explicit close/cleanup path | `src/rl/ppo_trainer.py`, `src/trainers/phase2_trainer.py` | MP4.5 | Done | Trainer close path exists; tests call close |
| MP5.5 | Trainer Integration | Pass worker specs from `Phase2Trainer` to sampler | `src/trainers/phase2_trainer.py` | MP2.3, MP5.1 | Done | Tiny process smoke reaches rollout |
| MP5.6 | Trainer Integration | Log process rollout configuration | `src/trainers/phase2_trainer.py` | MP1.2 | Done | Log includes mode/start method/device/workers |
| MP6.1 | Tests | Add config process tests | `tests/unit/config/` | MP1 | Done | Test passes |
| MP6.2 | Tests | Add worker spec tests | `tests/unit/trading/` | MP2 | Done | Test passes |
| MP6.3 | Tests | Add worker protocol tests | `tests/unit/rl/` | MP3 | Done | Test passes |
| MP6.4 | Tests | Add process sampler tests | `tests/unit/rl/` | MP4 | Done | Test passes |
| MP6.5 | Tests | Add process Phase II smoke | `tests/integration/` | MP5 | Done | Tiny train run passes |
| MP7.1 | Benchmark | Run representative serial vs process benchmark | logs/artifacts | MP6.5 | Pending | Timing comparison recorded |
| D1 | Documentation | Revise technical design for process backend | `changes/parallel-rollout-sampling-design.md` | User correction | Done | Design now targets process workers |
| D2 | Documentation | Revise execution plan for process backend | `changes/parallel-rollout-sampling-execution-plan.md` | D1 | Done | This document |

## Dependency Order

1. MP0 baseline.
2. MP1 config extension.
3. MP2 worker specs.
4. MP3 process worker protocol.
5. MP4 `ProcessRolloutSampler`.
6. MP5 trainer/checkpoint/resume integration.
7. MP6 tests and smoke.
8. MP7 benchmark and rollout-mode recommendation.

## Risk Register

| Risk | Impact | Mitigation | Status |
|---|---|---|---|
| Per-step process pool serialization destroys speedup | No meaningful performance gain | Use persistent worker processes | Done |
| Dataset pickle copies too much memory | High startup time/RAM | Start with pickle, add `memmap` if benchmark/memory shows pressure | Pending |
| CUDA/fork interaction breaks workers | Crashes or deadlocks | Default to `spawn` and worker CPU device | Done |
| Worker failure leaves child processes alive | Resource leak and stuck jobs | Explicit close/terminate cleanup in sampler/trainer | Done |
| Partial logical step enters buffer after worker error | Invalid GAE/PPO update | Commit only after all worker results arrive | Done |
| Checkpoint resume misses worker env state | Broken resume semantics | Add sampler env-state API | Done |
| Process mode changes sample ordering | GAE grouped incorrectly | Sort results by `env_id` before `buffer.add()` | Done |

## Verification Commands

Executed verification commands:

```bash
/home/lanceliang/miniconda3/envs/ArchetypeTrade/bin/python -m pytest tests/unit/rl tests/integration/test_phase2_pipeline_smoke.py
/home/lanceliang/miniconda3/envs/ArchetypeTrade/bin/python -m pytest tests/unit/config/test_phase2_config_docs.py tests/unit/trading/test_horizon_factory.py tests/unit/rl/test_rollout_worker.py tests/unit/rl/test_rollout_sampler.py tests/unit/rl/test_ppo_trainer.py
/home/lanceliang/miniconda3/envs/ArchetypeTrade/bin/python -m pytest tests/integration/test_phase2_pipeline_smoke.py
/home/lanceliang/miniconda3/envs/ArchetypeTrade/bin/python -m pytest tests/unit/rl/test_rollout_sampler.py tests/unit/rl/test_rollout_worker.py tests/unit/rl/test_ppo_trainer.py tests/integration/test_phase2_pipeline_smoke.py
```

Benchmark comparison:

```bash
# serial
rollout_collection.mode=serial

# process
rollout_collection.mode=process
rollout_collection.max_workers=<num_envs or CPU-bound tuned value>
rollout_collection.worker_device=cpu
```

Compare:

- `rollout_collect_seconds`
- `rollout_policy_forward_seconds`
- `rollout_env_step_seconds`
- `rollout_ipc_wait_seconds`
- `rollout_worker_startup_seconds`
- `rollout_samples_per_second`
- reward mean/std
- done/truncated/bootstrap counts
- PPO finite-loss and finite-gradient checks

## Current Next Step

Run MP7.1 on a representative non-smoke training job and compare serial vs process timing. Keep `serial` as the default until the benchmark confirms the right worker count and dataset sharing mode.
