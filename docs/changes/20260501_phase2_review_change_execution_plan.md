# Phase II Review 问题修复变更执行计划

**日期**: 2026-05-01
**来源 Review**: `docs/review/20260501_phase2_full_code_quality_review.md`
**设计依据**:
- `docs/design/phase2_archetype_selection_design.md`
- `docs/plan/phase2_archetype_selection_execution_plan.md`

---

## 1. 目标与范围

本计划用于把 Phase II 代码质量 Review 中确认的问题落成可执行的修复方案。范围覆盖:

- P0/P1 级正确性问题: rolling validation、GAE truncated 语义、多 env 仓位继承、composite score、dead code mask。
- 审计与可复现性问题: 配置字段、report 字段、horizon index 字段、checkpoint/resume、训练健康指标。
- 部署前评估能力: execution stress、OOD/distribution shift、online action throttle、KL/demo 消融、rolling validation 产物。

本文件只定义实施计划和技术细节，不包含代码改动。

---

## 2. 修复原则

1. **先修正确性，再补完整性**: P0/P1 必须先于配置补齐和工程重构落地。
2. **保持 Phase I 冻结语义**: Phase II 只使用 Phase I decoder/codebook/labels，不重新训练 Phase I，也不在线调用 DP。
3. **训练与回测语义一致**: horizon reward、成本、仓位连续、gap 处理、streaming decode 必须在 trainer/evaluator/backtest 三条路径一致。
4. **所有 sign-off 指标可审计**: checkpoint 选择、rolling validation、baseline、stress、OOD、guardrail 必须进入产物。
5. **向后兼容现有配置**: 新字段提供默认值；旧配置加载不应失败，除非缺失会造成错误复现或 label 泄漏。

---

## 3. 优先级与交付批次

| 批次 | 优先级 | 目标 | 主要文件 |
| --- | --- | --- | --- |
| A | P0 | 修复训练/评估正确性阻塞项 | `horizon_env.py`, `ppo_trainer.py`, `rollout_buffer.py`, `phase2_replay.py`, `phase2_evaluator.py`, `phase2_selection_policy.py` |
| B | P1 | 补齐选择、审计、mask、report 的 sign-off 能力 | `phase2_config.py`, `phase2_metrics.py`, `phase2_report.py`, `phase2_trainer.py`, `backtest_phase2.py` |
| C | P2 | 补齐部署前评估与实验编排 | evaluator/diagnostics/scripts/report |
| D | P3 | 工程结构优化和维护性提升 | trainer/config/data 模块 |

推荐以 4 到 5 个小 PR/commit 执行，避免一次性改动导致训练链路难以回归。

---

## 4. 批次 A: P0 正确性修复

### A1. 修复 Rolling Validation 真实 fold 执行

**Review 问题**: `Phase2Evaluator.evaluate_rolling_validation()` 每个 fold 都运行完整 val walk-forward，再切片 records，导致 rolling validation 退化为同一结果拆片。

**涉及文件**:
- `src/evaluation/phase2_replay.py`
- `src/evaluation/phase2_evaluator.py`
- `src/evaluation/phase2_report.py`
- `tests/unit/evaluation/test_phase2_rolling_validation.py`
- `tests/unit/evaluation/test_phase2_replay.py`

**实现方案**:

1. 为 `Phase2BacktestRunner.run_walk_forward()` 增加 subset 参数:

```python
def run_walk_forward(
    self,
    split: str,
    deterministic: bool = True,
    stochastic_seeds: Optional[List[int]] = None,
    entry_indices: Optional[Sequence[int]] = None,
    initial_position: int = 0,
    fold_id: Optional[int] = None,
) -> List[Phase2HorizonReplayRecord]:
    ...
```

2. 新增内部解析函数，避免外部直接传 entry 对象:

```python
def _resolve_walk_forward_entries(
    self,
    split: str,
    entry_indices: Optional[Sequence[int]],
) -> List[Tuple[int, HorizonEntry]]:
    # 返回 dataset index + entry，保持时间正序。
```

3. `entry_indices=None` 时保持现有 split 全量行为；非空时只运行指定 contiguous fold subset。

4. fold 内部仓位连续:
   - fold 起点默认 `initial_position=0`。
   - 如果 horizon index 后续补齐 `prev_terminal_position` 字段，则优先支持 `initial_position_policy="from_index"`。
   - report 中记录 `fold_initial_position_policy`，避免 rolling validation 初始仓位语义不透明。

5. `Phase2Evaluator.evaluate_rolling_validation()` 只构造一次 val entry index 列表，按 `num_folds` 切分后把每个 fold 的 index subset 传给 runner。

6. `Phase2HorizonReplayRecord` 增加可选字段:

```python
fold_id: Optional[int] = None
timestamp_start: Optional[str] = None
```

7. `Phase2ReportWriter.write_rolling_validation()` 同时写:
   - `phase2_rolling_validation.json`
   - `phase2_rolling_validation_records.feather`

**验收标准**:
- 每个 fold 的 `sample_id` 集合互不重叠，合并后等于 val split。
- 每个 fold 只调用一次 subset walk-forward，不重复跑完整 val。
- fold 内 `prev_terminal_position -> final_position` 连续继承。
- rolling validation 结果包含 `fold_mean`、`worst_fold_quantile`、`fold_volatility`、`fold_count`、`fold_sizes`。

**测试计划**:
- 新增/更新 `test_phase2_rolling_validation.py`:
  - 构造 9 个 val horizon、3 folds，断言每 fold 只含 3 个 sample。
  - mock runner 记录每次收到的 `entry_indices`，断言不是 `None` 且无重叠。
  - 验证 fold metrics 是各自 subset 计算，不是完整 val records 切片。
- 新增 replay subset 单测:
  - `run_walk_forward(split="val", entry_indices=[2,3])` 只输出对应 sample。
  - subset 内 position continuity 生效。

---

### A2. 补齐 truncated 语义，修复 rollout 边界 GAE 语义

**Review 问题**: `HorizonEnv.step()` 的 `truncated` 始终为 `False`，PPO rollout 到达 buffer 尾部但 episode 未结束时无法表达截断语义。

**涉及文件**:
- `src/rl/ppo_trainer.py`
- `src/rl/rollout_buffer.py`
- `tests/unit/rl/test_ppo_trainer.py`
- `tests/integration/test_phase2_gae_no_cross_env_leakage.py`

**实现方案**:

1. 保持 `HorizonEnv.step()` 只负责环境真实终止，`truncated=False` 仍可保留。

2. 在 `PPOTrainer.collect_rollout()` 中由 collector 明确标注 buffer 截断:

```python
is_rollout_tail = step == self.config.rollout_length - 1
next_obs, reward, done, env_truncated, info = env.step(action)
truncated = bool(env_truncated or (is_rollout_tail and not done))
```

3. `RolloutSample.truncated` 写入上述 `truncated`。

4. `RolloutBuffer.compute_gae()` 保持核心规则:
   - `done=True`: 不 bootstrap。
   - `truncated=True and done=False`: 使用 `last_values[env_id]` bootstrap。

5. 增加 debug 统计:
   - `rollout_truncated_count`
   - `rollout_done_count`
   - `rollout_bootstrap_count`

这些字段进入 `RolloutBuffer.get_stats()` 和 `PPOUpdateStats`。

**验收标准**:
- rollout 最后一行且 `done=False` 的样本 `truncated=True`。
- `done=True` 的 episode 末端不被错误标注为 truncated。
- GAE 在非 done 截断处使用 `last_values`。

**测试计划**:
- 单测构造 `rollout_length=2`、env 至少 4 个 horizon:
  - 第 2 步 sample: `done=False`, `truncated=True`。
  - `compute_gae(last_values=[V])` 的 return 包含 bootstrap。
- 单测构造 env 刚好结束:
  - 最后一行 `done=True`, `truncated=False`。
  - return 不使用 bootstrap。

---

### A3. 修复 HorizonEnv reset/restore 的仓位继承

**Review 问题**: `HorizonEnv.reset()` 总是把 `_prev_terminal_position` 重置为 0；多 env 训练和 resume 时仓位继承链断裂。

**涉及文件**:
- `src/trading/horizon_env.py`
- `src/rl/ppo_trainer.py`
- `src/trading/horizon_factory.py`
- `tests/unit/trading/test_horizon_env.py`
- `tests/integration/test_phase2_resume_checkpoint.py`

**实现方案**:

1. 修改 `HorizonEnv.reset()` 签名:

```python
def reset(
    self,
    prev_terminal_position: int = 0,
    cursor: int = 0,
    reset_risk_state: bool = True,
) -> np.ndarray:
    ...
```

2. 新增公开状态恢复接口，替代 `PPOTrainer.load_state()` 直接写私有属性:

```python
def restore_state(
    self,
    cursor: int,
    prev_terminal_position: int,
    cumulative_loss: float = 0.0,
    consecutive_losses: int = 0,
) -> np.ndarray:
    # 校验 cursor 范围，恢复后返回当前 obs。
```

3. `PPOTrainer.setup()` 初始化 env 时:
   - 默认仍从 `prev_terminal_position=0` 开始。
   - 若未来 `env_shards.chunk_reset_position == "inherit"` 且 horizon index 提供 shard 起点仓位，则传入 shard 起点仓位。

4. `PPOTrainer.load_state()` 改为:

```python
obs = env.restore_state(
    cursor=es.get("cursor", 0),
    prev_terminal_position=es.get("prev_terminal_position", 0),
    cumulative_loss=es.get("cumulative_loss", 0.0),
    consecutive_losses=es.get("consecutive_losses", 0),
)
self._current_obs[env_idx] = obs
```

5. `PPOTrainer.get_state()` 保存:
   - `cursor`
   - `prev_terminal_position`
   - `cumulative_loss`
   - `consecutive_losses`
   - 后续可扩展 `decoder_recurrent_state`

**验收标准**:
- `reset(prev_terminal_position=1)` 后第一条 selector state 使用 position=1 编码。
- resume 后下一次 rollout 不是从 flat 重新开始。
- `PPOTrainer` 不再直接访问 `env._cursor` / `env._prev_terminal_position`。

**测试计划**:
- `test_horizon_env_reset_accepts_prev_terminal_position`
- `test_horizon_env_restore_state_returns_current_obs`
- `test_phase2_resume_checkpoint_restores_env_position`

---

### A4. 实现 Composite Score 选择 best checkpoint

**Review 问题**: 当前 `Phase2SelectionPolicy` 只用单一 `val_net_return`，缺少设计要求的 composite score 和 sensitivity。

**涉及文件**:
- `src/config/phase2_config.py`
- `src/trainers/phase2_selection_policy.py`
- `src/evaluation/phase2_metrics.py`
- `src/evaluation/phase2_report.py`
- `src/trainers/phase2_trainer.py`
- `tests/unit/trainers/test_phase2_selection_policy.py`
- `tests/unit/evaluation/test_phase2_metrics.py`

**配置变更**:

```python
@dataclass(frozen=True)
class Phase2SelectionPolicyConfig:
    selection_metric: str = "phase2_composite_score"
    primary_metric: str = "val_net_return"  # backward compatible fallback
    primary_mode: Literal["max", "min"] = "max"
    metric_weights: Dict[str, float] = field(default_factory=lambda: {
        "net_return": 1.0,
        "sharpe_ratio": 0.5,
        "max_drawdown": -0.5,
        "turnover": -0.1,
        "action_dominance_ratio": -0.2,
        "active_archetype_ratio": 0.2,
    })
    composite_score_sensitivity_perturbations: List[float] = field(
        default_factory=lambda: [-0.2, 0.2]
    )
```

**实现方案**:

1. 新增函数:

```python
def compute_phase2_composite_score(
    metrics: Mapping[str, float],
    weights: Mapping[str, float],
) -> float:
    # 缺失 metric 记 warning 或用 0.0，不静默吞掉关键字段。
```

2. `phase2_composite_metrics()` 输出 `phase2_composite_score`。

3. `Phase2SelectionPolicy.evaluate()` 优先读取 `selection_metric`；若缺失则 fallback 到 `primary_metric` 并记录 reason。

4. guardrails 仍为硬约束，composite score 只决定通过 guardrail 后是否 promote。

5. 实现 sensitivity:
   - 对每个权重做 `weight * (1 + perturbation)`。
   - 重新计算 checkpoint 排名。
   - 输出 rank 是否变化、score_delta、top_checkpoint_stable。

6. `Phase2ReportWriter.write_sensitivity()` 写 `composite_score_sensitivity_phase2.json`。

**验收标准**:
- best checkpoint 选择默认使用 `phase2_composite_score`。
- guardrail fail 的 checkpoint 即使 composite score 更高也不能 promote。
- sensitivity 输出可说明 best 是否对权重扰动敏感。

**测试计划**:
- composite score 正负权重计算正确。
- `selection_metric` 缺失时 fallback 并给出 reason。
- sensitivity 中扰动权重导致排名变化时 `top_checkpoint_stable=False`。

---

## 5. 批次 B: P1 sign-off 完整性修复

### B1. 校准 dead code mask 逻辑

**Review 问题**: dead code mask 使用 `count == 0`，未按 Phase I global usage ratio 阈值判断；backtest 脚本写死全 False。

**涉及文件**:
- `src/config/phase2_config.py`
- `src/trainers/phase2_trainer.py`
- `scripts/backtest_phase2.py`
- `src/models/archetype_selector.py`
- `src/rl/actor_critic.py`

**实现方案**:

1. `SelectorNetworkConfig` 增加:

```python
action_mask_dead_codes: bool = True
dead_code_usage_threshold: float = 0.01
```

2. 从 Phase I `phase1_report.json` 或 manifest 中读取 code usage:
   - 优先读取 `code_usage_ratio`。
   - 若只有 count，则用 `count / sum(counts)`。
   - 若字段缺失: 训练阶段 fail-fast；backtest 阶段 warning + 全 False。

3. mask 计算:

```python
dead_code_mask[k] = usage_ratio[k] < config.selector_network.dead_code_usage_threshold
```

4. 所有路径复用同一个 helper:

```python
def build_dead_code_mask(phase1_report: Mapping[str, Any], num_codes: int, threshold: float) -> List[bool]:
    ...
```

**验收标准**:
- usage ratio 小于 1% 的 code 被 mask。
- mask 能写入 report。
- `scripts/backtest_phase2.py` 不再写死 `[False] * num_codes`。

---

### B2. 补齐 phase2_report.json 审计字段

**Review 问题**: report summary 字段约 20 个，设计要求约 60 个，缺少调度、gap、norm、env shards、reward scaling、baseline、rolling、stress、OOD、resume、guardrail 审计。

**涉及文件**:
- `src/evaluation/phase2_report.py`
- `src/trainers/phase2_trainer.py`
- `src/evaluation/phase2_evaluator.py`
- `tests/unit/evaluation/test_phase2_report.py`

**实现方案**:

1. 扩展 `REQUIRED_PHASE2_REPORT_KEYS`，至少包含:
   - `horizon_schedule`
   - `data_gap_filter`
   - `input_norm`
   - `env_shards`
   - `reward_scaling`
   - `cost_config_inherited`
   - `baselines_val`
   - `baselines_test`
   - `rolling_validation_summary`
   - `execution_stress_summary`
   - `distribution_shift_warning_count`
   - `resume_ready`
   - `guardrails_pass`
   - `val_guardrails_pass`
   - `test_guardrails_pass_report_only`

2. 在 `Phase2Trainer` 中新增 report builder 小函数，避免继续扩大 `run()`:

```python
def _build_phase2_report_summary(
    self,
    train_result,
    val_result,
    test_result,
    rolling_result,
    ppo_stats,
    artifacts_meta,
) -> Dict[str, Any]:
    ...
```

3. 对缺失信息采用明确的状态字段:
   - 已实现: 写真实统计。
   - 未启用: `enabled=False` + `reason`。
   - 未实现但配置存在: `implemented=False`，禁止伪造 pass。

4. report schema 单测覆盖必填字段和缺失字段报错。

**验收标准**:
- `phase2_report.json` 能回答: 用什么配置训练、用什么 checkpoint 选择、是否通过 val guardrail、test 是否仅 report-only、rolling/stress/OOD 是否执行。
- schema 校验缺关键 sign-off 字段时失败。

---

### B3. 补全配置层缺失字段与 strict reproduction

**Review 问题**: 多个设计必配字段缺失，`paper_strict_reproduction` 存在但未应用 override。

**涉及文件**:
- `src/config/phase2_config.py`
- `scripts/train_phase2.py`
- `scripts/backtest_phase2.py`
- `tests/unit/config` 或新增 `tests/unit/test_phase2_config.py`

**实现方案**:

1. 新增配置 dataclass:
   - `CostAlignmentCheckConfig`
   - `EarlyStoppingConfig`
   - `ResumeConfig`
   - `DeploymentLadderConfig`
   - `EnvShardsConfig`
   - `StateDimBreakdownConfig`
   - `RewardNormalizationConfig`

2. 扩展已有配置:
   - `HorizonScheduleConfig`: `walk_forward_enabled`, `walk_forward_seed`, `chunk_reset_position`, gap minute threshold, reward alignment lookahead check。
   - `PPOConfig`: `value_clip_range`, `entropy_warmup_coef`, `entropy_warmup_fraction`, `kl_demo_label_smoothing`, `kl_demo_anneal_to`, `kl_demo_anneal_fraction`, `batch_size`, `lr_schedule`, `reward_normalization`。
   - `SelectorNetworkConfig`: `input_norm`, `position_encoding`。

3. 实现:

```python
def apply_paper_strict_overrides(self) -> "Phase2Config":
    # 返回 replace(...) 后的新 config。
```

4. 在 `from_dict()` 或训练入口加载后调用:

```python
config = Phase2Config.from_dict(payload)
if config.paper_strict_reproduction:
    config = config.apply_paper_strict_overrides()
```

5. 增加 Phase I/II 一致性校验:
   - `max_position`
   - `cost_config`
   - `reward_alignment`
   - `input_schema`

**验收标准**:
- 旧 yaml 能加载，新字段有默认值。
- strict reproduction 开启后，最终落盘 `phase2_config.yaml` 显示 override 后的配置。
- cost/max_position 不一致时 fail-fast。

---

### B4. reward scaling clip 与 explained variance

**Review 问题**: `reward_scaling.clip_range` 未使用；`PPOUpdateStats.explained_variance` 始终为 0。

**涉及文件**:
- `src/rl/ppo_trainer.py`
- `src/rl/rollout_buffer.py`
- `src/evaluation/metrics/policy_health.py`
- `tests/unit/rl/test_ppo_trainer.py`

**实现方案**:

1. `_scale_reward()` 同时返回 scaled 和 clipped 标记:

```python
def _scale_reward(self, reward: float) -> Tuple[float, bool]:
    scaled = reward / horizon if method == "divide_by_horizon" else reward
    if clip_range is not None:
        clipped = float(np.clip(scaled, -clip_range, clip_range))
        return clipped, clipped != scaled
    return scaled, False
```

2. `RolloutSample` 增加 `reward_was_clipped`。

3. `RolloutBuffer.get_stats()` 输出:
   - `reward_clipped_ratio`
   - `reward_unclipped_mean`
   - `reward_unclipped_std`
   - `reward_scaled_mean`

4. `PPOTrainer.update()` 在 GAE 后使用 returns/value 计算 explained variance:

```python
stats.explained_variance = compute_explained_variance(values, returns)
```

**验收标准**:
- 开启 clip 后 clipped ratio > 0 时 report 中可见。
- explained variance 不再恒为 0。

---

### B5. Horizon index 和 Dataset 防御性校验

**Review 问题**: horizon index 产物字段缺失；`Phase2Dataset.__init__` 缺 input schema、timestamp、split 校验。

**涉及文件**:
- `src/data/phase2_horizon_index.py`
- `src/data/phase2_dataset.py`
- `tests/unit/data/test_phase2_horizon_index.py`
- `tests/unit/data/test_phase2_dataset.py`

**实现方案**:

1. horizon index 补齐字段:
   - `last_execution_row`
   - `last_markout_row`
   - `phase1_sample_id`
   - `prev_terminal_position`
   - `timestamp_start`
   - `max_timestamp_gap_minutes`

2. 保持旧字段兼容:
   - `horizon_start` 可继续存在，但同步输出 `start_index`。
   - `horizon_end` 可继续存在，但同步输出 `end_index`。

3. `Phase2Dataset.__init__` 增加校验:
   - `feature_columns == input_schema.json`。
   - timestamp 单调递增。
   - `horizon_entries` 的 split 与 dataset frame split 一致。
   - `horizon_end` 不超过 frame 长度。

**验收标准**:
- 产物 schema 与设计字段对齐。
- schema/timestamp/split 错误能在 dataset 构造阶段失败，而不是训练中途失败。

---

## 6. 批次 C: 部署前评估与实验能力

### C1. KL/demo 消融矩阵自动编排

**目标**: 自动运行 `kl_demo_coef in {0, 0.1, 0.5, 1.0}`，输出 `phase2_ablation_kl_demo.json` 和 `phase2_ablation_summary.csv`。

**实现要点**:
- 在 `scripts/train_phase2.py` 增加 `--run-kl-demo-ablation`。
- 每个 alpha 生成独立 phase2 batch suffix，例如 `{batch_id}_kl{alpha}`。
- 汇总 val/test composite score、KL、action dominance、active archetype ratio。
- 主训练不默认触发，避免训练成本意外放大。

---

### C2. Execution Stress runner

**目标**: 落实 commission/slippage/execution lag 压力测试。

**实现要点**:
- 新增 `Phase2ExecutionStressRunner` 或在 evaluator 中新增 `evaluate_execution_stress()`。
- 对 `ExecutionStressConfig` 中的倍率组合执行 test report-only backtest。
- 输出:
  - `phase2_execution_stress.json`
  - 每组 `net_return`, `max_drawdown`, `sharpe_ratio`, `turnover`, `pass_guardrail`
- 不参与 checkpoint 选择。

---

### C3. OOD / Distribution Shift 监控接入

**目标**: 将已有 distribution shift 配置接入评估路径和 report。

**实现要点**:
- 训练集拟合 baseline stats。
- val/test 计算 zscore/PSI/mahalanobis。
- 输出 `distribution_shift_warning_count`、top features、fallback action 触发次数。
- 如果 OOD 超阈，只作为 deployment readiness warning，不反向改训练结果。

---

### C4. Online Action Throttle 接入推理路径

**目标**: backtest/live inference 支持 selector 动作节流。

**实现要点**:
- 新增 throttle state:
  - rolling switch count
  - last chosen code
  - cooldown counter
  - confidence threshold
- 在 `Phase2BacktestRunner.run_walk_forward()` action 产生后、decode 前应用 throttle。
- record 增加:
  - `throttle_triggered`
  - `original_code`
  - `throttled_code`
  - `selector_confidence`

---

### C5. Live risk control 完整回测路径

**目标**: 让 mid-horizon emergency flatten 在 backtest 中可审计。

**实现要点**:
- 将 `HorizonEnv._handle_mid_horizon_flatten()` 的逻辑抽到共享 helper，避免 env/backtest 两套实现。
- `Phase2BacktestRunner` 调用相同风控 helper。
- per-horizon record 写 `risk_triggered`, `risk_trigger_step`, `risk_reason`。

---

## 7. 批次 D: 工程结构优化

### D1. 拆分 `Phase2Trainer.run()`

**目标**: 降低 `Phase2Trainer.run()` 的编排耦合，便于测试。

**建议拆分**:
- `_load_phase1_artifacts()`
- `_build_horizon_index()`
- `_build_datasets()`
- `_build_selector_and_ppo()`
- `_run_training_loop()`
- `_run_final_evaluations()`
- `_write_reports()`

**验收标准**:
- `run()` 保持高层编排，单个方法不超过约 80 行。
- 每个私有方法可单独单测或 smoke test。

---

### D2. 拆分 Phase I artifact validator

**目标**: 将 `Phase1ArtifactValidator` 从 `phase2_horizon_index.py` 拆出。

**建议新文件**:
- `src/data/phase1_artifact_validator.py`

**验收标准**:
- horizon indexer 只负责 horizon 生成。
- Phase I 产物校验可被 train/backtest 复用。

---

### D3. ScheduleManager 退火参数配置化

**目标**: 消除 KL demo 和 entropy schedule 硬编码。

**实现要点**:
- `kl_demo_anneal_to`
- `kl_demo_anneal_fraction`
- `entropy_warmup_coef`
- `entropy_warmup_fraction`
- `lr_schedule`
- report 写最终 schedule 曲线摘要。

---

## 8. 变更顺序建议

### Step 1: P0 bugfix commit

包含:
- A1 rolling validation subset runner。
- A2 rollout truncated 标注。
- A3 `HorizonEnv.reset/restore_state`。

必须通过:
- `pytest tests/unit/trading/test_horizon_env.py`
- `pytest tests/unit/rl/test_ppo_trainer.py`
- `pytest tests/unit/evaluation/test_phase2_rolling_validation.py`
- `pytest tests/integration/test_phase2_gae_no_cross_env_leakage.py`
- `pytest tests/integration/test_phase2_resume_checkpoint.py`

### Step 2: checkpoint selection commit

包含:
- A4 composite score。
- sensitivity report。
- selection policy 单测。

必须通过:
- `pytest tests/unit/trainers/test_phase2_selection_policy.py`
- `pytest tests/unit/evaluation/test_phase2_metrics.py`
- `pytest tests/unit/evaluation/test_phase2_report.py`

### Step 3: audit/report/config commit

包含:
- B1 dead code mask。
- B2 report 字段。
- B3 config 字段与 strict reproduction。
- B4 reward clipping/explained variance。
- B5 index/dataset schema。

必须通过:
- `pytest tests/unit/data/test_phase2_horizon_index.py`
- `pytest tests/unit/data/test_phase2_dataset.py`
- `pytest tests/unit/evaluation/test_phase2_report.py`
- `pytest tests/unit/trainers/test_phase2_trainer.py`

### Step 4: deployment evaluation commit

包含:
- C1 KL/demo ablation。
- C2 execution stress。
- C3 OOD。
- C4 online throttle。
- C5 live risk backtest path。

必须通过:
- `pytest tests/unit/evaluation/test_phase2_execution_stress.py`
- `pytest tests/unit/evaluation/test_phase2_distribution_shift.py`
- `pytest tests/unit/evaluation/test_phase2_online_action_throttle.py`
- `pytest tests/unit/evaluation/test_phase2_live_risk_controls.py`
- `pytest tests/integration/test_phase2_backtest_walk_forward.py`

### Step 5: refactor commit

包含:
- D1 trainer 拆分。
- D2 validator 拆分。
- D3 schedule 配置化。

必须通过:
- Phase II 全量 unit tests。
- Phase II pipeline smoke:
  - `pytest tests/integration/test_phase2_pipeline_smoke.py`

---

## 9. 风险与回滚策略

| 风险 | 触发点 | 缓解 |
| --- | --- | --- |
| rolling validation 子集语义改变历史指标 | A1 | report 记录 subset/fold 初始仓位策略；旧全量 val walk-forward 保持不变 |
| truncated 修复改变 PPO 训练曲线 | A2 | 增加 GAE 精确单测；记录 rollout truncated/done 统计 |
| reset 支持 prev position 后影响旧测试 | A3 | 默认参数保持 `prev_terminal_position=0` |
| composite score 改变 best checkpoint | A4 | report 同时保留旧 `primary_metric` 和新 `selection_metric` |
| report schema 变严导致旧 pipeline 失败 | B2 | 分阶段扩展 required keys；未实现字段先显式 `implemented=False` |
| config 字段增多导致旧 yaml 加载失败 | B3 | 全部新字段提供默认值；`from_dict()` 显式 nested map |

回滚策略:
- 每个批次独立提交。
- P0 bugfix 不依赖 P2 部署能力，可单独保留。
- 若 composite score 影响实验对比，可临时设置 `selection_metric=val_net_return` 回退旧选择逻辑，但 report 必须记录该回退。

---

## 10. 最终验收清单

Phase II 进入 Phase III 前，至少满足以下条件:

- [ ] rolling validation 每个 fold 真实独立执行 subset walk-forward。
- [ ] rollout buffer 尾部非 done 样本写入 `truncated=True`，GAE 使用 bootstrap。
- [ ] `HorizonEnv.reset()` 支持注入 `prev_terminal_position`，resume 不再绕过公开接口。
- [ ] checkpoint 选择默认使用 `phase2_composite_score`，guardrails 为硬约束。
- [ ] dead code mask 按 Phase I global usage ratio 阈值计算。
- [ ] `phase2_report.json` 包含 sign-off 所需审计字段。
- [ ] reward clipping 和 explained variance 进入 PPO 健康指标。
- [ ] horizon index schema 与设计字段对齐。
- [ ] dataset 构造阶段校验 input schema、timestamp 单调性和 split 一致性。
- [ ] execution stress、OOD、online throttle、live risk control 至少在 report-only 路径可执行。
- [ ] Phase II unit tests 和 pipeline smoke 全部通过。

