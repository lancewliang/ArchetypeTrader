# Phase II Review 问题修复变更执行计划

**日期**: 2026-05-01
**最后更新**: 2026-05-01（完成状态标记）
**来源 Review**: `docs/review/20260501_phase2_full_code_quality_review.md`
**设计依据**:
- `docs/design/phase2_archetype_selection_design.md`
- `docs/plan/phase2_archetype_selection_execution_plan.md`

---

## 0. 修复状态总览

| 编号 | 项目 | 优先级 | 状态 | 备注 |
| --- | --- | --- | --- | --- |
| A1 | Rolling Validation 真实 fold 执行 | P0 | ✅ 已修复 | entry_indices/fold_id/initial_position 已实现 |
| A2 | truncated 语义 + GAE bootstrap | P0 | ✅ 已修复 | PPOTrainer 标注 truncated，GAE 按 done/truncated 分支 |
| A3 | HorizonEnv reset/restore 仓位继承 | P0 | ✅ 已修复 | reset() 接受 prev_terminal_position，restore_state() 公开接口 |
| A4 | Composite Score 选择 best checkpoint | P0 | ✅ 已修复 | compute_phase2_composite_score + sensitivity 分析 |
| B1 | Dead code mask 阈值校准 | P1 | ✅ 已修复 | build_dead_code_mask() 使用 Phase I usage ratio + threshold |
| B2 | phase2_report.json 审计字段补齐 | P1 | ✅ 已修复 | PHASE2_AUDIT_REPORT_KEYS + trainer report builder |
| B3 | 配置层缺失字段 + strict reproduction | P1 | ✅ 已修复 | 7 个新 dataclass + apply_paper_strict_overrides() |
| B4 | Reward scaling clip + explained variance | P1 | ✅ 已修复 | _scale_reward() 返回 clipped 标记 + compute_explained_variance |
| B5 | Horizon index + Dataset 防御性校验 | P1 | ✅ 已修复 | 6 个新字段 + _validate_inputs() 5 项校验 |
| C1 | KL/demo 消融矩阵 | P2 | ✅ 已修复 | --run-kl-demo-ablation CLI + run_kl_demo_ablation() |
| C2 | Execution Stress runner | P2 | ✅ 已修复 | Phase2ExecutionStressRunner + trainer._execution_stress_summary() |
| C3 | OOD / Distribution Shift 监控 | P2 | ✅ 已修复 | Phase2DistributionShiftMonitor + trainer._distribution_shift_summary() |
| C4 | Online Action Throttle | P2 | ✅ 已修复 | Phase2OnlineActionThrottle + replay 集成 |
| C5 | Live risk control 完整回测路径 | P2 | ✅ 已修复 | replay 中应用 live_risk_controls + risk_triggered/reason 字段 |
| D1 | 拆分 Phase2Trainer.run() | P3 | ❌ 未修复 | run() 仍为单一大方法，仅新增了若干私有 helper |
| D2 | 拆分 Phase1ArtifactValidator | P3 | ❌ 未修复 | 仍在 phase2_horizon_index.py 中 |
| D3 | ScheduleManager 退火参数配置化 | P3 | ✅ 已修复 | kl_demo_anneal_to/anneal_fraction/entropy_warmup 均可配置 |

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

### A1. 修复 Rolling Validation 真实 fold 执行 ✅ 已修复

**Review 问题**: `Phase2Evaluator.evaluate_rolling_validation()` 每个 fold 都运行完整 val walk-forward，再切片 records，导致 rolling validation 退化为同一结果拆片。

**涉及文件**:
- `src/evaluation/phase2_replay.py`
- `src/evaluation/phase2_evaluator.py`
- `src/evaluation/phase2_report.py`
- `tests/unit/evaluation/test_phase2_rolling_validation.py`
- `tests/unit/evaluation/test_phase2_replay.py`

**实际实现**:

1. `Phase2BacktestRunner.run_walk_forward()` 已增加 `entry_indices`, `initial_position`, `fold_id` 参数（`phase2_replay.py:L84-L92`）。

2. `_resolve_walk_forward_entries()` 内部解析函数已实现（`phase2_replay.py:L218-L242`），支持 dataset-global index 和 split-relative fallback。

3. `entry_indices=None` 时保持全量行为；非空时只运行指定 subset。

4. fold 内仓位连续:
   - `evaluate_rolling_validation()` 中 `initial_position` 从 0 开始，后续 fold 继承上一 fold 末仓位（`phase2_evaluator.py:L177-L199`）。
   - `RollingValidationResult` 记录 `fold_initial_position_policy`（"inherit_previous_fold" / "flat"）和 `fold_initial_positions`。

5. `Phase2Evaluator.evaluate_rolling_validation()` 构造一次 val entry index 列表，按 `num_folds` 切分后传 `entry_indices` 给 runner（`phase2_evaluator.py:L164-L229`）。

6. `Phase2HorizonReplayRecord` 已增加 `fold_id`, `fold_initial_position`, `timestamp_start` 字段（`phase2_replay.py:L50-L52`）。

7. `Phase2ReportWriter.write_rolling_validation()` 同时写 JSON 和 feather（`phase2_report.py:L117-L134`）。

**疑问/偏差**:
- 计划中提到 `prev_terminal_position` 字段在 horizon index 中补齐以支持 `initial_position_policy="from_index"`，当前实现中 horizon index 的 `prev_terminal_position` 字段为 `None`（`phase2_horizon_index.py:L326`），fold 初始仓位策略仅支持 "inherit_previous_fold" 和 "flat" 两种，未实现 "from_index" 模式。

---

### A2. 补齐 truncated 语义，修复 rollout 边界 GAE 语义 ✅ 已修复

**Review 问题**: `HorizonEnv.step()` 的 `truncated` 始终为 `False`，PPO rollout 到达 buffer 尾部但 episode 未结束时无法表达截断语义。

**涉及文件**:
- `src/rl/ppo_trainer.py`
- `src/rl/rollout_buffer.py`
- `tests/unit/rl/test_ppo_trainer.py`
- `tests/integration/test_phase2_gae_no_cross_env_leakage.py`

**实际实现**:

1. `HorizonEnv.step()` 保持 `truncated=False`（`horizon_env.py:L231`），由 collector 标注。

2. `PPOTrainer.collect_rollout()` 中明确标注 buffer 截断（`ppo_trainer.py:L164`）:
   ```python
   truncated = bool(env_truncated or (step == self.config.rollout_length - 1 and not done))
   ```

3. `RolloutSample.truncated` 写入上述 `truncated`（`ppo_trainer.py:L179`）。

4. `RolloutBuffer.compute_gae()` 正确处理（`rollout_buffer.py:L115-L176`）:
   - `done=True`: 不 bootstrap（`next_value=0, next_non_terminal=0`）。
   - `truncated=True and done=False`: 使用 `last_values` bootstrap（`next_value=value, next_non_terminal=1`）。

5. Debug 统计已增加（`rollout_buffer.py:L275-L277`）:
   - `rollout_truncated_count`
   - `rollout_done_count`
   - `rollout_bootstrap_count`

6. 这些字段进入 `RolloutBuffer.get_stats()` 和 `PPOUpdateStats`（`ppo_trainer.py:L57-L59`）。

**疑问/偏差**:
- 无显著偏差，实现与计划一致。

---

### A3. 修复 HorizonEnv reset/restore 的仓位继承 ✅ 已修复

**Review 问题**: `HorizonEnv.reset()` 总是把 `_prev_terminal_position` 重置为 0；多 env 训练和 resume 时仓位继承链断裂。

**涉及文件**:
- `src/trading/horizon_env.py`
- `src/rl/ppo_trainer.py`
- `src/trading/horizon_factory.py`
- `tests/unit/trading/test_horizon_env.py`
- `tests/integration/test_phase2_resume_checkpoint.py`

**实际实现**:

1. `HorizonEnv.reset()` 签名已修改（`horizon_env.py:L84-L88`）:
   ```python
   def reset(self, prev_terminal_position: int = 0, cursor: int = 0, reset_risk_state: bool = True) -> np.ndarray
   ```

2. 公开状态恢复接口 `restore_state()` 已实现（`horizon_env.py:L111-L128`），替代 `PPOTrainer.load_state()` 直接写私有属性。

3. `PPOTrainer.setup()` 初始化 env 时默认从 `prev_terminal_position=0` 开始。

4. `PPOTrainer.load_state()` 改为使用 `env.restore_state()`（`ppo_trainer.py:L446-L452`），不再直接访问 `env._cursor` / `env._prev_terminal_position`。

5. `PPOTrainer.get_state()` 保存 `cursor`, `prev_terminal_position`, `cumulative_loss`, `consecutive_losses`（`ppo_trainer.py:L416-L423`）。

6. 公开属性 `prev_terminal_position`, `cursor`, `cumulative_loss`, `consecutive_losses` 已通过 `@property` 暴露（`horizon_env.py:L308-L325`）。

**疑问/偏差**:
- 计划中提到 "若未来 `env_shards.chunk_reset_position == 'inherit'` 且 horizon index 提供 shard 起点仓位，则传入 shard 起点仓位"，当前实现中 `HorizonFactory` 未根据 `env_shards.chunk_reset_position` 传入 shard 起点仓位，所有 env 默认从 0 开始。

---

### A4. 实现 Composite Score 选择 best checkpoint ✅ 已修复

**Review 问题**: 当前 `Phase2SelectionPolicy` 只用单一 `val_net_return`，缺少设计要求的 composite score 和 sensitivity。

**涉及文件**:
- `src/config/phase2_config.py`
- `src/trainers/phase2_selection_policy.py`
- `src/evaluation/phase2_metrics.py`
- `src/evaluation/phase2_report.py`
- `src/trainers/phase2_trainer.py`
- `tests/unit/trainers/test_phase2_selection_policy.py`
- `tests/unit/evaluation/test_phase2_metrics.py`

**实际实现**:

1. `Phase2SelectionPolicyConfig` 已增加 `selection_metric`, `metric_weights`, `composite_score_sensitivity_perturbations`（`phase2_config.py:L138-L151`），默认值与计划一致。

2. `compute_phase2_composite_score()` 已实现（`phase2_metrics.py:L46-L72`），缺失 metric 记录到 `missing_metrics` 列表。

3. `phase2_composite_metrics()` 输出 `phase2_composite_score` 和 `phase2_composite_score_debug`（`phase2_metrics.py:L209-L214`）。

4. `Phase2SelectionPolicy.evaluate()` 优先读取 `selection_metric`；若缺失则 fallback 到 `primary_metric` 并记录 reason（`phase2_selection_policy.py:L74-L93`）。

5. Guardrails 仍为硬约束，composite score 只决定通过 guardrail 后是否 promote（`phase2_selection_policy.py:L96-L131`）。

6. `phase2_composite_score_sensitivity()` 已实现（`phase2_metrics.py:L75-L130`），输出 `top_checkpoint_stable`, `best_update_indices`, 每个扰动的 `score_delta`。

7. `Phase2ReportWriter.write_sensitivity()` 写 `composite_score_sensitivity_phase2.json`（`phase2_report.py:L113-L115`）。

**疑问/偏差**:
- 计划中 `risk` 和 `behavior` 独立子配置（`Phase2RiskGuardrailConfig` / `Phase2BehaviorGuardrailConfig`）未拆分，相关字段仍散落在 `Phase2SelectionPolicyConfig` 中。

---

## 5. 批次 B: P1 sign-off 完整性修复

### B1. 校准 dead code mask 逻辑 ✅ 已修复

**Review 问题**: dead code mask 使用 `count == 0`，未按 Phase I global usage ratio 阈值判断；backtest 脚本写死全 False。

**实际实现**:

1. `SelectorNetworkConfig` 已增加 `action_mask_dead_codes: bool = True` 和 `dead_code_usage_threshold: float = 0.01`（`phase2_config.py:L91-L92`）。

2. `build_dead_code_mask()` helper 已实现在 `src/trainers/phase2_dead_code.py`，从 Phase I report 读取 `code_usage_ratio` / `per_code_usage_ratio` / `usage_ratio` / `code_usage.counts` / `code_usage_counts`，自动转换为 ratio。

3. mask 计算: `dead_code_mask[k] = usage_ratio[k] < threshold`（`phase2_dead_code.py:L57`）。

4. `phase2_trainer.py` 使用 `build_dead_code_mask()`（`phase2_trainer.py:L233-L238`）。

5. `backtest_phase2.py` 不再写死 `[False] * num_codes`，改用 `build_dead_code_mask()`（`backtest_phase2.py:L113-L117`）。

6. 缺失 Phase I report 时返回全 False（训练阶段应在调用前 fail-fast）。

**疑问/偏差**:
- 计划中提到 "训练阶段 fail-fast；backtest 阶段 warning + 全 False"，当前实现中训练阶段未对缺失 usage 数据做 fail-fast，而是静默返回全 False。

---

### B2. 补齐 phase2_report.json 审计字段 ✅ 已修复

**Review 问题**: report summary 字段约 20 个，设计要求约 60 个，缺少调度、gap、norm、env shards、reward scaling、baseline、rolling、stress、OOD、resume、guardrail 审计。

**实际实现**:

1. `PHASE2_AUDIT_REPORT_KEYS` 已扩展（`phase2_report.py:L36-L52`），包含 `horizon_schedule`, `data_gap_filter`, `input_norm`, `env_shards`, `reward_scaling`, `cost_config_inherited`, `baselines_val`, `baselines_test`, `rolling_validation_summary`, `execution_stress_summary`, `distribution_shift_warning_count`, `resume_ready`, `guardrails_pass`, `val_guardrails_pass`, `test_guardrails_pass_report_only`。

2. `phase2_trainer.py` `run()` 中 `report_summary` 已包含所有审计字段（`phase2_trainer.py:L523-L625`）。

3. 缺失信息采用明确状态字段:
   - `reward_normalization.implemented=False`（`phase2_trainer.py:L571`）
   - `reward_normalization.reward_normalization_rejected_for_signoff`（`phase2_trainer.py:L572-L574`）

4. `validate_schema()` 校验必填字段（`phase2_report.py:L160-L166`）。

**疑问/偏差**:
- 计划中提到 "在 Phase2Trainer 中新增 report builder 小函数 `_build_phase2_report_summary()`"，当前实现中 report builder 逻辑仍内联在 `run()` 中（`phase2_trainer.py:L523-L625`），未拆分为独立方法。这与 D1（拆分 `run()`）未修复相关。

---

### B3. 补全配置层缺失字段与 strict reproduction ✅ 已修复

**Review 问题**: 多个设计必配字段缺失，`paper_strict_reproduction` 存在但未应用 override。

**实际实现**:

1. 新增配置 dataclass（`phase2_config.py`）:
   - `CostAlignmentCheckConfig`（L188-L191）
   - `EarlyStoppingConfig`（L194-L199）
   - `ResumeConfig`（L202-L208）
   - `DeploymentLadderConfig`（L211-L216）
   - `EnvShardsConfig`（L219-L223）
   - `StateDimBreakdownConfig`（L226-L230）
   - `RewardNormalizationConfig`（L175-L184）

2. 扩展已有配置:
   - `HorizonScheduleConfig`: `walk_forward_enabled`, `walk_forward_seed`, `chunk_reset_position`, `data_gap_check_enabled`, `max_allowed_gap_minutes`, `drop_gap_horizons`, `gap_position_carry_threshold_minutes`, `gap_large_reset_mode`, `reward_alignment_lookahead_check`（L53-L74）
   - `PPOConfig`: `value_clip_range`, `entropy_warmup_coef`, `entropy_warmup_fraction`, `kl_demo_label_smoothing`, `kl_demo_anneal_to`, `kl_demo_anneal_fraction`, `batch_size`, `lr_schedule`, `reward_normalization`（L100-L126）
   - `SelectorNetworkConfig`: `input_norm`, `position_encoding`, `action_mask_dead_codes`, `dead_code_usage_threshold`（L86-L94）

3. `apply_paper_strict_overrides()` 已实现（`phase2_config.py:L440-L467`），返回 `replace(...)` 后的新 config。

4. `from_dict()` 加载后自动调用 `apply_paper_strict_overrides()`（`phase2_config.py:L436-L437`）。

5. Phase I/II 一致性校验在 `Phase1ArtifactValidator.validate()` 中实现（`phase2_horizon_index.py:L127-L140`），检查 `max_position` 和 `reward_alignment`。

**疑问/偏差**:
- `cost_config` 的一致性校验只检查了 `max_position` 和 `reward_alignment`，未检查 `commission_rate` / `book_levels` 等细节字段。

---

### B4. reward scaling clip 与 explained variance ✅ 已修复

**Review 问题**: `reward_scaling.clip_range` 未使用；`PPOUpdateStats.explained_variance` 始终为 0。

**实际实现**:

1. `_scale_reward()` 同时返回 scaled 和 clipped 标记（`ppo_trainer.py:L195-L206`）。

2. `RolloutSample` 增加 `reward_was_clipped`（`rollout_buffer.py:L38`）。

3. `RolloutBuffer.get_stats()` 输出 `reward_clipped_ratio`, `reward_unclipped_mean`, `reward_unclipped_std`, `rollout_done_count`, `rollout_truncated_count`, `rollout_bootstrap_count`（`rollout_buffer.py:L258-L278`）。

4. `PPOTrainer.update()` 使用 `compute_explained_variance()` 计算 explained variance（`ppo_trainer.py:L251-L254`），通过 `flat_values_returns()` 获取 GAE 后的 values/returns。

5. `PPOUpdateStats` 包含 `explained_variance`, `reward_clipped_ratio`, `reward_unclipped_mean`（`ppo_trainer.py:L50-L56`）。

**疑问/偏差**:
- 无显著偏差，实现与计划一致。

---

### B5. Horizon index 和 Dataset 防御性校验 ✅ 已修复

**Review 问题**: horizon index 产物字段缺失；`Phase2Dataset.__init__` 缺 input schema、timestamp、split 校验。

**实际实现**:

1. horizon index 补齐字段（`phase2_horizon_index.py` `Phase2HorizonEntry`，L153-L169）:
   - `last_execution_row` ✅
   - `last_markout_row` ✅
   - `phase1_sample_id` ✅
   - `prev_terminal_position` ✅（但生成时为 None）
   - `timestamp_start` ✅
   - `max_timestamp_gap_minutes` ✅

2. 旧字段兼容: `horizon_start` / `horizon_end` 保留，同时输出 `start_index` / `end_index`（`phase2_horizon_index.py:L367-L372`）。

3. `Phase2Dataset._validate_inputs()` 增加校验（`phase2_dataset.py:L115-L145`）:
   - feature_columns 在 frame 中存在（L117-L118）
   - price_column 在 frame 中存在（L119-L120）
   - timestamp 单调非递减（L122-L127）
   - horizon_entries split 一致性（L129-L138）
   - horizon_end 不超过 frame 长度（L139-L145）

**疑问/偏差**:
- `prev_terminal_position` 在 horizon index 生成时始终为 `None`（`phase2_horizon_index.py:L326`），需要后续在训练流程中填充。

---

## 6. 批次 C: 部署前评估与实验能力

### C1. KL/demo 消融矩阵自动编排 ✅ 已修复

**目标**: 自动运行 `kl_demo_coef in {0, 0.1, 0.5, 1.0}`，输出 `phase2_ablation_kl_demo.json` 和 `phase2_ablation_summary.csv`。

**实际实现**:
- `scripts/train_phase2.py` 增加 `--run-kl-demo-ablation`（L55）和 `--kl-demo-ablation-values`（L57-L59）。
- `run_kl_demo_ablation()` 函数已实现（`train_phase2.py:L128-L164`），每个 alpha 生成独立 phase2 batch suffix `{batch_id}_kl{tag}`。
- 输出 `phase2_ablation_kl_demo.json`（L161）。
- 主训练不默认触发。

**疑问/偏差**:
- 计划中提到输出 `phase2_ablation_summary.csv`，当前实现只输出 JSON，未输出 CSV。

---

### C2. Execution Stress runner ✅ 已修复

**目标**: 落实 commission/slippage/execution lag 压力测试。

**实际实现**:
- `src/evaluation/phase2_execution_stress.py` 已实现 `Phase2ExecutionStressRunner`（L27-L79）。
- 接受 callable `run_records` 以注入不同 cost 配置的 runner。
- 对 `ExecutionStressConfig` 中的倍率组合执行 test report-only backtest。
- 输出每组 `net_return`, `max_drawdown`, `sharpe_ratio`, `turnover`。
- `phase2_trainer.py` `_execution_stress_summary()` 方法已实现（L791-L846），结果写入 report。
- 不参与 checkpoint 选择。

**疑问/偏差**:
- 计划中提到输出 `phase2_execution_stress.json` 独立文件，当前实现将 stress summary 内联到 `phase2_report.json` 的 `execution_stress_summary` 字段中，未写独立文件。

---

### C3. OOD / Distribution Shift 监控接入 ✅ 已修复

**目标**: 将已有 distribution shift 配置接入评估路径和 report。

**实际实现**:
- `src/evaluation/phase2_distribution_shift.py` 已实现 `Phase2DistributionShiftMonitor`（L29-L76），支持 z-score OOD 检测。
- 训练集拟合 baseline stats（`fit()` 方法），val/test 计算 zscore（`score()` 方法）。
- `phase2_trainer.py` `_distribution_shift_summary()` 方法已实现（L741-L789）。
- 输出 `distribution_shift_warning_count`、`max_score_val`、`max_score_test`、`fallback_action`。
- OOD 超阈仅作为 deployment readiness warning，不反向改训练结果。

**疑问/偏差**:
- 计划中提到支持 PSI / mahalanobis，当前实现仅支持 zscore。配置中 `method` 字段存在但未在实现中区分。

---

### C4. Online Action Throttle 接入推理路径 ✅ 已修复

**目标**: backtest/live inference 支持 selector 动作节流。

**实际实现**:
- `src/evaluation/phase2_online_action_throttle.py` 已实现 `Phase2OnlineActionThrottle`（L20-L113），包含:
  - rolling switch count
  - last chosen code
  - cooldown counter
  - confidence threshold
- `Phase2BacktestRunner.run_walk_forward()` 在 action 产生后、decode 前应用 throttle（`phase2_replay.py:L119,L141-L143`）。
- `Phase2HorizonReplayRecord` 增加:
  - `throttle_triggered`（L55）
  - `original_code`（L56）
  - `throttled_code`（L57）
  - `selector_confidence`（L54）

**疑问/偏差**:
- 无显著偏差，实现与计划一致。

---

### C5. Live risk control 完整回测路径 ✅ 已修复

**目标**: 让 mid-horizon emergency flatten 在 backtest 中可审计。

**实际实现**:
- `phase2_replay.py` `run_walk_forward()` 中应用 live risk controls（L156-L172），逻辑与 `HorizonEnv._handle_mid_horizon_flatten()` 一致（均基于 config 中的 `daily_loss_limit` / `consecutive_loss_limit` / `flatten_on_trigger`）。
- per-horizon record 写 `risk_triggered`, `risk_trigger_step`, `risk_reason`（`phase2_replay.py:L203-L205`）。

**疑问/偏差**:
- 计划中提到 "将 `HorizonEnv._handle_mid_horizon_flatten()` 的逻辑抽到共享 helper，避免 env/backtest 两套实现"，当前实现中 replay 和 env 各自独立实现了风控逻辑，未抽取共享 helper。两处逻辑目前一致，但未来维护可能产生分歧。

---

## 7. 批次 D: 工程结构优化

### D1. 拆分 `Phase2Trainer.run()` ❌ 未修复

**目标**: 降低 `Phase2Trainer.run()` 的编排耦合，便于测试。

**当前状态**: `run()` 仍为单一大方法（约 540 行，`phase2_trainer.py:L107-L645`），但已新增若干私有 helper:
- `_validate_unsupported_features()`（L647-L653）
- `_history_from_manifest()`（L655-L676）
- `_maybe_resume()`（L678-L714）
- `_rolling_result_payload()`（L716-L726）
- `_is_metric_improved()`（L728-L739）
- `_distribution_shift_summary()`（L741-L789）
- `_execution_stress_summary()`（L791-L846）
- `_seed_everything()`（L848-L858）

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

### D2. 拆分 Phase I artifact validator ❌ 未修复

**目标**: 将 `Phase1ArtifactValidator` 从 `phase2_horizon_index.py` 拆出。

**当前状态**: `Phase1ArtifactValidator` 仍在 `src/data/phase2_horizon_index.py` 中（L48-L150），未拆分到独立模块 `src/data/phase1_artifact_validator.py`。

**建议新文件**:
- `src/data/phase1_artifact_validator.py`

**验收标准**:
- horizon indexer 只负责 horizon 生成。
- Phase I 产物校验可被 train/backtest 复用。

---

### D3. ScheduleManager 退火参数配置化 ✅ 已修复

**目标**: 消除 KL demo 和 entropy schedule 硬编码。

**实际实现**:
- `kl_demo_anneal_to` 可配置（`phase2_config.py:L114`），`ScheduleManager` 读取该值（`scheduling.py:L91-L99`）。
- `kl_demo_anneal_fraction` 可配置（`phase2_config.py:L115`），控制退火进度（`scheduling.py:L92-L99`）。
- `entropy_warmup_coef` / `entropy_warmup_fraction` 可配置（`phase2_config.py:L110-L111`），`ScheduleManager` 实现 warmup 逻辑（`scheduling.py:L79-L88`）。
- `entropy_min_coef` 作为退火下限（`phase2_config.py:L109`），`ScheduleManager` 使用 `max(current, entropy_min_coef)`（`scheduling.py:L85-L88`）。
- `lr_schedule` 支持 `constant` / `linear`（`phase2_config.py:L125`），`ScheduleManager` 实现（`scheduling.py:L67-L70`）。
- report 写入 `entropy_schedule` 和 `kl_demo_dominance_ratio`（`phase2_trainer.py:L610-L618`）。

**疑问/偏差**:
- 无显著偏差，实现与计划一致。

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

- [x] rolling validation 每个 fold 真实独立执行 subset walk-forward。
- [x] rollout buffer 尾部非 done 样本写入 `truncated=True`，GAE 使用 bootstrap。
- [x] `HorizonEnv.reset()` 支持注入 `prev_terminal_position`，resume 不再绕过公开接口。
- [x] checkpoint 选择默认使用 `phase2_composite_score`，guardrails 为硬约束。
- [x] dead code mask 按 Phase I global usage ratio 阈值计算。
- [x] `phase2_report.json` 包含 sign-off 所需审计字段。
- [x] reward clipping 和 explained variance 进入 PPO 健康指标。
- [x] horizon index schema 与设计字段对齐。
- [x] dataset 构造阶段校验 input schema、timestamp 单调性和 split 一致性。
- [x] execution stress、OOD、online throttle、live risk control 至少在 report-only 路径可执行。
- [ ] Phase II unit tests 和 pipeline smoke 全部通过。
- [ ] Phase2Trainer.run() 拆分为可测试的私有方法（D1 未修复）。
- [ ] Phase1ArtifactValidator 拆分到独立模块（D2 未修复）。

