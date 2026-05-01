# Phase I 全量代码质量审查记录

**日期**: 2026-05-01  
**范围**: Phase I 设计文档对应的源码、训练入口、测试与产物写出逻辑。  
**审查依据**:

- `docs/design/phase1_archetype_discovery_design.md`
- `docs/plan/phase1_archetype_discovery_execution_plan.md`
- 当前 `src/`、`scripts/train_phase1.py`、`run_pipeline.sh`、`tests/`

本轮只做 review 和记录，未修改业务代码。

## 验证命令

```bash
pytest -q
```

结果: 收集阶段失败。当前环境未安装 `torch`，但 `src/models/vector_quantizer.py` 在 `torch is None` 时仍执行 `@torch.no_grad()`，导致:

```text
AttributeError: 'NoneType' object has no attribute 'no_grad'
```

```bash
pytest -q tests/unit/data tests/unit/trading tests/unit/planners \
  tests/unit/evaluation/metrics \
  tests/unit/evaluation/diagnostics/test_failure_case_report.py \
  tests/unit/evaluation/diagnostics/test_latent_visualization.py
```

结果: `106 passed, 1 skipped, 3 failed`。失败项:

- `tests/unit/data/test_schema.py::test_close_non_positive_raises`
- `tests/unit/data/test_stratified_sampler.py::test_min_gap_between_samples_enforced`
- `tests/unit/planners/test_demo_generator.py::test_fail_when_dataset_reject_rate_exceeds`

## 问题总览

| ID | 严重度 | 分类 | 摘要 |
| --- | --- | --- | --- |
| P1-001 | Critical | DP 算法 | DP 末步估值允许切换，但 forward 又强制复制倒数第二步，能生成明显次优 teacher |
| P1-002 | High | DP 质量门禁 | reject_transition 统计只看最终 replay 动作，无法发现 DP 候选转移因盘口深度不足被大量拒绝 |
| P1-003 | High | sign-off | prospective 对照只检查 batch id，不读取对照报告、不比较指标、不阻止 sign-off |
| P1-004 | High | 采样 | `min_gap_between_samples` 只在局部 pick 中生效，最终样本全局间距可小于阈值 |
| P1-005 | High | 产物 | `next_row_execution` 的 horizon label 行号导出 off-by-one |
| P1-006 | High | checkpoint | `_train_loop` 更新的 history 没返回，最终 `best_epoch` 可能始终是 0 |
| P1-007 | High | checkpoint | 无 best checkpoint 时仍导出 last/current 模型产物并返回 `best_vq_model.pt` 路径 |
| P1-008 | High | report | `phase1_composite_score` 没进入 epoch metrics/final report，最终 report 可能固定为 0 |
| P1-009 | Medium | 采样 | 采样健康检查结果被丢弃，report 缺 `window_overlap_ratio`、`min_sample_gap` 等验收字段 |
| P1-010 | Medium | 分层 | horizon/past 分层统计存在 off-by-one，prospective unknown 桶没有被过滤 |
| P1-011 | Medium | VQ 健康 | 默认开启 dead-code restart，但 trainer 从未调用 `restart_dead_codes` |
| P1-012 | Medium | 评估 | horizon boundary、epoch stability、confusion/per-class 等设计要求指标没有进入 evaluator/report |
| P1-013 | Medium | report | weighted reconstruction 指标 key 不一致，final report 的 `weighted_reconstruction_accuracy` 可能是默认 0 |
| P1-014 | Medium | 测试/依赖 | `torch` 缺失时模型模块 fallback 不完整，导致非 torch 子集测试也无法完整收集 |
| P1-015 | Medium | 测试 | schema 单测 fixture 在断言 validator 前先 ZeroDivision |
| P1-016 | Low | cache/审计 | `_schema_hash` 实际使用 `config_hash`，无法表达真实 schema 变化 |
| P1-017 | Low | 集成脚本 | `run_pipeline.sh` 调 Phase I 时不提供必填 train/val/test 文件参数 |

## 详细问题

### P1-001: DP 末步估值与实际 action 约束不一致

**位置**:

- `src/planners/single_trade_dp.py:188-234`
- `src/planners/single_trade_dp.py:236-253`
- `src/planners/single_trade_dp.py:103-108`

**现象**:

`_backward()` 在 `t = h - 1` 仍允许 `target_a != prev_a`，因此 `V[h-1]` 可能包含“最后一步换仓”的收益。但 `_forward()` 只遍历到 `h-2`，随后 `plan()` 又强制:

```python
actions[-1] = actions[-2]
```

这意味着 DP 估值时允许的末步切换，实际 action 序列里不会执行。估值与执行约束不一致。

**复现**:

零成本、两步价格 `[100, 100, 110]`，如果第一步切到 long 并复制到末步，`[2, 2]` replay 收益为 `10`；当前 planner 输出 `[1, 1]`，收益为 `0`。

**影响**:

Teacher demonstration 会系统性次优，后续 VQ 学到的是错误 archetype label。这个问题直接影响论文第一阶段的核心训练数据。

**建议**:

重新定义末步 DP 递推语义。若必须满足 `actions[N-1] = actions[N-2]`，则 `V[h-1][prev, c]` 应只允许保持 `prev`，或者把末步 reward 合并到 `t=h-2` 的同仓位延续收益中。补一个回归测试覆盖上述两步上涨例子。

### P1-002: reject_transition 统计没有统计 DP 候选转移拒绝

**位置**:

- `src/planners/single_trade_dp.py:172-180`
- `src/planners/single_trade_dp.py:255-273`
- `src/planners/demo_generator.py:104-112`

**现象**:

DP precompute 中确实会把深度不足转移标为 `valid=False`，但这些拒绝没有进入 `DPResult`。`Phase1DemoGenerator` 只统计 `_replay(actions)` 中最终被选 action 的 reject events。

如果盘口深度很薄，DP 会避开所有不可成交换仓，最后可能选择全 flat。此时 replay 没有 reject，`dataset_reject_rate` 仍为 0。

**测试证据**:

`tests/unit/planners/test_demo_generator.py::test_fail_when_dataset_reject_rate_exceeds` 当前失败: 预期极小深度触发 `RejectTransitionExceeded`，实际未抛出。

**影响**:

设计 §9.4 要求用 reject rate 阻止数据质量差的 DP demo 生成。当前实现会静默放过“所有交易机会都因深度不足而不可行”的数据，导致 no-trade 样本和 codebook 容量问题被掩盖。

**建议**:

让 `SingleTradeDPPlanner` 返回 precompute 阶段的 rejected transition 统计，至少包含 per-step/per-action-pair count。`DemoGenerator` 的 `dataset_reject_rate` 应基于候选转移拒绝，而不是仅基于最终 replay 路径。

### P1-003: prospective 对照 sign-off 未真正执行

**位置**:

- `src/trainers/phase1_trainer.py:315-327`
- `src/trainers/phase1_trainer.py:354-387`
- `src/trainers/phase1_trainer.py:819-821`
- `scripts/train_phase1.py:149-179`

**现象**:

当前逻辑只检查 hindsight 主实验是否传入了 `diagnostic_pair_batch_id`，但没有:

- 读取 diagnostic batch 的 `phase1_report.json`
- 比较 `val_return_capture_ratio`、`val_sharpe_ratio`、`val_max_drawdown`、`code_usage_ratio` 等差异
- 写入 `hindsight_vs_prospective_metric_delta`
- 超阈值时标记 `hindsight_bias_warning="exceeded"`
- 阻止 `best_vq_model.pt` 被声明为 sign-off 版本

`_hindsight_warning_triggered()` 当前固定返回 `False`。

**影响**:

设计中最重要的 hindsight leakage 约束没有落地。只要传一个字符串 batch id，主实验就能通过 sign-off 前置检查。

**建议**:

在主实验 final report 阶段读取 `artifacts/{pair}/{diagnostic_pair_batch_id}/phase1/phase1_report.json`，比较 `StratificationConfig.hindsight_vs_prospective_max_delta` 中的指标，并将结果写入 `sampling_leakage_diagnostics.json` 和 `phase1_report.json`。

### P1-004: 采样全局 min gap 没被 sampler 保证

**位置**:

- `src/data/stratified_sampler.py:208-240`
- `src/data/stratified_sampler.py:258-274`
- `src/data/stratified_sampler.py:282-303`

**现象**:

`_pick_with_gap()` 只检查当前调用内的 `chosen_starts`。每个 strata 单独 pick 后，跨 strata 的样本可能非常接近。补采样时也没有把已经选出的样本传进去一起检查。

**测试证据**:

`tests/unit/data/test_stratified_sampler.py::test_min_gap_between_samples_enforced` 当前失败，出现相邻 starts `8` 和 `13`，gap 为 `5`，小于配置的 `10`。

**影响**:

设计要求通过 `min_gap_between_samples` 降低高度重叠样本的时间自相关。当前 sampler 不能保证这一点，后续只能靠 health checker 失败兜底，无法稳定产出满足约束的采样集。

**建议**:

把全局已选窗口集合传入 `_pick_with_gap()`，或采样完成后做一次全局 repair。`sorted(set(sampled_indices))[:num_samples]` 也应避免破坏 strata 配额和随机性。

### P1-005: `next_row_execution` label 行号导出 off-by-one

**位置**:

- `src/trainers/phase1_trainer.py:721-735`
- `src/data/horizon_builder.py:121-128`
- `src/trading/reward_alignment.py:85-90`

**现象**:

`_export_horizon_labels()` 固定写:

```python
last_execution_row = rec.start_index + len(rec.execution_books) - 1
last_markout_row = rec.start_index + len(rec.execution_books)
```

这只适用于 `paper_formula`。在 `next_row_execution` 下，最后一步 execution row 是 `start + h`，markout row 是 `start + h + 1`。

**影响**:

`horizon_labels_*.feather` 的边界审计字段错误，Phase II/III 或后续复盘会误判 reward alignment 和 markout 边界。

**建议**:

`HorizonRecord` 保留 `last_execution_row` / `last_markout_row`，由 `SampledHorizon` 或 `RewardAlignment` 在构建时写入；label 导出不要重新猜行号。

### P1-006: `_train_loop` 更新的 history 没返回给 `run()`

**位置**:

- `src/trainers/phase1_trainer.py:237-263`
- `src/trainers/phase1_trainer.py:678-682`
- `src/trainers/phase1_trainer.py:303-311`

**现象**:

`history` 在 `_train_loop()` 内被重新赋值:

```python
history = policy.update_history(history, metrics_for_select, verdict)
```

但 `_train_loop()` 没有 return history，`run()` 外层的 `history` 仍是初始对象。因此 final report 使用的:

```python
best_epoch=history.best_epoch or 0
```

可能始终为 `0`。

**影响**:

`phase1_report.json` 的 `best_epoch` 不可信，影响 checkpoint 审计和实验复现。

**建议**:

让 `_train_loop()` 返回最终 `SelectionHistory`，或把 history 作为 trainer 状态维护。

### P1-007: 无 best checkpoint 时仍导出 last/current 产物

**位置**:

- `src/trainers/phase1_trainer.py:268-282`
- `src/trainers/phase1_trainer.py:331-349`
- `src/trainers/phase1_trainer.py:811-814`

**现象**:

如果所有 epoch 都被 guardrail reject，`best_vq_model.pt` 不存在。当前流程会:

- `best_state = None`
- horizon labels 使用当前 model 直接导出
- Phase II artifacts 使用 `_snapshot_state(model)` 导出
- 返回 `best_vq_model` 路径
- final report 仍写 `best_checkpoint_path`

**影响**:

设计 §9.7 要求 `encoder.pt`、`decoder.pt`、`codebook.pt` 必须从 `best_vq_model.pt` 导出。当前可能在没有 best 的情况下产出可被后续阶段误用的 last/current 模型。

**建议**:

训练结束时若不存在 best checkpoint，应明确失败，或在 report 中标记 `no_signoff_best_checkpoint=true` 并禁止导出 Phase II artifacts。

### P1-008: `phase1_composite_score` 没进入 final report

**位置**:

- `src/trainers/selection_policy.py:77-82`
- `src/trainers/phase1_trainer.py:678-681`
- `src/trainers/phase1_trainer.py:779-789`
- `src/trainers/phase1_trainer.py:791-817`

**现象**:

selection policy 计算了 composite score，但 `ep_metrics.metrics` 没写入 `phase1_composite_score`。checkpoint manifest 记录了 verdict score，但 `epoch_metrics/*.json` 没有。最终 `_latest_metrics()` 读取的是 metrics JSON，`_build_final_summary()` 只能:

```python
summary.setdefault("phase1_composite_score", 0.0)
```

**影响**:

`phase1_report.json` 中的核心验收指标可能固定为 `0.0`，与 checkpoint 选择实际使用的分数不一致。

**建议**:

在 `metrics_for_select` 或 `ep_metrics.metrics` 中写入 `phase1_composite_score` 和 debug 信息，并确保 final report 读取 best epoch 的 metrics，而不是 latest epoch。

### P1-009: 采样健康报告被丢弃

**位置**:

- `src/trainers/phase1_trainer.py:439-458`
- `src/trainers/phase1_trainer.py:791-817`
- `src/data/sampling_health.py:17-27`

**现象**:

`checker.check(...)` 返回 `SamplingHealthReport`，但 trainer 没有保存返回值。最终 `phase1_report.json` 缺少设计 §9.2 要求的:

- `window_overlap_ratio`
- `min_sample_gap`
- `mean_sample_gap`
- `split_boundary_gap`
- `effective_min_gap_between_samples`
- `sampling_health_warnings`

**影响**:

采样健康即使通过，也无法审计；如果 `warn_only=true`，warning 也不会进入 final report。

**建议**:

把 train split 的 `SamplingHealthReport` 保存到 trainer 层，并合并进 final summary。

### P1-010: 分层统计 off-by-one 与 prospective unknown 桶

**位置**:

- `src/data/window_indexer.py:45-50`
- `src/data/window_indexer.py:73-80`
- `src/data/window_indexer.py:183-188`
- `src/data/stratified_sampler.py:57-67`
- `src/data/stratified_sampler.py:138-147`

**现象**:

设计 §3.4 中 hindsight horizon return 是:

```text
(close[t+h] - close[t]) / close[t]
```

当前 `_compute_window_stats(close, start, h)` 只读取 `start ... start+h-1`，实际变成 `(close[t+h-1] - close[t]) / close[t]`。

prospective 设计表中 past return 是 `(close[t] - close[t-L]) / close[t-L]`，当前 `_compute_past_stats()` 使用的是 `[start-lookback, start)`，最后一个价格是 `close[t-1]`。

此外，`enumerate()` 注释写“lookback 不足时采样阶段会丢弃 NaN strata”，但 sampler 会把它们归为 `unknown|unknown|mixed` 并正常参与采样。

**影响**:

分层样本覆盖与设计定义不一致；prospective 诊断可能被大量 unknown 早期窗口污染。

**建议**:

明确分层统计是否包含当前行和 markout 行，并与设计统一。prospective 模式下默认跳过 insufficient lookback 窗口，或至少把 unknown 桶比例写入 report 并默认阻止过高比例。

### P1-011: dead-code restart 默认开启但 trainer 未调用

**位置**:

- `src/models/vector_quantizer.py:228-267`
- `src/trainers/phase1_trainer.py:637-682`
- `src/trainers/selection_policy.py:84-98`

**现象**:

配置默认 `dead_code_restart=True`，`VectorQuantizer.restart_dead_codes()` 已实现，selection policy 也预留了 cooldown 逻辑。但 trainer 主循环从未计算 per-sample reconstruction error，也从未调用 restart。

**影响**:

设计 §9.5 要求 dead code 出现时默认执行 restart，并记录 `dead_code_restarts`。当前只会 reject/fatal collapse，不会执行设计中的恢复机制。

**建议**:

evaluator 或 train loop 暴露 per-sample reconstruction error，epoch 末按 `CodebookHealthConfig` 调用 restart，并把 `_dead_code_restart_triggered`、重启 code id 和来源样本写入 metrics/report。

### P1-012: 多项设计要求评估指标未进入 evaluator/report

**位置**:

- `src/evaluation/phase1_evaluator.py:90-304`
- `src/evaluation/phase1_replay.py:131-216`
- `src/evaluation/phase1_report.py:107-129`

**现象**:

设计 §9.5 与 §9.6 要求的部分指标/诊断没有进入主评估或最终 report:

- `horizon_boundary_turnover_cost`
- `horizon_boundary_position_consistency`
- `epoch_code_stability`
- `confusion_matrix`
- `action_precision_recall_per_class`
- `per_code_switch_point_distribution`
- `dp_teacher_return_distribution`
- latent snapshots / failure cases / action/risk/archetype diagnostics 文件

`Phase1ReplayEvaluator.evaluate_horizon_boundaries()` 已存在，但 `Phase1Evaluator` 没调用。`Phase1ReportWriter.write_diagnostics()` 也没有在 trainer 主流程中被调用。

**影响**:

当前产物无法满足设计文档的完整验收，尤其是 Phase II 关心的 horizon 边界持仓连续性风险。

**建议**:

先把必需验收指标接入 `Phase1Evaluator.evaluate_epoch()` 和 final report，再逐步接入 HTML/latent 诊断。

### P1-013: weighted reconstruction 指标 key 不一致

**位置**:

- `src/evaluation/phase1_evaluator.py:175-179`
- `src/evaluation/phase1_report.py:15-36`
- `src/trainers/phase1_trainer.py:795-797`

**现象**:

evaluator 写的是:

```python
out.metrics["val_weighted_reconstruction_accuracy"]
```

report schema 要求的是:

```python
"weighted_reconstruction_accuracy"
```

final summary 因此会补默认值 `0.0`，导致 report 中的 required key 不是实际验证指标。

**影响**:

验收报告容易误导，且指标命名和设计中 `val_weighted_reconstruction_accuracy` 不一致。

**建议**:

统一 metric key。建议 report 使用设计文档中的 `val_*` 命名，并兼容写入非 `val_` alias 时明确来源。

### P1-014: `torch` 缺失时 fallback 不完整，导致 pytest 收集失败

**位置**:

- `src/models/vector_quantizer.py:16-21`
- `src/models/vector_quantizer.py:87`
- `src/models/vector_quantizer.py:190`
- `src/models/vector_quantizer.py:228`
- `src/models/vector_quantizer.py:271`
- `src/models/vq_archetype.py:12-17`
- `src/models/vq_archetype.py:172`
- `requirements.txt:1-3`

**现象**:

`requirements.txt` 明确 `torch` 由 conda 或独立安装管理，不在 pip requirements 中。模型模块尝试在 ImportError 下设置 `torch = None`，但类定义阶段仍使用 `@torch.no_grad()`。

**影响**:

在未装 torch 的开发环境中，原本可以跳过的 torch 相关测试也无法收集，`pytest -q` 直接失败。

**建议**:

要么把 `torch` 作为测试必需依赖，要么移除模块级 `@torch.no_grad()` 装饰器，改成函数内部 `with torch.no_grad()` 或定义 no-op decorator。

### P1-015: schema 单测 fixture 自身先 ZeroDivision

**位置**:

- `tests/unit/data/test_schema.py:9-17`
- `tests/unit/data/test_schema.py:35-38`

**现象**:

`test_close_non_positive_raises()` 希望验证 validator 能拒绝 `close=0`，但 `_make_frame()` 先计算:

```python
(close[i] - close[i - 1]) / close[i - 1]
```

当上一根 close 是 0 时，测试在进入 validator 前就 `ZeroDivisionError`。

**影响**:

该测试没有覆盖目标逻辑，且会让数据单测失败。

**建议**:

测试 fixture 不应依赖待验证的非法 close 计算衍生收益；可直接传固定 `return_1m`，或在 close 非正测试中单独构造最小 frame。

### P1-016: `_schema_hash` 不是 schema hash

**位置**:

- `src/trainers/phase1_trainer.py:181`
- `src/data/demo_store.py:47-62`

**现象**:

trainer 里写:

```python
schema_hash = config_hash
```

`Phase1DemoStore` 随后把 `_schema_hash` 写入 demo/label 文件。

**影响**:

如果同一路径下数据 schema 变了但配置未变，artifact 中的 `_schema_hash` 无法表达真实 schema 变化，审计字段名与语义不符。

**建议**:

对 `schema.to_dict()` 做稳定 hash，单独写入 `_schema_hash`。

### P1-017: `run_pipeline.sh` Phase I 调用缺必填文件参数

**位置**:

- `run_pipeline.sh:34-38`
- `scripts/train_phase1.py:41-45`

**现象**:

`scripts/train_phase1.py` 要求 `--train-file`、`--val-file`、`--test-file`。`run_pipeline.sh` 默认调用只传:

```bash
python scripts/train_phase1.py --pair "${PAIR}" --train-batch-id "${BATCH_ID}" "${EXTRA_ARGS[@]}"
```

如果用户按脚本默认方式运行，会在 argparse 阶段失败。

**影响**:

全流程脚本与 Phase I CLI 不兼容，降低端到端可用性。

**建议**:

在脚本中补默认 `data/${PAIR}/train.feather` 等路径，或在脚本开头显式要求用户通过 `EXTRA_ARGS` 提供并做友好校验。

## 其他观察

- `NoTradeControlConfig` 当前基本未被 trainer 使用。`no_trade_ratio` 只在最终报告中统计，没有触发补采样、过滤或 warnings。设计 §9.4 对 no-trade ratio 的验收还未闭环。
- `Phase1DemoStore.save_demos()` 不保存 `execution_books`，但设计的数据契约中 horizon 样本包含该字段。若后续希望从 demo cache 重跑 replay 或 failure case，会缺必要盘口数据。
- `TrainingConfig.device`、`mixed_precision`、`early_stopping_patience` 当前未在 trainer 中使用，属于配置已暴露但行为未接线。
- `composite_score_sensitivity()` 当前只对一组 metrics 重算分数，没有按不同权重重新选择 best epoch；与设计 §9.5 的“观察 best epoch 是否漂移”仍有差距。

---

## 2026-05-01 执行结果与采纳状态

> 本节为追加执行记录，未修改上方原 review 内容。

### Review 意见采纳状态

| 标记 | ID/条目 | 采纳状态 |
| --- | --- | --- |
| 【✅】 | P1-001 | 已采纳。DP 末步估值与 action 复制约束已统一。 |
| 【✅】 | P1-002 | 已采纳。reject_transition 统计已覆盖 DP 候选拒绝。 |
| 【✅】 | P1-003 | 已采纳。prospective 对照 report 读取、delta 比较、sign-off 阻断已接入。 |
| 【✅】 | P1-004 | 已采纳。采样 min gap 已全局保证。 |
| 【✅】 | P1-005 | 已采纳。`next_row_execution` label 行号由真实边界字段传递。 |
| 【✅】 | P1-006 | 已采纳。训练循环返回更新后的 selection history。 |
| 【✅】 | P1-007 | 已采纳。无 best checkpoint 时禁止导出 Phase II artifacts。 |
| 【✅】 | P1-008 | 已采纳。composite score 写入 epoch metrics/final report。 |
| 【✅】 | P1-009 | 已采纳。sampling health 字段写入最终报告。 |
| 【✅】 | P1-010 | 已采纳。分层统计 off-by-one 已修复；prospective `unknown` 桶过滤未作为本轮 bug 采纳。 |
| 【✅】 | P1-011 | 已采纳。dead-code restart 已接入 trainer。 |
| 【】 | P1-012 | 部分采纳，未标记为完成。核心 evaluator/report 指标已接入；latent snapshot/failure case 主流程触发仍未完整接线。 |
| 【✅】 | P1-013 | 已采纳。weighted reconstruction key 已统一。 |
| 【✅】 | P1-014 | 已采纳。torch fallback 收集失败已修复。 |
| 【✅】 | P1-015 | 已采纳。schema 单测 fixture 已修复。 |
| 【✅】 | P1-016 | 已采纳。`_schema_hash` 使用真实 schema hash。 |
| 【✅】 | P1-017 | 已采纳。`run_pipeline.sh` 已补 Phase I 默认文件参数。 |
| 【】 | NoTradeControlConfig 闭环 | 未采纳，仍需单独确认策略口径。 |
| 【✅】 | demo cache 保存 `execution_books` | 已采纳。`Phase1DemoStore` 已序列化并恢复 execution books。 |
| 【】 | `device` / `mixed_precision` / `early_stopping_patience` | 未采纳，仍作为配置接线增强项。 |
| 【✅】 | composite score sensitivity 重新选择 best epoch | 已采纳。当前对所有 epoch metrics 重新打分并检测 best epoch drift。 |

### 本次补充调整

| 标记 | 文件 | 内容 |
| --- | --- | --- |
| 【✅】 | `tests/integration/test_phase1_pipeline_smoke.py` | smoke test 显式放宽 risk/behavior guardrail，避免与 P1-007 的“无 best 不导出”保护逻辑冲突；生产代码行为不变。 |

### 验证结果

| 标记 | 命令 | 结果 |
| --- | --- | --- |
| 【✅】 | `source /home/lanceliang/miniconda3/etc/profile.d/conda.sh && conda activate ArchetypeTrade && pytest -q` | `375 passed, 17 warnings`。warnings 为未注册 `pytest.mark.integration`。 |
| 【✅】 | `source /home/lanceliang/miniconda3/etc/profile.d/conda.sh && conda activate ArchetypeTrade && bash -n run_pipeline.sh` | 语法检查通过。 |
