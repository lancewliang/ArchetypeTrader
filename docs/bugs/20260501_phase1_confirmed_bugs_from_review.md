# Phase I Review 明确 Bug 筛选清单

**日期**: 2026-05-01
**来源**:

- `docs/review/20260501_phase1_full_code_quality_review.md`
- `docs/review/20260501_phase1_full_code_quality_review_v2.md`

## 筛选口径

本清单只收录 review 中可以明确归为 bug 的条目，而不是单纯设计建议或后续增强项。判定标准:

- 当前代码会产生错误结果、错误产物、错误报告或错误审计字段。
- 当前测试已失败，或 review 给出了可复现的错误行为。
- 配置、CLI 或质量门禁看似启用，但实际行为无效或与声明语义冲突。
- 对同一个 review ID，只收录其中明确是 bug 的部分；偏策略或设计取舍的部分会注明不纳入。

## 明确 Bug 总览

| ID | 严重度 | 类型 | 摘要 | 判定理由 | 修复状态 |
| --- | --- | --- | --- | --- | --- |
| P1-001 | Critical | 算法 | DP 末步估值允许切换，但 forward 又强制复制倒数第二步 | 估值约束和实际 action 约束矛盾，可生成次优 teacher | 已修复 |
| P1-002 | High | 质量门禁 | reject_transition 只统计最终 replay 路径，不统计 DP 候选拒绝 | 薄盘口下 DP 可避开全部不可成交转移，reject rate 被错误记为 0 | 已修复 |
| P1-003 | High | sign-off | prospective 对照 sign-off 固定不告警 | `_hindsight_warning_triggered()` 固定返回 `False`，质量门禁失效 | 已修复 |
| P1-004 | High | 采样 | `min_gap_between_samples` 未全局保证 | 跨 strata 和补采样会产生小于 min gap 的样本，已有失败测试 | 已修复 |
| P1-005 | High | 产物 | `next_row_execution` horizon label 行号 off-by-one | 导出的 execution/markout 边界行号错误 | 已修复 |
| P1-006 | High | checkpoint/report | `_train_loop` 更新后的 history 没返回给 `run()` | final report 的 `best_epoch` 可能固定为 0 | 已修复 |
| P1-007 | High | checkpoint/产物 | 无 best checkpoint 时仍导出 last/current 模型产物 | 下游会误用并不存在 sign-off 的模型产物 | 已修复 |
| P1-008 | High | report | `phase1_composite_score` 未进入 epoch metrics/final report | final report 可能把核心分数写成默认 0 | 已修复 |
| P1-009 | Medium | report | 采样健康检查结果被丢弃 | checker 已执行但返回值未保存，report 缺审计字段 | 已修复 |
| P1-010 | Medium | 分层 | horizon/past return 分层统计 off-by-one | 计算窗口和设计公式错一行；unknown 桶策略不在本清单中判为 bug | 已修复 |
| P1-011 | Medium | VQ 健康 | `dead_code_restart=True` 但 trainer 从未调用 restart | 默认开启的恢复机制实际无效 | 已修复 |
| P1-013 | Medium | report | weighted reconstruction 指标 key 不一致 | evaluator 写 `val_*`，report 读无前缀 key，最终 fallback 为 0 | 已修复 |
| P1-014 | Medium | 测试/依赖 | `torch` 缺失时 fallback 不完整 | `torch=None` 时模块级 `@torch.no_grad()` 直接导致 pytest 收集失败 | 已修复 |
| P1-015 | Medium | 测试 | schema 单测 fixture 自身先 ZeroDivision | 测试未进入 validator，覆盖目标失效 | 已修复 |
| P1-016 | Low | 审计 | `_schema_hash` 实际写的是 `config_hash` | 字段名和语义冲突，无法表达真实 schema 变化 | 已修复 |
| P1-017 | Low | 集成脚本 | `run_pipeline.sh` Phase I 调用缺必填文件参数 | 默认脚本路径会在 argparse 阶段失败 | 已修复 |
| P1-019 | Medium | report | final report 读取 latest epoch metrics，而非 best epoch metrics | report 指标和 `best_checkpoint_path` 可能不一致 | 已修复 |

## 修复状态摘要

**当前状态**: 上表 17 个明确 bug 均已修复。

**验证命令**:

```bash
pytest -q
bash -n run_pipeline.sh
```

**最近验证结果**: `153 passed, 8 skipped`，`run_pipeline.sh` 语法检查通过。

**主要修复落点**:

- P1-001/P1-002: `src/planners/single_trade_dp.py`, `src/planners/demo_generator.py`
- P1-004/P1-010: `src/data/stratified_sampler.py`, `src/data/window_indexer.py`
- P1-005/P1-006/P1-007/P1-008/P1-009/P1-011/P1-013/P1-016/P1-019: `src/trainers/phase1_trainer.py`, `src/evaluation/phase1_evaluator.py`, `src/data/horizon_builder.py`, `src/data/demo_store.py`
- P1-014: `src/models/vector_quantizer.py`, `src/models/vq_archetype.py`
- P1-015: `tests/unit/data/test_schema.py`
- P1-017: `run_pipeline.sh`

## 详细 Bug 条目

### P1-001: DP 末步估值与实际 action 约束不一致

**位置**:

- `src/planners/single_trade_dp.py`

**为什么是 bug**:

`_backward()` 在 `t = h - 1` 仍允许换仓，DP 估值中可以把最后一步切换收益计入价值函数；但 `_forward()` 只遍历到 `h - 2`，随后 `plan()` 强制 `actions[-1] = actions[-2]`。这会让 DP 在估值阶段依赖一个最终不会执行的动作。

**影响**:

Teacher demonstration 可能明显次优，直接污染 Phase I 训练标签。

**修复方向**:

末步 DP 递推只允许保持 `prev_a`，或把末步 reward 合并到 `t = h - 2` 的同仓位延续收益中，并补两步上涨场景的回归测试。

### P1-002: reject_transition 统计没有统计 DP 候选转移拒绝

**位置**:

- `src/planners/single_trade_dp.py`
- `src/planners/demo_generator.py`

**为什么是 bug**:

DP precompute 会把盘口深度不足的转移标成 invalid，但这些拒绝没有进入 `DPResult`。`Phase1DemoGenerator` 只看最终 replay 路径上的 reject events。如果 DP 避开所有不可成交换仓，dataset reject rate 会被错误记为 0。

**影响**:

数据质量门禁会静默放过薄盘口数据。review 中的 `test_fail_when_dataset_reject_rate_exceeds` 已体现该问题。

**修复方向**:

在 `DPResult` 中增加 precompute 阶段的拒绝计数和按 action pair 分组的统计，`DemoGenerator` 用候选拒绝统计计算 dataset reject rate。

### P1-003: prospective 对照 sign-off 未真正执行

**位置**:

- `src/trainers/phase1_trainer.py`
- `scripts/train_phase1.py`

**为什么是 bug**:

主流程只检查是否传入 `diagnostic_pair_batch_id`，但没有读取对照 batch 的 report，也没有比较指标差异。`_hindsight_warning_triggered()` 固定返回 `False`，因此超阈值时也不会阻止 sign-off。

**影响**:

hindsight leakage 质量门禁失效，主实验可能被错误声明为可 sign-off。

**修复方向**:

读取 paired prospective report，比较配置中定义的核心指标差异，写入 leakage diagnostics；超过阈值时标记 `hindsight_bias_warning="exceeded"` 并禁止 sign-off。

### P1-004: 采样全局 min gap 没被 sampler 保证

**位置**:

- `src/data/stratified_sampler.py`

**为什么是 bug**:

`_pick_with_gap()` 只检查当前 strata 内的局部选择。跨 strata 采样和补采样没有把全局已选窗口纳入 gap 检查，最终会出现相邻样本距离小于 `min_gap_between_samples`。

**影响**:

采样集不能稳定满足时间间隔约束，review 中的 `test_min_gap_between_samples_enforced` 已失败。

**修复方向**:

维护全局已选 `window_start` 集合，所有 strata 和补采样都基于同一 gap 约束挑选；避免最后排序截断破坏采样约束和随机性。

### P1-005: `next_row_execution` label 行号导出 off-by-one

**位置**:

- `src/trainers/phase1_trainer.py`
- `src/data/horizon_builder.py`
- `src/trading/reward_alignment.py`

**为什么是 bug**:

`_export_horizon_labels()` 固定按 `paper_formula` 计算最后 execution/markout 行号。在 `next_row_execution` 下，最后一步 execution row 应为 `start + h`，markout row 应为 `start + h + 1`，当前导出少了一行。

**影响**:

`horizon_labels_*.feather` 的边界审计字段错误，后续 Phase II/III 或复盘会误判 reward alignment。

**修复方向**:

在 horizon 构建阶段记录真实 `last_execution_row` / `last_markout_row`，label 导出直接使用记录值，不在导出阶段重新推断。

### P1-006: `_train_loop` 更新的 history 没返回给 `run()`

**位置**:

- `src/trainers/phase1_trainer.py`

**为什么是 bug**:

`_train_loop()` 内部调用 `policy.update_history()` 后把结果赋给局部变量 `history`，但函数没有返回更新后的 history。`run()` 外层仍持有初始 `SelectionHistory()`。

**影响**:

final report 的 `best_epoch` 可能始终为 0，checkpoint 审计不可信。

**修复方向**:

让 `_train_loop()` 返回最终 `SelectionHistory`，或把 history 作为 trainer 状态维护。

### P1-007: 无 best checkpoint 时仍导出 last/current 产物

**位置**:

- `src/trainers/phase1_trainer.py`

**为什么是 bug**:

如果所有 epoch 都被 guardrail reject，`best_vq_model.pt` 不存在。当前流程会用当前 model snapshot 导出 Phase II artifacts，并返回一个不存在的 `best_vq_model` 路径。

**影响**:

下游可能把未通过 sign-off 的 last/current 模型当成 best 模型使用。

**修复方向**:

训练结束时若不存在 best checkpoint，应明确失败或写入 `no_signoff_best_checkpoint=true`，并禁止导出 Phase II artifacts。

### P1-008: `phase1_composite_score` 没进入 final report

**位置**:

- `src/trainers/selection_policy.py`
- `src/trainers/phase1_trainer.py`

**为什么是 bug**:

selection policy 计算了 composite score，但 epoch metrics 没写入该 key。final summary 读取 metrics JSON 时找不到 `phase1_composite_score`，只能补默认值 0。

**影响**:

final report 的核心验收分数可能错误，且与 checkpoint 选择逻辑不一致。

**修复方向**:

把 `phase1_composite_score` 和 debug 信息写入 epoch metrics，并让 final report 读取 best epoch 对应 metrics。

### P1-009: 采样健康报告被丢弃

**位置**:

- `src/trainers/phase1_trainer.py`
- `src/data/sampling_health.py`

**为什么是 bug**:

`SamplingHealthChecker.check()` 已被调用，但返回的 `SamplingHealthReport` 没有被捕获或合并进 final summary。

**影响**:

即使采样健康检查已执行，产物中也缺少 `window_overlap_ratio`、`min_sample_gap`、`mean_sample_gap`、`split_boundary_gap` 等审计字段；warn-only 情况下 warning 也会丢失。

**修复方向**:

保存 train/val/test 的 sampling health report，并将关键字段合并到 `phase1_report.json`。

### P1-010: 分层统计 off-by-one

**位置**:

- `src/data/window_indexer.py`

**为什么是 bug**:

hindsight horizon return 设计语义是 `(close[t+h] - close[t]) / close[t]`，当前实现实际使用 `close[t+h-1]`。prospective past return 设计语义是 `(close[t] - close[t-L]) / close[t-L]`，当前实现实际使用 `close[t-1]`。

**影响**:

分层标签和设计定义错一行，采样覆盖会被系统性扰动。

**修复方向**:

统一窗口边界定义，明确 stats 是否包含当前行和 markout 行；修正 off-by-one 并补窗口边界单测。

**未纳入本 bug 的关联点**:

prospective unknown 桶是否默认过滤更偏策略/设计取舍，需单独确认验收口径。

### P1-011: dead-code restart 默认开启但 trainer 未调用

**位置**:

- `src/models/vector_quantizer.py`
- `src/trainers/phase1_trainer.py`
- `src/trainers/selection_policy.py`

**为什么是 bug**:

配置默认 `dead_code_restart=True`，`VectorQuantizer.restart_dead_codes()` 已实现，selection policy 也预留了 cooldown 字段；但 trainer 主循环没有任何调用路径。也就是说，默认开启的恢复机制实际不会发生。

**影响**:

codebook collapse 时只能 reject/fatal，无法执行配置声明的自动恢复。

**修复方向**:

epoch 末计算 dead-code 状态和候选样本，按配置调用 `restart_dead_codes()`，并把 restart 事件写入 metrics/report。

### P1-013: weighted reconstruction 指标 key 不一致

**位置**:

- `src/evaluation/phase1_evaluator.py`
- `src/evaluation/phase1_report.py`
- `src/trainers/phase1_trainer.py`

**为什么是 bug**:

evaluator 写入 `val_weighted_reconstruction_accuracy`，report schema 和 final summary 读取 `weighted_reconstruction_accuracy`。key 不一致会导致 report 中该指标 fallback 为 0。

**影响**:

验收报告中的 weighted reconstruction 指标错误。

**修复方向**:

统一指标 key，建议保留设计中的 `val_*` 命名；如需 alias，应明确从实际 val 指标复制。

### P1-014: `torch` 缺失时 fallback 不完整

**位置**:

- `src/models/vector_quantizer.py`
- `src/models/vq_archetype.py`

**为什么是 bug**:

模块 import 失败时把 `torch = None`，但类定义阶段仍执行 `@torch.no_grad()`。未安装 torch 时，pytest 在收集阶段直接 `AttributeError`。

**影响**:

非 torch 子集测试也无法完整收集。

**修复方向**:

要么把 torch 设为测试必需依赖，要么提供 no-op decorator 或改为函数内部 `with torch.no_grad()`。

### P1-015: schema 单测 fixture 自身先 ZeroDivision

**位置**:

- `tests/unit/data/test_schema.py`

**为什么是 bug**:

`test_close_non_positive_raises()` 期望验证 validator 拒绝非正 close，但 `_make_frame()` 在构造 `return_1m` 时先除以 0，测试根本没有进入 validator。

**影响**:

目标逻辑没有被覆盖，且数据单测失败。

**修复方向**:

非法 close 测试应绕开派生收益计算，直接构造固定 `return_1m` 或使用专门的最小 frame。

### P1-016: `_schema_hash` 不是 schema hash

**位置**:

- `src/trainers/phase1_trainer.py`
- `src/data/demo_store.py`

**为什么是 bug**:

trainer 把 `config_hash` 写入 `schema_hash` 变量，随后 `Phase1DemoStore` 写出 `_schema_hash`。字段名承诺的是 schema hash，但实际内容是 config hash。

**影响**:

artifact 无法表达真实 schema 变化，审计字段语义错误。

**修复方向**:

对 schema 的稳定序列化结果单独计算 hash，并写入 `_schema_hash`。

### P1-017: `run_pipeline.sh` Phase I 调用缺必填文件参数

**位置**:

- `run_pipeline.sh`
- `scripts/train_phase1.py`

**为什么是 bug**:

`scripts/train_phase1.py` 要求 `--train-file`、`--val-file`、`--test-file`，但 `run_pipeline.sh` 默认调用 Phase I 时不传这些必填参数。

**影响**:

按默认脚本运行会在 argparse 阶段失败。

**修复方向**:

脚本中补默认数据路径，或在脚本开头显式校验并提示用户通过 `EXTRA_ARGS` 提供。

### P1-019: final report 读取 latest 而非 best epoch metrics

**位置**:

- `src/trainers/phase1_trainer.py`

**为什么是 bug**:

`_latest_metrics()` 总是读取 manifest 最后一条 epoch metrics，而 final report 同时写 `best_checkpoint_path`。如果 best epoch 不是 last epoch，report 指标会和 best checkpoint 不一致。

**影响**:

最终报告无法准确描述被导出的 best 模型。

**修复方向**:

按 `best_epoch` 或 `best_checkpoint_path` 对应的 manifest entry 读取 metrics；没有 best 时禁止写正常 sign-off report。

## 暂不纳入“明确 Bug”的条目

| ID/条目 | 原因 |
| --- | --- |
| P1-012 | 多项 evaluator/report 指标未接入，整体更偏设计完整性和验收覆盖缺口；可拆成具体缺字段后再转 bug |
| P1-018 | `HorizonRecord` 缺字段更像 P1-005 的结构性根因，单独看是数据模型设计缺口 |
| P1-020 | `num_switches` 统计“不够健壮”，review 未给出当前可触发的错误输出 |
| NoTradeControlConfig 未使用 | 更偏未完成的设计闭环/功能接线 |
| `Phase1DemoStore.save_demos()` 不保存 `execution_books` | 目前 review 只指出后续 replay 可能缺数据，尚未证明当前流程错误 |
| `device` / `mixed_precision` / `early_stopping_patience` 未使用 | 属于配置接线缺口，需确认是否已承诺为当前阶段功能 |
| `composite_score_sensitivity()` 不重新选择 best epoch | 更偏诊断设计语义未完整实现 |

---

## 2026-05-01 执行结果追加

> 本节为追加执行记录，未修改上方原计划内容。

### 明确 Bug 执行状态

| 标记 | ID | 执行结果 |
| --- | --- | --- |
| 【✅】 | P1-001 | 已修复并验证。末步不可执行切换不再进入 DP 估值自由度，新增/保留两步上涨回归测试。 |
| 【✅】 | P1-002 | 已修复并验证。DP precompute 候选拒绝计入 `DPResult`，demo generator 用候选拒绝统计 dataset reject rate。 |
| 【✅】 | P1-003 | 已修复并验证。trainer 读取 prospective paired report，计算指标 delta，超阈值时写 `hindsight_bias_warning="exceeded"` 并阻止 sign-off。 |
| 【✅】 | P1-004 | 已修复并验证。采样阶段使用全局 `chosen_starts` 保证 min gap。 |
| 【✅】 | P1-005 | 已修复并验证。`WindowIndexEntry`/`SampledHorizon`/`HorizonRecord` 传递真实 `last_execution_row` 与 `last_markout_row`。 |
| 【✅】 | P1-006 | 已修复并验证。`_train_loop()` 返回更新后的 `SelectionHistory`。 |
| 【✅】 | P1-007 | 已修复并验证。无 best checkpoint 时 trainer 抛出 fatal，不导出 Phase II artifacts。 |
| 【✅】 | P1-008 | 已修复并验证。`phase1_composite_score` 与 debug 写入 epoch metrics/final report。 |
| 【✅】 | P1-009 | 已修复并验证。sampling health report 合并进入 final summary。 |
| 【✅】 | P1-010 | 已修复并验证。horizon/past return 边界按设计公式修正；`unknown` 桶策略未纳入本 bug。 |
| 【✅】 | P1-011 | 已修复并验证。trainer 主循环已调用 `restart_dead_codes()`，并写入 restart metrics/report。 |
| 【✅】 | P1-013 | 已修复并验证。`weighted_reconstruction_accuracy` 与 `val_weighted_reconstruction_accuracy` 建立一致 alias。 |
| 【✅】 | P1-014 | 已修复并验证。torch 缺失 fallback 不再因模块级 `@torch.no_grad()` 收集失败。 |
| 【✅】 | P1-015 | 已修复并验证。schema 非正 close 测试不再在 fixture 构造阶段 ZeroDivision。 |
| 【✅】 | P1-016 | 已修复并验证。`_schema_hash` 使用 schema 稳定序列化 hash，不再复用 config hash。 |
| 【✅】 | P1-017 | 已修复并验证。`run_pipeline.sh` 为 Phase I 补默认 train/val/test 文件参数。 |
| 【✅】 | P1-019 | 已修复并验证。final report 读取 best epoch 对应 metrics。 |

### 暂不纳入条目的执行状态

| 标记 | 条目 | 执行结果 |
| --- | --- | --- |
| 【】 | P1-012 | 部分采纳，未标记为完成。核心 evaluator/report 指标已接入：boundary、confusion/per-class、epoch stability、per-code switch、DP teacher distribution；latent snapshot/failure case 主流程触发仍未完整接线。 |
| 【✅】 | P1-018 | 已作为 P1-005 的结构性修复采纳。 |
| 【✅】 | P1-020 | 已采纳。`_export_horizon_labels` 的 `num_switches` 统计排除末步复制区间。 |
| 【】 | NoTradeControlConfig 未使用 | 未采纳，仍作为功能闭环待确认项。 |
| 【✅】 | `Phase1DemoStore.save_demos()` 不保存 `execution_books` | 已采纳。demo cache 当前序列化并恢复 `execution_books`。 |
| 【】 | `device` / `mixed_precision` / `early_stopping_patience` 未使用 | 未采纳，仍作为配置接线/后续增强项。 |
| 【✅】 | `composite_score_sensitivity()` 不重新选择 best epoch | 已采纳。当前使用 `composite_score_sensitivity_across_epochs()` 对所有 epoch 重新选择 best。 |

### 验证结果

| 标记 | 命令 | 结果 |
| --- | --- | --- |
| 【✅】 | `source /home/lanceliang/miniconda3/etc/profile.d/conda.sh && conda activate ArchetypeTrade && pytest -q` | `375 passed, 17 warnings`。warnings 为未注册 `pytest.mark.integration`。 |
| 【✅】 | `source /home/lanceliang/miniconda3/etc/profile.d/conda.sh && conda activate ArchetypeTrade && bash -n run_pipeline.sh` | 语法检查通过。 |
