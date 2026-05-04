# Phase I 完整时间分布与 No-trade 学习执行计划

**日期**: 2026-05-04
**来源设计**: `docs/changes/20260504_phase1_full_time_distribution_no_trade_design.md`
**影响阶段**: Phase I Archetype Discovery, Phase II Archetype Selection
**计划目标**: 将 Phase I 从“机会窗口采样训练”扩展为“train 覆盖完整时间分布 + val/test 全量 eligible label”的可执行代码批次，修复 Phase II 连续时间训练中缺少 no-trade / low-opportunity 行为的问题。

---

## 1. 执行原则

本计划遵循以下原则:

1. **先修评估 label 契约**: val/test 必须对所有 boundary eligible windows 生成 label，不再受 `num_demos` 和 `_num_samples_for_split()` 的 64 条限制。
2. **train 才做采样与扩充**: `num_demos` 只控制 train 样本预算；full-time pool、opportunity pool、no-trade 补样都只作用于 train。
3. **val/test 不做增强**: val/test 不做 stratified sampling、不做 temporal contrastive、不做 synthetic horizon，不人为改变评估分布。
4. **存储与计算可控**: val/test 全量 DP/label 可能很大，允许先用 chunk/shard 流式生成，避免一次性把全量 horizon tensor 放进内存。
5. **保持旧实验可读**: 新字段和新文件默认向后兼容；旧 Phase I 产物不能作为修复后 Phase II 连续时间实验的合格上游。
6. **测试先覆盖契约**: 每个批次必须至少有契约级单测；长训练实验不作为判断代码正确性的唯一依据。

---

## 2. 执行状态表

状态含义:

- `DONE`: 文档或代码变更已完成。
- `TODO`: 纳入本轮计划，尚未实现。
- `IN_PROGRESS`: 正在实现。
- `BLOCKED`: 存在外部阻塞。
- `DEFERRED`: 暂缓，后续单独批次。
- `DROP`: 明确不做。

| ID | 事项 | 优先级 | 执行状态 | 测试状态 | 主要文件 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| D0 | 生成执行计划文档 | P0 | DONE | 不适用 | `docs/plan/20260504_phase1_full_time_distribution_no_trade_execution_plan.md` | 本文件 |
| D1 | 在设计文档中回链执行计划 | P0 | DONE | 不适用 | `docs/changes/20260504_phase1_full_time_distribution_no_trade_design.md` | 文档索引 |
| A1 | 修正 val/test 全量 eligible label 生成 | P0 | DONE | DONE | `scripts/process_phase1_data.py`, `src/data/phase1_processed_store.py` | val/test 不再限制为 64 条；manifest 记录 eligible/labeled 数 |
| A2 | 新增 eval labeling 配置与 manifest 审计字段 | P0 | DONE | DONE | `src/config/phase1_config.py`, `scripts/process_phase1_data.py` | 已新增 `EvalLabelingConfig` 与 split 审计字段 |
| A3 | 实现 train full-time coverage pool | P0 | DONE | DONE | `scripts/process_phase1_data.py`, `src/data/stratified_sampler.py` | train 默认 full-time + opportunity 混合 |
| A4 | 合并 full-time pool 与 opportunity pool，写入 `sample_source` | P0 | DONE | DONE | `src/data/phase1_processed_store.py`, `src/data/demo_store.py` | `sample_source` 已写入 sampled/demos/labels |
| A5 | 实现 train no-trade / low-opportunity 最小覆盖补样 | P0 | DONE | DONE | `scripts/process_phase1_data.py`, `src/data/sampling_health.py` | DP 后从 unused full-time candidates 补样并替换 opportunity 样本 |
| A6 | 导出 `horizon_labels_full_time_train.feather` | P1 | DONE | DONE | `src/trainers/phase1_trainer.py`, `src/data/demo_store.py` | Phase I 训练后导出 full-time train subset labels |
| A7 | Phase I report 增加 full-time/no-trade 指标 | P1 | IN_PROGRESS | PARTIAL | `src/evaluation/phase1_report.py`, `src/evaluation/phase1_evaluator.py` | 已写基础覆盖/比例字段；full-time replay、cost leakage 与 failure cases 待补 |
| A8 | Phase II 支持选择 full-time train labels | P1 | DONE | DONE | `src/config/phase2_config.py`, `src/data/phase2_label_loader.py`, `src/trainers/phase2_trainer.py` | `phase1_label_source=full_time` 已接入并缺文件 fail-fast |
| A9 | 单元测试与集成 smoke | P0 | DONE | DONE | `tests/unit/**`, `tests/integration/**` | focused tests 与 Phase I smoke 已通过 |
| B1 | 立即要求 train 也对所有 stride=1 windows 生成 labels | P2 | DEFERRED | 不适用 | - | 计算/存储成本过高；先做 full-time train pool |
| B2 | 强制所有旧 Phase I 批次失效 | P2 | DEFERRED | 不适用 | - | 先通过 report guardrail 标记不合格 |
| B3 | 重写 DP planner 算法 | P2 | DROP | 不适用 | - | 本问题来自采样/label 分布，不改 single-trade DP |

### 2.1 本轮执行记录

**执行日期**: 2026-05-04

已落地:

1. `Phase1DataProcessConfig` 新增 full-time sampling、eval labeling 与 no-trade 最小覆盖配置。
2. `process_phase1_data.py` 将 train sampling 与 eval labeling 拆开；val/test 使用所有 boundary eligible windows。
3. train 默认生成 full-time coverage pool，并与 opportunity pool 混合；产物写入 `sample_source`。
4. train DP 后检查 no-trade / low-opportunity 覆盖，不足时从 unused full-time candidates 补样并替换 opportunity 样本。
5. manifest split 审计字段记录 eligible/labeled 数、labeling mode、sampling 状态、sample source 分布和 DP 后覆盖结果。
6. Phase I 训练导出 `horizon_labels_full_time_train.feather`。
7. Phase II 新增 `phase1_label_source: default|full_time`，选择 full-time 时缺文件 fail-fast。

剩余:

1. A7 深诊断仍需补充 full-time replay、`false_trade_on_no_trade_rate`、`low_opportunity_cost_leakage` 与 no-trade failure cases。

---

## 3. 批次 A1: 修正 Val/Test 全量 Label 生成

**目标**: val/test 的 DP teacher 和 `horizon_labels_{split}.feather` 覆盖所有 boundary eligible windows，不再被 `_num_samples_for_split()` 截断到 64。

**涉及文件**:

- `scripts/process_phase1_data.py`
- `src/data/phase1_processed_store.py`
- `src/data/demo_store.py`
- `src/trainers/phase1_trainer.py`
- `tests/unit/scripts/test_phase1_data_processor.py`
- `tests/integration/test_phase1_data_process_then_train.py`

**实现动作**:

1. 将 train 和 eval split 的目标集合生成逻辑拆开:
   - train: 继续使用 `num_demos` + sampler。
   - val/test: 使用所有 boundary eligible entries。
2. val/test 跳过 `StratifiedWindowSampler.sample()`。
3. val/test 跳过 data augmentation。
4. val/test 仍运行 DP teacher，输出 `dp_teacher_val.feather` / `dp_teacher_test.feather`。
5. 训练后导出的 `horizon_labels_val.feather` / `horizon_labels_test.feather` 必须覆盖所有 eligible val/test windows。
6. manifest 写入:
   - `num_eligible_windows`
   - `num_labeled_windows`
   - `labeling_mode=all_eligible`
   - `sampling_applied=false`
7. 对大 split 支持 chunked processing 或至少设计可插入 chunk 机制，避免一次性内存爆炸。

**验收**:

- 对 fixture 数据，val/test label 数量等于 boundary eligible window 数量。
- val/test `sampled_horizons_*` 或等价 eval horizon 文件不受 `num_demos` 改变影响。
- 现有 FU 类命令中，`--num-demos 18000` 只改变 train 样本数量。
- Phase II 读取 val labels 时不再出现只有 64 条 label 的覆盖限制。

---

## 4. 批次 A2: 配置与 Manifest 契约

**目标**: 用显式配置表达 eval split 全量 labeling 和 train full-time sampling，避免隐藏在脚本分支里。

**涉及文件**:

- `src/config/phase1_config.py`
- `scripts/process_phase1_data.py`
- `tests/unit/config/test_phase1_config_docs.py`
- `tests/unit/scripts/test_phase1_data_processor.py`

**实现动作**:

1. 新增 `EvalLabelingConfig`:

```python
@dataclass(frozen=True)
class EvalLabelingConfig:
    val_mode: Literal["all_eligible"] = "all_eligible"
    test_mode: Literal["all_eligible"] = "all_eligible"
    apply_sampling: bool = False
    apply_augmentation: bool = False
```

2. 新增 `TimeDistributionSamplingConfig`:

```python
@dataclass(frozen=True)
class TimeDistributionSamplingConfig:
    enabled: bool = True
    full_time_mode: Literal["non_overlap", "stride"] = "stride"
    full_time_stride: int = 36
    min_train_ratio: float = 0.40
    label_export_enabled: bool = True
```

3. 将两个配置挂到 `Phase1DataProcessConfig`。
4. 更新 config docs 和 `to_dict()` hash 影响范围。
5. manifest 中记录最终 resolved mode。

**验收**:

- 配置可序列化、hash 稳定。
- `paper_strict_reproduction` 或兼容模式下可显式关闭 full-time train sampling，但 val/test all-eligible labeling 仍默认开启。

---

## 5. 批次 A3/A4: Train Full-time Pool 与样本来源

**目标**: train 集合从单一 opportunity sampling 改为 full-time coverage pool + opportunity pool 混合。

**涉及文件**:

- `scripts/process_phase1_data.py`
- `src/data/stratified_sampler.py`
- `src/data/phase1_processed_store.py`
- `src/data/demo_store.py`
- `tests/unit/data/test_phase1_processed_store.py`
- `tests/unit/scripts/test_phase1_data_processor.py`

**实现动作**:

1. 从 train eligible windows 生成 full-time pool:
   - `non_overlap`: `start % horizon == 0`。
   - `stride`: `start % full_time_stride == 0`。
2. 继续用当前 prospective/stratified sampler 生成 opportunity pool。
3. 按 `window_start` 合并去重。
4. 每条 horizon 写入 `sample_source`:
   - `full_time`
   - `opportunity`
   - `both`
5. 保证最终 train 样本数不超过 `num_demos`。
6. 当 full-time 可用样本不足时:
   - 优先使用 stride 模式满足 `min_train_ratio`。
   - 若仍不足，report 写 warning，并记录 resolved ratio。

**验收**:

- `sample_source` 可读回。
- train `full_time_sample_ratio >= min_train_ratio`，或 report 明确记录不足原因。
- 相同 seed + 相同输入得到稳定 sample_id 和 window_start 集合。

---

## 6. 批次 A5: No-trade / Low-opportunity 最小覆盖补样

**目标**: 采样后运行 DP，再根据 DP 结果保证 train 中 no-trade / low-opportunity 有最低覆盖。

**涉及文件**:

- `scripts/process_phase1_data.py`
- `src/config/phase1_config.py`
- `src/data/sampling_health.py`
- `src/planners/demo_generator.py`
- `tests/unit/scripts/test_phase1_data_processor.py`
- `tests/unit/data/test_sampling_health.py`

**实现动作**:

1. 扩展 `NoTradeControlConfig`:

```python
min_no_trade_ratio: float = 0.10
min_low_opportunity_ratio: float = 0.25
low_opportunity_return_quantile: float = 0.30
resample_when_below_min: bool = True
```

2. 初次 DP 后计算:
   - `no_trade_ratio`
   - `low_opportunity_ratio`
   - `sample_source` 分布
3. 若低于阈值，从未使用的 full-time candidates 中补样。
4. 对补样 horizon 运行 DP。
5. 重新合并并裁剪 opportunity 样本，保持 `num_demos` 总预算。
6. report 写入:
   - 初始比例
   - 补样数量
   - 最终比例
   - 被替换的 opportunity 数量

**验收**:

- 人造 fixture 中 no-trade 不足时触发补样。
- low-opportunity 不足时触发补样。
- 补样只发生在 train。
- val/test 不受补样逻辑影响。

---

## 7. 批次 A6/A7: Labels、Report 与指标

**目标**: 产物和报告能明确说明该 Phase I 批次是否合格支撑 Phase II 连续时间训练。

**涉及文件**:

- `src/trainers/phase1_trainer.py`
- `src/evaluation/phase1_evaluator.py`
- `src/evaluation/phase1_report.py`
- `src/evaluation/diagnostics/failure_case_report.py`
- `src/data/demo_store.py`
- `tests/unit/evaluation/test_phase1_report.py`
- `tests/unit/evaluation/test_phase1_evaluator.py`

**实现动作**:

1. 导出 `horizon_labels_full_time_train.feather`。
2. 修正 `horizon_labels_val.feather` / `horizon_labels_test.feather` 为全量 eligible labels。
3. report 增加:
   - `full_time_training_enabled`
   - `full_time_sample_ratio`
   - `opportunity_sample_ratio`
   - `label_coverage_val`
   - `label_coverage_test`
   - `no_trade_ratio`
   - `low_opportunity_ratio`
   - `false_trade_on_no_trade_rate`
   - `low_opportunity_cost_leakage`
4. failure case report 增加:
   - teacher no-trade 但 student 交易。
   - low-opportunity 高成本泄漏。
   - 首步高 boundary cost。

**验收**:

- report 中有 full-time/no-trade 字段。
- 缺少 full-time labels 或 val/test label 覆盖不足时，报告标记不合格。
- failure case 可在 fixture 中生成对应案例。

---

## 8. 批次 A8: Phase II Label Source 支持

**目标**: Phase II 可以显式选择 full-time train labels，并直接使用 val/test 全量 labels。

**涉及文件**:

- `src/config/phase2_config.py`
- `src/data/phase2_label_loader.py`
- `src/trainers/phase2_trainer.py`
- `tests/unit/data/test_phase2_label_loader.py`
- `tests/unit/trainers/test_phase2_trainer.py`

**实现动作**:

1. `Phase2Config` 新增:

```python
phase1_label_source: Literal["default", "full_time"] = "default"
```

2. 当 `phase1_label_source="full_time"`:
   - train 优先读取 `horizon_labels_full_time_train.feather`。
   - val 读取 `horizon_labels_val.feather`。
   - test 仍不得进入训练决策路径，只能 posthoc。
3. log 输出 label coverage。
4. 若用户请求 full_time 但文件不存在，fail-fast。

**验收**:

- Phase II train label coverage 不再受 Phase I sampled train 起点限制。
- val label join 覆盖率显著高于旧 2/1173。
- test label 泄漏防护仍通过。

---

## 9. 测试与验证命令

### 单元测试

```bash
pytest tests/unit/config/test_phase1_config_docs.py \
       tests/unit/scripts/test_phase1_data_processor.py \
       tests/unit/data/test_phase1_processed_store.py \
       tests/unit/data/test_sampling_health.py \
       tests/unit/evaluation/test_phase1_report.py \
       -q
```

### Phase II 相关单测

```bash
pytest tests/unit/data/test_phase2_label_loader.py \
       tests/unit/trainers/test_phase2_trainer.py \
       tests/integration/test_phase2_no_test_label_leakage.py \
       -q
```

### 集成 smoke

```bash
pytest tests/integration/test_phase1_data_process_then_train.py \
       tests/integration/test_phase1_pipeline_smoke.py \
       -q
```

### FU 真实批次验收建议

```bash
python scripts/process_phase1_data.py \
  --pair FU \
  --data-batch-id batch_010_full_time_no_trade \
  --train-file data/FU/train.feather \
  --val-file data/FU/val.feather \
  --test-file data/FU/test.feather \
  --horizon 72 \
  --num-demos 18000 \
  --sampling-strategy stratified_uniform \
  --stratification-mode prospective_past \
  --prospective-lookback-minutes 720 \
  --max-position 10 \
  --seed 42 \
  --sampling-max-overlap-ratio 0.7
```

预期检查:

- train sampled horizons 约 18000。
- val labels 约等于 val boundary eligible windows，不再是 64。
- test labels 约等于 test boundary eligible windows，不再是 64。
- report 中 `no_trade_ratio` 和 `low_opportunity_ratio` 达到配置阈值或明确 warning。

---

## 10. 风险与缓解

| 风险 | 影响 | 缓解 |
| --- | --- | --- |
| val/test 全量 DP 运行时间显著增加 | 数据预处理耗时上升 | chunk/shard 生成，支持断点或至少明确进度日志 |
| val/test 全量 sampled horizons 文件过大 | 存储和加载压力 | 优先保存 compact window index + teacher/label；必要时分 shard |
| train full-time 样本稀释交易机会 | opportunity archetype 质量下降 | 默认 40/60 起步，report 分开记录 opportunity/full-time 指标 |
| no-trade 过多导致 code 容量被 flat 占据 | trading archetype 数量不足 | 保留 max_no_trade_ratio 和 no-trade code health guardrail |
| Phase II 误读 test labels | 泄漏风险 | 保持 `Phase2TestLabelRequestError` 与 no-test-label-leakage 测试 |

---

## 11. 完成定义

本计划视为完成需要满足:

1. 所有 P0 批次 A1-A5 完成。
2. 单元测试和集成 smoke 通过。
3. 新 Phase I 产物 report 明确记录 val/test 全量 label 覆盖。
4. Phase II 使用新产物时，val label join 不再被 64 条 label 限制。
5. FU 新批次至少完成一次数据预处理 dry-run 或真实 run，并记录关键覆盖率。
