# Phase I 完整时间分布与 No-trade 学习设计变更

**日期**: 2026-05-04
**影响阶段**: Phase I Archetype Discovery, Phase II Archetype Selection
**目标**: 让 Phase I 的训练与 label 产物覆盖完整时间分布中的 no-trade / low-opportunity 窗口，而不是几乎只覆盖 DP teacher 可盈利的机会窗口。
**对应执行计划**: `docs/plan/20260504_phase1_full_time_distribution_no_trade_execution_plan.md`

---

## 1. 背景

当前 FU `batch_009_prospective` 暴露出一个关键问题:

- Phase I train labels 共 18000 条，但 `is_no_trade` 只有约 `0.54%`。
- Phase I val teacher profitable ratio 约 `98.4%`，说明验证窗口几乎都是事后可交易机会。
- Phase II 使用连续 `non_overlap` 时间窗训练时，大量真实时间窗口没有有效交易机会，但 Phase I 冻结 decoder/codebook 并没有学到足够稳定的 no-trade 行为。
- 在 Phase II replay 中，很多 single archetype 即使不切换 code，内部 decoder 仍频繁产生底层交易动作，亏损几乎全部来自成本。

这说明 Phase I 当前更像是在学习“机会窗口里的交易形态压缩”，而不是学习完整市场时间分布下的“交易 / 不交易”行为谱系。Phase II selector 在连续时间上运行时，需要能选择 no-trade 或低活跃 archetype；如果 Phase I 没有提供这种行为，Phase II 只能在一组高成本交易 archetype 之间选择。

---

## 2. 设计目标

1. Phase I 数据预处理必须纳入完整时间分布中的 no-trade / low-opportunity 窗口。
2. No-trade 不再只有“最高占比上限”，还必须有“最低覆盖率”与时间覆盖要求。
3. Phase I 仍保留机会窗口采样能力，但训练集合必须由“时间覆盖样本池 + 机会增强样本池”混合构成。
4. train/val/test 的 sampled horizons 与 `horizon_labels_{split}.feather` 必须能服务 Phase II 的连续时间窗口训练。
5. 报告必须明确区分:
   - opportunity-only 指标，衡量交易 archetype 质量。
   - full-time 指标，衡量完整时间分布下的收益、成本和不交易能力。

---

## 3. 非目标

- 不修改 `SingleTradeDPPlanner` 的 single-trade 约束。
- 不把某个 code 手工指定为 flat code；code 语义仍由 VQ 学习得到。
- 不让 Phase II 在线调用 DP。
- 不使用 test labels 进入 Phase II 决策路径。
- 不用简单过滤负收益样本来提高 Phase I 指标。

---

## 4. 核心设计

### 4.0 现有采样机制确认

当前 FU `batch_009_prospective` 使用的 Phase I 数据预处理命令是:

```bash
python scripts/process_phase1_data.py \
  --pair FU \
  --data-batch-id batch_009_prospective \
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

这条命令对应的当前机制:

1. `SlidingWindowIndexer` 先按 stride=1 枚举所有候选窗口。
2. `prospective_past` 分层只使用 `start_index` 之前 `720` 分钟的 past return / past volatility / past draw pattern。
3. `StratifiedWindowSampler` 使用 `stratified_uniform` 对 strata 做均匀配额。
4. 采样器仍会限制 `flat|low|*` strata 的最高占比，默认 `flat_low_vol_max_ratio=0.15`。
5. `--num-demos 18000` 只直接决定 train 目标样本数。
6. val/test 当前由 `_num_samples_for_split()` 固定为 `min(64, num_demos // 16)`，因此该批次 val/test labels 各只有 64 条。这是当前实现行为，但不是正确的目标行为。
7. DP teacher 在采样完成后才运行；因此当前机制不会基于 DP 后的 `is_no_trade` 或 `low_opportunity` 最低覆盖率做补样。

该机制本身是 prospective 的，不是后视分层；本次变更不是否定这套机制，而是在 train 侧增加完整时间覆盖池和 DP 后覆盖率约束，并修正 val/test label 生成契约。

本次变更必须修正一个现有设计错误: **val/test 不应按 `num_demos` 抽样到 64 条 label，而应对对应 split 的所有 eligible windows 生成 label**。val/test 是评估和后续 Phase II label 对齐来源，不能因为 Phase I 训练样本预算被压缩。

### 4.1 两类样本池

Phase I 数据预处理新增两个逻辑样本池:

| 样本池 | 生成方式 | 作用 |
| --- | --- | --- |
| full-time coverage pool | 按时间顺序生成固定 stride 或 non-overlap horizon，不按未来收益筛选 | 覆盖真实连续时间分布，让 no-trade / low-opportunity 自然出现 |
| opportunity pool | 沿用现有 prospective / stratified 采样 | 增强可交易机会覆盖，保证交易 archetype 不被稀释 |

最终 train sampled horizons 来自二者混合。val/test 不参与这种混合采样，必须直接使用全量 eligible windows。推荐默认:

```yaml
time_distribution_sampling:
  enabled: true
  full_time_pool:
    mode: non_overlap
    stride: 72
    min_train_ratio: 0.40
  opportunity_pool:
    enabled: true
    max_train_ratio: 0.60
  eval_labeling:
    val_mode: all_eligible
    test_mode: all_eligible
    apply_sampling: false
    apply_augmentation: false
```

解释:

- train 至少 40% 来自完整时间覆盖池，避免训练集只剩机会窗口。
- val/test 不做采样扩充，不受 `num_demos` 限制，所有 eligible windows 都要运行 DP 并导出 label。
- opportunity pool 仍用于学习丰富交易形态，但不能挤掉完整时间分布样本。

### 4.1.1 `num-demos` 大小与配额

`--num-demos` 不应简单理解为“机会样本数量”。本变更后它只表示 Phase I train 的总样本预算:

```text
train_num_demos = full_time_train_quota + opportunity_train_quota - duplicated_starts
val_label_count = num_eligible_val_windows
test_label_count = num_eligible_test_windows
```

推荐先保持 `--num-demos 18000` 作为 FU 的对照预算，不立即增大。需要调整的是样本构成，而不是盲目增加总量。

以当前 FU `batch_009_prospective` 产物为例:

| split | boundary eligible windows | current sampled | eligible non-overlap windows (`start % 72 == 0`) |
| --- | ---: | ---: | ---: |
| train | 493087 | 18000 | 6849 |
| val | 116430 | 64 | 1618 |
| test | 160862 | 64 | 2235 |

上表中的 val/test `current sampled=64` 是需要修正的旧行为。新契约下:

- `sampled_horizons_train.feather` 仍由 train sampling 生成，数量由 `num_demos` 控制。
- `sampled_horizons_val.feather` / `sampled_horizons_test.feather` 应改为评估窗口集合，默认包含所有 boundary eligible windows。
- `dp_teacher_val.feather` / `dp_teacher_test.feather` 应为所有 eligible val/test windows 生成 teacher actions/rewards。
- `horizon_labels_val.feather` / `horizon_labels_test.feather` 应覆盖所有 eligible val/test windows，供 Phase II / posthoc 评估按需要 join。

如果 `num_demos=18000` 且 `min_train_ratio=0.40`，full-time train quota 需要约 `7200` 条。若 full-time pool 严格使用 non-overlap，当前 FU train 只有约 `6849` 条可用，略低于 40% 目标。因此实现时必须支持以下策略之一:

1. **推荐**: 保持 `num_demos=18000`，full-time train pool 使用 `stride`，例如 `full_time_stride=36` 或 `24`；同时额外导出 non-overlap full-time labels 供 Phase II 对齐。
2. 保持 full-time train pool 为 `non_overlap`，将 `min_train_ratio` 自适应降到可用上限，例如 `6849 / 18000 ~= 0.38`。
3. 保持 full-time train pool 为 `non_overlap` 且坚持 `min_train_ratio=0.40`，则把 `num_demos` 降到不超过 `floor(6849 / 0.40) = 17122`。

不建议仅为了 no-trade 覆盖而把 `num_demos` 从 18000 继续增大；如果 full-time pool 是 non-overlap，增大 `num_demos` 反而会降低 full-time quota 的可满足性。只有在 full-time pool 改为 stride、训练时间和显存预算允许，并且 codebook health 没有退化时，才考虑增大 `num_demos`。

### 4.2 No-trade / low-opportunity 最小覆盖

现有 `NoTradeControlConfig.max_no_trade_ratio` 只能防止 no-trade 过多，不能解决 no-trade 太少的问题。新增最低覆盖配置:

```yaml
no_trade_control:
  keep_no_trade: true
  max_no_trade_ratio: 0.35
  min_no_trade_ratio: 0.10
  min_low_opportunity_ratio: 0.25
  low_opportunity_return_quantile: 0.30
  resample_when_below_min: true
```

定义:

- `is_no_trade`: DP teacher actions 全程 flat。
- `low_opportunity`: `demo_return` 位于训练集 DP return 的低分位区间，或 `demo_return <= min_profit_gate`。
- `min_no_trade_ratio`: 训练集中全 flat teacher 的最低比例。
- `min_low_opportunity_ratio`: 训练集中低机会窗口的最低比例，包含但不限于 no-trade。

当 DP 运行后发现比例不足时，不允许只接受当前样本集。数据预处理必须从 full-time coverage pool 补入低机会窗口，并按需要减少 opportunity pool 样本。

### 4.3 Full-time labels 供 Phase II 对齐

Phase II 的 `non_overlap` 或 `stride` horizon index 需要高覆盖标签。Phase I 数据预处理应导出高覆盖 label。

对于 val/test，现有 label 文件本身就应该是全量 eligible windows:

```text
horizon_labels_val.feather
horizon_labels_test.feather
```

对于 train，为避免训练样本预算和全量 label 需求混在一起，建议额外导出完整时间覆盖 label:

```text
horizon_labels_full_time_train.feather
```

这些文件与现有 `horizon_labels_{split}.feather` 的字段保持兼容，但起点由 full-time coverage pool 或所有 eligible windows 决定。Phase II 后续可以选择:

- 继续读取 `horizon_labels_{split}.feather`，保持旧实验兼容。
- 显式读取 `horizon_labels_full_time_train.feather`，用于 train 连续时间训练。
- 对 val/test 直接使用全量 `horizon_labels_val.feather` / `horizon_labels_test.feather`。

推荐后续 Phase II 配置新增:

```yaml
phase1_label_source: full_time
```

### 4.4 指标分层

Phase I report 新增或强化以下指标:

| 指标 | 说明 |
| --- | --- |
| `full_time_sample_ratio` | sampled horizons 中来自完整时间覆盖池的比例 |
| `opportunity_sample_ratio` | sampled horizons 中来自机会增强池的比例 |
| `no_trade_ratio` | DP 全 flat 样本比例 |
| `low_opportunity_ratio` | 低机会样本比例 |
| `full_time_label_coverage_train/val/test` | full-time labels 对对应 split 连续窗口的覆盖率 |
| `full_time_student_net_return` | student decoder 在完整时间分布 replay 的净收益 |
| `full_time_total_cost_paid` | 完整时间分布 replay 的总成本 |
| `false_trade_on_no_trade_rate` | teacher no-trade 样本中 student 产生非 flat 动作的比例 |
| `low_opportunity_cost_leakage` | low-opportunity 样本中 student 成本 / abs gross pnl 的比例 |

Checkpoint selection 不应只看 opportunity validation。建议拆成两组:

- `opportunity_val_*`: 交易机会窗口上的重构和收益捕获。
- `full_time_val_*`: 完整时间分布上的成本控制、不交易能力和风险。

推荐 `phase1_composite_score` 加入:

```yaml
selection_policy:
  metric_weights:
    val_weighted_reconstruction_accuracy: 0.15
    switch_point_recall: 0.20
    switch_direction_accuracy: 0.15
    val_return_capture_ratio: 0.15
    val_sharpe_ratio: 0.10
    full_time_cost_control_score: 0.15
    no_trade_reconstruction_score: 0.10
```

### 4.5 No-trade 行为诊断

Phase I failure case report 必须能单独抽样 no-trade / low-opportunity 失败案例:

- teacher 全 flat，但 student 出现 long/short。
- teacher 低收益或低机会，但 student 成本过高。
- student 首步从 flat 切到高仓位，产生高 boundary cost。
- 某些 code 在 no-trade 样本上行为不稳定。

这类案例应写入 `phase1_failure_cases.html/json`，并在 `phase1_report.json` 中给出汇总计数。

---

## 5. 配置变更建议

新增配置:

```python
@dataclass(frozen=True)
class TimeDistributionSamplingConfig:
    enabled: bool = True
    full_time_mode: Literal["non_overlap", "stride"] = "non_overlap"
    full_time_stride: int = 72
    min_train_ratio: float = 0.40
    label_export_enabled: bool = True


@dataclass(frozen=True)
class EvalLabelingConfig:
    val_mode: Literal["all_eligible"] = "all_eligible"
    test_mode: Literal["all_eligible"] = "all_eligible"
    apply_sampling: bool = False
    apply_augmentation: bool = False
```

扩展 `NoTradeControlConfig`:

```python
min_no_trade_ratio: float = 0.10
min_low_opportunity_ratio: float = 0.25
low_opportunity_return_quantile: float = 0.30
resample_when_below_min: bool = True
```

`paper_strict_reproduction=True` 时:

- 可关闭 `time_distribution_sampling.enabled`，复现论文 opportunity-style 采样。
- 必须在 report 中标记 `full_time_training_enabled=false`。
- Phase II 正式连续时间实验不得使用该批次作为默认上游。

---

## 6. 数据流程变更

Phase I data processor 推荐流程:

1. 枚举完整 train/val/test window index。
2. 从完整 window index 生成 full-time coverage pool。
3. 按现有 prospective/stratified 逻辑生成 opportunity pool。
4. 合并两个 pool，按 `sample_id` 去重，记录 `sample_source`:
   - `full_time`
   - `opportunity`
   - `both`
5. 对合并后的 sampled horizons 运行 DP teacher。
6. 根据 DP 结果计算 no-trade / low-opportunity 覆盖率。
7. 如果覆盖率低于最低阈值，从 full-time pool 补样并重新运行 DP。
8. 对 val/test 的所有 eligible windows 运行 DP teacher，不做 sampling、不做 augmentation。
9. 写入 sampled horizons、DP teacher、full-time labels 和 report diagnostics。

重要约束:

- 补样不得跨 split。
- 补样只允许用于 train。
- val/test 不允许按 strata 配额抽样，也不允许合成扩充。
- 补样不得破坏 `min_gap_between_samples`，除非显式允许 overlap relaxation。
- `sample_id` 必须稳定，建议包含 split、start_index 和 sample_source。
- DP teacher hash 必须包含 full-time sampling 配置和 no-trade coverage 配置。

---

## 7. 影响文件

预计后续实现会涉及:

| 文件 | 变更 |
| --- | --- |
| `src/config/phase1_config.py` | 新增 `TimeDistributionSamplingConfig`，扩展 no-trade 配置 |
| `src/data/stratified_sampler.py` | 支持 full-time pool 与 opportunity pool 合并 |
| `src/data/sampling_health.py` | 新增 full-time/no-trade/low-opportunity 覆盖检查 |
| `scripts/process_phase1_data.py` | 编排完整时间覆盖样本生成、补样和 label export；修正 val/test 不再由 `_num_samples_for_split()` 限制到 64 |
| `src/trainers/phase1_trainer.py` | 读取新增字段，报告 full-time metrics |
| `src/evaluation/phase1_evaluator.py` | 新增 full-time replay 与 no-trade 行为指标 |
| `src/evaluation/diagnostics/failure_case_report.py` | 新增 no-trade / low-opportunity failure cases |
| `src/data/demo_store.py` / `src/data/phase1_processed_store.py` | 保存 `sample_source` 与 full-time labels |

---

## 8. 测试计划

必须新增或扩展测试:

1. `TimeDistributionSamplingConfig` 可序列化、hash 稳定、文档字段完整。
2. full-time pool 使用 non-overlap/stride 生成稳定窗口。
3. sampled horizons 中 `sample_source` 正确写入并可读回。
4. no-trade 比例低于 `min_no_trade_ratio` 时触发补样。
5. low-opportunity 比例低于 `min_low_opportunity_ratio` 时触发补样。
6. val/test 在 full-time 模式下不使用 opportunity-only 采样。
7. val/test label 数量等于各自 boundary eligible window 数量。
8. `horizon_labels_full_time_train.feather` 字段与现有 labels 契约兼容。
9. Phase I report 输出 full-time/no-trade/low-opportunity 指标。
10. `paper_strict_reproduction=True` 时关闭 full-time training，并在 report 标明。

---

## 9. 验收标准

一个 Phase I 批次只有满足以下条件，才可作为 Phase II 连续时间训练的默认上游:

1. `full_time_training_enabled=true`。
2. `full_time_label_coverage_train >= 0.95`。
3. `label_coverage_val >= 0.99`，且覆盖对象是所有 eligible val windows。
4. `label_coverage_test >= 0.99`，且覆盖对象是所有 eligible test windows。
5. `no_trade_ratio >= min_no_trade_ratio`，或 report 明确说明数据中真实 no-trade 极少且人工签收。
6. `low_opportunity_ratio >= min_low_opportunity_ratio`。
7. `false_trade_on_no_trade_rate` 低于配置阈值。
8. `full_time_total_cost_paid` 没有相对 always-flat baseline 出现异常成本泄漏。
9. Phase II 使用该批次时，val/test label join 覆盖率不应再被 64 条 label 限制。

---

## 10. 风险与取舍

- 增加 no-trade / low-opportunity 样本会拉低 opportunity-only validation 指标，这是预期现象，不应视为退化。
- 如果 no-trade 样本过多，codebook 可能把容量集中到 flat 行为，需要配合 no-trade code health 监控。
- 如果 full-time pool 占比太低，Phase II 仍学不到不交易；如果占比太高，交易 archetype 可能不足。默认 40/60 是起点，需要按品种调参。
- 该变更会改变 DP teacher hash 和 Phase I label 分布，旧 Phase I 产物不能与新 Phase II 实验混用。

---

## 11. 推荐落地顺序

1. 先修正 val/test label 生成: 对所有 boundary eligible windows 运行 DP 并导出 `horizon_labels_val.feather` / `horizon_labels_test.feather`，不做 sampling、不做 augmentation。
2. 实现 train full-time pool 生成与 `sample_source` 字段。
3. 实现 train no-trade / low-opportunity 最小覆盖补样。
4. 导出 `horizon_labels_full_time_train.feather`，供 Phase II train 连续时间训练对齐。
5. 最后把 full-time 指标纳入 Phase I report 和 checkpoint selection。

本变更完成后，Phase I 的职责从“压缩可盈利机会窗口”升级为“在完整时间分布上发现交易与不交易 archetype”。这是一条 Phase II 连续时间策略能否学会少交易、少付成本的前置条件。
