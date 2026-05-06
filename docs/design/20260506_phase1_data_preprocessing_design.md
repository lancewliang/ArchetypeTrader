# Phase I 前置数据处理技术设计

## 1. 目标

本文档定义 Phase I 训练之前的离线数据处理阶段。该阶段已经从 Phase I 训练中切分出来，目标是把论文第一阶段需要的 fixed-length chunks 和 DP demonstration trajectories 预先固化为可复用、可审计的 manifest 产物。

数据处理阶段负责：

1. 读取外部已经切分好的 train/val/test 行情与因子文件。
2. 校验输入 schema 和 factor list。
3. 枚举固定长度 horizon 窗口。
4. 对 train 做分层采样和采样健康检查。
5. 构建 horizon records：`states/prices/execution_books/meta`。
6. 运行 Single-trade DP teacher，生成 `actions/rewards`。
7. 生成 sampled、full-time train、non-overlap 三类训练/标注产物。
8. 写入 `data_process_manifest.json`，作为 Phase I 训练唯一数据入口。

Phase I 训练阶段只消费 manifest，不重新读取原始行情、不重新采样、不重新运行 DP。

## 2. 论文语义映射

论文第一阶段 Archetype Discovery 的核心输入是 demonstration trajectory：

```text
tau = (s_demo, a_demo, r_demo)
```

工程映射如下：

| 论文概念 | 当前数据处理产物 |
| --- | --- |
| 固定长度 chunk `h` | `window_index_{split}.feather` 中的 horizon window |
| market observations `s_demo` | `sampled_horizons_{split}.feather.states` |
| DP demonstration actions `a_demo` | `sampled_dp_teacher_{split}.feather.actions` |
| DP rewards `r_demo` | `sampled_dp_teacher_{split}.feather.rewards` |
| 训练 VQ 的 demonstration dataset | manifest join 后的 `HorizonRecord` |
| Phase II label 的基础 horizon | `non_overlap_horizons_{split}.feather` + `non_overlap_dp_teacher_{split}.feather` |

DP 只属于离线数据处理和标签生成。Phase I 模型训练、Phase II selector 训练、验证、测试和线上推理都不动态调用 DP。

## 3. 目录与代码边界

数据前置处理代码统一集中在一个目录：

```text
src/preprocess_data/
  __init__.py
  config.py
  processor.py
  market_reader.py
  feature_registry.py
  feature_provenance.py
  schema.py
  window_indexer.py
  stratified_sampler.py
  sampling_health.py
  horizon_builder.py
  data_augmentation.py
  processed_store.py
  demo_store.py
```

CLI 入口保留在：

```text
scripts/pre_process_data.py
```


职责边界：

- `scripts/pre_process_data.py` 只负责 CLI 参数接收、配置覆盖、日志初始化、随机种子初始化和调用 `Phase1DataProcessor.run()`。
- `src/preprocess_data/` 放所有前置数据处理代码，包括配置、编排、读文件、schema、窗口、采样、horizon 构建、processed store 和 manifest。
- `src/preprocess_data/` 可以调用共享的交易成本、reward 对齐和 DP planner，但不能把前置数据处理逻辑留在 `src/data/` 或 `scripts/`。
- `src/data/` 不再承载 Phase I 前置数据处理模块。若仍有通用 dataset 适配代码，应只保留训练/推理通用能力，不包含采样和 DP teacher 生成。
- `src/config/phase1_config.py` 已删除；`Phase1DataProcessConfig` 保留在 `src/preprocess_data/config.py`，Phase I 训练配置迁移到 `src/phase1/config.py`。

主要配置：

```text
src/preprocess_data/config.py::Phase1DataProcessConfig
```

主要编排类：

```text
src/preprocess_data/processor.py::Phase1DataProcessor
```

主要模块：

| 模块 | 职责 |
| --- | --- |
| `src/preprocess_data/config.py` | 前置数据处理配置 |
| `src/preprocess_data/processor.py` | 前置数据处理主流程 |
| `src/preprocess_data/market_reader.py` | 读取 train/val/test feather |
| `src/preprocess_data/feature_registry.py` | 加载 pair/profile 对应 factor list |
| `src/preprocess_data/feature_provenance.py` | 写入特征来源审计 |
| `src/preprocess_data/schema.py` | 校验价格列、特征列、盘口列和 schema |
| `src/preprocess_data/window_indexer.py` | 按 reward alignment 枚举 horizon window |
| `src/preprocess_data/stratified_sampler.py` | 生成 strata，并对 train 做分层采样 |
| `src/preprocess_data/sampling_health.py` | 检查 overlap、min gap、low-vol 占比等采样健康 |
| `src/preprocess_data/horizon_builder.py` | 从 window 切出 states、prices、execution books |
| `src/preprocess_data/data_augmentation.py` | train-only 数据增强 |
| `src/planners/single_trade_dp.py` | 单个 horizon 的 single-trade DP |
| `src/planners/demo_generator.py` | 批量生成 DP teacher actions/rewards |
| `src/preprocess_data/processed_store.py` | 保存 sampled horizons、DP teacher、manifest，并提供训练加载校验 |
| `src/preprocess_data/demo_store.py` | 保存兼容 demos 与 non-overlap demos |

从当前代码迁移时采用一对一搬迁，不改变算法语义：

| 当前文件 | 目标文件 |
| --- | --- |
| `scripts/pre_process_data.py::Phase1DataProcessor` | `src/preprocess_data/processor.py` |
| `src/config/phase1_config.py::Phase1DataProcessConfig` | `src/preprocess_data/config.py` |
| `src/config/phase1_config.py::Phase1Config` | `src/phase1/config.py` |
| `src/data/market_reader.py` | `src/preprocess_data/market_reader.py` |
| `src/data/feature_registry.py` | `src/preprocess_data/feature_registry.py` |
| `src/data/feature_provenance.py` | `src/preprocess_data/feature_provenance.py` |
| `src/data/schema.py` | `src/preprocess_data/schema.py` |
| `src/data/window_indexer.py` | `src/preprocess_data/window_indexer.py` |
| `src/data/stratified_sampler.py` | `src/preprocess_data/stratified_sampler.py` |
| `src/data/sampling_health.py` | `src/preprocess_data/sampling_health.py` |
| `src/data/horizon_builder.py` | `src/preprocess_data/horizon_builder.py` |
| `src/data/data_augmentation.py` | `src/preprocess_data/data_augmentation.py` |
| `src/data/phase1_processed_store.py` | `src/preprocess_data/processed_store.py` |
| `src/data/demo_store.py` | `src/preprocess_data/demo_store.py` |

迁移完成后，Phase I/Phase II 训练如果需要读取前置处理产物，必须从 `src/preprocess_data/processed_store.py` 导入 manifest/store，不再从 `src/data/phase1_processed_store.py` 导入。

Phase I 训练入口：

```text
scripts/train_phase1.py
```

训练入口必须接收：

```text
--data-process-manifest artifacts/{PAIR}/{DATA_BATCH_ID}/phase1/data_process_manifest.json
```

`scripts/train_phase1.py` 不负责任何数据处理行为。

## 4. 输入契约

数据处理阶段读取三份外部准备好的文件：

```text
--train-file data/{PAIR}/train.feather
--val-file data/{PAIR}/val.feather
--test-file data/{PAIR}/test.feather
```

输入文件要求：

- 必须可由 `MarketFileReader` 读取。
- 必须包含价格列，默认 `close`。
- 必须包含 factor list 要求的特征列。
- `close` 只作为价格列和 reward 结算列，不进入模型状态特征。
- train/val/test 必须通过同一个 `InputSchemaValidator` 校验。

factor list 选择：

```text
--pair FU
--factor-profile short
--factor-list-file src/factors/FU/short.txt
```

若存在固定 factor list，则 schema 使用固定特征清单；否则走 legacy numeric auto 模式。新实验应优先使用固定 factor list，避免不同文件列顺序或额外列导致训练输入漂移。

## 5. 配置

当前数据处理配置由 `src/preprocess_data/config.py::Phase1DataProcessConfig` 管理。会影响数据产物和 hash 的字段包括：

```yaml
pair: FU
data_batch_id: batch_001
train_file: data/FU/train.feather
val_file: data/FU/val.feather
test_file: data/FU/test.feather
artifact_root: artifacts
factor_profile: short
factor_list_file: null
horizon: 72
num_demos: 30000
sampling_strategy: stratified_uniform
seed: 42

stratification:
  mode: hindsight_horizon
  prospective_lookback_minutes: 1440
  require_prospective_diagnostic: true
  diagnostic_pair_batch_id: batch_001_prospective

sampling_health:
  max_no_trade_ratio: 0.25
  flat_low_vol_max_ratio: 0.15
  min_gap_between_samples: 12
  max_overlap_ratio: 0.5
  split_boundary_embargo: 73
  next_row_split_boundary_embargo: 74
  warn_only: false
  allow_overlap_relaxation: false

no_trade_control:
  keep_no_trade: true
  max_no_trade_ratio: 0.35
  min_no_trade_ratio: 0.10
  min_low_opportunity_ratio: 0.25

time_distribution_sampling:
  enabled: true
  full_time_mode: stride
  full_time_stride: 36
  min_train_ratio: 0.40
  label_export_enabled: true

eval_labeling:
  val_mode: horizon_stride
  test_mode: horizon_stride
  apply_sampling: false
  apply_augmentation: false

dp:
  horizon: 72
  gamma: 1.0
  max_position: 1
  cost_config:
    reward_alignment: paper_formula
    commission_rate: 0.0005
    slippage_model: lob_depth
    book_levels: 5
    mark_price: mid_price
    execution_lag: 0

dp_workers: 0
dp_worker_chunksize: 32
```

说明：

- `data_process_hash` 覆盖输入文件审计、factor source、horizon、采样、分层、no-trade、time distribution、增强和 seed。
- `dp_teacher_hash` 基于 `data_process_hash + DPConfig + CostConfig`。
- 修改模型训练超参不会改变 `data_process_hash` 或 `dp_teacher_hash`。

## 6. 主流程

`src/preprocess_data/processor.py::Phase1DataProcessor.run()` 的主流程：

```text
build artifacts_dir
seed_everything(seed)
check prospective diagnostic contract
read train/val/test frames
validate train schema
validate val/test against train schema
write input_schema.json

for split=train:
  enumerate all window entries
  apply split boundary embargo
  assign strata
  sample train windows
  run sampling health check
  build sampled horizon records
  run DP teacher
  backfill no-trade / low-opportunity coverage if needed
  optionally build full-time train records
  build non-overlap train records

for split=val/test:
  enumerate all window entries
  apply split boundary embargo
  assign strata
  select eval label windows by horizon_stride or all_eligible
  build sampled horizon records
  run DP teacher
  build non-overlap records

compute input file audit
compute schema_hash/data_process_hash/dp_teacher_hash
write feature_provenance.json
write split artifacts
write compatibility demos
write data_process_manifest.json
```

### 6.1 Window 枚举

`SlidingWindowIndexer` 根据 `reward_alignment` 生成候选窗口：

```text
paper_formula:
  decision/execution row = t ... t+h-1
  markout row = t+1 ... t+h

next_row_execution:
  observation row = t ... t+h-1
  execution row = t+1 ... t+h
  markout row = t+2 ... t+h+1
```

所有窗口必须满足 split 文件内部有完整 execution 和 markout 行。边界不足的窗口通过 `split_boundary_embargo` 排除。

### 6.2 Train 采样

train split 默认不是全量滑窗训练，而是预算采样：

```text
num_demos = min(config.num_demos, eligible_windows)
```

采样由两部分组成：

1. `full_time` 时间覆盖样本：从 eligible 窗口中按固定 stride 均匀覆盖训练时间轴。
2. `opportunity` 分层机会样本：在剩余额度内按 strata 采样。

`sample_source` 记录样本来源：

```text
full_time | opportunity | both | non_overlap
```

采样完成后必须执行 `SamplingHealthChecker`，检查：

- `window_overlap_ratio`
- `min_sample_gap`
- `flat_low_vol_sample_ratio`
- split boundary embargo
- overlap relaxation 是否发生

### 6.3 Val/Test 标注

val/test 不参与训练采样，不做增强。默认模式：

```text
eval_labeling.val_mode = horizon_stride
eval_labeling.test_mode = horizon_stride
```

即每 `horizon` 个 eligible 起点选择一个窗口做 DP label。可选 `all_eligible` 用于更密集诊断，但不作为默认。

### 6.4 Non-Overlap 产物

数据处理阶段为 train/val/test 额外生成 non-overlap horizon：

```text
non_overlap_horizons_{split}.feather
non_overlap_dp_teacher_{split}.feather
```

用途：

- Phase II selector 训练与验证。
- Horizon-level label 生成。
- 避免 Phase II 使用高度重叠窗口造成收益和 label 泄漏。

non-overlap 起点间隔至少为 `horizon`。

### 6.5 DP Teacher

DP teacher 使用：

```text
SingleTradeDPPlanner
Phase1DemoGenerator
LobDepthCostModel
RewardAlignment
```

输出动作语义：

```text
0 = short
1 = flat
2 = long
```

DP 约束：

- 每个 horizon 最多一次交易切换。
- 手续费、滑点、盘口深度不足 reject 通过统一 cost/reward 接口计算。
- DP 结果只写入 teacher 文件，不进入 sampled horizon 文件。

## 7. 产物目录

默认输出目录：

```text
artifacts/{PAIR}/{DATA_BATCH_ID}/phase1/
```

核心产物：

```text
input_schema.json
feature_provenance.json
data_process_manifest.json

window_index_train.feather
window_index_val.feather
window_index_test.feather

sampled_horizons_train.feather
sampled_horizons_val.feather
sampled_horizons_test.feather

sampled_dp_teacher_train.feather
sampled_dp_teacher_val.feather
sampled_dp_teacher_test.feather

reject_stats_train.json
reject_stats_val.json
reject_stats_test.json

non_overlap_horizons_train.feather
non_overlap_horizons_val.feather
non_overlap_horizons_test.feather

non_overlap_dp_teacher_train.feather
non_overlap_dp_teacher_val.feather
non_overlap_dp_teacher_test.feather
```

当 full-time train 启用时，额外产物：

```text
sampled_horizons_full_time_train.feather
sampled_dp_teacher_full_time_train.feather
```

兼容产物：

```text
demos_train.feather
demos_val.feather
demos_test.feather
non_overlap_demos_train.feather
non_overlap_demos_val.feather
non_overlap_demos_test.feather
```

兼容 demos 可以保留，但新训练主契约是：

```text
data_process_manifest.json
sampled_horizons_{split}.feather
sampled_dp_teacher_{split}.feather
```

## 8. 文件契约

### 8.1 `data_process_manifest.json`

manifest 是训练阶段唯一入口：

```json
{
  "version": 2,
  "phase": "phase1_data_process",
  "pair": "FU",
  "data_batch_id": "batch_001",
  "artifact_dir": "artifacts/FU/batch_001/phase1",
  "input_files": {
    "train": "data/FU/train.feather",
    "val": "data/FU/val.feather",
    "test": "data/FU/test.feather"
  },
  "input_schema_path": "input_schema.json",
  "feature_provenance_path": "feature_provenance.json",
  "schema_hash": "...",
  "data_process_hash": "...",
  "dp_teacher_hash": "...",
  "feature_source": {},
  "splits": {
    "train": {
      "window_index_path": "window_index_train.feather",
      "sampled_horizons_path": "sampled_horizons_train.feather",
      "dp_teacher_path": "sampled_dp_teacher_train.feather",
      "reject_stats_path": "reject_stats_train.json",
      "non_overlap_horizons_path": "non_overlap_horizons_train.feather",
      "non_overlap_dp_teacher_path": "non_overlap_dp_teacher_train.feather",
      "num_horizons": 30000,
      "labeling_mode": "sampled_train",
      "sampling_applied": true
    }
  }
}
```

manifest 中路径默认相对 `artifact_dir`，便于整体移动 artifact 目录。

### 8.2 `sampled_horizons_{split}.feather`

保存 horizon 本体，不保存 DP actions/rewards。

必需字段：

| 字段 | 说明 |
| --- | --- |
| `sample_id` | 与 teacher join 的稳定主键 |
| `pair` / `split` | 标的与 split |
| `start_index` / `end_index` | horizon 起止行 |
| `last_execution_row` / `last_markout_row` | reward alignment 审计字段 |
| `strata_label` | 分层标签 |
| `sample_source` | `full_time/opportunity/non_overlap` 等来源 |
| `states` | `[h, feature_dim]`，不包含 `close` |
| `prices` | `[h+1]` 或 `[h+2]` |
| `execution_books` | JSON 编码的盘口深度 |
| `is_augmented` / `augmentation_type` | 增强审计 |
| `_schema_hash` | schema hash |
| `_data_process_hash` | 数据处理 hash |

### 8.3 `sampled_dp_teacher_{split}.feather`

保存 DP teacher 结果。

必需字段：

| 字段 | 说明 |
| --- | --- |
| `sample_id` | 与 sampled horizon join 的稳定主键 |
| `pair` / `split` | 标的与 split |
| `sample_source` | 样本来源 |
| `actions` | `[h]`，DP teacher 目标仓位动作 |
| `rewards` | `[h]`，DP teacher step reward |
| `teacher_return` | `sum(rewards)` |
| `num_switches` | action 切换次数 |
| `is_no_trade` | 是否全程 flat |
| `reject_transition_count` | 被拒绝转移数量 |
| `reject_transition_rate` | 被拒绝转移比例 |
| `_schema_hash` | schema hash |
| `_data_process_hash` | 数据处理 hash |
| `_dp_teacher_hash` | DP teacher hash |

## 9. 训练消费契约

Phase I 训练使用：

```bash
python scripts/train_phase1.py \
  --pair FU \
  --train-batch-id phase1_vq_001 \
  --data-process-manifest artifacts/FU/batch_001/phase1/data_process_manifest.json
```

训练阶段通过 `src/preprocess_data/processed_store.py::Phase1ProcessedStore` 加载：

```python
store = Phase1ProcessedStore(artifact_dir)
manifest = store.load_manifest(path)
schema = store.load_schema(manifest)
train_records = store.load_records(manifest, "train")
val_records = store.load_records(manifest, "val")
test_records = store.load_records(manifest, "test")
```

必须校验：

- manifest `phase == "phase1_data_process"`。
- manifest 包含 train/val/test 三个 split。
- `input_schema.json` hash 等于 `schema_hash`。
- sampled horizons 与 DP teacher 的 `sample_id` 集合完全一致。
- `pair/split/_schema_hash/_data_process_hash/_dp_teacher_hash` 一致。
- `actions/rewards` 长度等于 horizon 长度。

训练阶段禁止调用：

```text
MarketFileReader.read_split
SlidingWindowIndexer.enumerate
StratifiedWindowSampler.sample
HorizonBuilder.build
Phase1DemoGenerator.generate
SingleTradeDPPlanner.plan
```

## 10. Phase II 衔接

数据处理阶段生成的 non-overlap 产物是 Phase II 的数据基础：

```text
non_overlap_horizons_train.feather
non_overlap_horizons_val.feather
non_overlap_horizons_test.feather
non_overlap_dp_teacher_train.feather
non_overlap_dp_teacher_val.feather
non_overlap_dp_teacher_test.feather
```

Phase I 训练完成后，用 best VQ model 对这些 non-overlap demonstrations 编码，导出 Phase II selector label：

```text
non_overlap_horizon_labels_train.feather
non_overlap_horizon_labels_val.feather
non_overlap_horizon_labels_test.feather
```

语义边界：

- 数据处理阶段只生成 DP teacher actions/rewards，不生成 VQ code label。
- VQ code label 必须由 Phase I best checkpoint 生成。
- Phase II 只能消费已固化的 non-overlap horizon 和 label，不重新运行 DP。

## 11. 命令示例

Hindsight 主实验：

```bash
python scripts/pre_process_data.py \
  --pair FU \
  --data-batch-id batch_001 \
  --train-file data/FU/train.feather \
  --val-file data/FU/val.feather \
  --test-file data/FU/test.feather \
  --factor-profile short \
  --horizon 72 \
  --num-demos 30000 \
  --sampling-strategy stratified_uniform \
  --stratification-mode hindsight_horizon \
  --diagnostic-pair-batch-id batch_001_prospective \
  --reward-alignment paper_formula \
  --max-position 1 \
  --artifact-root artifacts
```

Prospective 诊断批次：

```bash
python scripts/pre_process_data.py \
  --pair FU \
  --data-batch-id batch_001_prospective \
  --train-file data/FU/train.feather \
  --val-file data/FU/val.feather \
  --test-file data/FU/test.feather \
  --factor-profile short \
  --horizon 72 \
  --num-demos 30000 \
  --sampling-strategy stratified_uniform \
  --stratification-mode prospective_past \
  --reward-alignment paper_formula \
  --max-position 1 \
  --artifact-root artifacts
```

Phase I 训练：

```bash
python scripts/train_phase1.py \
  --pair FU \
  --train-batch-id phase1_vq_001 \
  --data-process-manifest artifacts/FU/batch_001/phase1/data_process_manifest.json \
  --epochs 100 \
  --batch-size 256 \
  --lr 0.001 \
  --device cuda
```

## 12. 审计与失败条件

必须 fail-fast 的情况：

- 缺少 train/val/test 任一输入文件。
- factor list 中字段不存在。
- `close` 被放入 `feature_columns`。
- val/test schema 与 train schema 不一致。
- hindsight 主实验缺少 prospective diagnostic，且未显式风险声明。
- sampled horizon 与 DP teacher 的 `sample_id` 不完全一致。
- hash 不一致。
- `actions/rewards` 长度不等于 horizon 长度。
- sampling health 超阈值且 `warn_only=false`。
- DP reject rate 超过配置阈值。

可 warning 但必须写入 manifest/report 的情况：

- `sampling_health.warn_only=true`。
- overlap relaxation 被启用并实际发生。
- no-trade / low-opportunity 回填仍未达到目标比例。
- 使用 `--allow-missing-prospective-diagnostic` 放行 hindsight 主实验。

## 13. 测试计划

单元测试：

```text
tests/unit/scripts/test_phase1_data_processor.py
tests/unit/preprocess_data/test_processed_store.py
tests/unit/preprocess_data/test_schema.py
tests/unit/preprocess_data/test_window_indexer.py
tests/unit/preprocess_data/test_stratified_sampler.py
tests/unit/preprocess_data/test_sampling_health.py
```

集成 smoke：

```text
tests/integration/test_phase1_data_process_then_train.py
```

必须覆盖：

- `scripts/pre_process_data.py` 能写出 manifest 和所有 split 产物。
- `Phase1ProcessedStore` 能从 manifest 加载 train/val/test records。
- 缺 teacher、额外 teacher、hash mismatch、wrong split 均会失败。
- `train_phase1.py` 在 manifest 模式下不重新枚举窗口、不重新运行 DP。
- non-overlap 产物存在并可被 Phase II 数据集消费。
