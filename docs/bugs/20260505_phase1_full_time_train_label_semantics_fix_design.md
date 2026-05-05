# Phase I `horizon_labels_full_time_train.feather` 语义偏差修复设计

**日期**: 2026-05-05
**状态**: 设计中
**影响范围**: Phase I 数据预处理、Phase I label 导出、Phase II `phase1_label_source=full_time`

## 1. Bug 判断

用户怀疑:

> `sampled_horizons_train.feather` 没有生成 `sampled_horizons_full_time_train.feather`，导致 `horizon_labels_full_time_train.feather` 的含义不对。

结论: 这个判断基本成立，但更准确地说，当前实现缺少一个独立的 full-time train horizon 输入产物，导致 `horizon_labels_full_time_train.feather` 即使被导出，也不是“训练 split 的 full-time coverage labels”，而只是“`sampled_horizons_train.feather` 中 `sample_source in {"full_time", "both"}` 的子集 labels”。

当前语义:

```text
sampled_horizons_train.feather
  -> Phase1ProcessedStore.load_records(..., "train")
  -> train_horizons
  -> filter sample_source in {"full_time", "both"}
  -> horizon_labels_full_time_train.feather
```

期望语义:

```text
sampled_horizons_full_time_train.feather
  + dp_teacher_full_time_train.feather
  -> full_time_train_horizons
  -> horizon_labels_full_time_train.feather
```

因此，问题不是文件名本身，而是 full-time label 缺少独立、可审计、可复现的 horizon/teacher 来源。

## 2. 现状证据

1. `scripts/process_phase1_data.py` 的 train sampling 会构造 `full_time_pool_entries`，但最终只把混合后的 `sampled` 保存为 `sampled_horizons_train.feather`。
2. `src/trainers/phase1_trainer.py` 导出 label 时，只从 `train_horizons` 里筛选 `sample_source in {"full_time", "both"}` 后写 `horizon_labels_full_time_train.feather`。
3. `src/trainers/phase2_trainer.py` 在 `phase1_label_source="full_time"` 时会读取 `horizon_labels_full_time_train.feather`，并把它当作 full-time train label source。

这会造成命名和下游假设不一致:

- 文件名看起来表示 train split 的 full-time coverage labels。
- 实际内容只覆盖训练预算中的 full-time 采样子集。
- 被 opportunity sampling、`num_demos`、`min_train_ratio`、no-trade backfill 替换策略共同影响。
- 无法覆盖 full-time pool 中未进入训练样本预算的窗口。

## 3. 风险

### 3.1 Phase II label coverage 偏低

Phase II 的连续时间 horizon index 可能比 Phase I 训练样本预算更密。如果 `phase1_label_source=full_time` 只拿到 train sampled 子集，join 后会出现大量 unlabeled horizon。

这会使 KL/demo regularization 实际只作用在少量窗口上，训练日志里的 full-time label source 容易被误解为“完整时间分布训练标签”。

### 3.2 指标含义失真

`full_time_label_coverage_train` 当前容易退化为 `full_time_sample_ratio`，它描述的是采样构成，不是 full-time coverage pool 的 label 覆盖率。

### 3.3 复现实验困难

`horizon_labels_full_time_train.feather` 没有自己的 manifest artifact 入口。后续只能从 label 文件倒推来源，无法严格验证:

- 这些 label 来自哪个 full-time pool。
- 是否与 `full_time_mode/full_time_stride` 一致。
- 是否包含所有应覆盖的 train full-time windows。
- DP teacher 是否对同一批 windows 生成。

## 4. 修复目标

新增明确契约:

1. `sampled_horizons_train.feather` 保持现有含义: Phase I VQ 训练用的混合训练样本，数量受 `num_demos` 控制。
2. 新增 `sampled_horizons_full_time_train.feather`: train split 的 full-time coverage horizon 集合，来源为 `full_time_pool_entries`，不受 opportunity sampling 预算截断。
3. 新增 `dp_teacher_full_time_train.feather`: 与 `sampled_horizons_full_time_train.feather` 一一对应的 DP actions/rewards。
4. `horizon_labels_full_time_train.feather` 必须只从 `sampled_horizons_full_time_train.feather + dp_teacher_full_time_train.feather` 导出。
5. 如果启用了 full-time label export，但 full-time source artifacts 缺失或 sample_id 不匹配，Phase I 应 fail-fast，不允许静默回退到 train sampled 子集。

## 5. 设计方案

### 5.1 数据预处理产物

在 `scripts/process_phase1_data.py` 中，当 `time_distribution_sampling.enabled=true` 时:

1. train split 正常生成 `sampled_horizons_train.feather` 和 `dp_teacher_train.feather`。
2. 同时基于 `full_time_pool_entries` 构造 `full_time_train_sampled`。
3. 对 `full_time_train_sampled` 调用 `HorizonBuilder.build(...)`。
4. 对这些 full-time horizons 调用同一套 `_generate_demos(...)`。
5. 保存为:

```text
sampled_horizons_full_time_train.feather
dp_teacher_full_time_train.feather
```

字段应与现有 `sampled_horizons_{split}.feather` / `dp_teacher_{split}.feather` 兼容。建议额外保证:

- `split = "train"`，不要写成 `full_time_train`，避免破坏数据语义。
- `sample_source = "full_time"`。
- `_schema_hash`、`_data_process_hash` 与 manifest 一致。
- `sample_id` 可以使用稳定前缀，例如 `train_full_time_{window_start}`，或沿用现有 `_sampled_from_entries(...)` 的规则，但必须避免与 `sampled_horizons_train.feather` 冲突。

### 5.2 Manifest 扩展

在 `data_process_manifest.json` 中为 train split 增加可选字段:

```json
{
  "splits": {
    "train": {
      "sampled_horizons_path": "sampled_horizons_train.feather",
      "dp_teacher_path": "dp_teacher_train.feather",
      "full_time_sampled_horizons_path": "sampled_horizons_full_time_train.feather",
      "full_time_dp_teacher_path": "dp_teacher_full_time_train.feather",
      "full_time_num_horizons": 6849
    }
  }
}
```

兼容策略:

- 旧 manifest 没有这些字段时，默认 `full_time_label_export_available=false`。
- 如果 Phase I 配置要求导出 full-time labels，则旧 manifest fail-fast。
- 不建议继续用“从 sampled train 子集筛选”的方式自动兼容，因为这会延续当前语义偏差。

### 5.3 Store 加载接口

在 `src/data/phase1_processed_store.py` 新增接口:

```python
load_full_time_train_records(manifest) -> List[HorizonRecord]
```

行为:

- 读取 train artifact 的 `full_time_sampled_horizons_path`。
- 读取 train artifact 的 `full_time_dp_teacher_path`。
- 复用 `join_horizons_with_teacher(...)` 做 sample_id 一致性校验。
- 校验 `pair/split/_schema_hash/_data_process_hash`。
- 如果路径缺失，抛出 `Phase1ProcessedStoreError`，错误信息包含缺失字段名和 manifest 路径。

### 5.4 Phase I label 导出

修改 `src/trainers/phase1_trainer.py` 的 label 输入构造:

当前:

```python
full_time_train_horizons = [
    rec for rec in train_horizons
    if rec.sample_source in {"full_time", "both"}
]
```

修复后:

```python
full_time_train_horizons = processed_store.load_full_time_train_records(manifest)
```

并且:

- `horizon_labels_train.feather` 继续由 `train_horizons` 导出，表示 VQ 训练样本标签。
- `horizon_labels_full_time_train.feather` 只由 `full_time_train_horizons` 导出，表示 full-time coverage 标签。
- 导出日志同时打印 `train_count`、`full_time_train_count`、`full_time_label_source=independent_artifact`。

### 5.5 Phase II 读取语义

`src/trainers/phase2_trainer.py` 现有 fail-fast 行为保留:

- `phase1_label_source="default"` 读取 `horizon_labels_train.feather`。
- `phase1_label_source="full_time"` 读取 `horizon_labels_full_time_train.feather`。

但 Phase II report 需要记录:

- `phase1_label_source`
- `phase1_label_path`
- `phase1_label_count`
- `phase1_label_coverage_on_phase2_index`

这样可以直接发现 full-time labels 与 Phase II horizon index 的覆盖关系。

## 6. 验收标准

1. 数据预处理启用 full-time sampling 后，产物目录包含:

```text
sampled_horizons_full_time_train.feather
dp_teacher_full_time_train.feather
horizon_labels_full_time_train.feather
```

2. `horizon_labels_full_time_train.feather` 行数等于 `sampled_horizons_full_time_train.feather` 行数。
3. `horizon_labels_full_time_train.feather.sample_id` 与 `sampled_horizons_full_time_train.feather.sample_id` 完全一致。
4. `horizon_labels_train.feather` 行数仍等于 `sampled_horizons_train.feather` 行数。
5. `horizon_labels_full_time_train.feather` 行数不再被 `num_demos * min_train_ratio` 截断，而由 `full_time_mode/full_time_stride` 产生的 pool 决定。
6. 当 manifest 缺少 full-time source artifacts 且配置要求 full-time label export 时，Phase I fail-fast。
7. Phase II 设置 `phase1_label_source=full_time` 时仍然缺文件 fail-fast，但读到的新文件语义为 full-time coverage labels。

## 7. 测试计划

### 7.1 Unit tests

新增或更新:

- `tests/unit/scripts/test_phase1_data_processor.py`
  - 验证 full-time train source artifacts 被生成。
  - 验证 `full_time_num_horizons` 不等于训练 sampled full-time 子集时仍可保存。
  - 验证 full-time source artifacts 的 `sample_source` 全部为 `full_time`。

- `tests/unit/data/test_phase1_processed_store.py`
  - 验证 `load_full_time_train_records(...)` 成功 join。
  - 验证缺少 `full_time_sampled_horizons_path` fail-fast。
  - 验证 `sample_id` mismatch fail-fast。

- `tests/unit/trainers/test_phase1_trainer.py`
  - 验证 `_export_horizon_labels(...)` 对 `full_time_train` 写出 `horizon_labels_full_time_train.feather`。
  - 验证 trainer 不再通过筛选 `train_horizons.sample_source` 构造 full-time labels。

### 7.2 Integration tests

更新 `tests/integration/test_phase1_pipeline_smoke.py`:

- 断言 full-time train source artifacts 存在。
- 断言 `horizon_labels_full_time_train.feather` 存在。
- 断言 full-time label 行数等于 source artifact 行数。
- 断言 `horizon_labels_full_time_train.feather` 与 `horizon_labels_train.feather` 允许不同大小、不同 sample_id 集合。

### 7.3 Regression check

保留旧行为兼容:

- `phase1_label_source=default` 不依赖 full-time artifacts。
- 旧批次没有 `horizon_labels_full_time_train.feather` 时，Phase II `phase1_label_source=full_time` 继续 fail-fast。

## 8. 非目标

本修复不改变:

- VQ 模型训练样本预算 `num_demos`。
- `horizon_labels_train.feather` 的含义。
- val/test 全量 eligible label 的既定目标。
- Phase II 默认 label source。

## 9. 建议实施顺序

1. 扩展 manifest schema，新增 full-time train artifact 字段。
2. 在数据预处理阶段保存 `sampled_horizons_full_time_train.feather` 和 `dp_teacher_full_time_train.feather`。
3. 在 `Phase1ProcessedStore` 增加 full-time train 加载接口。
4. 修改 Phase I trainer，从独立 full-time records 导出 `horizon_labels_full_time_train.feather`。
5. 增加单元测试和 smoke integration 断言。
6. 更新 report 字段，区分 sample ratio 与 label coverage。

