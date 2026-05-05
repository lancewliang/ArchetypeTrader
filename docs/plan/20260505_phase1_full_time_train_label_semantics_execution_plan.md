# Phase I Full-time Train Label 语义修复执行计划

**日期**: 2026-05-05
**来源设计**: `docs/bugs/20260505_phase1_full_time_train_label_semantics_fix_design.md`
**影响阶段**: Phase I Data Process, Phase I Archetype Discovery, Phase II Archetype Selection
**计划目标**: 修复 `horizon_labels_full_time_train.feather` 当前只表示 train sampled full-time 子集的问题，新增独立的 full-time train horizon/teacher 来源产物，使该 label 文件真正表示 train split 的 full-time coverage labels。

---

## 1. 执行原则

1. **分离训练样本与 label 覆盖**: `sampled_horizons_train.feather` 继续只表示 VQ 训练样本；full-time train labels 使用独立 source artifacts。
2. **不静默回退**: 需要 full-time labels 时，如果缺少 `sampled_horizons_full_time_train.feather` 或 `dp_teacher_full_time_train.feather`，Phase I 必须 fail-fast。
3. **保持旧默认路径可用**: `phase1_label_source=default` 和 `horizon_labels_train.feather` 语义不变。
4. **manifest 可审计**: full-time train source artifacts 必须进入 `data_process_manifest.json`，并记录数量、路径和 hash 校验。
5. **测试先锁契约**: 先覆盖文件存在性、行数、sample_id 一致性、缺文件 fail-fast，再扩展性能优化。

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
| D0 | 生成 bug 修复设计文档 | P0 | DONE | 不适用 | `docs/bugs/20260505_phase1_full_time_train_label_semantics_fix_design.md` | 已确认当前语义偏差 |
| D1 | 生成执行计划文档 | P0 | DONE | 不适用 | `docs/plan/20260505_phase1_full_time_train_label_semantics_execution_plan.md` | 本文件 |
| A1 | 扩展 manifest schema | P0 | DONE | DONE | `src/data/phase1_processed_store.py`, `scripts/process_phase1_data.py` | train split 已增加 full-time source artifact 字段 |
| A2 | 数据预处理保存 full-time train source artifacts | P0 | DONE | DONE | `scripts/process_phase1_data.py`, `src/data/phase1_processed_store.py` | 已新增 `sampled_horizons_full_time_train.feather` / `dp_teacher_full_time_train.feather` |
| A3 | Store 增加 full-time train 加载接口 | P0 | DONE | DONE | `src/data/phase1_processed_store.py` | 已复用 sample_id join 和 hash 校验 |
| A4 | Phase I trainer 改用独立 full-time records 导出 labels | P0 | DONE | DONE | `src/trainers/phase1_trainer.py`, `src/data/demo_store.py` | 已移除从 `train_horizons` 筛子集的导出逻辑；Phase I 训练目录会复制 source artifacts 供审计 |
| A5 | Phase I report 区分 sample ratio 与 label coverage | P1 | DONE | DONE | `src/trainers/phase1_trainer.py` | 已新增 full-time label count/source/path 字段 |
| A6 | Phase II coverage 诊断增强 | P1 | DONE | DONE | `src/trainers/phase2_trainer.py` | 已记录 label source/path/count 与 Phase II index 覆盖率 |
| A7 | 单元测试与集成 smoke | P0 | DONE | DONE | `tests/unit/**`, `tests/integration/**` | `ArchetypeTrade` 环境 focused suite 27 passed |
| B1 | train split stride=1 全量 labels | P2 | DEFERRED | 不适用 | - | 成本较高；本批先按 full-time pool 导出 |
| B2 | 改变 Phase II 默认 label source 为 full_time | P2 | DEFERRED | 不适用 | - | 保持默认兼容，实验显式启用 |

---

### 2.1 本轮执行记录

**执行日期**: 2026-05-05

已落地:

1. `Phase1SplitArtifact` 新增 `full_time_sampled_horizons_path`、`full_time_dp_teacher_path`、`full_time_num_horizons`，并保持旧 manifest 向后兼容。
2. `Phase1ProcessedStore.load_full_time_train_records(...)` 已实现，复用 sampled/teacher 的 pair、split、schema hash、data process hash、DP teacher hash 与 sample_id 校验。
3. `scripts/process_phase1_data.py` 在启用 `time_distribution_sampling` 时，基于 `full_time_pool_entries` 独立生成 full-time train horizons 和 DP teacher，并保存:
   - `sampled_horizons_full_time_train.feather`
   - `dp_teacher_full_time_train.feather`
4. `Phase1Trainer.run()` 已改为从 manifest 的独立 full-time records 导出 `horizon_labels_full_time_train.feather`，不再从 `train_horizons` 中筛选 `sample_source` 子集。
5. Phase I report 已新增 full-time label source/count/path 字段，并将 train sample ratio 与 label coverage 字段拆开。
6. Phase II report 已新增 `phase1_label_source`、`phase1_train_label_path`、`phase1_train_label_count`、`phase1_label_coverage_on_phase2_index`。
7. 新增/更新测试覆盖:
   - `tests/unit/data/test_phase1_processed_store.py`
   - `tests/unit/scripts/test_phase1_data_processor.py`
   - `tests/integration/test_phase1_pipeline_smoke.py`

测试结果:

1. `conda run -n ArchetypeTrade python -c "import sys, torch; print(sys.executable); print(torch.__version__)"`: **通过**，Python 路径为 `/home/lanceliang/miniconda3/envs/ArchetypeTrade/bin/python`，torch 版本为 `2.9.1+cu130`。
2. `conda run -n ArchetypeTrade pytest tests/unit/data/test_phase1_processed_store.py tests/unit/scripts/test_phase1_data_processor.py tests/unit/trainers/test_phase2_trainer.py tests/integration/test_phase1_pipeline_smoke.py`: **通过，27 passed**。

剩余验证:

1. 用一个真实 Phase I data process batch 确认 full-time train source artifacts 行数由 `full_time_mode/full_time_stride` 决定，而不是由 `num_demos * min_train_ratio` 截断。

## 3. 批次 A1: Manifest Schema 扩展

**目标**: 让 `data_process_manifest.json` 显式记录 train full-time source artifacts，避免 label 文件来源不可追溯。

**涉及文件**:

- `src/data/phase1_processed_store.py`
- `scripts/process_phase1_data.py`
- `tests/unit/data/test_phase1_processed_store.py`
- `tests/unit/scripts/test_phase1_data_processor.py`

**实现动作**:

1. 在 `Phase1SplitArtifact` 增加可选字段:
   - `full_time_sampled_horizons_path`
   - `full_time_dp_teacher_path`
   - `full_time_num_horizons`
2. `from_dict()` / `to_dict()` 保持向后兼容。
3. `process_phase1_data.py` 在 train split 且 full-time sampling 启用时写入上述字段。
4. 对旧 manifest:
   - default 训练路径可继续读取。
   - full-time label export 路径必须显式报缺字段。

**验收**:

- 新 manifest 包含 full-time source artifact 字段。
- 旧 manifest 不因默认 `load_records(..., "train")` 失败。
- 缺少 full-time 字段时，调用 full-time 加载接口报错信息包含字段名和 manifest 路径。

---

## 4. 批次 A2: 数据预处理保存 Full-time Train Source Artifacts

**目标**: 从 `full_time_pool_entries` 生成独立的 full-time train horizon/teacher 文件。

**涉及文件**:

- `scripts/process_phase1_data.py`
- `src/data/phase1_processed_store.py`
- `tests/unit/scripts/test_phase1_data_processor.py`

**实现动作**:

1. 在 train split 构建完成后保留 `full_time_pool_entries`。
2. 基于 `full_time_pool_entries` 构造 full-time sampled records:
   - `split = "train"`
   - `sample_source = "full_time"`
   - `strata_label` 从 `strata_by_start` 读取
3. 使用 `HorizonBuilder.build(...)` 生成 full-time train horizons。
4. 对 full-time train horizons 调用 `_generate_demos(...)`，生成 actions/rewards/reject stats。
5. 保存:
   - `sampled_horizons_full_time_train.feather`
   - `dp_teacher_full_time_train.feather`
6. 不把这些 records 混入 VQ 训练 `train_horizons`，避免改变训练预算。

**验收**:

- 启用 `time_distribution_sampling.enabled=true` 后，两个新文件存在。
- `sampled_horizons_full_time_train.feather.sample_source` 全部为 `full_time`。
- full-time 文件行数由 `full_time_mode/full_time_stride` 决定，不被 `num_demos * min_train_ratio` 截断。
- `sampled_horizons_train.feather` 行数与语义不变。

---

## 5. 批次 A3: Store 加载接口

**目标**: 为 Phase I trainer 提供明确的 full-time train records 输入。

**涉及文件**:

- `src/data/phase1_processed_store.py`
- `tests/unit/data/test_phase1_processed_store.py`

**实现动作**:

1. 新增 `load_full_time_train_records(manifest)`。
2. 读取 train artifact 的:
   - `full_time_sampled_horizons_path`
   - `full_time_dp_teacher_path`
3. 复用 `_load_sampled_horizons(...)` 或抽取通用加载函数，避免重复校验逻辑。
4. 复用 `join_horizons_with_teacher(...)` 校验 sample_id 一致。
5. 校验返回数量等于 `full_time_num_horizons`。

**验收**:

- full-time sampled/teacher sample_id 完全匹配时成功返回 records。
- teacher 缺行、额外行、重复 sample_id 均 fail-fast。
- hash、pair、split 不一致时 fail-fast。

---

## 6. 批次 A4: Phase I Label 导出改造

**目标**: `horizon_labels_full_time_train.feather` 只从独立 full-time records 导出。

**涉及文件**:

- `src/trainers/phase1_trainer.py`
- `src/data/demo_store.py`
- `tests/unit/trainers/test_phase1_trainer.py`
- `tests/integration/test_phase1_data_process_then_train.py`
- `tests/integration/test_phase1_pipeline_smoke.py`

**实现动作**:

1. `Phase1Trainer.run()` 在加载 train/val/test records 后，调用 `store.load_full_time_train_records(...)`。
2. 构造 label input:

```python
horizon_label_inputs = {
    "train": train_horizons,
    "val": val_horizons,
    "test": test_horizons,
    "full_time_train": full_time_train_horizons,
}
```

3. 删除或停止使用从 `train_horizons` 筛选 `sample_source in {"full_time", "both"}` 的逻辑。
4. 如果配置启用了 full-time sampling 但 full-time records 缺失，Phase I fail-fast。
5. 日志记录:
   - `train_label_count`
   - `full_time_train_label_count`
   - `full_time_label_source=independent_artifact`

**验收**:

- `horizon_labels_train.feather` 行数等于 `sampled_horizons_train.feather`。
- `horizon_labels_full_time_train.feather` 行数等于 `sampled_horizons_full_time_train.feather`。
- 两个 label 文件允许 sample_id 集合不同。
- 缺少 full-time source artifacts 时不生成误导性的 full-time labels。

---

## 7. 批次 A5: Report 与指标语义修正

**目标**: 明确区分 train sample composition 与 full-time label coverage。

**涉及文件**:

- `src/trainers/phase1_trainer.py`
- `src/evaluation/phase1_report.py`
- `tests/unit/trainers/test_phase1_trainer.py`

**实现动作**:

1. 保留 `full_time_sample_ratio`: 表示 VQ train sampled horizons 中 full-time/both 的比例。
2. 修正或新增:
   - `full_time_label_count_train`
   - `full_time_label_pool_count_train`
   - `full_time_label_coverage_train`
3. `full_time_label_coverage_train` 应基于 full-time source artifact 或 Phase II target index 计算，不再直接等于 `full_time_sample_ratio`。
4. report 中记录 full-time source artifact path。

**验收**:

- report 中 sample ratio 与 label coverage 为两个独立字段。
- 当 full-time sampled train 子集小于 full-time source artifact 时，两个指标不相等。

---

## 8. 批次 A6: Phase II Coverage 诊断增强

**目标**: Phase II 使用 `phase1_label_source=full_time` 时，清楚记录实际 join 覆盖。

**涉及文件**:

- `src/trainers/phase2_trainer.py`
- `src/data/phase2_label_loader.py`
- `tests/unit/trainers/test_phase2_trainer.py`

**实现动作**:

1. 保留现有缺 `horizon_labels_full_time_train.feather` fail-fast。
2. 在加载 labels 后记录:
   - `phase1_label_source`
   - `phase1_label_path`
   - `phase1_label_count`
   - `phase1_label_coverage_on_phase2_index`
3. 如果 coverage 低于配置阈值，给出明确 warning 或 fail-fast。

**验收**:

- `phase1_label_source=full_time` 时读取新语义文件。
- 缺文件仍 fail-fast。
- report/log 可以看到 full-time labels 对 Phase II train horizon index 的覆盖率。

---

## 9. 批次 A7: 测试与回归

**目标**: 用测试锁住新增契约，避免再次把 full-time labels 退化成 train sampled 子集。

**测试清单**:

1. `tests/unit/scripts/test_phase1_data_processor.py`
   - full-time source artifacts 存在。
   - source artifacts 行数与 manifest `full_time_num_horizons` 一致。
   - `sample_source` 全部为 `full_time`。
2. `tests/unit/data/test_phase1_processed_store.py`
   - `load_full_time_train_records(...)` 成功 join。
   - 缺路径字段 fail-fast。
   - sample_id mismatch fail-fast。
3. `tests/unit/trainers/test_phase1_trainer.py`
   - label input 使用独立 full-time records。
   - `horizon_labels_full_time_train.feather` 行数匹配 source artifact。
4. `tests/integration/test_phase1_pipeline_smoke.py`
   - 端到端生成:
     - `sampled_horizons_full_time_train.feather`
     - `dp_teacher_full_time_train.feather`
     - `horizon_labels_full_time_train.feather`
   - `horizon_labels_full_time_train.feather` 与 `horizon_labels_train.feather` sample_id 集合允许不同。
5. `tests/unit/trainers/test_phase2_trainer.py`
   - `phase1_label_source=full_time` 缺文件 fail-fast。
   - 读到文件时记录 label count / coverage。

**推荐执行命令**:

```bash
pytest tests/unit/scripts/test_phase1_data_processor.py \
  tests/unit/data/test_phase1_processed_store.py \
  tests/unit/trainers/test_phase1_trainer.py \
  tests/unit/trainers/test_phase2_trainer.py
pytest tests/integration/test_phase1_pipeline_smoke.py
```

---

## 10. 最终验收标准

1. 新 Phase I 批次产物中存在:

```text
sampled_horizons_full_time_train.feather
dp_teacher_full_time_train.feather
horizon_labels_full_time_train.feather
```

2. `horizon_labels_full_time_train.feather` 的 source 是独立 full-time train records，不是 `sampled_horizons_train.feather` 的筛选子集。
3. `horizon_labels_full_time_train.feather` 行数等于 `sampled_horizons_full_time_train.feather` 行数。
4. `horizon_labels_train.feather` 行数等于 `sampled_horizons_train.feather` 行数。
5. `phase1_label_source=full_time` 在 Phase II 中读取的是新语义文件。
6. 旧批次缺少新 source artifacts 时，不会被误当作修复后的 full-time label 批次。
