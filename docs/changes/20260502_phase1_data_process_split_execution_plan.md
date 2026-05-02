# Phase I 数据预处理拆分变更执行计划

**日期**: 2026-05-02
**来源设计**: `docs/design/20260502_phase1_data_process_split_design.md`
**影响阶段**: Phase I Archetype Discovery

---

## 1. 执行目标

本计划把 Phase I 数据预处理拆分设计转换为可实施的代码变更批次。核心目标是把分层采样、horizon 构建和 DP teacher 生成从训练主流程拆出，让 `scripts/train_phase1.py` 在 manifest 模式下只读取已经固化的分片训练集合和 DP teacher 数据。

本次执行坚持四个边界:

- 只拆 Phase I 数据准备与训练编排，不改 DP、采样、模型、loss 和 Phase II 语义。
- 新增 manifest 训练模式，但保留当前一体化路径作为迁移期兼容路径。
- 引入必要的数据 manifest/hash 校验，但不做完整 artifact manifest/cache 体系重构。
- 每个代码批次必须包含相应单元测试或集成 smoke，并执行指定 pytest 命令。

---

## 2. 执行情况看板

| ID | 事项 | 决策 | 优先级 | 当前状态 | 测试状态 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| D0 | 生成变更执行计划文档 | 采纳 | P0 | 【✅】DONE | 不适用 | 本文件 |
| D1 | 回写 design 中的不采纳原因 | 采纳 | P0 | 【✅】DONE | 不适用 | 见来源设计 §13 |
| A1 | 新增 processed store 与 manifest 校验 | 采纳 | P0 | 【✅】DONE | 【✅】通过 | 固化 sampled horizons + DP teacher 文件契约 |
| A2 | 新增 `scripts/phase1_data_processor.py` 与 processor | 采纳 | P0 | 【✅】DONE | 【✅】通过 | 先跑脚本生成固化数据，再由训练读取产物 |
| A3 | `train_phase1.py` 支持 manifest 训练模式 | 采纳 | P0 | 【✅】DONE | 【✅】通过 | manifest 模式跳过 raw reader、采样和 DP |
| A4 | report/config 记录数据预处理来源与 hash | 采纳 | P1 | 【✅】DONE | 【✅】通过 | 提供训练报告审计链 |
| A5 | 单元测试与集成 smoke 覆盖新流程 | 采纳 | P0 | 【✅】DONE | 【✅】通过 | store、processor、trainer manifest 模式 |
| B1 | 立即移除旧一体化训练路径 | 不采纳 | P1 | DROP | 不适用 | 结构性风险高，先保留兼容 |
| B2 | 将 manifest 改为所有训练的唯一强制入口 | 暂缓 | P1 | DEFER | 不适用 | 新流程稳定后再决定 |
| B3 | 重构完整 artifact manifest/cache 体系 | 不采纳 | P2 | DROP | 不适用 | 本批次只做必要数据 manifest |
| B4 | 大规模拆分 `Phase1Config` 与训练配置 hash 体系 | 暂缓 | P2 | DEFER | 不适用 | 先做轻量配置扩展 |
| B5 | 生成 `demos_val.feather` / `demos_test.feather` 兼容产物 | 不采纳 | P2 | DROP | 不适用 | 会扩大存储与契约面 |
| B6 | 对输入 feather 做全文件内容 hash | 暂缓 | P2 | DEFER | 不适用 | 大文件成本高，先记录路径/大小/mtime 审计 |
| B7 | 新增 tensor cache / shard 存储层 | 不采纳 | P2 | DROP | 不适用 | 属于性能专项，不是本次拆分必需 |
| B8 | 重命名 `scripts/train_phase1.py` 为 `phase1_train.py` | 不采纳 | P2 | DROP | 不适用 | 避免破坏现有命令与测试引用 |

状态说明:

- `TODO`: 尚未实现。
- `DONE`: 本计划文档层面已完成。
- `DROP`: 明确不纳入本批次。
- `DEFER`: 后续单独设计或新批次处理。

---

## 3. 采纳结论

### 3.1 采纳项

| 设计项 | 决策 | 批次 | 采纳原因 |
| --- | --- | --- | --- |
| 新增 `scripts/phase1_data_processor.py` | 采纳 | A2 | 是实现“先生成数据，再训练”的最小入口变更 |
| 在 `scripts/phase1_data_processor.py` 中抽出 `Phase1DataProcessor` 或等价编排函数 | 采纳 | A2 | 让数据预处理逻辑离开 trainer 主训练循环，同时复用现有组件 |
| 新增 `sampled_horizons_{split}.feather` | 采纳 | A1/A2 | 固化分层采样后的训练集合，训练阶段不再重新采样 |
| 新增 `dp_teacher_{split}.feather` | 采纳 | A1/A2 | DP teacher actions/rewards 与 sampled horizons 分离，便于审计和 hash 校验 |
| 生成 `data_process_manifest.json` | 采纳 | A1/A2 | 作为训练读取预处理数据的入口，集中记录路径、hash、split 行数 |
| train/val/test 都持久化 DP teacher | 采纳 | A2/A3 | validation、labels 和 teacher replay 都需要同一批固化 teacher 数据 |
| `train_phase1.py --data-process-manifest` | 采纳 | A3 | 满足训练从文件读取分片训练集合和 DP teacher 数据的核心需求 |
| manifest 模式禁止调用采样和 DP | 采纳 | A3/A5 | 防止训练超参变化意外重生成数据，保证实验可比 |
| report 记录 manifest/hash | 采纳 | A4 | 训练报告必须能回溯到数据预处理批次 |
| processed store 与 trainer 单元测试 | 采纳 | A1/A3/A5 | 新文件契约属于训练入口基础保障，必须防回归 |
| data process + train 集成 smoke | 采纳 | A5 | 验证两个命令串联能跑通，覆盖真实编排边界 |

### 3.2 部分采纳项

| 设计项 | 决策 | 批次 | 采纳边界 |
| --- | --- | --- | --- |
| `Phase1DataProcessConfig` | 部分采纳 | A2 | 新增轻量 dataclass 或 builder，复用现有嵌套配置类；不重写整个 Phase I 配置体系 |
| hash 拆分 | 部分采纳 | A1/A4 | 本批次落地 `schema_hash`、`data_process_hash`、`dp_teacher_hash`；训练 hash 继续复用现有 `config_hash` |
| manifest 是训练入口 | 部分采纳 | A3 | 仅在 manifest 模式下作为唯一数据入口；迁移期不强制所有训练都必须走 manifest |
| 兼容导出 `demos_train.feather` | 部分采纳 | A2 | 可继续生成 `demos_train.feather` 以兼容现有测试/下游；不新增 `demos_val/test` 兼容文件 |

### 3.3 不采纳或暂缓项

| 设计项 | 决策 | 原因 | 后续处理 |
| --- | --- | --- | --- |
| 立即移除旧一体化训练路径 | 不采纳 | 训练主流程体量大，一次性移除会扩大回归面；当前 smoke、旧实验和本地调试仍依赖旧路径 | manifest 模式稳定后再单独评估 |
| 将 manifest 设为所有训练强制必填 | 暂缓 | 需要先完成数据处理产物兼容、文档和 CI 覆盖，否则会阻塞现有实验 | 先支持可选 `--data-process-manifest` |
| 完整 artifact manifest/cache 体系重构 | 不采纳 | 属于跨阶段结构性工程，不是拆分采样/DP 的必要条件 | 保持本批次 manifest 只服务 Phase I data process |
| 大规模拆分训练配置和新增独立 `training_config_hash` | 暂缓 | 配置体系是多个测试与报告的公共契约，贸然重写风险高 | 继续使用现有 `config_hash`，后续配置治理专项处理 |
| 生成 `demos_val.feather` 和 `demos_test.feather` | 不采纳 | 新主契约已是 sampled horizons + DP teacher；额外兼容文件会增加存储、测试和混用风险 | 只保留必要的 `demos_train.feather` 兼容 |
| 输入 feather 全文件 sha256 | 暂缓 | 生产数据文件可能很大，全量 hash 会增加预处理启动成本 | manifest 先记录路径、size、mtime_ns；必要时后续加 `--hash-input-files` |
| tensor cache / shard 存储 | 不采纳 | 属于性能和大规模数据专项，本批次先保持 feather 主产物 | 后续依据运行瓶颈单独设计 |
| `train_phase1.py` 重命名为 `phase1_train.py` | 不采纳 | 当前仓库和测试引用均使用 `train_phase1.py`；重命名收益低且容易破坏脚本调用 | 如确有兼容需要，后续只加薄 wrapper |

---

## 4. 批次 A1: Processed Store 与 Manifest

**涉及文件**:

- `src/data/phase1_processed_store.py`
- `src/data/demo_store.py`
- `src/data/__init__.py`
- `tests/unit/data/test_phase1_processed_store.py`

**实现方案**:

1. 新增 `Phase1DataProcessManifest` / `Phase1SplitArtifact` 数据结构，支持从 `data_process_manifest.json` 读取和校验。
2. 新增 `Phase1ProcessedStore`:
   - `save_sampled_horizons(split, records, schema_hash, data_process_hash)`
   - `save_dp_teacher(split, records, reject_stats, schema_hash, data_process_hash, dp_teacher_hash)`
   - `write_manifest(payload)`
   - `load_records(manifest, split)`
   - `join_horizons_with_teacher(sampled, teacher)`
3. `sampled_horizons_{split}.feather` 只保存 states/prices/execution_books/meta，不保存 `actions/rewards`。
4. `dp_teacher_{split}.feather` 保存 `actions/rewards/teacher_return/num_switches/is_no_trade/reject_transition_*`。
5. 读取时校验:
   - `sample_id` 集合完全一致。
   - split、pair、hash 与 manifest 一致。
   - `actions/rewards` 长度与 horizon 长度一致。
   - 缺文件、重复 `sample_id`、多余或缺失 teacher 都 fail-fast。
6. 迁移期可复用 `Phase1DemoStore.save_demos()` 生成 `demos_train.feather`，但 processed store 的加载不依赖它。

**测试修复**:

- `test_processed_store_saves_and_loads_records`
- `test_processed_store_joins_teacher_by_sample_id`
- `test_processed_store_rejects_missing_teacher_sample`
- `test_processed_store_rejects_extra_teacher_sample`
- `test_processed_store_rejects_hash_mismatch`
- `test_processed_store_rejects_wrong_split`
- `test_processed_store_rejects_action_reward_length_mismatch`

**执行命令**:

```bash
pytest tests/unit/data/test_phase1_processed_store.py tests/unit/data/test_demo_store.py -q
```

---

## 5. 批次 A2: 数据预处理入口与 Processor

**涉及文件**:

- `scripts/phase1_data_processor.py`
- `src/config/phase1_config.py`
- `src/trainers/phase1_trainer.py`
- `tests/unit/scripts/test_phase1_data_processor.py`

**实现方案**:

1. 新增 `scripts/phase1_data_processor.py`，CLI 参数复用现有 `train_phase1.py` 中影响数据生成的字段:
   - `--pair`
   - `--data-batch-id`
   - `--train-file/--val-file/--test-file`
   - `--artifact-root`
   - `--factor-profile/--factor-list-file`
   - `--horizon/--num-demos`
   - `--sampling-strategy/--stratification-mode`
   - `--diagnostic-pair-batch-id`
   - `--max-position`
   - sampling health 和 prospective diagnostic 相关参数
2. 新增轻量 `Phase1DataProcessConfig` 或 `build_data_process_config()`:
   - 复用 `StratificationConfig`、`SamplingHealthConfig`、`DataAugmentationConfig`、`DPConfig`。
   - 不引入新的训练模型字段。
3. 在 `scripts/phase1_data_processor.py` 中新增 `Phase1DataProcessor.run()` 或等价编排函数，从 trainer 现有流程中复用/抽取:
   - seed 设置。
   - prospective diagnostic 检查。
   - market file 读取。
   - schema validation。
   - window index + stratified sampling + horizon build。
   - train temporal contrastive augmentation。
   - train/val/test DP teacher 生成。
   - manifest、sampled horizons、teacher 文件写入。
4. `data_process_hash` 由会影响 sampled horizons 的配置生成:
   - input path audit 信息。
   - factor source。
   - horizon、num_demos、sampling、stratification、sampling health、seed、augmentation。
5. `dp_teacher_hash` 基于 `data_process_hash + DPConfig + CostConfig` 生成。

**测试修复**:

- `test_phase1_data_processor_writes_manifest_and_split_files`
- `test_phase1_data_processor_records_schema_and_hashes`
- `test_phase1_data_processor_writes_val_and_test_teacher`
- `test_phase1_data_processor_preserves_deterministic_sample_ids`
- `test_phase1_data_processor_rejects_missing_prospective_diagnostic`

**执行命令**:

```bash
pytest tests/unit/scripts/test_phase1_data_processor.py tests/unit/data/test_phase1_processed_store.py -q
```

---

## 6. 批次 A3: Trainer Manifest 训练模式

**涉及文件**:

- `scripts/train_phase1.py`
- `src/config/phase1_config.py`
- `src/trainers/phase1_trainer.py`
- `src/data/phase1_processed_store.py`
- `tests/unit/trainers/test_phase1_trainer.py`
- `tests/unit/scripts/test_train_phase1_cli.py`

**实现方案**:

1. `scripts/train_phase1.py` 新增:

```text
--data-process-manifest artifacts/AL/al_short_20260502/phase1/data_process_manifest.json
```

2. 当传入 manifest 时，`--train-file/--val-file/--test-file` 不再必填；未传 manifest 时保持当前必填行为。
3. `Phase1Config` 新增低风险字段:

```python
data_process_manifest: Optional[str] = None
```

4. `Phase1Trainer.run()` 开始后分支:
   - manifest 模式: 调 `Phase1ProcessedStore.load_records()` 加载 `scripts/phase1_data_processor.py` 已生成的 train/val/test records 和 schema。
   - legacy 模式: 保持当前 raw reader + schema + sampler + DP 的一体化流程。
5. manifest 模式下禁止调用:
   - `MarketFileReader.read_split`
   - `SlidingWindowIndexer.enumerate`
   - `StratifiedWindowSampler.sample`
   - `HorizonBuilder.build`
   - `Phase1DemoGenerator.generate`
   - `SingleTradeDPPlanner.plan`
   - `scripts/phase1_data_processor.py` 的运行时入口
6. manifest 模式继续执行:
   - `RewardNormalizer.fit_train`
   - model/evaluator/loss/optimizer 初始化
   - codebook warmup
   - train loop
   - labels、Phase II artifacts、report 导出
7. manifest 中的 `pair` 与 CLI/config `pair` 不一致时 fail-fast。

**测试修复**:

- `test_train_phase1_cli_manifest_mode_does_not_require_raw_files`
- `test_train_phase1_cli_legacy_mode_still_requires_raw_files`
- `test_phase1_trainer_manifest_mode_loads_processed_records`
- `test_phase1_trainer_manifest_mode_does_not_sample_or_run_dp`
- `test_phase1_trainer_manifest_pair_mismatch_fails`
- `test_phase1_trainer_manifest_missing_file_fails`

**执行命令**:

```bash
pytest tests/unit/trainers/test_phase1_trainer.py tests/unit/scripts/test_train_phase1_cli.py -q
```

---

## 7. 批次 A4: Report、Config 与审计字段

**涉及文件**:

- `src/evaluation/phase1_report.py`
- `src/trainers/phase1_trainer.py`
- `src/config/phase1_config.py`
- `tests/unit/evaluation/test_phase1_report.py`
- `tests/unit/trainers/test_phase1_trainer.py`

**实现方案**:

1. `phase1_config.yaml` 记录 `data_process_manifest`。
2. `phase1_report.json` 写入:
   - `data_process_manifest`
   - `data_batch_id`
   - `schema_hash`
   - `data_process_hash`
   - `dp_teacher_hash`
   - `processed_data_mode`
3. legacy 模式写入 `processed_data_mode="legacy_inline"`。
4. manifest 模式写入 `processed_data_mode="manifest"`。
5. report 中不要把 `training_config_hash` 作为新顶层语义强推；继续记录现有 `config_hash`，并在说明中把它视为训练配置审计 hash。

**测试修复**:

- `test_phase1_report_records_data_process_manifest`
- `test_phase1_report_records_processed_data_hashes`
- `test_phase1_report_marks_legacy_inline_mode`
- `test_phase1_config_yaml_includes_data_process_manifest`

**执行命令**:

```bash
pytest tests/unit/evaluation/test_phase1_report.py tests/unit/trainers/test_phase1_trainer.py -q
```

---

## 8. 批次 A5: 集成 Smoke 与执行矩阵

**涉及文件**:

- `tests/integration/test_phase1_data_process_then_train.py`
- `tests/integration/test_phase1_pipeline_smoke.py`
- `tests/fixtures/phase1/README.md`

**实现方案**:

1. 新增集成测试:
   - 用 fixture market data 运行 data processor。
   - 读取生成的 manifest。
   - 用 manifest 模式启动 trainer smoke。
   - 断言 checkpoint、labels、report 生成。
2. 保留现有 `test_phase1_pipeline_smoke.py` 覆盖 legacy 一体化路径。
3. smoke 中使用小样本、小 epoch、`--local-smoke-relaxed-guardrails`，避免测试耗时过高。
4. 断言 manifest 模式下日志/spy 没有进入采样和 DP 生成路径。

**执行命令**:

```bash
pytest tests/unit/data/test_phase1_processed_store.py -q
pytest tests/unit/scripts/test_phase1_data_processor.py -q
pytest tests/unit/trainers/test_phase1_trainer.py tests/unit/scripts/test_train_phase1_cli.py -q
pytest tests/unit/evaluation/test_phase1_report.py -q
pytest tests/integration/test_phase1_data_process_then_train.py tests/integration/test_phase1_pipeline_smoke.py -q
```

---

## 9. 最终验收

完成本计划后必须满足:

1. `python scripts/phase1_data_processor.py ...` 能生成 `data_process_manifest.json`、`sampled_horizons_{split}.feather`、`dp_teacher_{split}.feather`。
2. `python scripts/train_phase1.py --data-process-manifest ...` 可以不传 raw split 文件完成训练。
3. manifest 模式下 trainer 不会重新执行分层采样或 DP teacher 生成。
4. 修改或删除任意 processed data 文件会 fail-fast。
5. sampled horizons 与 DP teacher 的 `sample_id` 不一致会 fail-fast。
6. 改变训练超参不会改变或重写 data process 产物。
7. legacy 一体化路径仍能通过原有 smoke。
8. `phase1_report.json` 能回溯 manifest 与三类 hash。
9. 所有新增/更新的 unit tests 和 integration smoke 已执行通过。

---

## 10. 推荐执行顺序

1. 先做 A1 store 和 manifest，锁定文件契约。
2. 再做 A2 data processor，确认能单独产出数据。
3. 再做 A3 trainer manifest 模式，切断训练对采样/DP 的依赖。
4. 补 A4 report/config 审计字段。
5. 最后跑 A5 全量 smoke，并根据失败点回补单测。

这个顺序可以让每一步都有可运行的边界，避免把 processor、trainer、report 和测试一次性搅在一起。

---

## 11. 执行结果回写

**执行日期**: 2026-05-02

本次已完成 A1-A5:

- A1: 新增 `src/data/phase1_processed_store.py`，实现 `data_process_manifest.json` 读取校验、`sampled_horizons_{split}.feather` 与 `dp_teacher_{split}.feather` 分离保存、`sample_id` join、split/pair/hash/action-reward 长度 fail-fast 校验。
- A2: 新增 `scripts/phase1_data_processor.py`，可先离线读取 raw split、生成 window index / sampled horizons / DP teacher / reject stats / `data_process_manifest.json`，并保留 `demos_train.feather` 兼容导出。
- A3: `scripts/train_phase1.py` 支持 `--data-process-manifest`；manifest 模式下 raw split 文件不再必填，`Phase1Trainer.run()` 只加载 data processor 产物并跳过 raw reader、window sampling、horizon build 和 DP teacher 生成。
- A4: `phase1_config.yaml` 记录 `data_process_manifest`；`phase1_report.json` 写入 `processed_data_mode`、`data_process_manifest`、`data_batch_id`、`schema_hash`、`data_process_hash`、`dp_teacher_hash`。
- A5: 新增 processed store、data processor、train CLI、trainer manifest、两阶段 integration smoke 覆盖；保留旧 `test_phase1_pipeline_smoke.py` 验证 legacy inline 路径。

已在 `conda activate ArchetypeTrade` 环境执行:

```bash
pytest tests/unit/data/test_phase1_processed_store.py tests/unit/data/test_demo_store.py -q
pytest tests/unit/scripts/test_phase1_data_processor.py tests/unit/data/test_phase1_processed_store.py -q
pytest tests/unit/trainers/test_phase1_trainer.py tests/unit/scripts/test_train_phase1_cli.py -q
pytest tests/unit/evaluation/test_phase1_report.py tests/unit/trainers/test_phase1_trainer.py tests/unit/config/test_phase1_config_docs.py -q
pytest tests/integration/test_phase1_data_process_then_train.py tests/integration/test_phase1_pipeline_smoke.py -q
```

结果:

- `10 passed`
- `12 passed`
- `18 passed`
- `23 passed`
- `5 passed`

额外检查:

```bash
git diff --check
```

结果: 通过。

修正 `data_process_hash` 仅覆盖真实数据处理输入后，追加执行:

```bash
pytest tests/unit/scripts/test_phase1_data_processor.py tests/integration/test_phase1_data_process_then_train.py -q
```

结果: `6 passed`。

最终合并执行本批次测试集合:

```bash
pytest tests/unit/data/test_phase1_processed_store.py tests/unit/data/test_demo_store.py tests/unit/scripts/test_phase1_data_processor.py tests/unit/trainers/test_phase1_trainer.py tests/unit/scripts/test_train_phase1_cli.py tests/unit/evaluation/test_phase1_report.py tests/unit/config/test_phase1_config_docs.py tests/integration/test_phase1_data_process_then_train.py tests/integration/test_phase1_pipeline_smoke.py -q
```

结果: `45 passed`。
