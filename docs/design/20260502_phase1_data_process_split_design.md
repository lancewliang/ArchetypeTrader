# Phase I 数据预处理拆分设计变更

**日期**: 2026-05-02
**影响阶段**: Phase I Archetype Discovery
**目标**: 将 Phase I 的分片采样生成与 DP teacher 生成从训练入口拆出，训练阶段只读取已固化的数据文件。

> 命名说明: 当前仓库中的训练入口是 `scripts/train_phase1.py`，不是 `scripts/phase1_train.py`；数据预处理入口是 `scripts/phase1_data_processor.py`。本文以下均以当前实际文件名为准；如后续需要兼容旧口径，可额外增加薄 wrapper。

---

## 1. 背景

当前 Phase I 主流程由 `scripts/train_phase1.py` 启动，并在 `Phase1Trainer.run()` 中连续完成:

1. 读取 train/val/test 原始特征文件。
2. 校验 schema 与因子清单。
3. 枚举滑动窗口并做分层采样。
4. 构建 horizon records。
5. 对 train/val/test horizon 跑 DP teacher，生成 `actions/rewards`。
6. 基于生成好的 demos 训练 VQ archetype 模型。
7. 导出 checkpoint、labels、Phase II artifacts 和报告。

这导致数据准备、DP 标注和模型训练强耦合。任何只改变训练超参的实验，例如 `epochs`、`lr`、`batch_size`、codebook 配置，都可能重新触发耗时的窗口采样和 DP teacher 生成，不利于复现实验、缓存复用和离线审计。

本次变更将 Phase I 拆为两个显式阶段:

| 阶段 | 入口 | 核心职责 | 主要产物 |
| --- | --- | --- | --- |
| 数据预处理 | `scripts/phase1_data_processor.py` | schema 校验、分层采样、horizon 构建、DP teacher 生成、manifest/hash 写入 | `sampled_horizons_*.feather`、`dp_teacher_*.feather`、`window_index_*.feather`、`data_process_manifest.json` |
| 模型训练 | `scripts/train_phase1.py` | 从文件读取分片训练集合和 DP teacher 数据，训练/验证/导出模型 | checkpoint、labels、report、Phase II artifacts |

本文中的“分片训练集合”指经过 window index + stratified sampler 选出的 horizon records，不是 train/val/test split 本身。

---

## 2. 设计目标

1. 新增 `scripts/phase1_data_processor.py`，专门生成 Phase I 训练所需的数据产物。
2. `scripts/train_phase1.py` 不再从原始行情文件生成分层采样结果，也不再运行 DP teacher；训练只读取文件中的分片训练集合与 DP teacher 数据。
3. 数据预处理产物必须可复用、可审计、可校验，训练超参变化不应使采样与 DP 结果失效。
4. train/val/test 都应有固化的 sampled horizons 与 DP teacher 数据，保证 validation/replay 与训练一致。
5. 所有文件必须携带 `schema_hash`、`data_process_hash`、`dp_teacher_hash` 等校验信息，避免混用不同配置批次。
6. 保持现有模型、loss、DP planner、采样算法和因子输入契约不变，只调整编排边界和持久化契约。

---

## 3. 非目标

- 不修改 `SingleTradeDPPlanner` 的算法语义。
- 不修改 `StratifiedWindowSampler` 的采样策略。
- 不改变 `close` 只作为价格列、不进入模型状态的约束。
- 不改变 Phase II 的数据消费接口；Phase II 仍通过 Phase I 导出的 schema、labels、encoder/decoder/codebook 工作。
- 不在本批次重构完整 artifact manifest 体系；只为本次数据预处理拆分增加必要 manifest。

---

## 4. 新流程

### 4.1 数据预处理命令

推荐新增命令:

```bash
python scripts/phase1_data_processor.py \
  --pair AL \
  --data-batch-id al_short_20260502 \
  --train-file data/AL/df_train.feather \
  --val-file data/AL/df_val.feather \
  --test-file data/AL/df_test.feather \
  --factor-profile short \
  --factor-list-file src/factors/AL/short.txt \
  --horizon 72 \
  --num-demos 30000 \
  --sampling-strategy stratified_uniform \
  --stratification-mode hindsight_horizon \
  --diagnostic-pair-batch-id al_short_20260502_prospective \
  --max-position 10 \
  --artifact-root artifacts
```

输出目录:

```text
artifacts/{PAIR}/{DATA_BATCH_ID}/phase1/
```

数据预处理阶段负责:

1. 读取 `train_file/val_file/test_file`。
2. 加载 `src/factors/{PAIR}/{profile}.txt` 或 `--factor-list-file`。
3. 生成并写入 `input_schema.json`。
4. 对 train/val/test 枚举 window index，并写入 `window_index_{split}.feather`。
5. 按现有 `StratifiedWindowSampler` 生成 sampled horizons。
6. 构建 states/prices/execution_books。
7. 对 train/val/test sampled horizons 运行 DP teacher。
8. 写入 sampled horizon 文件、DP teacher 文件、reject stats 和 `data_process_manifest.json`。

### 4.2 模型训练命令

训练阶段推荐改为读取 manifest:

```bash
python scripts/train_phase1.py \
  --pair AL \
  --train-batch-id al_short_20260502_vq_run01 \
  --data-process-manifest artifacts/AL/al_short_20260502/phase1/data_process_manifest.json \
  --epochs 100 \
  --batch-size 256 \
  --lr 1e-3 \
  --device cuda
```

训练阶段负责:

1. 读取 `data_process_manifest.json`。
2. 校验 CLI `pair` 与 manifest `pair` 一致。
3. 读取 `input_schema.json`，并校验 `schema_hash`。
4. 读取 `sampled_horizons_{split}.feather` 和 `dp_teacher_{split}.feather`。
5. 通过 `sample_id` 将 sampled horizons 与 teacher actions/rewards 合并为 `HorizonRecord`。
6. 用 train rewards fit `RewardNormalizer`。
7. 执行模型训练、validation、checkpoint selection。
8. 用 best checkpoint 导出 `horizon_labels_{split}.feather`、Phase II artifacts 与报告。

训练主流程只消费 `scripts/phase1_data_processor.py` 已经落盘的结果；不得在 manifest 模式下 import 或调用该脚本去现算数据。

训练阶段不得重新调用:

- `SlidingWindowIndexer.enumerate`
- `StratifiedWindowSampler.sample`
- `HorizonBuilder.build`
- `Phase1DemoGenerator.generate`
- `SingleTradeDPPlanner.plan`

---

## 5. 文件契约

### 5.1 `data_process_manifest.json`

新增 manifest 是训练阶段的唯一入口。

```json
{
  "version": 1,
  "phase": "phase1_data_process",
  "pair": "AL",
  "data_batch_id": "al_short_20260502",
  "artifact_dir": "artifacts/AL/al_short_20260502/phase1",
  "created_at": "2026-05-02T00:00:00Z",
  "input_files": {
    "train": "data/AL/df_train.feather",
    "val": "data/AL/df_val.feather",
    "test": "data/AL/df_test.feather"
  },
  "input_schema_path": "input_schema.json",
  "schema_hash": "xxxxxxxxxxxxxxxx",
  "data_process_hash": "xxxxxxxxxxxxxxxx",
  "dp_teacher_hash": "xxxxxxxxxxxxxxxx",
  "feature_source": {
    "mode": "fixed_plus_factor_list",
    "pair": "AL",
    "profile": "short",
    "factor_list_path": "src/factors/AL/short.txt"
  },
  "splits": {
    "train": {
      "window_index_path": "window_index_train.feather",
      "sampled_horizons_path": "sampled_horizons_train.feather",
      "dp_teacher_path": "dp_teacher_train.feather",
      "num_horizons": 30000
    },
    "val": {
      "window_index_path": "window_index_val.feather",
      "sampled_horizons_path": "sampled_horizons_val.feather",
      "dp_teacher_path": "dp_teacher_val.feather",
      "num_horizons": 64
    },
    "test": {
      "window_index_path": "window_index_test.feather",
      "sampled_horizons_path": "sampled_horizons_test.feather",
      "dp_teacher_path": "dp_teacher_test.feather",
      "num_horizons": 64
    }
  }
}
```

路径建议保存为相对 `artifact_dir` 的相对路径，训练加载时统一 resolve，便于移动整个 artifact 目录。

### 5.2 `sampled_horizons_{split}.feather`

该文件保存分片训练集合，即被采样后的 horizon records，不包含 DP teacher 的 `actions/rewards`。

必需字段:

| 字段 | 说明 |
| --- | --- |
| `sample_id` | 与 DP teacher join 的稳定主键 |
| `pair` / `split` | 标的与 split |
| `start_index` / `end_index` | horizon 起止行 |
| `last_execution_row` / `last_markout_row` | reward alignment 审计字段 |
| `strata_label` | 分层采样标签 |
| `states` | 按 `input_schema.feature_columns` 切出的状态序列 |
| `prices` | 按 `input_schema.price_column` 切出的价格序列 |
| `execution_books` | DP/replay 所需盘口深度 JSON |
| `is_augmented` / `augmentation_type` | 增强来源审计 |
| `_schema_hash` | schema 校验 |
| `_data_process_hash` | 数据预处理配置校验 |

### 5.3 `dp_teacher_{split}.feather`

该文件保存 DP teacher 结果，与 sampled horizons 通过 `sample_id` 一对一关联。

必需字段:

| 字段 | 说明 |
| --- | --- |
| `sample_id` | join 主键 |
| `actions` | DP teacher action 序列 |
| `rewards` | DP teacher reward 序列 |
| `teacher_return` | 原始 reward 求和 |
| `num_switches` | action 切换次数 |
| `is_no_trade` | 是否全程 no-trade |
| `reject_transition_count` | 当前 horizon 被拒绝的转移数 |
| `reject_transition_rate` | 当前 horizon reject rate |
| `_schema_hash` | schema 校验 |
| `_data_process_hash` | 数据预处理配置校验 |
| `_dp_teacher_hash` | DP/cost 配置校验 |

训练加载时必须校验:

- sampled horizons 与 DP teacher 的 `sample_id` 集合完全一致。
- 每个 `actions/rewards` 长度与 horizon 长度一致。
- 三个 hash 与 manifest 一致。
- train/val/test split 不可互相混用。

### 5.4 兼容产物

当前代码已有 `demos_train.feather` 合并格式。迁移期可以保留:

```text
demos_train.feather
demos_val.feather
demos_test.feather
```

但它们应视为兼容导出，不作为新训练入口的主契约。新的主契约是:

```text
sampled_horizons_{split}.feather + dp_teacher_{split}.feather + data_process_manifest.json
```

---

## 6. 配置拆分

建议新增 `Phase1DataProcessConfig`，把会影响采样和 DP teacher 的字段从训练配置中独立出来:

```python
@dataclass(frozen=True)
class Phase1DataProcessConfig:
    pair: str
    data_batch_id: str
    train_file: str
    val_file: str
    test_file: str
    artifact_root: str = "artifacts"
    factor_profile: str = "short"
    factor_list_file: Optional[str] = None
    horizon: int = 72
    num_demos: int = 30000
    sampling_strategy: str = "stratified_uniform"
    stratification: StratificationConfig = field(default_factory=StratificationConfig)
    sampling_health: SamplingHealthConfig = field(default_factory=SamplingHealthConfig)
    data_augmentation: DataAugmentationConfig = field(default_factory=DataAugmentationConfig)
    dp: DPConfig = field(default_factory=DPConfig)
```

`Phase1Config` 保留模型训练字段，并新增:

```python
data_process_manifest: Optional[str] = None
```

迁移期允许 `Phase1Config` 继续包含 `train_file/val_file/test_file`，但当 `data_process_manifest` 存在时，训练必须忽略这些原始文件路径，避免重新读取 raw market data。

Hash 拆分规则:

| Hash | 包含内容 | 用途 |
| --- | --- | --- |
| `schema_hash` | `input_schema.json` canonical payload | 防止 states 维度/字段混用 |
| `data_process_hash` | 输入文件签名、采样配置、horizon、seed、factor source、augmentation 配置 | 防止 sampled horizons 混用 |
| `dp_teacher_hash` | `data_process_hash` + DPConfig + CostConfig + reward alignment | 防止 teacher actions/rewards 混用 |
| `training_config_hash` | 模型、loss、optimizer、selection policy、training seed 等 | checkpoint/report 审计 |

训练超参变化只应改变 `training_config_hash`，不应改变 `data_process_hash` 或 `dp_teacher_hash`。

---

## 7. 代码改造方案

### 7.1 新增数据处理入口

新增:

```text
scripts/phase1_data_processor.py
```

`scripts/phase1_data_processor.py` 是本批次的数据预处理脚本入口，负责 CLI parse、配置构建和调用数据处理编排。当前不新增 trainer 层 processor 文件；如后续需要复用，再把脚本内稳定逻辑下沉到 `src/`。

脚本内的 `Phase1DataProcessor.run()` 或等价编排函数从当前 `Phase1Trainer.run()` 中抽出以下步骤:

1. `_seed_everything`
2. `_check_prospective_diagnostic`
3. schema validation
4. `_build_horizons_for_split`
5. temporal contrastive augmentation
6. `_generate_demos`
7. sampled horizon / DP teacher / manifest 写入

### 7.2 新增 processed data store

新增或扩展:

```text
src/data/phase1_processed_store.py
```

职责:

- `save_sampled_horizons(split, records)`
- `save_dp_teacher(split, records, reject_stats)`
- `load_records(manifest, split)`
- `validate_manifest(manifest)`
- `join_horizons_with_teacher(sampled, teacher)`

`Phase1DemoStore` 可保留，用于现有 `demos_train.feather` 兼容读写和 label 导出。

### 7.3 改造训练入口

`scripts/train_phase1.py` 新增:

```text
--data-process-manifest artifacts/AL/al_short_20260502/phase1/data_process_manifest.json
```

当该参数存在时:

1. `build_config()` 写入 `Phase1Config.data_process_manifest`。
2. `Phase1Trainer.run()` 直接从 processed store 加载 train/val/test records。
3. 跳过 raw market reader、schema 推导、window sampling、DP teacher 生成。
4. 继续执行 RewardNormalizer、训练、validation、labels、checkpoint、report。
5. 不依赖 `scripts/phase1_data_processor.py` 的运行时逻辑，只依赖 manifest 中声明的固化产物。

迁移期可保留旧路径:

- 未传 `--data-process-manifest` 时走当前一体化流程。
- 日志打印 deprecation warning。
- 待新流程稳定后，再将 manifest 改为必填。

---

## 8. 验收标准

1. `phase1_data_processor.py` 单独运行后，输出 manifest、schema、window index、sampled horizons、DP teacher 文件。
2. `train_phase1.py --data-process-manifest ...` 可在不传 `--train-file/--val-file/--test-file` 的情况下完成训练。
3. manifest 训练路径下，单测 monkeypatch `StratifiedWindowSampler.sample` 与 `Phase1DemoGenerator.generate` 抛错，训练仍可正常加载数据并进入训练组件初始化。
4. 删除任意 `sampled_horizons_{split}.feather` 或 `dp_teacher_{split}.feather` 时，训练 fail-fast，错误信息包含缺失路径。
5. 修改 `dp_teacher_{split}.feather` 中的 `_dp_teacher_hash` 后，训练 fail-fast。
6. sampled horizons 与 DP teacher 的 `sample_id` 不一致时，训练 fail-fast。
7. 同一数据预处理配置、同一 seed 重跑，`sample_id/window_start/actions/rewards` 稳定一致。
8. 改变训练超参后，只生成新的训练产物，不重新生成 sampled horizons 或 DP teacher。
9. `phase1_report.json` 记录 `data_process_manifest`、`schema_hash`、`data_process_hash`、`dp_teacher_hash`。
10. `input_schema.json` 继续满足 `"close" not in feature_columns`。

---

## 9. 测试计划

### 9.1 单元测试

新增:

- `tests/unit/data/test_phase1_processed_store.py`
  - 保存并读取 sampled horizons。
  - 保存并读取 DP teacher。
  - join 后恢复完整 `HorizonRecord.actions/rewards`。
  - hash 不一致时报错。
  - `sample_id` 缺失或多余时报错。

- `tests/unit/scripts/test_phase1_data_processor.py`
  - data processor 写出 manifest 和三个 split 文件。
  - data processor 记录 reject stats。
  - 相同 seed 产物稳定。

更新:

- `tests/unit/trainers/test_phase1_trainer.py`
  - manifest 模式下 trainer 不调用采样和 DP。
  - manifest 模式下 trainer 从文件加载 train/val/test records。
  - manifest 缺文件或 hash 不一致时 fail-fast。

### 9.2 集成测试

新增:

```bash
pytest tests/integration/test_phase1_data_process_then_train.py -q
```

测试流程:

1. 使用 fixture market data 运行 `scripts/phase1_data_processor.py`。
2. 使用生成的 `data_process_manifest.json` 运行 `scripts/train_phase1.py` smoke。
3. 断言输出 checkpoint、labels、report。
4. 断言训练日志中没有 raw sampling / DP generation 的步骤。

---

## 10. 迁移批次

| 批次 | 事项 | 优先级 |
| --- | --- | --- |
| A1 | 新增 `Phase1DataProcessConfig` 与 `scripts/phase1_data_processor.py` | P0 |
| A2 | 在脚本入口内抽出 `Phase1DataProcessor` 或等价编排函数，复用现有 schema、sampler、horizon builder、DP generator | P0 |
| A3 | 新增 `phase1_processed_store` 和 manifest/hash 校验 | P0 |
| A4 | `train_phase1.py` 支持 `--data-process-manifest` 并跳过采样/DP | P0 |
| A5 | 将 val/test DP teacher 也持久化，训练从文件读取 validation records | P0 |
| A6 | 更新 report/config，记录 manifest 与三类 hash | P1 |
| A7 | 保留旧一体化路径并打印 deprecation warning | P1 |
| A8 | 稳定后移除旧一体化路径或将其降为内部 debug 模式 | P2 |

---

## 11. 风险与处理

| 风险 | 影响 | 处理 |
| --- | --- | --- |
| sampled horizons 与 teacher 文件错配 | 训练标签污染 | 使用 `sample_id` 全量集合校验和 hash 校验，错配直接失败 |
| 训练仍隐式读取 raw market 文件 | 破坏“训练只读固化数据”目标 | manifest 模式下禁止调用 `MarketFileReader.read_split` |
| val/test teacher 未持久化 | validation 仍依赖在线 DP | data process 必须生成 `dp_teacher_val/test.feather` |
| 训练配置 hash 与数据 hash 混在一起 | 改训练超参导致数据缓存失效 | 拆分 `data_process_hash`、`dp_teacher_hash`、`training_config_hash` |
| 旧 `demos_train.feather` 使用方断裂 | 影响既有测试和下游 | 迁移期保留兼容导出，但新训练入口以 manifest 为准 |
| 文件体积增加 | artifact 占用上升 | 先保留 feather；后续可增加 tensor cache 或分 shard 写入 |

---

## 12. 最终状态

变更完成后，Phase I 的责任边界应变为:

```text
raw market data + factor list
        |
        v
scripts/phase1_data_processor.py
        |
        +-- input_schema.json
        +-- window_index_{train,val,test}.feather
        +-- sampled_horizons_{train,val,test}.feather
        +-- dp_teacher_{train,val,test}.feather
        +-- data_process_manifest.json
        |
        v
scripts/train_phase1.py --data-process-manifest ...
        |
        +-- reward_normalizer.json
        +-- best_vq_model.pt / last_vq_model.pt
        +-- horizon_labels_{train,val,test}.feather
        +-- encoder.pt / decoder.pt / codebook.pt
        +-- phase1_report.json
```

训练入口从“生成数据再训练”变为“读取已审计数据再训练”。这使 Phase I 可以复用同一批分片采样与 DP teacher 数据，对不同模型训练超参做可比实验。

---

## 13. 执行计划采纳回写

**执行计划**: `docs/changes/20260502_phase1_data_process_split_execution_plan.md`

本节记录从设计到执行计划的取舍结果。执行批次不机械采纳所有设计建议；对结构性风险较高、收益不足或不属于本次拆分目标的内容，明确不纳入本批次。

### 13.1 采纳项

| 设计项 | 决策 | 原因 |
| --- | --- | --- |
| 新增 `scripts/phase1_data_processor.py` | 采纳 | 满足“数据预处理独立运行”的核心目标，且入口层改动可控 |
| 在 `scripts/phase1_data_processor.py` 中抽出 `Phase1DataProcessor` 或等价编排函数 | 采纳 | 将 schema、采样、horizon 构建、DP teacher 生成从 trainer 训练循环中拆出，同时复用现有组件 |
| `sampled_horizons_{split}.feather` | 采纳 | 固化分层采样后的 horizon 集合，训练阶段不再重新采样 |
| `dp_teacher_{split}.feather` | 采纳 | 将 DP teacher actions/rewards 作为独立文件保存，便于审计和 hash 校验 |
| `data_process_manifest.json` | 采纳 | 作为 manifest 模式训练的数据入口，集中记录 split 文件、schema 和 hash |
| train/val/test 都生成 DP teacher | 采纳 | validation、labels 和 teacher replay 都需要固化的 teacher 数据 |
| `train_phase1.py --data-process-manifest` | 采纳 | 训练可以从文件读取分片训练集合和 DP teacher 数据 |
| manifest 模式跳过采样与 DP | 采纳 | 防止训练超参变化导致数据预处理结果被重算，保证实验可比 |
| 单元测试和集成 smoke | 采纳 | 新文件契约和新编排路径必须有测试防回归 |

### 13.2 部分采纳项

| 设计项 | 决策 | 采纳边界 |
| --- | --- | --- |
| `Phase1DataProcessConfig` | 部分采纳 | 只新增轻量配置结构或 builder，复用现有 `StratificationConfig`、`SamplingHealthConfig`、`DPConfig`；不重写整个 Phase I 配置体系 |
| hash 拆分 | 部分采纳 | 本批次落地 `schema_hash`、`data_process_hash`、`dp_teacher_hash`；训练配置继续复用现有 `config_hash` |
| manifest 是训练入口 | 部分采纳 | manifest 模式下是唯一数据入口；迁移期不强制所有训练都必须使用 manifest |
| 兼容 `demos_train.feather` | 部分采纳 | 可保留 `demos_train.feather` 兼容旧测试和下游；不新增 `demos_val.feather` / `demos_test.feather` 兼容文件 |

### 13.3 不采纳或暂缓项

| 设计项 | 决策 | 不采纳原因 | 后续处理 |
| --- | --- | --- | --- |
| 立即移除旧一体化训练路径 | 不采纳 | 属于高风险结构性改动，会同时影响现有 smoke、旧实验和本地调试入口 | 先保留 legacy inline 路径，manifest 模式稳定后再评估 |
| 将 manifest 改为所有训练的唯一强制入口 | 暂缓 | 新 data process 产物、文档和 CI 覆盖未稳定前，强制切换会阻塞现有实验 | 本批次只新增可选 `--data-process-manifest` |
| 完整 artifact manifest/cache 体系重构 | 不采纳 | 超出“拆分采样与 DP 生成”的必要范围，容易牵动 Phase II/III 产物管理 | 本批次只做 Phase I data process manifest |
| 大规模拆分 `Phase1Config` 与新增独立 `training_config_hash` 体系 | 暂缓 | 配置体系被 CLI、报告和测试广泛引用，贸然重构回归面过大 | 继续使用现有 `config_hash` 作为训练配置审计 hash |
| 生成 `demos_val.feather` 和 `demos_test.feather` | 不采纳 | 新主契约已经是 sampled horizons + DP teacher；额外兼容文件增加存储和混用风险 | 只保留必要的 `demos_train.feather` 兼容 |
| 输入 feather 全文件 sha256 | 暂缓 | 生产数据文件可能较大，全量 hash 会增加预处理启动成本 | manifest 先记录路径、size、mtime_ns；需要强审计时另加 `--hash-input-files` |
| tensor cache / shard 存储层 | 不采纳 | 属于性能专项，不是本次数据预处理拆分的必要条件 | 后续根据实际瓶颈单独设计 |
| `scripts/train_phase1.py` 重命名为 `scripts/phase1_train.py` | 不采纳 | 当前仓库、测试和运行命令都使用 `train_phase1.py`，重命名收益低且破坏现有引用 | 如需兼容旧称谓，后续只新增薄 wrapper |

### 13.4 测试要求

执行计划要求所有采纳变更包含对应测试:

- processed store 单元测试: `tests/unit/data/test_phase1_processed_store.py`
- data processor 单元测试: `tests/unit/scripts/test_phase1_data_processor.py`
- trainer manifest 模式测试: `tests/unit/trainers/test_phase1_trainer.py`
- CLI 条件必填测试: `tests/unit/scripts/test_train_phase1_cli.py`
- report 审计字段测试: `tests/unit/evaluation/test_phase1_report.py`
- 两阶段 smoke: `tests/integration/test_phase1_data_process_then_train.py`

每个代码批次完成时必须执行执行计划中对应的 pytest 命令；最终验收必须同时覆盖新 manifest 流程和旧 legacy inline 流程。

---

## 14. 实施结果回写

**实施日期**: 2026-05-02
**对应执行计划**: `docs/changes/20260502_phase1_data_process_split_execution_plan.md`

本次实现已落地两阶段 Phase I 流程:

1. `scripts/phase1_data_processor.py` 先离线生成 `window_index_{split}.feather`、`sampled_horizons_{split}.feather`、`dp_teacher_{split}.feather`、`reject_stats_{split}.json`、`input_schema.json` 和 `data_process_manifest.json`。
2. `scripts/train_phase1.py --data-process-manifest ...` 在 manifest 模式下只加载 data processor 产物，不重新读取 raw split、不重新采样、不重新构建 horizon、不重新运行 DP teacher。
3. `phase1_report.json` 增加 `processed_data_mode`、`data_process_manifest`、`data_batch_id`、`schema_hash`、`data_process_hash`、`dp_teacher_hash`，用于回溯离线数据处理批次。
4. 旧 legacy inline 路径保留，并由原有 Phase I smoke 继续覆盖。

已在 `conda activate ArchetypeTrade` 环境执行并通过:

```bash
pytest tests/unit/data/test_phase1_processed_store.py tests/unit/data/test_demo_store.py -q
pytest tests/unit/scripts/test_phase1_data_processor.py tests/unit/data/test_phase1_processed_store.py -q
pytest tests/unit/trainers/test_phase1_trainer.py tests/unit/scripts/test_train_phase1_cli.py -q
pytest tests/unit/evaluation/test_phase1_report.py tests/unit/trainers/test_phase1_trainer.py tests/unit/config/test_phase1_config_docs.py -q
pytest tests/integration/test_phase1_data_process_then_train.py tests/integration/test_phase1_pipeline_smoke.py -q
```

测试结果分别为 `10 passed`、`12 passed`、`18 passed`、`23 passed`、`5 passed`。`git diff --check` 通过。

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
