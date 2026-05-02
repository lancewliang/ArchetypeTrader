# Phase I 因子配置需求变更技术设计与执行计划

## 1. 变更日志

| 日期 | 变更项 | 说明 |
| --- | --- | --- |
| 2026-05-02 | Phase I 因子输入契约变更 | 第一阶段不再默认把所有数值列自动作为模型状态输入，改为按交易标的读取固定字段 + 标的级因子清单。 |

## 2. 背景与目标

当前 Phase I 的输入 schema 由 `InputSchemaValidator` 自动推导: 除 `timestamp/symbol/split/sample_id/close` 外，所有数值列都会进入 `feature_columns`。这会让不同标的之间的输入维度和字段含义不可控，也无法明确复现实验所用因子清单。

本次变更目标:

1. 支持不同交易标的使用不同因子清单。
2. AL 标的使用 `src/factors/AL/short.txt` 作为短周期因子清单。
3. 固定字段由系统配置统一纳入，其中 `close` 只作为价格列，不进入模型输入状态。
4. Phase I 训练数据切换为已经准备好的 AL 文件:

```text
data/AL/df_train.feather
data/AL/df_val.feather
data/AL/df_test.feather
```

5. AL 最大交易量配置为 `10`。在当前代码语义下建议落地为 `max_position=10`，即 action `0/1/2` 对应目标仓位 `-10/0/+10`。如果业务语义要求“单笔换仓量不得超过 10”，需额外引入 `max_order_volume`，不能只依赖现有 `max_position` 字段。

## 3. 输入字段契约

### 3.1 固定字段

所有标的共享以下固定字段:

```python
FIXED_FEATURES = [
    "close",
    "ask1_price", "ask1_size", "bid1_price", "bid1_size",
    "ask2_price", "ask2_size", "bid2_price", "bid2_size",
    "ask3_price", "ask3_size", "bid3_price", "bid3_size",
    "ask4_price", "ask4_size", "bid4_price", "bid4_size",
    "ask5_price", "ask5_size", "bid5_price", "bid5_size",
    "total_trade_volume", "turnover", "open_interest"
]
```

字段语义:

| 字段类别 | 用途 | 是否进入模型状态 |
| --- | --- | --- |
| `close` | 价格列，用于 reward、DP、replay、指标、mark price fallback | 否 |
| 五档盘口价格与数量 | 模型状态输入，同时供成本模型切 `execution_books` | 是 |
| `total_trade_volume/turnover/open_interest` | 模型状态输入 | 是 |

`close` 必须在 `input_schema.json` 中记录为 `price_column="close"`，并且必须满足 `close not in feature_columns`。这是一条阻塞式验收规则，不能通过配置绕过。

### 3.2 标的级因子清单

每个标的可以配置独立因子文件:

```text
src/factors/{PAIR}/{profile}.txt
```

AL 第一阶段使用:

```text
src/factors/AL/short.txt
```

因子文件格式:

- 每行一个字段名。
- 允许空行和前后空白。
- 建议支持 `#` 注释行。
- 建议兼容字段名外层单引号或双引号，但写入 `input_schema.json` 时统一保存裸字段名。
- 文件中不得包含精确字段 `close`；如果出现，应直接报错并提示 `close` 只能作为价格列使用。
- 允许包含 `close_price_ratio_60`、`close_price_zscore_30` 等由历史 `close` 派生且已在数据文件中准备好的因子，但必须由后续 `feature_provenance.json` 证明不使用未来行。

### 3.3 最终模型输入列

Phase I 的最终状态列按以下顺序确定:

```text
feature_columns =
  FIXED_FEATURES 去掉 close
  + src/factors/{PAIR}/{profile}.txt 中列出的标的级因子
  - 去重后的重复字段
```

排序规则必须稳定:

1. 固定字段按 `FIXED_FEATURES` 原始顺序。
2. 标的级因子按 txt 文件顺序。
3. 若标的级因子与固定字段重复，只保留固定字段位置，并在 schema 诊断中记录 `deduplicated_features`。

输入数据必须同时满足:

- train/val/test 三个 split 都包含全部 `feature_columns` 和 `close`。
- 所有 `feature_columns` 都是数值列，可转为 `float32`。
- 任意 `feature_columns` 不得包含 null、NaN、Inf。
- `close > 0` 且无 null、NaN、Inf。
- train/val/test 的 `feature_columns` 必须完全一致，不能各自自动推导。

## 4. 代码改造方案

### 4.1 新增因子清单加载模块

建议新增:

```text
src/data/feature_registry.py
```

职责:

- 定义 `FIXED_FEATURES`。
- 加载 `src/factors/{PAIR}/{profile}.txt`。
- 解析空行、注释、引号、重复字段。
- 产出 `FeatureSelectionSpec`:

```python
@dataclass(frozen=True)
class FeatureSelectionSpec:
    pair: str
    profile: str
    factor_list_path: str
    fixed_features: list[str]
    configured_factors: list[str]
    feature_columns: list[str]
    price_column: str = "close"
    deduplicated_features: list[str] = field(default_factory=list)
```

硬约束:

- `close` 只能存在于 `fixed_features` 中作为 `price_column`，不得进入 `feature_columns`。
- 因子文件不存在时正式训练直接失败；如需兼容旧实验，只允许通过显式 `--allow-auto-feature-discovery` 走旧逻辑。

### 4.2 扩展 Phase I 配置与 CLI

建议在 `Phase1Config` 增加:

```python
factor_profile: str = "short"
factor_list_file: Optional[str] = None
allow_auto_feature_discovery: bool = False
max_trade_volume: int = 10
```

同时在 CLI 增加:

```text
--factor-profile short
--factor-list-file src/factors/AL/short.txt
--allow-auto-feature-discovery
--max-position 10
```

`max_trade_volume=10` 的落点:

- 当前 `TradingEnv` 已有 `max_position`，`DPConfig` 也有 `max_position`，因此本次可先把 AL 的最大交易量落到 `DPConfig.max_position=10` 与 `TradingEnv(max_position=10)`。
- `phase1_config.yaml` 和 `phase1_report.json` 必须记录 `max_trade_volume=10` 与实际传入 env 的 `max_position=10`。
- 若后续要限制单笔成交量，新增 `max_order_volume` 并在 `LobDepthCostModel.execute()` 前校验 `abs(target_position - prev_position) <= max_order_volume`。

### 4.3 改造 InputSchemaValidator

当前 `InputSchemaValidator.validate(frame)` 会自动扫描所有数值列。改造后应支持显式特征列:

```python
InputSchemaValidator(
    timestamp_column="timestamp",
    price_column="close",
    feature_columns=feature_spec.feature_columns,
)
```

新增行为:

- 如果传入显式 `feature_columns`，只校验这些列，不再自动纳入其他数值列。
- 额外数值列允许存在，但默认不进入模型状态。
- 校验 train 后生成唯一 schema；val/test 必须调用 `validate_against_schema(frame, schema)`，确保 split 间字段一致。
- `input_schema.json` 增加审计字段:

```json
{
  "timestamp_column": "timestamp",
  "price_column": "close",
  "feature_columns": ["ask1_price", "..."],
  "excluded_columns": ["timestamp", "symbol", "split", "sample_id", "close"],
  "feature_source": {
    "mode": "fixed_plus_factor_list",
    "pair": "AL",
    "profile": "short",
    "factor_list_path": "src/factors/AL/short.txt",
    "fixed_features": ["close", "ask1_price", "..."],
    "configured_factors": ["ask_gap_3_4", "..."],
    "deduplicated_features": []
  }
}
```

### 4.4 训练入口使用新数据路径

AL Phase I 推荐命令:

```bash
python scripts/train_phase1.py \
  --pair AL \
  --train-batch-id al_short_20260502 \
  --train-file data/AL/df_train.feather \
  --val-file data/AL/df_val.feather \
  --test-file data/AL/df_test.feather \
  --factor-profile short \
  --factor-list-file src/factors/AL/short.txt \
  --max-position 10
```

可选兼容增强:

- 当用户传 `--pair AL --factor-profile short` 但未传 `--factor-list-file` 时，默认解析 `src/factors/AL/short.txt`。
- 当用户未传 `--train-file/--val-file/--test-file` 时，可默认查找 `data/{PAIR}/df_train.feather`、`data/{PAIR}/df_val.feather`、`data/{PAIR}/df_test.feather`；旧命名 `train.feather/val.feather/test.feather` 只作为 fallback，并在日志中明确记录。

### 4.5 下游模块影响

| 模块 | 改动 |
| --- | --- |
| `src/data/horizon_builder.py` | 无需大改，继续从 `schema.feature_columns` 切 `states`；确认 `close` 只从 `price_column` 切 `prices`。 |
| `src/data/dataset.py` | 无需改动，`feature_dim` 会随 schema 变化。 |
| `src/models/vq_archetype.py` | 无需改动，模型按 `schema.feature_dim()` 初始化。 |
| `src/trainers/phase1_trainer.py` | 在读取数据后先加载 `FeatureSelectionSpec`，再把显式 `feature_columns` 传给 schema validator。构造 env 时传入 `max_position=10`。 |
| `src/data/phase2_dataset.py` | 无需大改，Phase II 继续读取 Phase I 导出的 `input_schema.json`。 |
| `artifacts/{PAIR}/{BATCH_ID}/phase1/input_schema.json` | schema hash 会变化，旧 cache 必须失效重算。 |

## 5. 测试计划

### 5.1 单元测试

新增或更新测试:

1. `test_feature_registry_loads_pair_factor_list`
   - 输入 `src/factors/AL/short.txt`。
   - 验证返回字段顺序稳定，空白被清理。

2. `test_feature_registry_rejects_close_in_factor_file`
   - 构造包含 `close` 的临时因子文件。
   - 期望直接报错。

3. `test_schema_uses_explicit_feature_columns_only`
   - DataFrame 中额外放一个数值列 `unused_numeric_feature`。
   - 验证该列不进入 `feature_columns`。

4. `test_schema_requires_all_configured_features`
   - 缺少 `src/factors/AL/short.txt` 中任一字段时失败。

5. `test_close_price_column_not_model_state`
   - 验证 `input_schema.price_column == "close"` 且 `"close" not in feature_columns`。

6. `test_phase1_max_position_from_cli`
   - CLI 传 `--max-position 10`。
   - 验证 `TradingEnv.max_position == 10`，action 映射为 `-10/0/+10`。

### 5.2 集成测试

新增 AL schema dry-run:

```bash
python scripts/train_phase1.py \
  --pair AL \
  --train-batch-id al_short_schema_smoke \
  --train-file data/AL/df_train.feather \
  --val-file data/AL/df_val.feather \
  --test-file data/AL/df_test.feather \
  --factor-profile short \
  --factor-list-file src/factors/AL/short.txt \
  --max-position 10 \
  --epochs 1 \
  --num-demos 128 \
  --device cpu \
  --allow-missing-prospective-diagnostic \
  --risk-acknowledged-by local_smoke \
  --expected-sign-off-followup-batch-id al_short_full
```

验收:

- 产出 `artifacts/AL/al_short_schema_smoke/phase1/input_schema.json`。
- `feature_columns` 等于固定字段去掉 `close` 后拼接 AL short 因子。
- `close` 不出现在 `states`、`feature_columns`、模型输入维度中。
- `phase1_config.yaml` 记录 AL 文件路径、factor list 路径、`max_position=10`。

## 6. 执行步骤

| 顺序 | 任务 | 产出 |
| --- | --- | --- |
| 1 | 新增 `src/data/feature_registry.py` | 固定字段、因子文件解析、`FeatureSelectionSpec` |
| 2 | 扩展 `Phase1Config` 与 `scripts/train_phase1.py` | `--factor-profile`、`--factor-list-file`、`--max-position` |
| 3 | 改造 `InputSchemaValidator` | 支持显式特征列和 split 间 schema 一致性校验 |
| 4 | 接入 `Phase1Trainer` | 训练前加载 AL 因子配置，构造 env 时应用 `max_position=10` |
| 5 | 更新 tests | 覆盖因子清单、`close` 隔离、最大交易量配置 |
| 6 | 跑 smoke | 生成 AL schema 与最小训练产物 |
| 7 | 更新执行文档或 README | 记录 AL 正式训练命令 |

## 7. 验收标准

本变更完成后必须满足:

- `data/AL/df_train.feather`、`data/AL/df_val.feather`、`data/AL/df_test.feather` 可被训练入口读取。
- `src/factors/AL/short.txt` 是 AL short profile 的唯一因子清单来源。
- `input_schema.json.feature_columns` 不再由“所有数值列”自动推导，而由固定字段 + AL 因子清单确定。
- `close` 只作为价格列进入 `prices/reward/DP/replay/metrics`，不作为模型状态输入。
- train/val/test 的 schema 校验使用同一份 feature list；任一 split 缺列、类型错误、NaN/Inf 都失败。
- AL 最大交易量 `10` 在配置、env、报告中可追溯。
- 旧 cache 因 schema hash 变化自动失效，避免复用旧 `demos_train.feather`。

## 8. 风险与待确认

| 风险 | 影响 | 处理 |
| --- | --- | --- |
| “最大交易量 10” 语义可能是最大仓位，也可能是单笔下单量 | 若按 `max_position=10`，从 short 到 long 的换仓 delta 可能为 20 | 本次按现有代码落到 `max_position=10`；如需单笔上限，追加 `max_order_volume` 设计。 |
| 因子文件含未来函数派生字段 | 训练和验证指标可能泄漏未来信息 | 后续必须为 `feature_provenance.json` 补齐每个因子的可见时间和 lookback 范围。 |
| 固定字段缺失或盘口深度不足 | DP/replay 可能大量 reject | schema 阶段先硬校验固定字段存在；训练报告继续监控 reject rate。 |
| 不同 profile 输入维度不同 | 旧模型 checkpoint 不可复用 | `input_schema_hash` 必须进入 cache 和 artifact manifest。 |

## 9. 采纳范围与不采纳说明

本节回写 `docs/changes/20260502_phase1_factor_change_execution_plan.md` 的执行裁剪结论。若设计文本中较早章节出现“建议”项，与本节冲突时，以本节和 changes 执行计划为准。

### 9.1 本批次采纳

| 设计项 | 采纳结论 | 说明 |
| --- | --- | --- |
| 每标的独立因子清单 | 采纳 | AL short profile 使用 `src/factors/AL/short.txt`。 |
| 固定字段 + 标的级因子拼接 | 采纳 | `FIXED_FEATURES` 中除 `close` 外的字段进入模型状态，再按 txt 顺序追加因子。 |
| `close` 价格列隔离 | 采纳 | `close` 只进入 reward、DP、replay、指标和 mark price fallback，不进入 `feature_columns`。 |
| 显式 `feature_columns` 校验 | 采纳 | `InputSchemaValidator` 需要支持显式字段列表，正式 AL 路径不再自动纳入所有数值列。 |
| `input_schema.json` 记录因子来源 | 采纳 | 增加 `feature_source` 审计信息，记录 pair/profile/factor list/fixed/configured/deduplicated。 |
| AL 最大交易量 `10` | 部分采纳 | 本批次按现有代码语义落到 `DPConfig.max_position=10` 和 `TradingEnv(max_position=10)`。 |
| 单元测试与 smoke 测试 | 采纳 | 变更执行计划要求补 `feature_registry`、schema、trainer、max position 相关测试并执行。 |

### 9.2 本批次不采纳或暂缓

| 设计项 | 结论 | 为什么不采纳 | 后续处理 |
| --- | --- | --- | --- |
| 新增 `max_trade_volume` 顶层配置字段 | 不采纳 | 现有系统已经通过 `DPConfig.max_position` 和 `TradingEnv.max_position` 表达最大仓位；再新增字段会产生双重配置、报告不一致和 cache hash 语义不清的问题。 | 使用 CLI `--max-position 10`，并在 `phase1_config.yaml` 中记录 `dp.max_position=10`。 |
| 新增 `max_order_volume` 单笔换仓约束 | 暂缓 | 单笔成交上限会改变 `LobDepthCostModel.execute()` 与 action/position 语义，属于交易执行模型扩展，不应混入因子 schema 变更。 | 单独设计“最大仓位 vs 单笔最大换仓量”的交易约束批次。 |
| 自动 fallback 到 `data/{PAIR}/df_train.feather`、旧命名 fallback | 不采纳 | 训练数据路径静默推导会增加误读文件风险；正式训练应显式传入三个 split 路径，便于审计和复现。 | AL 命令继续显式传 `--train-file/--val-file/--test-file`。 |
| 正式新增 `--allow-auto-feature-discovery` 开关 | 不采纳 | 本次需求是收紧输入契约；正式入口继续允许自动发现会削弱“每标的因子清单”的约束。 | legacy 自动数值列推导只可保留为内部兼容或单测 fallback，不作为 AL 正式训练入口。 |
| 本批次生产完整 `feature_provenance.json` | 暂缓 | provenance 需要外部因子工程链路提供每个因子的 lookback、可见时间、fit scope；当前仓库无法从 feather 字段名可靠反推。 | 本批次只在 `input_schema.json` 记录因子来源；完整 provenance 作为独立 sign-off 批次。 |
| 修改 Phase II dataset / selector | 不采纳 | Phase II 通过 Phase I 导出的 `input_schema.json` 间接受益，本批次不需要改 selector 或 Phase II 数据结构。 | 重新训练 Phase I 后，Phase II 自然读取新 schema。 |
| 重构 artifact manifest / cache 体系 | 不采纳 | 结构性风险高，且当前 `input_schema_hash` 已可让旧 `demos_train.feather` cache 失效。 | 保持现有 hash/cache 机制，后续有 manifest 专项需求时再改。 |

## 10. 执行结果记录

**执行日期**: 2026-05-02
**执行环境**: `conda activate ArchetypeTrade`
**执行计划**: `docs/changes/20260502_phase1_factor_change_execution_plan.md`

### 10.1 完成项

| 项目 | 完成标记 | 结果 |
| --- | --- | --- |
| 因子清单加载器 | 【✅】 | 已新增 `src/data/feature_registry.py`，支持 `FIXED_FEATURES`、`src/factors/{PAIR}/{profile}.txt`、`close` 拒绝、重复字段记录和审计 dict。 |
| 显式 schema 校验 | 【✅】 | `InputSchemaValidator` 已支持显式 `feature_columns`，额外数值列不再自动进入模型输入；val/test 使用 train schema 校验。 |
| Phase I 接入 | 【✅】 | `Phase1Config`、`scripts/train_phase1.py`、`Phase1Trainer` 已接入 `factor_profile/factor_list_file`。 |
| AL 最大交易量 | 【✅】 | 已按采纳结论落到 `--max-position 10` -> `DPConfig.max_position=10` -> `TradingEnv(max_position=10)`。 |
| 测试修复 | 【✅】 | 已补充因子清单、schema、trainer/CLI、env、DP 和 Phase I smoke 测试。 |
| `.gitignore` 修正 | 【✅】 | 已将 `data` 收窄为 `/data/`，避免新增 `src/data/*` 和 `tests/unit/data/*` 被误忽略。 |

### 10.2 测试结果

已执行:

```bash
pytest tests/unit/data/test_feature_registry.py tests/unit/data/test_schema.py tests/unit/trainers/test_phase1_trainer.py tests/unit/trading/test_env.py tests/unit/planners/test_single_trade_dp.py -q
```

结果: `44 passed in 3.23s`

```bash
pytest tests/integration/test_phase1_pipeline_smoke.py -q
```

结果: `4 passed in 4.90s`

```bash
pytest tests/unit/data/test_feature_registry.py tests/unit/data/test_schema.py tests/unit/trainers/test_phase1_trainer.py tests/integration/test_phase1_pipeline_smoke.py -q
```

结果: `31 passed in 4.47s`

```bash
pytest tests/integration/test_phase1_reproducibility.py tests/integration/test_phase1_next_row_alignment.py -q
```

结果: `4 passed in 0.28s`

### 10.3 AL 数据验证

真实 AL 数据字段预检结果:

- `data/AL/df_train.feather`: 缺失字段数 `0`，`feature_dim=53`
- `data/AL/df_val.feather`: 缺失字段数 `0`，`feature_dim=53`
- `data/AL/df_test.feather`: 缺失字段数 `0`，`feature_dim=53`

AL 小样本 dry-run 已执行到采样健康检查，随后因 `flat_low_vol_sample_ratio=0.188 > max=0.15` 停止。该失败与本次因子 schema 变更无关；schema 阶段已经通过，且未发现 AL 因子缺列或 `close` 进入模型状态的问题。
