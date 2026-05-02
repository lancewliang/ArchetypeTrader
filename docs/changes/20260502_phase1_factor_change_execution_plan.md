# Phase I 因子配置变更执行计划

**日期**: 2026-05-02
**来源设计**: `docs/design/20260502_phase1_factor_change_log_design.md`
**影响阶段**: Phase I Archetype Discovery
**目标标的**: AL

---

## 1. 执行目标

本计划把 Phase I 因子配置设计转换为可落地的代码变更批次。执行目标是让 AL 训练显式使用固定字段 + `src/factors/AL/short.txt` 的因子清单，避免当前“所有数值列自动进入模型状态”的不可控行为。

本次执行坚持三个边界:

- 只改 Phase I 输入 schema、训练入口和相关测试，不牵动 Phase II。
- `close` 继续只作为价格列，用于 reward、DP、replay 和指标，不进入模型输入状态。
- 高风险结构性改动不纳入本批次，例如单笔下单量模型、数据路径隐式 fallback、artifact manifest 重构。

---

## 2. 执行情况看板

| ID | 事项 | 决策 | 优先级 | 当前状态 | 测试状态 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| D0 | 生成变更执行计划文档 | 采纳 | P0 | DONE | 不适用 | 本文件 |
| D1 | 回写 design 中的不采纳原因 | 采纳 | P0 | DONE | 不适用 | 见来源设计 §9 |
| A1 | 新增因子清单加载器 `feature_registry` | 采纳 | P0 | TODO | TODO | 支持固定字段 + 标的级 txt |
| A2 | `InputSchemaValidator` 支持显式 `feature_columns` | 采纳 | P0 | TODO | TODO | 关闭正式路径的全数值列自动纳入 |
| A3 | Phase I config / CLI / trainer 接入因子清单 | 采纳 | P0 | TODO | TODO | AL 显式配置 `short.txt` |
| A4 | AL 最大交易量落到 `max_position=10` | 部分采纳 | P0 | TODO | TODO | 不新增 `max_trade_volume` 顶层字段 |
| A5 | 更新 unit tests 与 integration smoke | 采纳 | P0 | TODO | TODO | 每个代码批次必须带测试 |
| B1 | 默认推导 `data/{PAIR}/df_*.feather` 路径 | 不采纳 | P1 | DROP | 不适用 | 避免静默读错数据 |
| B2 | 新增正式 CLI `--allow-auto-feature-discovery` | 不采纳 | P1 | DROP | 不适用 | 正式 AL 路径必须显式因子清单 |
| B3 | 新增 `max_trade_volume` 独立配置字段 | 不采纳 | P1 | DROP | 不适用 | 避免与现有 `max_position` 双重语义 |
| B4 | 新增单笔 `max_order_volume` 成交约束 | 暂缓 | P2 | DEFER | 不适用 | 属于交易执行模型扩展 |
| B5 | 本批次生成完整 `feature_provenance.json` | 暂缓 | P2 | DEFER | 不适用 | 只记录因子来源，不做 provenance 生产链 |
| B6 | 修改 Phase II dataset / selector | 不采纳 | P2 | DROP | 不适用 | Phase II 通过 Phase I schema 间接受益 |
| B7 | 重构 artifact manifest / cache 体系 | 不采纳 | P2 | DROP | 不适用 | 只依赖现有 `input_schema_hash` 失效 |

状态说明:

- `TODO`: 尚未实现。
- `DONE`: 本计划文档层面已完成。
- `DROP`: 明确不纳入本批次。
- `DEFER`: 后续单独设计或批次处理。

---

## 3. 采纳结论

### 3.1 采纳项

| 设计项 | 决策 | 批次 | 采纳原因 |
| --- | --- | --- | --- |
| 每标的使用独立因子清单 | 采纳 | A1/A3 | 是本次需求核心，能固定 AL 输入维度和字段含义 |
| `FIXED_FEATURES` 作为统一固定字段 | 采纳 | A1/A2 | 保持盘口和成交字段稳定进入状态，同时让 `close` 明确隔离 |
| `close` 不进入模型状态 | 采纳 | A2/A5 | 已是 Phase I 原设计硬约束，本次需在显式配置路径继续锁死 |
| `src/factors/AL/short.txt` 作为 AL short profile 来源 | 采纳 | A1/A3 | 用户指定路径，且现有仓库已有该文件 |
| 显式 feature list 替代自动数值列推导 | 采纳 | A2 | 降低特征漂移和不可复现实验风险 |
| `input_schema.json` 记录因子来源审计信息 | 采纳 | A2 | 让 schema hash 和后续审计能追踪因子文件来源 |
| AL 最大交易量 `10` | 部分采纳 | A4 | 本批次按现有交易语义落到 `DPConfig.max_position=10` 和 `TradingEnv(max_position=10)` |
| 单元测试和 smoke 测试更新 | 采纳 | A5 | 输入 schema 属于训练入口基础契约，必须有测试防回归 |

### 3.2 不采纳或暂缓项

| 设计项 | 决策 | 原因 | 后续处理 |
| --- | --- | --- | --- |
| 新增 `max_trade_volume` 顶层配置字段 | 不采纳 | 现有代码已经用 `DPConfig.max_position` 和 `TradingEnv.max_position` 表达最大仓位；再加一个字段会产生双写和不一致风险 | CLI 使用 `--max-position 10`，配置落到 `DPConfig.max_position=10` |
| 新增 `max_order_volume` 单笔成交约束 | 暂缓 | 这会改变 `LobDepthCostModel` 和 action/position 语义，属于交易执行模型扩展，不应混入因子 schema 变更 | 另开交易约束设计，明确“最大仓位”和“单笔最大换仓量”的区别 |
| 自动 fallback 到 `data/{PAIR}/df_train.feather` 等路径 | 不采纳 | 训练数据路径一旦被静默推导，容易在不同批次间误读文件；当前 CLI 显式路径更可审计 | AL 命令继续显式传 `--train-file/--val-file/--test-file` |
| 正式新增 `--allow-auto-feature-discovery` | 不采纳 | 本次目标是收紧输入契约；给正式入口加自动发现开关会削弱约束 | 旧自动推导只允许作为内部兼容或单测 fallback，不作为 AL 正式训练入口 |
| 本批次生产完整 `feature_provenance.json` | 暂缓 | provenance 需要外部因子工程链路提供 lookback / 可见时间 / fit scope，当前仓库无法可靠反推 | 本批次只在 `input_schema.json` 记录因子来源；provenance 另设 sign-off 批次 |
| Phase II 同步改造 | 不采纳 | Phase II 已消费 Phase I `input_schema.json`，本批次不需要改 selector 或 dataset 结构 | Phase II 在重新训练 Phase I 后自然读取新 schema |
| artifact manifest / cache 体系重构 | 不采纳 | 结构性风险高，且当前 schema hash 已能触发 demo cache 失效 | 保持现有 `input_schema_hash` 机制 |

---

## 4. 批次 A1: 因子清单加载器

**涉及文件**:

- `src/data/feature_registry.py`
- `src/data/__init__.py`
- `tests/unit/data/test_feature_registry.py`

**实现方案**:

1. 新增 `FIXED_FEATURES` 常量，包含 `close`、五档盘口、成交量、成交额、持仓量字段。
2. 新增 `FeatureSelectionSpec` dataclass，记录:
   - `pair`
   - `profile`
   - `factor_list_path`
   - `fixed_features`
   - `configured_factors`
   - `feature_columns`
   - `price_column`
   - `deduplicated_features`
3. 新增 `load_feature_selection(pair, profile, factor_list_file=None)`:
   - 默认路径为 `src/factors/{PAIR}/{profile}.txt`，但最终 resolved path 必须写入 spec。
   - 支持空行、前后空白、`#` 注释行。
   - 兼容字段名外层 `'` 或 `"`。
   - `close` 出现在因子文件中直接报错。
   - `FIXED_FEATURES` 中的 `close` 不进入 `feature_columns`。
   - fixed 和 factor list 重复时保留 fixed 位置，重复字段写入 `deduplicated_features`。
4. 因子文件不存在时 fail-fast，错误信息包含 pair/profile/path。

**测试修复**:

- `test_feature_registry_loads_pair_factor_list`
- `test_feature_registry_strips_quotes_comments_and_blank_lines`
- `test_feature_registry_rejects_close_in_factor_file`
- `test_feature_registry_deduplicates_fixed_feature`
- `test_feature_registry_missing_file_raises`

**执行命令**:

```bash
pytest tests/unit/data/test_feature_registry.py -q
```

---

## 5. 批次 A2: 显式 Schema 校验

**涉及文件**:

- `src/data/schema.py`
- `tests/unit/data/test_schema.py`

**实现方案**:

1. `InputSchema` 增加可选审计字段 `feature_source: dict | None`。
2. `InputSchemaValidator.__init__()` 增加可选参数:

```python
feature_columns: Optional[List[str]] = None
feature_source: Optional[dict] = None
```

3. 当 `feature_columns` 显式传入时:
   - 只校验这些字段。
   - 额外数值列允许存在，但不进入 `schema.feature_columns`。
   - 缺少任一配置字段直接失败。
   - 任一配置字段非数值、null、NaN、Inf 直接失败。
4. 保留 legacy 自动推导路径，用于现有 fixture 和未迁移的实验，但正式 AL CLI 不走该路径。
5. 新增 `validate_against_schema(frame, schema)`:
   - 用 train 生成的 schema 校验 val/test。
   - 校验 `price_column` 存在且合法。
   - 校验所有 `feature_columns` 存在且数值合法。
   - 不允许 val/test 自行推导出不同 feature list。
6. `write_schema_json()` 输出 `feature_source`，用于审计和 schema hash。

**测试修复**:

- `test_schema_uses_explicit_feature_columns_only`
- `test_schema_requires_all_configured_features`
- `test_schema_rejects_non_numeric_configured_feature`
- `test_schema_validate_against_schema_rejects_missing_val_feature`
- `test_schema_feature_source_written_to_json`
- 更新既有 `close` 隔离测试，确认显式路径下仍满足 `"close" not in feature_columns`。

**执行命令**:

```bash
pytest tests/unit/data/test_schema.py -q
```

---

## 6. 批次 A3: Phase I 配置、CLI 与 Trainer 接入

**涉及文件**:

- `src/config/phase1_config.py`
- `scripts/train_phase1.py`
- `src/trainers/phase1_trainer.py`
- `tests/unit/trainers/test_phase1_trainer.py`
- `tests/integration/test_phase1_pipeline_smoke.py`

**实现方案**:

1. `Phase1Config` 增加低风险字段:

```python
factor_profile: str = "short"
factor_list_file: Optional[str] = None
```

2. `scripts/train_phase1.py` 增加 CLI:

```text
--factor-profile short
--factor-list-file src/factors/AL/short.txt
--max-position 10
```

3. `build_config()` 中:
   - `--max-position` 写入 `DPConfig(max_position=args.max_position)`。
   - `factor_profile` / `factor_list_file` 写入 `Phase1Config`。
4. `Phase1Trainer.run()` 中:
   - 读取 frames 后，调用 `load_feature_selection()`。
   - 用 `feature_spec.feature_columns` 构造 `InputSchemaValidator`。
   - train 调 `validate()`，val/test 调 `validate_against_schema()`。
   - `input_schema.json.feature_source` 写入 `FeatureSelectionSpec` 的审计信息。
5. 若 `factor_list_file` 为空:
   - 正式 CLI 使用 `src/factors/{PAIR}/{factor_profile}.txt`。
   - 单元测试可继续直接构造旧配置；如果没有对应因子文件，可保留 legacy 自动推导路径，但必须在 trainer 日志/报告中标记 `feature_source.mode="legacy_auto_numeric"`。

**测试修复**:

- `test_phase1_trainer_uses_factor_list_schema`
- `test_phase1_trainer_rejects_missing_factor_column`
- `test_phase1_trainer_records_feature_source`
- `test_train_phase1_cli_sets_max_position`
- 更新 smoke fixture，确保显式 factor list 字段存在。

**执行命令**:

```bash
pytest tests/unit/trainers/test_phase1_trainer.py tests/integration/test_phase1_pipeline_smoke.py -q
```

---

## 7. 批次 A4: AL 最大交易量 10

**涉及文件**:

- `scripts/train_phase1.py`
- `src/config/phase1_config.py`
- `src/trainers/phase1_trainer.py`
- `tests/unit/trading/test_env.py`
- `tests/unit/planners/test_single_trade_dp.py`
- `tests/unit/trainers/test_phase1_trainer.py`

**实现方案**:

1. 只采用现有 `max_position` 语义，不新增 `max_trade_volume`。
2. CLI 默认仍保持当前行为；AL 正式训练命令显式传:

```text
--max-position 10
```

3. `Phase1Trainer` 已通过 `self.config.dp.max_position` 构造 planner/env，本批次只补齐 CLI 入参和测试。
4. `phase1_config.yaml` 自然记录 `dp.max_position=10`。
5. `phase1_report.json` 如已有 config dump，则不额外重复字段；如没有，应在 summary 中补 `max_position`。

**测试修复**:

- `test_trading_env_action_mapping_respects_max_position_10`
- `test_single_trade_dp_respects_max_position_10`
- `test_train_phase1_cli_sets_dp_max_position_10`

**执行命令**:

```bash
pytest tests/unit/trading/test_env.py tests/unit/planners/test_single_trade_dp.py tests/unit/trainers/test_phase1_trainer.py -q
```

---

## 8. 批次 A5: AL Smoke 与最终验收

**涉及文件**:

- `tests/integration/test_phase1_pipeline_smoke.py`
- `docs/plan/phase1_archetype_discovery_execution_plan.md` 或 README 类入口文档（如需要记录命令）

**实现方案**:

1. 增加一个轻量 AL schema smoke，使用显式 factor list 和小 `num_demos`。
2. smoke 验证:
   - `input_schema.json.price_column == "close"`。
   - `"close" not in input_schema.json.feature_columns`。
   - `feature_columns` 等于 `FIXED_FEATURES - close + src/factors/AL/short.txt` 去重结果。
   - `feature_source.factor_list_path` 指向 `src/factors/AL/short.txt`。
   - `phase1_config.yaml.dp.max_position == 10`。

**执行命令**:

```bash
pytest tests/unit/data/test_feature_registry.py tests/unit/data/test_schema.py tests/unit/trainers/test_phase1_trainer.py tests/integration/test_phase1_pipeline_smoke.py -q
```

可选本地 AL 数据 dry-run:

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

---

## 9. 验收标准

代码批次完成后必须满足:

- `src/factors/AL/short.txt` 是 AL short profile 的显式因子来源。
- `input_schema.json.feature_columns` 由固定字段去掉 `close` 后拼接 AL short 因子清单得到。
- `close` 不进入模型状态、demo `states` 或 Phase I 模型输入维度。
- train/val/test 共用 train 生成的 schema 校验，不能各自自动推导。
- `data/AL/df_train.feather`、`data/AL/df_val.feather`、`data/AL/df_test.feather` 通过显式 CLI 路径读取。
- `--max-position 10` 落到 `DPConfig.max_position=10` 和 `TradingEnv(max_position=10)`。
- 新增和更新的单元测试全部通过。
- 最终执行记录更新到本文件看板，至少填入执行命令和结果。

---

## 10. 执行顺序

1. A1 因子清单加载器。
2. A2 显式 schema 校验。
3. A3 Phase I trainer 接入。
4. A4 CLI `--max-position` 和 AL 最大仓位测试。
5. A5 smoke 与最终验收。

如果 A2 或 A3 发现现有 fixture 大量依赖 legacy 自动推导，优先补最小 fixture factor list，不扩大 trainer 结构。

---

## 11. 实际执行结果

**执行日期**: 2026-05-02
**执行环境**: `conda activate ArchetypeTrade`

### 11.1 执行结果看板

| ID | 完成标记 | 执行结果 |
| --- | --- | --- |
| D0 | 【✅】 | 已生成本执行计划文档。 |
| D1 | 【✅】 | 已在 `docs/design/20260502_phase1_factor_change_log_design.md` 末尾回写采纳/不采纳原因。 |
| A1 | 【✅】 | 已新增 `src/data/feature_registry.py`，实现 `FIXED_FEATURES`、`FeatureSelectionSpec`、默认路径解析、因子文件解析、`close` 拒绝和重复字段记录。 |
| A2 | 【✅】 | 已扩展 `InputSchemaValidator`，支持显式 `feature_columns`、`feature_source`、`validate_against_schema()`，并保留 legacy 自动数值列路径。 |
| A3 | 【✅】 | 已接入 `Phase1Config`、`scripts/train_phase1.py` 和 `Phase1Trainer`，支持 `--factor-profile`、`--factor-list-file`，trainer 在因子文件存在时走固定字段 + 因子清单路径。 |
| A4 | 【✅】 | 已增加 CLI `--max-position`，并写入 `DPConfig.max_position`；测试覆盖 `max_position=10` 的 env 和 DP 行为。 |
| A5 | 【✅】 | 已补充 `feature_registry`、schema、trainer/CLI、trading env、single-trade DP 和 Phase I smoke 测试，并完成目标 pytest。 |
| B1 | 【✅】 | 已确认不采纳数据路径隐式 fallback；正式路径继续显式传 `--train-file/--val-file/--test-file`。 |
| B2 | 【✅】 | 已确认不采纳正式 `--allow-auto-feature-discovery`；仅保留 trainer 内部 legacy fallback 兼容旧 fixture。 |
| B3 | 【✅】 | 已确认不新增 `max_trade_volume` 顶层字段；采用现有 `max_position` 语义。 |
| B4 | 【✅】 | 已确认单笔 `max_order_volume` 暂缓，未改成本模型和 action/position 结构。 |
| B5 | 【✅】 | 已确认完整 `feature_provenance.json` 暂缓；本批次只写 `input_schema.json.feature_source`。 |
| B6 | 【✅】 | 已确认不改 Phase II；Phase II 后续读取 Phase I 新 schema。 |
| B7 | 【✅】 | 已确认不重构 artifact/cache；继续依赖现有 schema hash 机制。 |

### 11.2 实际代码变更

| 文件 | 执行结果 |
| --- | --- |
| `.gitignore` | 将 `data` 收窄为 `/data/`，避免新增 `src/data/*` 与 `tests/unit/data/*` 被误忽略，同时继续忽略根目录训练数据。 |
| `src/data/feature_registry.py` | 新增因子清单加载器。 |
| `src/data/schema.py` | 显式 feature list 校验、split schema 一致性校验和 `feature_source` 审计信息落地。 |
| `src/config/phase1_config.py` | 新增 `factor_profile`、`factor_list_file`。 |
| `scripts/train_phase1.py` | 新增 `--factor-profile`、`--factor-list-file`、`--max-position`。 |
| `src/trainers/phase1_trainer.py` | 训练前按因子清单构造 schema validator；val/test 沿用 train schema；最终 report 追加 `max_position/factor_profile/factor_list_file`。 |
| `tests/fixtures/phase1/build_fixtures.py` | fixture 补齐固定成交字段 `total_trade_volume/turnover/open_interest`。 |
| `tests/integration/test_phase1_pipeline_smoke.py` | smoke 改为显式 factor list 路径，并断言 `feature_source` 与 `close` 隔离。 |
| `tests/unit/data/test_feature_registry.py` | 新增因子清单加载器测试。 |
| `tests/unit/data/test_schema.py` | 新增显式 schema 测试。 |
| `tests/unit/trainers/test_phase1_trainer.py` | 新增 trainer 因子清单接入和 CLI max-position 测试。 |
| `tests/unit/trading/test_env.py` | 新增 `max_position=10` action 映射测试。 |
| `tests/unit/planners/test_single_trade_dp.py` | 新增 DP `max_position=10` 行为测试。 |

### 11.3 测试执行记录

```bash
source /home/lanceliang/miniconda3/etc/profile.d/conda.sh && conda activate ArchetypeTrade && \
pytest tests/unit/data/test_feature_registry.py tests/unit/data/test_schema.py tests/unit/trainers/test_phase1_trainer.py tests/unit/trading/test_env.py tests/unit/planners/test_single_trade_dp.py -q
```

结果: `44 passed in 3.23s`

```bash
source /home/lanceliang/miniconda3/etc/profile.d/conda.sh && conda activate ArchetypeTrade && \
pytest tests/integration/test_phase1_pipeline_smoke.py -q
```

结果: `4 passed in 4.90s`

```bash
source /home/lanceliang/miniconda3/etc/profile.d/conda.sh && conda activate ArchetypeTrade && \
pytest tests/unit/data/test_feature_registry.py tests/unit/data/test_schema.py tests/unit/trainers/test_phase1_trainer.py tests/integration/test_phase1_pipeline_smoke.py -q
```

结果: `31 passed in 4.47s`

```bash
source /home/lanceliang/miniconda3/etc/profile.d/conda.sh && conda activate ArchetypeTrade && \
pytest tests/integration/test_phase1_reproducibility.py tests/integration/test_phase1_next_row_alignment.py -q
```

结果: `4 passed in 0.28s`

### 11.4 AL 真实数据预检与 dry-run

AL 字段预检已执行:

- `data/AL/df_train.feather`: `missing_count=0`，`feature_dim=53`
- `data/AL/df_val.feather`: `missing_count=0`，`feature_dim=53`
- `data/AL/df_test.feather`: `missing_count=0`，`feature_dim=53`

执行 AL 小样本 dry-run:

```bash
source /home/lanceliang/miniconda3/etc/profile.d/conda.sh && conda activate ArchetypeTrade && \
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
  --batch-size 32 \
  --device cpu \
  --allow-missing-prospective-diagnostic \
  --risk-acknowledged-by local_smoke \
  --expected-sign-off-followup-batch-id al_short_full
```

结果: 因采样健康检查停止，错误为 `flat_low_vol_sample_ratio=0.188 > max=0.15`。该失败发生在 schema 写出和真实字段校验之后，不是因子清单缺列或 `close` 进入模型状态导致。临时生成的 `artifacts/AL/al_short_schema_smoke/phase1/input_schema.json` 与 `phase1_config.yaml` 已清理。
