# Phase I Archetype Discovery 可执行代码生成计划

本文档根据 `docs/design/phase1_archetype_discovery_design.md` 生成可执行实施计划。目标不是在本计划中粘贴生产代码，而是明确后续代码生成、审查、测试数据、单元测试、集成测试和验收命令的执行顺序。

## 1. 实施目标

Phase I 需要交付一条可运行的离线训练链路:

```text
读取 train/val/test 数据
  -> schema 校验
  -> horizon 滑窗索引
  -> 分层采样
  -> 采样健康检查
  -> horizon 构造
  -> Single-trade DP demonstration
  -> VQ encoder-decoder 训练
  -> validation 评估
  -> checkpoint 选择
  -> codebook/decoder/labels/report 导出
```

实施边界:

- 数据处理统一使用 `polars`，不使用 `pandas`。
- 输入与输出表格统一使用 Feather/Arrow IPC 格式，生产代码用 `polars.read_ipc` / `DataFrame.write_ipc` 读写 `.feather` 文件。
- `close` 只作为价格列使用，用于 DP reward、分层统计、replay 和风险指标计算；`close` 不纳入模型输入状态 `states`。
- 模型训练使用 `torch`。
- 测试框架使用 `pytest`。
- Phase I 不实现 Phase II/III 的 RL 训练，只导出它们需要的 `decoder.pt`、`codebook.pt` 和 `horizon_labels_*.feather`。
- DP 只能用于离线 demonstration 和 horizon label 生成，不能进入后续线上推理。

## 2. 依赖安装计划

建议更新 `requirements.txt`，补齐 Phase I 生产代码和测试依赖。

必需依赖:

```text
polars>=0.20.0
pyarrow>=14.0.0          # Feather/Arrow IPC 兼容与生态互操作
numpy>=1.24.0
torch>=2.2.0
tqdm>=4.64.0
PyYAML>=6.0.0
pydantic>=2.0.0
pytest>=8.0.0
pytest-cov>=5.0.0
```

建议依赖:

```text
scikit-learn>=1.4.0      # kmeans warmup、silhouette、诊断指标
tensorboard>=2.15.0      # latent/codebook 训练诊断
matplotlib>=3.8.0        # failure case HTML 图表，第一版可暂缓
```

安装与验证命令:

```bash
python3 -m pip install -r requirements.txt
python3 - <<'PY'
import polars, pyarrow, numpy, torch, pytest, yaml, pydantic
print("phase1 dependencies ok")
PY
```

## 3. 代码生成顺序

按以下顺序生成代码，每一步完成后运行对应测试，避免一次性生成大面积不可定位的问题。

### Step 1: 配置与基础 IO

生成文件:

```text
src/config/phase1_config.py
src/utils/io.py
scripts/train_phase1.py
```

核心内容:

- `Phase1Config`: 输入路径、输出路径、horizon、采样、成本、模型、训练、checkpoint 配置。
- CLI 参数解析: `--pair`、`--train-batch-id`、`--train-file`、`--val-file`、`--test-file`、`--horizon`、`--num-demos`、`--epochs`、`--seed` 等。
- 输出目录: `artifacts/{PAIR}/{BATCH_ID}/phase1/`。
- 写入 `phase1_config.yaml`。

验收:

```bash
python scripts/train_phase1.py --help
```

### Step 2: 数据读取与 schema 校验

生成文件:

```text
src/data/market_reader.py
src/data/schema.py
```

核心内容:

- `MarketFileReader` 使用 `polars.read_ipc` 读取 `.feather`，可选使用 `polars.read_csv` 读取调试 CSV。
- `InputSchemaValidator` 识别 `timestamp`、`close`、状态特征列、盘口列。
- 排除 `timestamp/symbol/split/sample_id/close` 等非模型输入列。
- `close` 必须记录为 `price_column`，不能出现在 `feature_columns` 或模型输入 `states` 中。
- 检查数值列 NaN/Inf、`close` 正数、时间排序。
- 输出 `input_schema.json`。

验收:

```bash
pytest tests/unit/data/test_market_reader.py tests/unit/data/test_schema.py
```

### Step 3: 滑窗、分层采样与采样健康检查

生成文件:

```text
src/data/window_indexer.py
src/data/stratified_sampler.py
src/data/sampling_health.py
```

核心内容:

- `SlidingWindowIndexer` 支持 `paper_formula` 和 `next_row_execution`。
- `StratifiedWindowSampler` 支持 `hindsight_horizon` 和 `prospective_past`。
- 采样策略支持 `stratified_uniform` 和 `stratified_proportional`。
- 实现 `min_gap_between_samples`、`flat_low_vol_max_ratio`。
- `SamplingHealthChecker` 输出:
  - `window_overlap_ratio`
  - `min_sample_gap`
  - `mean_sample_gap`
  - `flat_low_vol_sample_ratio`
  - `sampling_health_warnings`
- 写入 `window_index_train.feather`、`window_index_val.feather`、`window_index_test.feather`。

验收:

```bash
pytest tests/unit/data/test_window_indexer.py
pytest tests/unit/data/test_stratified_sampler.py
pytest tests/unit/data/test_sampling_health.py
```

### Step 4: 成本模型、reward alignment 与交易环境

生成文件:

```text
src/envs/reward_alignment.py
src/trading/cost_model.py
src/envs/trading_env.py
```

核心内容:

- `RewardAlignment.rows(decision_offset)`:
  - `paper_formula`: execution row = `t`，markout row = `t + 1`
  - `next_row_execution`: execution row = `t + 1`，markout row = `t + 2`
- `LobDepthCostModel`:
  - long/平空走 ask 档。
  - short/平多走 bid 档。
  - 五档深度不足按 `reject_transition` 处理。
  - 计算 fee、slippage、fill price、filled qty。
- `TradingEnv`:
  - action `{0,1,2}` 映射到 position `{-1,0,1}`。
  - 支持 `initial_position != 0`。
  - `step()` 返回净 reward 和成交 info。

验收:

```bash
pytest tests/unit/envs/test_reward_alignment.py
pytest tests/unit/trading/test_cost_model.py
pytest tests/unit/envs/test_trading_env.py
```

### Step 5: horizon 构造与 demonstration store

生成文件:

```text
src/data/horizon_builder.py
src/data/demo_store.py
src/data/dataset.py
```

核心内容:

- `HorizonBuilder` 根据窗口索引切出:
  - `states`
  - `prices`
  - `execution_books`
  - `meta`
- `states` 只包含 `feature_columns`，不得包含 `close`。
- `prices` 从 `close` 列切出，只用于 reward、DP、replay 和指标。
- `Phase1DemoStore` 保存:
  - `demos_train.feather`
  - `horizon_labels_train.feather`
  - `horizon_labels_val.feather`
  - `horizon_labels_test.feather`
- `Phase1DemoDataset` 只做 PyTorch Dataset/DataLoader 适配，不读取原始文件，不调用 DP。

验收:

```bash
pytest tests/unit/data/test_horizon_builder.py
pytest tests/unit/data/test_demo_store.py
pytest tests/unit/data/test_dataset.py
```

### Step 6: Single-trade DP planner

生成文件:

```text
src/planners/single_trade_dp.py
src/planners/demo_generator.py
```

核心内容:

- DP 状态为 `(t, action, changed)`。
- 初始 action 为 flat，即 `1`。
- 最多一次动作切换。
- 转移 reward 必须通过 `TradingEnv` 或共享 cost/reward 接口计算。
- 输出:
  - `actions`
  - `rewards`
  - `total_return`
  - `num_switches`
  - `is_no_trade`
- `Phase1DemoGenerator` 批量生成 train/val/test horizon 的 DP 标签。

验收:

```bash
pytest tests/unit/planners/test_single_trade_dp.py
pytest tests/unit/planners/test_demo_generator.py
```

### Step 7: VQ 模型组件

生成文件:

```text
src/models/encoder_inputs.py
src/models/vector_quantizer.py
src/models/vq_losses.py
src/models/vq_archetype.py
```

核心内容:

- `RewardNormalizer` 只在 train rewards 上 fit。
- `EncoderInputAdapter` 分别处理 state/action/reward。
- `ArchetypeEncoder`: LSTM encoder，输出 `z_e`。
- `VectorQuantizer`:
  - 支持 `random_normal`、`sample_encoder_outputs`、`kmeans_warmup`。
  - 支持 `gradient` 和 `ema` 更新。
  - 输出 `code_id`、`z_q`、usage stats。
- `ArchetypeDecoder`:
  - 单向 LSTM。
  - 只使用 `state_t/past states + z_q`。
  - 禁止双向 LSTM 或全 horizon pooling。
- `Phase1Loss`:
  - reconstruction CE
  - VQ loss
  - commitment loss
  - 可选 usage regularization

验收:

```bash
pytest tests/unit/models/test_reward_normalizer.py
pytest tests/unit/models/test_vector_quantizer.py
pytest tests/unit/models/test_vq_losses.py
pytest tests/unit/models/test_vq_archetype.py
```

### Step 8: Phase I 指标与 replay 评估

生成文件:

```text
src/evaluation/action_metrics.py
src/evaluation/risk_metrics.py
src/evaluation/archetype_diagnostics.py
src/evaluation/behavior_diagnostics.py
src/evaluation/phase1_metrics.py
src/evaluation/phase1_replay.py
src/evaluation/phase1_evaluator.py
```

核心内容:

- action 指标:
  - reconstruction accuracy
  - weighted reconstruction accuracy
  - non-flat accuracy
  - confusion matrix
  - switch point recall
  - switch direction accuracy
- VQ 指标:
  - code usage ratio
  - perplexity
  - dominant code ratio
  - dead code count
- replay 指标:
  - student online net return
  - DP teacher net return
  - return capture ratio
  - regret to DP
  - cost paid
- 风险指标:
  - sharpe
  - sortino
  - max drawdown
  - calmar
- 行为诊断:
  - single-trade consistency
  - inter-code action diversity
  - decoder sensitivity to code

验收:

```bash
pytest tests/unit/evaluation/test_action_metrics.py
pytest tests/unit/evaluation/test_risk_metrics.py
pytest tests/unit/evaluation/test_phase1_metrics.py
pytest tests/unit/evaluation/test_phase1_replay.py
pytest tests/unit/evaluation/test_phase1_evaluator.py
```

### Step 9: checkpoint、报告与训练器

生成文件:

```text
src/trainers/phase1_checkpoint.py
src/evaluation/phase1_report.py
src/trainers/phase1_trainer.py
```

核心内容:

- `Phase1CheckpointManager`:
  - 保存 `last_vq_model.pt`
  - 保存 `best_vq_model.pt`
  - 可选保存 `checkpoints/epoch_*.pt`
  - 写入 `checkpoint_manifest.json`
- best checkpoint 选择:
  - 默认使用 `phase1_composite_score`
  - `code_usage_ratio < min_code_usage_ratio` 不可成为 best
  - 风险 guardrail 不通过时记录拒绝原因
- `Phase1ReportWriter`:
  - 写入 `phase1_report.json`
  - 写入 `action_diagnostics.json`
  - 写入 `risk_diagnostics.json`
  - 写入 `archetype_behavior_diagnostics.json`
- `Phase1Trainer` 编排完整训练流程。

验收:

```bash
pytest tests/unit/trainers/test_phase1_checkpoint.py
pytest tests/unit/evaluation/test_phase1_report.py
pytest tests/unit/trainers/test_phase1_trainer.py
```

### Step 10: 集成入口

完善文件:

```text
scripts/train_phase1.py
run_pipeline.sh
```

核心内容:

- `scripts/train_phase1.py` 调用 `Phase1Trainer`。
- `run_pipeline.sh` 已预留 Phase I 调用，保持参数兼容。
- 支持小数据 smoke run:

```bash
python scripts/train_phase1.py \
  --pair AL \
  --train-batch-id smoke_phase1 \
  --train-file tests/fixtures/phase1/market_train.feather \
  --val-file tests/fixtures/phase1/market_val.feather \
  --test-file tests/fixtures/phase1/market_test.feather \
  --horizon 8 \
  --num-demos 12 \
  --num-archetypes 4 \
  --epochs 2 \
  --batch-size 4 \
  --seed 42
```

验收:

```bash
pytest tests/integration/test_phase1_pipeline_smoke.py
```

## 4. 单元测试用例计划

### 数据层

`tests/unit/data/test_market_reader.py`

- 读取 feather 成功。
- 读取 csv 成功。
- 不支持的扩展名报错。
- 文件不存在报错。

`tests/unit/data/test_schema.py`

- `timestamp + close + numeric features` 通过。
- 缺少 `close` 报错。
- `close <= 0` 报错。
- NaN/Inf 报错。
- 元信息列不进入 feature columns。
- `close` 不进入 feature columns。
- 盘口字段识别正确。

`tests/unit/data/test_window_indexer.py`

- `paper_formula` 下 `num_rows=20,h=8` 生成 12 个窗口。
- `next_row_execution` 下 `num_rows=20,h=8` 生成 11 个窗口。
- `last_execution_row/last_markout_row` 正确。
- stride 生效。

`tests/unit/data/test_stratified_sampler.py`

- 相同 seed 采样结果稳定。
- `stratified_uniform` 尽量均匀覆盖 strata。
- `stratified_proportional` 按候选数量比例采样。
- `min_gap_between_samples` 生效。
- flat low vol 样本比例不超过配置。

`tests/unit/data/test_sampling_health.py`

- 计算 overlap ratio 正确。
- 计算 min/mean gap 正确。
- 超过 max overlap 产生 warning。
- `warn_only=false` 时健康检查失败。

`tests/unit/data/test_horizon_builder.py`

- states shape 为 `[h, feature_dim]`。
- states 不包含 `close`，feature_dim 等于数值特征列数量减去元信息列、盘口元数据排除列和 `close`。
- `paper_formula` prices shape 为 `[h + 1]`。
- `next_row_execution` prices shape 为 `[h + 2]`。
- sample_id、start_index、end_index 正确。

### 环境与成本层

`tests/unit/envs/test_reward_alignment.py`

- `paper_formula` 行号映射正确。
- `next_row_execution` 行号映射正确。
- 非法 alignment 报错。

`tests/unit/trading/test_cost_model.py`

- 买入使用 ask 档。
- 卖出使用 bid 档。
- 五档深度足够时 fill price 为加权均价。
- 深度不足时 reject transition。
- fee 与 slippage 计算正确。

`tests/unit/envs/test_trading_env.py`

- action 到 position 映射正确。
- flat -> long 的 reward 扣除成本。
- long -> flat 的 reward 扣除成本。
- 支持 non-flat initial position。
- replay actions 汇总收益等于逐步 step 收益之和。

### DP 层

`tests/unit/planners/test_single_trade_dp.py`

- 单调上涨价格下选择 flat -> long，且最多一次切换。
- 单调下跌价格下选择 flat -> short，且最多一次切换。
- 横盘且有手续费时选择全 flat。
- 所有输出 actions 长度为 h。
- `num_switches <= 1`。
- DP total_return 等于 env replay return。

`tests/unit/planners/test_demo_generator.py`

- 批量 horizon 都生成 actions/rewards。
- no-trade horizon 标记正确。
- metadata 保留 strata_label 和 sample_id。

### 模型层

`tests/unit/models/test_reward_normalizer.py`

- 只用 train rewards fit mean/std。
- transform 后均值接近 0。
- clip ratio 统计正确。
- std 过小时使用 epsilon。

`tests/unit/models/test_vector_quantizer.py`

- nearest code id 正确。
- straight-through 输出 shape 正确。
- usage stats 正确。
- EMA update 修改 codebook。
- dead code 统计正确。

`tests/unit/models/test_vq_archetype.py`

- forward 输出 logits `[batch,h,3]`。
- 输出 `code_id` `[batch]`。
- decoder 是单向因果结构。
- 修改未来 states 不改变过去 timestep logits。

### 指标与评估层

`tests/unit/evaluation/test_action_metrics.py`

- reconstruction accuracy 正确。
- weighted accuracy 正确。
- non-flat accuracy 正确。
- confusion matrix 正确。
- switch recall 和 direction accuracy 正确。

`tests/unit/evaluation/test_risk_metrics.py`

- 全正收益 sharpe 为正。
- 下行收益 sortino 有效。
- max drawdown 正确。
- calmar 在 drawdown 为 0 时稳定处理。

`tests/unit/evaluation/test_phase1_metrics.py`

- codebook perplexity 正确。
- code_usage_ratio 正确。
- return_capture_ratio 对 teacher return 接近 0 时稳定。
- phase1_composite_score 可计算。

`tests/unit/evaluation/test_phase1_replay.py`

- teacher replay 与 DP total_return 对齐。
- student replay 使用相同 reward alignment。
- boundary turnover cost 可计算。

### 训练与产物层

`tests/unit/trainers/test_phase1_checkpoint.py`

- 保存 last checkpoint。
- metric 变好时保存 best。
- code usage guardrail 不通过时拒绝 best。
- manifest 写入拒绝原因。

`tests/unit/evaluation/test_phase1_report.py`

- report 包含配置、采样健康、模型指标、风险指标。
- JSON 可读。
- 必需字段缺失时报错。

## 5. 集成测试用例计划

`tests/integration/test_phase1_pipeline_smoke.py`

目标: 使用小型 fixture 数据跑通完整 Phase I。

输入:

```text
tests/fixtures/phase1/market_train.feather
tests/fixtures/phase1/market_val.feather
tests/fixtures/phase1/market_test.feather
```

命令:

```bash
python scripts/train_phase1.py \
  --pair TEST \
  --train-batch-id integration_smoke \
  --train-file tests/fixtures/phase1/market_train.feather \
  --val-file tests/fixtures/phase1/market_val.feather \
  --test-file tests/fixtures/phase1/market_test.feather \
  --horizon 8 \
  --window-stride 1 \
  --num-demos 12 \
  --num-archetypes 4 \
  --epochs 2 \
  --batch-size 4 \
  --seed 7
```

断言:

- 进程退出码为 0。
- 以下文件存在:

```text
artifacts/TEST/integration_smoke/phase1/phase1_config.yaml
artifacts/TEST/integration_smoke/phase1/input_schema.json
artifacts/TEST/integration_smoke/phase1/window_index_train.feather
artifacts/TEST/integration_smoke/phase1/demos_train.feather
artifacts/TEST/integration_smoke/phase1/horizon_labels_train.feather
artifacts/TEST/integration_smoke/phase1/horizon_labels_val.feather
artifacts/TEST/integration_smoke/phase1/horizon_labels_test.feather
artifacts/TEST/integration_smoke/phase1/best_vq_model.pt
artifacts/TEST/integration_smoke/phase1/last_vq_model.pt
artifacts/TEST/integration_smoke/phase1/decoder.pt
artifacts/TEST/integration_smoke/phase1/codebook.pt
artifacts/TEST/integration_smoke/phase1/checkpoint_manifest.json
artifacts/TEST/integration_smoke/phase1/phase1_report.json
```

- `phase1_report.json` 中:
  - `num_sampled_horizons == 12`
  - `reward_alignment == "paper_formula"`
  - `code_usage_ratio >= 0`
  - `sampling_health_warnings` 字段存在
- `input_schema.json` 中:
  - `price_column == "close"`
  - `feature_columns` 不包含 `close`
- `demos_train.feather` 中保存的 `states` 维度与 `feature_columns` 一致，且不包含 `close`。
- labels 文件中 `code_label` 范围在 `[0, num_archetypes - 1]`。

`tests/integration/test_phase1_next_row_alignment.py`

目标: 验证 `next_row_execution` 全链路行号一致。

断言:

- `window_index_train.feather` 的 `last_execution_row = start_index + h`。
- `last_markout_row = start_index + h + 1`。
- DP replay 和 student replay 都记录 `reward_alignment=next_row_execution`。

`tests/integration/test_phase1_resume_checkpoint.py`

目标: 验证 checkpoint 恢复训练。

步骤:

1. 先跑 `epochs=1`。
2. 再用 `--resume-from last_vq_model.pt --epochs 2`。

断言:

- 第二次训练从 epoch 2 继续。
- `checkpoint_manifest.json` 保留两次训练记录。

## 6. 单元测试数据计划

生成 fixture 脚本:

```text
tests/fixtures/phase1/generate_phase1_fixtures.py
```

生成文件:

```text
tests/fixtures/phase1/market_train.feather
tests/fixtures/phase1/market_val.feather
tests/fixtures/phase1/market_test.feather
tests/fixtures/phase1/market_bad_missing_close.feather
tests/fixtures/phase1/market_bad_nan.feather
tests/fixtures/phase1/market_bad_crossed_book.feather
```

正常 fixture 字段:

```text
timestamp
close
ask1_price ... ask5_price
ask1_size ... ask5_size
bid1_price ... bid5_price
bid1_size ... bid5_size
total_trade_volume
turnover
open_interest
feature_return_1
feature_vol_4
feature_momentum_8
```

fixture 约束:

- `close` 是价格列，只用于 `prices`、reward、DP、replay 和指标。
- 模型输入状态只使用 `feature_return_1/feature_vol_4/feature_momentum_8` 以及经 schema 允许的非价格数值特征。
- `close`、`timestamp` 和元信息列不得进入 `feature_columns`。

数据规模:

- train: 64 行。
- val: 32 行。
- test: 32 行。
- horizon: 集成测试默认用 8。

价格场景:

- 前 16 行单调上涨，用于验证 long DP。
- 中间 16 行横盘，用于验证 no-trade。
- 后 16 行单调下跌，用于验证 short DP。
- 额外加入轻微噪声，用于分层采样和 volatility bucket。

盘口规则:

- `mid = close`
- `ask{level}_price = close + spread * level`
- `bid{level}_price = close - spread * level`
- size 全部为正数，默认大于 1，保证普通测试深度充足。
- `market_bad_crossed_book.feather` 中故意设置 `bid1_price > ask1_price`，用于 schema/cost 测试。

fixture 生成命令:

```bash
python tests/fixtures/phase1/generate_phase1_fixtures.py
```

## 7. 代码审查计划

每个 Step 完成后按以下清单审查。

数据与泄漏审查:

- train/val/test 分开读取，不跨 split fit 统计量。
- reward normalizer 只 fit train。
- DP 和 encoder 可以看完整 horizon，但 decoder 不能看未来。
- validation/test 不做数据增强。
- `hindsight_horizon` 分层在 report 中明确标记为 hindsight。

交易语义审查:

- DP、teacher replay、student replay 使用同一 `TradingEnv` 和 `CostModel`。
- `paper_formula` 与 `next_row_execution` 行号映射不混用。
- action `{0,1,2}` 到 position `{-1,0,1}` 一致。
- 手续费和滑点在 DP demonstration 中启用。
- 深度不足策略在 DP 和 replay 中有明确行为。

模型审查:

- decoder 是 unidirectional LSTM。
- 未来 state 改动不会影响过去 logits。
- codebook update method 写入 config/report。
- `code_usage_ratio` guardrail 参与 best checkpoint 选择。

产物审查:

- 所有输出写入 `artifacts/{PAIR}/{BATCH_ID}/phase1/`。
- `phase1_config.yaml` 可复现实验配置。
- `checkpoint_manifest.json` 记录 best/last/拒绝原因。
- `phase1_report.json` 包含采样、模型、风险、行为诊断。

测试审查:

- 单元测试覆盖 DP 单次切换约束。
- 单元测试覆盖 decoder 因果性。
- 集成测试覆盖完整 CLI。
- fixture 小而确定，seed 固定。

## 8. 执行命令总览

安装依赖:

```bash
python3 -m pip install -r requirements.txt
```

生成测试数据:

```bash
python tests/fixtures/phase1/generate_phase1_fixtures.py
```

运行单元测试:

```bash
pytest tests/unit -q
```

运行集成测试:

```bash
pytest tests/integration -q
```

运行全部测试并统计覆盖率:

```bash
pytest tests --cov=src --cov=scripts --cov-report=term-missing
```

运行 Phase I smoke 训练:

```bash
python scripts/train_phase1.py \
  --pair TEST \
  --train-batch-id smoke_phase1 \
  --train-file tests/fixtures/phase1/market_train.feather \
  --val-file tests/fixtures/phase1/market_val.feather \
  --test-file tests/fixtures/phase1/market_test.feather \
  --horizon 8 \
  --num-demos 12 \
  --num-archetypes 4 \
  --epochs 2 \
  --batch-size 4 \
  --seed 42
```

运行真实数据 Phase I:

```bash
python scripts/train_phase1.py \
  --pair AL \
  --train-batch-id batch_001 \
  --train-file data/AL/train.feather \
  --val-file data/AL/val.feather \
  --test-file data/AL/test.feather \
  --horizon 72 \
  --window-stride 1 \
  --sampling-strategy stratified_uniform \
  --stratification-mode hindsight_horizon \
  --num-demos 30000 \
  --num-archetypes 10 \
  --epochs 100 \
  --seed 42
```

运行前瞻性分层诊断批次:

```bash
python scripts/train_phase1.py \
  --pair AL \
  --train-batch-id batch_002_prospective_strata \
  --train-file data/AL/train.feather \
  --val-file data/AL/val.feather \
  --test-file data/AL/test.feather \
  --horizon 72 \
  --window-stride 1 \
  --sampling-strategy stratified_uniform \
  --stratification-mode prospective_past \
  --prospective-lookback-minutes 1440 \
  --num-demos 30000 \
  --num-archetypes 10 \
  --epochs 100 \
  --seed 42
```

## 9. 完成定义

Phase I 代码生成完成需要同时满足:

- `pytest tests/unit -q` 通过。
- `pytest tests/integration -q` 通过。
- smoke 训练能生成完整 artifacts。
- `phase1_report.json` 包含采样健康、VQ、重构、replay、风险、codebook 诊断字段。
- `best_vq_model.pt`、`decoder.pt`、`codebook.pt`、`horizon_labels_*.feather` 可被 Phase II 读取。
- 代码审查清单全部通过，特别是 decoder 因果性和 DP 使用边界。
