# Phase I Archetype Discovery 技术设计

## 1. 目标与范围

本文档根据论文 `docs/paper/AAAI26_ArchetypeTrader_core.md` 中第一阶段 Archetype Discovery 的要求，设计可落地的工程方案。第一阶段负责从训练集历史行情中生成高质量 demonstration trajectories，并通过 VQ encoder-decoder 学习离散、可复用的 trading archetypes。

第一阶段只做三件事:

1. 直接读取外部准备好的 train/val/test 三个数据文件，并在文件内部生成固定长度 horizon。
2. 使用 Single-trade DP planner 生成单次交易 demonstration。
3. 训练 VQ encoder-decoder，导出 codebook、decoder 和 horizon-level archetype labels。

DP 只允许在 Phase I 离线 demonstration 生成和 horizon label 生成时使用。Phase II/III 的验证、测试、回测推理和线上推理不能动态调用 DP，避免未来信息泄漏。

## 2. 论文要求映射

| 论文要求 | 工程设计 |
| --- | --- |
| 采样 `n=30000` 个固定长度 chunk | `Phase1DatasetBuilder` 在 1 分钟训练数据上用滑动窗口枚举候选 horizon，再做分层采样 |
| 每个 chunk 限制单次交易 | `SingleTradeDPPlanner` 用 `(t, action, changed)` 状态约束最多一次动作切换 |
| action 属于 `{0,1,2}` | 统一定义 `0=short`, `1=flat`, `2=long` |
| demonstration tuple 为 `(s_demo, a_demo, r_demo)` | 缓存为 horizon 样本，包含 `states/actions/rewards/prices/meta` |
| LSTM encoder 输出连续 latent | `ArchetypeEncoder` 输出 `z_e` |
| VQ codebook 离散化 latent | `VectorQuantizer` 输出 `code_id` 和 `z_q` |
| decoder 根据 state 和 code 重构 action | `ArchetypeDecoder` 输出每步三分类 action logits |
| 损失为 reconstruction + VQ + commitment | `Phase1Loss` 实现论文公式 (4) |
| Phase II 需要 demonstration label | 对 train/val/test horizon 生成 `code_label` 文件 |

### 2.1 与论文第一阶段的一致性边界

设计文档保留论文第一阶段的核心架构和算法公式:

```text
sample horizons -> Single-trade DP demonstration -> LSTM encoder -> VQ codebook -> decoder reconstructs actions
```

以下内容属于工程落地增强，不改变论文第一阶段主干:

| 设计项 | 与论文关系 | 是否改变核心公式 |
| --- | --- | --- |
| 滑动窗口后分层采样 30000 个 horizon | 论文只要求采样 `n` 个固定长度 chunk；本设计给出小数据量下的采样实现 | 否 |
| 手续费和盘口深度滑点 | 论文 MDP 已包含 execution loss 和 commission；本设计细化 `O_t` 的成本计算 | 否 |
| Reward 时间对齐 | 默认 `paper_formula` 严格对应论文 `p_{t+1}^{mark}-p_t^{mark}`；可选 `next_row_execution` 只用于保守在线仿真 | `paper_formula` 不改变；`next_row_execution` 会改变 reward timing，需单独标注且不可直接与论文实验比较 |
| `TradingEnv` / `CostModel` | 工程上统一 reward、成交和 replay 语义 | 否 |
| checkpoint online replay | 额外验证 student 与 DP teacher 的收益差距 | 否 |
| causal decoder 约束 | 为避免部署时未来信息泄漏，对 decoder 施加实现约束 | 不改变公式 (3)，但约束其因果实现 |
| no-trade 样本保留 | 工程配置项；严格复现论文可过滤 no-trade | 不改变 DP 转移，只影响数据集纳入规则 |
| usage regularization / dead-code restart | codebook collapse 防护；可关闭以严格复现公式 (4) | 开启时会扩展训练 loss，需记录配置 |

## 3. 数据契约

### 3.1 输入数据

本项目第一阶段不负责特征工程，不在当前仓库内重新计算盘口因子或技术指标。输入数据视为已经完成清洗、对齐和因子构造，Phase I 只负责读取文件、校验字段、生成 horizon、运行 DP 和训练 VQ。

默认输入为三份已经切分好的数据文件:

```text
data/{PAIR}/train.parquet
data/{PAIR}/val.parquet
data/{PAIR}/test.parquet
```

也可以通过 CLI 显式指定:

```text
--train-file data/{PAIR}/train.parquet
--val-file data/{PAIR}/val.parquet
--test-file data/{PAIR}/test.parquet
```

后续建议支持 csv/parquet 两类格式。每个文件至少需要:

- 时间列: `timestamp` 或可等价排序的 index。
- 价格列: `close`，用于 DP reward 计算。
- 特征列: 除去元信息列后可直接作为状态 $s_t$ 的数值列。

如果输入文件包含以下盘口和成交字段，则可直接纳入状态:

- `close`
- `ask1_price` 到 `ask5_price`
- `ask1_size` 到 `ask5_size`
- `bid1_price` 到 `bid5_price`
- `bid1_size` 到 `bid5_size`
- `total_trade_volume`
- `turnover`
- `open_interest`

派生因子也直接从文件读取。`docs/paper/因子清单.txt` 只作为字段参考，不要求本项目重新生成这些因子。

### 3.2 状态字段

每个 bar 的状态为:

$$
s_t \in R^d
$$

状态字段选择规则:

- `close` 保留为价格列，同时也可以作为状态特征列。
- `timestamp`、`symbol`、`split`、`sample_id` 等元信息列不进入模型。
- 其余数值列默认进入 `states`。
- 本阶段只做字段校验、类型转换和 NaN/Inf 检查，不做滚动因子生成和标准化拟合。

建议保存:

```text
artifacts/{PAIR}/{BATCH_ID}/phase1/input_schema.json
artifacts/{PAIR}/{BATCH_ID}/phase1/window_index_train.parquet
artifacts/{PAIR}/{BATCH_ID}/phase1/window_index_val.parquet
artifacts/{PAIR}/{BATCH_ID}/phase1/window_index_test.parquet
```

### 3.3 Horizon 样本

每个样本长度为 `h=72`，数据结构如下:

| 字段 | shape | dtype | 说明 |
| --- | --- | --- | --- |
| `states` | `[h, feature_dim]` | `float32` | 从输入文件直接读取的状态特征 |
| `prices` | `[h + 1]` 或 `[h + 2]` | `float32` | `paper_formula` 用 `[h+1]`；`next_row_execution` 用 `[h+2]` |
| `execution_books` | `[h, levels, 4]` | `float32` | 每步实际成交行的 bid/ask 盘口，由 `reward_alignment` 决定来自当前行或下一行 |
| `actions` | `[h]` | `int64` | DP demonstration action |
| `rewards` | `[h]` | `float32` | 执行 demonstration action 的逐步收益 |
| `start_index` | scalar | `int64` | horizon 起点 |
| `end_index` | scalar | `int64` | horizon 终点 |
| `pair` | scalar | `str` | 交易标的 |
| `split` | scalar | `str` | train/val/test |
| `sample_id` | scalar | `str` | 可复现样本 ID |

Reward 时间对齐由 `reward_alignment` 显式配置:

- `paper_formula`: 与论文公式对齐，动作在第 `k` 行成交，用第 `k+1` 行 mark price 结算，窗口需要覆盖 `h + 1` 行价格。
- `next_row_execution`: 保守在线执行模式，第 `k` 行只用于观察，动作在第 `k+1` 行成交，用第 `k+2` 行 mark price 结算，窗口需要覆盖 `h + 2` 行价格。

### 3.4 滑动窗口与分层采样

用户提供的是约 45 万行的分钟级训练数据。Phase I 的目标不是把全部滑窗候选都作为训练样本，而是先用滑动窗口从这 45 万行中枚举候选 horizon，再通过分层采样选出最终 `30000` 个 horizon 进入 DP demonstration 和 VQ 训练。

候选窗口生成:

```text
if reward_alignment == paper_formula:
    window_start = 0, 1, 2, ..., num_rows - h - 1
    window_end = window_start + h - 1
    last_execution_row = window_start + h - 1
    last_markout_row = window_start + h

if reward_alignment == next_row_execution:
    window_start = 0, 1, 2, ..., num_rows - h - 2
    window_end = window_start + h - 1
    last_execution_row = window_start + h
    last_markout_row = window_start + h + 1
```

当 `h=72` 且训练集约 `450000` 行时:

```text
paper_formula:       450000 - 72 = 449928
next_row_execution:  450000 - 72 - 1 = 449927
```

`paper_formula` 用于复现论文 reward 公式和实验可比性；`next_row_execution` 用于更保守的在线可执行评估。二者不能混用，产物中必须记录 `reward_alignment`。随后只从这些候选中采样 `30000` 个作为最终训练 horizon。

分层采样流程:

1. 对所有候选窗口计算轻量级窗口统计，不先运行完整 VQ 训练。
2. 根据窗口统计生成 strata label。
3. 在每个 strata 内按比例或配额采样，最终选出 `num_demos=30000` 个 horizon。
4. 只对这 `30000` 个被采样 horizon 运行 Single-trade DP，得到 `actions/rewards`。

建议第一版使用以下分层维度:

| 维度 | 计算方式 | 分桶 |
| --- | --- | --- |
| horizon return | `(close[t+h] - close[t]) / close[t]` | down / flat / up |
| realized volatility | horizon 内 1 分钟收益标准差 | low / mid / high |
| draw pattern | `max_drawup` 与 `max_drawdown` 的相对强弱 | upward / downward / mixed |

组合后得到 strata，例如:

```text
return_bin=up, vol_bin=high, pattern_bin=mixed
```

采样策略:

- 默认 `stratified_uniform`: 每个非空 strata 尽量采样相同数量，剩余额度按 strata 大小补齐。
- 可选 `stratified_proportional`: 按每个 strata 的候选窗口数量等比例采样。
- 对 `return_bin=flat` 且 `vol_bin=low` 的 strata 设置采样上限，避免 DP 全 flat 样本过多。
- 若初次 DP 标注后的 `no_trade_ratio` 超过阈值，触发二次补采样: 从非 flat 或中高波动 strata 中补足被过滤的 horizon。
- 设置 `min_gap_between_samples` 和 `max_overlap_ratio` 控制相邻采样窗口重叠度；默认在采样阶段强制执行，避免训练样本时间自相关过高。
- 对 train/val/test 文件边界执行 `split_boundary_embargo` 检查，避免 validation horizon 与 train 末尾时间过近导致验证指标虚高。
- 同一批次必须固定 `seed`，并把被采样的 `window_start` 保存到 `window_index_train.parquet`。
- val/test 不参与 VQ 训练采样；需要标签时可按固定 stride 枚举，或使用同一分层策略生成评估窗口索引。

采样健康检查配置:

```yaml
sampling_health:
  max_no_trade_ratio: 0.25
  flat_low_vol_max_ratio: 0.15
  min_gap_between_samples: 36   # h=72 时取 h/2，最多允许 50% 重叠
  max_overlap_ratio: 0.5
  split_boundary_embargo: 72    # train/val/test 边界至少间隔一个 horizon
  warn_only: false
  allow_overlap_relaxation: false
```

`window_overlap_ratio` 定义为相邻已采样窗口的平均重叠比例，单对窗口可按 `max(0, h - gap) / h` 计算。`h=72` 时，`min_gap_between_samples=36` 对应最多 36 分钟重叠，即 50% overlap；不再使用 `min_gap=12` / `max_overlap=0.85` 作为默认值，因为 83%-85% 重叠会显著提高样本时间自相关。

`StratifiedWindowSampler` 应先在每个 strata 内按 `min_gap_between_samples` 去相关采样；若某些 strata 因约束过强导致采不满，默认应从其他 strata 补齐，而不是自动放宽重叠约束。只有显式设置 `allow_overlap_relaxation=true` 时，才允许降低 `min_gap_between_samples`，并必须把放宽后的实际值、受影响 strata 和原因写入 `phase1_report.json`。

`phase1_evaluator.py` 必须在每次数据构建后输出 `sampling_health_warnings`。当 `warn_only=false` 时，若 `window_overlap_ratio > max_overlap_ratio`、`min_sample_gap < min_gap_between_samples` 或 `split_boundary_gap < split_boundary_embargo`，数据构建应失败；当 `warn_only=true` 时，训练可以继续，但报告必须明确提示泛化风险和建议调整项。

## 4. 目录与模块设计

随着采样健康、reward 对齐、真实 replay、codebook 诊断、no-trade 容量和 horizon 边界衔接等设计加入，Phase I 不应继续用少数大文件承载全部逻辑。目录设计采用“编排层薄、能力模块细”的原则:

- `dataset.py` 只保留 PyTorch Dataset/DataLoader 适配，不负责采样策略和 DP 生成。
- `phase1_evaluator.py` 只做评估编排，不直接实现每类指标。
- `vq_archetype.py` 只组装模型，不把 encoder input adapter、quantizer 更新、loss 全部写在一个文件里。
- reward 对齐、成本、env replay 必须是共享基础设施，DP teacher 和 student replay 不各自实现一套。

建议模块结构如下，和 `run_pipeline.sh` 中预留的 `scripts/train_phase1.py` 对齐:

```text
scripts/train_phase1.py

src/config/phase1_config.py

src/data/market_reader.py
src/data/schema.py
src/data/window_indexer.py
src/data/stratified_sampler.py
src/data/sampling_health.py
src/data/horizon_builder.py
src/data/demo_store.py
src/data/dataset.py

src/planners/single_trade_dp.py
src/planners/demo_generator.py

src/models/vq_archetype.py
src/models/encoder_inputs.py
src/models/vector_quantizer.py
src/models/vq_losses.py

src/trainers/phase1_trainer.py
src/trainers/phase1_checkpoint.py

src/evaluation/phase1_evaluator.py
src/evaluation/phase1_replay.py
src/evaluation/phase1_metrics.py
src/evaluation/action_metrics.py
src/evaluation/risk_metrics.py
src/evaluation/archetype_diagnostics.py
src/evaluation/behavior_diagnostics.py
src/evaluation/code_stability.py
src/evaluation/phase1_report.py

src/envs/trading_env.py
src/envs/reward_alignment.py

src/trading/cost_model.py

src/utils/io.py
```

### 4.1 `scripts/train_phase1.py`

命令行入口，负责解析参数、加载配置、启动训练。

建议 CLI:

```bash
python scripts/train_phase1.py \
  --pair AL \
  --train-batch-id batch_001 \
  --train-file data/AL/train.parquet \
  --val-file data/AL/val.parquet \
  --test-file data/AL/test.parquet \
  --horizon 72 \
  --window-stride 1 \
  --sampling-strategy stratified_uniform \
  --num-demos 30000 \
  --num-archetypes 10 \
  --epochs 100 \
  --seed 42
```

### 4.2 `src/config/phase1_config.py`

集中管理第一阶段配置:

- 数据路径和输出路径
- train/val/test 三个输入文件路径
- horizon 长度 `h`
- 滑动窗口参数: `window_stride`, `num_demos`, `sampling_strategy`, `strata_bins`
- 采样健康检查参数: `max_no_trade_ratio`, `flat_low_vol_max_ratio`, `min_gap_between_samples`, `max_overlap_ratio`, `split_boundary_embargo`, `allow_overlap_relaxation`
- DP/交易成本参数: `gamma`, `reward_alignment`, `commission_rate`, `slippage_model=lob_depth`, `execution_lag`, `max_position`, `cost_model`。其中 `execution_lag` 只在 `reward_alignment=next_row_execution` 时作为成交延迟语义使用；`paper_formula` 下行号映射固定为 `execution_row=t, markout_row=t+1`。
- VQ 参数: `hidden_dim=128`, `code_dim=16`, `num_codes=10`, `beta0=0.25`, `encoder_input`, `codebook.init_method`, `codebook.update_method`, `usage_regularization_weight`, `dead_code_restart`
- 训练参数: `batch_size`, `lr`, `epochs`, `seed`, `device`
- checkpoint 参数: `save_every`, `selection_metric`, `selection_mode`, `early_stopping_patience`

### 4.3 `src/data/*`

数据层按职责拆分，避免 `dataset.py` 变成读文件、滑窗、采样、健康检查和 demo 构造的大杂烩。

| 模块 | 职责 | 主要输出 |
| --- | --- | --- |
| `market_reader.py` | 直接读取 train/val/test 三个数据文件，保持外部特征原样 | `DataFrame` |
| `schema.py` | 校验输入 schema，识别时间列、价格列、盘口列、状态特征列 | `input_schema.json` |
| `window_indexer.py` | 按 `reward_alignment` 枚举候选 horizon，计算 `last_execution_row/last_markout_row` | `window_index_*.parquet` |
| `stratified_sampler.py` | 按 strata、no-trade 控制和 `min_gap_between_samples` 做去相关采样 | sampled window index |
| `sampling_health.py` | 计算 `window_overlap_ratio`、`split_boundary_gap`、采样 warning/fail 条件 | sampling health report |
| `horizon_builder.py` | 根据窗口索引切出 `states/prices/execution_books/meta` | horizon records |
| `demo_store.py` | 保存和加载 DP demonstrations、horizon labels、诊断字段 | `demos_train.parquet` / labels |
| `dataset.py` | 只负责 PyTorch `Dataset` / `DataLoader` 适配 | tensors for training |

核心类建议:

```python
class MarketFileReader:
    def read(self, path): ...

class InputSchemaValidator:
    def validate(self, frame): ...

class SlidingWindowIndexer:
    def enumerate(self, num_rows, horizon, stride, reward_alignment): ...

class StratifiedWindowSampler:
    def sample(self, window_index, num_samples, strategy, seed): ...

class SamplingHealthChecker:
    def check(self, sampled_windows, split_boundaries, config): ...

class HorizonBuilder:
    def build(self, frame, window_index): ...

class Phase1DemoStore:
    def save_demos(self, demos): ...
    def load_demos(self, path): ...

class Phase1DemoDataset(torch.utils.data.Dataset):
    def __getitem__(self, idx): ...
```

边界约束:

- `dataset.py` 不调用 DP，不执行分层采样，不计算 schema。
- `stratified_sampler.py` 不读取原始文件，只接收已经生成的 window index 和 strata。
- `sampling_health.py` 必须能在数据构建后独立重跑，用于审计已有产物。

### 4.4 `src/planners/*`

Planner 层负责生成离线 teacher demonstration，不参与 Phase II/III 在线推理。

| 模块 | 职责 |
| --- | --- |
| `single_trade_dp.py` | 实现论文 Algorithm 1 的 single-trade DP |
| `demo_generator.py` | 批量调用 DP，为 sampled horizons 生成 actions/rewards/DP metadata |
| `src/envs/reward_alignment.py` | 提供 `paper_formula` / `next_row_execution` 的统一行号映射，供 DP、env、replay 共用 |

`SingleTradeDPPlanner` 输入:

- `prices`: `[h + 1]` 或 `[h + 2]`，由 `reward_alignment` 决定
- `order_books`: 至少包含成交行的 ask1-ask5/bid1-bid5 价量
- `actions`: `[0, 1, 2]`
- `gamma`
- `commission_rate`
- `slippage_model`
- `reward_alignment`
- `execution_lag`: 仅 `next_row_execution` 模式使用；`paper_formula` 下不应覆盖论文公式行号对齐
- `max_position`
- `cost_model`

输出:

- `actions`: `[h]`
- `rewards`: `[h]`
- `total_return`
- `num_switches`
- `is_no_trade`

核心类建议:

```python
class RewardAlignment:
    def rows(self, decision_row): ...

class SingleTradeDPPlanner:
    def plan(self, horizon): ...

class Phase1DemoGenerator:
    def generate(self, sampled_horizons): ...
```

边界约束:

- `single_trade_dp.py` 只实现单个 horizon 的 DP，不负责批量 IO。
- `demo_generator.py` 负责断点续跑、进度记录和 demo metadata，不实现 DP 转移公式。
- DP reward 必须通过 `TradingEnv` 或共享 reward/cost 接口计算，不能在 planner 内另写一套手续费和滑点逻辑。

### 4.5 `src/models/*`

模型层拆分为可测试组件:

| 模块 | 职责 |
| --- | --- |
| `encoder_inputs.py` | state/action/reward 三路 input adapter、reward normalizer 应用、输入健康统计 |
| `vector_quantizer.py` | codebook 初始化、nearest-neighbor quantization、gradient/EMA 更新、dead-code restart 接口 |
| `vq_losses.py` | reconstruction loss、VQ loss、commitment loss、usage regularization |
| `vq_archetype.py` | 组装 `ArchetypeEncoder`、`VectorQuantizer`、`ArchetypeDecoder`、`VQArchetypeModel` |

输入 demonstration tuple:

$$
\tau=(\mathbf{s}_{demo}, \mathbf{a}_{demo}, \mathbf{r}_{demo})
$$

输出:

- `logits`: `[batch, h, 3]`
- `z_e`: `[batch, code_dim]`
- `z_q`: `[batch, code_dim]`
- `code_id`: `[batch]`
- `vq_loss`
- `commitment_loss`

核心类建议:

```python
class EncoderInputAdapter:
    def forward(self, states, actions, rewards): ...

class RewardNormalizer:
    def fit_train(self, rewards): ...
    def transform(self, rewards): ...

class VectorQuantizer:
    def quantize(self, z_e): ...
    def update_codebook(self, z_e, code_id): ...

class Phase1Loss:
    def forward(self, outputs, batch): ...

class VQArchetypeModel:
    def forward(self, states, actions, rewards): ...
```

边界约束:

- `encoder_inputs.py` 不读取数据文件，只接收 tensor。
- `vector_quantizer.py` 必须显式支持 `update_method=gradient|ema`，并暴露 code usage stats。
- `vq_archetype.py` 不负责 checkpoint 选择、不生成 labels。

### 4.6 `src/trainers/phase1_trainer.py`

职责:

- 编排数据构建、demo 生成、模型训练、评估、checkpoint 和产物导出。
- 调用 `Phase1DemoGenerator` 生成或加载 `demos_train.parquet`。
- 训练 VQ encoder-decoder。
- 每个 epoch 结束后调用 evaluator 在 validation horizon 上计算指标。
- 根据 `selection_metric` 调用 checkpoint manager 保存 `best_vq_model.pt`。
- 保存 `last_vq_model.pt` 和可选中间 checkpoint。
- 用 best checkpoint 生成 horizon code labels。
- 输出 `phase1_report.json`。

边界约束:

- trainer 不直接计算复杂指标，只消费 `Phase1Evaluator` 输出。
- trainer 不直接写 checkpoint 文件，统一交给 `Phase1CheckpointManager`。
- trainer 不直接写 report 文件，统一交给 `Phase1ReportWriter`。

### 4.7 `src/evaluation/phase1_evaluator.py`

职责:

- 在 train/val horizon 上评估 VQ encoder-decoder。
- 计算 checkpoint 选择所需指标。
- 检查 codebook 是否塌缩。
- 调用 `Phase1ReplayEvaluator` 获取真实收益指标，但不直接实现交易 replay 逻辑。
- 汇总风险调整收益、per-archetype、切换点、per-class action、archetype 可区分性、decoder 行为多样性、horizon 边界衔接、epoch 稳定性和 DP teacher 质量诊断。
- 输出 epoch-level metrics，供 trainer 和 checkpoint manager 使用。

评估层按指标域拆分，`phase1_evaluator.py` 只做调度和聚合:

| 模块 | 职责 |
| --- | --- |
| `phase1_evaluator.py` | 调用模型推理、收集 batch outputs、调度各类 metric/replay、形成 epoch metrics |
| `phase1_replay.py` | student/teacher replay、边界换仓 replay |
| `phase1_metrics.py` | 通用指标门面和组合指标计算 |
| `action_metrics.py` | reconstruction、per-class precision/recall、confusion matrix、switch metrics |
| `risk_metrics.py` | Sharpe、Sortino、MDD、Calmar、equity curve 统计 |
| `archetype_diagnostics.py` | per-code return/win/no-trade/switch distribution |
| `behavior_diagnostics.py` | action entropy、inter-code action diversity、decoder sensitivity |
| `code_stability.py` | epoch code stability、Hungarian matched stability |
| `phase1_report.py` | `phase1_report.json`、诊断 JSON/parquet 的写入和 schema 校验 |

边界约束:

- `phase1_evaluator.py` 不实现手续费、滑点、成交逻辑。
- `phase1_evaluator.py` 不直接写 report 文件。
- 各 metric 模块必须是纯函数或轻状态对象，便于离线重算和单元测试。

建议指标:

| 指标 | 说明 | 用途 |
| --- | --- | --- |
| `val_reconstruction_accuracy` | validation action 重构准确率 | 只作为辅助指标，不能单独决定 best |
| `val_weighted_reconstruction_accuracy` | 按类别权重加权后的 action 重构准确率 | 缓解 flat 类占比过高导致的虚高 accuracy |
| `val_non_flat_accuracy` | short/long 非 flat timestep 的重构准确率 | 检查模型是否真的学到交易动作 |
| `val_cross_entropy` | validation action 重构 CE loss | 辅助判断过拟合，越低越好 |
| `val_perplexity` | validation codebook perplexity | 判断 code 使用是否健康 |
| `code_usage_ratio` | 被使用 code 数 / `K` | 低于阈值说明 codebook collapse |
| `single_trade_consistency_rate` | decoder 输出动作满足单次切换约束的比例 | 衡量 decoder 是否学到 demonstration 结构 |
| `val_student_online_net_return` | checkpoint 在 validation horizon 上按因果方式推理动作后 replay 的净收益 | 衡量学生模型在线执行能赚多少 |
| `val_dp_teacher_net_return` | DP teacher 在同一批 validation horizon 上的 hindsight 净收益 | 作为老师上限 |
| `val_dp_teacher_sharpe` | DP teacher 在 validation replay step return 上的 Sharpe | 判断老师本身的风险调整收益质量 |
| `val_dp_teacher_profitable_ratio` | DP teacher 正收益 horizon 占比 | 判断老师标签中真实可交易机会的比例 |
| `val_return_capture_ratio` | `student_net_return / max(abs(dp_teacher_net_return), eps)` | 衡量学生学到老师多少收益能力 |
| `val_regret_to_dp` | `dp_teacher_net_return - student_net_return` | 衡量学生与老师的收益差距 |
| `val_cost_paid` | validation replay 中手续费、滑点、成交成本总和 | 检查收益是否真实扣成本 |
| `val_sharpe_ratio` | validation replay step return 的年化 Sharpe | 风险调整收益，防止只看净收益 |
| `val_sortino_ratio` | 只惩罚下行波动的年化 Sortino | 更适合交易收益序列 |
| `val_max_drawdown` | validation equity curve 最大回撤 | 核心风险约束 |
| `val_calmar_ratio` | 年化收益 / 最大回撤 | 高收益高回撤 checkpoint 的惩罚指标 |
| `per_code_*` | 每个 archetype 的收益、胜率、样本数、切换点分布 | 判断 archetype 是否有意义 |
| `per_code_no_trade_ratio` | 每个 code 内 no-trade horizon 占比 | 判断 no-trade 样本是否挤占少数 code 容量 |
| `switch_*` | 切换点 recall、方向准确率、切换时机误差 | 检查单次交易核心行为是否学到 |
| `action_precision_recall_per_class` | short/flat/long 精确率和召回率 | 检查类别不均衡下的动作质量 |
| `confusion_matrix` | 三分类动作混淆矩阵 | 定位 short/flat/long 混淆 |
| `inter_code_distance` | codebook 向量间平均距离 | 判断 archetype 是否足够区分 |
| `silhouette_score` | latent space 中按 code_id 分组的轮廓系数 | 判断聚类质量 |
| `per_code_action_entropy` | 每个 code 解码出的 action 分布熵 | 判断 archetype 行为是否退化为单一动作 |
| `inter_code_action_diversity` | 固定同一批 states 后，不同 code 输出 action 序列的平均 Hamming 距离或 `1-DTW` 距离 | 判断不同 code 是否产生不同策略行为 |
| `decoder_sensitivity_to_code` | 固定 states，只替换 code_id 后 decoder logits/prob/action 的变化幅度 | 判断 decoder 是否真正使用 `z_q` |
| `horizon_boundary_turnover_cost` | 相邻 horizon 拼接时，前一 horizon 末仓位到后一 horizon 初始动作的换仓成本估算 | 衡量 Phase II 独立选 horizon 时的边界成本 |
| `horizon_boundary_position_consistency` | 相邻 horizon 边界仓位一致的比例 | 检查 archetype 序列是否容易在边界频繁反手 |
| `epoch_code_stability` | best epoch 与 last epoch 在同一 validation horizon 上的 code 分配一致率 | 判断 codebook 和标签是否仍在漂移 |

风险指标按分钟级 step return 计算，默认年化因子:

```yaml
risk_metrics:
  annualization_factor: 525600  # 365 * 24 * 60
```

默认 checkpoint 选择不再使用普通 `val_reconstruction_accuracy` 作为主指标。Phase I 的核心是学到“何时切换、切换到什么、交易后风险收益如何”，因此使用组合指标:

```yaml
selection_metric: phase1_composite_score
selection_mode: max
min_code_usage_ratio: 0.7
metric_weights:
  switch_point_recall: 0.30
  switch_direction_accuracy: 0.20
  val_weighted_reconstruction_accuracy: 0.20
  val_return_capture_ratio: 0.20
  val_sharpe_ratio: 0.10
risk_guardrails:
  max_drawdown: 0.2
  min_sharpe_ratio: 0.0
behavior_guardrails:
  min_inter_code_action_diversity: 0.15
  min_decoder_sensitivity_to_code: 0.05
  min_epoch_code_stability: 0.8
teacher_quality_guardrails:
  min_dp_teacher_profitable_ratio: 0.3
```

`val_reconstruction_accuracy` 仍然记录，但只能作为 sanity check。若 checkpoint 只是 flat 类预测准确率高，而 `switch_point_recall`、`switch_direction_accuracy` 或 `val_non_flat_accuracy` 较低，则不能成为 best。

如果 `phase1_composite_score` 更高但 `code_usage_ratio < min_code_usage_ratio`，该 checkpoint 不应成为 best，应继续训练或触发告警。

若 checkpoint 的 `val_max_drawdown` 超过 `risk_guardrails.max_drawdown`，或 `val_sharpe_ratio` 低于 `risk_guardrails.min_sharpe_ratio`，即使净收益较高，也不能直接成为 best，必须在 `checkpoint_manifest.json` 中记录风险拒绝原因。

若 `inter_code_action_diversity` 或 `decoder_sensitivity_to_code` 低于阈值，说明 codebook 虽然可能在 latent space 中分开，但 decoder 行为没有充分利用 code，应在 `phase1_report.json` 中写入 warning。若 `val_dp_teacher_profitable_ratio` 很低，则 `val_return_capture_ratio` 不应单独解读为学生学得好，因为老师本身没有足够高质量的可交易收益。

Online-style replay 规则:

1. 对每个 validation horizon，先用同一套 Single-trade DP 和 `cost_config` 离线生成 teacher action 与 `dp_teacher_net_return`。
2. 对同一 horizon，用当前 checkpoint 的 encoder 得到 `code_id`。这是 Phase I 的离线标签评估，不代表 Phase II 线上 selector。
3. 冻结 decoder，按分钟因果执行: 第 `\tau` 步只能输入 `s_t, ..., s_\tau` 和 `codebook[code_id]`，输出当前动作。
4. 用输出动作序列按 `cost_config.reward_alignment` replay，成交行和 markout 行必须与 DP teacher 完全一致:
   - `paper_formula`: 第 `\tau` 步动作使用 `\tau` 行 bid/ask 盘口逐档成交，并用 `\tau+1` 行 mark price 结算持仓收益。
   - `next_row_execution`: 第 `\tau` 步动作使用 `\tau+1` 行 bid/ask 盘口逐档成交，并用 `\tau+2` 行 mark price 结算持仓收益。
5. 汇总所有 validation horizon，写入 epoch metrics 和 `checkpoint_manifest.json`。

该验证回答的问题是: 在给定老师压缩出的 archetype code 后，学生 decoder 以因果方式复现老师交易能力的程度。使用 `paper_formula` 时，它是论文公式可比的 Phase I validation replay；使用 `next_row_execution` 时，它是更保守的部署仿真诊断，不能直接与论文 reward 或实验结果比较。它不能替代 Phase II selector 的线上评估，因为 Phase II 还需要学习如何在 horizon 起点选择 `code_id`。

Horizon 边界衔接诊断:

Phase II selector 会按 horizon 选择 archetype，但真实账户持仓不会在 horizon 边界自动归零。因此边界换仓成本应由下一段 horizon 的第一步 action 承担: env reset 新 horizon 时必须接收上一段结束仓位 `prev_terminal_position`，第一步 target position 与该仓位的差额通过同一 `CostModel` 成交并扣费。Phase I validation 可在按时间排序的非重叠或指定 stride 的 validation horizons 上模拟这种拼接，输出:

- `horizon_boundary_turnover_cost`: 边界处从上一 horizon 末仓位切到下一 horizon 首个 target position 的总成本或均值。
- `horizon_boundary_position_consistency`: `prev_terminal_position == next_initial_target_position` 的比例。

该指标不改变论文 Phase I 的 VQ 训练目标，但会提前暴露“单个 horizon 内表现好、拼起来成本很高”的 archetype 组合风险。

必须明确边界限制: Phase I 的 DP demonstration 按论文口径从 flat 初始状态生成，而真实 Phase II/III 执行时账户可能从上一 horizon 继承 long/short/flat 任一仓位。因此 Phase I 只能报告边界风险，不能完全解决该问题。Phase II 设计必须让 environment 支持 `initial_position != flat`，并在每个新 horizon 的第一步把 `prev_terminal_position -> first_target_position` 的换仓成本计入 reward；否则 selector 逐 horizon 独立选 archetype 会系统性低估跨 horizon 反手成本。

### 4.8 `src/evaluation/phase1_replay.py`

职责:

- 独立实现 Phase I checkpoint 的 validation replay。
- 对同一批 validation horizon 分别计算 DP teacher 净收益和 student online 净收益。
- 在按时间排序的 validation horizons 上模拟 horizon 拼接，估算边界换仓成本和仓位一致性。
- 调用 `TradingEnv` 逐步执行动作；env 内部调用 `CostModel` 扣除手续费和盘口深度滑点。
- 输出 student/teacher 收益差距指标，交给 `phase1_evaluator.py` 汇总。

`phase1_replay.py` 是 Phase I 的 replay 编排器，不直接实现 env 或 cost:

| Replay 类型 | 输入 | 输出 |
| --- | --- | --- |
| teacher replay | validation horizon + DP planner/env | `dp_teacher_net_return`、teacher step returns、teacher quality |
| student causal replay | decoder + codebook + code_id + horizon | `student_online_net_return`、student step returns、cost paid |
| boundary replay | 按时间排序的 horizons + decoded actions | `horizon_boundary_turnover_cost`、`horizon_boundary_position_consistency` |

核心类建议:

```python
class Phase1ReplayEvaluator:
    def evaluate_checkpoint(self, model, val_dataset, env_factory): ...
    def replay_student_online(self, decoder, codebook, horizon, code_id, env): ...
    def replay_dp_teacher(self, horizon, env): ...
    def evaluate_horizon_boundaries(self, ordered_horizons, decoded_actions, env_factory): ...
```

模块边界:

| 模块 | 负责 | 不负责 |
| --- | --- | --- |
| `phase1_evaluator.py` | 重构指标、codebook 指标、汇总 replay 指标、生成 epoch metrics | 逐笔成交、盘口滑点、收益 replay |
| `phase1_replay.py` | validation online replay、student/teacher 净收益、regret/capture ratio、horizon 边界换仓成本 | checkpoint 保存、best 选择、底层成交成本计算 |
| `envs/trading_env.py` | 状态推进、动作执行、position/cash/nav/reward/done 统一语义 | 模型训练、checkpoint 选择 |
| `trading/cost_model.py` | 盘口逐档成交、手续费、滑点和未成交处理 | 模型推理、指标汇总 |

### 4.9 `src/envs/trading_env.py`

职责:

- 提供统一的分钟级交易环境，供 DP、student replay、后续 Phase II/III 复用。
- 管理当前步索引、持仓、现金、净值、成交记录和 episode 结束条件。
- 执行 action 到目标仓位的转换，并按 `reward_alignment` 决定 `execution_row` 与 `markout_row`。
- 返回扣除手续费和盘口滑点后的 step reward。
- 保证所有阶段使用同一套 reward 和成交语义。

核心接口建议:

```python
class TradingEnv:
    def reset(self, horizon, initial_position=0): ...
    def step(self, action): ...
    def current_observation(self): ...
    def replay(self, actions): ...
```

step 语义:

| 输入/输出 | 说明 |
| --- | --- |
| `action` | 目标仓位动作 `{0,1,2}`，映射为 short/flat/long |
| `observation` | 当前可见状态，不包含未来行 |
| `reward` | 当前 action 在 `execution_row` 成交并 markout 后的净收益 |
| `done` | horizon 结束或成交深度不足触发终止/保持仓位规则 |
| `info` | fee、slippage、fill_price、filled_qty、position、nav、execution_row、markout_row |

Phase I 使用方式:

- `SingleTradeDPPlanner` 可以在内部调用 env 的 reward/cost 计算接口构建 DP transition reward。
- `Phase1ReplayEvaluator` 用 env 逐步执行 student decoder 输出的 action，得到与配置对齐的 validation net return。
- DP teacher 和 student replay 必须使用相同 env 配置。
- Phase I DP 默认 `initial_position=0`，但 env 必须支持非 flat 初始仓位，供 boundary replay 和 Phase II/III 复用。

### 4.10 `src/trading/cost_model.py`

职责:

- 使用 `execution_row` 的 bid/ask 五档盘口估算成交均价。
- 计算手续费、盘口滑点和实际成交量。
- 对五档深度不足执行 `insufficient_depth_policy`。
- 为 DP teacher、student replay 和后续阶段提供统一成本接口。

核心类建议:

```python
class LobDepthCostModel:
    def execute(self, prev_position, target_position, execution_book, mark_price): ...
```

### 4.11 `src/evaluation/phase1_metrics.py`

职责:

- 作为 Phase I 专用指标的门面模块，聚合各子指标模块。
- 为 `phase1_evaluator.py` 和 `phase1_replay.py` 提供可复用的 metric 实现。
- 避免 Phase II/III 的 RL 指标和 Phase I 的 VQ/replay 指标混在一个通用 `metrics.py` 中。

实现上可以将具体函数拆到 `action_metrics.py`、`risk_metrics.py`、`archetype_diagnostics.py`、`behavior_diagnostics.py`、`code_stability.py`，再由 `phase1_metrics.py` 统一导出稳定 API。

指标范围:

| 指标类别 | 具体指标 |
| --- | --- |
| 重构指标 | `reconstruction_accuracy`、`weighted_reconstruction_accuracy`、`non_flat_accuracy`、`cross_entropy` |
| VQ 指标 | `code_usage_ratio`、`perplexity`、`vq_loss`、`commitment_loss` |
| encoder 输入健康指标 | `reward_norm_clip_ratio`、`encoder_input_modality_norms`、`reward_embedding_norm_ratio` |
| 行为结构指标 | `single_trade_consistency_rate`、`no_trade_ratio` |
| replay 收益指标 | `student_online_net_return`、`dp_teacher_net_return`、`return_capture_ratio`、`regret_to_dp`、`cost_paid` |
| 风险调整收益指标 | `sharpe_ratio`、`sortino_ratio`、`max_drawdown`、`calmar_ratio` |
| per-archetype 指标 | `per_code_avg_return`、`per_code_win_rate`、`per_code_count`、`per_code_no_trade_ratio`、`per_code_switch_point_distribution` |
| 切换点指标 | `switch_point_recall`、`switch_direction_accuracy`、`switch_timing_error_mean`、`switch_timing_error_distribution` |
| per-class action 指标 | `action_precision_recall_per_class`、`confusion_matrix` |
| archetype 可区分性指标 | `inter_code_distance`、`silhouette_score` |
| archetype 行为多样性指标 | `per_code_action_entropy`、`inter_code_action_diversity`、`decoder_sensitivity_to_code` |
| horizon 边界衔接指标 | `horizon_boundary_turnover_cost`、`horizon_boundary_position_consistency` |
| epoch 稳定性指标 | `epoch_code_stability`、`epoch_code_stability_matched`、`code_assignment_drift_warning` |
| DP teacher 质量指标 | `val_dp_teacher_sharpe`、`val_dp_teacher_profitable_ratio`、`dp_teacher_return_distribution` |
| 采样健康指标 | `strata_distribution`、`num_candidate_windows`、`num_sampled_horizons`、`flat_low_vol_sample_ratio`、`window_overlap_ratio`、`min_sample_gap`、`mean_sample_gap`、`split_boundary_gap`、`effective_min_gap_between_samples`、`sampling_health_warnings` |

核心函数建议:

```python
def reconstruction_accuracy(logits, actions): ...
def weighted_reconstruction_accuracy(logits, actions, class_weights): ...
def non_flat_accuracy(logits, actions): ...
def codebook_perplexity(code_ids, num_codes): ...
def encoder_input_health(state_emb, action_emb, reward_emb, reward_norm): ...
def single_trade_consistency(actions): ...
def return_capture_ratio(student_return, teacher_return): ...
def regret_to_dp(student_return, teacher_return): ...
def sharpe_ratio(step_returns, annualization_factor): ...
def sortino_ratio(step_returns, annualization_factor): ...
def max_drawdown(equity_curve): ...
def calmar_ratio(annual_return, max_drawdown): ...
def per_code_summary(horizon_metrics, code_ids): ...
def switch_metrics(dp_actions, pred_actions): ...
def action_confusion_matrix(dp_actions, pred_actions): ...
def inter_code_distance(codebook): ...
def latent_silhouette_score(latents, code_ids): ...
def per_code_action_entropy(decoded_logits, code_ids): ...
def inter_code_action_diversity(decoded_actions_by_code, distance="hamming"): ...
def decoder_sensitivity_to_code(decoded_logits_by_code): ...
def horizon_boundary_metrics(boundary_positions, boundary_books, cost_model): ...
def epoch_code_stability(best_code_ids, last_code_ids): ...
def matched_epoch_code_stability(best_code_ids, last_code_ids, best_codebook, last_codebook): ...
def dp_teacher_quality(dp_horizon_returns, dp_step_returns, annualization_factor): ...
def window_overlap_ratio(sampled_windows, horizon): ...
def split_boundary_gap(train_frame, val_frame, test_frame): ...
def sampling_health_warnings(report, config): ...
def behavior_health_warnings(report, config): ...
def teacher_quality_warnings(report, config): ...
def boundary_health_warnings(report, config): ...
```

行为多样性指标必须固定同一批 validation `states`，分别用 `K` 个 code 解码，再比较输出动作或 logits。这样可以发现“codebook 向量距离很远，但 decoder 对 `z_q` 不敏感，输出几乎相同”的问题。`epoch_code_stability` 默认使用 code id 直接一致率；若出现 code id 交换，可额外用 codebook 距离做 Hungarian matching 后输出 `epoch_code_stability_matched`，用于区分标签漂移和纯编号交换。

### 4.12 `src/trainers/phase1_checkpoint.py`

职责:

- 统一保存和加载 Phase I checkpoint。
- 比较当前 epoch 指标与历史 best。
- 原子写入 `best_vq_model.pt`、`last_vq_model.pt` 和 `checkpoints/epoch_*.pt`。
- 在 `checkpoint_manifest.json` 中记录每个 checkpoint 的 epoch、指标、路径、是否 best。
- 支持从 `last_vq_model.pt` 恢复训练。
- 避免和 Phase II/III 的 RL checkpoint 管理混用。

核心类建议:

```python
class Phase1CheckpointManager:
    def save_last(self, state, metrics): ...
    def save_periodic(self, state, metrics, epoch): ...
    def maybe_save_best(self, state, metrics): ...
    def load(self, path): ...
```

### 4.13 `src/evaluation/phase1_report.py`

职责:

- 统一写入 `phase1_report.json`、`checkpoint_manifest.json` 之外的诊断 JSON/parquet。
- 校验 report schema，确保新增指标不会只存在于内存 metrics 中。
- 汇总 data health、model health、replay health、codebook health、boundary health warning。
- 保存 report 生成时使用的 config hash、input schema hash 和 checkpoint path，便于审计。

核心类建议:

```python
class Phase1ReportWriter:
    def write_epoch_metrics(self, metrics, epoch): ...
    def write_final_report(self, summary): ...
    def write_diagnostics(self, diagnostics): ...
    def validate_schema(self, report): ...
```

边界约束:

- `phase1_report.py` 只负责序列化和 schema 校验，不重新计算指标。
- `phase1_checkpoint.py` 负责 checkpoint manifest；`phase1_report.py` 可以引用 manifest，但不决定 best checkpoint。

## 5. Single-trade DP 设计

### 5.1 状态定义

DP 状态为:

$$
(t, i, c)
$$

其中:

- `t`: horizon 内时间步。
- `i`: 当前 action，取值 `{0,1,2}`。
- `c`: 是否已经发生过一次动作切换，取值 `{0,1}`。

初始状态:

$$
i=1,\quad c=0
$$

即从 flat 开始。

### 5.2 转移约束

每个 horizon 最多只允许一次动作切换:

$$
c + \mathbf{1}[i \neq j] \leq 1
$$

这会过滤掉频繁开平仓的噪声轨迹，使 demonstration 更接近论文中“捕捉主要交易机会”的设定。

### 5.3 Reward 计算

DP demonstration 必须包含手续费和滑点交易成本。第一阶段训练、validation replay、checkpoint 评估都必须使用同一套 `cost_config`，不能用无成本收益训练模型，否则 archetype 会偏向过度交易。

action 到目标仓位的映射:

$$
P(a)=m(a-1)
$$

### 5.3.1 Reward 时间对齐

论文公式使用:

$$
r_t=P_t(p_{t+1}^{mark}-p_t^{mark})-O_t
$$

即 1-step mark price difference。为了避免工程实现中把论文复现和在线保守执行混在一起，本设计显式提供两种 reward alignment。

| 概念 | 行号 |
| --- | --- |
| `paper_formula` | `decision_row=t`, `execution_row=t`, `markout_row=t+1` |
| `next_row_execution` | `decision_row=t`, `execution_row=t+1`, `markout_row=t+2` |

`paper_formula` 与论文公式完全对齐，DP 找到的交易时机、reward 量级和论文实验更可比。`next_row_execution` 是保守在线执行模式，避免“使用第 t 行特征后又按第 t 行价格成交”的隐性泄漏，但它会把可交易收益窗口整体后移一行: action 在 `t+1` 成交，收益用 `p_{t+2}^{mark}-p_{t+1}^{mark}` 计算。因此该模式下的 DP 最优交易时机和 reward 量级不能直接与论文结果比较。

默认建议:

```yaml
reward_alignment: paper_formula      # 论文复现和公式对齐
# reward_alignment: next_row_execution  # 保守在线仿真
```

无论选择哪种模式，DP teacher、student online replay、checkpoint validation 和 Phase II/III 环境必须使用同一 `reward_alignment`。

逐步收益:

$$
r_k(i \rightarrow j)=P(j)(p_{markout}^{mark}-p_{exec}^{mark})-O_{exec}(i,j)
$$

其中 `paper_formula` 下 `p_exec = p_k` 且 `p_markout = p_{k+1}`；`next_row_execution` 下 `p_exec = p_{k+1}` 且 `p_markout = p_{k+2}`。

交易成本:

$$
O_{exec}(i,j)=fee_{exec}(i,j)+slippage_{exec}(i,j)
$$

其中:

$$
fee_{exec}(i,j)=\delta |P(j)-P(i)|p_{exec}^{mark}
$$

论文设置手续费:

$$
\delta=0.02\%
$$

滑点必须通过盘口深度算法计算，不使用 `fixed_bps`。默认 `slippage_model=lob_depth`，根据动作导致的仓位变化量 `\Delta P=P(j)-P(i)` 在 `execution_row` 的 bid/ask 五档盘口逐档成交:

- 增加多头或平空时，按 ask1 到 ask5 的价格和数量估算平均成交价。
- 增加空头或平多时，按 bid1 到 bid5 的价格和数量估算平均成交价。
- 若五档深度不足，默认 `insufficient_depth_policy=reject_transition`，即 DP 中该换仓转移不可选；student replay 中记录 `unfilled_transition` 并保持上一仓位。

成交滑点计算:

$$
p_{fill}=\frac{\sum_{\ell=1}^{5} q_{\ell}^{fill}p_{\ell}}{\sum_{\ell=1}^{5} q_{\ell}^{fill}}
$$

$$
slippage_{exec}=|\Delta P| \cdot |p_{fill}-p_{exec}^{mark}|
$$

其中 `p_exec_mark` 默认使用 `execution_row` 的 mid price，即 `(ask1_price + bid1_price) / 2`。

所有 DP teacher、student replay、checkpoint evaluator 必须通过同一个 `TradingEnv` 执行动作，并由 env 调用同一个 `CostModel`，保证老师收益和学生收益可比。

默认成本配置建议:

```yaml
cost_config:
  reward_alignment: paper_formula
  commission_rate: 0.0002
  slippage_model: lob_depth
  book_levels: 5
  mark_price: mid_price
  execution_lag: 0  # paper_formula 固定为同 row 成交；next_row_execution 时设为 1
  insufficient_depth_policy: reject_transition
```

约束:

- `fee_exec` 和 `slippage_exec` 在 DP 生成 demonstration 时必须启用。
- `rewards` 字段保存的是扣除手续费和滑点后的净收益。
- evaluator 的 `val_demo_return_replay` 必须使用同一份 `cost_config` 重新 replay。
- `phase1_config.yaml` 和 `phase1_report.json` 必须记录最终使用的 `cost_config`。

### 5.4 No-trade 样本处理

DP 可能发现全程 flat 的收益最高。No-trade 处理必须作为数据构建机制实现，而不是只在风险表中提示。

配置建议:

```yaml
no_trade_control:
  keep_no_trade: true
  max_no_trade_ratio: 0.25
  min_profit_gate: 0.0
  cap_flat_low_vol_strata: true
  flat_low_vol_max_ratio: 0.15
  resample_when_exceeded: true
```

处理流程:

1. 初始分层采样时，对 `return_bin=flat, vol_bin=low` 的 strata 设置配额上限。
2. 对采样出的 horizon 运行 DP，并标记 `is_no_trade`。
3. 计算 `no_trade_ratio = no_trade_count / num_demos`。
4. 若 `no_trade_ratio <= max_no_trade_ratio`，保留当前样本集。
5. 若 `no_trade_ratio > max_no_trade_ratio`:
   - 当 `keep_no_trade=True` 时，只保留不超过 `max_no_trade_ratio` 的 no-trade 样本，其余从非 flat 或中高波动 strata 中补采样。
   - 当 `keep_no_trade=False` 时，过滤 `total_return <= min_profit_gate` 或全 flat 样本，并从候选池补采样直到达到 `num_demos=30000`。
6. 所有被过滤、补采样和保留的样本数量写入 `phase1_report.json`。

论文文字强调 single trade，且原始表述可理解为每个 horizon 有一次主要交易机会。工程实现中，DP 转移约束保持“最多一次动作切换”，不允许多次交易；若最优解为全 flat，则视为 no-trade horizon。为严格复现论文口径，可设置 `keep_no_trade=False` 过滤 no-trade 样本；为真实交易训练，默认建议 `keep_no_trade=True` 保留负样本，并在报告中记录 `no_trade_ratio`。该配置不改变 Algorithm 1 的 DP 状态和转移形式，只决定是否把全 flat 样本纳入 $\mathcal{D}$。

No-trade archetype 容量监控:

保留 no-trade horizon 有助于让模型学会“不交易”场景，但如果大量 no-trade 样本集中落到 1-2 个 code，这些 code 会退化成纯 no-trade archetype，剩余 code 需要覆盖全部交易模式，容量可能不足。因此 `archetype_diagnostics.parquet` 和 `phase1_report.json` 必须输出:

- `per_code_no_trade_ratio`: 每个 code 内 `is_no_trade=true` 样本占比。
- `no_trade_code_concentration`: no-trade 样本在 top-1/top-2 code 中的集中度。
- `active_trade_code_count`: `per_code_no_trade_ratio` 低于阈值且样本数足够的交易型 code 数。

配置建议:

```yaml
no_trade_code_health:
  max_per_code_no_trade_ratio: 0.8
  max_top2_no_trade_concentration: 0.7
  min_active_trade_code_count: 6
```

若 no-trade 过度集中，应优先调整 `max_no_trade_ratio`、flat/低波动 strata 配额或 `min_profit_gate`；不建议通过手工指定 code 语义来干预 VQ，因为这会破坏无监督 archetype discovery 的设定。

## 6. VQ Encoder-Decoder 设计

### 6.1 Encoder

论文层面的每步输入仍然是:

```text
[state_t, action_embedding_t, reward_t]
```

但工程实现不能把三者 raw concat 后直接送入 LSTM，因为 `state_t`、`action_embedding_t` 和 `reward_t` 的量级来源不同。Encoder 必须使用分模态 input adapter，再合并成统一 hidden 表示。

默认结构:

```text
state_t
  -> state_adapter(Linear + LayerNorm + GELU)
action_t
  -> action_embedding + LayerNorm
reward_t
  -> reward_normalizer + reward_adapter(Linear/MLP + LayerNorm + GELU)
concat(state_emb, action_emb, reward_emb)
  -> fusion_layer(Linear + LayerNorm)
  -> LSTM(hidden_dim=128)
  -> last hidden
  -> MLP
  -> z_e(16)
```

输出:

$$
z_e \in R^{16}
$$

Reward normalization:

```yaml
encoder_input:
  state_adapter_dim: 96
  action_embedding_dim: 16
  reward_embedding_dim: 16
  fusion_dim: 128
  reward_normalization: train_reward_standard
  reward_clip_value: 5.0
```

`reward_normalization=train_reward_standard` 表示只在 train demonstration 的 `rewards` 上拟合:

$$
\hat r_t=\operatorname{clip}\left(\frac{r_t-\mu_r}{\max(\sigma_r,\epsilon)}, -c, c\right)
$$

其中 `\mu_r`、`\sigma_r` 和 `clip_value=c` 必须保存到 `reward_normalizer.json` 或 `phase1_config.yaml`，并原样用于 val/test。若 reward 分布重尾，可配置为 `train_reward_robust`，使用 median/MAD 替代 mean/std。

这一步属于 Phase I encoder 的模型输入适配，不属于市场特征工程: 不重新生成因子、不改写原始状态列，也不在 val/test 上拟合统计量。`states` 若已由外部数据文件标准化则直接进入 `state_adapter`；若状态量级差异较大，优先使用 `state_adapter` 内部的 `LayerNorm`，不要在本阶段重新拟合状态特征 scaler。

健康检查:

- `reward_norm_clip_ratio`: 被 clip 的 reward 比例，过高说明 reward 尺度或异常值需要检查。
- `encoder_input_modality_norms`: 记录 `state_emb/action_emb/reward_emb` 的平均 L2 norm。
- `reward_embedding_norm_ratio`: `reward_emb_norm / state_emb_norm`，过低说明 reward 信号仍可能被淹没，过高说明 reward 支配 encoder。

### 6.2 Vector Quantizer

codebook:

$$
\epsilon=\{e_0,\ldots,e_{K-1}\},\quad K=10
$$

最近邻量化:

$$
k=\arg\min_i \|z_e-e_i\|^2,\qquad z_q=e_k
$$

训练使用 straight-through estimator:

```python
z_q_st = z_e + (z_q - z_e).detach()
```

### 6.2.1 Codebook 初始化与更新

Codebook 初始化和更新方式必须配置化，并写入 `phase1_config.yaml`、`checkpoint_manifest.json` 和 `phase1_report.json`。

默认建议:

```yaml
codebook:
  init_method: kmeans_warmup
  kmeans_warmup_batches: 32
  update_method: ema
  ema_decay: 0.99
  ema_epsilon: 1.0e-5
```

初始化方式:

| `init_method` | 说明 | 适用场景 |
| --- | --- | --- |
| `random_normal` | 随机初始化 code embedding | 最简单，但早期更容易 dead code |
| `sample_encoder_outputs` | 用首批 `z_e` 样本初始化 codebook | 比随机稳定，成本低 |
| `kmeans_warmup` | 先用若干 batch 的 `z_e` 做 K-means，再初始化 codebook | 默认推荐，减少 collapse 和无效 code |

更新方式:

| `update_method` | 说明 | 与论文公式关系 |
| --- | --- | --- |
| `gradient` | codebook loss `||sg[z_e]-z_q||^2` 直接通过梯度更新 `e_k` | 严格贴近论文公式 (4) |
| `ema` | 使用 assigned `z_e` 的指数滑动均值更新 codebook | 工程稳定性更好，但属于实现增强，需记录配置 |

严格复现论文公式 (4) 时应设置:

```yaml
codebook:
  init_method: random_normal
  update_method: gradient
```

工程默认可使用 `kmeans_warmup + ema`，因为 30k horizon 且样本高度筛选后，EMA 对 codebook 使用率和 dead-code 风险更稳定。EMA 更新时，VQ loss 仍用于 encoder commitment 和 straight-through 路径，但 codebook embedding 本身由 EMA buffer 更新，不再由 optimizer 对 embedding 直接梯度更新。

EMA 更新:

$$
N_i \leftarrow \lambda N_i + (1-\lambda)n_i
$$

$$
m_i \leftarrow \lambda m_i + (1-\lambda)\sum_{b:k_b=i} z_{e,b}
$$

$$
e_i \leftarrow \frac{m_i}{N_i+\epsilon}
$$

其中 `\lambda=ema_decay`。若某个 code 在 `dead_code_patience` 个 epoch 内 `N_i` 过低，应触发 6.5 的 dead-code restart。

### 6.3 Decoder

每步输入:

```text
[state_t, z_q]
```

结构:

```text
input projection -> LSTM/MLP -> action logits(3)
```

输出:

```text
logits: [batch, h, 3]
base_actions: argmax(logits)
```

Phase II/III 中冻结 decoder，并使用选中的 codebook entry 解码 base action sequence。

重要约束: decoder 必须是 causal decoder。训练时可以把 `[batch, h, feature_dim]` 作为张量一次性送入模型以提高计算效率，但第 `\tau` 步 action logits 只能依赖 `s_t, ..., s_\tau` 和选中的 `z_q`，不能依赖 `s_{\tau+1}, ..., s_{t+h-1}`。

默认 decoder 架构:

```text
state projection
  -> concat repeated z_q at each timestep
  -> unidirectional LSTM(batch_first=True)
  -> timestep-wise MLP head
  -> action logits [batch, h, 3]
```

实现要求:

- LSTM 必须是 `bidirectional=False`。
- hidden state 按时间从 `0 -> h-1` 正向递推。
- `z_q` 可以在每个 timestep 作为条件输入，但不能由未来 state pooling 生成 decoder 输入。
- 不允许使用 `sequence_summary`、全 horizon pooling 或未来收益作为 decoder 输入。
- 若后续改用 Transformer/attention，必须使用严格 causal mask，使位置 `\tau` 只能 attend 到 `0..\tau`。
- 单元测试必须验证: 修改 `s_{\tau+1:h-1}` 不会改变第 `\tau` 步 logits。

### 6.4 Loss

总损失:

$$
L = L_{rec}
+ \|\operatorname{sg}[z_e]-z_q\|^2
+ \beta_0\|z_e-\operatorname{sg}[z_q]\|^2
$$

其中:

- `L_rec`: action reconstruction cross entropy。
- 第二项: codebook loss。
- 第三项: commitment loss。
- `beta0=0.25`。

### 6.5 Codebook Collapse 防护

Codebook collapse 指大部分样本长期落入少数 code，导致 `K=10` 个 archetype 退化成 1-2 个可用策略。Phase I 训练必须内置 collapse 监控和处理，不只在风险表中提示。

监控指标:

| 指标 | 告警阈值 | 处理 |
| --- | --- | --- |
| `code_usage_ratio` | `< 0.7` | checkpoint 不可成为 best，启动 usage regularization 或 dead-code restart |
| `perplexity` | 长期接近 `1.0` | 降低学习率、增加 commitment 调节、重启低频 code |
| `dead_code_count` | 连续多个 epoch 使用次数为 0 | 从高误差样本的 encoder 输出重置 code |
| `dominant_code_ratio` | 单个 code 占比 `> 0.5` | 提高 usage regularization 权重 |

训练配置:

```yaml
codebook_health:
  min_code_usage_ratio: 0.7
  max_dominant_code_ratio: 0.5
  usage_regularization_weight: 0.01
  dead_code_patience: 5
  dead_code_restart: true
  restart_source: high_reconstruction_error_samples
```

usage regularization 是工程稳定项，用于降低 codebook collapse 风险。严格复现论文公式 (4) 时应设置 `usage_regularization_weight=0`；当训练中出现 collapse 风险时，可启用该辅助项。它不改变 VQ encoder-decoder 架构和最近邻量化公式，只是在训练目标上增加可配置正则。

usage regularization 目标是鼓励 batch 内 code 分布接近均匀分布:

$$
L_{usage}=KL(U(K)\|p_{code})
$$

启用后训练总损失扩展为:

$$
L_{total}=L_{rec}+L_{vq}+\beta_0L_{commit}+\lambda_{usage}L_{usage}
$$

低频 code 重启策略:

1. 统计每个 code 连续未被使用的 epoch 数。
2. 若某 code 超过 `dead_code_patience` 未被使用，从当前 epoch 中 reconstruction error 最高的一批样本抽取 encoder 输出。
3. 用这些 `z_e` 均值或随机样本重置 dead code embedding。
4. 在 `phase1_report.json` 和 `checkpoint_manifest.json` 中记录 `dead_code_restarts`。

checkpoint 约束:

- `code_usage_ratio < 0.7` 的 checkpoint 不能成为 `best_vq_model.pt`。
- 若发生 dead-code restart，该 epoch 的 checkpoint 只可保存为 periodic/last，不直接成为 best；至少经过一个完整 validation epoch 后才能参与 best 选择。
- `phase1_report.json` 必须记录 `code_usage`、`perplexity`、`dominant_code_ratio`、`dead_code_count` 和 `dead_code_restarts`。

### 6.6 训练架构选择

Phase I 不采用 DQN、AC 或 PPO。第一阶段是离线 self-supervised / supervised representation learning，训练目标是重构 DP demonstration action，并学习离散 archetype codebook。

明确选择:

| 阶段 | 是否 RL | 训练架构 | 原因 |
| --- | --- | --- | --- |
| Phase I Archetype Discovery | 否 | LSTM VQ encoder-decoder + cross entropy + VQ loss | 目标是从 DP demonstration 中学习离散 archetype，不与环境交互 |
| Phase II Archetype Selection | 是 | PPO-style discrete Actor-Critic | action 是 `K` 个离散 archetype，论文目标包含 reward 与 KL/demo regularization，PPO 比 DQN 更适合加入策略正则和稳定更新 |
| Phase III Archetype Refinement | 是 | PPO-style discrete Actor-Critic | action 是 `{-1,0,1}`，需要处理 action mask、单次 override 和 episode early termination |

因此，当前 Phase I 实现只需要 VQ 模型训练器和 AdamW 优化器；DQN/AC/PPO 不进入 Phase I 训练代码。后续实现 Phase II/III 时，统一采用带 value head 的 PPO Actor-Critic，而不是 DQN。

### 6.7 未来信息与因果性边界

第一阶段确实会在离线训练中使用完整 `h=72` 的 horizon，但不同组件的使用边界不同:

| 组件 | 是否可看完整 72 行 | 原因 | 是否进入线上决策 |
| --- | --- | --- | --- |
| Single-trade DP planner | 可以 | DP 是 hindsight expert，用完整 horizon 生成 demonstration label | 否 |
| VQ encoder | 可以 | encoder 只负责把完整 demonstration trajectory 压缩成 archetype code label | 否 |
| VQ decoder | 不可以看未来 | decoder 会在 Phase II/III 被复用，必须满足在线决策因果性 | 是 |
| Phase II selector | 不可以看未来 | selector 在 horizon 起点只观察当前状态 `s_t` 并选择 archetype | 是 |

因此，Phase I 的“整段 horizon 输入”只允许发生在两个离线环节:

1. DP 用完整未来价格生成 hindsight demonstration。
2. encoder 用完整 demonstration 分配 archetype label。

这不等价于线上策略提前看到未来，因为 DP 和 encoder 都不会在 Phase II/III 推理时调用。真正会进入后续交易策略的是 `decoder.pt` 和 `codebook.pt`，其中 decoder 必须按时间步因果解码:

```text
at horizon start:
  selector observes s_t
  selector chooses code_id

for each minute tau in [t, t+h-1]:
  observe current state s_tau
  decoder receives current/past states and codebook[code_id]
  decoder outputs action_tau
```

离线训练或回测中可以为了 GPU 效率一次性传入 `[h, d]` 的张量，但必须保证模型结构和 mask 使其等价于逐分钟在线执行。禁止用以下结构训练可部署 decoder:

- bidirectional LSTM decoder。
- 使用整段 horizon pooling 后再生成每一步 action。
- 未使用 causal mask 的 Transformer decoder。
- 在第 `\tau` 步 action 输入中拼接未来收益、未来价格或未来状态。

换句话说，Phase I 允许用未来信息构造“老师答案”，但不允许训练出一个在执行时依赖未来信息的学生模型。

### 6.8 训练轮数与版本策略

Phase I 默认训练 `100` epochs，与论文实验设置一致。训练轮数必须配置化:

```yaml
epochs: 100
early_stopping_patience: null
save_every: 10
selection_metric: phase1_composite_score
```

版本策略:

| 类型 | 是否默认保留 | 作用 |
| --- | --- | --- |
| `best_vq_model.pt` | 是 | 验证集 `phase1_composite_score` 最优且通过 guardrail 的模型，是本次 Phase I 的正式版本 |
| `last_vq_model.pt` | 是 | 最后一个 epoch 的模型，用于断点恢复或排查训练后期退化 |
| `checkpoints/epoch_*.pt` | 可选 | 每 `save_every` 个 epoch 保存一次，仅用于调试和恢复，不作为后续阶段默认输入 |
| `encoder.pt` / `decoder.pt` / `codebook.pt` | 是 | 从 `best_vq_model.pt` 导出的 Phase II/III 消费产物 |

一次 `{PAIR}/{BATCH_ID}` 训练只产生一个正式 Phase I 版本，即 `best_vq_model.pt` 及其导出的 `encoder.pt`、`decoder.pt`、`codebook.pt`。如果需要比较不同采样策略、seed 或超参数，应创建新的 `BATCH_ID`，例如 `batch_001`、`batch_002`，而不是在同一目录下混用多个正式版本。

## 7. 训练流程

第一阶段流水线:

```text
prepared train/val/test files
  -> file reader and schema validator
  -> sliding-window indexer
  -> stratified window sampler
  -> single-trade DP demonstrations
  -> VQ encoder-decoder training
  -> codebook / decoder / labels export
```

详细步骤:

1. 读取外部准备好的 train/val/test 三个数据文件。
2. 校验 schema，确定 `timestamp`、`close` 和状态特征列。
3. 在 train 文件内用 stride=1 滑动窗口枚举候选 horizon，`h=72` 时约 45 万行可枚举约 44.99 万个候选窗口。
4. 为每个候选窗口计算 `horizon_return`、`realized_volatility`、`draw_pattern` 等分层统计。
5. 按 `stratified_uniform` 或 `stratified_proportional` 从候选窗口中最终采样 `num_demos=30000` 个 train horizon。
6. 只对这 `30000` 个 train horizon 运行 Single-trade DP，得到 `actions` 和 `rewards`。
7. 构造 demonstration dataset。
8. 训练 VQ encoder-decoder `100` epochs。
9. 选择 `phase1_composite_score` 最优且通过风险 guardrail 的 checkpoint。validation horizon 可从 val 文件按固定 stride 生成，或按同一分层策略采样。
10. 使用 best checkpoint 为需要进入 Phase II 的 horizon 生成 `code_label`。
11. 导出 Phase II/III 所需产物。

## 8. 输出产物

建议输出目录:

```text
artifacts/{PAIR}/{BATCH_ID}/phase1/
```

输出产物说明:

| 文件 | 作用 | 主要内容 | 后续使用方 |
| --- | --- | --- | --- |
| `phase1_config.yaml` | 固化本次 Phase I 实验配置，保证可复现 | 输入文件路径、horizon 长度、采样策略、DP 参数、交易成本参数、VQ 参数、训练参数、seed | 复现实验、Phase II/III 读取上下文 |
| `input_schema.json` | 记录三份输入文件的字段契约 | 时间列、价格列、状态特征列、排除列、字段 dtype、文件行数 | Dataset 构建、Phase II/III 状态列对齐 |
| `reward_normalizer.json` | 保存 encoder reward 输入归一化参数 | train reward mean/std 或 median/MAD、clip value、clip ratio | Phase I 复现、val/test 编码一致性 |
| `window_index_train.parquet` | 保存训练集滑动窗口候选与最终采样结果 | 全部候选 `window_start/window_end/last_execution_row/last_markout_row`、分层统计、`strata_label`、`is_sampled` | DP demonstration 生成、采样审计 |
| `window_index_val.parquet` | 保存验证集 horizon 索引 | 验证窗口位置、分层统计、`strata_label` | VQ 验证、Phase II validation label 生成 |
| `window_index_test.parquet` | 保存测试集 horizon 索引 | 测试窗口位置、分层统计、`strata_label` | Phase II/III 离线评估对齐 |
| `demos_train.parquet` | 保存最终 30000 个训练 Horizon 的 DP demonstration | `states` 引用或压缩数组、`prices`、`actions`、扣除手续费和滑点后的 `rewards`、DP net return、切换次数 | VQ encoder-decoder 训练 |
| `horizon_labels_train.parquet` | 保存训练 horizon 的 archetype 标签 | `sample_id`、窗口位置、`code_label`、demo return、strata 信息 | Phase II selector 的 KL/demo regularization |
| `horizon_labels_val.parquet` | 保存验证 horizon 的 archetype 标签 | `sample_id`、窗口位置、`code_label`、demo return、strata 信息 | Phase II 验证、checkpoint 选择 |
| `horizon_labels_test.parquet` | 保存测试 horizon 的 archetype 标签 | `sample_id`、窗口位置、`code_label`、demo return、strata 信息 | 离线分析与可解释性评估 |
| `archetype_diagnostics.parquet` | 保存 per-archetype 诊断指标 | `code_id`、样本数、平均收益、胜率、no-trade ratio、切换点分布、平均持仓方向 | 判断每个 archetype 是否有意义，识别 no-trade code 是否挤占容量 |
| `action_diagnostics.json` | 保存动作分类和切换点诊断 | confusion matrix、per-class precision/recall、switch recall、direction accuracy、timing error 分布 | 诊断 decoder 是否学到单次交易行为 |
| `risk_diagnostics.json` | 保存 validation replay 风险指标 | Sharpe、Sortino、MDD、Calmar、equity curve 摘要 | checkpoint 风险 guardrail 和人工审查 |
| `archetype_separation.json` | 保存 archetype 可区分性指标 | inter-code distance、silhouette score、code distance matrix 摘要 | 判断 codebook 是否学出可区分策略 |
| `archetype_behavior_diagnostics.json` | 保存 decoder 行为多样性诊断 | per-code action entropy、inter-code action diversity、decoder sensitivity to code | 判断不同 code 是否真的产生不同 action 序列 |
| `horizon_boundary_diagnostics.json` | 保存 horizon 间仓位衔接诊断 | boundary turnover cost、boundary position consistency、边界换仓方向分布 | 估算 Phase II 独立 horizon 选择带来的边界成本 |
| `code_stability_diagnostics.json` | 保存 epoch 间 code 分配稳定性 | best/last epoch code agreement、matched agreement、漂移 warning | 判断 horizon labels 是否稳定 |
| `encoder.pt` | 保存训练后的 encoder 权重 | LSTM encoder 与 latent projection 参数 | 离线重新编码 horizon、分析 code 分布 |
| `decoder.pt` | 保存冻结 decoder 权重 | 根据 `states + codebook entry` 输出 base action logits 的参数 | Phase II/III 推理 base action sequence |
| `codebook.pt` | 保存离散 archetype codebook | `K=10` 个 code embedding，维度 16 | Phase II selector 动作空间、Phase III archetype context |
| `best_vq_model.pt` | 保存完整最优 VQ 模型 checkpoint | encoder、decoder、codebook、optimizer 可选、epoch、指标 | 继续训练、回滚、完整复现实验 |
| `last_vq_model.pt` | 保存最后一个 epoch 的完整 checkpoint | 最终 epoch 的模型状态、optimizer 状态、epoch、指标 | 断点恢复、训练退化排查 |
| `checkpoints/epoch_*.pt` | 可选保存的中间 checkpoint | 每 `save_every` 个 epoch 的模型状态 | 调试、恢复，不作为 Phase II 默认输入 |
| `checkpoint_manifest.json` | 记录 checkpoint 验证与选择过程 | 每个 checkpoint 的 epoch、路径、validation 指标、是否 best、被拒绝原因 | 审计 best 模型选择、复现实验 |
| `phase1_report.json` | 保存训练与数据诊断指标 | 重构准确率、VQ loss、code usage、perplexity、风险指标、per-code 指标、切换点指标、采样分布、成本配置 | 验收、实验对比、问题排查 |

`horizon_labels_*.parquet` 字段:

| 字段 | 说明 |
| --- | --- |
| `sample_id` | horizon ID |
| `start_index` | 起始位置 |
| `end_index` | 结束位置 |
| `last_execution_row` | 最后一步 action 实际成交使用的盘口行 |
| `last_markout_row` | 最后一步 reward 持仓收益结算使用的 mark price 行 |
| `strata_label` | 分层采样标签 |
| `code_label` | VQ encoder 分配的 archetype ID |
| `demo_return` | DP demonstration horizon return |
| `num_switches` | action 切换次数 |
| `is_no_trade` | 是否全程 flat |

`phase1_report.json` 字段:

```json
{
  "reconstruction_accuracy": 0.0,
  "weighted_reconstruction_accuracy": 0.0,
  "non_flat_accuracy": 0.0,
  "cross_entropy": 0.0,
  "vq_loss": 0.0,
  "commitment_loss": 0.0,
  "codebook_init_method": "kmeans_warmup",
  "codebook_update_method": "ema",
  "codebook_ema_decay": 0.99,
  "reward_normalization": "train_reward_standard",
  "reward_mean": 0.0,
  "reward_std": 0.0,
  "reward_norm_clip_ratio": 0.0,
  "encoder_input_modality_norms": {
    "state_emb": 0.0,
    "action_emb": 0.0,
    "reward_emb": 0.0
  },
  "reward_embedding_norm_ratio": 0.0,
  "code_usage": {},
  "perplexity": 0.0,
  "dominant_code_ratio": 0.0,
  "dead_code_count": 0,
  "dead_code_restarts": 0,
  "single_trade_consistency_rate": 0.0,
  "mean_demo_return": 0.0,
  "median_demo_return": 0.0,
  "no_trade_ratio": 0.0,
  "no_trade_count": 0,
  "filtered_no_trade_count": 0,
  "resampled_horizon_count": 0,
  "no_trade_code_concentration": {
    "top1": 0.0,
    "top2": 0.0
  },
  "active_trade_code_count": 0,
  "flat_low_vol_sample_ratio": 0.0,
  "window_overlap_ratio": 0.0,
  "min_sample_gap": 0,
  "mean_sample_gap": 0.0,
  "split_boundary_gap": 0,
  "effective_min_gap_between_samples": 36,
  "overlap_relaxation_applied": false,
  "sampling_health_warnings": [],
  "behavior_health_warnings": [],
  "teacher_quality_warnings": [],
  "boundary_health_warnings": [],
  "num_train_rows": 450000,
  "reward_alignment": "paper_formula",
  "num_candidate_windows": 449928,
  "sampling_strategy": "stratified_uniform",
  "strata_distribution": {},
  "cost_config": {
    "reward_alignment": "paper_formula",
    "commission_rate": 0.0002,
    "slippage_model": "lob_depth",
    "book_levels": 5,
    "mark_price": "mid_price",
    "execution_lag": 0,
    "insufficient_depth_policy": "reject_transition"
  },
  "val_student_online_net_return": 0.0,
  "val_dp_teacher_net_return": 0.0,
  "val_dp_teacher_sharpe": 0.0,
  "val_dp_teacher_profitable_ratio": 0.0,
  "dp_teacher_return_distribution": {},
  "val_return_capture_ratio": 0.0,
  "val_regret_to_dp": 0.0,
  "val_cost_paid": 0.0,
  "val_sharpe_ratio": 0.0,
  "val_sortino_ratio": 0.0,
  "val_max_drawdown": 0.0,
  "val_calmar_ratio": 0.0,
  "per_code_avg_return": {},
  "per_code_win_rate": {},
  "per_code_count": {},
  "per_code_no_trade_ratio": {},
  "per_code_switch_point_distribution": {},
  "switch_point_recall": 0.0,
  "switch_direction_accuracy": 0.0,
  "switch_timing_error_mean": 0.0,
  "switch_timing_error_distribution": {},
  "action_precision_recall_per_class": {},
  "confusion_matrix": [],
  "inter_code_distance": 0.0,
  "silhouette_score": 0.0,
  "per_code_action_entropy": {},
  "inter_code_action_diversity": 0.0,
  "inter_code_action_diversity_method": "hamming",
  "decoder_sensitivity_to_code": 0.0,
  "horizon_boundary_turnover_cost": 0.0,
  "horizon_boundary_position_consistency": 0.0,
  "epoch_code_stability": 0.0,
  "epoch_code_stability_matched": 0.0,
  "code_assignment_drift_warning": false,
  "best_epoch": 0,
  "best_checkpoint_path": "best_vq_model.pt",
  "selection_metric": "phase1_composite_score",
  "phase1_composite_score": 0.0
}
```

## 9. 验收标准

### 9.1 数据验收

- 必须能读取 train/val/test 三个文件。
- 每个文件按时间排序后生成 horizon，horizon 不能跨文件边界。
- 状态特征列无 NaN/Inf，且均可转换为 `float32`。
- 本阶段不做特征工程、不在状态特征上拟合 scaler、不生成滚动因子；Phase I encoder 只允许基于 train demonstration rewards 拟合 `reward_normalizer`，并必须复用于 val/test。
- `input_schema.json` 必须记录价格列、时间列、特征列和被排除的元信息列。

### 9.2 滑动窗口与采样验收

- 训练集 1 分钟数据约 45 万行时，候选窗口数必须与 `reward_alignment` 一致: `paper_formula` 接近 `num_rows - h`，`next_row_execution` 接近 `num_rows - h - 1`。
- 最终进入 `demos_train.parquet` 的训练 horizon 数量应等于 `num_demos=30000`。
- `window_index_train.parquet` 必须保存全部候选窗口统计和最终采样标记。
- 分层采样后，各 strata 的样本数应进入 `phase1_report.json`。
- flat/低波动 strata 的最终样本比例不得超过 `flat_low_vol_max_ratio`。
- 默认 `min_gap_between_samples` 应不低于 `h/2`；`h=72` 时默认值为 `36`。
- 默认 `max_overlap_ratio` 应不高于 `0.5`；不再接受 `0.85` 作为默认健康阈值。
- `window_overlap_ratio`、`min_sample_gap`、`mean_sample_gap`、`split_boundary_gap`、`effective_min_gap_between_samples` 必须进入 `phase1_report.json`。
- 若 `window_overlap_ratio > max_overlap_ratio` 或 `min_sample_gap < min_gap_between_samples`，默认应阻止数据构建；只有 `warn_only=true` 时才降级为 `sampling_health_warnings`。
- 若 `split_boundary_gap < split_boundary_embargo`，必须在 `sampling_health_warnings` 中提醒 train/val/test 时间边界过近；默认 `warn_only=false` 时应阻止数据构建或裁掉边界窗口。
- 固定 `seed` 后，重复运行得到相同的 `sample_id/window_start`。

### 9.3 DP 验收

- action 只包含 `{0,1,2}`。
- strict single-trade 模式下，每个样本最多一次 action 切换。
- reward 计算必须按 `reward_alignment` 对齐: 默认 `paper_formula` 使用 `p_{t+1}^{mark}-p_t^{mark}`，可选 `next_row_execution` 使用后移一行的成交和结算。两种模式都必须使用成交行盘口逐档估算滑点，并可由 `prices/order_books/actions/cost_config` 复现。
- 若使用 `next_row_execution`，`phase1_report.json` 和 `checkpoint_manifest.json` 必须标注该结果不直接与论文 Phase I reward 公式比较。
- `phase1_config.yaml`、`demos_train.parquet`、DP planner 和 replay env 使用的 `env_config/cost_config` 必须一致。
- `no_trade_ratio` 必须小于等于 `max_no_trade_ratio`，并被记录进报告。
- 若触发 no-trade 过滤或补采样，`filtered_no_trade_count` 和 `resampled_horizon_count` 必须写入报告。
- 若 `no_trade_ratio > max_no_trade_ratio`，必须在 `sampling_health_warnings` 中提醒调整 `flat_low_vol_max_ratio` 或 `min_profit_gate`。

### 9.4 VQ 验收

- `phase1_composite_score` 达到配置阈值，且普通 `val_reconstruction_accuracy` 不能作为唯一通过条件。
- `val_weighted_reconstruction_accuracy`、`val_non_flat_accuracy`、`switch_point_recall` 必须进入 report。
- codebook 使用率不低于 70%。
- perplexity 不能长期塌缩到 1。
- 若出现 dead code，必须执行配置化 dead-code restart，并在报告中记录。
- encoder 不允许 raw concat `[state_t, action_embedding_t, reward_t]` 后直接输入 LSTM；必须通过 state/action/reward 三路 adapter，并对 `reward_t` 做 train-only normalization。
- `reward_normalizer.json` 必须保存 train reward 统计量，val/test 不得重新拟合；`reward_norm_clip_ratio` 和 `reward_embedding_norm_ratio` 必须进入 report。
- decoder 能在不给 DP 的情况下，根据 `states + code_id` 生成 base actions。
- evaluator 必须输出 validation 的 `student_online_net_return`、`dp_teacher_net_return`、`return_capture_ratio` 和 `regret_to_dp`。
- evaluator 必须输出风险调整收益指标: `val_sharpe_ratio`、`val_sortino_ratio`、`val_max_drawdown`、`val_calmar_ratio`。
- evaluator 必须输出 per-archetype 指标: `per_code_avg_return`、`per_code_win_rate`、`per_code_count`、`per_code_no_trade_ratio`、`per_code_switch_point_distribution`。
- 若 `per_code_no_trade_ratio` 高于阈值或 no-trade 样本 top-2 code 集中度过高，必须在报告中提示 no-trade archetype 挤占 codebook 容量。
- evaluator 必须输出切换点指标: `switch_point_recall`、`switch_direction_accuracy`、`switch_timing_error_distribution`。
- evaluator 必须输出 per-class action 指标: `action_precision_recall_per_class` 和 `confusion_matrix`。
- evaluator 必须输出 archetype 可区分性指标: `inter_code_distance` 和 `silhouette_score`。
- evaluator 必须输出 archetype 行为多样性指标: `per_code_action_entropy`、`inter_code_action_diversity`、`decoder_sensitivity_to_code`。
- evaluator 必须输出 DP teacher 质量指标: `val_dp_teacher_sharpe`、`val_dp_teacher_profitable_ratio`、`dp_teacher_return_distribution`。
- evaluator 必须输出 horizon 边界衔接指标: `horizon_boundary_turnover_cost` 和 `horizon_boundary_position_consistency`。
- evaluator 必须输出 epoch 间 code 稳定性指标: `epoch_code_stability`；若低于阈值，必须在 `phase1_report.json` 中写入 `code_assignment_drift_warning=true`。
- validation student replay 必须按因果在线方式逐分钟生成动作，不能让 decoder 使用未来状态。
- teacher 和 student 的 validation replay 必须使用同一个 `TradingEnv` 和 `CostModel` 扣除手续费、算法滑点和成交成本。

### 9.5 Checkpoint 验证验收

- 每个 epoch 必须产出 train/val metrics。
- `best_vq_model.pt` 必须对应 `checkpoint_manifest.json` 中 `is_best=true` 的 epoch。
- best checkpoint 默认由 `phase1_composite_score` 最大化选择。
- 普通 `val_reconstruction_accuracy` 只可作为 sanity check，不得单独作为 best 选择主指标。
- 若多个 checkpoint 组合分数接近，可用 `val_student_online_net_return` 或 `val_return_capture_ratio` 作为 tie-breaker。
- 若 checkpoint 的 `val_max_drawdown` 超过风险阈值，或 `val_sharpe_ratio` 低于风险阈值，不能直接成为 best，除非配置显式关闭 risk guardrail。
- 若 checkpoint 的 `code_usage_ratio < 0.7`，即使重构准确率更高，也不能被选为 best。
- checkpoint manifest 必须记录 `codebook_init_method` 和 `codebook_update_method`；若使用 EMA 更新，必须记录 `ema_decay`、dead-code restart 次数和 code usage。
- 若 `inter_code_distance` 和 `silhouette_score` 正常但 `inter_code_action_diversity` 或 `decoder_sensitivity_to_code` 过低，checkpoint 不能自动视为通过，需要在 manifest 中记录 decoder 行为退化原因。
- 若 `val_dp_teacher_profitable_ratio` 低于阈值，`val_return_capture_ratio` 不能作为主要 tie-breaker，应优先检查采样策略和 `min_profit_gate`。
- `encoder.pt`、`decoder.pt`、`codebook.pt` 必须从 `best_vq_model.pt` 导出，而不是从 `last_vq_model.pt` 导出。
- 恢复训练时必须能从 `last_vq_model.pt` 读取模型、optimizer、epoch 和历史 best 指标。

### 9.6 产物验收

- Phase II 可以只依赖 `decoder.pt`、`codebook.pt`、`horizon_labels_*.parquet` 启动训练。
- 所有 checkpoint、配置和报告位于同一个 `artifacts/{PAIR}/{BATCH_ID}/phase1/` 目录。
- 固定 seed 后，重复运行能得到一致的 sample IDs 和可比指标。

## 10. 风险与处理

| 风险 | 表现 | 处理 |
| --- | --- | --- |
| codebook collapse | 大部分样本落到同一个 code | 已内置 `codebook_health` 监控、usage regularization、dead-code restart 和 best checkpoint 拒绝规则 |
| codebook 初始化/更新不稳定 | 随机初始化导致 early dead code，或梯度更新导致 code 抖动 | 默认 `kmeans_warmup + ema`；严格复现论文公式时用 `random_normal + gradient`，并在 manifest 记录 |
| DP 全 flat 过多 | `no_trade_ratio` 过高 | eval 输出 `sampling_health_warnings`，提醒调整 `flat_low_vol_max_ratio`、`min_profit_gate` 或 `max_no_trade_ratio` |
| no-trade code 挤占容量 | 大量 no-trade horizon 集中分配到 1-2 个 code | 输出 `per_code_no_trade_ratio`、`no_trade_code_concentration`、`active_trade_code_count`，提醒调整 no-trade 配额 |
| 滑动窗口高度重叠 | 相邻样本过于相似，梯度方差虚低，validation 指标虚高 | 默认 `min_gap_between_samples=h/2`、`max_overlap_ratio=0.5` 且 `warn_only=false`；超阈值时阻止数据构建或要求显式放宽并记录原因 |
| decoder 忽略 code | codebook 向量可分，但不同 code 解码动作几乎相同 | 输出 `inter_code_action_diversity`、`decoder_sensitivity_to_code`；过低时拒绝自动通过或触发 warning |
| horizon 边界频繁换仓 | 相邻 horizon 末仓位和下一 horizon 初始仓位冲突；DP 从 flat 开始但实盘可能从非 flat 开始 | Phase I 输出 `horizon_boundary_turnover_cost` 和 `horizon_boundary_position_consistency`；Phase II 必须支持 `initial_position != flat` 并在第一步扣除边界换仓成本 |
| code 分配不稳定 | best epoch 和 last epoch 对同一 horizon 分配不同 code | 输出 `epoch_code_stability`，过低时写入 `code_assignment_drift_warning` 并考虑延长训练或降低学习率 |
| DP teacher 质量弱 | 老师收益接近 0 或正收益 horizon 过少 | 输出 `val_dp_teacher_sharpe`、`val_dp_teacher_profitable_ratio`，提醒调整采样分层或 `min_profit_gate` |
| encoder reward 信号被淹没 | `reward_t` 量级远小于状态特征，encoder 忽略收益信息 | 使用 train-only `reward_normalizer` 和 reward adapter；监控 `reward_embedding_norm_ratio` 与 `reward_norm_clip_ratio` |
| demonstration 太理想化 | Phase II 回测收益弱 | 第一阶段强制启用手续费和盘口深度逐档滑点，禁止使用 `fixed_bps` |
| 输入数据存在未来信息 | 验证/测试表现异常好 | 审计外部数据文件的字段来源，本项目不重新生成因子 |
| decoder 只记忆动作位置 | 泛化弱 | 增加 validation horizon、多资产训练或 action label smoothing |

## 11. 与后续阶段的接口

Phase II 读取:

- `codebook.pt`
- `decoder.pt`
- `horizon_labels_train.parquet`
- `horizon_labels_val.parquet`
- `input_schema.json`

Phase II 使用 `code_label` 作为 KL/demo regularization 的 ground-truth archetype label:

$$
\hat{a}^{sel}_t = code\_label_t
$$

推理时流程为:

```text
selector chooses code_id
  -> frozen causal decoder receives current/past states and codebook[code_id]
  -> decoder emits the current step base action
```

Phase II 必须额外处理 horizon 间仓位连续性:

- selector 不能假设每个 horizon 从 flat 开始。
- Phase II env 的 `reset()` 需要接收上一 horizon 的 `terminal_position` 作为下一 horizon 的 `initial_position`。
- 下一 horizon 第一笔 target position 与 inherited position 不一致时，必须通过同一 `CostModel` 扣除换仓成本。
- Phase II 的评估报告应继续输出 `horizon_boundary_turnover_cost` 和 `horizon_boundary_position_consistency`，与 Phase I 的边界诊断对齐。

因此，Phase I 的最终验收不是单纯 reconstruction accuracy，而是能否提供稳定、可复用、可解释的 discrete archetype interface。
