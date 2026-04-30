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
| 手续费和盘口深度滑点 | 论文 MDP 已包含 execution loss 和 commission；本设计细化 `r_t(i->j)` 的成本计算 | 否 |
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
| `prices` | `[h + 2]` | `float32` | 用于 next-row execution 下计算完整 `h` 步 reward |
| `execution_books` | `[h, levels, 4]` | `float32` | 每步实际成交行的 bid/ask 盘口，默认来自状态行的下一行 |
| `actions` | `[h]` | `int64` | DP demonstration action |
| `rewards` | `[h]` | `float32` | 执行 demonstration action 的逐步收益 |
| `start_index` | scalar | `int64` | horizon 起点 |
| `end_index` | scalar | `int64` | horizon 终点 |
| `pair` | scalar | `str` | 交易标的 |
| `split` | scalar | `str` | train/val/test |
| `sample_id` | scalar | `str` | 可复现样本 ID |

默认采用 `execution_lag=1` 的保守对齐方式: 第 `k` 个状态行只用于生成动作，动作在第 `k+1` 行盘口成交，并用第 `k+2` 行 mark price 计算持仓收益。因此一个 horizon 候选窗口需要覆盖 `h + 2` 行价格和盘口，其中前 `h` 行进入 `states`。

### 3.4 滑动窗口与分层采样

用户提供的是约 45 万行的分钟级训练数据。Phase I 的目标不是把全部滑窗候选都作为训练样本，而是先用滑动窗口从这 45 万行中枚举候选 horizon，再通过分层采样选出最终 `30000` 个 horizon 进入 DP demonstration 和 VQ 训练。

候选窗口生成:

```text
window_start = 0, 1, 2, ..., num_rows - h - 2
window_end = window_start + h - 1
last_execution_row = window_start + h
last_markout_row = window_start + h + 1
```

当 `h=72` 且训练集约 `450000` 行时，stride 为 1 的滑动窗口可枚举约:

```text
450000 - 72 - 1 = 449927
```

个候选 horizon。随后只从这些候选中采样 `30000` 个作为最终训练 horizon。

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
- 设置 `min_gap_between_samples` 和 `max_overlap_ratio` 检查相邻采样窗口是否过度重叠；超阈值时在 eval/report 中给出调参提醒。
- 同一批次必须固定 `seed`，并把被采样的 `window_start` 保存到 `window_index_train.parquet`。
- val/test 不参与 VQ 训练采样；需要标签时可按固定 stride 枚举，或使用同一分层策略生成评估窗口索引。

采样健康检查配置:

```yaml
sampling_health:
  max_no_trade_ratio: 0.25
  flat_low_vol_max_ratio: 0.15
  min_gap_between_samples: 12
  max_overlap_ratio: 0.85
  warn_only: true
```

`phase1_evaluator.py` 必须在每次数据构建后输出 `sampling_health_warnings`。这些 warning 不一定中断训练，但必须提示是否需要调整 `flat_low_vol_max_ratio`、`min_profit_gate`、`min_gap_between_samples` 或采样策略。

## 4. 目录与模块设计

建议新增以下模块，和 `run_pipeline.sh` 中预留的 `scripts/train_phase1.py` 对齐:

```text
scripts/train_phase1.py
src/config/phase1_config.py
src/data/dataset.py
src/planners/single_trade_dp.py
src/models/vq_archetype.py
src/trainers/phase1_trainer.py
src/evaluation/phase1_evaluator.py
src/evaluation/phase1_replay.py
src/envs/trading_env.py
src/trading/cost_model.py
src/utils/io.py
src/evaluation/phase1_metrics.py
src/trainers/phase1_checkpoint.py
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
- 采样健康检查参数: `max_no_trade_ratio`, `flat_low_vol_max_ratio`, `min_gap_between_samples`, `max_overlap_ratio`
- DP/交易成本参数: `gamma`, `commission_rate`, `slippage_model=lob_depth`, `execution_lag`, `max_position`, `cost_model`
- VQ 参数: `hidden_dim=128`, `code_dim=16`, `num_codes=10`, `beta0=0.25`, `usage_regularization_weight`, `dead_code_restart`
- 训练参数: `batch_size`, `lr`, `epochs`, `seed`, `device`
- checkpoint 参数: `save_every`, `selection_metric`, `selection_mode`, `early_stopping_patience`

### 4.3 `src/data/dataset.py`

职责:

- 直接读取 train/val/test 三个数据文件。
- 校验输入 schema，识别价格列、时间列、状态特征列。
- 用滑动窗口在每个文件内部枚举候选 horizon。
- 为候选 horizon 计算分层统计并生成 strata label。
- 从训练候选窗口中分层采样 `num_demos` 个 horizon。
- 调用或读取 DP demonstration。
- 构造 PyTorch `Dataset` 和 `DataLoader`。
- 保证 horizon 只在单个输入文件内部生成，不跨 train/val/test 文件。

核心类建议:

```python
class MarketFileReader:
    def read(self, path): ...

class SlidingWindowIndexer:
    def enumerate(self, num_rows, horizon, stride): ...

class StratifiedWindowSampler:
    def sample(self, window_index, num_samples, strategy, seed): ...

class HorizonSampler:
    def build(self, frame, window_index): ...

class Phase1DemoDataset(torch.utils.data.Dataset):
    def __getitem__(self, idx): ...
```

### 4.4 `src/planners/single_trade_dp.py`

实现论文 Algorithm 1。

输入:

- `prices`: `[h + 2]`
- `order_books`: 至少包含成交行的 ask1-ask5/bid1-bid5 价量
- `actions`: `[0, 1, 2]`
- `gamma`
- `commission_rate`
- `slippage_model`
- `execution_lag`
- `max_position`
- `cost_model`

输出:

- `actions`: `[h]`
- `rewards`: `[h]`
- `total_return`
- `num_switches`
- `is_no_trade`

### 4.5 `src/models/vq_archetype.py`

包含:

- `ArchetypeEncoder`
- `VectorQuantizer`
- `ArchetypeDecoder`
- `VQArchetypeModel`

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

### 4.6 `src/trainers/phase1_trainer.py`

职责:

- 生成或加载 `demos_train.parquet`。
- 训练 VQ encoder-decoder。
- 每个 epoch 结束后调用 evaluator 在 validation horizon 上计算指标。
- 根据 `selection_metric` 调用 checkpoint manager 保存 `best_vq_model.pt`。
- 保存 `last_vq_model.pt` 和可选中间 checkpoint。
- 用 best checkpoint 生成 horizon code labels。
- 输出 `phase1_report.json`。

### 4.7 `src/evaluation/phase1_evaluator.py`

职责:

- 在 train/val horizon 上评估 VQ encoder-decoder。
- 计算 checkpoint 选择所需指标。
- 检查 codebook 是否塌缩。
- 调用 `Phase1ReplayEvaluator` 获取真实收益指标，但不直接实现交易 replay 逻辑。
- 输出 epoch-level metrics，供 trainer 和 checkpoint manager 使用。

建议指标:

| 指标 | 说明 | 用途 |
| --- | --- | --- |
| `val_reconstruction_accuracy` | validation action 重构准确率 | 默认 best checkpoint 选择指标，越高越好 |
| `val_cross_entropy` | validation action 重构 CE loss | 辅助判断过拟合，越低越好 |
| `val_perplexity` | validation codebook perplexity | 判断 code 使用是否健康 |
| `code_usage_ratio` | 被使用 code 数 / `K` | 低于阈值说明 codebook collapse |
| `single_trade_consistency_rate` | decoder 输出动作满足单次切换约束的比例 | 衡量 decoder 是否学到 demonstration 结构 |
| `val_student_online_net_return` | checkpoint 在 validation horizon 上按因果方式推理动作后 replay 的净收益 | 衡量学生模型在线执行能赚多少 |
| `val_dp_teacher_net_return` | DP teacher 在同一批 validation horizon 上的 hindsight 净收益 | 作为老师上限 |
| `val_return_capture_ratio` | `student_net_return / max(abs(dp_teacher_net_return), eps)` | 衡量学生学到老师多少收益能力 |
| `val_regret_to_dp` | `dp_teacher_net_return - student_net_return` | 衡量学生与老师的收益差距 |
| `val_cost_paid` | validation replay 中手续费、滑点、成交成本总和 | 检查收益是否真实扣成本 |

默认 checkpoint 选择:

```yaml
selection_metric: val_reconstruction_accuracy
selection_mode: max
min_code_usage_ratio: 0.7
secondary_metric: val_student_online_net_return
```

如果 `val_reconstruction_accuracy` 更高但 `code_usage_ratio < min_code_usage_ratio`，该 checkpoint 不应成为 best，应继续训练或触发告警。

Online-style replay 规则:

1. 对每个 validation horizon，先用同一套 Single-trade DP 和 `cost_config` 离线生成 teacher action 与 `dp_teacher_net_return`。
2. 对同一 horizon，用当前 checkpoint 的 encoder 得到 `code_id`。这是 Phase I 的离线标签评估，不代表 Phase II 线上 selector。
3. 冻结 decoder，按分钟因果执行: 第 `\tau` 步只能输入 `s_t, ..., s_\tau` 和 `codebook[code_id]`，输出当前动作。
4. 用输出动作序列按 `cost_config` replay: 第 `\tau` 步动作使用 `\tau+1` 行 bid/ask 盘口逐档成交，并用 `\tau+2` 行 mark price 结算持仓收益，得到 `student_online_net_return`。
5. 汇总所有 validation horizon，写入 epoch metrics 和 `checkpoint_manifest.json`。

该验证回答的问题是: 在给定老师压缩出的 archetype code 后，学生 decoder 以在线可执行方式复现老师交易能力的程度。它不能替代 Phase II selector 的线上评估，因为 Phase II 还需要学习如何在 horizon 起点选择 `code_id`。

### 4.8 `src/evaluation/phase1_replay.py`

职责:

- 独立实现 Phase I checkpoint 的 validation replay。
- 对同一批 validation horizon 分别计算 DP teacher 净收益和 student online 净收益。
- 调用 `TradingEnv` 逐步执行动作；env 内部调用 `CostModel` 扣除手续费和盘口深度滑点。
- 输出 student/teacher 收益差距指标，交给 `phase1_evaluator.py` 汇总。

核心类建议:

```python
class Phase1ReplayEvaluator:
    def evaluate_checkpoint(self, model, val_dataset, env_factory): ...
    def replay_student_online(self, decoder, codebook, horizon, code_id, env): ...
    def replay_dp_teacher(self, horizon, env): ...
```

模块边界:

| 模块 | 负责 | 不负责 |
| --- | --- | --- |
| `phase1_evaluator.py` | 重构指标、codebook 指标、汇总 replay 指标、生成 epoch metrics | 逐笔成交、盘口滑点、收益 replay |
| `phase1_replay.py` | validation online replay、student/teacher 净收益、regret/capture ratio | checkpoint 保存、best 选择、底层成交成本计算 |
| `envs/trading_env.py` | 状态推进、动作执行、position/cash/nav/reward/done 统一语义 | 模型训练、checkpoint 选择 |
| `trading/cost_model.py` | 盘口逐档成交、手续费、滑点和未成交处理 | 模型推理、指标汇总 |

### 4.9 `src/envs/trading_env.py`

职责:

- 提供统一的分钟级交易环境，供 DP、student replay、后续 Phase II/III 复用。
- 管理当前步索引、持仓、现金、净值、成交记录和 episode 结束条件。
- 执行 action 到目标仓位的转换，并调用 `CostModel` 完成 next-row execution。
- 返回扣除手续费和盘口滑点后的 step reward。
- 保证所有阶段使用同一套 reward 和成交语义。

核心接口建议:

```python
class TradingEnv:
    def reset(self, horizon): ...
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
- `Phase1ReplayEvaluator` 用 env 逐步执行 student decoder 输出的 action，得到 online-style net return。
- DP teacher 和 student replay 必须使用相同 env 配置。

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

- 定义 Phase I 专用指标的计算函数。
- 为 `phase1_evaluator.py` 和 `phase1_replay.py` 提供可复用的 metric 实现。
- 避免 Phase II/III 的 RL 指标和 Phase I 的 VQ/replay 指标混在一个通用 `metrics.py` 中。

指标范围:

| 指标类别 | 具体指标 |
| --- | --- |
| 重构指标 | `reconstruction_accuracy`、`cross_entropy` |
| VQ 指标 | `code_usage_ratio`、`perplexity`、`vq_loss`、`commitment_loss` |
| 行为结构指标 | `single_trade_consistency_rate`、`no_trade_ratio` |
| replay 收益指标 | `student_online_net_return`、`dp_teacher_net_return`、`return_capture_ratio`、`regret_to_dp`、`cost_paid` |
| 采样健康指标 | `strata_distribution`、`num_candidate_windows`、`num_sampled_horizons`、`flat_low_vol_sample_ratio`、`window_overlap_ratio`、`min_sample_gap`、`mean_sample_gap`、`sampling_health_warnings` |

核心函数建议:

```python
def reconstruction_accuracy(logits, actions): ...
def codebook_perplexity(code_ids, num_codes): ...
def single_trade_consistency(actions): ...
def return_capture_ratio(student_return, teacher_return): ...
def regret_to_dp(student_return, teacher_return): ...
def window_overlap_ratio(sampled_windows, horizon): ...
def sampling_health_warnings(report, config): ...
```

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

### 5.3.1 动作与成交行对齐

分钟级数据通常包含该分钟收盘价、盘口快照和基于该行计算出的特征。为了避免“看到当前分钟收盘信息后又按当前分钟价格成交”的隐性泄漏，默认采用 next-row execution:

| 概念 | 行号 |
| --- | --- |
| 决策状态 | `decision_row = window_start + k` |
| 动作 | `a_k = policy(s_{decision_row})` |
| 成交盘口 | `execution_row = decision_row + execution_lag`，默认 `execution_lag=1` |
| 持仓结算 | `markout_row = execution_row + 1` |

因此，第 `k` 步 action 对应的是 horizon 中第 `k+1` 行的盘口成交，而不是第 `k` 行。第 `k` 行只用于观察和决策。该约定同时用于 DP teacher、student online replay 和 checkpoint validation。

逐步收益:

$$
r_k(i \rightarrow j)=P(j)(p_{markout}^{mark}-p_{exec}^{mark})-O_{exec}(i,j)
$$

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
  commission_rate: 0.0002
  slippage_model: lob_depth
  book_levels: 5
  mark_price: mid_price
  execution_lag: 1
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

## 6. VQ Encoder-Decoder 设计

### 6.1 Encoder

每步输入:

```text
[state_t, action_embedding_t, reward_t]
```

结构:

```text
input projection -> LSTM(hidden_dim=128) -> last hidden -> MLP -> z_e(16)
```

输出:

$$
z_e \in R^{16}
$$

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

重要约束: decoder 必须是 causal decoder。训练时可以把 `[batch, h, feature_dim]` 作为张量一次性送入模型以提高计算效率，但第 `\tau` 步 action logits 只能依赖 `s_t, ..., s_\tau` 和选中的 `z_q`，不能依赖 `s_{\tau+1}, ..., s_{t+h-1}`。实现上可以使用逐步 MLP 或单向 LSTM；不能使用 bidirectional LSTM，也不能使用未加 causal mask 的 Transformer/self-attention。

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
selection_metric: val_reconstruction_accuracy
```

版本策略:

| 类型 | 是否默认保留 | 作用 |
| --- | --- | --- |
| `best_vq_model.pt` | 是 | 验证集 `val_reconstruction_accuracy` 最优的模型，是本次 Phase I 的正式版本 |
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
9. 选择 validation reconstruction accuracy 最优的 checkpoint。validation horizon 可从 val 文件按固定 stride 生成，或按同一分层策略采样。
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
| `window_index_train.parquet` | 保存训练集滑动窗口候选与最终采样结果 | 全部候选 `window_start/window_end/last_execution_row/last_markout_row`、分层统计、`strata_label`、`is_sampled` | DP demonstration 生成、采样审计 |
| `window_index_val.parquet` | 保存验证集 horizon 索引 | 验证窗口位置、分层统计、`strata_label` | VQ 验证、Phase II validation label 生成 |
| `window_index_test.parquet` | 保存测试集 horizon 索引 | 测试窗口位置、分层统计、`strata_label` | Phase II/III 离线评估对齐 |
| `demos_train.parquet` | 保存最终 30000 个训练 Horizon 的 DP demonstration | `states` 引用或压缩数组、`prices`、`actions`、扣除手续费和滑点后的 `rewards`、DP net return、切换次数 | VQ encoder-decoder 训练 |
| `horizon_labels_train.parquet` | 保存训练 horizon 的 archetype 标签 | `sample_id`、窗口位置、`code_label`、demo return、strata 信息 | Phase II selector 的 KL/demo regularization |
| `horizon_labels_val.parquet` | 保存验证 horizon 的 archetype 标签 | `sample_id`、窗口位置、`code_label`、demo return、strata 信息 | Phase II 验证、checkpoint 选择 |
| `horizon_labels_test.parquet` | 保存测试 horizon 的 archetype 标签 | `sample_id`、窗口位置、`code_label`、demo return、strata 信息 | 离线分析与可解释性评估 |
| `encoder.pt` | 保存训练后的 encoder 权重 | LSTM encoder 与 latent projection 参数 | 离线重新编码 horizon、分析 code 分布 |
| `decoder.pt` | 保存冻结 decoder 权重 | 根据 `states + codebook entry` 输出 base action logits 的参数 | Phase II/III 推理 base action sequence |
| `codebook.pt` | 保存离散 archetype codebook | `K=10` 个 code embedding，维度 16 | Phase II selector 动作空间、Phase III archetype context |
| `best_vq_model.pt` | 保存完整最优 VQ 模型 checkpoint | encoder、decoder、codebook、optimizer 可选、epoch、指标 | 继续训练、回滚、完整复现实验 |
| `last_vq_model.pt` | 保存最后一个 epoch 的完整 checkpoint | 最终 epoch 的模型状态、optimizer 状态、epoch、指标 | 断点恢复、训练退化排查 |
| `checkpoints/epoch_*.pt` | 可选保存的中间 checkpoint | 每 `save_every` 个 epoch 的模型状态 | 调试、恢复，不作为 Phase II 默认输入 |
| `checkpoint_manifest.json` | 记录 checkpoint 验证与选择过程 | 每个 checkpoint 的 epoch、路径、validation 指标、是否 best、被拒绝原因 | 审计 best 模型选择、复现实验 |
| `phase1_report.json` | 保存训练与数据诊断指标 | 重构准确率、VQ loss、code usage、perplexity、采样分布、成本配置、no-trade ratio | 验收、实验对比、问题排查 |

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
  "cross_entropy": 0.0,
  "vq_loss": 0.0,
  "commitment_loss": 0.0,
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
  "flat_low_vol_sample_ratio": 0.0,
  "window_overlap_ratio": 0.0,
  "min_sample_gap": 0,
  "mean_sample_gap": 0.0,
  "sampling_health_warnings": [],
  "num_train_rows": 450000,
  "num_candidate_windows": 449927,
  "sampling_strategy": "stratified_uniform",
  "strata_distribution": {},
  "cost_config": {
    "commission_rate": 0.0002,
    "slippage_model": "lob_depth",
    "book_levels": 5,
    "mark_price": "mid_price",
    "execution_lag": 1,
    "insufficient_depth_policy": "reject_transition"
  },
  "val_student_online_net_return": 0.0,
  "val_dp_teacher_net_return": 0.0,
  "val_return_capture_ratio": 0.0,
  "val_regret_to_dp": 0.0,
  "val_cost_paid": 0.0,
  "best_epoch": 0,
  "best_checkpoint_path": "best_vq_model.pt",
  "selection_metric": "val_reconstruction_accuracy"
}
```

## 9. 验收标准

### 9.1 数据验收

- 必须能读取 train/val/test 三个文件。
- 每个文件按时间排序后生成 horizon，horizon 不能跨文件边界。
- 状态特征列无 NaN/Inf，且均可转换为 `float32`。
- 本阶段不做特征工程、不拟合 scaler、不生成滚动因子。
- `input_schema.json` 必须记录价格列、时间列、特征列和被排除的元信息列。

### 9.2 滑动窗口与采样验收

- 训练集 1 分钟数据约 45 万行且 `execution_lag=1` 时，候选窗口数应接近 `num_rows - h - 1`。
- 最终进入 `demos_train.parquet` 的训练 horizon 数量应等于 `num_demos=30000`。
- `window_index_train.parquet` 必须保存全部候选窗口统计和最终采样标记。
- 分层采样后，各 strata 的样本数应进入 `phase1_report.json`。
- flat/低波动 strata 的最终样本比例不得超过 `flat_low_vol_max_ratio`。
- `window_overlap_ratio`、`min_sample_gap`、`mean_sample_gap` 必须进入 `phase1_report.json`。
- 若 `window_overlap_ratio > max_overlap_ratio` 或 `min_sample_gap < min_gap_between_samples`，必须在 `sampling_health_warnings` 中提醒调整 `min_gap_between_samples` 或采样策略。
- 固定 `seed` 后，重复运行得到相同的 `sample_id/window_start`。

### 9.3 DP 验收

- action 只包含 `{0,1,2}`。
- strict single-trade 模式下，每个样本最多一次 action 切换。
- reward 计算必须按 next-row execution 对齐，使用成交行盘口逐档估算滑点，并可由 `prices/order_books/actions/cost_config` 复现。
- `phase1_config.yaml`、`demos_train.parquet`、DP planner 和 replay env 使用的 `env_config/cost_config` 必须一致。
- `no_trade_ratio` 必须小于等于 `max_no_trade_ratio`，并被记录进报告。
- 若触发 no-trade 过滤或补采样，`filtered_no_trade_count` 和 `resampled_horizon_count` 必须写入报告。
- 若 `no_trade_ratio > max_no_trade_ratio`，必须在 `sampling_health_warnings` 中提醒调整 `flat_low_vol_max_ratio` 或 `min_profit_gate`。

### 9.4 VQ 验收

- validation reconstruction accuracy 达到配置阈值。
- codebook 使用率不低于 70%。
- perplexity 不能长期塌缩到 1。
- 若出现 dead code，必须执行配置化 dead-code restart，并在报告中记录。
- decoder 能在不给 DP 的情况下，根据 `states + code_id` 生成 base actions。
- evaluator 必须输出 validation 的 `student_online_net_return`、`dp_teacher_net_return`、`return_capture_ratio` 和 `regret_to_dp`。
- validation student replay 必须按因果在线方式逐分钟生成动作，不能让 decoder 使用未来状态。
- teacher 和 student 的 validation replay 必须使用同一个 `TradingEnv` 和 `CostModel` 扣除手续费、算法滑点和成交成本。

### 9.5 Checkpoint 验证验收

- 每个 epoch 必须产出 train/val metrics。
- `best_vq_model.pt` 必须对应 `checkpoint_manifest.json` 中 `is_best=true` 的 epoch。
- best checkpoint 默认由 `val_reconstruction_accuracy` 最大化选择。
- 若多个 checkpoint 重构指标接近，可用 `val_student_online_net_return` 或 `val_return_capture_ratio` 作为 tie-breaker。
- 若 checkpoint 的 `code_usage_ratio < 0.7`，即使重构准确率更高，也不能被选为 best。
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
| DP 全 flat 过多 | `no_trade_ratio` 过高 | eval 输出 `sampling_health_warnings`，提醒调整 `flat_low_vol_max_ratio`、`min_profit_gate` 或 `max_no_trade_ratio` |
| 滑动窗口高度重叠 | 相邻样本过于相似 | eval 输出 `window_overlap_ratio/min_sample_gap` 和 warning，提醒设置或增大 `min_gap_between_samples` |
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

因此，Phase I 的最终验收不是单纯 reconstruction accuracy，而是能否提供稳定、可复用、可解释的 discrete archetype interface。
