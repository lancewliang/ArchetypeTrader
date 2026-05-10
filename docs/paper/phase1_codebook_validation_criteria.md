# 第一阶段 Codebook 验证与 Checkpoint 选择标准

本文档是对 ArchetypeTrader 论文第一阶段的工程化补充。论文定义了第一阶段的训练目标，即用 DP 示范轨迹训练 VQ encoder-decoder，并通过重建损失、codebook loss、commitment loss 学习离散策略原型；但论文没有明确给出第一阶段 codebook 合理性验证标准，也没有明确说明第一阶段 checkpoint 如何选择。

因此，本文档给出一套只依赖第一阶段即可完成的三层验证标准：

1. VQ 内部质量；
2. 原型行为质量；
3. 第一阶段盈利性验证。

最终 checkpoint 必须同时满足三层硬性阈值；在满足硬性阈值的候选中，再按综合评分选择。

## 基本符号

- 验证集 horizon 数量记为 `N_val`。
- codebook 大小记为 `K`。论文实验中 `K = 10`。
- 第 `i` 个 validation horizon 的 DP 示范轨迹为：
  `tau_i = (s_i, a_i^demo, r_i^demo)`。
- encoder 分配的 label 为：
  `k_i = Encoder(s_i, a_i^demo, r_i^demo)`。
- 使用 assigned label 经 decoder 重构得到动作：
  `a_i^dec = Decoder(s_i, e_{k_i})`。
- 执行 DP 示范动作得到收益：
  `R_i^DP`。
- 执行 decoder 重构动作得到收益：
  `R_i^dec`。
- 空仓或 no-trade 基准收益：
  `R_i^flat`。通常可取 0；如果环境中有资金利息或其他成本，应按环境实际定义计算。
- 随机 label 基准收益：
  `R_i^rand = Execute(Decoder(s_i, e_{u_i}))`，其中 `u_i` 从 `{0, ..., K-1}` 随机采样。

所有收益均应扣除手续费和执行成本。

## 第一层：VQ 内部质量

这一层回答：VQ 模型是否学到了稳定、可用、未塌缩的离散表示。

### 指标与阈值

| 指标 | 定义 | 作用 | 通过阈值 |
|---|---|---|---|
| validation action accuracy | `mean(a_i^dec == a_i^demo)`，按所有 horizon 和 timestep 统计 | 检查 decoder 是否能在 assigned code 条件下重建 DP 示范动作；如果该指标过低，说明 code 没有保留基本动作信息。 | `>= 85%` |
| validation reconstruction loss gap | `L_rec_val / L_rec_train` | 检查第一阶段是否过拟合训练示范；如果验证损失明显高于训练损失，后续 selector 使用该 decoder 时泛化风险较大。 | `<= 1.25` |
| active code ratio | 被分配样本比例 `p_k >= 1%` 的 code 数量 / `K` | 检查 codebook 是否被充分使用；如果 active code 太少，第二阶段名义上有 `K` 个 label，实际只有少数可选策略。 | `>= 80%` |
| max code occupancy | `max_k p_k` | 检查是否发生 label 塌缩；如果单个 code 吃掉过多样本，说明不同交易模式被混在同一个 label 中。 | `<= 40%` |
| normalized code perplexity | `exp(-sum_k p_k log p_k) / K` | 衡量 label 分布的有效多样性；过低表示塌缩，过高可能表示分配近似随机、缺乏结构。 | `[0.50, 0.90]` |
| dead code ratio | 被分配样本比例 `p_k < 0.1%` 的 code 数量 / `K` | 检查无效 code 比例；dead code 会浪费 codebook 容量，并增加第二阶段探索无意义 label 的概率。 | `<= 20%` |

### 淘汰规则

满足以下任一条件，直接淘汰该 checkpoint：

- validation action accuracy `< 85%`；
- active code ratio `< 80%`；
- max code occupancy `> 40%`；
- normalized code perplexity `< 0.50`；
- validation reconstruction loss gap `> 1.25`。

### 解释

第一阶段不能只看训练损失。低 reconstruction loss 可能对应 code collapse，即大多数样本都落到少数几个 code；这会导致第二阶段 selector 实际可选的策略很少。合理 codebook 应该同时满足：

- 能重建 DP 示范动作；
- 多数 code 被有效使用；
- code 分布不过度集中；
- 验证集和训练集差距不过大。

## 第二层：原型行为质量

这一层回答：每个 code 是否对应可解释、相对稳定、彼此有区分度的交易行为。

由于论文第一阶段的 DP 示范被限制为每个 horizon 至多一次交易，行为质量可以围绕交易方向、入场时点、持仓长度和动作序列相似度进行统计。

### 指标与阈值

| 指标 | 定义 | 作用 | 通过阈值 |
|---|---|---|---|
| per-code minimum support | 每个 active code 的样本数 `n_k` | 保证每个原型的统计结论有足够样本支撑；样本过少的 code 可能只是偶然噪声或训练残留。 | `n_k >= max(100, 0.02 * N_val)` |
| dominant market morphology ratio | 每个 code 内占比最高的市场形态比例 | 检查一个 code 是否集中出现在相似市场结构中；如果比例过低，selector 很难仅凭市场状态学习何时选择该 code。 | `>= 35%` |
| dominant motif ratio | 每个 code 内占比最高的行为 motif 比例 | 检查一个 code 是否对应清晰交易意图；如果最高 motif 占比太低，该 code 内部行为混乱，selector 难以学习其适用场景。 | `>= 40%` |
| dominant morphology-motif pair ratio | 每个 code 内占比最高的 `(market morphology, trading motif)` 组合比例 | 检查市场形态和交易行为是否形成稳定对应关系；如果 pair 不集中，说明该 code 的盈利可能缺少可预测结构。 | `>= 30%` |
| morphology distribution lift | 某 code 的 dominant morphology 占比 / 全体验证集同类 morphology 占比 | 检查该 code 是否真的偏向某类市场，而不是和全体验证集分布几乎一样；lift 越高，说明 label 对市场结构越有辨识度。 | `>= 1.25` |
| intra-code action similarity | 同一 code 内 decoded action sequence 的平均相似度 | 衡量同一 archetype 内行为一致性；过低表示一个 label 聚合了互相矛盾的动作模式。 | `>= 65%` |
| inter/intra separation | code 间动作中心距离 / code 内动作距离 | 衡量不同 archetype 之间是否有足够区分度；如果该比值低，说明 codebook 只是形式上离散，行为上没有分开。 | `>= 1.30` |
| duplicate code similarity | 任意两个 code 原型 decoded action 的平均相似度 | 检查是否多个 code 学成同一种策略；重复 code 会浪费容量，并让第二阶段在等价 label 间无效探索。 | `<= 85%` |
| profitable-code coverage | 满足第三层 per-code 盈利条件的 active code 数量 / active code 数量 | 检查可解释原型中有多少同时具备盈利潜力；如果覆盖率太低，codebook 虽有行为模式但可交易价值不足。 | `>= 60%` |

### Horizon 市场形态分类

需要对每个 validation horizon 做自动化市场形态分类。这里的形态分类不是人工标注，也不参与训练；它只用于 validation 诊断，回答两个问题：

1. 每个 code 是否集中对应某类市场结构；
2. 每个 code 的交易行为是否和市场结构形成稳定对应关系。

市场形态描述“horizon 长什么样”，行为 motif 描述“原型怎么交易”。二者应分别统计，再联合统计。

#### 基础变量

对每个 horizon 的价格序列 `p_0, ..., p_{h-1}`，先计算：

- `ret_total = p_{h-1} / p_0 - 1`：horizon 总收益率。
- `ret_first = p_{mid} / p_0 - 1`：前半段收益率，其中 `mid = floor(h / 2)`。
- `ret_second = p_{h-1} / p_{mid} - 1`：后半段收益率。
- `realized_vol = std(log(p_t / p_{t-1})) * sqrt(h)`：horizon 内实现波动。
- `max_drawdown`：horizon 内从局部高点到后续低点的最大跌幅。
- `max_runup`：horizon 内从局部低点到后续高点的最大涨幅。
- `range_ratio = (max(p) - min(p)) / p_0`：horizon 内振幅。
- `trend_efficiency = abs(p_{h-1} - p_0) / sum_t abs(p_t - p_{t-1})`：趋势效率，越高表示走势越单边。

阈值建议用 validation set 的分位数自适应确定：

- `vol_high`：`realized_vol` 的 70% 分位数。
- `vol_low`：`realized_vol` 的 30% 分位数。
- `range_high`：`range_ratio` 的 70% 分位数。
- `trend_ret_threshold`：`abs(ret_total)` 的 60% 分位数，且至少大于单边交易成本的 `3` 倍。
- `reversal_leg_threshold`：`abs(ret_first)` 和 `abs(ret_second)` 合并后的 60% 分位数，且至少大于单边交易成本的 `2` 倍。

使用分位数阈值的原因是不同资产和不同时间段波动水平不同，固定收益率阈值容易使某些资产几乎全部落入同一类。

#### 市场形态类别

| 类别 | 判定规则 | 含义 |
|---|---|---|
| `uptrend` | `ret_total > trend_ret_threshold` 且 `trend_efficiency >= 0.35` | 单边上涨或上涨占主导，适合检查 long/hold 类原型。 |
| `downtrend` | `ret_total < -trend_ret_threshold` 且 `trend_efficiency >= 0.35` | 单边下跌或下跌占主导，适合检查 short/hold 类原型。 |
| `reversal-up` | `ret_first < -reversal_leg_threshold` 且 `ret_second > reversal_leg_threshold` | 先跌后涨，适合检查 short-to-long 或 against-recent-move 类原型。 |
| `reversal-down` | `ret_first > reversal_leg_threshold` 且 `ret_second < -reversal_leg_threshold` | 先涨后跌，适合检查 long-to-short 或 against-recent-move 类原型。 |
| `range-high-vol` | `abs(ret_total) <= trend_ret_threshold` 且 `range_ratio >= range_high` | 首尾变化不大但内部振幅较高，可能适合反转或短交易原型。 |
| `range-low-vol` | `abs(ret_total) <= trend_ret_threshold` 且 `realized_vol <= vol_low` | 低波动横盘，可能适合 mostly-flat 或低交易频率原型。 |
| `volatile-mixed` | `realized_vol >= vol_high` 且不满足以上类别 | 高波动但方向不清晰，需警惕 code 行为混杂。 |
| `neutral` | 不满足以上类别 | 没有明显结构，通常不应成为大多数盈利 code 的主要来源。 |

#### 市场形态归类优先级

为保证每个 horizon 只有一个主形态，建议按以下顺序归类：

1. 先判断 `reversal-up` 和 `reversal-down`。
2. 再判断 `uptrend` 和 `downtrend`。
3. 再判断 `range-high-vol` 和 `range-low-vol`。
4. 再判断 `volatile-mixed`。
5. 最后归为 `neutral`。

反转类优先于趋势类，是因为很多强反转 horizon 的首尾收益可能也较大；如果不优先识别反转，容易把反转机会误归为单边趋势。

#### 按 code 的检查方法

对每个 active code `k`，统计：

- `P(morphology | code=k)`：该 code 内各市场形态占比。
- `P(motif | code=k)`：该 code 内各交易 motif 占比。
- `P(morphology, motif | code=k)`：该 code 内市场形态和交易行为组合占比。
- `lift(morphology, k) = P(morphology | code=k) / P(morphology)`：该 code 对某类市场形态的富集倍数。
- `return_by_pair(k, morphology, motif)`：该 code 在不同 pair 下的 decoded return、win rate、retention ratio。

验证时应输出一张 code 级别诊断表：

| code | support | dominant morphology | morph ratio | morph lift | dominant motif | motif ratio | dominant pair | pair ratio | decoded return | win rate |
|---|---:|---|---:|---:|---|---:|---|---:|---:|---:|

一个健康 code 的典型结果应类似：

- `code 3`: dominant morphology 为 `downtrend`，dominant motif 为 `short + early + hold`，dominant pair 占比高，decoded return 为正。
- `code 5`: dominant morphology 为 `reversal-up`，dominant motif 为 `long + middle + delayed-hold + against-recent-move`，retention ratio 较高。
- `code 1`: dominant morphology 为 `range-low-vol`，dominant motif 为 `flat + none + mostly-flat`，收益不一定高，但回撤和交易成本较低。

如果某个 code 的 `P(morphology | code=k)` 与全体验证集 `P(morphology)` 很接近，且 `morphology distribution lift < 1.25`，说明这个 code 没有明显市场结构偏好。这样的 code 不一定必须淘汰，但如果同时 motif ratio 低或盈利性差，应视为弱原型。

### 行为 motif 定义

建议把每条 decoded action sequence 映射为一个可统计的粗粒度 motif。motif 的目标不是精确描述每个 timestep，而是把一个 horizon 内的主要交易意图归类，便于判断同一 code 内是否聚合了相似行为。

#### 基础变量

对长度为 `h` 的 decoded action sequence `a_0, ..., a_{h-1}`，先计算：

- `position_t`：由动作映射得到的持仓方向，取值为 `short`、`flat`、`long`。
- `change_points`：满足 `position_t != position_{t-1}` 的 timestep 集合。
- `non_flat_ratio`：非空仓 timestep 占比。
- `long_ratio`：多头 timestep 占比。
- `short_ratio`：空头 timestep 占比。
- `first_trade_t`：第一个从 `flat` 进入 `long/short`，或从 `long/short` 切换到另一方向的 timestep。
- `main_position`：占比最高的非空仓方向；如果 `long_ratio` 和 `short_ratio` 接近，则记为 `mixed`。
- `holding_ratio_after_entry`：从 `first_trade_t` 到 horizon 结束期间，保持 `main_position` 的 timestep 占比。

如果使用论文第一阶段的 DP 示范，理论上每个 horizon 至多一次主要交易。但 decoder 可能生成轻微抖动，因此 motif 统计应允许少量短暂切换。

#### 维度一：方向 direction

方向描述该 horizon 的主导持仓方向。

| 类别 | 判定规则 | 含义 |
|---|---|---|
| `long` | `long_ratio >= 0.35` 且 `long_ratio >= short_ratio + 0.15` | 主要意图是建立多头或持有多头，通常对应上涨或反弹机会。 |
| `short` | `short_ratio >= 0.35` 且 `short_ratio >= long_ratio + 0.15` | 主要意图是建立空头或持有空头，通常对应下跌或回落机会。 |
| `flat` | `non_flat_ratio < 0.20` | 主要意图是不交易或规避风险。 |
| `mixed` | 不满足以上条件 | 多空占比接近或行为不够单一，可能对应反转、震荡或 decoder 不稳定。 |

#### 维度二：入场时间 entry bucket

入场时间描述主要交易发生在 horizon 的哪个阶段。

| 类别 | 判定规则 | 含义 |
|---|---|---|
| `early` | `first_trade_t / h < 1/3` | 接近 horizon 开始就建仓，偏向趋势延续或快速识别机会。 |
| `middle` | `1/3 <= first_trade_t / h < 2/3` | 等待一段市场演化后建仓，偏向确认信号或中段反转。 |
| `late` | `first_trade_t / h >= 2/3` | horizon 后段才建仓，偏向尾部机会或延迟确认。 |
| `none` | 不存在 `first_trade_t` | 基本无交易，通常与 `flat` 方向配套。 |

#### 维度三：持仓风格 holding style

持仓风格描述建仓后是否持续持有，还是频繁变化。

| 类别 | 判定规则 | 含义 |
|---|---|---|
| `hold` | `holding_ratio_after_entry >= 0.70` 且 `len(change_points) <= 2` | 建仓后基本持有到 horizon 末端，是最符合论文单次交易设定的稳定原型。 |
| `delayed-hold` | `entry bucket` 为 `middle/late`，且 `holding_ratio_after_entry >= 0.70` | 等待后建仓并持有，常见于突破确认或局部反转确认。 |
| `brief-trade` | `0.20 <= non_flat_ratio < 0.50` 且非空仓片段连续 | 只在 horizon 的一小段时间暴露风险，偏向捕捉短机会。 |
| `switching` | `len(change_points) > 2` | 多次切换，可能表示震荡策略，也可能表示 decoder 行为不稳定。 |
| `mostly-flat` | `non_flat_ratio < 0.20` | 基本空仓，偏向风险规避。 |

#### 维度四：反转类型 reversal type

反转类型用于单独识别均值回归或多空切换行为。该维度不是每条样本都必须有，只有满足条件时才附加到 motif 上。

| 类别 | 判定规则 | 含义 |
|---|---|---|
| `long-to-short` | horizon 内主方向从 `long` 切换到 `short`，且两段占比都 `>= 20%` | 先做多后做空，可能对应冲高回落或局部顶部。 |
| `short-to-long` | horizon 内主方向从 `short` 切换到 `long`，且两段占比都 `>= 20%` | 先做空后做多，可能对应探底反弹或局部底部。 |
| `against-recent-move` | 入场方向与入场前价格变化方向相反 | 偏向均值回归或逆势交易。 |
| `with-recent-move` | 入场方向与入场前价格变化方向一致 | 偏向趋势跟随或动量交易。 |
| `none` | 不满足以上条件 | 无明显反转结构。 |

其中 `against-recent-move` 和 `with-recent-move` 需要使用对应 horizon 的价格序列。可用 `first_trade_t` 前 `min(12, first_trade_t)` 个 bar 的价格变化方向作为 recent move。

#### 推荐 motif 字符串

最终 motif 可以按以下格式拼接：

```text
{direction} + {entry_bucket} + {holding_style} [+ {reversal_type}]
```

示例：

- `short + early + hold + with-recent-move`：接近 horizon 开始做空并持有，偏趋势下行。
- `long + middle + delayed-hold + against-recent-move`：中段逆近期下跌方向做多并持有，偏均值回归。
- `short + late + brief-trade`：后段短暂做空，只捕捉尾部下跌机会。
- `mixed + middle + switching + short-to-long`：中段附近发生空到多切换，可能是反转原型，也可能需要检查 decoder 是否抖动。
- `flat + none + mostly-flat`：基本不交易，偏风险规避。

#### 归类优先级

为避免同一条 sequence 被归入多个 motif，建议按以下优先级归类：

1. 如果 `non_flat_ratio < 0.20`，直接归为 `flat + none + mostly-flat`。
2. 如果存在明确 `long-to-short` 或 `short-to-long`，优先标记 reversal type。
3. 再判断 `direction` 和 `entry bucket`。
4. 最后判断 `holding style`。

对于 `switching` 占比较高的 code，需要额外检查：

- 如果该 code 的收益和胜率稳定，可以保留为震荡或反转类原型；
- 如果该 code 的收益不稳定且 action similarity 低，应视为行为混杂或 decoder 不稳定。

### 淘汰规则

满足以下任一条件，直接淘汰该 checkpoint：

- 超过 `20%` 的 active code 样本数低于 `max(100, 0.02 * N_val)`；
- 超过 `40%` 的 active code 的 dominant market morphology ratio `< 35%`；
- 超过 `40%` 的 active code 无法形成 dominant motif，即最高 motif 占比 `< 40%`；
- 超过 `40%` 的 active code 的 dominant morphology-motif pair ratio `< 30%`；
- 超过 `40%` 的 active code 的 morphology distribution lift `< 1.25`，且这些 code 同时不满足第三层 per-code 盈利条件；
- intra-code action similarity `< 65%`；
- inter/intra separation `< 1.30`；
- 存在多个高度重复 code，且 duplicate code similarity `> 85%` 的 code pair 数量超过 `K`。

### 解释

第二层不是要求每个 code 都机械重复同一动作序列。论文强调 archetype 应该抽象出高层交易相似性，而不是僵硬复刻持仓。因此，阈值不应设置到过高。

合理状态是：

- 同一 code 内有可解释的共同行为；
- 不同 code 之间确实代表不同交易意图；
- codebook 不是把同一个策略复制成多个 label；
- 也不是把完全无关的行为强行混到一个 label 里。

## 第三层：第一阶段盈利性验证

这一层回答：encoder 分配出的 label 经 decoder 执行后，是否仍然保留了 DP 示范中的盈利能力。

注意：label 本身只是离散 ID，不能直接交易。要验证的是：

`assigned label + frozen decoder -> decoded action sequence -> validation return`

如果这个 oracle assigned-label 策略在 validation 上都没有盈利能力，第二阶段 selector 基本不可能凭空选出可盈利 label。

### 核心验证方式

对每个 validation horizon：

1. 用 encoder 给 DP 示范轨迹分配 label：`k_i`；
2. 用 frozen decoder 根据 `s_i` 和 `e_{k_i}` 重构动作：`a_i^dec`；
3. 执行 `a_i^dec`，计算扣除手续费和执行成本后的 `R_i^dec`；
4. 与 `R_i^flat`、`R_i^rand`、`R_i^DP` 比较。

这个结果可称为 `oracle-label decoded performance`。它不是线上策略表现，因为真实交易时没有未来 DP label；它是第一阶段 codebook 的可交易性上界检查。

### 全局盈利指标与阈值

| 指标 | 定义 | 作用 | 通过阈值 |
|---|---|---|---|
| mean decoded advantage vs flat | `mean(R_i^dec - R_i^flat)` | 检查 assigned label 经 decoder 执行后是否整体优于不交易；这是第一阶段 codebook 具备交易价值的最低要求。 | `> 0` |
| win rate vs flat | `P(R_i^dec > R_i^flat)` | 检查盈利不是只由少数极端 horizon 贡献；胜率过低说明收益分布不稳定，第二阶段学习难度会变高。 | `>= 55%` |
| mean advantage vs random label | `mean(R_i^dec - R_i^rand)` | 检查 encoder 分配的 label 是否真的有信息量；如果不优于随机 label，说明 label 与市场/示范结构的对应关系弱。 | `> 0`，且相对提升 `>= 20%` |
| retention ratio | `sum(R_i^dec - R_i^flat) / sum(R_i^DP - R_i^flat)` | 衡量 decoder 保留了多少 DP 示范盈利能力；过低表示虽然能重建部分动作，但关键盈利意图在压缩后丢失。 | `>= 50%` |
| downside control | `max_drawdown(R^dec cumulative)` 相对 DP 示范 | 检查 decoded 策略是否用过大回撤换取收益；第一阶段如果已经带来严重下行风险，后续 selector 很难完全修复。 | `<= 1.5 * max_drawdown(R^DP cumulative)` |
| validation Sharpe or risk-adjusted score | 基于 `R_i^dec` 的风险调整收益 | 检查收益质量，而不只看累计收益；用于避免选择高波动、高尾部风险的 codebook。 | `> 0`，且优于 random label |

### Per-code 盈利指标与阈值

对每个 active code `k`，只在被 encoder 分配到该 code 的 validation samples 上统计：

| 指标 | 定义 | 作用 | 通过阈值 |
|---|---|---|---|
| per-code mean advantage | `mean_{i:k_i=k}(R_i^dec - R_i^flat)` | 检查单个 archetype 在其被分配到的市场片段上是否有正期望；避免整体盈利被少数 code 掩盖。 | `> 0` |
| per-code win rate | `P(R_i^dec > R_i^flat | k_i=k)` | 检查单个 archetype 的盈利稳定性；如果某个 code 胜率过低，selector 选中它时风险较高。 | `>= 52%` |
| per-code retention ratio | `sum_{i:k_i=k}(R_i^dec - R_i^flat) / sum_{i:k_i=k}(R_i^DP - R_i^flat)` | 衡量每个 code 对其对应 DP 示范盈利能力的保留程度；用于发现只在全局上看似有效、局部却失真的 code。 | `>= 40%` |
| bad-code ratio | per-code mean advantage `< 0` 的 active code 数量 / active code 数量 | 衡量 codebook 中负价值原型的比例；坏 code 过多会扩大第二阶段 selector 的动作噪声。 | `<= 30%` |

### 淘汰规则

满足以下任一条件，直接淘汰该 checkpoint：

- `mean(R_i^dec - R_i^flat) <= 0`；
- win rate vs flat `< 55%`；
- retention ratio `< 50%`；
- decoded performance 没有显著优于 random label；
- bad-code ratio `> 30%`；
- oracle-label decoded cumulative return 曲线明显由极少数 horizon 贡献，且去掉收益最高的 `5%` horizon 后总收益 `<= 0`。

### 解释

第一阶段盈利性验证是必要条件，但不是充分条件。

必要性在于：第二阶段 selector 只能从第一阶段 codebook 中选择 code。如果 assigned-label decoder 在 oracle 情况下都不赚钱，说明 codebook 没有保留可交易行为。

非充分性在于：第二阶段真实输入只有 horizon 起点附近的市场状态，不能看到完整未来轨迹。如果第一阶段 label 虽然盈利，但从起点状态不可预测，selector 仍然可能选不准。因此，第三层只证明 codebook 有可交易潜力，不证明完整系统最终盈利。

## Checkpoint 选择流程

### 硬过滤

每个候选 checkpoint 按顺序执行：

1. 通过第一层 VQ 内部质量阈值；
2. 通过第二层原型行为质量阈值；
3. 通过第三层第一阶段盈利性验证阈值。

任一层失败，则该 checkpoint 不进入最终候选集合。

### 综合评分

对通过硬过滤的 checkpoint，使用以下评分排序：

```text
Score =
  0.30 * normalized_reconstruction_score
+ 0.20 * normalized_codebook_health_score
+ 0.20 * normalized_behavior_separation_score
+ 0.30 * normalized_oracle_profitability_score
```

建议定义：

- `normalized_reconstruction_score`：validation action accuracy，截断到 `[0.85, 1.00]` 后归一化；
- `normalized_codebook_health_score`：active code ratio、perplexity、max occupancy 的综合分；
- `normalized_behavior_separation_score`：intra-code similarity 和 inter/intra separation 的综合分；
- `normalized_oracle_profitability_score`：retention ratio、win rate、risk-adjusted return 的综合分。

最终选择 `Score` 最高的 checkpoint。

### Tie-breaker

如果多个 checkpoint 综合评分接近，差距 `< 3%`，按以下顺序选择：

1. oracle-label decoded risk-adjusted return 更高者；
2. retention ratio 更高者；
3. active code ratio 更高且 max occupancy 更低者；
4. validation reconstruction loss 更低者。

## 推荐默认阈值汇总

| 层级 | 关键阈值 |
|---|---|
| 第一层 VQ 内部质量 | action accuracy `>= 85%`；active code ratio `>= 80%`；max occupancy `<= 40%`；normalized perplexity `[0.50, 0.90]`；val/train loss gap `<= 1.25` |
| 第二层 原型行为质量 | per-code support `>= max(100, 2% N_val)`；dominant market morphology `>= 35%`；dominant motif `>= 40%`；dominant morphology-motif pair `>= 30%`；morphology lift `>= 1.25`；intra-code similarity `>= 65%`；inter/intra separation `>= 1.30`；duplicate similarity `<= 85%` |
| 第三层 盈利性验证 | mean advantage vs flat `> 0`；win rate vs flat `>= 55%`；advantage vs random label `>= 20%`；retention ratio `>= 50%`；bad-code ratio `<= 30%` |

## 适用边界

这套标准只用于第一阶段 validation 和 checkpoint 选择，不替代第二阶段 selector validation。

第一阶段应回答：

- codebook 是否没有塌缩；
- archetype 是否有稳定且区分明确的行为含义；
- assigned label 经 decoder 执行后是否保留可盈利能力。

第二阶段仍需回答：

- 在没有未来 DP label 的情况下，selector 是否能根据当前市场状态选中合适 archetype；
- 选择策略是否能在真实 validation trading 环境中获得稳定收益和风险控制。
