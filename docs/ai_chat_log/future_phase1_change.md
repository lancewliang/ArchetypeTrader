# Phase I 后续改动校对记录

早期 Phase I 诊断数据告诉我们（以下数值保留当时实验口径；当前代码状态以文末总表为准）：

## 2026-04-22 补充：BiLSTM Encoder 首轮实验结果

实验：`logs/AL/batch_02-short/AL_pipeline_20260422_215103.log`，在 `short` 因子组上将 Phase I encoder 从单向 LSTM 改为 BiLSTM。对照参考为同一天较早的 `logs/AL/batch_01-short/AL_pipeline_20260422_151607.log`。

结论：BiLSTM encoder 已实现，但这次首轮 `short` 对照效果不好，不建议把“单方向 hidden_dim=128 的 BiLSTM encoder”直接当成默认增强主线。更准确的判断是：它提升了 encoder 的表达能力，同时也更容易把完整示范轨迹中的 hindsight 信息压进 code；Phase II selector 只看 horizon 市场状态，未必能稳定预测这种更强、更后验的 code。

关键指标对照：

| 指标 | batch_01-short（单向 encoder） | batch_02-short（BiLSTM encoder） |
|---|---:|---:|
| Encoder 参数量 | 332,064 | 664,096 |
| Phase I profit gate 命中 | 9/13 | 1/13 |
| 选中 checkpoint | epoch 240 | epoch 120 |
| realizable proxy mean | 182.91 | 152.18 |
| realizability score | 0.965 | 0.777 |
| Phase II best val avg_return | 154.13 | 125.20 |
| 选中 checkpoint token accuracy | 0.9998 | 0.9706 |
| 选中 checkpoint exact match | 0.9906 | 0.7373 |
| quantization MSE | 0.375 | 8.202 |

这次 BiLSTM 版本的问题不在于 Phase II “只训练了一半”：`batch_02-short` 在 Phase II step 61,440 已达到最佳验证收益 125.20，后续到 step 488,448 反而降到 114.15。更核心的问题发生在 Phase I checkpoint 选择：13 个候选只有 1 个通过 profit gate，最后选中的 epoch 120 是收益 proxy 过关但重构/量化健康度较差的早期 checkpoint。

后续建议：

- 不再优先继续堆 Phase I 架构容量；先修 Phase I/Phase II 窗口标签对齐问题。
- 如果继续保留 BiLSTM encoder，建议先试 `hidden_dim=64`（单方向），让双向拼接后的宽度接近原单向 128，避免参数量直接翻倍。
- Phase I checkpoint gate 增加健康约束，例如 `exact_match >= 0.95`、`quantization_mse` 上限、`phase2_realizability_score` 下限，避免选中“收益 proxy 好但 latent 很散”的 checkpoint。
- 用 `middle` 因子组做同配置对照；历史记录里 `middle` 通常明显强于 `short`。

问题 1：change point 处 56.7% 准确率，混淆矩阵显示主要错误是 short→flat (361/881=41%) 和 long→flat (343/892=38%)

decoder 在 change point 处倾向于预测 flat 而不是正确的方向。这不主要是模型容量问题：当时的 BiLSTM 128 维配置已经足够暴露出该现象。问题在于 72 步中只有 1 步是 change point，标准 CE loss 被 71 步的"保持不变"主导，decoder 学到了"有疑问就预测 flat"的保守策略。

问题 2：pairwise agreement 0.75，codebook cosine_max 0.9955

10 个 archetype 之间 75% 动作相同，有些码本向量几乎重叠。但 codebook perplexity=9.84（接近满分 10），说明 encoder 能区分不同轨迹，问题出在 decoder 端——不同的 z_q 输入产生了过于相似的输出。

问题 3：decoded_return (2087) 远超 dp_return (-0.99)

这看起来反直觉，但原因是 single-trade constraint post-processing 在 decoder 的 soft logits 上做了全局最优搜索，有时候找到了比 DP 更好的交易时机。这说明 decoder 的 logits 质量其实不错，瓶颈在于 change point 处的方向判断。

真正有希望的 Phase I 改动（按预期收益排序）：

1. 增加 DP 轨迹数量和多样性（改 DP 采样策略）

早期配置的 30k 轨迹从 525k 个合法起点中无放回采样，覆盖率只有 5.7%。而且 DP 的 single-trade 约束意味着每条轨迹只有 3 种模式（flat→short、flat→long、全 flat）。当前代码默认已把 `num_trajectories` 提高到 50k；如果继续做数据量对照，60k-100k 仍能让 VQ 模型看到更多市场状态下的最优交易模式。这是相对安全的改动：不改模型架构，只增加数据量和训练成本。

1b. DP 起点采样策略提升多样性（已实现）

代码阅读确认这一项已经落地：`Config` 暴露了 `phase1_start_sampling_mode`，`DPPlanner` 支持 `uniform`、`stratified`、`hybrid_stratified_importance` 三种起点采样方式，并会把采样模式、分层/重要性比例、可用起点数量、采样起点索引等元数据写入 trajectory cache。后续实验记录里，单纯改采样不能完全解决 Phase I 的收益语义问题，但它已经成为当前 Phase I 数据结构的一部分。

2. 增大 codebook size K（10 → 16 或 20）

当前默认 K=10。早期诊断里的 pairwise cosine_max=0.9955 说明 10 个 code 可能不够表达当时轨迹的多样性。增大 K 的动机是让每个 archetype 更专注于一种特定的市场模式，减少 archetype 之间的重叠；但后续尝试收益不佳，因此目前不建议把它作为优先主线。论文的 ablation study 也提到 K 太小会 under-represent diverse trading motifs。

3. 增大 latent_dim（16 → 32）

早期 z_e 只有 16 维，要编码 72 步 × 3 种动作的轨迹信息。信息瓶颈太紧，encoder 被迫丢弃细节（比如 change point 的精确位置），decoder 只能恢复粗略的方向。后续代码已把默认 `latent_dim` 调整为 32，以保留更多时序细节。论文说"archetype dimension is set smaller than network dimension to create information bottleneck"，但 16 维可能过于激进。

这几个改动后来已部分落地和验证：1 / 1b / 3 已进入代码，2 已可配置但尝试效果不好。剩余项更适合作为明确实验分支，而不是默认主线。

还有几个方向，我按"改动成本 vs 预期收益"排：

4. 特征增强 — 固定特征 + 周期特征集（已完成）

早期设想是在论文式 45 维状态上补充 price change 和 order imbalance，因为 decoder 需要判断"这一步该不该开仓"，最直接的信号是短期价格变化率和买卖力量对比。当时建议加入的衍生特征包括：

price_return_1：单步收益率 (close[t] - close[t-1]) / close[t-1]
price_return_6：6 步（1 小时）收益率
order_imbalance：(bid_total - ask_total) / (bid_total + ask_total)
spread_ratio：(ask1_price - bid1_price) / close
这些特征直接帮助 decoder 判断 change point 处的方向，不改任何模型架构。

当前代码的实际落地方式不是新增这些同名列，而是把 `feature_pipeline.py` 改成 `24 fixed features + optional cycle feature sets`。CLI 未指定 `--cycle-feature-sets` 时默认启用 `middle`，因此默认状态维度为 `24 + 33 = 57`。`short` / `middle` / `long` 周期特征已经覆盖价格比率、价差、imbalance、成交量、持仓变化和 regime 类信号。

5. Decoder 输入加入 reward 信号

当前 decoder 输入是 [state, z_q]，没有 reward 信息。但 encoder 输入了 [state, action, reward]。这个不对称意味着 decoder 要从 state + z_q 中"猜"出 reward 隐含的信息。如果把 DP 轨迹的 reward 序列也喂给 decoder（训练时用 ground-truth reward，推理时用 0 或估计值），decoder 能更直接地学到"高 reward 时刻应该持仓、低 reward 时刻应该变动"。

6. 训练时数据增强 — 时间翻转

DP 轨迹的 single-trade 结构是 [flat...flat, action, action...action]。可以对每条轨迹做时间翻转：[action...action, flat, flat...flat]（对应"先持仓后平仓"的镜像模式）。这能让 decoder 同时学会"开仓"和"平仓"的 change point 模式，等效于把训练数据翻倍。不过需要注意翻转后 reward 序列也要相应调整。

7. 多尺度 horizon 训练（已尝试，效果不好）

当前固定 h=72。可以在训练时随机截取 h=36、48、72、96 的子序列（padding 到统一长度），让 encoder-decoder 学会在不同时间尺度上识别交易模式。这能提升泛化能力，但改动量较大。

实际尝试后效果不好，暂不建议作为主线继续推进。代码阅读确认当前主线仍是单一 `config.horizon` 流程，没有保留多尺度 horizon / padding 的训练实现。可能原因是多尺度/padding 改变了论文固定 `h=72` 的训练协议，也会破坏 Phase I decoder、Phase II horizon selector 和评估流程之间的长度一致性，使收益信号和单次交易结构更难对齐。

8. Encoder 用 BiLSTM 替代单向 LSTM（已实现；首轮 short 对照效果不好）

当前 encoder 已从单向 LSTM + attention pooling 改为 BiLSTM + attention pooling，让 encoder 同时看到轨迹的前后文。但 `batch_02-short` 首轮结果显示，直接使用单方向 `hidden_dim=128` 的 BiLSTM 会让 encoder 参数量翻倍，并可能把完整示范轨迹中的后验信息编码进 z_e，导致 Phase II selector 更难稳定选择。该方向暂不建议继续作为默认增强，除非配合更小单方向 hidden_dim、checkpoint 健康约束，以及窗口标签对齐修复后重新评估。

9. 增加 VQ 训练的 pretrain epochs

当前 `pretrain_epochs=10`，Phase A 只跑 10 个 epoch 就切到 Phase B 开始 VQ 量化。如果 encoder-decoder 在 Phase A 还没充分收敛，Phase B 的 codebook 初始化质量就差。可以增加到 20-30 个 epoch，让连续 latent space 先充分成型。

## 改动分类与实现状态总表

| 编号 | 改动 | 是否超出论文架构 | 当前代码状态 | 备注 |
|---|---|---|---|---|
| 1 | 增加 DP 轨迹数量 | 否 | 已实现（可直接改配置） | `num_trajectories` 已接入 `Config` 和 `train_phase1.py` |
| 1b | 改 DP 采样策略提升多样性 | 不改模型架构，但超出论文默认数据协议 | 已实现 | `phase1_start_sampling_mode` 支持 `uniform` / `stratified` / `hybrid_stratified_importance`，并写入 cache 元数据 |
| 2 | 增大 codebook size K（10 → 16 / 20） | 否 | 已实现（可直接改配置；已尝试效果不好） | `num_archetypes` 已接入训练和存档，但实验收益不佳 |
| 3 | 增大 latent_dim（16 → 32） | 否 | 已实现（当前默认 32） | `latent_dim` 已接入 encoder / decoder / codebook |
| 4 | 特征增强（固定特征 + 周期特征集） | 不改模型架构，但超出论文固定状态定义 | 已实现 | `feature_pipeline.py` 使用 `24 fixed features + optional cycle feature sets`；CLI 默认 `middle` 时 state_dim=57 |
| 5 | Decoder 输入加入 reward 信号 | 是 | 未实现 | 会把 decoder 从 `p(a_hat | s, z_q)` 改成更强输入版本 |
| 6 | 训练时数据增强（时间翻转） | 不改网络结构，但超出论文原始训练流程 | 未实现 | 属于训练技巧扩展 |
| 7 | 多尺度 horizon 训练 | 是 | 当前未保留/未启用；已尝试效果不好 | 会改固定长度 `h=72` 的数据组织和训练协议；暂不建议继续作为主线 |
| 8 | Encoder 用 BiLSTM 替代单向 LSTM | 是 | 已实现；首轮 `short` 对照效果不好 | 当前 encoder 是 BiLSTM + temporal attention；`hidden_dim` 表示单方向隐藏维度。`batch_02-short` 对照中 Phase II best val 从 154.13 降到 125.20，需先降容量/加 checkpoint 健康约束再重试 |
| 9 | 增加 VQ 训练的 pretrain epochs | 否 | 已实现（可直接改配置） | `pretrain_epochs` 已接入两阶段训练流程 |

补充说明：

- 当前论文定义的 Phase I 是 `encoder: q(z_e | s, a, r)`，`decoder: p(a_hat | s, z_q)`。
- 当前代码里 decoder 和 encoder 都已经是 BiLSTM。
- 第 8 项已经完成一次结构实验；首轮 `short` 结果提醒我们，encoder 侧双向上下文不一定自动提升下游 selector 可用性。

如果要从这些里再挑最值得试的，4 已经完成；第 8 项首轮不佳；在剩余未完成项里，优先考虑 5（decoder 加 reward 输入），但建议等窗口对齐和 checkpoint gate 修正后再判断。理由：

- 特征增强是低风险改动，直接给 decoder 更好的输入信号来判断 change point 方向。
- Decoder 加 reward 修复了 encoder-decoder 之间的信息不对称，让 decoder 不用从 z_q 里"猜" reward 信息。

## 仍可考虑的 DP / 下游优化

- 调整 `gamma`，做接近 1 的对照实验。
- 放宽 single-trade 约束，允许 2 次或更多 change。
- 给窗口末端加 continuation value，而不是把窗口外价值当 0。
- 降低下游 imitation 权重，让 RL 学会偏离 DP 先验。
- 多尺度 horizon 已尝试效果不好，暂不列为优先方向。

## 因子组测试记录

- `short` / `long` 因子组收益约 20%。
- `middle` 因子组收益约 80%。
