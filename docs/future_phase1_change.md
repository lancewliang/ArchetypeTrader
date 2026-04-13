当前 Phase I 的数据告诉我们：

问题 1：change point 处 56.7% 准确率，混淆矩阵显示主要错误是 short→flat (361/881=41%) 和 long→flat (343/892=38%)

decoder 在 change point 处倾向于预测 flat 而不是正确的方向。这不是模型容量问题——BiLSTM 128 维已经足够。问题在于 72 步中只有 1 步是 change point，标准 CE loss 被 71 步的"保持不变"主导，decoder 学到了"有疑问就预测 flat"的保守策略。

问题 2：pairwise agreement 0.75，codebook cosine_max 0.9955

10 个 archetype 之间 75% 动作相同，有些码本向量几乎重叠。但 codebook perplexity=9.84（接近满分 10），说明 encoder 能区分不同轨迹，问题出在 decoder 端——不同的 z_q 输入产生了过于相似的输出。

问题 3：decoded_return (2087) 远超 dp_return (-0.99)

这看起来反直觉，但原因是 single-trade constraint post-processing 在 decoder 的 soft logits 上做了全局最优搜索，有时候找到了比 DP 更好的交易时机。这说明 decoder 的 logits 质量其实不错，瓶颈在于 change point 处的方向判断。

真正有希望的 Phase I 改动（按预期收益排序）：

1. 增加 DP 轨迹数量和多样性（改 DP 采样策略）

当前 30k 轨迹从 525k 个合法起点中无放回采样，覆盖率只有 5.7%。而且 DP 的 single-trade 约束意味着每条轨迹只有 3 种模式（flat→short、flat→long、全 flat）。增加到 60k-100k 轨迹能让 VQ 模型看到更多市场状态下的最优交易模式，直接提升 encoder 的区分能力和 decoder 的泛化能力。这是最安全的改动——不改任何模型架构，只增加数据量。

2. 增大 codebook size K（10 → 16 或 20）

当前 K=10 但 pairwise cosine_max=0.9955，说明 10 个 code 不够表达 30k 条轨迹的多样性。增大 K 能让每个 archetype 更专注于一种特定的市场模式，减少 archetype 之间的重叠。论文的 ablation study 也提到 K 太小会 under-represent diverse trading motifs。

3. 增大 latent_dim（16 → 32）

当前 z_e 只有 16 维，要编码 72 步 × 3 种动作的轨迹信息。信息瓶颈太紧，encoder 被迫丢弃细节（比如 change point 的精确位置），decoder 只能恢复粗略的方向。增大到 32 维能保留更多时序细节。论文说"archetype dimension is set smaller than network dimension to create information bottleneck"，但 16 维可能过于激进。

这三个改动互相独立，可以逐个测试。建议先试第 1 个（增加轨迹数量），因为零风险、不改架构、只需要改 config 里的 num_trajectories。你想从哪个开始？

还有几个方向，我按"改动成本 vs 预期收益"排：

4. 特征增强 — 加入 price change 和 order imbalance（已完成）

当前 45 维特征是原始 LOB 价格/数量 + WAP + trend_60。但 decoder 需要判断"这一步该不该开仓"，最直接的信号是短期价格变化率和买卖力量对比。可以在 feature_pipeline.py 里加 3-5 个衍生特征：

price_return_1：单步收益率 (close[t] - close[t-1]) / close[t-1]
price_return_6：6 步（1 小时）收益率
order_imbalance：(bid_total - ask_total) / (bid_total + ask_total)
spread_ratio：(ask1_price - bid1_price) / close
这些特征直接帮助 decoder 判断 change point 处的方向，不改任何模型架构。

5. Decoder 输入加入 reward 信号

当前 decoder 输入是 [state, z_q]，没有 reward 信息。但 encoder 输入了 [state, action, reward]。这个不对称意味着 decoder 要从 state + z_q 中"猜"出 reward 隐含的信息。如果把 DP 轨迹的 reward 序列也喂给 decoder（训练时用 ground-truth reward，推理时用 0 或估计值），decoder 能更直接地学到"高 reward 时刻应该持仓、低 reward 时刻应该变动"。

6. 训练时数据增强 — 时间翻转

DP 轨迹的 single-trade 结构是 [flat...flat, action, action...action]。可以对每条轨迹做时间翻转：[action...action, flat, flat...flat]（对应"先持仓后平仓"的镜像模式）。这能让 decoder 同时学会"开仓"和"平仓"的 change point 模式，等效于把训练数据翻倍。不过需要注意翻转后 reward 序列也要相应调整。

7. 多尺度 horizon 训练

当前固定 h=72。可以在训练时随机截取 h=36、48、72、96 的子序列（padding 到统一长度），让 encoder-decoder 学会在不同时间尺度上识别交易模式。这能提升泛化能力，但改动量较大。

8. Encoder 用 BiLSTM 替代单向 LSTM

当前 encoder 是单向 LSTM + attention pooling。改成 BiLSTM 能让 encoder 同时看到轨迹的前后文，更准确地编码 change point 的位置和方向信息到 z_e 中。z_e 质量提升 → codebook 分配更准确 → decoder 收到更有区分度的 z_q。

9. 增加 VQ 训练的 pretrain epochs

当前 pretrain_epochs=10，Phase A 只跑 10 个 epoch 就切到 Phase B 开始 VQ 量化。如果 encoder-decoder 在 Phase A 还没充分收敛，Phase B 的 codebook 初始化质量就差。可以增加到 20-30 个 epoch，让连续 latent space 先充分成型。

## 改动分类与实现状态总表

| 编号 | 改动 | 是否超出论文架构 | 当前代码状态 | 备注 |
|---|---|---|---|---|
| 1 | 增加 DP 轨迹数量 | 否 | 已实现（可直接改配置） | `num_trajectories` 已接入 `Config` 和 `train_phase1.py` |
| 1b | 改 DP 采样策略提升多样性 | 不改模型架构，但超出论文默认数据协议 | 未实现 | 当前只有滑窗合法起点 + 随机采样， |
| 2 | 增大 codebook size K（10 → 16 / 20） | 否 | 已实现（可直接改配置） | `num_archetypes` 已接入训练和存档 效果不好  |
| 3 | 增大 latent_dim（16 → 32） | 否 | 已实现（可直接改配置） | `latent_dim` 已接入 encoder / decoder / codebook。|
| 4 | 特征增强（price change / imbalance / spread） | 否 | 已实现 | 属于输入特征增强，不改 Phase I 核心结构 |
| 5 | Decoder 输入加入 reward 信号 | 是 | 未实现 | 会把 decoder 从 `p(a_hat | s, z_q)` 改成更强输入版本 |
| 6 | 训练时数据增强（时间翻转） | 不改网络结构，但超出论文原始训练流程 | 未实现 | 属于训练技巧扩展 |
| 7 | 多尺度 horizon 训练 | 是 | 未实现 | 会改固定长度 `h=72` 的数据组织和训练协议 |
| 8 | Encoder 用 BiLSTM 替代单向 LSTM | 是 | 未实现 | 当前 encoder 仍是单向 LSTM + temporal attention |
| 9 | 增加 VQ 训练的 pretrain epochs | 否 | 已实现（可直接改配置） | `pretrain_epochs` 已接入两阶段训练流程 |

补充说明：

- 当前论文定义的 Phase I 是 `encoder: q(z_e | s, a, r)`，`decoder: p(a_hat | s, z_q)`。
- 当前代码里 decoder 已经是 BiLSTM，但 encoder 还不是 BiLSTM。
- 因此第 8 项不是“继续沿用现状”，而是一次新的结构修改。

如果要我从这些里再挑最值得试的，我会把 4 标记为已完成；在剩余未完成项里，优先考虑 5（decoder 加 reward 输入）。理由：

特征增强是零风险改动，直接给 decoder 更好的输入信号来判断 change point 方向
Decoder 加 reward 修复了 encoder-decoder 之间的信息不对称，让 decoder 不用从 z_q 里"猜" reward 信息



dp优化
增大或多尺度化 horizon
把 gamma 往 1 调近一点做对照实验
放宽 single-trade 约束，允许 2 次或更多 change
给窗口末端加 continuation value，而不是把窗口外价值当 0
降低下游 imitation 权重，让 RL 学会偏离 DP 先验


因子组测试完了
short/long 只有20%收益
middle 有80%收益
