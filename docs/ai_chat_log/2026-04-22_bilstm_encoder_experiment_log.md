# 2026-04-22 BiLSTM Encoder 实验记录

## 背景

本次实验验证 Phase I 的第 8 个候选改动：将 VQ encoder 从单向 LSTM + temporal attention pooling 替换为 BiLSTM + temporal attention pooling。

实验日志：

- 对照实验：`logs/AL/batch_01-short/AL_pipeline_20260422_151607.log`
- BiLSTM 实验：`logs/AL/batch_02-short/AL_pipeline_20260422_215103.log`

两次实验都使用 `AL`、`short` 因子组。`batch_02-short` 的 Phase II 训练在 step 488,448 / 1,000,000 截止，但它的最佳验证收益出现在 step 61,440，因此已经足够暴露该配置的主要问题。

## 关键结果

| 指标 | batch_01-short（单向 encoder） | batch_02-short（BiLSTM encoder） |
|---|---:|---:|
| Encoder 参数量 | 332,064 | 664,096 |
| Phase I profit gate 命中 | 9/13 | 1/13 |
| 选中 checkpoint | epoch 240 | epoch 120 |
| realizable proxy mean | 182.91 | 152.18 |
| realizability score | 0.965 | 0.777 |
| Phase II best val avg_return | 154.13 | 125.20 |
| Phase II last observed val avg_return | 133.39 | 114.15 |
| 选中 checkpoint token accuracy | 0.9998 | 0.9706 |
| 选中 checkpoint exact match | 0.9906 | 0.7373 |
| quantization MSE | 0.375 | 8.202 |

## 诊断

1. BiLSTM encoder 没有带来更好的下游可选原型。

   `batch_02-short` 的 Phase I 只有 1/13 个 checkpoint 通过 profit gate，而 `batch_01-short` 有 9/13 个通过。这说明问题不是 Phase II 后半段没跑完，而是 Phase I 候选 checkpoint 的下游可交易性整体变差。

2. 被选中的 epoch 120 是一个收益 proxy 过关但健康度较差的 checkpoint。

   epoch 120 的 `exact_match=0.7373`、`quantization_mse=8.202`，显著差于 `batch_01-short` 选中的 epoch 240（`exact_match=0.9906`、`quantization_mse=0.375`）。它能过 profit gate，说明当前 gate 还缺少足够强的重构/量化健康约束。

3. Phase II 选择分布塌到少数原型。

   `batch_02-short` 在 step 61,440 的验证集选择分布几乎集中在 `k=4` 和 `k=7`：`k4=406`、`k7=1056`。这更像是在押少数固定策略，而不是按每个 horizon 稳定选择不同 archetype。

4. BiLSTM 可能放大了 encoder/selector 的信息不对称。

   Phase I encoder 输入完整 `(s_demo, a_demo, r_demo)` 轨迹，BiLSTM 又能同时看前后文；Phase II selector 只看当前 horizon 的市场状态。更强的 encoder 可能把完整示范轨迹里的后验信息压进 z_e/code，导致 selector 更难预测。

5. `short` 因子组本身偏弱，但不是这次退化的唯一原因。

   历史记录显示 `middle` 因子组通常明显强于 `short` / `long`。不过这次对照的两组都是 `short`，因此 batch_02 相对 batch_01 的退化仍主要来自 BiLSTM encoder 配置和 checkpoint 选择。

## 结论

第 8 项“Encoder 用 BiLSTM 替代单向 LSTM”已经完成实现和首轮实验，但 `hidden_dim=128` 单方向的 BiLSTM encoder 不应直接作为默认增强。当前证据显示，它让 Phase I 更强表达，但没有让 Phase II 更容易选择，反而降低了可实现的验证收益。

## 后续建议

- 先修 Phase I/Phase II 的窗口标签对齐问题，不急着继续堆 Phase I 架构。
- 如果继续保留 BiLSTM encoder，先试 `hidden_dim=64` 单方向，使双向拼接后的维度接近原单向 128。
- Phase I checkpoint gate 加入健康约束，例如 `exact_match >= 0.95`、`quantization_mse` 上限、`phase2_realizability_score` 下限。
- 用 `middle` 因子组做同配置对照，确认这次退化是否会在更强特征组上缓解。
- 在窗口对齐和 gate 修正前，暂缓 decoder 加 reward、时间翻转等更大结构实验。
