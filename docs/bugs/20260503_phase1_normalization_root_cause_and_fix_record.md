# Phase I 归一化问题根因分析与修复记录

**日期**: 2026-05-03
**影响阶段**: Phase I Archetype Discovery, Phase II state 输入对齐
**涉及标的与批次**: `FU / batch_007`
**主日志链路**:
- `logs/FU/batch_007/ts-20260503-111514-1550742.log`
- `logs/FU/batch_007/ts-20260503-144240-1811188.log`
- `logs/FU/batch_007/ts-20260503-150029-1837767.log`
- `logs/FU/batch_007/ts-20260503-151246-1856218.log`

---

## 1. 背景与问题定义

本次问题最初表现为 Phase I 训练结果很差，具体症状是:

- Phase A 重构损失长期停留在接近随机分类水平。
- Phase B 很快进入 codebook collapse, 大部分样本只使用 1 个 code。
- validation 上 decoder 对 code 不敏感，不同 code 解码出的 action 几乎一样。
- best checkpoint 无法产生，训练结束后 `best_epoch=None`。

后来排查确认，主要根因不是 VQ loss 本身，也不是 DP teacher 完全失效，而是**模型 state 输入没有做训练集拟合的尺度归一化**。原始状态特征中包含 `turnover`、`total_trade_volume`、`open_interest`、盘口 size/volume 等大数量级字段，最大绝对值达到 `4e10` 量级，直接进入 `state_adapter` 和 LSTM，导致连续输入尺度压倒其他特征。

本次记录重点覆盖两个归一化修复:

1. **状态特征归一化**: 解决模型训练不收敛和 codebook collapse 的核心问题。
2. **风险指标归一化**: 解决修复状态输入后，`val_max_drawdown` 仍因 raw PnL 口径错误而阻止 best 晋升的问题。

---

## 2. 发现过程

### 2.1 第一版日志显示输入尺度异常

在 `ts-20260503-111514-1550742.log` 中，训练刚开始就出现:

```text
phase1_amp_disabled_for_input_scale
max_abs=4.561249e+10
state_abs_max=4.561249e+10
reward_abs_max=5.000000e+00
```

这个诊断说明状态输入尺度已经超过 fp16 安全阈值，因此训练被迫对超阈值 batch 使用 fp32 前向。`reward_abs_max=5` 相对正常，异常主要来自 `states`。

同一 run 的 Phase A 指标也异常:

| epoch | phase | loss_rec | 现象 |
|---:|---|---:|---|
| 0 | A | 1.130507 | 接近三分类随机水平 |
| 1 | A | 1.092113 | 几乎没有有效下降 |
| 9 | A | 1.081578 | Phase A 结束仍接近随机 |

三分类随机交叉熵约为 `ln(3)=1.099`。Phase A 训练 10 个 epoch 后仍在这个区间，说明 encoder/decoder 没有从状态序列中学到可用重构信息。

### 2.2 Codebook collapse 是后果，不是第一原因

同一日志进入 Phase B 后，collapse 迅速发生:

| epoch | code_unique | code_entropy | loss_rec | selection 结果 |
|---:|---:|---:|---:|---|
| 10 | 2/10 | 0.6913/2.3026 | 1.078594 | skipped |
| 12 | 1/10 | 0.0000/2.3026 | 1.077054 | skipped |
| 14 | 1/10 | 0.0000/2.3026 | 1.073293 | reject |

epoch 14 的 validation:

```text
decision=reject
code_usage_ratio=0.1
val_return_capture_ratio=-0.376968
val_sharpe_ratio=-19.494
reasons=codebook_collapse, risk_guardrail, behavior_guardrail
```

这说明 codebook collapse 确实发生了，但它是在模型没有学到有效 latent 表示之后发生的。真正需要先处理的是输入尺度。

### 2.3 代码路径确认没有 state normalizer

排查训练链路后发现:

- `HorizonBuilder` 将 schema 中的 feature columns 切成 `HorizonRecord.states`。
- `Phase1DemoDataset` 将 `states` 直接转成 tensor。
- `VQArchetypeModel.input_adapter` 直接消费 raw states。
- 训练中已有 reward normalizer，但没有 state normalizer。

也就是说，价格、盘口 size、成交量、持仓量、turnover 等完全不同量纲的字段被直接拼到一起进入网络。

---

## 3. 根因分析

### 3.1 原始大数量级字段支配网络输入

FU batch 中状态特征最大绝对值达到 `4.561249e+10`。这种尺度会造成:

- `state_adapter` 的线性层输出被少数大字段主导。
- LSTM 输入动态范围过大，梯度主要围绕大数量级字段震荡。
- action/reward embedding 和较小量纲的价格变化类特征被淹没。
- mixed precision 无法安全使用，虽然 fp32 避免 NaN，但不解决可学习性。

### 3.2 Phase A 没学好导致 Phase B 无法健康量化

Phase A 的目标是先让 encoder/decoder 学会重构 action 序列。只有 Phase A 输出的 `z_e` 有意义，后面的 codebook warmup/k-means 才有价值。

在未归一化时，Phase A 重构损失没有明显下降，导致:

1. encoder 输出的 `z_e` 信息量低。
2. codebook warmup 基于低质量 latent 初始化。
3. Phase B commitment 和 usage 正则无法挽救表示质量。
4. 最终表现为 1 个 code 被大量使用，decoder 对 code 不敏感。

### 3.3 只靠 dead code restart 不能解决输入尺度

旧 run 中 dead code restart 能短暂提高 `code_unique`，例如 epoch 34/39 出现重启，但 validation 仍无法通过。原因是 restart 只是在 codebook 空间重置向量，不能改变 encoder 输入已经失衡的问题。

---

## 4. 状态特征归一化方案

### 4.1 设计原则

状态归一化必须满足:

- **只用 train split 拟合**，避免 validation/test 泄漏。
- **特征列顺序可审计**，Phase II 必须使用完全相同的 feature columns 和变换。
- **保留 prices/execution books/rewards 原始口径**，只处理模型 state features。
- **对大数量级成交/持仓字段先压缩长尾**，再做稳健缩放。
- **输出范围可控**，避免再次触发 AMP 尺度保护。

### 4.2 具体算法

新增 `StateNormalizer`，方法名为 `train_state_robust_v1`。

处理流程:

1. 收集 train horizons 中所有 timestep 的 state matrix, shape 为 `[N * horizon, feature_dim]`。
2. 对大数量级字段做 signed `log1p`:
   - `turnover`
   - `total_trade_volume`
   - `open_interest`
   - 以 `_size` 或 `_volume` 结尾的字段
   - 排除包含 `ratio`、`zscore` 或以 `log_return` 开头的字段
3. 对每个 feature 用 train split 计算 median。
4. 用 `1.4826 * MAD` 作为 robust scale。
5. 对近似常数或 MAD 太小的字段 fallback 到 standard deviation。
6. 如果仍然太小，则使用 scale=1。
7. 输出 `(x - center) / scale` 并 clip 到 `[-8, 8]`。

选择 robust z-score 而不是普通 mean/std 的原因:

- 成交量、open interest、turnover 类字段通常长尾明显。
- 普通均值和标准差会被极端值拉偏。
- median/MAD 对异常点更稳，和 clip 配合后能保证网络输入范围稳定。

---

## 5. 代码修改记录

### 5.1 新增 StateNormalizer

文件: `src/data/state_normalizer.py`

主要能力:

- `StateNormalizer.fit_records(...)`
- `StateNormalizer.fit_matrix(...)`
- `StateNormalizer.transform_records(...)`
- `StateNormalizer.transform_array(...)`
- `StateNormalizer.to_dict() / from_dict() / load_json()`

持久化内容包括:

- `method`
- `feature_columns`
- `transform_kinds`
- `center`
- `scale`
- `clip_value`
- `scale_floor`
- `max_abs_before`
- `max_abs_after_fit`
- `fallback_to_standard_count`

### 5.2 Phase I 训练入口接入 state normalizer

文件: `src/trainers/phase1_trainer.py`

修改点:

- 加载 manifest 后，在 reward normalizer 之前拟合 `StateNormalizer`。
- 只用 `train_horizons` 拟合。
- 将 `state_normalizer.json` 写入 `artifacts/{pair}/{batch}/phase1/`。
- 对 train/val/test 的 `HorizonRecord.states` 原地 transform。
- 日志输出 fit 诊断:

```text
phase1_state_normalizer_fit
train_max_abs_before=...
train_max_abs_after=...
val_max_abs_before=...
val_max_abs_after=...
test_max_abs_before=...
test_max_abs_after=...
log_transform_columns=...
fallback_to_standard_count=...
```

- `TrainerArtifacts` 增加 `state_normalizer_json`。
- final summary 增加:
  - `state_normalization_resolved`
  - `state_norm_max_abs_before`
  - `state_norm_max_abs_after_fit`

### 5.3 Phase II 读取相同 state normalizer

文件: `src/data/phase2_dataset.py`

修改点:

- 初始化 Phase II dataset 时尝试读取 `phase1_dir/state_normalizer.json`。
- 读取 raw frame 的 feature matrix 后，先应用 Phase I state normalizer。
- 校验 `state_normalizer.feature_columns` 必须与 Phase II input schema 一致。
- 对旧 fixture 或旧 artifact 兼容: 缺少 normalizer 时允许继续使用 raw states。

文件: `src/data/phase2_horizon_index.py`

修改点:

- Phase I artifact 校验时，如果缺少 `state_normalizer.json`，增加 warning。
- 不把缺失 normalizer 作为硬错误，以兼容历史产物。

### 5.4 测试补充

新增或更新测试:

- `tests/unit/data/test_state_normalizer.py`
  - 大数量级特征会被压到 clip 范围内。
  - `turnover/open_interest` 会走 signed `log1p`。
  - dict roundtrip 后 transform 一致。
  - feature dim mismatch 会报错。
- `tests/unit/data/test_phase2_dataset.py`
  - Phase II dataset 会应用 Phase I 的 `state_normalizer.json`。
- `tests/integration/test_phase1_pipeline_smoke.py`
  - Phase I smoke artifact 包含 `state_normalizer_json`。
- `tests/integration/test_phase1_data_process_then_train.py`
  - data process 到 train 的集成产物包含 `state_normalizer_json`。

---

## 6. 修复后第一轮观察

### 6.1 输入尺度被压住

在 `ts-20260503-144240-1811188.log` 和后续 run 中出现:

```text
phase1_state_normalizer_fit
method=train_state_robust_v1
feature_dim=47
clip=8.00
train_max_abs_before=4.665508e+10
train_max_abs_after=8.000000e+00
val_max_abs_before=4.096696e+10
val_max_abs_after=8.000000e+00
test_max_abs_before=2.859775e+10
test_max_abs_after=8.000000e+00
log_transform_columns=13
fallback_to_standard_count=5
```

对比未修复 run:

| 项目 | 未归一化 | 状态归一化后 |
|---|---:|---:|
| train state max abs | `4.561249e+10` | `8.0` |
| AMP 尺度保护 | 触发 `phase1_amp_disabled_for_input_scale` | 未再看到同类警告 |
| log transform columns | 无 | 13 |
| fallback to std | 无 | 5 |

### 6.2 Phase A 终于开始学习

| 指标 | 未归一化 run `11:15` | 归一化后 run `14:42/15:12` |
|---|---:|---:|
| Phase A epoch 0 loss_rec | 1.130507 | 0.966135 |
| Phase A epoch 1 loss_rec | 1.092113 | 0.751121 |
| Phase A epoch 2 loss_rec | 1.075884 | 0.571640 |
| Phase A epoch 9 loss_rec | 1.081578 | 0.304119 |

未归一化时 Phase A 基本停在随机分类附近。归一化后，Phase A loss 从 `0.966` 降到 `0.304`，说明 encoder/decoder 已经能从 state 序列中提取可用信号。

### 6.3 Phase B codebook 使用恢复

| 指标 | 未归一化 run | 归一化后 run |
|---|---:|---:|
| Phase B epoch 10 code_unique | 2/10 | 10/10 |
| Phase B epoch 12 code_unique | 1/10 | 10/10 |
| epoch 14 code_usage_ratio | 0.1 | 0.8 到 0.9 |
| epoch 39 code_usage_ratio | 0.5 | 1.0 |
| `inter_code_action_diversity` | 0.000 | 约 0.63 |
| `decoder_sensitivity_to_code` | 接近 0 | 约 3.6 到 3.9 |

这说明原先的 hard collapse 和 decoder 对 code 不敏感问题已经被实质性缓解。

---

## 7. 后续发现: 风险指标也存在归一化口径问题

状态归一化修复后，训练质量明显改善，但 `ts-20260503-144240-1811188.log` 仍然没有 best。原因变成:

```text
risk_guardrail: val_max_drawdown=13.479 > 0.2
```

当时 epoch 39 的其他指标已经较好:

- `code_usage_ratio=1.0`
- `val_return_capture_ratio=0.562102`
- `val_sharpe_ratio=30.498590`
- `weighted_reconstruction_accuracy` 约 `0.96`

继续排查发现，`src/evaluation/metrics/risk.py` 的 `equity_curve_from_step_returns` 假定输入是 step return ratio，并从初始净值 `1.0` 开始累加。但 Phase I replay 输出的是 raw PnL:

```text
reward = position * (p_markout - p_exec) - cost
```

也就是说，原逻辑等价于把几百、几千的 raw PnL 当成百分比收益率累加，导致 `val_max_drawdown` 出现 `373`、`142`、`13` 这类不合理数值。selection policy 的 `risk.max_drawdown=0.2` 本来表示 20% 回撤，但输入指标不是同一量纲。

### 7.1 风险指标归一化方案

文件:

- `src/evaluation/metrics/risk.py`
- `src/evaluation/phase1_evaluator.py`

修改点:

- 新增 `step_returns_from_pnl(step_pnl, capital_base)`。
- 新增 `equity_curve_from_step_pnl(...)`。
- 新增 `cumulative_pnl_curve(...)`。
- 新增 `max_drawdown_abs(...)`，保留 raw PnL 绝对回撤审计。
- Phase I evaluator 中估算 `risk_capital_base`:
  - 优先使用显式传入的 `risk_capital_base`。
  - 否则用 validation/probe records 的 median price。
  - 再乘以 env 的 `max_position`。
- `val_max_drawdown` 改为基于 normalized return ratio 的净值曲线。
- 输出审计字段:
  - `val_risk_capital_base`
  - `val_max_drawdown_abs`
  - `val_annual_return_ratio`

### 7.2 风险口径修正后的效果

在 `ts-20260503-150029-1837767.log` 和 `ts-20260503-151246-1856218.log` 中，best 开始正常晋升。

epoch 59:

| 指标 | 数值 |
|---|---:|
| `decision` | `promote_to_best` |
| `phase1_composite_score` | 4.664933 |
| `code_usage_ratio` | 1.0 |
| `val_return_capture_ratio` | 0.680050 |
| `val_sharpe_ratio` | 38.634662 |
| `val_max_drawdown` | 0.014319 |
| `val_max_drawdown_abs` | 551.9275 |
| `val_risk_capital_base` | 32385.0 |
| `val_student_online_net_return` | 7183.065 |
| `val_dp_teacher_net_return` | 10562.5475 |
| `val_weighted_reconstruction_accuracy` | 0.972591 |

这说明 risk guardrail 现在比较的是同一量纲的回撤比例，而 raw PnL 绝对回撤仍保留在 `val_max_drawdown_abs` 中用于审计。

---

## 8. 最终效果与剩余状态

### 8.1 训练与 artifact 导出状态

最新日志 `ts-20260503-151246-1856218.log` 显示:

```text
epoch=59 decision=promote_to_best score=4.664933
phase1_train_loop_done best_epoch=59
phase1_horizon_labels_exported
phase1_phase2_artifacts_exported
```

这说明:

- Phase I 已经能产生 best checkpoint。
- horizon labels 已经能导出。
- Phase II 所需的 `encoder.pt`、`decoder.pt`、`codebook.pt` 已经能导出。

该 run 最后仍触发 fatal:

```text
缺少 prospective 对照报告:
artifacts/FU/batch_002_prospective_strata/phase1/phase1_report.json;
主实验不可 sign-off。
```

这个阻塞与本次归一化修复无关，它属于实验 sign-off 的 prospective 对照报告缺失。

### 8.2 指标对比总表

| 阶段 | 日志 | 关键状态 | 结果 |
|---|---|---|---|
| 未修复 | `ts-20260503-111514-1550742.log` | `state_abs_max=4.56e10`, Phase A loss 约 1.08, code collapse 到 1/10 | `best_epoch=None` |
| 状态归一化后 | `ts-20260503-144240-1811188.log` | state max abs 压到 8, Phase A loss 到 0.304, code usage 恢复 | 被错误 MDD 口径拦住 |
| 风险口径修正后 | `ts-20260503-150029-1837767.log` | epoch 59 晋升 best | 标签导出 OOM 暴露 |
| 分批导出后 | `ts-20260503-151246-1856218.log` | best、labels、Phase II artifacts 均导出 | 仅剩 prospective report sign-off 阻塞 |

---

## 9. 为什么这个方案是正确修复

本次修复不是单纯放宽 guardrail，而是把输入和指标恢复到合理量纲:

- State normalizer 解决的是训练输入量纲问题，直接改善 Phase A 学习和 Phase B codebook 使用。
- Risk PnL normalizer 解决的是评估指标量纲问题，让 `max_drawdown=0.2` 重新表示 20% 回撤。
- Code usage、decoder sensitivity、inter-code diversity、weighted reconstruction、return capture、Sharpe 都同步改善，说明不是某个单点指标被调参“骗过”。
- Guardrail 没有被关闭，epoch 19/24/29/34/39/44/49/54 仍会因为 `epoch_code_stability < 0.8` 被拒绝，直到 epoch 59 稳定性和收益指标同时达标后才 promote。

---

## 10. 已执行验证

相关测试曾分批执行并通过:

```bash
/home/lanceliang/miniconda3/envs/ArchetypeTrade/bin/python -m pytest \
  tests/unit/data/test_state_normalizer.py \
  tests/unit/data/test_phase2_dataset.py \
  tests/unit/trainers/test_phase1_trainer.py
```

```bash
/home/lanceliang/miniconda3/envs/ArchetypeTrade/bin/python -m pytest \
  tests/unit/evaluation/metrics/test_risk.py \
  tests/unit/evaluation/test_phase1_evaluator.py \
  tests/unit/trainers/test_phase1_selection_policy.py \
  tests/unit/trainers/test_phase1_trainer.py \
  tests/integration/test_phase1_pipeline_smoke.py \
  tests/integration/test_phase1_data_process_then_train.py
```

最后一次相关测试结果:

```text
32 passed
```

---

## 11. 后续建议

1. 把 `state_normalizer.json` 视为 Phase I 到 Phase II 的必要语义产物。旧 artifact 可以 warning 兼容，但正式实验不应缺失。
2. 在最终 report 中持续记录 `state_norm_max_abs_before` 和 `state_norm_max_abs_after_fit`，用于快速判断新 batch 是否存在输入尺度异常。
3. 如果更换 feature schema，必须重新拟合 state normalizer，禁止复用旧 normalizer。
4. 风险指标报告中同时保留 ratio MDD 和 absolute PnL MDD，避免以后再次混淆交易 PnL 和收益率。
5. 当前剩余阻塞是 prospective 对照报告缺失，不是 Phase I 训练或归一化问题。
