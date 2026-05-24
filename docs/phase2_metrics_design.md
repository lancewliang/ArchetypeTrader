# Phase II 指标设计文档

## 1. 设计目标

Phase II 的目标不是简单复现 Phase I 的 assigned archetype label，而是在当前在线可见市场状态下选择最合适的 archetype，并通过冻结的 Phase I decoder 生成 horizon 内的 base actions，最终获得更好的 horizon-level trading return。

因此 Phase II 指标需要同时回答以下问题：

1. 评估数据和执行口径是否可信，是否存在泄露、错位或无效 rollout。
2. selector greedy policy 是否真的产生正收益，并具备可接受的风险特征。
3. selector 是否优于 assigned-label baseline、random baseline，并接近 hindsight oracle code 上界。
4. selector 偏离 Phase I assigned label 时，是否用收益证明了这种偏离。
5. selector 是否合理使用 Phase I 学到的 archetype set，而不是发生 code collapse。
6. checkpoint 的表现是否稳定，是否存在训练过拟合、validation 偶然尖峰或 Q value 发散。

Phase II 指标建议采用分层结构，参考 Phase I 的 validation layer 思路，将 hard gate、reference diagnostics、score 和 tie-breaker 分开。

## 2. 分层总览

| Layer | 名称 | 核心问题 | 是否建议作为 hard gate |
| --- | --- | --- | --- |
| Layer 0 | Evaluation Validity | 评估结果是否可信 | 是 |
| Layer 1 | Selector Profitability | selector 自身是否赚钱 | 是 |
| Layer 2 | Baseline Uplift | 是否优于 baseline | 是 |
| Layer 3 | Demonstration Consistency | 偏离 Phase I label 是否合理 | 部分 hard gate |
| Layer 4 | Code Usage And Collapse | 是否充分且稳定地使用 archetype | 部分 hard gate + warn |
| Layer 5 | Generalization And Stability | 是否泛化、收敛、稳定 | 主要 warn + tie-breaker |

## 3. Layer 0: Evaluation Validity

### 目标

确认 Phase II validation/test 的指标口径可靠。该层不评价模型好坏，只判断评估结果是否可以被后续 checkpoint selector 使用。

### 指标

`num_samples`  
评估 split 中实际参与计算的 horizon 样本数。样本数过低时，收益、分位数和 per-code 统计都不稳定。

`valid_rollout_ratio`  
成功完成 selector action、frozen decoder、execution reward 计算的样本比例。低于 1 说明部分样本执行失败，结果不可完整审计。

`finite_reward_ratio`  
reward、gross return、fee、turnover 中非 NaN、非 inf 的比例。任何非有限值都可能污染均值、Sharpe 或 checkpoint 排序。

`valid_selected_code_ratio`  
selected code 落在 `[0, K)` 范围内的比例。低于 1 表示 selector 输出或后处理存在严重错误。

`deterministic_eval`  
validation/test 是否使用 deterministic greedy action。正式 checkpoint selection 应禁用 epsilon exploration。

`label_alignment_valid`  
`sample_id/code_label` 是否和 horizon dataset 对齐。错位会让 imitation、baseline 和 consistency 指标失真。

`visible_state_contract_valid`  
selector observation 是否只包含 previous horizon 和 current t-window，不包含当前 horizon 的未来状态、价格、reward 或 teacher action。

### 建议 gate

- `num_samples >= min_eval_samples`
- `valid_rollout_ratio == 1.0`
- `finite_reward_ratio == 1.0`
- `valid_selected_code_ratio == 1.0`
- `deterministic_eval == true`
- `label_alignment_valid == true`
- `visible_state_contract_valid == true`

## 4. Layer 1: Selector Profitability

### 目标

直接衡量 selector greedy policy 在统一交易执行口径下的收益、风险和交易成本。

### 指标

`mean_return`  
selector 选择 archetype 后的平均 horizon return，是 Phase II 最核心的收益指标。

`median_return`  
收益中位数，降低极端样本对整体判断的影响。若 mean 为正但 median 为负，说明收益可能依赖少数大盈利样本。

`total_return`  
所有 horizon return 的累计值，用于和整体资金曲线方向对齐。

`win_rate`  
`return > 0` 的样本比例。高 win rate 不一定代表高收益，但低 win rate 通常表示策略稳定性不足。

`sharpe_like`  
`mean_return / std_return`，衡量单位波动下的平均收益。这里是 horizon-level 类 Sharpe，不等同于年化 Sharpe。

`downside_sharpe_like`  
`mean_return / std(negative_returns)`，更关注下行波动。

`p05_return`  
return 的 5% 分位数，衡量左尾风险。该指标比最小值更稳健。

`loss_rate`  
`return < 0` 的比例。可与 win rate 互相校验。

`mean_gross_return`  
扣费前平均收益，用于区分策略方向是否有效和手续费是否过高。

`mean_fee`  
平均手续费。

`fee_drag_ratio`  
`mean_fee / abs(mean_gross_return)`。如果该值过高，说明策略可能被频繁换手和交易成本吞噬。

`mean_turnover`  
平均换手率或行为强度。过高可能带来 fee drag，过低可能说明 selector 退化为长期 flat 或固定低活动策略。

### 建议 gate

- `mean_return > 0`
- `win_rate >= 0.5`
- `sharpe_like > 0`
- `p05_return >= max_allowed_tail_loss`
- `fee_drag_ratio <= max_fee_drag_ratio`

## 5. Layer 2: Baseline Uplift

### 目标

判断 Phase II selector 是否真正带来了选择价值，而不是只复现 assigned label 或随机选 code。

### Baseline

`assigned_label_baseline`  
使用 Phase I assigned code label 执行 frozen decoder。该 baseline 表示“不训练 Phase II selector，只按 Phase I label 选择”的效果。

`random_baseline`  
随机选择 archetype。建议多 seed 平均，用于衡量 selector 是否至少优于随机 code selection。

`oracle_code_baseline`  
对每个样本枚举所有 K 个 code，取 hindsight return 最高的 code。该 baseline 使用未来信息，只作为上界参考，不能用于训练或推理。

### 指标

`assigned_mean_return`  
assigned-label baseline 的平均 return。

`random_mean_return`  
random baseline 的平均 return。

`oracle_mean_return`  
hindsight oracle code baseline 的平均 return。

`uplift_vs_assigned`  
`selector_mean_return - assigned_mean_return`。衡量 Phase II 是否优于 Phase I assigned label。

`uplift_vs_random`  
`selector_mean_return - random_mean_return`。衡量 selector 是否学到了有效选择。

`relative_uplift_vs_assigned`  
`uplift_vs_assigned / abs(assigned_mean_return)`。用于不同标的、不同收益尺度间比较。

`oracle_capture_ratio`  
`selector_mean_return / oracle_mean_return`。衡量 selector 捕获 hindsight oracle 上界的比例。

`regret_to_oracle`  
`oracle_mean_return - selector_mean_return`。越小表示离 hindsight 最优 code 越近。

`beat_assigned_rate`  
样本级 `selector_return > assigned_return` 的比例。

`beat_random_rate`  
样本级 `selector_return > random_return` 的比例。

### 建议 gate

- `uplift_vs_random > 0`
- `beat_random_rate > 0.5`
- 保守版本：`uplift_vs_assigned >= -small_tolerance`
- 激进版本：`uplift_vs_assigned > 0`

## 6. Layer 3: Demonstration Consistency

### 目标

对应论文中的 imitation/KL regularization。selector 可以偏离 assigned label，但偏离应当带来收益，而不是破坏 Phase I 已学习到的 archetype 语义。

### 指标

`label_match_rate`  
selected code 等于 Phase I assigned label 的比例。过高可能表示 Phase II 只是模仿，过低且无收益提升则表示选择漂移。

`cross_entropy_to_assigned`  
selector policy softmax 后对 assigned label 的 cross entropy。用于衡量和 demonstration label 的距离。

`kl_to_assigned_onehot`  
与 assigned one-hot label 的 KL/CE 等价指标。可作为 imitation regularization 的评估口径。

`label_q_margin`  
selected code Q value 与 assigned label Q value 的差值。用于判断 selector 偏离 assigned label 时是否有明确 Q-value 支持。

`profitable_deviation_rate`  
偏离 assigned label 且 `selector_return > assigned_return` 的样本比例。

`unprofitable_deviation_rate`  
偏离 assigned label 且 `selector_return < assigned_return` 的样本比例。该指标是关键风险指标。

`deviation_return_delta`  
只在偏离样本上计算 `selector_return - assigned_return` 的平均值。

### 建议 gate

- `unprofitable_deviation_rate <= max_bad_deviation_rate`
- `label_match_rate >= min_consistency_floor`
- 如果 `label_match_rate < min_consistency_floor`，则要求 `uplift_vs_assigned > stronger_uplift_threshold`

## 7. Layer 4: Code Usage And Collapse

### 目标

判断 selector 是否真正根据市场状态选择不同 archetype，还是塌缩为只选一个或少数几个 code。Phase I 学到的是 reusable archetype set，Phase II 应该学习何时使用哪个 archetype。

### 指标

`selected_code_entropy`  
衡量 selector selected code 分布的熵。熵越高表示 code 使用越分散；熵接近 0 表示几乎只选择一个 code。  
异常解释：过低通常表示 code collapse；过高但收益较差，可能表示 selector 接近随机乱选。

`selected_code_perplexity`  
`exp(selected_code_entropy)`，可理解为“等效使用了多少个 code”。例如 K=10 但 perplexity=1.3，说明实际接近只使用 1 个 code。  
异常解释：perplexity 远低于 Phase I 的有效 code 数，说明 selector 没有充分利用 archetype set。

`active_code_count`  
使用比例超过阈值的 code 数，例如 usage ratio > 1%。  
异常解释：active code 太少表示 collapse；active code 很多但收益差，可能说明选择缺乏区分度。

`max_code_usage_ratio`  
最常被选择的 code 占全部样本的比例。  
异常解释：若单个 code 占比过高，例如超过 80%，selector 可能退化为固定策略。

`min_code_usage_ratio`  
最少被选择的 code 占比。  
异常解释：接近 0 不一定是坏事，因为某些 archetype 可能确实不适合当前 split；但如果该 code 在 assigned-label baseline 中盈利，却长期不被选择，需要诊断。

`usage_kl_to_train_label_distribution`  
selected code 分布相对 Phase I train assigned label 分布的 KL divergence。  
异常解释：KL 很大说明 Phase II 的使用分布偏离 Phase I 学到的 archetype 先验。若同时收益提升，这是合理适应；若收益没有提升，则是危险漂移。

`usage_kl_to_val_label_distribution`  
selected code 分布相对当前 validation split assigned label 分布的 KL divergence。  
异常解释：若 KL 很大且 `uplift_vs_assigned <= 0`，说明 selector 偏离 Phase I label 后没有带来收益，应触发 fail 或 warn。

`per_code_mean_return`  
按 selected code 分组统计平均 return。  
用途：识别哪些 code 被 selector 选择后贡献收益，哪些 code 被选中后持续亏损。

`per_code_sample_count`  
每个 selected code 的样本数。  
用途：给 `per_code_mean_return` 提供置信度。样本太少时，per-code return 不应过度解释。

`dead_profitable_code_count`  
在 Phase I assigned-label baseline 或 oracle baseline 中表现较好、但 Phase II 几乎不选择的 code 数。  
异常解释：如果很多 profitable code 被忽略，说明 selector 可能存在选择偏置，或 Q-network 没学到这些 archetype 的适用市场状态。

### 建议 gate 和判断逻辑

- `active_code_count >= min_active_codes`
- `max_code_usage_ratio <= max_dominant_code_ratio`
- `selected_code_entropy >= min_entropy`
- 低 entropy + 高 max usage：强烈怀疑 code collapse。
- 高 entropy + 低收益：可能是无效探索或 Q value 区分度差。
- 分布偏离 Phase I label，但收益明显提升：可以接受。
- 分布偏离 Phase I label，收益没有提升：应 fail 或 warn。

## 8. Layer 5: Generalization And Stability

### 目标

判断 Phase II selector 的高分 checkpoint 是否稳定可靠，而不是训练过拟合、validation 偶然尖峰或 Q value 发散。

### 指标

`train_val_return_gap`  
训练集平均 return 与验证集平均 return 的差距。  
异常解释：train 明显高于 val，说明 selector 可能过拟合训练行情、replay buffer 或训练 split 的局部市场结构。

`val_test_return_gap`  
验证集平均 return 与测试集平均 return 的差距。  
异常解释：val 好但 test 差，说明 checkpoint selection 可能过拟合 validation split。test 不参与选 checkpoint，但应作为最终审计指标。

`train_val_usage_kl`  
train split selected code 分布与 val split selected code 分布的 KL divergence。  
异常解释：如果两个 split 市场分布相近但 code usage 差异很大，说明 selector 选择不稳定；如果市场 regime 本身不同，则需要结合收益解释。

`validation_score_churn`  
相邻 validation epoch 的综合 score 波动幅度。  
异常解释：波动过大说明训练不稳定，某个高分 checkpoint 可能只是偶然尖峰。

`selected_action_churn`  
对同一批 validation 样本，比较相邻 epoch 的 selected code 变化率。  
异常解释：高 churn 表示 selector 决策边界不稳定；若 return 没有同步提升，说明 Q-network 尚未稳定收敛。

`q_value_scale_mean`  
所有 Q value 的平均量级。  
异常解释：绝对值持续变大可能表示 Q-value overestimation 或训练发散。

`q_value_scale_std`  
Q value 的标准差。  
异常解释：过大说明估值尺度不稳定；过小说明模型可能无法区分不同 archetype。

`q_margin_mean`  
top1 Q value 与 top2 Q value 的平均差距。  
用途：衡量 selector 对所选 code 的置信度。

`low_confidence_selection_rate`  
`top1_q - top2_q` 小于阈值的样本比例。  
异常解释：比例过高说明大量选择只是微弱优势，checkpoint 排名和 selected code 可能不稳定。

`td_loss_trend`  
训练过程中 TD loss 的趋势。  
异常解释：持续上升或剧烈震荡说明 value learning 不稳定。

`imitation_loss_trend`  
训练过程中 imitation loss 的趋势。  
异常解释：如果 imitation loss 持续升高，同时收益没有提升，说明 selector 偏离 Phase I 先验但没有获得回报。

`reward_mean_trend`  
训练 rollout 或 replay batch 中 reward 均值趋势。  
异常解释：如果 train reward 上升但 validation return 不提升，可能存在训练-验证分布不一致或过拟合。

### 建议 gate 和判断逻辑

- `train_val_return_gap <= max_generalization_gap`
- `q_value_scale_mean` 和 `q_value_scale_std` 不应爆炸。
- `low_confidence_selection_rate` 过高时 warn。
- `validation_score_churn` 过高时，不建议只选最高点，应偏好稳定高分 checkpoint。
- validation 好、test 差时，不能回头用 test 选 checkpoint，但 report 必须标记泛化失败。

## 9. 各层指标阈值明细

本节给出 Phase II 第一版报表和 checkpoint selection 可采用的默认阈值。阈值分为三类：

- `hard gate`: 失败时该 checkpoint 不应进入 best checkpoint 候选。
- `warn`: 不直接淘汰，但需要在 report 中标记风险。
- `reference`: 只用于解释、排序或人工审计。

默认阈值应保存在 Phase II validation config 中，后续可以按交易标的、手续费率、horizon 长度和 validation 样本数调整。

### Layer 0: Evaluation Validity

| 指标 | 方向 | 默认阈值 | 严重级别 | 说明 |
| --- | --- | --- | --- | --- |
| `num_samples` | greater_is_better | `>= 500` | hard gate | 样本过少时收益和 per-code 统计不稳定。 |
| `valid_rollout_ratio` | greater_is_better | `== 1.0` | hard gate | 所有样本都应成功完成 decoder 和 execution。 |
| `finite_reward_ratio` | greater_is_better | `== 1.0` | hard gate | reward、fee、turnover 不允许出现 NaN/inf。 |
| `valid_selected_code_ratio` | greater_is_better | `== 1.0` | hard gate | selected code 必须全部落在 `[0, K)`。 |
| `deterministic_eval` | equal | `true` | hard gate | validation/test 必须使用 greedy deterministic action。 |
| `label_alignment_valid` | equal | `true` | hard gate | assigned label 和 horizon sample 必须对齐。 |
| `visible_state_contract_valid` | equal | `true` | hard gate | observation 不能混入未来状态、价格或 reward。 |

### Layer 1: Selector Profitability

| 指标 | 方向 | 默认阈值 | 严重级别 | 说明 |
| --- | --- | --- | --- | --- |
| `mean_return` | greater_is_better | `> 0` | hard gate | selector 平均 horizon return 必须为正。 |
| `median_return` | greater_is_better | `>= -small_return_tolerance` | warn | median 明显为负时，收益可能依赖少数尾部样本。 |
| `win_rate` | greater_is_better | `>= 0.50` | hard gate | 正收益样本比例不能低于随机方向判断。 |
| `sharpe_like` | greater_is_better | `> 0` | hard gate | 风险调整收益至少为正。 |
| `downside_sharpe_like` | greater_is_better | `> 0` | warn | 下行风险调整后仍应有正收益。 |
| `p05_return` | greater_is_better | `>= -tail_loss_limit` | hard gate | 控制左尾亏损。第一版可设为 validation return 标准差的 `-2.5x`。 |
| `fee_drag_ratio` | less_is_better | `<= 0.40` | hard gate | 手续费不应吞噬过多 gross profit。 |
| `mean_turnover` | less_is_better | `<= turnover_upper_limit` | warn | 过高换手通常带来成本和滑点风险。 |

### Layer 2: Baseline Uplift

| 指标 | 方向 | 默认阈值 | 严重级别 | 说明 |
| --- | --- | --- | --- | --- |
| `uplift_vs_random` | greater_is_better | `> 0` | hard gate | selector 必须优于随机 code selection。 |
| `beat_random_rate` | greater_is_better | `> 0.50` | hard gate | 样本级表现应多数优于 random。 |
| `uplift_vs_assigned` | greater_is_better | `>= -small_return_tolerance` | hard gate | 保守口径允许轻微低于 KL/assigned baseline，但不能明显退化。 |
| `beat_assigned_rate` | greater_is_better | `>= 0.48` | warn | 低于该值说明 selector 的 uplift 可能依赖少数大样本。 |
| `oracle_capture_ratio` | greater_is_better | `>= 0.30` | warn | 捕获 hindsight oracle 上界过低时，selector 仍有明显选择空间。 |
| `regret_to_oracle` | less_is_better | reference | reference | 用于 tie-breaker，不建议第一版 hard gate。 |

### Layer 3: Demonstration Consistency

| 指标 | 方向 | 默认阈值 | 严重级别 | 说明 |
| --- | --- | --- | --- | --- |
| `label_match_rate` | greater_is_better | `>= 0.20` | hard gate | selector 不能完全脱离 Phase I assigned label 先验。 |
| `label_match_rate` | less_is_better | `<= 0.90` | warn | 过高说明 Phase II 可能只是复现 KL/assigned baseline。 |
| `unprofitable_deviation_rate` | less_is_better | `<= 0.25` | hard gate | 偏离 assigned label 且亏于 KL 的比例不能过高。 |
| `profitable_deviation_rate` | greater_is_better | `>= 0.20` | warn | 有收益证明的偏离比例过低，说明选择价值不足。 |
| `deviation_return_delta` | greater_is_better | `>= 0` | warn | 偏离样本上的平均收益差应非负。 |
| `cross_entropy_to_assigned` | less_is_better | reference | reference | 主要解释 imitation regularization，不建议直接 hard gate。 |
| `label_q_margin` | greater_is_better | `>= 0` | warn | 偏离 assigned label 时，selected code 的 Q 值应有优势。 |

### Layer 4: Code Usage And Collapse

| 指标 | 方向 | 默认阈值 | 严重级别 | 说明 |
| --- | --- | --- | --- | --- |
| `active_code_count` | greater_is_better | `>= max(3, ceil(0.4 * K))` | hard gate | 防止 selector 只使用极少数 code。 |
| `selected_code_entropy` | greater_is_better | `>= log(active_code_count_min)` | hard gate | code 分布熵过低表示 collapse。 |
| `selected_code_perplexity` | greater_is_better | `>= active_code_count_min` | warn | 等效使用 code 数过低需要标记。 |
| `max_code_usage_ratio` | less_is_better | `<= 0.60` | hard gate | 单个 code 不能支配大多数样本。 |
| `usage_kl_to_val_label_distribution` | less_is_better | `<= 0.50` | warn | selected 分布相对 KL label 分布偏离过大时需要解释。 |
| `dead_profitable_code_count` | less_is_better | `<= 1` | warn | 盈利 code 被 selector 忽略时需要诊断。 |
| `per_code_sample_count` | greater_is_better | `>= 30` | reference | 低 support code 的 per-code return 只作参考。 |

### Layer 5: Generalization And Stability

| 指标 | 方向 | 默认阈值 | 严重级别 | 说明 |
| --- | --- | --- | --- | --- |
| `train_val_return_gap` | less_is_better | `<= max(0.5 * abs(val_mean_return), gap_abs_limit)` | warn | train 明显好于 val 时标记过拟合风险。 |
| `val_test_return_gap` | less_is_better | reference | reference | test 不参与选 checkpoint，只做最终审计。 |
| `train_val_usage_kl` | less_is_better | `<= 0.50` | warn | train/val code 使用分布差异过大需要解释。 |
| `validation_score_churn` | less_is_better | `<= 0.15` | warn | 分数波动过大时，最高点可能不稳定。 |
| `selected_action_churn` | less_is_better | `<= 0.35` | warn | 同一样本跨 epoch 选择频繁变化，说明决策边界不稳。 |
| `q_value_scale_mean` | less_is_better | `<= q_scale_abs_limit` | warn | Q 值尺度持续膨胀可能表示 overestimation。 |
| `q_value_scale_std` | less_is_better | `<= q_scale_std_limit` | warn | Q 值方差过大说明估值不稳定。 |
| `q_margin_mean` | greater_is_better | `>= q_margin_floor` | warn | top1/top2 Q margin 太低表示选择置信度不足。 |
| `low_confidence_selection_rate` | less_is_better | `<= 0.40` | warn | 大量低置信选择会降低 checkpoint 稳定性。 |

## 10. 综合评分建议

Phase II checkpoint selection 不建议只按 `mean_return` 排序。建议在通过 Layer 0 到 Layer 3 的必要 gate 后，使用综合分排序。

```text
phase2_score =
  0.35 * profitability_score
+ 0.25 * baseline_uplift_score
+ 0.15 * risk_score
+ 0.10 * consistency_score
+ 0.10 * code_usage_score
+ 0.05 * stability_score
```

### 子分数含义

`profitability_score`  
由 `mean_return`、`median_return`、`win_rate` 归一化得到。

`baseline_uplift_score`  
由 `uplift_vs_assigned`、`uplift_vs_random`、`oracle_capture_ratio` 得到。

`risk_score`  
由 `sharpe_like`、`downside_sharpe_like`、`p05_return`、`fee_drag_ratio` 得到。

`consistency_score`  
由 `label_match_rate`、`profitable_deviation_rate`、`unprofitable_deviation_rate` 得到。

`code_usage_score`  
由 `selected_code_entropy`、`active_code_count`、`max_code_usage_ratio` 得到。

`stability_score`  
由 `validation_score_churn`、`selected_action_churn`、`q_margin_mean`、`low_confidence_selection_rate` 得到。

## 11. Tie-breaker 建议

当多个 checkpoint 综合分接近时，按以下顺序做稳定排序：

1. `uplift_vs_assigned` 越高越好。
2. `sharpe_like` 越高越好。
3. `p05_return` 越高越好。
4. `oracle_capture_ratio` 越高越好。
5. `unprofitable_deviation_rate` 越低越好。
6. `selected_code_entropy` 越合理越好，避免过低 collapse。
7. `validation_score_churn` 越低越好。
8. epoch 越早越好，用于降低后期过拟合风险。

## 12. 第一版落地范围建议

第一版 Phase II validation 可以先实现最小闭环：

1. Layer 0: evaluation validity。
2. Layer 1: selector return、risk、fee、turnover。
3. Layer 2: assigned/random/oracle baseline uplift。
4. Layer 3: label match、profitable/unprofitable deviation。
5. Layer 4: entropy、active code、max usage、per-code return。

Layer 5 可先作为训练日志和 report diagnostics，不必立即作为 hard gate。等 validation checkpoint 数量和测试结果积累后，再把 stability score 纳入 checkpoint selector。

## 13. Phase II 报表卡片与过程数据

Phase II 报表需要复用 Phase I 已有的 Code 级诊断、per-code 盈利、code distribution 和 Dominant Pair 热力图卡片，但口径要从“Phase I assigned code / decoded action”扩展为“Phase II selector selected code / selected code decoded action”。同时，为了和论文中的 KL imitation 约束对齐，报表中建议把 `KL` 统一解释为 `KL/assigned-label baseline`：即使用 Phase I 离线 assigned code label 执行 frozen decoder 得到的基线结果。

如果后续实现中 `KL` 指代的是 selector policy 的 KL loss 或 KL-regularized policy 输出，需要在报表 schema 中把命名拆成 `assigned_label_baseline` 和 `policy_kl_diagnostics`，避免混淆。

### 13.1 Per-code 盈利图表

#### 展示目标

按 code 展示 selector 和 KL/assigned-label baseline 的盈利差异。该图回答：Phase II selector 选择某个 code 后是否真的赚钱，以及它相对 Phase I assigned label baseline 是否有提升。

#### 对比对象

- `selector`: 按 `selected_code_id` 分组。
- `kl`: 按 `assigned_code_label` 分组，执行 assigned label 对应 frozen decoder action。

#### 建议展示字段

每个 code 一组 bars 或 grouped rows：

- `selector_mean_return`
- `kl_mean_return`
- `selector_total_return`
- `kl_total_return`
- `selector_win_rate`
- `kl_win_rate`
- `selector_support`
- `kl_support`
- `selector_fee_drag_ratio`
- `kl_fee_drag_ratio`
- `uplift_vs_kl = selector_mean_return - kl_mean_return`

#### 需要新增的过程数据

按样本保存：

- `sample_id`
- `selected_code_id`
- `assigned_code_label`
- `selector_return`
- `selector_gross_return`
- `selector_fee`
- `selector_turnover`
- `kl_return`
- `kl_gross_return`
- `kl_fee`
- `kl_turnover`

按 code 聚合保存：

- `code_id`
- `selector_support`
- `kl_support`
- `selector_mean_return`
- `kl_mean_return`
- `selector_total_return`
- `kl_total_return`
- `selector_win_rate`
- `kl_win_rate`
- `selector_fee_drag_ratio`
- `kl_fee_drag_ratio`
- `uplift_vs_kl`

### 13.2 Codebook 使用分布

#### 展示目标

对比 selector 实际使用 codebook 的分布和 KL/assigned-label baseline 的 code 分布，判断 Phase II 是否发生 code collapse、选择漂移或有效重分配。

#### 对比对象

- `selector_codebook_distribution`: `selected_code_id` 的 count/ratio。
- `kl_codebook_distribution`: `assigned_code_label` 的 count/ratio。

#### 建议展示字段

每个 code 一行或一组 stacked bars：

- `code_id`
- `selector_count`
- `selector_ratio`
- `kl_count`
- `kl_ratio`
- `ratio_delta = selector_ratio - kl_ratio`
- `support_delta = selector_count - kl_count`

全局摘要：

- `selector_entropy`
- `kl_entropy`
- `selector_perplexity`
- `kl_perplexity`
- `selector_active_code_count`
- `kl_active_code_count`
- `selector_max_code_usage_ratio`
- `kl_max_code_usage_ratio`
- `usage_kl_selector_to_kl`

#### 需要新增的过程数据

该卡片主要依赖每个样本的 code assignment：

- `sample_id`
- `selected_code_id`
- `assigned_code_label`
- `selector_q_top1`
- `selector_q_top2`
- `selector_q_margin`
- `assigned_label_q_value`

其中 Q-value 字段不是分布图必需项，但能解释 selector 为什么偏离 assigned label。

### 13.3 Oracle-label 累计收益图表

#### 展示目标

展示不同策略在 validation/test split 上按样本顺序累计后的收益曲线。该图回答：selector 的收益曲线是否稳定优于多个基线，而不是只在均值上占优。

#### 五条线

`selector`  
Phase II selector greedy action 选择 code 后，frozen decoder 执行得到的 return。

`dp`  
Phase I DP teacher action 直接执行得到的 return。该线使用未来信息，只作为 teacher/oracle 质量参考，不参与推理和 checkpoint selection 的线上决策。

`kl`  
Phase I assigned code label baseline，即按 assigned label 执行 frozen decoder。

`random`  
随机选择 code 的 baseline。建议保存 random seed，并可以选择展示多 seed 均值线或固定 seed 线。

`hold`  
一直持有基线。需要明确持有方向，建议至少支持：

- `hold_long`: 全 horizon long。
- `hold_short`: 全 horizon short。
- 如果报表只允许一条 hold 线，应在 metadata 中写清楚使用哪一种。

#### 建议展示字段

每条线保存：

- `step_index`
- `sample_id`
- `horizon_return`
- `cumulative_return`
- `drawdown`

卡片摘要：

- `selector_final_cumulative_return`
- `dp_final_cumulative_return`
- `kl_final_cumulative_return`
- `random_final_cumulative_return`
- `hold_final_cumulative_return`
- `selector_max_drawdown`
- `selector_curve_above_kl_ratio`
- `selector_curve_above_random_ratio`

#### 需要新增的过程数据

按样本保存五类 return：

- `sample_id`
- `sequence_index`
- `selector_return`
- `dp_return`
- `kl_return`
- `random_return`
- `hold_return`
- `selector_cumulative_return`
- `dp_cumulative_return`
- `kl_cumulative_return`
- `random_cumulative_return`
- `hold_cumulative_return`

如果当前 Phase II dataset 不包含 DP teacher actions 或 DP horizon returns，则需要从 Phase I 产物中额外读取或导出：

- `dp_actions`
- `dp_rewards`
- `dp_horizon_return`

如果当前 Phase II dataset 不包含 hold baseline，则 evaluator 需要用同一 `ActionExecutionCalculator` 临时构造 hold actions 并执行，保证手续费、滑点和 return 口径一致。

### 13.4 Dominant Pair 热力图

#### 展示目标

复用 Phase I 的 Dominant Pair 热力图设计，但 Phase II 的样本归因应基于 selector 产生的结果。该卡片回答：selector 选择出的 archetype 在哪些市场形态 morphology 和交易行为 motif 组合上贡献收益或风险。

#### Phase II 口径

- morphology: 仍按 Phase I 相同规则，从 market states/prices 中识别市场形态。
- motif: 基于 selector selected code 经过 frozen decoder 产生的 base actions 识别交易行为 motif。
- return: 使用 selector 对应的 execution return。
- advantage: 建议同时计算相对 KL baseline 和 random baseline 的 advantage。

#### 建议展示字段

每个 morphology x motif cell：

- `morphology`
- `motif`
- `support`
- `selector_mean_return`
- `kl_mean_return`
- `random_mean_return`
- `mean_advantage_vs_kl`
- `mean_advantage_vs_random`
- `win_rate`
- `fee_drag_ratio`
- `selected_code_top`
- `selected_code_top_ratio`

#### Cell 展示建议

每个热力图 cell 建议同时显示绝对收益和相对收益：

```text
selector_mean_return
adv_vs_kl: mean_advantage_vs_kl
n=support, c=dominant_selected_code
```

示例：

```text
+12.4
adv +4.8
n=612 c2
```

含义：

- `+12.4`: selector 在该 morphology/motif cell 内的平均 horizon return。
- `adv +4.8`: selector 相对 KL/assigned-label baseline 的平均收益优势，即 `selector_mean_return - kl_mean_return`。
- `n=612`: 该 cell 的样本数。
- `c2`: 该 cell 内最常被 selector 选择的 code 是 code 2。

颜色建议仍然使用 `mean_advantage_vs_kl`，因为热力图主要用于定位 selector 相对 KL baseline 的改进和退化；cell 文本中的第一行补充展示 selector 的绝对盈利水平。

#### 需要新增的过程数据

按样本保存：

- `sample_id`
- `selected_code_id`
- `assigned_code_label`
- `morphology`
- `selector_motif`
- `kl_motif`
- `selector_return`
- `kl_return`
- `random_return`
- `selector_fee`
- `selector_turnover`

按 cell 聚合保存：

- `morphology`
- `motif`
- `support`
- `selector_mean_return`
- `kl_mean_return`
- `random_mean_return`
- `mean_advantage_vs_kl`
- `mean_advantage_vs_random`
- `win_rate`
- `fee_drag_ratio`
- `dominant_selected_code`
- `dominant_selected_code_ratio`

### 13.5 Code 级诊断表

#### 展示目标

把每个 code 的使用、收益、baseline 对比、dominant pair 和风险状态放到一张表里。Phase I 的 Code Diagnostics 表主要解释 codebook 学得是否好；Phase II 的 Code Diagnostics 表还需要解释 selector 是否正确使用了这些 code。

#### 建议字段

基础字段：

- `code_id`
- `selector_support`
- `selector_usage_ratio`
- `kl_support`
- `kl_usage_ratio`
- `usage_delta`

收益字段：

- `selector_mean_return`
- `kl_mean_return`
- `uplift_vs_kl`
- `selector_win_rate`
- `selector_fee_drag_ratio`
- `selector_turnover`

行为解释字段：

- `dominant_morphology`
- `dominant_morphology_ratio`
- `dominant_motif`
- `dominant_motif_ratio`
- `dominant_pair`
- `dominant_pair_ratio`

selector 诊断字段：

- `mean_q_margin`
- `low_confidence_ratio`
- `profitable_deviation_count`
- `unprofitable_deviation_count`
- `unprofitable_deviation_rate`
- `status`
- `risk_reason`

#### status 建议

- `pass`: 使用充分、收益为正、相对 KL 有 uplift 或无明显坏偏离。
- `warn_low_support`: selector 使用样本太少，无法可靠解释。
- `warn_collapse_dominant`: 该 code 使用占比过高，可能造成 collapse。
- `warn_unprofitable`: selector 选择该 code 后平均收益为负。
- `warn_bad_deviation`: selector 偏离 assigned label 选择该 code，但收益低于 KL baseline。
- `fail_invalid`: code id、return 或过程数据不合法。

#### 需要新增的过程数据

Code 级诊断表本身可以由前面四类过程数据聚合得到，但为了报表渲染和审计稳定，建议在 validation payload 中显式保存 per-code rows：

- `code_id`
- `selector_support`
- `selector_usage_ratio`
- `kl_support`
- `kl_usage_ratio`
- `selector_mean_return`
- `kl_mean_return`
- `uplift_vs_kl`
- `selector_win_rate`
- `selector_fee_drag_ratio`
- `dominant_morphology`
- `dominant_morphology_ratio`
- `dominant_motif`
- `dominant_motif_ratio`
- `dominant_pair`
- `dominant_pair_ratio`
- `mean_q_margin`
- `low_confidence_ratio`
- `unprofitable_deviation_rate`
- `status`
- `risk_reason`

## 14. Phase II 报表聚合数据建议

Phase II 报表不需要保存完整逐样本过程表，也不需要持久化一行一个 horizon 的 trace。Evaluator 在 validation/test 期间可以在内存中临时保留逐样本执行结果，用于计算图表；落盘 payload 只保存报表卡片需要的聚合结果和曲线点。

### 14.1 聚合 payload

validation result/report payload 建议保存以下聚合结构：

- `per_code_profitability_comparison`
- `codebook_usage_comparison`
- `cumulative_return_curves`
- `selector_pair_profitability_matrix`
- `code_diagnostics`

这样 HTML report 不需要重新执行 evaluator，也不需要重新 decode actions；它只消费已经落盘的聚合结果。

### 14.2 最小聚合字段

`per_code_profitability_comparison`:

- `code_id`
- `selector_support`
- `kl_support`
- `selector_mean_return`
- `kl_mean_return`
- `selector_total_return`
- `kl_total_return`
- `selector_win_rate`
- `kl_win_rate`
- `selector_fee_drag_ratio`
- `kl_fee_drag_ratio`
- `uplift_vs_kl`

`codebook_usage_comparison`:

- `code_id`
- `selector_count`
- `selector_ratio`
- `kl_count`
- `kl_ratio`
- `ratio_delta`
- `support_delta`
- 全局摘要：`selector_entropy`、`kl_entropy`、`selector_perplexity`、`kl_perplexity`、`usage_kl_selector_to_kl`

`cumulative_return_curves`:

- 每条线只保存降采样或完整曲线点：`x`、`selector`、`dp`、`kl`、`random`、`hold`
- 曲线摘要：`final_return`、`max_drawdown`、`curve_above_kl_ratio`、`curve_above_random_ratio`

`selector_pair_profitability_matrix`:

- `morphology`
- `motif`
- `support`
- `selector_mean_return`
- `kl_mean_return`
- `random_mean_return`
- `mean_advantage_vs_kl`
- `mean_advantage_vs_random`
- `win_rate`
- `fee_drag_ratio`
- `dominant_selected_code`
- `dominant_selected_code_ratio`

`code_diagnostics`:

- `code_id`
- `selector_support`
- `selector_usage_ratio`
- `kl_support`
- `kl_usage_ratio`
- `selector_mean_return`
- `kl_mean_return`
- `uplift_vs_kl`
- `dominant_morphology`
- `dominant_motif`
- `dominant_pair`
- `mean_q_margin`
- `low_confidence_ratio`
- `unprofitable_deviation_rate`
- `status`
- `risk_reason`
