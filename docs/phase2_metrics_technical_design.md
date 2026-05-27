# Phase II 分层指标骨架技术设计

## 1. 目标

本设计文档描述 Phase II validation metrics 的骨架代码结构。当前版本只定义强类型数据结构、阈值配置和 rule 入口，不实现 evaluator 推理、decoder 执行、收益聚合、probe 训练或报表渲染。

设计原则：

1. 1 个 validation layer 对应 1 个代码文件。
2. 每层文件只保存本层的 `Payload`、`Metrics`、`Thresholds` 和 `evaluate_*_rules()`。
3. 公共 metric result、layer result 和阈值比较 helper 单独放在公共文件中。
4. 可预测性相关的中间 payload 和阈值配置先放入 Layer 5，作为 stability/predictability reference，不新增第 6 层。
5. 报表不保存完整逐样本 trace，只消费 evaluator 已经聚合好的 payload。

## 2. 文件结构

新增或补齐的文件：

```text
src/phase2/metrics/__init__.py
src/phase2/metrics/phase2_layer_computation.py
src/phase2/metrics/phase2_metric_results.py
src/phase2/metrics/phase2_validation_rule_helpers.py
src/phase2/metrics/phase2_validation_layer0_evaluation_validity.py
src/phase2/metrics/phase2_validation_layer1_selector_profitability.py
src/phase2/metrics/phase2_validation_layer2_baseline_uplift.py
src/phase2/metrics/phase2_validation_layer3_demonstration_consistency.py
src/phase2/metrics/phase2_validation_layer4_code_usage_collapse.py
src/phase2/metrics/phase2_validation_layer5_generalization_stability.py
src/phase2/evaluators/phase2_validation_layers/__init__.py
src/phase2/evaluators/phase2_validation_layers/_numeric.py
src/phase2/evaluators/phase2_validation_layers/layer0_evaluation_validity.py
src/phase2/evaluators/phase2_validation_layers/layer1_selector_profitability.py
src/phase2/evaluators/phase2_validation_layers/layer2_baseline_uplift.py
src/phase2/evaluators/phase2_validation_layers/layer3_demonstration_consistency.py
src/phase2/evaluators/phase2_validation_layers/layer4_code_usage_collapse.py
src/phase2/evaluators/phase2_validation_layers/layer5_generalization_stability.py
```

已有文件保持：

```text
src/phase2/metrics/phase2_metrics.py
src/phase2/metrics/phase2_metric_results.py
```

`phase2_metrics.py` 仍用于 Double DQN 训练期基础指标聚合；本次新增的 validation layer 文件用于 validation/test checkpoint 评估结果。

`src/phase2/metrics/phase2_validation_layer*.py` 定义 schema、payload、threshold 和 rule；`src/phase2/evaluators/phase2_validation_layers/layer*.py` 负责从 payload/聚合数组计算 raw metrics。

## 3. 公共结果结构与规则 helper

文件：

```text
src/phase2/metrics/phase2_metric_results.py
```

核心类：

`Phase2MetricResult`

- 表达单个 metric 的值、阈值、方向、严重级别、是否通过和说明。
- 对齐 Phase I 的 metric result 风格。
- 供 report 的“指标阈值明细”卡片消费。

`Phase2LayerResult`

- 聚合一个 layer 内的多个 `Phase2MetricResult`。
- `passed` 只表达该 layer 的 gate 结果。

同一文件还保留 Phase II validation checkpoint/report 已有的核心 payload：

- `Phase2ValidationMetrics`
- `Phase2ValidationResult`
- `Phase2ValidationPayloads`

各层 `Computation` 类型放在对应 `phase2_validation_layer*.py` 文件中：

- `Phase2Layer0EvaluationValidityComputation`
- `Phase2Layer1SelectorProfitabilityComputation`
- `Phase2Layer2BaselineUpliftComputation`
- `Phase2Layer3DemonstrationConsistencyComputation`
- `Phase2Layer4CodeUsageCollapseComputation`
- `Phase2Layer5GeneralizationStabilityComputation`

`Phase2ValidationResult` 保存三类结果：

- `metrics`: checkpoint selector 直接消费的核心摘要指标。
- `layers`: hard-gate/reference layer 的阈值判定结果。
- `layer_computations`: Layer 0-5 的强类型 raw metrics 和本层中间 payload。
- `payloads`: report 复用的聚合 payload，不保存完整 `selection_trace`。

规则 helper 文件：

```text
src/phase2/metrics/phase2_validation_rule_helpers.py
```

公共 helper：

- `_ge()`
- `_gt()`
- `_le()`
- `_between()`
- `_eq_bool()`
- `_build_layer_result()`
- `_is_missing()`

这些 helper 只负责阈值比较、result 包装和 layer 聚合，不计算 raw metrics。Layer 5 当前可以在 `_build_layer_result(force_passed=True)` 下作为 warn/reference 层，不阻断 checkpoint selection。

## 4. Layer 文件设计

### 4.1 Layer 0: Evaluation Validity

文件：

```text
src/phase2/metrics/phase2_validation_layer0_evaluation_validity.py
```

类：

`Phase2EvaluationValidityPayload`

- 保存评估有效性 raw metrics 计算所需的中间计数。
- 例如失败 rollout 数、非有限 reward 数、非法 code 数。

`Phase2EvaluationValidityMetrics`

- `num_samples`
- `valid_rollout_ratio`
- `finite_reward_ratio`
- `valid_selected_code_ratio`
- `deterministic_eval`
- `label_alignment_valid`
- `visible_state_contract_valid`

`Phase2EvaluationValidityThresholds`

- `min_eval_samples`
- `valid_rollout_ratio_min`
- `finite_reward_ratio_min`
- `valid_selected_code_ratio_min`
- deterministic / alignment / no-leakage bool 要求

入口：

```python
evaluate_evaluation_validity_rules(metrics, thresholds) -> Phase2LayerResult
```

### 4.2 Layer 1: Selector Profitability

文件：

```text
src/phase2/metrics/phase2_validation_layer1_selector_profitability.py
```

类：

`Phase2SelectorProfitabilityPayload`

- 保存 selector return、gross return、fee、turnover 序列。
- 该 payload 是计算聚合 raw metrics 的中间对象；报表不需要保存完整序列。

`Phase2SelectorProfitabilityMetrics`

- `mean_return`
- `median_return`
- `total_return`
- `win_rate`
- `sharpe_like`
- `downside_sharpe_like`
- `p05_return`
- `loss_rate`
- `mean_gross_return`
- `mean_fee`
- `fee_drag_ratio`
- `mean_turnover`

`Phase2SelectorProfitabilityThresholds`

- 收益、胜率、风险、手续费和换手阈值。

入口：

```python
evaluate_selector_profitability_rules(metrics, thresholds) -> Phase2LayerResult
```

### 4.3 Layer 2: Baseline Uplift

文件：

```text
src/phase2/metrics/phase2_validation_layer2_baseline_uplift.py
```

类：

`Phase2BaselineUpliftPayload`

- 保存 selector、assigned-label/KL、random、oracle return 序列。
- 仅作为 evaluator 内部聚合输入。

`Phase2BaselineUpliftMetrics`

- `assigned_mean_return`
- `random_mean_return`
- `oracle_mean_return`
- `uplift_vs_assigned`
- `uplift_vs_random`
- `relative_uplift_vs_assigned`
- `oracle_capture_ratio`
- `regret_to_oracle`
- `beat_assigned_rate`
- `beat_random_rate`

`Phase2BaselineUpliftThresholds`

- random uplift hard gate。
- assigned-label/KL uplift hard gate。
- oracle capture 和 regret warning。

入口：

```python
evaluate_baseline_uplift_rules(metrics, thresholds) -> Phase2LayerResult
```

### 4.4 Layer 3: Demonstration Consistency

文件：

```text
src/phase2/metrics/phase2_validation_layer3_demonstration_consistency.py
```

类：

`Phase2DemonstrationConsistencyPayload`

- 保存 selected code、assigned label、selector return、assigned-label return、selected Q value 和 assigned-label Q value。

`Phase2DemonstrationConsistencyMetrics`

- `label_match_rate`
- `cross_entropy_to_assigned`
- `kl_to_assigned_onehot`
- `label_q_margin`
- `profitable_deviation_rate`
- `unprofitable_deviation_rate`
- `deviation_return_delta`

`Phase2DemonstrationConsistencyThresholds`

- label match 下限/上限。
- bad deviation hard gate。
- profitable deviation、deviation delta、Q margin warning。

入口：

```python
evaluate_demonstration_consistency_rules(metrics, thresholds) -> Phase2LayerResult
```

### 4.5 Layer 4: Code Usage And Collapse

文件：

```text
src/phase2/metrics/phase2_validation_layer4_code_usage_collapse.py
```

类：

`Phase2PerCodeUsageDiagnostic`

- 每个 code 的 usage、KL usage、收益 uplift 和 dead-profitable 标记。
- 可直接作为 Code 级诊断表的数据来源之一。

`Phase2CodeUsageCollapsePayload`

- 保存 selected code、assigned label 和 per-code diagnostic rows。

`Phase2CodeUsageCollapseMetrics`

- `selected_code_entropy`
- `selected_code_perplexity`
- `active_code_count`
- `max_code_usage_ratio`
- `min_code_usage_ratio`
- `usage_kl_to_train_label_distribution`
- `usage_kl_to_val_label_distribution`
- `dead_profitable_code_count`
- `min_per_code_sample_count`

`Phase2CodeUsageCollapseThresholds`

- active code 下限。
- entropy/perplexity 下限。
- dominant code ratio 上限。
- usage KL、dead profitable code warning。

入口：

```python
evaluate_code_usage_collapse_rules(metrics, thresholds, *, num_archetypes) -> Phase2LayerResult
```

### 4.6 Layer 5: Generalization, Stability, Predictability

文件：

```text
src/phase2/metrics/phase2_validation_layer5_generalization_stability.py
```

类：

`Phase2PredictabilityPayload`

- 可预测性 raw metrics 计算的中间 payload。
- 保存 probe train/validation accuracy、gap、confusion matrix、probe seed。
- 不保存逐样本 probe 预测。

`Phase2PredictabilityMetrics`

- `probe_top1_accuracy`
- `probe_top3_accuracy`
- `probe_balanced_accuracy`
- `selected_code_entropy_given_morphology`
- `selected_code_entropy`
- `mutual_information_lift`

`Phase2PredictabilityThresholds`

- top-1/top-3 自适应阈值。
- balanced accuracy 下限。
- mutual information lift 下限。
- conditional entropy ratio 上限。

`Phase2GeneralizationStabilityPayload`

- 保存 score history、selected action churn history、Q scale history。
- 可选挂载 `Phase2PredictabilityPayload`。

`Phase2GeneralizationStabilityMetrics`

- `train_val_return_gap`
- `val_test_return_gap`
- `train_val_usage_kl`
- `validation_score_churn`
- `selected_action_churn`
- `q_value_scale_mean`
- `q_value_scale_std`
- `q_margin_mean`
- `low_confidence_selection_rate`
- `td_loss_trend`
- `imitation_loss_trend`
- `reward_mean_trend`
- `predictability`

`Phase2GeneralizationStabilityThresholds`

- gap、usage KL、churn、Q scale、Q margin、low confidence warning 阈值。
- 内嵌 `Phase2PredictabilityThresholds`。

入口：

```python
evaluate_generalization_stability_rules(
    metrics,
    thresholds,
    *,
    num_archetypes,
) -> Phase2LayerResult
```

Layer 5 当前作为 warning/reference 层，默认不阻断 checkpoint selection。

## 5. 后续接入计划

后续 evaluator 接入时建议分三步：

1. 在 `Phase2Evaluator` 内部计算 selector、KL/assigned、random、oracle、hold 和 DP baseline 的聚合结果。
2. 将聚合结果转换为各 layer 的 `Metrics` 对象，并调用 `evaluate_*_rules()` 得到 `Phase2LayerResult`。
3. 在 `Phase2ValidationResult` 中写入：
   - `metrics`: checkpoint selector 需要的稳定排序字段。
   - `layers`: hard-gate/reference layer 判定结果。
   - `layer_computations`: Layer 0-5 各自的强类型 computation 对象。
   - `payloads`: report 聚合 payload。

## 6. 报表 payload 边界

报表不保存完整逐样本 trace。推荐保存以下聚合结构：

- `per_code_profitability_comparison`
- `codebook_usage_comparison`
- `cumulative_return_curves`
- `selector_pair_profitability_matrix`
- `code_diagnostics`
- `layer_results`

HTML report 只消费这些聚合 payload，不重新执行 decoder、reward 或 probe。
