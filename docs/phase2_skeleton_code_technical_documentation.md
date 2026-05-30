# Phase II 骨架代码技术文档

本文档基于 `src/phase2` 当前骨架代码整理，描述 Phase II 文件、文件中的类名，以及类中与维度、shape、容量、样本数、code 数等相关的字段。不展开普通业务字段细节。

## 1. 维度约定

| 符号 | 含义 |
| --- | --- |
| `N` | split 内可训练/评估样本数，通常是 horizon 样本数减 1 |
| `B` | batch size |
| `H` | horizon 长度 |
| `T` | selector 当前分片可见窗口长度，来自 `tsize` |
| `F` | 原始 state feature 维度，来自 `state_dim` |
| `R` | relative state feature 维度，来自 `relative_state_dim` |
| `G` | trend state feature 维度，来自 `trend_state_dim` |
| `P` | price feature 维度 |
| `D` | depthprice feature 维度；没有盘口时可为 0 或 `None` |
| `K` | Phase I codebook / Phase II action 空间大小，来自 `num_archetypes` |
| `L` | Phase I latent/code embedding 维度 |
| `A` | Phase I decoder 基础动作类别数 |

核心数据 shape：

- `HorizonDataset`: `(states, relative_states, trend_states, prices, depthprices)`
  - `states`: `[N_raw, H, F]`
  - `relative_states`: `[N_raw, H, R]`
  - `trend_states`: `[N_raw, H, G]`
  - `prices`: `[N_raw, H, P]`
  - `depthprices`: `[N_raw, H, D]` 或 `None`
- `VisibleStatesDataset`: 六路 selector observation
  - previous 三路: `[N, H, F/R/G]`
  - current 三路: `[N, T, F/R/G]`
- Q-network 输入: 六路 `[B, time, feature]`
- Q-network 输出: `[B, K]`
- decoder 输入: 三路 `[B, H, F/R/G]` 与 selected code `[B]`
- decoder 输出: `[B, H]`；全 code 解码为 `[B, K, H]`
- evaluator rollout 矩阵: return/fee/turnover `[N, K]`，actions `[N, K, H]`

## 2. 顶层编排与配置

| 文件 | 类名 | 维度字段 |
| --- | --- | --- |
| `src/phase2/phase2_config.py` | `Phase2DatasetConfig` | `tsize` |
| `src/phase2/phase2_config.py` | `Phase2ModelConfig` | `state_dim`, `relative_state_dim`, `trend_state_dim`, `num_archetypes`, `hidden_dim`, `num_layers` |
| `src/phase2/phase2_config.py` | `Phase2RewardConfig` | 无显式 shape 字段 |
| `src/phase2/phase2_config.py` | `Phase2TrainConfig` | `epochs`, `batch_size`, `replay_capacity`, `learning_start_epoch`, `updates_per_epoch`, `rollout_batch_size`, `validation_interval`, `target_update_interval_epochs`, `epsilon_decay_epochs` |
| `src/phase2/phase2_config.py` | `Phase2MainConfig` | 无显式 shape 字段 |
| `src/phase2/phase2_main.py` | `Phase2FatalError` | 无 |
| `src/phase2/phase2_main.py` | `Phase2MainFlow` | `state_dim`, `relative_state_dim`, `trend_state_dim`, `validation_checkpoints` |
| `src/phase2/phase2_artifact_store.py` | `Phase2ArtifactStore` | `train_batch_id` 参与产物命名，不承载 tensor shape |

## 3. 数据集与环境

| 文件 | 类名 | 维度字段 |
| --- | --- | --- |
| `src/phase2/phase2_selection_dataset.py` | `Phase2SelectionDataset` | `visible_states`, `horizon_dataset`, `demonstration_horizon_label_dataset` |
| `src/phase2/phase2_selection_dataset.py` | `Phase2SelectionDatasetBuilder` | `tsize`, `REQUIRED_LABEL_COLUMNS` |
| `src/phase2/phase2_env.py` | `Phase2SelectionStepResultInfo` | `selected_code_id`, `demo_assigned_code_label` |
| `src/phase2/phase2_env.py` | `Phase2SelectionStepResult` | `observation` |
| `src/phase2/phase2_env.py` | `ArchetypeSelectionEnv` | `sample_count`, `decoder_policy.num_archetypes` |
| `src/phase2/phase2_batch_env.py` | `Phase2SelectionBatchResult` | `observations`, `next_observations`, `rewards`, `dones`, `sample_ids`, `selected_code_ids`, `assigned_code_labels`, `actions` |
| `src/phase2/phase2_batch_env.py` | `ArchetypeSelectionBatchEnv` | `indices`, `selected_code_ids`, `expected_size`, `sample_count` |

维度说明：

- `Phase2SelectionDatasetBuilder.build_visible_states()` 将原始 `N_raw` 个 horizon 样本变成 `N = N_raw - 1` 个 selector 样本。
- `Phase2SelectionBatchResult.actions` 为 `[B, H]`。
- `Phase2SelectionBatchResult.rewards/dones/sample_ids/selected_code_ids/assigned_code_labels` 为 `[B]`。

## 4. 模型

| 文件 | 类名 | 维度字段 |
| --- | --- | --- |
| `src/phase2/model/phase2_q_network.py` | `Phase2QNetwork` | `VISIBLE_STATE_COUNT`, `STREAM_POOL_COUNT`, `visible_state_feature_dims`, `stream_encoders`, `config.num_archetypes`, `config.hidden_dim`, `config.num_layers` |
| `src/phase2/model/phase2_decoder_policy.py` | `FrozenArchetypeDecoderPolicy` | `phase1_model.num_archetypes`, `selected_code_ids`, `z_q` |

维度说明：

- `Phase2QNetwork.VISIBLE_STATE_COUNT = 6`，对应 previous/current 各三路状态。
- `Phase2QNetwork.STREAM_POOL_COUNT = 3`，对应 mean/max/last pooling。
- `Phase2QNetwork.visible_state_feature_dims = (F, R, G, F, R, G)`。
- Q head 输入维度为 `hidden_dim * STREAM_POOL_COUNT * VISIBLE_STATE_COUNT`，输出维度为 `K`。
- `FrozenArchetypeDecoderPolicy.get_code_embeddings()` 输出 `[B, L]`。

`src/phase2/model/__init__.py` 当前只作为包入口，无类定义。

## 5. RL 训练

| 文件 | 类名 | 维度字段 |
| --- | --- | --- |
| `src/phase2/rl/phase2_replay_buffer.py` | `Phase2ReplayTransition` | `visible_states`, `next_visible_states`, `action`, `demonstration_horizon_label` |
| `src/phase2/rl/phase2_replay_buffer.py` | `Phase2SelectionTransitionTensorBatch` | `visible_states`, `actions`, `rewards`, `next_visible_states`, `dones`, `demonstration_horizon_label_batch` |
| `src/phase2/rl/phase2_replay_buffer.py` | `Phase2ReplayBuffer` | `capacity`, `visible_state_shapes`, `_size`, `_actions`, `_sample_ids`, `_code_labels` |
| `src/phase2/rl/phase2_double_dqn_loss.py` | `Phase2DoubleDqnLossOutput` | `total_loss`, `td_loss`, `imitation_loss` 为 scalar tensor；`greedy_next_action_mean` 为 action 统计 |
| `src/phase2/rl/phase2_double_dqn_trainer.py` | `Phase2DoubleDqnTrainer` | `online_q_network.config.num_archetypes`, `train_config.batch_size`, `updates_per_epoch`, `target_update_interval_epochs` |
| `src/phase2/rl/phase2_double_dqn_batch_trainer.py` | `Phase2DoubleDqnBatchTrainer` | `rollout_batch_size`, `batch_size`, `num_archetypes` |

维度说明：

- replay buffer 的 `visible_state_shapes` 不包含 batch 维，固定为六路 `[(H,F), (H,R), (H,G), (T,F), (T,R), (T,G)]`。
- `Phase2SelectionTransitionTensorBatch.actions/rewards/dones` 为 `[B]`。
- `demonstration_horizon_label_batch = (sample_ids, code_labels)`，两路均为 `[B]`。
- Double DQN loss 中 `q_values` 为 `[B, K]`，`selected_q_values/td_targets` 为 `[B]`。

`src/phase2/rl/__init__.py` 当前只作为包入口，无类定义。

## 6. Checkpoint

| 文件 | 类名 | 维度字段 |
| --- | --- | --- |
| `src/phase2/checkpoint/phase2_checkpoint.py` | `Phase2Checkpoint` | `epoch`, `config`, `q_network_state_dict`, `optimizer_state_dict` |
| `src/phase2/checkpoint/phase2_checkpoint.py` | `Phase2ValidationCheckpoint` | `epoch`, `validation_result` |
| `src/phase2/checkpoint/phase2_checkpoint_selector.py` | `Phase2CheckpointSelectionResult` | `selected_epoch`, `selected_score` |
| `src/phase2/checkpoint/phase2_checkpoint_selector.py` | `Phase2CheckpointSelector` | 无显式 shape 字段 |

## 7. 评估器与 Validation Layer 计算入口

| 文件 | 类名 | 维度字段 |
| --- | --- | --- |
| `src/phase2/evaluators/phase2_evaluator.py` | `_RolloutMatrices` | `returns`, `gross_returns`, `fees`, `turnover`, `actions`, `failed_mask` |
| `src/phase2/evaluators/phase2_evaluator.py` | `Phase2Evaluator` | `rollout_batch_size`, `validation_score_history`, `selected_action_churn_history`, `td_loss_history`, `imitation_loss_history`, `reward_mean_history`, `train_usage_distribution` |

维度说明：

- `_RolloutMatrices.returns/gross_returns/fees/turnover/failed_mask` 为 `[N, K]`。
- `_RolloutMatrices.actions` 为 `[N, K, H]`。
- `Phase2Evaluator._compute_q_values()` 输出 `[N, K]`。

以下文件只定义函数入口，无类定义：

- `src/phase2/evaluators/phase2_validation_layers/__init__.py`
- `src/phase2/evaluators/phase2_validation_layers/layer0_evaluation_validity.py`
- `src/phase2/evaluators/phase2_validation_layers/layer1_selector_profitability.py`
- `src/phase2/evaluators/phase2_validation_layers/layer2_baseline_uplift.py`
- `src/phase2/evaluators/phase2_validation_layers/layer3_demonstration_consistency.py`
- `src/phase2/evaluators/phase2_validation_layers/layer4_code_usage_collapse.py`
- `src/phase2/evaluators/phase2_validation_layers/layer5_generalization_stability.py`
- `src/phase2/evaluators/phase2_validation_layers/report_aggregates.py`

## 8. Metrics Schema

| 文件 | 类名 | 维度字段 |
| --- | --- | --- |
| `src/phase2/metrics/phase2_layer_computation.py` | `Phase2LayerComputationBase` | `layer_id`, `metrics` |
| `src/phase2/metrics/phase2_metrics.py` | `Phase2Metrics` | `epoch`, `num_samples`, `num_updates`, `greedy_next_action_mean` |
| `src/phase2/metrics/phase2_metric_results.py` | `Phase2MetricResult` | 无显式 shape 字段 |
| `src/phase2/metrics/phase2_metric_results.py` | `Phase2LayerResult` | `layer_id`, `metrics` |
| `src/phase2/metrics/phase2_metric_results.py` | `Phase2ValidationMetrics` | 无显式 shape 字段 |
| `src/phase2/metrics/phase2_metric_results.py` | `Phase2ReportPairProfitabilityPayloadRow` | `support`, `dominant_selected_code`, `dominant_selected_code_ratio` |
| `src/phase2/metrics/phase2_metric_results.py` | `Phase2ReportCodeDiagnosticPayloadRow` | `code_id`, `selector_support`, `kl_support`, `profitable_deviation_count`, `unprofitable_deviation_count` |
| `src/phase2/metrics/phase2_metric_results.py` | `Phase2ReportCodeCount` | `code_id`, `count` |
| `src/phase2/metrics/phase2_metric_results.py` | `Phase2ReportCodeUsageDistribution` | `selector`, `kl` |
| `src/phase2/metrics/phase2_metric_results.py` | `Phase2ReportCumulativeReturns` | `selector`, `kl`, `random`, `oracle`, `hold` |
| `src/phase2/metrics/phase2_metric_results.py` | `Phase2ValidationPayloads` | 各 layer payload tuple、`selector_pair_profitability_matrix`, `code_diagnostics`, `codebook_usage_distribution`, `oracle_label_cumulative_returns` |
| `src/phase2/metrics/phase2_metric_results.py` | `Phase2ValidationResult` | `layers`, `layer_computations`, `payloads` |

`src/phase2/metrics/__init__.py` 和 `src/phase2/metrics/phase2_validation_rule_helpers.py` 当前不定义类。

## 9. Layer 0-5 Schema

| 文件 | 类名 | 维度字段 |
| --- | --- | --- |
| `src/phase2/metrics/phase2_validation_layer0_evaluation_validity.py` | `Phase2EvaluationValidityPayload` | `epoch`, `num_samples`, `failed_rollout_count`, `non_finite_reward_count`, `invalid_selected_code_count`, `num_archetypes` |
| `src/phase2/metrics/phase2_validation_layer0_evaluation_validity.py` | `Phase2EvaluationValidityMetrics` | `num_samples`, `valid_selected_code_ratio` |
| `src/phase2/metrics/phase2_validation_layer0_evaluation_validity.py` | `Phase2Layer0EvaluationValidityComputation` | `layer_id`, `metrics`, `evaluation_validity_payload` |
| `src/phase2/metrics/phase2_validation_layer0_evaluation_validity.py` | `Phase2EvaluationValidityThresholds` | `min_eval_samples`, `valid_selected_code_ratio_min` |
| `src/phase2/metrics/phase2_validation_layer1_selector_profitability.py` | `Phase2SelectorProfitabilityPayload` | `selector_returns`, `selector_gross_returns`, `selector_fees`, `selector_turnover` |
| `src/phase2/metrics/phase2_validation_layer1_selector_profitability.py` | `Phase2SelectorProfitabilityMetrics` | 无显式 shape 字段 |
| `src/phase2/metrics/phase2_validation_layer1_selector_profitability.py` | `Phase2Layer1SelectorProfitabilityComputation` | `layer_id`, `metrics`, `selector_profitability_payload` |
| `src/phase2/metrics/phase2_validation_layer1_selector_profitability.py` | `Phase2SelectorProfitabilityThresholds` | 无显式 shape 字段 |
| `src/phase2/metrics/phase2_validation_layer2_baseline_uplift.py` | `Phase2BaselineUpliftPayload` | `selector_returns`, `assigned_label_returns`, `random_returns`, `oracle_returns`, `random_seed` |
| `src/phase2/metrics/phase2_validation_layer2_baseline_uplift.py` | `Phase2BaselineUpliftMetrics` | 无显式 shape 字段 |
| `src/phase2/metrics/phase2_validation_layer2_baseline_uplift.py` | `Phase2Layer2BaselineUpliftComputation` | `layer_id`, `metrics`, `baseline_uplift_payload` |
| `src/phase2/metrics/phase2_validation_layer2_baseline_uplift.py` | `Phase2BaselineUpliftThresholds` | 无显式 shape 字段 |
| `src/phase2/metrics/phase2_validation_layer3_demonstration_consistency.py` | `Phase2DemonstrationConsistencyPayload` | `selected_code_ids`, `assigned_code_labels`, `selector_returns`, `assigned_label_returns`, `selected_q_values`, `assigned_label_q_values` |
| `src/phase2/metrics/phase2_validation_layer3_demonstration_consistency.py` | `Phase2DemonstrationConsistencyMetrics` | 无显式 shape 字段 |
| `src/phase2/metrics/phase2_validation_layer3_demonstration_consistency.py` | `Phase2Layer3DemonstrationConsistencyComputation` | `layer_id`, `metrics`, `demonstration_consistency_payload` |
| `src/phase2/metrics/phase2_validation_layer3_demonstration_consistency.py` | `Phase2DemonstrationConsistencyThresholds` | 无显式 shape 字段 |
| `src/phase2/metrics/phase2_validation_layer4_code_usage_collapse.py` | `Phase2PerCodeUsageDiagnostic` | `code_id`, `selector_count`, `kl_count` |
| `src/phase2/metrics/phase2_validation_layer4_code_usage_collapse.py` | `Phase2CodeUsageCollapsePayload` | `selected_code_ids`, `assigned_code_labels`, `per_code_diagnostics` |
| `src/phase2/metrics/phase2_validation_layer4_code_usage_collapse.py` | `Phase2CodeUsageCollapseMetrics` | `active_code_count`, `dead_profitable_code_count`, `min_per_code_sample_count` |
| `src/phase2/metrics/phase2_validation_layer4_code_usage_collapse.py` | `Phase2Layer4CodeUsageCollapseComputation` | `layer_id`, `metrics`, `code_usage_collapse_payload`, `per_code_diagnostics` |
| `src/phase2/metrics/phase2_validation_layer4_code_usage_collapse.py` | `Phase2CodeUsageCollapseThresholds` | `active_code_count_min`, `active_code_ratio_min`, `dead_profitable_code_count_warn_max`, `per_code_sample_count_reference_min` |
| `src/phase2/metrics/phase2_validation_layer5_generalization_stability.py` | `Phase2PredictabilityPayload` | `probe_confusion_matrix`, `probe_seed` |
| `src/phase2/metrics/phase2_validation_layer5_generalization_stability.py` | `Phase2PredictabilityMetrics` | 无显式 shape 字段 |
| `src/phase2/metrics/phase2_validation_layer5_generalization_stability.py` | `Phase2PredictabilityThresholds` | `top1_threshold(num_archetypes)`, `top3_threshold(num_archetypes)` |
| `src/phase2/metrics/phase2_validation_layer5_generalization_stability.py` | `Phase2GeneralizationStabilityPayload` | `validation_score_history`, `selected_action_churn_history`, `q_value_scale_history`, `predictability_payload` |
| `src/phase2/metrics/phase2_validation_layer5_generalization_stability.py` | `Phase2GeneralizationStabilityMetrics` | `selected_action_churn` |
| `src/phase2/metrics/phase2_validation_layer5_generalization_stability.py` | `Phase2Layer5GeneralizationStabilityComputation` | `layer_id`, `metrics`, `generalization_stability_payload` |
| `src/phase2/metrics/phase2_validation_layer5_generalization_stability.py` | `Phase2GeneralizationStabilityThresholds` | `selected_action_churn_warn_max`, `predictability_thresholds` |

## 10. Report Schema

| 文件 | 类名 | 维度字段 |
| --- | --- | --- |
| `src/phase2/report/phase2_selector_report.py` | `Phase2SelectorReport` | 无显式 shape 字段 |
| `src/phase2/report/phase2_selector_report_context.py` | `Phase2SelectorReportContextBuilder` | 无显式 shape 字段 |
| `src/phase2/report/phase2_selector_report_schema.py` | `Phase2ReportMeta` | 无显式 shape 字段 |
| `src/phase2/report/phase2_selector_report_schema.py` | `Phase2ReportDocument` | 无显式 shape 字段 |
| `src/phase2/report/phase2_selector_report_schema.py` | `Phase2ReportHeaderItem` | 无显式 shape 字段 |
| `src/phase2/report/phase2_selector_report_schema.py` | `Phase2ReportHeader` | `k`, `n_val`, `horizon`, `meta_items` |
| `src/phase2/report/phase2_selector_report_schema.py` | `Phase2ReportSummaryView` | `epoch`, `failed_layers`, `layer_count` |
| `src/phase2/report/phase2_selector_report_schema.py` | `Phase2ReportMetricView` | 无显式 shape 字段 |
| `src/phase2/report/phase2_selector_report_schema.py` | `Phase2ReportLayerView` | `layer_id`, `metric_count`, `failed_count`, `metrics` |
| `src/phase2/report/phase2_selector_report_schema.py` | `Phase2ReportMappingRow` | 无显式 shape 字段 |
| `src/phase2/report/phase2_selector_report_schema.py` | `Phase2ReportKpiRow` | 无显式 shape 字段 |
| `src/phase2/report/phase2_selector_report_schema.py` | `Phase2ReportBaselineRow` | 无显式 shape 字段 |
| `src/phase2/report/phase2_selector_report_schema.py` | `Phase2ReportPerCodeProfitabilityRow` | `code_id`, `selector_support`, `kl_support` |
| `src/phase2/report/phase2_selector_report_schema.py` | `Phase2ReportCodeUsageRow` | `code_id`, `selector_count`, `kl_count` |
| `src/phase2/report/phase2_selector_report_schema.py` | `Phase2ReportPairProfitabilityCell` | `support`, `dominant_selected_code`, `dominant_selected_code_ratio` |
| `src/phase2/report/phase2_selector_report_schema.py` | `Phase2ReportPairProfitabilityRow` | `cells` |
| `src/phase2/report/phase2_selector_report_schema.py` | `Phase2ReportPairProfitabilityMatrix` | `motifs`, `motif_headers`, `rows`, `cells` |
| `src/phase2/report/phase2_selector_report_schema.py` | `Phase2ReportCodeDiagnosticRow` | `code_id`, `selector_support`, `kl_support`, `profitable_deviation_count`, `unprofitable_deviation_count` |
| `src/phase2/report/phase2_selector_report_schema.py` | `Phase2ReportSeriesPoint` | `step` |
| `src/phase2/report/phase2_selector_report_schema.py` | `Phase2ReportSeries` | `points` |
| `src/phase2/report/phase2_selector_report_schema.py` | `Phase2ReportChartGridLine` | 无显式 shape 字段 |
| `src/phase2/report/phase2_selector_report_schema.py` | `Phase2ReportChartSeries` | `points` |
| `src/phase2/report/phase2_selector_report_schema.py` | `Phase2ReportLineChart` | `width`, `height`, `grid_lines`, `series` |
| `src/phase2/report/phase2_selector_report_schema.py` | `Phase2ReportHtmlContext` | `layers`, `core_metric_rows`, `baseline_rows`, `per_code_profitability_rows`, `code_usage_rows`, `cumulative_return_series`, `config_rows`, `artifact_rows`, `code_diagnostic_rows` |

`src/phase2/report/__init__.py` 当前只作为包入口，无类定义。

