# Phase I 骨架代码技术文档

## 1. 范围

本文档基于当前 Phase I 骨架代码，描述 Phase I 的文件、文件内类名，以及类中与维度、形状或规模相关的字段。本文不展开普通业务字段和指标字段的细节。

Phase I 代码范围分为三层：

1. Phase I 主流程、checkpoint、评估、指标和报告：`src/phase1/**`
2. Phase I 直接使用的模型骨架：`src/model/**`
3. Phase I 数据准备与 horizon 构建骨架：`src/data/**`

## 2. 维度约定

| 符号 | 含义 |
| --- | --- |
| `x` | 固定 horizon 样本数量 |
| `batch` | 训练或评估 batch size |
| `h` / `horizon` | 每条 trajectory 的时间步长度，当前默认 72 |
| `state_dim` | 主状态特征维度 |
| `relative_state_dim` | 相对状态特征维度 |
| `trend_state_dim` | 趋势状态特征维度 |
| `hidden_dim` | 模型内部隐藏层维度，当前默认 128 |
| `latent_dim` | VQ 连续 latent / codebook embedding 维度，当前默认 16 |
| `K` / `num_archetypes` | codebook 中 archetype 数量，当前默认 10 |
| `action_dim` | 动作类别数，当前默认 3，对应 short / flat / long |

核心张量形状：

| 数据 | 形状 |
| --- | --- |
| `states` | `[x or batch, horizon, state_dim]` |
| `relative_states` | `[x or batch, horizon, relative_state_dim]` |
| `trend_states` | `[x or batch, horizon, trend_state_dim]` |
| `prices` | `[x or batch, horizon, 1]` |
| `depthprices` | `[x or batch, horizon, 20]` |
| `actions` | `[batch, horizon]` |
| `rewards` | `[batch, horizon]` 或 `[batch, horizon, 1]` |
| `sample_ids` | `[batch]` |
| `z_e` / `z_q` | `[batch, latent_dim]` |
| `action_logits` | `[batch, horizon, action_dim]` |
| `distances` | `[batch, num_archetypes]` |
| `code_ids` / `code_indices` | `[batch]` |

## 3. Phase I 入口与主流程

### `scripts/train_phase1.py`

训练入口脚本。

| 类 / 函数 | 说明 | 维度字段 |
| --- | --- | --- |
| `build_default_phase1_config()` | 构建默认 `Phase1MainConfig` | 间接设置 `horizon`、`batch_size`、模型维度 |
| `main()` | 创建并运行 `Phase1MainFlow` | 无新增维度字段 |

### `src/phase1/phase1_main.py`

Phase I 主流程编排文件。

| 类名 | 说明 | 维度字段 |
| --- | --- | --- |
| `Phase1FatalError` | Phase I 统一致命错误类型 | 无 |
| `Phase1MainConfig` | Phase I 主流程配置 | `batch_size`、`horizon`、`hidden_dim`、`latent_dim`、`num_archetypes`、`action_dim`、`num_layers`、`codebook_init_max_samples`、`dead_code_reset_max_samples`、`dead_code_reset_max_fraction` |
| `Phase1MainFlow` | 串联数据加载、模型构建、预训练、VQ 训练、checkpoint 选择和 label 导出 | 从训练集首条 trajectory 推导 `state_dim`、`relative_state_dim`、`trend_state_dim`；维护 dataloader 的 `batch_size`；训练中消费 `[batch, horizon, *]` 张量 |

## 4. Phase I 产物与 Checkpoint

### `src/phase1/phase1_artifact_store.py`

| 类名 | 说明 | 维度字段 |
| --- | --- | --- |
| `Phase1ArtifactStore` | 管理 Phase I 标准产物路径、checkpoint、metrics、HTML report 和 horizon labels | 无训练张量维度字段 |

### `src/phase1/checkpoint/phase1_checkpoint.py`

| 类名 | 说明 | 维度字段 |
| --- | --- | --- |
| `Phase1Checkpoint` | 保存训练 checkpoint 的阶段、epoch、配置和状态字典 | 维度来自 `config` 中的模型参数 |
| `Phase1ValidationCheckpoint` | 保存一个验证 checkpoint 的 train / val 指标和 codebook validation 结果 | 维度来自嵌套的 validation metrics / result |

### `src/phase1/checkpoint/phase1_checkpoint_selector.py`

| 类名 | 说明 | 维度字段 |
| --- | --- | --- |
| `Phase1RejectedCheckpointSummary` | 记录被拒 checkpoint 的摘要 | 无训练张量维度字段 |
| `Phase1CheckpointSelectionResult` | 记录 checkpoint selection 的结果集合 | `candidate_count`、`eligible_count` 表示候选规模 |
| `Phase1CheckpointSelector` | 根据 validation 结果选择 best checkpoint | 无训练张量维度字段 |

## 5. Horizon Label 导出

### `src/phase1/horizon_train_label_builder.py`

| 类名 | 说明 | 维度字段 |
| --- | --- | --- |
| `HorizonTrainLabelBuilderConfig` | 离线 horizon label 生成配置 | `horizon`、`batch_size` |
| `HorizonTrainLabelRow` | 单条 horizon-level label 输出行 | `latent_values` 长度为 `latent_dim`；`horizon_start_idx` / `horizon_end_idx` 由 `sample_id * horizon` 推导 |
| `HorizonTrainLabelBuilder` | 批量调用 Phase I encoder/codebook 生成 `code_label` | 输入 `TrajectoryDataset`，推理 batch 为 `[batch, horizon, *]`，输出每条样本一个 code label |

## 6. Phase I 训练评估器

### `src/phase1/evaluators/phase1_evaluator.py`

| 类名 | 说明 | 维度字段 |
| --- | --- | --- |
| `Phase1Evaluator` | 基础训练/验证指标评估器 | 消费 `TrajectoryTensorBatch`：`states [batch,h,state_dim]`、`actions [batch,h]` |

### `src/phase1/evaluators/phase1_codebook_evaluator.py`

| 类名 | 说明 | 维度字段 |
| --- | --- | --- |
| `Phase1CodebookEvaluator` | 五层 codebook validation 编排器 | `collect_snapshot()` 聚合 `n` 条样本：`states [n,h,state_dim]`、`decoded_logits [n,h,action_dim]`、`code_ids [n]`、`z_e/z_q [n,latent_dim]`、`distances [n,K]` |

### `src/phase1/evaluators/phase1_validation_layers/layer4_label_predictability.py`

| 类名 | 说明 | 维度字段 |
| --- | --- | --- |
| `LinearProbe` | label predictability 的线性探针 | `weight` 形状通常为 `[num_labels, feature_dim]`；`bias` 为 `[num_labels]`；`feature_mean/std` 为 `[feature_dim]` |
| `ProbeMetrics` | 探针预测结果与准确率 | `probe_probs [n,num_labels]`、`ranked_labels [n,num_labels]`、`top1_predictions [n]` |

### `src/phase1/evaluators/phase1_validation_layers/*.py`

其余 layer 文件以函数为主，不定义类：

| 文件 | 说明 |
| --- | --- |
| `layer0_teacher_quality.py` | 计算 DP teacher 质量 raw metrics |
| `layer1_vq_internal.py` | 计算 VQ 内部健康度 raw metrics |
| `layer2_behavior_quality.py` | 计算行为结构质量 raw metrics |
| `layer3_oracle_profitability.py` | 计算 oracle profitability raw metrics |
| `hungarian_matching_helper.py` | code 对齐辅助函数 |
| `__init__.py` | 汇总导出 layer calculator |

## 7. Phase I Metrics Schema

### `src/phase1/metrics/phase1_metrics.py`

| 类名 | 说明 | 维度字段 |
| --- | --- | --- |
| `Phase1Metrics` | 训练期基础指标聚合 | `num_samples`、`correct_actions`、`total_actions` 表示样本和动作 token 规模 |

### `src/phase1/metrics/phase1_layer_computation.py`

| 类名 | 说明 | 维度字段 |
| --- | --- | --- |
| `Phase1LayerComputationBase` | 五层 validation computation 基类 | `layer_id` 表示 layer 维度 |

### `src/phase1/metrics/phase1_metric_results.py`

| 类名 | 说明 | 维度字段 |
| --- | --- | --- |
| `Phase1MetricResult` | 单个指标判定结果 | 无训练张量维度字段 |
| `Phase1RiskFinding` | 风险发现记录 | `related_codes` 表示关联 code 集合 |
| `Phase1LayerResult` | 单层 validation 结果 | `layer_id`；`metrics` 为该层指标集合 |
| `Phase1ValidationResult` | 一个 checkpoint 的完整 validation 结果 | `layers`、`code_diagnostics`、`drift_diagnostics`、`risk_findings` 为聚合集合；嵌套 payload 保留数组维度 |

### `src/phase1/metrics/phase1_validation_data_schema.py`

| 类名 | 说明 | 维度字段 |
| --- | --- | --- |
| `Phase1EvaluationSnapshot` | 评估器收集的 checkpoint 快照 | `sample_ids [n]`、`states [n,h,state_dim]`、`relative_states [n,h,relative_state_dim]`、`trend_states [n,h,trend_state_dim]`、`prices [n,h,1]`、`demo_actions [n,h]`、`demo_rewards [n,h]`、`decoded_actions [n,h]`、`decoded_logits [n,h,action_dim]`、`code_ids [n]`、`z_e/z_q [n,latent_dim]`、`distances [n,K]`、`depthprices [n,h,20]` |
| `Phase1TieBreakerMetrics` | checkpoint tie-breaker 指标 | 无训练张量维度字段 |
| `Phase1ValidationMetrics` | 五层 metrics 聚合对象 | 维度来自各层 metrics |

### `src/phase1/metrics/phase1_validation_teacher_quality.py`

| 类名 | 说明 | 维度字段 |
| --- | --- | --- |
| `Phase1TeacherQualityPayload` | Layer 0 中间 payload | `dp_returns`、`flat_returns`、`advantages` 长度为样本数 `n` |
| `Phase1TeacherQualityMetrics` | Layer 0 指标集合 | 无训练张量维度字段 |
| `Phase1TeacherQualityComputation` | Layer 0 computation | `layer_id = 0` |
| `Phase1TeacherQualityThresholds` | Layer 0 阈值配置 | 无训练张量维度字段 |

### `src/phase1/metrics/phase1_validation_vq_internal.py`

| 类名 | 说明 | 维度字段 |
| --- | --- | --- |
| `CodeAssignmentSnapshot` | code assignment 快照 | `sample_ids [n]`、`code_ids [n]`、`active_codes <= K`、`code_prototypes [K,latent_dim]`、`action_prototypes [K,h]` |
| `Phase1VQInternalPayload` | Layer 1 中间 payload | `code_distribution [K]`、`active_codes <= K`、`codebook_size = K`、`code_distribution_sample_count = n` |
| `Phase1VQInternalMetrics` | Layer 1 指标集合 | 无训练张量维度字段 |
| `Phase1VQInternalComputation` | Layer 1 computation | `layer_id = 1` |
| `Phase1VQInternalThresholds` | Layer 1 阈值配置 | 无训练张量维度字段 |

### `src/phase1/metrics/phase1_validation_behavior_quality.py`

| 类名 | 说明 | 维度字段 |
| --- | --- | --- |
| `Phase1BehaviorQualityPayload` | Layer 2 中间 payload | `morphology_labels`、`motif_labels`、`active_codes <= K` |
| `Phase1CodeDiagnostic` | 单个 code 的诊断摘要 | `code_id`、`support`、`occupancy` 表示 code 级规模 |
| `Phase1BehaviorQualityMetrics` | Layer 2 指标集合 | `num_codes = K`、`duplicate_code_pair_count` |
| `Phase1BehaviorQualityComputation` | Layer 2 computation | `layer_id = 2`；`code_diagnostics` 长度最多为 `K` |
| `Phase1BehaviorQualityThresholds` | Layer 2 阈值配置 | `min_code_support_abs`、`min_code_support_ratio` 等 code 支持度阈值 |

### `src/phase1/metrics/phase1_validation_oracle_profitability.py`

| 类名 | 说明 | 维度字段 |
| --- | --- | --- |
| `Phase1PairProfitabilityCell` | morphology / motif 组合收益单元 | `support` 表示该组合样本数 |
| `Phase1PerCodeProfitability` | 单 code 收益摘要 | `code_id`，集合长度最多为 `K` |
| `Phase1OracleProfitabilityPayload` | Layer 3 中间 payload | `per_code_profitability <= K`；`decoded_returns`、`dp_returns`、`flat_returns`、`random_label_returns` 长度为 `n`；`pair_profitability_matrix` 为 morphology × motif 组合 |
| `Phase1OracleProfitabilityMetrics` | Layer 3 指标集合 | 无训练张量维度字段 |
| `Phase1OracleProfitabilityComputation` | Layer 3 computation | `layer_id = 3` |
| `Phase1OracleProfitabilityThresholds` | Layer 3 阈值配置 | per-code 阈值作用于最多 `K` 个 code |

### `src/phase1/metrics/phase1_validation_label_predictability.py`

| 类名 | 说明 | 维度字段 |
| --- | --- | --- |
| `Phase1LabelPredictabilityPayload` | Layer 4 中间 payload | `probe_confusion_matrix [K,K]` |
| `Phase1LabelPredictabilityMetrics` | Layer 4 指标集合 | `num_codes = K` |
| `Phase1LabelPredictabilityComputation` | Layer 4 computation | `layer_id = 4` |
| `Phase1LabelPredictabilityThresholds` | Layer 4 阈值配置 | top-k 探针阈值和 label entropy 阈值 |

### `src/phase1/metrics/phase1_validation_config.py`

| 类名 | 说明 | 维度字段 |
| --- | --- | --- |
| `Phase1ValidationScoreWeights` | 五层 score 权重 | 五个评分分量 |
| `Phase1ValidationRuntimeConfig` | validation 运行配置 | `random_label_trials`、`churn_window_epochs`、`probe_epochs`、`probe_batch_size`、`codebook_size` |

### `src/phase1/metrics/phase1_validation_score.py`

| 类名 | 说明 | 维度字段 |
| --- | --- | --- |
| `Phase1ValidationScoreComponent` | 单个 score 分量 | 无训练张量维度字段 |
| `Phase1ValidationScore` | checkpoint 综合 score | `components` 为评分分量集合 |

### `src/phase1/metrics` 中的函数型文件

| 文件 | 说明 |
| --- | --- |
| `phase1_validation_rules.py` | 五层 rule 入口 |
| `phase1_validation_rule_helpers.py` | 阈值比较和 layer result 构造 helper |
| `phase1_validation_score_helpers.py` | score 计算辅助函数 |
| `phase1_validation_risk_findings.py` | risk finding 构造逻辑 |
| `__init__.py` | 汇总导出 Phase I metrics schema 与 helper |

## 8. Phase I Report Schema

### `src/phase1/report/phase1_codebook_report.py`

| 类名 | 说明 | 维度字段 |
| --- | --- | --- |
| `Phase1CodebookReport` | 生成 codebook validation HTML | 无训练张量维度字段 |

### `src/phase1/report/phase1_checkpoint_selection_report.py`

| 类名 | 说明 | 维度字段 |
| --- | --- | --- |
| `Phase1CheckpointSelectionReport` | 生成 checkpoint selection HTML | 无训练张量维度字段 |

### `src/phase1/report/phase1_codebook_report_context.py`

| 类名 | 说明 | 维度字段 |
| --- | --- | --- |
| `Phase1CodebookReportContextBuilder` | 把 validation result 转为 HTML 上下文 | 输出集合维度来自 layers、code diagnostics、risk findings 和 chart series |

### `src/phase1/report/phase1_codebook_report_schema.py`

报告 schema 主要是 HTML 展示模型，不承载训练张量。类名如下：

`Phase1CodebookReportMeta`、`Phase1CodebookReportDocument`、`Phase1ReportHeaderItem`、`Phase1ReportHeader`、`Phase1ReportSummaryView`、`Phase1ReportMetricView`、`Phase1ReportLayerView`、`Phase1ReportCodeDiagnosticView`、`Phase1ReportRiskSummaryView`、`Phase1ReportRiskFindingView`、`Phase1ReportMappingRow`、`Phase1ReportKpiRow`、`Phase1ReportProfitSeriesRow`、`Phase1ReportCodeDistributionRow`、`Phase1ReportLabelTip`、`Phase1ReportScoreBreakdownRow`、`Phase1ReportSeriesPoint`、`Phase1ReportSeries`、`Phase1ReportChartGridLine`、`Phase1ReportChartSeries`、`Phase1ReportLineChart`、`Phase1ReportPairProfitabilityCell`、`Phase1ReportPairProfitabilityRow`、`Phase1ReportPairProfitabilityMatrix`、`Phase1CodebookReportHtmlContext`。

维度字段集中在展示集合：

| 类名 | 维度字段 |
| --- | --- |
| `Phase1ReportLineChart` | `width`、`height`、`grid_lines`、`series`、`detail_charts` |
| `Phase1ReportPairProfitabilityMatrix` | `morphologies`、`motifs`、`rows`、`cells` |
| `Phase1CodebookReportHtmlContext` | `layers`、`code_diagnostics`、`oracle_*_series`、`per_code_profit_series`、`code_distribution`、`risk_findings`、`config_rows`、`artifact_rows` |

### `src/phase1/report/_template.py`

| 类名 | 说明 | 维度字段 |
| --- | --- | --- |
| `TemplateSyntaxError` | 模板解析错误 | 无 |
| `_TextNode` | 模板文本节点 | 无 |
| `_VariableNode` | 模板变量节点 | 无 |
| `_IfNode` | 条件节点 | `true_nodes`、`false_nodes` 为子节点集合 |
| `_ForNode` | 循环节点 | `body_nodes` 为子节点集合 |

### `src/phase1/report/__init__.py`

汇总导出报告相关类，不定义类。

## 9. Phase I 直接使用的模型骨架

### `src/model/data_types.py`

该文件定义 numpy 数据结构类型别名，不定义类。Phase I 相关维度如下：

| 类型 | 形状 |
| --- | --- |
| `HorizonDataset` | `(states [x,h,state_dim], relative_states [x,h,relative_state_dim], trend_states [x,h,trend_state_dim], prices [x,h,1], depthprices [x,h,20])` |
| `DemonstrationTrajectory` | `(s_demo [h,state_dim], relative_s_demo [h,relative_state_dim], trend_s_demo [h,trend_state_dim], a_demo [h], r_demo [h], sample_id scalar)` |
| `TrajectoryDataset` | `list[DemonstrationTrajectory]`，长度为样本数 `n` |

### `src/model/tensor_data_types.py`

该文件定义 PyTorch tensor 类型别名和转换函数，不定义类。Phase I 相关维度如下：

| 类型 | 形状 |
| --- | --- |
| `HorizonTensorDataset` | 与 `HorizonDataset` 相同，但使用 `torch.Tensor` |
| `DemonstrationTrajectoryTensor` | 与 `DemonstrationTrajectory` 相同，但使用 `torch.Tensor` |
| `TrajectoryTensorDataset` | 逐样本 TensorDataset |
| `TrajectoryTensorBatch` | `(states [batch,h,state_dim], relative_states [batch,h,relative_state_dim], trend_states [batch,h,trend_state_dim], actions [batch,h], rewards [batch,h], sample_ids [batch])` |

### `src/model/trajectory_batch.py`

该文件定义 batch 形状校验函数，不定义类。

| 函数 | 维度约束 |
| --- | --- |
| `normalize_trajectory_batch()` | 校验三路状态共享 `[batch,horizon]`，`actions [batch,horizon]`，`rewards [batch,horizon]` 或 `[batch,horizon,1]`，`sample_ids [batch]` |

### `src/model/market_state_input.py`

| 类名 | 说明 | 维度字段 |
| --- | --- | --- |
| `MarketStateInputEncoder` | 三路市场状态输入 adapter 和融合层 | `state_dim`、`relative_state_dim`、`trend_state_dim`、`hidden_dim`；输出 `[batch,horizon,hidden_dim]` |

### `src/model/archetype_encoder.py`

| 类名 | 说明 | 维度字段 |
| --- | --- | --- |
| `ArchetypeTrajectoryEncoder` | 把 demonstration trajectory 编码为连续 latent | `state_dim`、`relative_state_dim`、`trend_state_dim`、`hidden_dim`、`latent_dim`、`action_dim`、`num_layers`；输出 `z_e [batch,latent_dim]` |

### `src/model/archetype_decoder.py`

| 类名 | 说明 | 维度字段 |
| --- | --- | --- |
| `ArchetypeActionDecoder` | 根据市场状态和 archetype latent 重构动作 logits | `state_dim`、`relative_state_dim`、`trend_state_dim`、`hidden_dim`、`latent_dim`、`action_dim`、`num_layers`；输入 `z_q [batch,latent_dim]`，输出 `action_logits [batch,horizon,action_dim]` |

### `src/model/codebook.py`

| 类名 | 说明 | 维度字段 |
| --- | --- | --- |
| `QuantizeOutput` | VQ 最近邻量化输出 | `quantized [batch,latent_dim]`、`code_indices [batch]`、`distances [batch,K]`、`z_q_no_grad [batch,latent_dim]` |
| `CodebookInitResult` | codebook 初始化摘要 | `num_samples`、`num_centers` |
| `CodebookDeadCodeResetResult` | dead code 重置摘要 | `num_samples`、`num_centers`、`dead_code_indices`、`reset_code_indices`、`occupancy [K]` |
| `VectorQuantizer` | VQ codebook 模块 | `num_archetypes = K`、`latent_dim`；embedding 形状 `[K,latent_dim]` |

### `src/model/vq_archetype.py`

| 类名 | 说明 | 维度字段 |
| --- | --- | --- |
| `VqModelOutputs` | Phase I VQ 模型前向输出 | `action_logits [batch,horizon,action_dim]`、`z_e/z_q/z_q_no_grad [batch,latent_dim]`、`code_id/code_indices [batch]` |
| `ArchetypeVQModel` | Phase I encoder + quantizer + decoder 总模型 | `state_dim`、`relative_state_dim`、`trend_state_dim`、`action_dim`、`hidden_dim`、`latent_dim`、`num_archetypes`、`num_layers`；forward 消费 `TrajectoryTensorBatch` |

## 10. Phase I 数据准备骨架

### `src/data/data_load.py`

| 类名 | 说明 | 维度字段 |
| --- | --- | --- |
| `DataLoad` | 读取 feature 数据 | `feature_columns` 决定输出 feature 表的列维度 |

### `src/data/data_preparer.py`

| 类名 | 说明 | 维度字段 |
| --- | --- | --- |
| `DataPreparer` | 数据准备编排器 | `horizon` 决定切片长度 |

### `src/data/feature_spec.py`

| 类名 | 说明 | 维度字段 |
| --- | --- | --- |
| `FeatureBlock` | 一组 feature columns | `columns` 长度决定 block feature 维度 |
| `FeatureInputSpec` | 三路输入 feature spec | `state_blocks`、`relative_state_blocks`、`trend_state_blocks` 的列数分别汇总为 `state_dim`、`relative_state_dim`、`trend_state_dim` |

### `src/data/horizon_builder.py`

| 类名 | 说明 | 维度字段 |
| --- | --- | --- |
| `HorizonBuilder` | 从连续时间序列构建固定 horizon 数据立方体 | `horizon`；输出 `states [x,h,state_dim]`、`relative_states [x,h,relative_state_dim]`、`trend_states [x,h,trend_state_dim]`、`prices [x,h,1]`、`depthprices [x,h,20]` |

### `src/data/state_normalizer.py`

| 类名 | 说明 | 维度字段 |
| --- | --- | --- |
| `StateNormalizer` | 对 state feature 做逐列标准化 | `feature_columns`、`mean [feature_dim]`、`std [feature_dim]` |

### `src/data/resolve_factor.py` 与 `src/data/__init__.py`

这两个文件不定义类。前者用于 factor 解析，后者用于包级导出。

## 11. Phase I 数据流摘要

1. `DataLoad` 读取 feature 表。
2. `HorizonBuilder` 按 `horizon` 切成 `HorizonDataset`。
3. DP planner 生成 `TrajectoryDataset`，每条样本包含三路状态、动作、reward 和 `sample_id`。
4. `build_trajectory_tensor_dataset()` 转成 `TrajectoryTensorBatch`。
5. `ArchetypeTrajectoryEncoder` 输出 `z_e [batch,latent_dim]`。
6. `VectorQuantizer` 输出 `code_indices [batch]` 和 `z_q [batch,latent_dim]`。
7. `ArchetypeActionDecoder` 输出 `action_logits [batch,horizon,action_dim]`。
8. `Phase1CodebookEvaluator` 聚合 validation snapshot 和五层指标。
9. `Phase1CheckpointSelector` 选择 best checkpoint。
10. `HorizonTrainLabelBuilder` 导出每个 horizon 的 `code_label` 和 latent。
