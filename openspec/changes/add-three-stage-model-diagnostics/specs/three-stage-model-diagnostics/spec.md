## ADDED Requirements

### Requirement: Split-scoped diagnostic document
系统 SHALL 产出机器可读的三阶段诊断文档，其中包含 `train` 和 `validation` 两个 split 的独立结果，并且两个结果使用相同 schema。

#### Scenario: Train and validation diagnostics are present
- **WHEN** 某次运行同时具备训练集和验证集评估数据，并生成诊断结果
- **THEN** 诊断文档同时包含 `train` 和 `validation` split 结果，且两者拥有相同的图表和表格 payload key

#### Scenario: Missing split is explicit
- **WHEN** 请求的某个 split 无法被评估
- **THEN** 诊断文档将该 split 记录为不可用并给出原因，而不是静默省略

### Requirement: Standard strategy variants
系统 SHALL 为 `demo_codebook`、`dp_upper_bound`、`sample_best_codebook`、`selector_action`、`selector_step_by_step`、`selector_tuning_step_by_step` 和 `long_hold_baseline` 使用稳定的策略变体标识。

#### Scenario: Variant metadata distinguishes oracle and deployable results
- **WHEN** 诊断文档列出策略变体
- **THEN** 每个变体都包含稳定标识、展示标签、可用性标记以及分类字段，用于区分 reference/oracle 变体和可部署 selector 变体

#### Scenario: Phase III tuning variant is unavailable
- **WHEN** 某次运行没有可用的 Phase III tuning 输出
- **THEN** `selector_tuning_step_by_step` 仍保留在变体 metadata 中，并标记 `available=false`，且不会生成误导性的零利润图表序列

### Requirement: Code sample distribution
系统 SHALL 对每个会产生 code 的变体，在每个 split 上报告每个 codebook code 的命中样本数和命中比例。

#### Scenario: Code usage rows sum to split total
- **WHEN** 为某个 split 和变体生成 code 样本分布
- **THEN** 所有 code 命中数量之和等于该 split 被统计的样本总数，且每个命中比例都使用同一个分母计算

#### Scenario: Empty codes are visible
- **WHEN** 某个 code 在 split 中命中样本数为零
- **THEN** 该 code 仍出现在分布中，count 为 `0`，ratio 为 `0`

### Requirement: Cumulative profit curves
系统 SHALL 为每个 split 上所有可用变体报告累积净利润序列。

#### Scenario: All requested profit curves are generated
- **WHEN** 为某个 split 生成累积利润诊断
- **THEN** 输出包含 demo codebook、DP upper bound、样本最佳 codebook、选择器 + 动作、选择器 + 逐步生成动作、可用时的选择器 + 微调 + 逐步生成动作，以及一直持有 baseline 的序列

#### Scenario: Cumulative curve uses consistent ordering
- **WHEN** 在同一个 split 上为多个变体计算累积利润点
- **THEN** 所有变体序列使用相同样本顺序，并在 metadata 中暴露该排序依据

### Requirement: Per-code profit distribution
系统 SHALL 为每个支持的策略变体报告单 code 的收益、亏损、毛利润、手续费、滑点、净利润、样本数、样本比例和利润比例。

#### Scenario: Per-code profit accounting includes costs
- **WHEN** 生成单 code 利润分布
- **THEN** 每一行都暴露毛利润、手续费、滑点和净利润，使成本拖累可以被审计

#### Scenario: Profit ratio handles zero denominator
- **WHEN** split 总利润分母为零
- **THEN** 单 code 利润比例仍是有限值，并且 payload 记录零分母情况

### Requirement: Market-regime taxonomy
系统 SHALL 将每个被评估样本分类到确定性的市场形态标签，并在可用时保留市场形态组件字段。

#### Scenario: Market regime is assigned for every counted sample
- **WHEN** 生成市场形态诊断
- **THEN** 每个被统计样本都有市场形态标签；只有在必要输入不可用时才使用明确的 unknown 标签

#### Scenario: Regime definitions are serialized
- **WHEN** 产出诊断文档
- **THEN** 文档包含该次运行使用的市场形态分类版本和定义

### Requirement: Action-type taxonomy
系统 SHALL 将动作行为分类为由 action 或 position 转移推导出的确定性动作类型。

#### Scenario: Action type is assigned from transition
- **WHEN** 某个样本从 flat 切换到 long
- **THEN** 动作类型分类器将该转移标记为 `enter_long`

#### Scenario: Action definitions are serialized
- **WHEN** 产出诊断文档
- **THEN** 文档包含该次运行使用的动作类型分类版本和定义

### Requirement: Market-regime by action-type summary table
系统 SHALL 为每个可用策略变体和 split 产出以市场形态为行、动作类型为列的汇总矩阵。

#### Scenario: Summary table contains sample and profit metrics
- **WHEN** 生成市场形态/动作汇总矩阵
- **THEN** 每个 cell 包含命中样本数、命中样本比例、毛利润、净利润、利润比例、手续费和滑点

#### Scenario: Summary table is variant scoped
- **WHEN** 诊断包含多个策略变体
- **THEN** 每个可用变体都分别拥有训练集和验证集上的市场形态/动作汇总矩阵

### Requirement: Per-code market-action diagnostics
系统 SHALL 为每个会产生 code 的变体，按市场形态和动作类型产出最细粒度的单 code 诊断。

#### Scenario: Per-code market-action hit metrics are available
- **WHEN** 生成单 code 市场/动作诊断
- **THEN** 每行包含 code id、市场形态、动作类型、命中样本数、命中样本比例，以及在该 code 内部的比例

#### Scenario: Per-code market-action profit metrics are available
- **WHEN** 生成单 code 市场/动作诊断
- **THEN** 每行包含该 code/市场形态/动作 cell 的毛利润、净利润、利润比例、手续费和滑点

### Requirement: Code interpretability diagnostics
系统 SHALL 提供 code 级总计和集中度指标，用于判断某个 code 是否具备足够行为可解释性。

#### Scenario: Code-level totals are reported
- **WHEN** 生成 code 诊断
- **THEN** 每个 code 包含总利润、总样本数、总利润比例、总样本命中比例、手续费和滑点

#### Scenario: Dominant behavior concentration is reported
- **WHEN** 某个 code 具备市场形态/动作 cell
- **THEN** 诊断识别该 code 的主导市场形态、主导动作类型、主导市场形态/动作 pair，以及它们在该 code 内的比例

### Requirement: Report-ready chart and table payloads
系统 SHALL 以 report-ready 的聚合 payload 暴露诊断，使 HTML 或静态图表渲染器不需要重新执行模型推理、decoder 生成、DP planning 或收益模拟。

#### Scenario: Report consumes aggregates only
- **WHEN** report 渲染三阶段诊断图表和表格
- **THEN** report 只读取聚合 diagnostic payload，不需要访问模型 checkpoint、dataloader 或原始逐时间步 trace

#### Scenario: Payload supports existing report sections
- **WHEN** Phase I 或 Phase II validation report 包含三阶段诊断
- **THEN** code 分布、累积利润、单 code 利润、市场/动作汇总和 code 诊断区块都可以从共用 payload 渲染
