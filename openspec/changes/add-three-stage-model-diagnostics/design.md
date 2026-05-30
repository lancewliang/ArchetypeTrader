## Context

现有 validation/report 代码已经把指标计算和报告渲染分开。Phase I 有 evaluation snapshot 和 validation result schema；Phase II 已经为单 code 盈利性、codebook 使用、累积收益、pair profitability matrix 和 code diagnostics 预留了 report payload。`src/analysis/per_code_model.py` 包含这次需求中部分诊断概念的早期草稿，但它还不是稳定的共用 schema，也没有完整定义训练/验证 split 的一致性、模型变体命名、市场形态/动作分类，以及手续费/滑点归因。

本次变更新增一个由 evaluator 生成、由 report 消费的共用诊断层。诊断需要在训练集和验证集上同时比较 Phase I codebook 行为、Phase II selector 行为和 Phase III tuning 行为，并且 HTML 模板不需要重新执行 decoder 推理、selector 推理、动态规划或收益模拟。

## Goals / Non-Goals

**Goals:**

- 为训练集和验证集定义一套共用 diagnostic payload schema。
- 覆盖所有请求的策略变体：demo codebook、DP、样本最佳 codebook、选择器 + 动作、选择器 + 逐步生成动作、选择器 + 微调 + 逐步生成动作，以及一直持有 baseline。
- 提供可直接用于图表/表格的聚合数据：
  - codebook 命中样本数和比例。
  - 累积利润曲线。
  - 单 code 收益/亏损/净利润分布。
  - 市场形态 x 动作类型矩阵。
  - 单 code 在市场形态/动作下的命中与利润诊断。
  - 市场形态/动作汇总表。
- 标准化市场形态和动作类型定义，使不同 evaluator 和 report 能以一致方式分组数据。
- 保持现有 checkpoint selection metrics 不变；诊断作为额外 report/audit payload 提供。

**Non-Goals:**

- 不训练或重新设计 Phase I、Phase II、Phase III 模型。
- 不改变 checkpoint 排序、hard gate 或训练 loss 公式。
- 不在普通 validation result JSON 中保存完整逐时间步 trace。共用 payload 只保存聚合数据和有边界的图表序列。
- 如果 Phase III 模型内部尚不存在，本次不实现它。schema 需要在 Phase III 输出可用时接收它，否则显式标记对应变体不可用。

## Decisions

### Decision 1: 创建共用分析 Schema

在 `src/analysis/` 下引入共用 diagnostic model，由各阶段 evaluator 填充。schema 应定义：

- `DiagnosticSplitResult`: 每个 split 一个结果，例如 `train`、`validation`，后续可扩展 `test`。
- `DiagnosticVariant`: 稳定的变体标识，例如 `demo_codebook`、`dp_upper_bound`、`sample_best_codebook`、`selector_action`、`selector_step_by_step`、`selector_tuning_step_by_step` 和 `long_hold_baseline`。
- `CodeUsageRow`: code id、命中数量、命中比例、split 总样本数。
- `CumulativeProfitSeries`: 每个变体的有序累积净利润点。
- `PerCodeProfitRow`: code id、收益、亏损、净利润、毛利润、手续费、滑点、利润比例、样本数、样本比例。
- `MarketActionCell`: 市场形态、动作类型、样本数、样本比例、毛利润、净利润、利润比例、手续费、滑点。
- `PerCodeMarketActionDiagnosticRow`: code id 加上市场/动作 cell 指标和 code 级总计。

备选方案：保留 Phase I 和 Phase II 各自独立的 report payload。该方案会重复分类、格式化和比例计算逻辑，也会让 Phase III 接入不一致。

### Decision 2: 在 Evaluator 中计算诊断，而不是在 Report 中计算

evaluator 或离线 analysis builder 负责计算所有策略收益、code assignment、动作分组、市场分组、手续费、滑点和聚合比例。report context builder 和模板只把已经计算好的 payload 格式化成 SVG/HTML 表格。

备选方案：在渲染 HTML 时计算聚合数据。该方案会把模型/收益逻辑混入展示层，并且当依赖或 checkpoint 变化时报告难以复现。

### Decision 3: 使用稳定且可扩展的分类标签

市场形态应由价格/状态特征通过确定性分类器生成，默认分类包括：

- 趋势：`uptrend`、`downtrend`、`sideways`。
- 波动：`low_volatility`、`normal_volatility`、`high_volatility`。
- 当可用时，盘口压力或微观结构状态：`buy_pressure`、`sell_pressure`、`balanced_pressure`。

报告矩阵的行 key 应使用标准化的 `market_regime` 标签。实现可以编码组合标签，例如 `uptrend|high_volatility|buy_pressure`，但 schema 也必须保留组件字段，便于后续改变分组方式时不必重新计算模型输出。

动作类型应由仓位/动作转移推导：

- `enter_long`, `hold_long`, `exit_long`.
- `enter_short`, `hold_short`, `exit_short`.
- `stay_flat`.
- `flip_long_to_short`, `flip_short_to_long`.

备选方案：只使用原始 action id。原始 id 更紧凑，但无法解释行为，尤其是在手续费和动作转移重要时。

### Decision 4: 统一定义利润归因

所有变体的利润都应使用同一套执行口径：

- 扣除交易成本前的毛利润。
- 手续费成本。
- 当深度数据可用时，计算来自深度数据的滑点成本。
- 扣除手续费和滑点后的净利润。
- 收益定义为正净样本利润之和。
- 亏损可定义为负净样本利润的绝对值或带符号求和，但必须在 schema 中统一记录。

比例应基于 split 总量计算：

- 样本比例 = cell 样本数 / split 样本数。
- 利润比例 = cell 净利润 / 总绝对净利润分母，并安全处理零分母。
- 手续费/滑点拖累比例 = 成本 / 绝对毛利润分母，并安全处理零分母。

备选方案：让每个 report 自己计算比例。该方案会导致训练/验证对比口径不一致，也会让诊断难以测试。

### Decision 5: 并排保留训练集和验证集结果

顶层 diagnostic document 应包含按 split 分组的结果，使每个图表/表格都能把训练集和验证集一起展示，或以相邻 tab 展示。两个 split 必须使用相同的变体和表格列；当某个变体无法计算时，使用 `available=false` 或空序列表示。

备选方案：生成互不相关的训练集报告和验证集报告。该方案会让泛化差距更难观察，也会增加产物发现复杂度。

## Risks / Trade-offs

- 单 code x 市场形态 x 动作矩阵可能导致 payload 很大 -> 按有限类别 cell 聚合，普通 report payload 不持久化完整逐时间步 trace。
- 市场形态定义可能过粗，或过度依赖当前特征 -> 同时保存组件标签和组合标签，并集中实现分类器，使定义可以演进。
- Phase III 输出可能尚不存在 -> schema 中保留该变体并提供明确 availability metadata；report 只在数据存在时渲染它。
- 样本最佳 codebook 可能成本较高，因为需要为每个样本评估每个 code -> 将其作为离线/evaluator 诊断路径实现，支持 batching 和可选限制，不放入训练热路径。
- DP 和 demo codebook 可能包含未来信息 -> 将这些变体标记为 reference/oracle 诊断，并在视觉上与可部署 selector 变体区分。

## Migration Plan

1. 在 `src/analysis/` 下新增共用 diagnostic schema 和纯聚合 helper。
2. 重构或替换 `src/analysis/per_code_model.py`，使字段校验、序列化和命名保持一致。
3. 使用小型合成 split 添加 calculator 测试，覆盖已知 code id、动作、市场形态、手续费、滑点和利润。
4. 在不改变 checkpoint selection metrics 的前提下，把共用 payload 接入 Phase I 和 Phase II validation result/report payload。
5. 为训练集和验证集诊断区块添加 report context/table/chart builder。
6. 当 tuning evaluator 可用时，接入 Phase III 变体。

回滚是增量安全的：report 可以忽略新的 diagnostic payload 字段，checkpoint selection 仍然使用现有 metrics。

## Open Questions

- 第一版市场形态标签应由哪些具体状态列定义：只用价格特征、使用现有 factor 输出，还是两者都用？
- `loss` 应序列化为带负号的数值，还是正数幅度？实现前 schema 需要选择统一约定。
- 当 dataset 样本顺序、时间戳顺序和 evaluation batch 顺序不一致时，累积利润曲线应按哪一种顺序排列？
