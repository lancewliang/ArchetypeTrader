## Why

项目需要一套共用诊断视图，用来解释三阶段模型是否学到了有用且可解释的交易行为，而不只是输出单一的 validation 分数。同一套诊断必须能同时比较训练集和验证集表现，从而直观看到过拟合、code collapse、选择器错配以及微调后退化等问题。

## What Changes

- 新增训练集和验证集共用的诊断能力，覆盖 code 使用分布、累积利润曲线、单 code 收益/亏损分布、市场形态/动作矩阵、单 code 可解释性诊断以及市场形态/动作汇总表。
- 标准化需要评估的策略变体：
  - demo codebook，包含使用未来信息的超额利润。
  - DP 理论最大利润上限。
  - 样本最佳 codebook，对每个样本穷举所有 code 并选择利润最大的 code。
  - 选择器 + 动作，使用第二阶段选择器产生的 code，并统计一个样本分片中所有时间步生成动作的累积利润。
  - 选择器 + 逐步生成动作，使用第二阶段选择器产生的 code，并按自回归方式逐步生成样本分片内动作。
  - 选择器 + 微调 + 逐步生成动作，使用第三阶段选择器产生的 code，经微调层后逐步生成动作。
  - 一直持有做多 baseline。
- 定义所有图表和表格共用的稳定市场形态分类与动作类型分类。
- 产出机器可读聚合 payload，记录 code 命中样本数、命中比例、利润、利润比例、收益、亏损、净利润、手续费、滑点以及相对 baseline 的利润。
- 让这些诊断能被 Phase I codebook validation、Phase II selector validation 和 Phase III tuning evaluation 报告复用，报告渲染阶段不重新执行模型推理。

## Capabilities

### New Capabilities
- `three-stage-model-diagnostics`: 共用训练/验证诊断 payload、图表定义和表格定义，用于解释 codebook、selector 和 tuning 输出下的三阶段模型质量。

### Modified Capabilities
- 无。

## Impact

- 受影响代码区域：
  - `src/analysis/per_code_model.py`，或替代它的共用诊断 schema/计算层。
  - `src/phase1/` 与 `src/phase2/` 下的 Phase I、Phase II validation/report payload 生成逻辑。
  - 后续 Phase III tuning 层存在后，对应 evaluator/report 接入逻辑。
  - `src/phase1/report/templates/` 与 `src/phase2/report/templates/` 下的报告模板。
- 受影响产物：
  - 训练集与验证集诊断 JSON payload。
  - 消费这些 payload 的 HTML 报告区块或静态图表/表格资源。
- 不计划破坏现有 checkpoint selection metrics；本次诊断是增量的、面向报告和审计的能力。
