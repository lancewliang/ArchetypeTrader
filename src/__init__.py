"""ArchetypeTrader Phase I 源码根包。

模块职责按照 docs/design/phase1_archetype_discovery_design.md §4 切分:

- ``src.config``  : Phase I 集中配置。
- ``src.data``    : 数据读取、滑窗、采样、horizon 切片、demonstration store、Dataset 适配。
- ``src.planners``: 单次交易 DP 与 demo 批生成。
- ``src.models``  : VQ encoder-decoder 模型组件（含因果 decoder 约束）。
- ``src.trainers``: 训练编排、checkpoint IO 与 best 选择策略。
- ``src.trading`` : 统一的 TradingEnv、CostModel 与 reward 行号映射。
- ``src.evaluation``: 指标、replay、诊断与报告。
- ``src.utils``   : Feather/IPC 与 JSON 等底层 IO 工具。

任何对模型行为产生影响的工程改动（默认值变更、guardrail、行号映射）必须先反映到设计文档，再落到代码。
"""
