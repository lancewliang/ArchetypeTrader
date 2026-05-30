## 1. 共用诊断 Schema

- [ ] 1.1 在 `src/analysis/` 中定义共用诊断模型，覆盖 split 结果、变体 metadata、code 使用、累积利润序列、单 code 利润行、市场/动作 cell，以及单 code 市场/动作诊断。
- [ ] 1.2 重构或替换 `src/analysis/per_code_model.py`，使现有草稿概念使用共用 schema、合法 import、兼容 float 的数值类型和稳定变体标识。
- [ ] 1.3 添加市场形态和动作类型的分类定义，并序列化分类版本 metadata。
- [ ] 1.4 添加安全比例 helper，覆盖样本比例、利润比例、手续费拖累和滑点拖累，并记录零分母 metadata。

## 2. 诊断计算器

- [ ] 2.1 基于 action 或 position 转移实现动作类型分类。
- [ ] 2.2 基于可用价格/状态/深度特征实现市场形态分类，并提供明确 unknown fallback。
- [ ] 2.3 为所有会产生 code 的变体实现 code 样本分布聚合。
- [ ] 2.4 为所有可用策略变体实现累积净利润序列聚合，并使用一致样本顺序。
- [ ] 2.5 实现单 code 收益/亏损/毛利润/手续费/滑点/净利润分布聚合。
- [ ] 2.6 实现市场形态 x 动作类型汇总矩阵聚合。
- [ ] 2.7 实现单 code 市场形态/动作诊断和 code 级可解释性总计。

## 3. 变体接入

- [ ] 3.1 从 Phase I evaluation data 接入 demo codebook 和 DP upper-bound 诊断输入。
- [ ] 3.2 实现样本最佳 codebook 诊断评估，使用 batched 单 code 利润比较。
- [ ] 3.3 接入 Phase II 选择器 + 动作，以及选择器 + 逐步生成动作诊断。
- [ ] 3.4 添加 Phase III 选择器 + 微调 + 逐步生成动作诊断接入；当 Phase III 输出缺失时写入不可用 metadata。
- [ ] 3.5 通过同一利润归因路径添加一直持有 baseline 利润计算。

## 4. Report Payload 与渲染

- [ ] 4.1 在不改变 checkpoint selection metrics 的前提下，把共用 diagnostic payload 加入 Phase I validation/report document。
- [ ] 4.2 在不改变 checkpoint selection metrics 的前提下，把共用 diagnostic payload 加入 Phase II validation/report document。
- [ ] 4.3 扩展 report context builder，暴露训练集/验证集诊断图表和表格 view model。
- [ ] 4.4 扩展 HTML 模板，为训练集和验证集 split 渲染 code 样本分布、累积利润曲线、单 code 利润分布、市场/动作汇总矩阵和 code 诊断表。

## 5. 测试与验证

- [ ] 5.1 为分类器和安全比例 helper 添加单元测试。
- [ ] 5.2 使用已知 code id、动作、利润、手续费、滑点和市场形态添加合成聚合测试。
- [ ] 5.3 添加 schema 序列化测试，证明训练集和验证集 split payload 使用相同 schema。
- [ ] 5.4 添加 report context 测试，证明渲染只使用聚合 payload，不需要模型 checkpoint 或 dataloader。
- [ ] 5.5 在 `ArachetypeTrade` conda 环境中运行相关测试套件。
