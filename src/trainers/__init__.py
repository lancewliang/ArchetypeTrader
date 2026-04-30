"""训练编排包。

切分:
- ``phase1_trainer.py``    : 编排数据构建、DP、训练、评估、checkpoint 写入。
- ``phase1_checkpoint.py`` : 仅 IO，根据 ``selection_policy`` 的 verdict 决定是否 promote 为 best。
- ``selection_policy.py``  : 集中 best 选择规则、guardrail（codebook/risk/behavior/teacher quality）与 fatal collapse 判定。
"""
