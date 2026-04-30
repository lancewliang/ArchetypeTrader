"""统一的交易语义层（合并自原 ``src/envs/`` 与 ``src/trading/``）。

包含 env、cost 与 reward 行号对齐。所有阶段（Phase I DP、Phase I student replay、
Phase II/III 训练与回测）必须经由本包获取 reward 与成交语义，避免老师/学生不可比。
"""
