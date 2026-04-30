"""数据层包。

边界:
- 不调用 DP（DP 只在 ``src.planners``）。
- 不重新拟合状态特征 scaler（外部数据视为已对齐和清洗）。
- ``close`` 列只能进入 ``prices``；``feature_columns`` 不得包含 ``close``。
"""
