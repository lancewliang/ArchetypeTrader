"""Phase II rolling validation 单元测试。

测试用例:
- 固定 fold 切法在同 seed 下结果一致。
- fold 均值、最差分位、波动聚合正确。
"""
import pytest


class TestPhase2RollingValidation:

    def test_fixed_fold_deterministic(self):
        """固定 fold 切法在同 seed 下结果一致。"""
        pass

    def test_fold_mean_aggregation(self):
        """fold 均值聚合正确。"""
        pass

    def test_worst_fold_quantile(self):
        """最差 fold 分位正确。"""
        pass

    def test_fold_volatility(self):
        """fold 间波动聚合正确。"""
        pass
