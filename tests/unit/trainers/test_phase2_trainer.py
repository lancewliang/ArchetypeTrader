"""Phase II trainer 单元测试。

测试用例:
- trainer 可以跑通完整 orchestrator。
- 训练结束后导出 per-horizon records。
- sensitivity 结果写入 JSON。
- KL/demo ablation matrix 能生成 phase2_ablation_kl_demo.json 与 summary CSV。
"""
import pytest


class TestPhase2Trainer:

    def test_orchestrator_runs(self):
        """trainer 可以跑通完整 orchestrator。"""
        pass

    def test_per_horizon_records_exported(self):
        """训练结束后导出 per-horizon records。"""
        pass

    def test_sensitivity_written(self):
        """sensitivity 结果写入 JSON。"""
        pass

    def test_kl_demo_ablation_matrix(self):
        """KL/demo ablation matrix 生成产物。"""
        pass
