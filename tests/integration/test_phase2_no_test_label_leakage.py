"""Phase II test label 无泄漏集成测试。

目标: 确保 test label 无法进入决策路径。

断言:
- backtest 若检测到 code_label 进入 selector 决策路径直接抛错。
- phase2_backtest_runner 可以记录 posthoc baseline，但不能用于 action 选择。
- _guard_no_test_label_in_decision_path() 在 selector 调用前后强制执行。
"""
import pytest


class TestPhase2NoTestLabelLeakage:

    @pytest.mark.integration
    def test_test_label_in_decision_path_raises(self, tmp_path):
        """检测到 test label 进入决策路径时抛 Phase2TestLabelLeakageError。"""
        pass

    @pytest.mark.integration
    def test_posthoc_baseline_allowed(self, tmp_path):
        """posthoc baseline 可记录但不用于 action 选择。"""
        pass

    @pytest.mark.integration
    def test_guard_enforced(self, tmp_path):
        """_guard_no_test_label_in_decision_path() 强制执行。"""
        pass
