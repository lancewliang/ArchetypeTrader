"""Phase II test label 无泄漏集成测试。"""
from __future__ import annotations

import pytest

from src.evaluation.phase2_replay import Phase2TestLabelLeakageError
from tests.unit.evaluation.test_phase2_replay import _runner


class TestPhase2NoTestLabelLeakage:

    @pytest.mark.integration
    def test_test_label_in_decision_path_raises(self, tmp_path):
        """检测到 test label 进入决策路径时抛 Phase2TestLabelLeakageError。"""
        runner = _runner(tmp_path, split="test", labeled=True)
        with pytest.raises(Phase2TestLabelLeakageError):
            runner.run_walk_forward("test", deterministic=True)

    @pytest.mark.integration
    def test_posthoc_baseline_allowed(self, tmp_path):
        """posthoc baseline 可记录但不用于 action 选择。"""
        runner = _runner(tmp_path, split="test", labeled=True)
        baselines = runner.run_baselines("test", include_posthoc_demo_label=True)
        assert "phase1_demo_label" in baselines

    @pytest.mark.integration
    def test_guard_enforced(self, tmp_path):
        """_guard_no_test_label_in_decision_path() 强制执行。"""
        runner = _runner(tmp_path, split="test", labeled=True)
        with pytest.raises(Phase2TestLabelLeakageError):
            runner._guard_no_test_label_in_decision_path("test")
