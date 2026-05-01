"""Selection metrics 单元测试。

测试用例:
- action dominance / active archetype ratio 计算正确。
- dead code usage 检查正确。
"""
import pytest

from src.evaluation.metrics.selection import (
    action_dominance_ratio,
    active_archetype_ratio,
    dead_code_usage_check,
)


class TestSelectionMetrics:

    def test_action_dominance_ratio_uniform(self):
        """均匀分布时 dominance = 1/K。"""
        actions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert action_dominance_ratio(actions, 10) == pytest.approx(0.1)

    def test_action_dominance_ratio_single(self):
        """全部选同一个 code 时 dominance = 1.0。"""
        actions = [3, 3, 3, 3, 3]
        assert action_dominance_ratio(actions, 10) == pytest.approx(1.0)

    def test_action_dominance_ratio_empty(self):
        assert action_dominance_ratio([], 10) == 0.0

    def test_active_archetype_ratio_all_used(self):
        """所有 code 都被使用时 ratio = 1.0。"""
        actions = list(range(10))
        assert active_archetype_ratio(actions, 10) == pytest.approx(1.0)

    def test_active_archetype_ratio_partial(self):
        """只用了 3 个 code。"""
        actions = [0, 0, 1, 1, 2, 2]
        assert active_archetype_ratio(actions, 10) == pytest.approx(0.3)

    def test_active_archetype_ratio_empty(self):
        assert active_archetype_ratio([], 10) == 0.0

    def test_dead_code_usage_check_no_dead(self):
        """没有 dead code 时 count=0。"""
        actions = [0, 1, 2]
        mask = [False] * 10
        result = dead_code_usage_check(actions, mask)
        assert result["dead_code_selected_count"] == 0
        assert result["dead_code_selected_ratio"] == 0.0

    def test_dead_code_usage_check_with_dead(self):
        """选择了 dead code。"""
        actions = [0, 1, 5, 5]
        mask = [False] * 10
        mask[5] = True  # code 5 is dead
        result = dead_code_usage_check(actions, mask)
        assert result["dead_code_selected_count"] == 2
        assert result["dead_code_selected_ratio"] == pytest.approx(0.5)
