"""PolicyAdapter 单元测试"""

import pytest

from src.phase3.policy_adapter import PolicyAdapter


class TestPolicyAdapterInit:
    """初始化测试"""

    def test_initial_state(self):
        adapter = PolicyAdapter()
        assert adapter.adjusted_in_horizon is False


class TestComputeFinalAction:
    """compute_final_action 核心逻辑测试"""

    def test_no_adjustment_signal(self):
        """a_ref = 0 → a_final = a_base"""
        adapter = PolicyAdapter()
        assert adapter.compute_final_action(a_base=1, a_base_prev=1, a_ref=0) == 1
        assert adapter.adjusted_in_horizon is False

    def test_base_action_changed(self):
        """a_base ≠ a_base_prev → a_final = a_base（不精炼）"""
        adapter = PolicyAdapter()
        assert adapter.compute_final_action(a_base=2, a_base_prev=1, a_ref=1) == 2
        assert adapter.adjusted_in_horizon is False

    def test_reduce_signal(self):
        """a_ref = -1 → a_final = 0"""
        adapter = PolicyAdapter()
        assert adapter.compute_final_action(a_base=1, a_base_prev=1, a_ref=-1) == 0
        assert adapter.adjusted_in_horizon is True

    def test_increase_signal(self):
        """a_ref = 1 → a_final = 2"""
        adapter = PolicyAdapter()
        assert adapter.compute_final_action(a_base=1, a_base_prev=1, a_ref=1) == 2
        assert adapter.adjusted_in_horizon is True

    def test_already_adjusted_ignores_ref(self):
        """已调整过 → 忽略 a_ref，返回 a_base"""
        adapter = PolicyAdapter()
        # 第一次调整
        adapter.compute_final_action(a_base=1, a_base_prev=1, a_ref=-1)
        assert adapter.adjusted_in_horizon is True
        # 第二次应被忽略
        result = adapter.compute_final_action(a_base=1, a_base_prev=1, a_ref=1)
        assert result == 1  # 返回 a_base，不是 2

    def test_at_most_one_adjustment_per_horizon(self):
        """验证每 horizon 最多一次调整"""
        adapter = PolicyAdapter()
        # 第一次: 调整生效
        r1 = adapter.compute_final_action(a_base=1, a_base_prev=1, a_ref=1)
        assert r1 == 2
        # 后续多次调整请求均被忽略
        r2 = adapter.compute_final_action(a_base=1, a_base_prev=1, a_ref=-1)
        assert r2 == 1
        r3 = adapter.compute_final_action(a_base=0, a_base_prev=0, a_ref=1)
        assert r3 == 0


class TestReset:
    """reset 方法测试"""

    def test_reset_clears_adjusted_flag(self):
        adapter = PolicyAdapter()
        adapter.compute_final_action(a_base=1, a_base_prev=1, a_ref=-1)
        assert adapter.adjusted_in_horizon is True
        adapter.reset()
        assert adapter.adjusted_in_horizon is False

    def test_adjustment_works_after_reset(self):
        """reset 后可以再次调整"""
        adapter = PolicyAdapter()
        # 第一个 horizon: 调整
        adapter.compute_final_action(a_base=1, a_base_prev=1, a_ref=1)
        assert adapter.adjusted_in_horizon is True
        # 新 horizon
        adapter.reset()
        # 应该可以再次调整
        result = adapter.compute_final_action(a_base=1, a_base_prev=1, a_ref=-1)
        assert result == 0
        assert adapter.adjusted_in_horizon is True


class TestEdgeCases:
    """边界情况测试"""

    def test_all_actions_as_base(self):
        """测试所有有效 a_base 值"""
        for a_base in [0, 1, 2]:
            adapter = PolicyAdapter()
            assert adapter.compute_final_action(a_base=a_base, a_base_prev=a_base, a_ref=0) == a_base

    def test_reduce_from_any_base(self):
        """从任意 a_base 减仓"""
        for a_base in [0, 1, 2]:
            adapter = PolicyAdapter()
            assert adapter.compute_final_action(a_base=a_base, a_base_prev=a_base, a_ref=-1) == 0

    def test_increase_from_any_base(self):
        """从任意 a_base 加仓"""
        for a_base in [0, 1, 2]:
            adapter = PolicyAdapter()
            assert adapter.compute_final_action(a_base=a_base, a_base_prev=a_base, a_ref=1) == 2

    def test_base_changed_overrides_ref(self):
        """a_base 变化时，即使 a_ref 非零也不调整"""
        adapter = PolicyAdapter()
        result = adapter.compute_final_action(a_base=0, a_base_prev=2, a_ref=-1)
        assert result == 0  # a_base, not because of a_ref=-1
        assert adapter.adjusted_in_horizon is False  # 不算调整


# ---------------------------------------------------------------------------
# Property-Based Tests (hypothesis)
# ---------------------------------------------------------------------------
from hypothesis import given, settings
from hypothesis import strategies as st


class TestPolicyAdapterProperties:
    """PolicyAdapter 属性测试。"""

    @given(
        a_ref_seq=st.lists(
            st.sampled_from([-1, 0, 1]), min_size=1, max_size=72
        ),
        a_base_seq=st.lists(
            st.sampled_from([0, 1, 2]), min_size=1, max_size=72
        ),
    )
    @settings(max_examples=100)
    def test_prop18_at_most_one_adjustment_per_horizon(
        self, a_ref_seq, a_base_seq
    ):
        """Property 18: 每 Horizon 最多一次调整 — 非零调整次数 ≤ 1。

        **Validates: Requirements 6.2**

        # Feature: archetype-trader, Property 18: 每 Horizon 最多一次调整
        """
        # Align lengths
        length = min(len(a_ref_seq), len(a_base_seq))
        a_ref_seq = a_ref_seq[:length]
        a_base_seq = a_base_seq[:length]

        adapter = PolicyAdapter()
        adjustment_count = 0

        for i in range(length):
            a_base = a_base_seq[i]
            # Use previous a_base as a_base_prev (first step uses same value)
            a_base_prev = a_base_seq[i - 1] if i > 0 else a_base
            a_ref = a_ref_seq[i]

            a_final = adapter.compute_final_action(a_base, a_base_prev, a_ref)

            # Count adjustments: a_final differs from a_base AND it was due to
            # refinement (a_base == a_base_prev and a_ref != 0)
            if a_base == a_base_prev and a_ref != 0 and a_final != a_base:
                adjustment_count += 1

        assert adjustment_count <= 1, (
            f"Expected at most 1 adjustment per horizon, got {adjustment_count}"
        )

    @given(
        a_base=st.sampled_from([0, 1, 2]),
        a_base_prev=st.sampled_from([0, 1, 2]),
        a_ref=st.sampled_from([-1, 0, 1]),
    )
    @settings(max_examples=100)
    def test_prop22_final_action_eq6(self, a_base, a_base_prev, a_ref):
        """Property 22: 最终动作计算正确性（Eq. 6）。

        **Validates: Requirements 6.2, 6.3**

        # Feature: archetype-trader, Property 22: 最终动作计算正确性

        验证:
        - a_base ≠ a_base_prev → a_final = a_base
        - a_ref = 0 → a_final = a_base
        - a_ref = -1 (且 a_base == a_base_prev) → a_final = 0
        - a_ref = 1 (且 a_base == a_base_prev) → a_final = 2
        """
        adapter = PolicyAdapter()
        a_final = adapter.compute_final_action(a_base, a_base_prev, a_ref)

        if a_base != a_base_prev:
            assert a_final == a_base, (
                f"When a_base({a_base}) != a_base_prev({a_base_prev}), "
                f"a_final should be a_base, got {a_final}"
            )
        elif a_ref == 0:
            assert a_final == a_base, (
                f"When a_ref=0, a_final should be a_base({a_base}), got {a_final}"
            )
        elif a_ref == -1:
            assert a_final == 0, (
                f"When a_ref=-1 and a_base==a_base_prev, "
                f"a_final should be 0, got {a_final}"
            )
        elif a_ref == 1:
            assert a_final == 2, (
                f"When a_ref=1 and a_base==a_base_prev, "
                f"a_final should be 2, got {a_final}"
            )
