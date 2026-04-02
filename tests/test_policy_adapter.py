"""PolicyAdapter 单元测试"""

import pytest

from src.phase3.policy_adapter import PolicyAdapter


class TestComputeFinalAction:
    """compute_final_action 核心逻辑测试"""

    def test_no_adjustment_signal(self):
        """a_ref = 0 → a_final = a_base"""
        adapter = PolicyAdapter()
        assert adapter.compute_final_action(a_base=1, a_base_prev=1, a_ref=0, has_adjusted=False) == (1, False)

    def test_base_action_changed(self):
        """a_base ≠ a_base_prev → a_final = a_base（不精炼）"""
        adapter = PolicyAdapter()
        assert adapter.compute_final_action(a_base=2, a_base_prev=1, a_ref=1, has_adjusted=False) == (2, False)

    def test_reduce_signal(self):
        """a_ref = -1 → a_final = 0"""
        adapter = PolicyAdapter()
        assert adapter.compute_final_action(a_base=1, a_base_prev=1, a_ref=-1, has_adjusted=False) == (0, True)

    def test_increase_signal(self):
        """a_ref = 1 → a_final = 2"""
        adapter = PolicyAdapter()
        assert adapter.compute_final_action(a_base=1, a_base_prev=1, a_ref=1, has_adjusted=False) == (2, True)

    def test_already_adjusted_ignores_ref(self):
        """已调整过 → 忽略 a_ref，返回 a_base"""
        adapter = PolicyAdapter()
        assert adapter.compute_final_action(a_base=1, a_base_prev=1, a_ref=1, has_adjusted=True) == (1, True)

    def test_at_most_one_adjustment_per_horizon(self):
        """验证每 horizon 最多一次调整（通过显式 has_adjusted 状态跟踪）"""
        adapter = PolicyAdapter()
        # 第一次: 调整生效
        a_final, has_adjusted = adapter.compute_final_action(a_base=1, a_base_prev=1, a_ref=1, has_adjusted=False)
        assert a_final == 2
        assert has_adjusted is True
        # 后续多次调整请求均被忽略（传入 has_adjusted=True）
        a_final, has_adjusted = adapter.compute_final_action(a_base=1, a_base_prev=1, a_ref=-1, has_adjusted=has_adjusted)
        assert a_final == 1
        assert has_adjusted is True
        a_final, has_adjusted = adapter.compute_final_action(a_base=0, a_base_prev=0, a_ref=1, has_adjusted=has_adjusted)
        assert a_final == 0
        assert has_adjusted is True


class TestEdgeCases:
    """边界情况测试"""

    def test_all_actions_as_base(self):
        """测试所有有效 a_base 值"""
        adapter = PolicyAdapter()
        for a_base in [0, 1, 2]:
            assert adapter.compute_final_action(a_base=a_base, a_base_prev=a_base, a_ref=0, has_adjusted=False) == (a_base, False)

    def test_reduce_from_any_base(self):
        """从任意 a_base 减仓"""
        adapter = PolicyAdapter()
        for a_base in [0, 1, 2]:
            assert adapter.compute_final_action(a_base=a_base, a_base_prev=a_base, a_ref=-1, has_adjusted=False) == (0, True)

    def test_increase_from_any_base(self):
        """从任意 a_base 加仓"""
        adapter = PolicyAdapter()
        for a_base in [0, 1, 2]:
            assert adapter.compute_final_action(a_base=a_base, a_base_prev=a_base, a_ref=1, has_adjusted=False) == (2, True)

    def test_base_changed_overrides_ref(self):
        """a_base 变化时，即使 a_ref 非零也不调整"""
        adapter = PolicyAdapter()
        result = adapter.compute_final_action(a_base=0, a_base_prev=2, a_ref=-1, has_adjusted=False)
        assert result == (0, False)  # a_base passthrough, not counted as adjustment


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
        has_adjusted = False

        for i in range(length):
            a_base = a_base_seq[i]
            # Use previous a_base as a_base_prev (first step uses same value)
            a_base_prev = a_base_seq[i - 1] if i > 0 else a_base
            a_ref = a_ref_seq[i]

            a_final, has_adjusted = adapter.compute_final_action(a_base, a_base_prev, a_ref, has_adjusted)

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
        a_final, _ = adapter.compute_final_action(a_base, a_base_prev, a_ref, has_adjusted=False)

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
