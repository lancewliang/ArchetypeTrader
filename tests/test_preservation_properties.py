"""Preservation Property Tests — Eq. 6 Action Mapping Consistency.

Property 2: Preservation — For any valid input tuple
(a_base ∈ {0,1,2}, a_base_prev ∈ {0,1,2}, a_ref ∈ {-1,0,1}),
a fresh PolicyAdapter() instance (equivalent to has_adjusted=False)
MUST produce the correct Eq. 6 action mapping.

These tests capture the correct baseline behavior on UNFIXED code
so we can verify it's preserved after the fix.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6**

EXPECTED OUTCOME on unfixed code: PASS (confirms baseline Eq. 6 behavior)
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from src.phase3.policy_adapter import PolicyAdapter


class TestEq6MappingProperty:
    """Property-based test: Eq. 6 action mapping on fresh instances."""

    @given(
        a_base=st.sampled_from([0, 1, 2]),
        a_base_prev=st.sampled_from([0, 1, 2]),
        a_ref=st.sampled_from([-1, 0, 1]),
    )
    @settings(max_examples=200)
    def test_eq6_action_mapping(
        self, a_base: int, a_base_prev: int, a_ref: int
    ):
        """Property 2: Preservation — Eq. 6 Action Mapping Consistency.

        **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6**

        For a fresh PolicyAdapter (has_adjusted=False equivalent), verify
        the complete Eq. 6 action mapping:
        - a_base != a_base_prev → result == a_base
        - a_ref == 0 → result == a_base
        - a_ref == -1 and a_base == a_base_prev → result == 0
        - a_ref == 1 and a_base == a_base_prev → result == 2
        """
        adapter = PolicyAdapter()
        a_final, new_has_adjusted = adapter.compute_final_action(a_base, a_base_prev, a_ref, has_adjusted=False)

        if a_base != a_base_prev:
            assert a_final == a_base, (
                f"Eq. 6 violation: when a_base({a_base}) != a_base_prev({a_base_prev}), "
                f"expected a_base={a_base}, got {a_final}"
            )
        elif a_ref == 0:
            assert a_final == a_base, (
                f"Eq. 6 violation: when a_ref=0, "
                f"expected a_base={a_base}, got {a_final}"
            )
        elif a_ref == -1:
            assert a_final == 0, (
                f"Eq. 6 violation: when a_ref=-1 and a_base==a_base_prev, "
                f"expected 0, got {a_final}"
            )
        elif a_ref == 1:
            assert a_final == 2, (
                f"Eq. 6 violation: when a_ref=1 and a_base==a_base_prev, "
                f"expected 2, got {a_final}"
            )


class TestSequenceProperty:
    """Property-based test: at most one adjustment per horizon in a sequence."""

    @given(
        a_ref_seq=st.lists(
            st.sampled_from([-1, 0, 1]), min_size=1, max_size=72
        ),
        a_base_seq=st.lists(
            st.sampled_from([0, 1, 2]), min_size=1, max_size=72
        ),
    )
    @settings(max_examples=200)
    def test_at_most_one_adjustment_per_horizon(
        self, a_ref_seq: list, a_base_seq: list
    ):
        """Property 2: Preservation — Sequence at-most-once adjustment.

        **Validates: Requirements 3.5, 3.6**

        For a random horizon action sequence run through a single
        PolicyAdapter instance, count adjustments where a_final != a_base
        due to refinement. Assert count <= 1.
        """
        length = min(len(a_ref_seq), len(a_base_seq))
        a_ref_seq = a_ref_seq[:length]
        a_base_seq = a_base_seq[:length]

        adapter = PolicyAdapter()
        adjustment_count = 0
        has_adjusted = False

        for i in range(length):
            a_base = a_base_seq[i]
            a_base_prev = a_base_seq[i - 1] if i > 0 else a_base
            a_ref = a_ref_seq[i]

            a_final, has_adjusted = adapter.compute_final_action(a_base, a_base_prev, a_ref, has_adjusted)

            # Count refinement-driven adjustments
            if a_base == a_base_prev and a_ref != 0 and a_final != a_base:
                adjustment_count += 1

        assert adjustment_count <= 1, (
            f"Expected at most 1 adjustment per horizon, got {adjustment_count}"
        )


class TestConcreteObservations:
    """Concrete observation tests on fresh instances to verify Eq. 6 baseline."""

    def test_reduce_signal_returns_zero(self):
        """compute_final_action(1, 1, -1, False) returns (0, True) (reduce to short/flat).

        **Validates: Requirements 3.1**
        """
        adapter = PolicyAdapter()
        a_final, new_has_adjusted = adapter.compute_final_action(1, 1, -1, has_adjusted=False)
        assert a_final == 0, f"Expected 0, got {a_final}"

    def test_increase_signal_returns_two(self):
        """compute_final_action(1, 1, 1, False) returns (2, True) (increase to long).

        **Validates: Requirements 3.2**
        """
        adapter = PolicyAdapter()
        a_final, new_has_adjusted = adapter.compute_final_action(1, 1, 1, has_adjusted=False)
        assert a_final == 2, f"Expected 2, got {a_final}"

    def test_no_adjustment_signal_returns_base(self):
        """compute_final_action(1, 1, 0, False) returns (1, False) (no adjustment).

        **Validates: Requirements 3.3**
        """
        adapter = PolicyAdapter()
        a_final, new_has_adjusted = adapter.compute_final_action(1, 1, 0, has_adjusted=False)
        assert a_final == 1, f"Expected 1, got {a_final}"

    def test_base_changed_passthrough(self):
        """compute_final_action(0, 2, -1, False) returns (0, False) (base changed, passthrough).

        **Validates: Requirements 3.4**
        """
        adapter = PolicyAdapter()
        a_final, new_has_adjusted = adapter.compute_final_action(0, 2, -1, has_adjusted=False)
        assert a_final == 0, f"Expected 0, got {a_final}"
