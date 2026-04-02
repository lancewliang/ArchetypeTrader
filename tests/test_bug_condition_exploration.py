"""Bug Condition Exploration Test — Stateful Side Effects Violate Pure Function Semantics.

Property 1: Bug Condition — For any valid input tuple
(a_base ∈ {0,1,2}, a_base_prev ∈ {0,1,2}, a_ref ∈ {-1,0,1}, has_adjusted ∈ {True, False}),
calling compute_final_action twice on the SAME PolicyAdapter instance with identical arguments
MUST return identical results. On unfixed code, cases where a_ref ∈ {-1, 1} and
a_base == a_base_prev will fail because the first call sets self.adjusted_in_horizon = True,
causing the second call to return a_base instead of the adjusted action.

**Validates: Requirements 1.5, 2.5, 2.6**

EXPECTED OUTCOME on unfixed code: FAIL (proves the bug exists)
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from src.phase3.policy_adapter import PolicyAdapter


class TestBugConditionExploration:
    """Bug condition exploration: same inputs → different outputs due to hidden mutable state."""

    @given(
        a_base=st.sampled_from([0, 1, 2]),
        a_base_prev=st.sampled_from([0, 1, 2]),
        a_ref=st.sampled_from([-1, 0, 1]),
    )
    @settings(max_examples=200)
    def test_pure_function_referential_transparency(
        self, a_base: int, a_base_prev: int, a_ref: int
    ):
        """Property 1: Bug Condition — Stateful Side Effects Violate Pure Function Semantics.

        **Validates: Requirements 1.5, 2.5, 2.6**

        A pure function must return the same output for the same input.
        On unfixed code, compute_final_action mutates self.adjusted_in_horizon,
        so calling it twice with the same args on the same instance yields
        different results when a_ref ∈ {-1, 1} and a_base == a_base_prev.
        """
        adapter = PolicyAdapter()

        # First call — has_adjusted=False (testing pure function with same inputs)
        result1 = adapter.compute_final_action(a_base, a_base_prev, a_ref, has_adjusted=False)
        # Second call — identical arguments, same instance
        result2 = adapter.compute_final_action(a_base, a_base_prev, a_ref, has_adjusted=False)

        assert result1 == result2, (
            f"Pure function violation: compute_final_action({a_base}, {a_base_prev}, {a_ref}, False) "
            f"returned {result1} on first call, then {result2} on second call. "
            f"Function is not referentially transparent."
        )

    @given(
        a_base=st.sampled_from([0, 1, 2]),
        a_base_prev=st.sampled_from([0, 1, 2]),
        a_ref=st.sampled_from([-1, 0, 1]),
    )
    @settings(max_examples=200)
    def test_cross_horizon_state_leakage(
        self, a_base: int, a_base_prev: int, a_ref: int
    ):
        """Cross-horizon state leakage: after an adjustment in horizon 1,
        a new horizon's adjustment signals are ignored without reset().

        **Validates: Requirements 1.5, 2.5, 2.6**

        With the pure function design, state leakage is impossible since
        has_adjusted is passed explicitly. Both calls use has_adjusted=False
        (simulating fresh horizon start), so results must be identical
        regardless of previous calls on the same instance.
        """
        adapter = PolicyAdapter()

        # --- Horizon 1: force an adjustment ---
        adapter.compute_final_action(a_base=1, a_base_prev=1, a_ref=-1, has_adjusted=False)

        # --- Horizon 2: NO reset() needed (pure function, no state to leak) ---
        # A fresh horizon passes has_adjusted=False explicitly.
        # Compute what a fresh adapter would return for the same inputs:
        fresh_adapter = PolicyAdapter()
        expected = fresh_adapter.compute_final_action(a_base, a_base_prev, a_ref, has_adjusted=False)

        # The stale adapter should produce the same result since state is explicit:
        actual = adapter.compute_final_action(a_base, a_base_prev, a_ref, has_adjusted=False)

        assert actual == expected, (
            f"Cross-horizon state leakage: after horizon 1 adjustment, "
            f"compute_final_action({a_base}, {a_base_prev}, {a_ref}, False) "
            f"returned {actual} (stale instance) vs {expected} (fresh instance). "
            f"Pure function should not depend on instance history."
        )
