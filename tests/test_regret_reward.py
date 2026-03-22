"""Regret-aware Reward 计算模块单元测试"""

import numpy as np
import pytest

from src.phase3.regret_reward import (
    compute_regret_reward,
    compute_top5_hindsight_optimal,
    _simulate_adaptation,
)
from src.env.trading_env import TradingEnv


class TestComputeRegretReward:
    """compute_regret_reward 函数测试。"""

    def test_zero_reward_when_no_adjustment(self):
        """a_ref=0 时奖励应为 0。"""
        result = compute_regret_reward(R=10.0, R_base=5.0, R_1_opt=8.0, a_ref=0)
        assert result == 0.0

    def test_zero_reward_when_no_adjustment_any_beta(self):
        """a_ref=0 时，无论 beta1 取何值，奖励都应为 0。"""
        for beta1 in [0.3, 0.5, 0.7]:
            result = compute_regret_reward(
                R=100.0, R_base=50.0, R_1_opt=80.0, a_ref=0, beta1=beta1
            )
            assert result == 0.0

    def test_positive_adjustment_formula(self):
        """a_ref=1 时验证公式 r_ref = (R - R_base) + β_1 × (R - R_1_opt)。"""
        R, R_base, R_1_opt, beta1 = 10.0, 5.0, 8.0, 0.5
        expected = (10.0 - 5.0) + 0.5 * (10.0 - 8.0)  # 5.0 + 1.0 = 6.0
        result = compute_regret_reward(R, R_base, R_1_opt, a_ref=1, beta1=beta1)
        assert result == pytest.approx(expected)

    def test_negative_adjustment_formula(self):
        """a_ref=-1 时验证公式同样适用。"""
        R, R_base, R_1_opt, beta1 = 3.0, 5.0, 8.0, 0.5
        expected = (3.0 - 5.0) + 0.5 * (3.0 - 8.0)  # -2.0 + (-2.5) = -4.5
        result = compute_regret_reward(R, R_base, R_1_opt, a_ref=-1, beta1=beta1)
        assert result == pytest.approx(expected)

    def test_default_beta1(self):
        """默认 beta1=0.5。"""
        result = compute_regret_reward(R=10.0, R_base=5.0, R_1_opt=8.0, a_ref=1)
        expected = (10.0 - 5.0) + 0.5 * (10.0 - 8.0)
        assert result == pytest.approx(expected)

    def test_beta1_values(self):
        """支持 β_1 ∈ {0.3, 0.5, 0.7} 调优。"""
        R, R_base, R_1_opt = 10.0, 5.0, 8.0
        for beta1 in [0.3, 0.5, 0.7]:
            result = compute_regret_reward(R, R_base, R_1_opt, a_ref=1, beta1=beta1)
            expected = (R - R_base) + beta1 * (R - R_1_opt)
            assert result == pytest.approx(expected)

    def test_perfect_performance(self):
        """R == R_1_opt 时 regret 项为 0。"""
        R = R_1_opt = 10.0
        R_base = 5.0
        result = compute_regret_reward(R, R_base, R_1_opt, a_ref=1, beta1=0.5)
        expected = (10.0 - 5.0) + 0.5 * 0.0  # 5.0
        assert result == pytest.approx(expected)

    def test_worse_than_base(self):
        """R < R_base 时 improvement 为负。"""
        result = compute_regret_reward(R=3.0, R_base=5.0, R_1_opt=8.0, a_ref=1, beta1=0.5)
        assert result < 0.0

    def test_all_equal_returns(self):
        """R == R_base == R_1_opt 时奖励应为 0。"""
        result = compute_regret_reward(R=5.0, R_base=5.0, R_1_opt=5.0, a_ref=1, beta1=0.5)
        assert result == pytest.approx(0.0)

    def test_negative_returns(self):
        """所有收益为负值时公式仍正确。"""
        R, R_base, R_1_opt, beta1 = -2.0, -5.0, -1.0, 0.3
        expected = (-2.0 - (-5.0)) + 0.3 * (-2.0 - (-1.0))  # 3.0 + 0.3*(-1.0) = 2.7
        result = compute_regret_reward(R, R_base, R_1_opt, a_ref=-1, beta1=beta1)
        assert result == pytest.approx(expected)

    def test_large_regret_penalty(self):
        """R 远低于 R_1_opt 时 regret 项应为大负值。"""
        R, R_base, R_1_opt, beta1 = 1.0, 0.0, 100.0, 0.7
        expected = (1.0 - 0.0) + 0.7 * (1.0 - 100.0)  # 1.0 + 0.7*(-99) = 1.0 - 69.3 = -68.3
        result = compute_regret_reward(R, R_base, R_1_opt, a_ref=1, beta1=beta1)
        assert result == pytest.approx(expected)

    def test_beta1_zero_no_regret(self):
        """beta1=0 时 regret 项消失，只剩 improvement。"""
        R, R_base, R_1_opt = 10.0, 5.0, 100.0
        result = compute_regret_reward(R, R_base, R_1_opt, a_ref=1, beta1=0.0)
        expected = 10.0 - 5.0  # 5.0, no regret term
        assert result == pytest.approx(expected)

    def test_symmetry_a_ref_positive_negative(self):
        """a_ref=1 和 a_ref=-1 使用相同公式，结果应相同。"""
        R, R_base, R_1_opt, beta1 = 7.0, 4.0, 9.0, 0.5
        r_pos = compute_regret_reward(R, R_base, R_1_opt, a_ref=1, beta1=beta1)
        r_neg = compute_regret_reward(R, R_base, R_1_opt, a_ref=-1, beta1=beta1)
        assert r_pos == pytest.approx(r_neg)

    def test_concrete_example_paper_scenario(self):
        """具体数值示例：模拟论文场景。

        假设 horizon 收益:
        - R = 12.5 (精炼后)
        - R_base = 8.0 (原始原型)
        - R_1_opt = 15.0 (最优 hindsight)
        - beta1 = 0.5
        r_ref = (12.5 - 8.0) + 0.5 * (12.5 - 15.0) = 4.5 + (-1.25) = 3.25
        """
        result = compute_regret_reward(R=12.5, R_base=8.0, R_1_opt=15.0, a_ref=1, beta1=0.5)
        assert result == pytest.approx(3.25)

    def test_zero_base_and_opt(self):
        """R_base=0, R_1_opt=0 时奖励等于 R*(1+beta1)。"""
        R, beta1 = 6.0, 0.5
        expected = R + beta1 * R  # 6.0 + 3.0 = 9.0
        result = compute_regret_reward(R=R, R_base=0.0, R_1_opt=0.0, a_ref=1, beta1=beta1)
        assert result == pytest.approx(expected)


class TestSimulateAdaptation:
    """_simulate_adaptation 内部函数测试。"""

    def test_flat_prices_no_reward(self):
        """价格不变时，持仓收益为 0（仅有佣金损失）。"""
        prices = np.array([100.0, 100.0, 100.0, 100.0])
        base_actions = np.array([1, 1, 1, 1])  # 全 flat
        result = _simulate_adaptation(
            prices, base_actions, adapt_step=1, a_ref=1, m=8, commission_rate=0.0002
        )
        # adapt_step=1 时 action=2 (long), position=8
        # 佣金 = 0.0002 * 8 * 100 = 0.16
        # 后续步 flat 价格不变，收益 = 0 - 佣金
        assert result < 0.0  # 有佣金损失

    def test_rising_prices_long_positive(self):
        """价格上涨时做多应有正收益。"""
        prices = np.array([100.0, 101.0, 102.0, 103.0])
        base_actions = np.array([1, 1, 1, 1])  # 全 flat
        result = _simulate_adaptation(
            prices, base_actions, adapt_step=0, a_ref=1, m=8, commission_rate=0.0
        )
        # step 0: action=2 (long), position=8, reward = 8*(101-100) = 8
        # step 1: action=1 (flat), position=0, cost=0.0002*8*101, reward = 0*(102-101) - cost
        # 简化: commission=0 时
        # step 0: 8*1 = 8, step 1: 0*1 - 0 = 0, step 2: 0*1 - 0 = 0, step 3: 0*0 = 0
        # total = 8
        assert result > 0.0


class TestComputeTop5HindsightOptimal:
    """compute_top5_hindsight_optimal 函数测试。"""

    def _make_env(self, h: int = 10) -> TradingEnv:
        """创建简单的测试环境。"""
        states = np.random.randn(h, 45)
        prices = np.linspace(100.0, 110.0, h)  # 线性上涨
        return TradingEnv(states=states, prices=prices, pair="BTC", horizon=h)

    def test_returns_at_most_5(self):
        """返回结果最多 5 个。"""
        h = 10
        env = self._make_env(h)
        prices = np.linspace(100.0, 110.0, h)
        base_actions = np.ones(h, dtype=int)  # 全 flat

        result = compute_top5_hindsight_optimal(prices, base_actions, step_idx=0, env=env)
        assert len(result) <= 5

    def test_sorted_descending(self):
        """返回结果按收益降序排列。"""
        h = 10
        env = self._make_env(h)
        prices = np.linspace(100.0, 110.0, h)
        base_actions = np.ones(h, dtype=int)

        result = compute_top5_hindsight_optimal(prices, base_actions, step_idx=0, env=env)
        returns = [r for _, _, r in result]
        assert returns == sorted(returns, reverse=True)

    def test_result_tuple_structure(self):
        """每个结果是 (τ_opt, adaptation_action, resulting_return) 三元组。"""
        h = 6
        env = self._make_env(h)
        prices = np.linspace(100.0, 106.0, h)
        base_actions = np.ones(h, dtype=int)

        result = compute_top5_hindsight_optimal(prices, base_actions, step_idx=0, env=env)
        for tau_opt, action, ret in result:
            assert isinstance(tau_opt, int), f"τ_opt 应为 int，得到 {type(tau_opt)}"
            assert 0 <= tau_opt < h, f"τ_opt 应在 [0, {h}) 范围内，得到 {tau_opt}"
            assert action in (-1, 1), f"调整动作应为 -1 或 1，得到 {action}"
            assert isinstance(ret, float)

    def test_step_idx_limits_candidates(self):
        """step_idx 越大，候选调整点越少。"""
        h = 10
        env = self._make_env(h)
        prices = np.linspace(100.0, 110.0, h)
        base_actions = np.ones(h, dtype=int)

        result_early = compute_top5_hindsight_optimal(prices, base_actions, step_idx=0, env=env)
        result_late = compute_top5_hindsight_optimal(prices, base_actions, step_idx=8, env=env)

        # step_idx=8 时只有 2 个调整点 (step 8, 9) × 2 动作 = 4 候选
        assert len(result_late) <= 4

    def test_rising_prices_prefers_long(self):
        """价格持续上涨时，top-1 应倾向于做多（a_ref=1）。"""
        h = 10
        env = self._make_env(h)
        prices = np.linspace(100.0, 120.0, h)  # 强上涨
        base_actions = np.ones(h, dtype=int)  # 全 flat

        result = compute_top5_hindsight_optimal(prices, base_actions, step_idx=0, env=env)
        # top-1 应该是在最早的步做多
        tau_opt, top_action, top_return = result[0]
        assert top_action == 1  # long
        assert top_return > 0.0

    def test_export_from_package(self):
        """测试从 phase3 包导入新函数。"""
        from src.phase3 import compute_regret_reward, compute_top5_hindsight_optimal
        assert callable(compute_regret_reward)
        assert callable(compute_top5_hindsight_optimal)


# ---------------------------------------------------------------------------
# Property-Based Tests (hypothesis)
# ---------------------------------------------------------------------------
from hypothesis import given, settings, assume
from hypothesis import strategies as st


class TestRegretRewardProperties:
    """Regret-aware Reward 属性测试。"""

    @given(
        R=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        R_base=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        R_1_opt=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        a_ref=st.sampled_from([-1, 0, 1]),
        beta1=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_prop20_regret_reward_correctness(
        self, R, R_base, R_1_opt, a_ref, beta1
    ):
        """Property 20: Regret-aware Reward 计算正确性。

        **Validates: Requirements 6.4**

        # Feature: archetype-trader, Property 20: Regret-aware Reward 计算正确性

        验证:
        - a_ref ≠ 0 时 r = (R - R_base) + β_1 × (R - R_1_opt)
        - a_ref = 0 时 r = 0
        """
        result = compute_regret_reward(R, R_base, R_1_opt, a_ref, beta1)

        if a_ref == 0:
            assert result == 0.0, (
                f"When a_ref=0, reward should be 0.0, got {result}"
            )
        else:
            expected = (R - R_base) + beta1 * (R - R_1_opt)
            assert result == pytest.approx(expected, rel=1e-6, abs=1e-9), (
                f"Expected {expected}, got {result} "
                f"(R={R}, R_base={R_base}, R_1_opt={R_1_opt}, "
                f"a_ref={a_ref}, beta1={beta1})"
            )

    @given(
        price_length=st.integers(min_value=5, max_value=20),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_prop21_top5_hindsight_sorted_descending(self, price_length, data):
        """Property 21: Top-5 Hindsight 排序 — 返回结果按收益降序排列。

        **Validates: Requirements 6.5**

        # Feature: archetype-trader, Property 21: Top-5 Hindsight 排序
        """
        # Generate random prices (positive, reasonable range)
        prices = np.array(
            data.draw(
                st.lists(
                    st.floats(min_value=50.0, max_value=200.0, allow_nan=False, allow_infinity=False),
                    min_size=price_length,
                    max_size=price_length,
                )
            )
        )

        # Generate random base_actions
        base_actions = np.array(
            data.draw(
                st.lists(
                    st.sampled_from([0, 1, 2]),
                    min_size=price_length,
                    max_size=price_length,
                )
            )
        )

        # Create a minimal TradingEnv
        states = np.random.randn(price_length, 45)
        env = TradingEnv(
            states=states, prices=prices, pair="BTC", horizon=price_length
        )

        step_idx = data.draw(st.integers(min_value=0, max_value=price_length - 1))

        result = compute_top5_hindsight_optimal(
            prices, base_actions, step_idx, env
        )

        # Verify sorted descending by return
        returns = [r for _, _, r in result]
        for i in range(len(returns) - 1):
            assert returns[i] >= returns[i + 1], (
                f"Results not sorted descending: {returns}"
            )

        # Verify at most 5 results
        assert len(result) <= 5, f"Expected at most 5 results, got {len(result)}"
