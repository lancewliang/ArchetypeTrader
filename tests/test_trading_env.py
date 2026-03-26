"""MDP 交易环境单元测试

测试 TradingEnv 的核心功能：reset、step、奖励计算、执行损失、持仓管理。
"""

import numpy as np
import polars as pl
import pytest

from src.data.feature_pipeline import SINGLE_FEATURES, TREND_FEATURES
from src.env.trading_env import TradingEnv


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_env(
    T: int = 144,
    state_dim: int = 45,
    pair: str = "BTC",
    horizon: int = 72,
    price_start: float = 50000.0,
    price_step: float = 10.0,
) -> TradingEnv:
    """创建一个简单的测试环境。"""
    rng = np.random.RandomState(42)
    states = rng.randn(T, state_dim).astype(np.float64)
    prices = np.arange(price_start, price_start + T * price_step, price_step)[:T]
    feature_cols = SINGLE_FEATURES + TREND_FEATURES
    states_df = pl.DataFrame(states, schema=feature_cols[:state_dim])
    return TradingEnv(states=states, prices=prices, pair=pair, horizon=horizon, states_dataframe=states_df)


@pytest.fixture
def env_btc():
    return _make_env(pair="BTC")


@pytest.fixture
def env_eth():
    return _make_env(pair="ETH")


# ---------------------------------------------------------------------------
# 构造函数验证
# ---------------------------------------------------------------------------

class TestInit:
    def test_valid_pairs(self):
        for pair in ["BTC", "ETH", "DOT", "BNB"]:
            env = _make_env(pair=pair)
            assert env.pair == pair
            assert env.m == TradingEnv.MAX_POSITIONS[pair]

    def test_invalid_pair_raises(self):
        states = np.zeros((144, 45))
        prices = np.zeros(144)
        with pytest.raises(ValueError, match="不支持的交易对"):
            TradingEnv(states, prices, pair="XRP")

    def test_states_prices_length_mismatch(self):
        states = np.zeros((100, 45))
        prices = np.zeros(50)
        with pytest.raises(ValueError, match="不一致"):
            TradingEnv(states, prices, pair="BTC")

    def test_states_wrong_ndim(self):
        states = np.zeros((10,))
        prices = np.zeros(10)
        with pytest.raises(ValueError, match="2 维"):
            TradingEnv(states, prices, pair="BTC")

    def test_num_horizons(self, env_btc):
        assert env_btc.num_horizons == 2  # 144 // 72


# ---------------------------------------------------------------------------
# reset 测试
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_returns_initial_state(self, env_btc):
        state = env_btc.reset(0)
        np.testing.assert_array_equal(state, env_btc.states[0])

    def test_reset_second_horizon(self, env_btc):
        state = env_btc.reset(1)
        np.testing.assert_array_equal(state, env_btc.states[72])

    def test_reset_clears_position(self, env_btc):
        env_btc.reset(0)
        env_btc.step(2)  # go long
        assert env_btc._position != 0
        env_btc.reset(0)
        assert env_btc._position == 0

    def test_reset_invalid_horizon_raises(self, env_btc):
        with pytest.raises(IndexError, match="越界"):
            env_btc.reset(2)
        with pytest.raises(IndexError, match="越界"):
            env_btc.reset(-1)


# ---------------------------------------------------------------------------
# step 测试 — 无效动作
# ---------------------------------------------------------------------------

class TestStepInvalidAction:
    def test_invalid_action_raises(self, env_btc):
        env_btc.reset(0)
        with pytest.raises(ValueError, match="无效动作"):
            env_btc.step(3)
        with pytest.raises(ValueError, match="无效动作"):
            env_btc.step(-1)

    def test_step_after_done_raises(self, env_btc):
        env_btc.reset(0)
        for _ in range(72):
            env_btc.step(1)
        with pytest.raises(RuntimeError, match="已结束"):
            env_btc.step(1)


# ---------------------------------------------------------------------------
# 持仓状态测试
# ---------------------------------------------------------------------------

class TestPosition:
    def test_flat_action_keeps_zero(self, env_btc):
        env_btc.reset(0)
        env_btc.step(1)  # flat
        assert env_btc._position == 0

    def test_long_action_sets_positive(self, env_btc):
        env_btc.reset(0)
        env_btc.step(2)  # long
        assert env_btc._position == 8  # BTC m=8

    def test_short_action_sets_negative(self, env_btc):
        env_btc.reset(0)
        env_btc.step(0)  # short
        assert env_btc._position == -8

    def test_position_always_in_valid_set(self, env_btc):
        """持仓始终属于 {-m, 0, m}"""
        env_btc.reset(0)
        m = env_btc.m
        valid = {-m, 0, m}
        actions = [2, 2, 0, 1, 0, 2, 1, 1, 0]
        for a in actions:
            env_btc.step(a)
            assert env_btc._position in valid

    def test_eth_position_values(self, env_eth):
        env_eth.reset(0)
        env_eth.step(2)
        assert env_eth._position == 100  # ETH m=100


# ---------------------------------------------------------------------------
# 奖励计算测试
# ---------------------------------------------------------------------------

class TestReward:
    def test_flat_position_zero_reward_no_cost(self):
        """flat 持仓 + flat 动作 → 奖励 ≈ 0（无持仓收益，无执行损失）"""
        T = 144
        states = np.zeros((T, 45))
        prices = np.ones(T) * 100.0
        env = TradingEnv(states, prices, pair="BTC")
        env.reset(0)
        _, reward, _, _ = env.step(1)  # flat
        assert reward == 0.0

    def test_long_position_price_increase(self):
        """long 持仓 + 价格上涨 → 正奖励（扣除执行损失）"""
        T = 144
        states = np.zeros((T, 45))
        prices = np.zeros(T)
        prices[0] = 100.0
        prices[1] = 110.0
        for i in range(2, T):
            prices[i] = 110.0
        env = TradingEnv(states, prices, pair="BTC")
        env.reset(0)
        # 第一步：从 flat(0) → long(8)，价差 = 110 - 100 = 10
        # 奖励 = 8 * 10 - commission
        # commission = 0.0002 * 8 * 100 = 0.16
        _, reward, _, info = env.step(2)
        expected_reward = 8 * (110.0 - 100.0) - 0.0002 * 8 * 100.0
        assert abs(reward - expected_reward) < 1e-10

    def test_short_position_price_decrease(self):
        """short 持仓 + 价格下跌 → 正奖励"""
        T = 144
        states = np.zeros((T, 45))
        prices = np.zeros(T)
        prices[0] = 100.0
        prices[1] = 90.0
        for i in range(2, T):
            prices[i] = 90.0
        env = TradingEnv(states, prices, pair="BTC")
        env.reset(0)
        _, reward, _, _ = env.step(0)  # short
        # P_t = -8, price_diff = 90 - 100 = -10
        # reward = -8 * (-10) - commission = 80 - 0.0002 * 8 * 100
        expected = -8 * (90.0 - 100.0) - 0.0002 * 8 * 100.0
        assert abs(reward - expected) < 1e-10

    def test_reward_formula_eq1(self):
        """直接验证 Eq. 1: r = P_t × (p_{t+1} - p_t) - O_t"""
        T = 144
        states = np.zeros((T, 45))
        prices = np.linspace(100, 200, T)
        env = TradingEnv(states, prices, pair="ETH")
        env.reset(0)

        # 先进入 long 持仓
        _, r1, _, info1 = env.step(2)
        p0, p1 = prices[0], prices[1]
        m = 100
        commission = 0.0002 * m * p0  # 从 0 → 100
        expected_r1 = m * (p1 - p0) - commission
        assert abs(r1 - expected_r1) < 1e-8

        # 保持 long，无执行损失
        _, r2, _, info2 = env.step(2)
        p2 = prices[2]
        expected_r2 = m * (p2 - p1) - 0.0  # 持仓不变，无佣金
        assert abs(r2 - expected_r2) < 1e-8


# ---------------------------------------------------------------------------
# 执行损失测试
# ---------------------------------------------------------------------------

class TestExecutionCost:
    def test_no_position_change_zero_cost(self, env_btc):
        """持仓不变 → 执行损失为 0"""
        cost = env_btc.compute_execution_cost(1, 0, 50000.0)  # flat → flat
        assert cost == 0.0

    def test_position_change_has_commission(self, env_btc):
        """持仓变化 → 执行损失包含佣金"""
        # flat(0) → long(8): delta = 8
        cost = env_btc.compute_execution_cost(2, 0, 50000.0)
        expected = 0.0002 * 8 * 50000.0  # = 80.0
        assert abs(cost - expected) < 1e-10

    def test_short_to_long_cost(self, env_btc):
        """short(-8) → long(8): delta = 16"""
        cost = env_btc.compute_execution_cost(2, -8, 50000.0)
        expected = 0.0002 * 16 * 50000.0  # = 160.0
        assert abs(cost - expected) < 1e-10

    def test_execution_cost_non_negative(self, env_btc):
        """执行损失始终 ≥ 0"""
        for action in [0, 1, 2]:
            for pos in [-8, 0, 8]:
                cost = env_btc.compute_execution_cost(action, pos, 50000.0)
                assert cost >= 0.0

    def test_fill_cost_non_negative(self, env_btc):
        """fill cost (slippage) 始终 ≥ 0"""
        state = env_btc.states_dataframe.row(0, named=True)
        price = 50000.0
        for delta_pos in [-16, -8, 0, 8, 16]:
            fc = env_btc.compute_fill_cost(delta_pos, state, price)
            assert fc >= 0.0


# ---------------------------------------------------------------------------
# Episode 长度测试
# ---------------------------------------------------------------------------

class TestEpisodeLength:
    def test_episode_terminates_at_horizon(self, env_btc):
        """episode 在 h=72 步后终止"""
        env_btc.reset(0)
        for i in range(71):
            _, _, done, _ = env_btc.step(1)
            assert not done, f"Episode 在第 {i+1} 步提前终止"
        _, _, done, _ = env_btc.step(1)
        assert done, "Episode 应在第 72 步终止"

    def test_episode_step_count(self, env_btc):
        """验证 step_in_horizon 计数"""
        env_btc.reset(0)
        for i in range(72):
            _, _, _, info = env_btc.step(1)
            assert info["step_in_horizon"] == i + 1

    def test_custom_horizon(self):
        """自定义 horizon 长度"""
        env = _make_env(T=20, horizon=10)
        env.reset(0)
        for i in range(9):
            _, _, done, _ = env.step(1)
            assert not done
        _, _, done, _ = env.step(1)
        assert done


# ---------------------------------------------------------------------------
# Property-Based Tests (hypothesis)
# ---------------------------------------------------------------------------

from hypothesis import given, settings, assume
from hypothesis import strategies as st


def _make_prop_env(
    pair: str,
    horizon: int,
    prices: np.ndarray | None = None,
    T: int | None = None,
    state_dim: int = 45,
) -> TradingEnv:
    """Helper to create a TradingEnv for property tests.

    If *prices* is provided, *T* is inferred from its length.
    Otherwise a linearly-spaced price array of length *T* is generated.
    """
    if prices is not None:
        T_actual = len(prices)
    else:
        assert T is not None
        T_actual = T
        prices = np.linspace(100.0, 200.0, T_actual)
    rng = np.random.RandomState(0)
    states = rng.randn(T_actual, state_dim).astype(np.float64)
    feature_cols = SINGLE_FEATURES + TREND_FEATURES
    states_df = pl.DataFrame(states, schema=feature_cols)
    return TradingEnv(states=states, prices=prices, pair=pair, horizon=horizon, states_dataframe=states_df)


# Feature: archetype-trader, Property 5: 持仓状态不变量
class TestPropPositionInvariant:
    """P_t 始终属于 {-m, 0, m}。

    **Validates: Requirements 2.3**
    """

    @given(
        pair=st.sampled_from(["BTC", "ETH", "DOT", "BNB"]),
        actions=st.lists(st.integers(0, 2), min_size=1, max_size=72),
    )
    @settings(max_examples=100)
    def test_prop_position_always_valid(self, pair: str, actions: list[int]):
        horizon = len(actions)
        # Need at least horizon+1 price points for reward calculation
        T = horizon + 1
        env = _make_prop_env(pair=pair, horizon=horizon, T=T)
        m = env.m
        valid_positions = {-m, 0, m}

        env.reset(0)
        for a in actions:
            _, _, done, info = env.step(a)
            assert info["position"] in valid_positions, (
                f"Position {info['position']} not in {valid_positions} "
                f"for pair={pair}, action={a}"
            )
            if done:
                break


# Feature: archetype-trader, Property 6: 奖励计算公式正确性
class TestPropRewardFormula:
    """r = P_t × (p_{t+1} - p_t) - O_t

    **Validates: Requirements 2.4**
    """

    @given(
        pair=st.sampled_from(["BTC", "ETH", "DOT", "BNB"]),
        actions=st.lists(st.integers(0, 2), min_size=1, max_size=72),
        price_base=st.floats(min_value=10.0, max_value=100000.0, allow_nan=False, allow_infinity=False),
        price_step=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_prop_reward_matches_formula(
        self,
        pair: str,
        actions: list[int],
        price_base: float,
        price_step: float,
    ):
        horizon = len(actions)
        T = horizon + 1
        prices = np.array([price_base + i * price_step for i in range(T)])
        # Ensure all prices are positive (required for meaningful commission calc)
        assume(np.all(prices > 0))

        env = _make_prop_env(pair=pair, horizon=horizon, prices=prices)
        env.reset(0)

        for step_idx, a in enumerate(actions):
            old_position = env._position
            _, reward, done, info = env.step(a)

            # Manually compute expected reward: r = P_t × (p_{t+1} - p_t) - O_t
            new_position = info["position"]
            p_t = prices[step_idx]
            p_t1 = prices[step_idx + 1] if step_idx + 1 < len(prices) else p_t
            execution_cost = info["execution_cost"]
            expected_reward = new_position * (p_t1 - p_t) - execution_cost

            assert abs(reward - expected_reward) < 1e-6, (
                f"Step {step_idx}: reward={reward}, expected={expected_reward}, "
                f"pos={new_position}, p_t={p_t}, p_t1={p_t1}, cost={execution_cost}"
            )
            if done:
                break


# Feature: archetype-trader, Property 7: 执行损失非负性
class TestPropExecutionCostNonNegative:
    """O_t ≥ 0 for all steps.

    **Validates: Requirements 2.5, 2.6**
    """

    @given(
        pair=st.sampled_from(["BTC", "ETH", "DOT", "BNB"]),
        actions=st.lists(st.integers(0, 2), min_size=1, max_size=72),
    )
    @settings(max_examples=100)
    def test_prop_execution_cost_non_negative(self, pair: str, actions: list[int]):
        horizon = len(actions)
        T = horizon + 1
        env = _make_prop_env(pair=pair, horizon=horizon, T=T)
        env.reset(0)

        for a in actions:
            _, _, done, info = env.step(a)
            assert info["execution_cost"] >= 0.0, (
                f"Execution cost {info['execution_cost']} < 0 "
                f"for pair={pair}, action={a}"
            )
            if done:
                break


# Feature: archetype-trader, Property 8: Episode 长度不变量
class TestPropEpisodeLength:
    """Episode terminates at exactly h steps.

    **Validates: Requirements 2.7**
    """

    @given(
        horizon=st.integers(min_value=5, max_value=100),
    )
    @settings(max_examples=100)
    def test_prop_episode_length_equals_horizon(self, horizon: int):
        T = horizon + 1  # need at least horizon+1 prices
        env = _make_prop_env(pair="BTC", horizon=horizon, T=T)
        env.reset(0)

        step_count = 0
        done = False
        while not done:
            _, _, done, _ = env.step(1)  # always flat — simplest action
            step_count += 1

        assert step_count == horizon, (
            f"Episode lasted {step_count} steps, expected {horizon}"
        )
