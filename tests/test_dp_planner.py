"""DP Planner 单元测试

测试 DPPlanner 的核心功能：
1. 全 flat 轨迹边界情况（价格恒定时无盈利交易）
2. 小规模轨迹生成
3. 单次交易约束：任意轨迹中持仓变化 ≤ 2
4. 轨迹结构：s_demo shape (h, state_dim), a_demo shape (h,) 值域 {0,1,2}, r_demo shape (h,)
5. DP 最优性（小规模实例）：DP 结果匹配暴力枚举
"""

import itertools

import numpy as np
import pytest

from src.env.trading_env import TradingEnv
from src.phase1.dp_planner import DPPlanner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env(
    prices: np.ndarray,
    state_dim: int = 45,
    pair: str = "BTC",
    horizon: int | None = None,
) -> TradingEnv:
    """根据给定价格序列创建测试环境。"""
    T = len(prices)
    if horizon is None:
        horizon = T
    states = np.random.RandomState(0).randn(T, state_dim).astype(np.float64)
    return TradingEnv(states=states, prices=prices, pair=pair, horizon=horizon)


def _count_position_changes(actions: np.ndarray, m: int) -> int:
    """统计动作序列中持仓状态变化的次数。

    从 flat 开始，每次持仓值发生变化计数 +1。
    """
    changes = 0
    prev_pos = 0  # 初始 flat
    for a in actions:
        pos = TradingEnv.POSITION_MAP[int(a)] * m
        if pos != prev_pos:
            changes += 1
        prev_pos = pos
    return changes


def _brute_force_max_reward(
    prices: np.ndarray,
    m: int,
    commission_rate: float = 0.0002,
    gamma: float = 0.99,
) -> float:
    """暴力枚举所有满足单次交易约束的动作序列，返回最大总奖励。

    单次交易约束：从 flat 开始，最多一次开仓 + 一次平仓。
    允许的路径模式：
      - 全 flat
      - flat...flat, open(long/short)...hold, flat...flat
      - flat...flat, open(long/short)...hold（不平仓，持有到结束）

    注意：使用折扣因子 γ 与 DP planner 保持一致。
    最后一步动作复制倒数第二步（Algorithm 1, Step 13）。
    """
    N = len(prices)
    if N == 0:
        return 0.0

    pos_map = TradingEnv.POSITION_MAP  # {0: -1, 1: 0, 2: 1}

    def _is_valid(actions):
        """检查动作序列是否满足单次交易约束。"""
        pos = 0  # flat
        traded = False  # 是否已开仓
        closed = False  # 是否已平仓
        for a in actions:
            new_pos = pos_map[a] * m
            if new_pos != pos:
                if not traded:
                    # 首次开仓
                    if pos != 0:
                        return False  # 不应该发生（初始为 flat）
                    traded = True
                elif closed:
                    # 已平仓后又变化 → 违反约束
                    return False
                else:
                    # 已开仓，检查是否平仓或切换方向
                    if new_pos == 0:
                        closed = True
                    elif new_pos != pos:
                        # 切换方向（short↔long）→ 违反约束
                        return False
            pos = new_pos
        return True

    def _compute_total_reward(actions):
        """计算动作序列的折扣总奖励（含 Algorithm 1 Step 13 最后一步复制规则）。"""
        # 应用 Algorithm 1 Step 13: â_{N-1} ← â_{N-2}
        actions = list(actions)
        if N >= 2:
            actions[N - 1] = actions[N - 2]

        total = 0.0
        current_pos = 0  # flat
        for t, a in enumerate(actions):
            new_pos = pos_map[a] * m
            # 执行损失
            delta = abs(new_pos - current_pos)
            cost = commission_rate * delta * prices[t] if delta > 0 else 0.0
            # 价差
            p_next = prices[t + 1] if t + 1 < len(prices) else prices[t]
            reward = new_pos * (p_next - prices[t]) - cost
            total += (gamma ** t) * reward
            current_pos = new_pos
        return total

    best = -np.inf
    # 只枚举前 N-1 步（最后一步由 Step 13 决定）
    if N >= 2:
        for actions in itertools.product([0, 1, 2], repeat=N):
            # 应用 Step 13 规则后检查有效性
            actions_list = list(actions)
            actions_list[N - 1] = actions_list[N - 2]
            if _is_valid(actions_list):
                r = _compute_total_reward(tuple(actions_list))
                if r > best:
                    best = r
    else:
        for actions in itertools.product([0, 1, 2], repeat=N):
            if _is_valid(actions):
                r = _compute_total_reward(actions)
                if r > best:
                    best = r

    return best


# ---------------------------------------------------------------------------
# 1. 全 flat 轨迹边界情况
# ---------------------------------------------------------------------------

class TestAllFlatTrajectory:
    """当价格恒定时，DP 应输出全 flat 轨迹（无盈利交易机会）。"""

    def test_constant_prices_all_flat(self):
        """恒定价格 → 任何交易都只有佣金损失，最优策略是全 flat。"""
        N = 10
        prices = np.ones(N) * 100.0
        env = _make_env(prices, horizon=N)
        planner = DPPlanner(env)

        s_demo, a_demo, r_demo = planner.plan(env.states, prices)

        # 全 flat：所有动作应为 1
        np.testing.assert_array_equal(a_demo, np.ones(N, dtype=np.int32))
        # 全 flat 时奖励应全为 0（无持仓收益，无执行损失）
        np.testing.assert_array_almost_equal(r_demo, np.zeros(N))

    def test_constant_prices_longer_horizon(self):
        """较长 horizon 下恒定价格仍应全 flat。"""
        N = 72
        prices = np.ones(N) * 50000.0
        env = _make_env(prices, horizon=N)
        planner = DPPlanner(env)

        _, a_demo, r_demo = planner.plan(env.states, prices)

        np.testing.assert_array_equal(a_demo, np.ones(N, dtype=np.int32))
        np.testing.assert_array_almost_equal(r_demo, np.zeros(N))

    def test_generate_trajectories_no_horizons(self):
        """环境中无可用 horizon 时，generate_trajectories 应返回全 flat 轨迹。"""
        # T < horizon → num_horizons = 0
        prices = np.ones(5) * 100.0
        env = _make_env(prices, horizon=72)
        assert env.num_horizons == 0

        planner = DPPlanner(env)
        result = planner.generate_trajectories(num_trajectories=3)

        assert result["actions"].shape == (3, 72)
        assert np.all(result["actions"] == 1)
        assert np.all(result["rewards"] == 0.0)


# ---------------------------------------------------------------------------
# 2. 小规模轨迹生成
# ---------------------------------------------------------------------------

class TestSmallScaleGeneration:
    """测试小规模轨迹生成功能。"""

    def test_generate_5_trajectories(self):
        """生成 5 条轨迹，验证输出结构。"""
        N = 10
        T = N * 3  # 3 个 horizon
        prices = np.linspace(100, 200, T)
        env = _make_env(prices, horizon=N)
        planner = DPPlanner(env)

        result = planner.generate_trajectories(num_trajectories=5)

        assert result["states"].shape == (5, N, 45)
        assert result["actions"].shape == (5, N)
        assert result["rewards"].shape == (5, N)

    def test_generate_fewer_than_horizons(self):
        """请求的轨迹数少于可用 horizon 数。"""
        N = 10
        T = N * 5  # 5 个 horizon
        prices = np.linspace(100, 200, T)
        env = _make_env(prices, horizon=N)
        planner = DPPlanner(env)

        result = planner.generate_trajectories(num_trajectories=3)

        assert result["states"].shape[0] == 3
        assert result["actions"].shape[0] == 3
        assert result["rewards"].shape[0] == 3

    def test_generate_more_than_horizons_cycles(self):
        """请求的轨迹数多于可用 horizon 数时，应循环复用。"""
        N = 10
        T = N * 2  # 2 个 horizon
        prices = np.linspace(100, 200, T)
        env = _make_env(prices, horizon=N)
        planner = DPPlanner(env)

        result = planner.generate_trajectories(num_trajectories=5)

        assert result["actions"].shape[0] == 5
        # 第 0 条和第 2 条应相同（循环复用 horizon 0）
        np.testing.assert_array_equal(
            result["actions"][0], result["actions"][2]
        )
        np.testing.assert_array_equal(
            result["actions"][1], result["actions"][3]
        )


# ---------------------------------------------------------------------------
# 3. 单次交易约束
# ---------------------------------------------------------------------------

class TestSingleTradeConstraint:
    """任意 DP 轨迹中持仓变化次数 ≤ 2（最多一次开仓 + 一次平仓）。"""

    def test_constraint_monotone_prices(self):
        """单调递增价格下，持仓变化 ≤ 2。"""
        N = 20
        prices = np.linspace(100, 200, N)
        env = _make_env(prices, horizon=N)
        planner = DPPlanner(env)

        _, a_demo, _ = planner.plan(env.states, prices)
        changes = _count_position_changes(a_demo, env.m)
        assert changes <= 2, f"持仓变化 {changes} 次，超过约束上限 2"

    def test_constraint_volatile_prices(self):
        """波动价格下，持仓变化 ≤ 2。"""
        N = 15
        rng = np.random.RandomState(42)
        prices = 100.0 + np.cumsum(rng.randn(N) * 5)
        prices = np.maximum(prices, 1.0)  # 确保价格为正
        env = _make_env(prices, horizon=N)
        planner = DPPlanner(env)

        _, a_demo, _ = planner.plan(env.states, prices)
        changes = _count_position_changes(a_demo, env.m)
        assert changes <= 2

    def test_constraint_on_generated_trajectories(self):
        """批量生成的轨迹都应满足单次交易约束。"""
        N = 10
        T = N * 3
        rng = np.random.RandomState(123)
        prices = 100.0 + np.cumsum(rng.randn(T) * 2)
        prices = np.maximum(prices, 1.0)
        env = _make_env(prices, horizon=N)
        planner = DPPlanner(env)

        result = planner.generate_trajectories(num_trajectories=5)

        for i in range(5):
            changes = _count_position_changes(result["actions"][i], env.m)
            assert changes <= 2, f"轨迹 {i}: 持仓变化 {changes} 次"


# ---------------------------------------------------------------------------
# 4. 轨迹结构验证
# ---------------------------------------------------------------------------

class TestTrajectoryStructure:
    """验证 DP 输出的轨迹结构。"""

    def test_plan_output_shapes(self):
        """plan() 输出的 shape 应正确。"""
        N = 10
        state_dim = 45
        prices = np.linspace(100, 200, N)
        env = _make_env(prices, state_dim=state_dim, horizon=N)
        planner = DPPlanner(env)

        s_demo, a_demo, r_demo = planner.plan(env.states, prices)

        assert s_demo.shape == (N, state_dim)
        assert a_demo.shape == (N,)
        assert r_demo.shape == (N,)

    def test_action_values_in_valid_set(self):
        """动作值应在 {0, 1, 2} 中。"""
        N = 15
        rng = np.random.RandomState(7)
        prices = 100.0 + np.cumsum(rng.randn(N) * 3)
        prices = np.maximum(prices, 1.0)
        env = _make_env(prices, horizon=N)
        planner = DPPlanner(env)

        _, a_demo, _ = planner.plan(env.states, prices)

        assert set(a_demo.tolist()).issubset({0, 1, 2})

    def test_generate_trajectories_shapes(self):
        """generate_trajectories() 输出的 shape 应正确。"""
        N = 10
        state_dim = 45
        T = N * 2
        prices = np.linspace(100, 200, T)
        env = _make_env(prices, state_dim=state_dim, horizon=N)
        planner = DPPlanner(env)

        result = planner.generate_trajectories(num_trajectories=4)

        assert result["states"].shape == (4, N, state_dim)
        assert result["actions"].shape == (4, N)
        assert result["rewards"].shape == (4, N)

    def test_generate_trajectories_action_values(self):
        """批量生成的轨迹动作值应在 {0, 1, 2} 中。"""
        N = 10
        T = N * 2
        prices = np.linspace(100, 200, T)
        env = _make_env(prices, horizon=N)
        planner = DPPlanner(env)

        result = planner.generate_trajectories(num_trajectories=3)

        for i in range(3):
            assert set(result["actions"][i].tolist()).issubset({0, 1, 2})

    def test_s_demo_is_copy_of_states(self):
        """s_demo 应为 states 的副本（不共享内存）。"""
        N = 8
        prices = np.linspace(100, 200, N)
        env = _make_env(prices, horizon=N)
        planner = DPPlanner(env)

        s_demo, _, _ = planner.plan(env.states, prices)

        np.testing.assert_array_equal(s_demo, env.states)
        # 修改 s_demo 不应影响 env.states
        s_demo[0, 0] = 999.0
        assert env.states[0, 0] != 999.0


# ---------------------------------------------------------------------------
# 5. DP 最优性（小规模暴力枚举验证）
# ---------------------------------------------------------------------------

class TestDPOptimality:
    """对长度 ≤ 10 的小规模实例，DP 结果应匹配暴力枚举最大收益。"""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_optimality_random_prices(self, seed):
        """随机价格序列下 DP 折扣收益 == 暴力枚举最大折扣收益。"""
        rng = np.random.RandomState(seed)
        N = 6  # 小规模以保证暴力枚举可行 (3^6 = 729)
        prices = 100.0 + np.cumsum(rng.randn(N) * 5)
        prices = np.maximum(prices, 1.0)

        env = _make_env(prices, pair="BTC", horizon=N)
        planner = DPPlanner(env)
        gamma = planner.gamma

        _, a_demo, r_demo = planner.plan(env.states, prices)
        dp_reward = sum(gamma ** t * r_demo[t] for t in range(N))

        bf_reward = _brute_force_max_reward(prices, m=env.m, gamma=gamma)

        assert abs(dp_reward - bf_reward) < 1e-6, (
            f"seed={seed}: DP reward={dp_reward:.6f} != BF reward={bf_reward:.6f}"
        )

    def test_optimality_trending_up(self):
        """单调递增价格：最优策略应为尽早 long。"""
        N = 5
        prices = np.array([100.0, 110.0, 120.0, 130.0, 140.0])
        env = _make_env(prices, pair="BTC", horizon=N)
        planner = DPPlanner(env)
        gamma = planner.gamma

        _, a_demo, r_demo = planner.plan(env.states, prices)
        dp_reward = sum(gamma ** t * r_demo[t] for t in range(N))
        bf_reward = _brute_force_max_reward(prices, m=env.m, gamma=gamma)

        assert abs(dp_reward - bf_reward) < 1e-6

    def test_optimality_trending_down(self):
        """单调递减价格：最优策略应为尽早 short。"""
        N = 5
        prices = np.array([140.0, 130.0, 120.0, 110.0, 100.0])
        env = _make_env(prices, pair="BTC", horizon=N)
        planner = DPPlanner(env)
        gamma = planner.gamma

        _, a_demo, r_demo = planner.plan(env.states, prices)
        dp_reward = sum(gamma ** t * r_demo[t] for t in range(N))
        bf_reward = _brute_force_max_reward(prices, m=env.m, gamma=gamma)

        assert abs(dp_reward - bf_reward) < 1e-6

    def test_optimality_v_shape(self):
        """V 形价格：先跌后涨。"""
        N = 7
        prices = np.array([100.0, 90.0, 80.0, 70.0, 80.0, 90.0, 100.0])
        env = _make_env(prices, pair="BTC", horizon=N)
        planner = DPPlanner(env)
        gamma = planner.gamma

        _, a_demo, r_demo = planner.plan(env.states, prices)
        dp_reward = sum(gamma ** t * r_demo[t] for t in range(N))
        bf_reward = _brute_force_max_reward(prices, m=env.m, gamma=gamma)

        assert abs(dp_reward - bf_reward) < 1e-6

    def test_optimality_length_10(self):
        """长度 10 的实例（3^10 = 59049 种组合）。"""
        rng = np.random.RandomState(99)
        N = 10
        prices = 100.0 + np.cumsum(rng.randn(N) * 3)
        prices = np.maximum(prices, 1.0)

        env = _make_env(prices, pair="BTC", horizon=N)
        planner = DPPlanner(env)
        gamma = planner.gamma

        _, a_demo, r_demo = planner.plan(env.states, prices)
        dp_reward = sum(gamma ** t * r_demo[t] for t in range(N))
        bf_reward = _brute_force_max_reward(prices, m=env.m, gamma=gamma)

        assert abs(dp_reward - bf_reward) < 1e-6, (
            f"N=10: DP reward={dp_reward:.6f} != BF reward={bf_reward:.6f}"
        )

    def test_optimality_eth_pair(self):
        """使用 ETH 交易对（m=100）验证最优性。"""
        rng = np.random.RandomState(55)
        N = 6
        prices = 2000.0 + np.cumsum(rng.randn(N) * 20)
        prices = np.maximum(prices, 1.0)

        env = _make_env(prices, pair="ETH", horizon=N)
        planner = DPPlanner(env)
        gamma = planner.gamma

        _, a_demo, r_demo = planner.plan(env.states, prices)
        dp_reward = sum(gamma ** t * r_demo[t] for t in range(N))
        bf_reward = _brute_force_max_reward(prices, m=env.m, gamma=gamma)

        assert abs(dp_reward - bf_reward) < 1e-4, (
            f"ETH: DP reward={dp_reward:.6f} != BF reward={bf_reward:.6f}"
        )


# ---------------------------------------------------------------------------
# Property-Based Tests (hypothesis)
# ---------------------------------------------------------------------------

from hypothesis import given, settings
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Property 9: DP 单次交易约束
# Feature: archetype-trader, Property 9: DP 单次交易约束
# 对于任意价格序列，DP Planner 生成的动作序列中持仓变化次数 ≤ 2
# ---------------------------------------------------------------------------

class TestPropDPSingleTradeConstraint:
    """**Validates: Requirements 3.1**"""

    @given(
        h=st.integers(min_value=5, max_value=30),
        seed=st.integers(min_value=0, max_value=2**32 - 1),
    )
    @settings(max_examples=100)
    def test_prop_single_trade_constraint(self, h, seed):
        """任意随机价格序列下，DP 轨迹持仓变化 ≤ 2。"""
        rng = np.random.RandomState(seed)
        prices = 100.0 + np.cumsum(rng.randn(h) * 5)
        prices = np.maximum(prices, 1.0)

        states = rng.randn(h, 45).astype(np.float64)
        env = TradingEnv(states=states, prices=prices, pair="BTC", horizon=h)
        planner = DPPlanner(env)

        _, a_demo, _ = planner.plan(env.states, prices)
        changes = _count_position_changes(a_demo, env.m)
        assert changes <= 2, (
            f"h={h}, seed={seed}: 持仓变化 {changes} 次，超过约束上限 2"
        )


# ---------------------------------------------------------------------------
# Property 10: DP 最优性（小规模模型测试）
# Feature: archetype-trader, Property 10: DP 最优性
# 对于长度 ≤ 10 的价格序列，DP 收益应等于暴力枚举最大收益
# ---------------------------------------------------------------------------

class TestPropDPOptimality:
    """**Validates: Requirements 3.2**"""

    @given(
        h=st.integers(min_value=3, max_value=10),
        seed=st.integers(min_value=0, max_value=2**32 - 1),
    )
    @settings(max_examples=100, deadline=None)
    def test_prop_dp_optimality_small_scale(self, h, seed):
        """长度 ≤ 10 时，DP 折扣收益 ≥ 暴力枚举最大折扣收益（容差内）。"""
        rng = np.random.RandomState(seed)
        prices = 100.0 + np.cumsum(rng.randn(h) * 5)
        prices = np.maximum(prices, 1.0)

        states = rng.randn(h, 45).astype(np.float64)
        env = TradingEnv(states=states, prices=prices, pair="BTC", horizon=h)
        planner = DPPlanner(env)
        gamma = planner.gamma

        _, a_demo, r_demo = planner.plan(env.states, prices)
        dp_reward = sum(gamma ** t * r_demo[t] for t in range(h))

        bf_reward = _brute_force_max_reward(prices, m=env.m, gamma=gamma)

        assert dp_reward >= bf_reward - 1e-6, (
            f"h={h}, seed={seed}: DP reward={dp_reward:.6f} < BF reward={bf_reward:.6f}"
        )


# ---------------------------------------------------------------------------
# Property 11: DP 轨迹结构完整性
# Feature: archetype-trader, Property 11: DP 轨迹结构完整性
# s_demo shape (h, state_dim), a_demo shape (h,) 值域 {0,1,2}, r_demo shape (h,)
# ---------------------------------------------------------------------------

class TestPropDPTrajectoryStructure:
    """**Validates: Requirements 3.5**"""

    @given(
        h=st.integers(min_value=5, max_value=50),
        seed=st.integers(min_value=0, max_value=2**32 - 1),
    )
    @settings(max_examples=100)
    def test_prop_trajectory_structure(self, h, seed):
        """DP 输出的轨迹结构应满足 shape 和值域约束。"""
        rng = np.random.RandomState(seed)
        prices = 100.0 + np.cumsum(rng.randn(h) * 5)
        prices = np.maximum(prices, 1.0)

        state_dim = 45
        states = rng.randn(h, state_dim).astype(np.float64)
        env = TradingEnv(states=states, prices=prices, pair="BTC", horizon=h)
        planner = DPPlanner(env)

        s_demo, a_demo, r_demo = planner.plan(env.states, prices)

        # s_demo shape (h, state_dim)
        assert s_demo.shape == (h, state_dim), (
            f"s_demo shape {s_demo.shape} != expected ({h}, {state_dim})"
        )
        # a_demo shape (h,) with values in {0, 1, 2}
        assert a_demo.shape == (h,), (
            f"a_demo shape {a_demo.shape} != expected ({h},)"
        )
        assert set(a_demo.tolist()).issubset({0, 1, 2}), (
            f"a_demo contains invalid values: {set(a_demo.tolist()) - {0, 1, 2}}"
        )
        # r_demo shape (h,)
        assert r_demo.shape == (h,), (
            f"r_demo shape {r_demo.shape} != expected ({h},)"
        )
