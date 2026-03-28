"""DP / TradingEnv 单元测试

# 验证目标:
# 1. Section 3.1 / Eq. 1 的 reward 与 execution cost 实现一致。
# 2. Algorithm 1 的 DP 结果与 brute-force oracle 完全一致。
# 3. DP 输出的 (a_demo, r_demo) 能被环境逐步回放精确复现。
"""

from __future__ import annotations

import numpy as np
import polars as pl

from src.env.trading_env import TradingEnv
from src.phase1.dp_planner import DPPlanner
from src.phase1.validation import (
    brute_force_optimal_sequence,
    replay_actions_on_horizon,
)



def build_synthetic_lob_dataframe(length: int, base_mark: float = 100.0) -> pl.DataFrame:
    """构造满足 TradingEnv LOB 列要求的最小可测 DataFrame。"""
    rows = []
    for i in range(length):
        mark = base_mark + i
        row = {
            # ask side
            "ask1_price": mark + 1.0,
            "ask2_price": mark + 2.0,
            "ask3_price": mark + 3.0,
            "ask4_price": mark + 4.0,
            "ask5_price": mark + 5.0,
            "ask1_size": 1.0,
            "ask2_size": 1.0,
            "ask3_size": 1.0,
            "ask4_size": 1.0,
            "ask5_size": 1.0,
            # bid side
            "bid1_price": mark - 1.0,
            "bid2_price": mark - 2.0,
            "bid3_price": mark - 3.0,
            "bid4_price": mark - 4.0,
            "bid5_price": mark - 5.0,
            "bid1_size": 1.0,
            "bid2_size": 1.0,
            "bid3_size": 1.0,
            "bid4_size": 1.0,
            "bid5_size": 1.0,
            # 额外特征，模拟 state vector 中的其它字段
            "feat_0": float(i),
            "feat_1": float(i % 2),
        }
        rows.append(row)
    return pl.DataFrame(rows)



def test_trading_env_step_matches_manual_reward_formula() -> None:
    """验证 env.step() 与手工 reward 计算一致。"""
    states_df = build_synthetic_lob_dataframe(length=4, base_mark=100.0)
    states = states_df.to_numpy()
    prices = np.asarray([100.0, 101.5, 101.0, 102.0], dtype=np.float64)

    env = TradingEnv(
        states=states,
        prices=prices,
        pair="BTC",
        horizon=3,
        states_dataframe=states_df,
    )

    env.reset(0)
    _, reward, _, info = env.step(2)  # flat -> long

    # 手工计算:
    # slippage = ask1 - mark = 101 - 100 = 1
    # commission = 0.0002 * 1 * 100 = 0.02
    # execution_cost = 1.02
    # 持仓变为 +1, 价格变化 = 101.5 - 100 = 1.5
    # reward = 1 * 1.5 - 1.02 = 0.48
    manual_reward = 1.0 * (101.5 - 100.0) - (1.0 + 0.0002 * 100.0)

    assert abs(reward - manual_reward) < 1e-8
    assert abs(info["execution_cost"] - 1.02) < 1e-8



def test_dp_matches_bruteforce_oracle_on_small_horizon() -> None:
    """验证短 horizon 上 DP 与 brute-force oracle 的折扣回报完全一致。"""
    states_df = build_synthetic_lob_dataframe(length=6, base_mark=100.0)
    prices = np.asarray([100.0, 100.5, 103.0, 102.0, 101.0, 100.5], dtype=np.float64)
    env = TradingEnv(
        states=states_df.to_numpy(),
        prices=prices,
        pair="BTC",
        horizon=5,
        states_dataframe=states_df,
    )
    planner = DPPlanner(env=env, gamma=0.99)

    # 对前 4 步做严格 oracle 对照，避免组合爆炸。
    sub_states = states_df[:4]
    sub_prices = prices[:5]

    _, dp_actions, dp_rewards = planner.plan(sub_states, sub_prices)
    dp_return = sum((planner.gamma ** t) * float(r) for t, r in enumerate(dp_rewards))

    brute_actions, brute_rewards, brute_return = brute_force_optimal_sequence(
        planner=planner,
        states=sub_states,
        prices=sub_prices,
    )

    assert abs(dp_return - brute_return) < 1e-8
    assert np.allclose(dp_rewards, brute_rewards, atol=1e-8) or np.array_equal(dp_actions, brute_actions)



def test_dp_demo_rewards_can_be_replayed_by_environment() -> None:
    """验证 plan() 输出的 r_demo 与环境逐步回放一致。"""
    states_df = build_synthetic_lob_dataframe(length=5, base_mark=100.0)
    # 注意这里 states/prices 长度为 5, horizon=4；最后一步 reward 仍可访问全局 t+1。
    prices = np.asarray([100.0, 101.0, 102.5, 101.5, 101.0], dtype=np.float64)
    env = TradingEnv(
        states=states_df.to_numpy(),
        prices=prices,
        pair="BTC",
        horizon=4,
        states_dataframe=states_df,
    )
    planner = DPPlanner(env=env, gamma=0.99)

    sub_states = states_df[:4]
    sub_prices = prices[:5]
    _, dp_actions, dp_rewards = planner.plan(sub_states, sub_prices)

    replay_rewards = replay_actions_on_horizon(env, horizon_idx=0, actions=dp_actions)
    assert np.allclose(dp_rewards, replay_rewards, atol=1e-8)
