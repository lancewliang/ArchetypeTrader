"""``TradingEnv`` 单元测试."""
from __future__ import annotations

import pytest

from src.trading.cost_model import ExecutionBook, LobDepthCostModel
from src.trading.env import HorizonInputs, TradingEnv
from src.trading.reward_alignment import RewardAlignment


def _book(mark: float, spread_bps: float = 5.0):
    factor = spread_bps / 10000.0
    asks = tuple(mark * (1 + factor * (i + 1)) for i in range(5))
    bids = tuple(mark * (1 - factor * (i + 1)) for i in range(5))
    sizes = (100.0,) * 5
    return ExecutionBook(
        ask_prices=asks, ask_sizes=sizes,
        bid_prices=bids, bid_sizes=sizes,
        mark_price=mark,
    )


def _make_env(alignment="paper_formula"):
    cm = LobDepthCostModel(commission_rate=0.0002)
    align = RewardAlignment(alignment)
    return TradingEnv(cost_model=cm, reward_alignment=align, max_position=1)


def test_action_to_position_mapping():
    env = _make_env()
    assert env._action_to_position(0) == -1
    assert env._action_to_position(1) == 0
    assert env._action_to_position(2) == 1


def test_invalid_action_raises():
    env = _make_env()
    with pytest.raises(ValueError):
        env._action_to_position(3)


def test_replay_equals_iterating_step_paper_formula():
    """Given 一段 actions / When replay vs 反复 step / Then step rewards 相同。

    注意: 实现里 replay 内部就是反复调用 step，故必然一致；
    本测试主要验证 done 触发时 replay 不跑越界。
    """
    env = _make_env()
    prices = [100.0, 101.0, 102.0, 103.0]  # h=3 + 1 markout
    books = [_book(p) for p in prices[:-1]]  # h=3 个盘口
    horizon = HorizonInputs(prices=prices, execution_books=books)

    env.reset(horizon)
    actions = [2, 2, 1]  # long, long, flat
    rewards, infos = env.replay(actions)
    assert len(rewards) == 3
    assert len(infos) == 3
    # done 在最后一步触发
    env.reset(horizon)
    rs2 = []
    for a in actions:
        r, done, _ = env.step(a)
        rs2.append(r)
        if done:
            break
    assert rewards == pytest.approx(rs2)


def test_supports_non_flat_initial_position():
    """Given reset(initial_position=+1) + step(flat) / Then 第一步扣换仓成本。"""
    env = _make_env()
    horizon = HorizonInputs(prices=[100.0, 100.0], execution_books=[_book(100.0)])
    env.reset(horizon, initial_position=1)
    reward, done, info = env.step(1)  # action=1 → target=0 (flat)
    assert done
    # markout - exec 的位置收益贡献 = 0 * (100-100)；reward 应为负（仅扣手续费/滑点）
    assert reward < 0


def test_step_info_contains_reject_event():
    env = _make_env()
    horizon = HorizonInputs(prices=[100.0, 100.0], execution_books=[_book(100.0)])
    env.reset(horizon)
    _, _, info = env.step(1)
    assert hasattr(info, "reject_event")


def test_invalid_initial_position_raises():
    env = _make_env()
    horizon = HorizonInputs(prices=[100.0, 100.0], execution_books=[_book(100.0)])
    with pytest.raises(ValueError):
        env.reset(horizon, initial_position=99)
