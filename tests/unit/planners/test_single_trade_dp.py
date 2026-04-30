"""``SingleTradeDPPlanner`` 单元测试 (含 Algorithm 1 末步处理)."""
from __future__ import annotations

import pytest

from src.planners.single_trade_dp import DPInputs, SingleTradeDPPlanner
from src.trading.cost_model import ExecutionBook, LobDepthCostModel
from src.trading.reward_alignment import RewardAlignment


def _book(mark: float, spread_bps: float = 1.0, depth: float = 100.0) -> ExecutionBook:
    factor = spread_bps / 10000.0
    asks = tuple(mark * (1 + factor * (i + 1)) for i in range(5))
    bids = tuple(mark * (1 - factor * (i + 1)) for i in range(5))
    sizes = (depth,) * 5
    return ExecutionBook(
        ask_prices=asks, ask_sizes=sizes,
        bid_prices=bids, bid_sizes=sizes,
        mark_price=mark,
    )


def _make_planner(commission: float = 0.0):
    cm = LobDepthCostModel(commission_rate=commission)
    align = RewardAlignment("paper_formula")
    return SingleTradeDPPlanner(cost_model=cm, reward_alignment=align, max_position=1)


def test_monotonic_uptrend_chooses_flat_to_long():
    """单调上涨 → flat→long，至多 1 切换。"""
    planner = _make_planner(commission=0.0)
    # h=8: 价格从 100 涨到 108
    prices = [100.0 + i for i in range(9)]  # h+1
    books = [_book(p, spread_bps=0.1) for p in prices[:-1]]
    inp = DPInputs(prices=prices, execution_books=books, horizon=8)
    res = planner.plan(inp)
    assert res.num_switches <= 1
    # 期望 long 仓段比 short 多
    assert res.actions.count(2) >= res.actions.count(0)


def test_monotonic_downtrend_chooses_flat_to_short():
    planner = _make_planner(commission=0.0)
    prices = [108.0 - i for i in range(9)]
    books = [_book(p, spread_bps=0.1) for p in prices[:-1]]
    inp = DPInputs(prices=prices, execution_books=books, horizon=8)
    res = planner.plan(inp)
    assert res.num_switches <= 1
    assert res.actions.count(0) >= res.actions.count(2)


def test_sideways_with_fees_chooses_all_flat():
    """横盘 + 较高手续费 → 全 flat 是最优。"""
    planner = _make_planner(commission=0.01)  # 1% 手续费夸张化让 DP 不交易
    prices = [100.0] * 9
    books = [_book(100.0, spread_bps=20.0) for _ in range(8)]
    inp = DPInputs(prices=prices, execution_books=books, horizon=8)
    res = planner.plan(inp)
    assert res.is_no_trade
    assert all(a == 1 for a in res.actions)


def test_actions_length_equals_horizon():
    planner = _make_planner()
    prices = [100.0 + i * 0.1 for i in range(9)]
    books = [_book(p) for p in prices[:-1]]
    res = planner.plan(DPInputs(prices=prices, execution_books=books, horizon=8))
    assert len(res.actions) == 8
    assert len(res.rewards) == 8


def test_dp_total_return_matches_replay_sum():
    """DP total_return == sum(rewards) (env replay)。"""
    planner = _make_planner()
    prices = [100.0 + i * 0.5 for i in range(9)]
    books = [_book(p, spread_bps=0.1) for p in prices[:-1]]
    res = planner.plan(DPInputs(prices=prices, execution_books=books, horizon=8))
    assert res.total_return == pytest.approx(sum(res.rewards), abs=1e-6)


def test_last_step_copies_second_last_when_holding_long():
    """Algorithm 1 末步: actions[N-1] == actions[N-2]。"""
    planner = _make_planner()
    prices = [100.0 + i for i in range(9)]
    books = [_book(p, spread_bps=0.1) for p in prices[:-1]]
    res = planner.plan(DPInputs(prices=prices, execution_books=books, horizon=8))
    # 末步必须复制倒数第二步
    assert res.actions[-1] == res.actions[-2]


def test_last_step_copies_second_last_when_back_to_flat():
    """末步已回到 flat 也必须复制。"""
    planner = _make_planner(commission=0.001)
    # 设计一个先涨后跌的场景；single-trade 限制下 DP 通常会选不交易，
    # 但此处只验证末步规则。
    prices = [100.0, 100.5, 101.0, 101.5, 101.0, 100.5, 100.0, 99.5, 99.0]
    books = [_book(p, spread_bps=2.0) for p in prices[:-1]]
    res = planner.plan(DPInputs(prices=prices, execution_books=books, horizon=8))
    assert res.actions[-1] == res.actions[-2]


def test_last_step_copy_does_not_count_as_switch():
    """末步复制不进入 num_switches 计数。"""
    planner = _make_planner()
    prices = [100.0 + i for i in range(9)]
    books = [_book(p, spread_bps=0.1) for p in prices[:-1]]
    res = planner.plan(DPInputs(prices=prices, execution_books=books, horizon=8))
    # num_switches 在 actions[0..N-2] 上数；末步赋值不影响该范围。
    raw = sum(1 for i in range(1, len(res.actions) - 1) if res.actions[i] != res.actions[i - 1])
    assert res.num_switches == raw
