"""``LobDepthCostModel`` 单元测试."""
from __future__ import annotations

import pytest

from src.trading.cost_model import ExecutionBook, LobDepthCostModel


def _book(asks=(101.0, 101.5, 102.0, 102.5, 103.0),
          ask_sizes=(10.0, 10.0, 10.0, 10.0, 10.0),
          bids=(99.0, 98.5, 98.0, 97.5, 97.0),
          bid_sizes=(10.0, 10.0, 10.0, 10.0, 10.0),
          mark=100.0) -> ExecutionBook:
    return ExecutionBook(
        ask_prices=tuple(asks),
        ask_sizes=tuple(ask_sizes),
        bid_prices=tuple(bids),
        bid_sizes=tuple(bid_sizes),
        mark_price=mark,
    )


def test_zero_delta_returns_zero_cost():
    """Given prev == target / When execute / Then fee=0, slippage=0, fill_price=None。"""
    cm = LobDepthCostModel(commission_rate=0.0002)
    res = cm.execute(prev_position=1, target_position=1, execution_book=_book())
    assert res.fee == 0.0
    assert res.slippage == 0.0
    assert res.fill_price is None
    assert not res.rejected


def test_buy_consumes_ask_levels():
    """Given prev=0,target=+1 / When execute / Then 走 ask 档。"""
    cm = LobDepthCostModel(commission_rate=0.0002)
    res = cm.execute(prev_position=0, target_position=1, execution_book=_book())
    assert not res.rejected
    # delta=1 < ask1 size (10) → 全部 ask1 成交
    assert res.fill_price == pytest.approx(101.0)


def test_sell_consumes_bid_levels():
    cm = LobDepthCostModel(commission_rate=0.0002)
    res = cm.execute(prev_position=0, target_position=-1, execution_book=_book())
    assert not res.rejected
    assert res.fill_price == pytest.approx(99.0)


def test_fill_price_volume_weighted_across_levels():
    """Given delta 跨多档 / When execute / Then fill_price = Σq*p / Σq。"""
    cm = LobDepthCostModel(commission_rate=0.0)
    book = _book(
        asks=(100.0, 110.0, 120.0, 130.0, 140.0),
        ask_sizes=(2.0, 3.0, 1.0, 1.0, 1.0),
        mark=100.0,
    )
    # delta=4 → ask1(2 @ 100) + ask2(2 @ 110) = (2*100 + 2*110)/4 = 105.0
    res = cm.execute(prev_position=0, target_position=4, execution_book=book)
    assert res.fill_price == pytest.approx(105.0)


def test_reject_when_depth_insufficient():
    """Given 总深度 < |delta| / When execute / Then rejected。"""
    cm = LobDepthCostModel(commission_rate=0.0)
    book = _book(ask_sizes=(0.5, 0.5, 0.5, 0.5, 0.5))  # 总 ask 深度 = 2.5
    res = cm.execute(prev_position=0, target_position=3, execution_book=book)
    assert res.rejected
    assert res.filled_position == 0
    assert res.fill_price is None


def test_fee_uses_commission_rate():
    """Given delta=1, mark=100, δ=0.0002 / When execute / Then fee = 0.02。"""
    cm = LobDepthCostModel(commission_rate=0.0002)
    res = cm.execute(prev_position=0, target_position=1, execution_book=_book(mark=100.0))
    assert res.fee == pytest.approx(0.0002 * 1 * 100.0)


def test_slippage_uses_abs_diff_to_mark():
    """Given fill > mark / Then slippage = |delta| * |fill - mark|。"""
    cm = LobDepthCostModel(commission_rate=0.0)
    res = cm.execute(prev_position=0, target_position=1, execution_book=_book(mark=100.0))
    assert res.slippage == pytest.approx(1 * abs(101.0 - 100.0))


def test_invalid_policy_raises():
    with pytest.raises(ValueError):
        LobDepthCostModel(commission_rate=0.0, insufficient_depth_policy="other")
