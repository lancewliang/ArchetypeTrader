"""交易审计模块测试"""

import pytest

from src.evaluation.portfolio_tracker import PortfolioTracker
from src.evaluation.trade_auditor import TradeAuditor


def _build_records(schedule, T=12, m=100, rate=0.0002, price_start=2000.0):
    """构造 tracker records。schedule: {bar: target_position}"""
    import numpy as np
    prices = np.arange(price_start, price_start + T, 1.0)[:T]
    cap = float(m * prices[0])
    tracker = PortfolioTracker(cap)
    pos = 0
    for t in range(T):
        new = schedule.get(t, pos)
        delta = new - pos
        price = float(prices[t])
        comm = round(rate * abs(delta) * price, 2) if delta else 0.0
        slip = round(0.0001 * abs(delta) * price, 2) if delta else 0.0
        tracker.update_cash_for_trade(pos, new, price, t)
        action = 2 if new > 0 else (0 if new < 0 else 1)
        tracker.record_step(t, action, price, pos, new, comm, slip)
        pos = new
    return tracker.records, cap


class TestTradeAuditorFlat:

    def test_all_flat(self):
        records, cap = _build_records({})
        report = TradeAuditor(records, cap).audit()
        stats = report["statistics"]
        assert stats["total_commission"] == 0.0
        assert stats["total_slippage"] == 0.0
        assert stats["trade_counts"]["hold_flat"] == 12
        assert all(c["pass"] for c in report["checks"])


class TestTradeAuditorLongClose:

    def test_open_and_close(self):
        records, cap = _build_records({0: 100, 11: 0})
        report = TradeAuditor(records, cap).audit()
        tc = report["statistics"]["trade_counts"]
        assert tc["flat_to_long"] == 1
        assert tc["long_to_flat"] == 1
        assert tc["hold_long"] == 10
        assert report["statistics"]["total_commission"] > 0
        assert all(c["pass"] for c in report["checks"])


class TestTradeAuditorShortClose:

    def test_open_and_close(self):
        records, cap = _build_records({0: -100, 11: 0})
        report = TradeAuditor(records, cap).audit()
        tc = report["statistics"]["trade_counts"]
        assert tc["flat_to_short"] == 1
        assert tc["short_to_flat"] == 1
        assert tc["hold_short"] == 10
        assert all(c["pass"] for c in report["checks"])


class TestTradeAuditorFlip:

    def test_long_to_short(self):
        records, cap = _build_records({0: 100, 6: -100, 11: 0})
        report = TradeAuditor(records, cap).audit()
        tc = report["statistics"]["trade_counts"]
        assert tc["flat_to_long"] == 1
        assert tc["long_to_short"] == 1
        assert tc["short_to_flat"] == 1
        assert all(c["pass"] for c in report["checks"])


class TestTradeAuditorChecks:

    def test_record_count_check(self):
        records, cap = _build_records({0: 100, 5: 0})
        report = TradeAuditor(records, cap).audit()
        rc = next(c for c in report["checks"] if c["name"] == "record_count")
        assert rc["pass"] is True

    def test_profit_consistency(self):
        records, cap = _build_records({0: 100, 11: 0})
        report = TradeAuditor(records, cap).audit()
        pc = next(c for c in report["checks"] if c["name"] == "profit_consistency")
        assert pc["pass"] is True

    def test_flat_cash_equals_value(self):
        records, cap = _build_records({0: 100, 11: 0})
        report = TradeAuditor(records, cap).audit()
        fc = next(c for c in report["checks"] if c["name"] == "flat_cash_equals_value")
        assert fc["pass"] is True
