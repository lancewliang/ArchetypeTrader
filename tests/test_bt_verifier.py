"""Backtrader 交叉验证测试

使用 backtrader 独立回放交易信号，逐步对比 PortfolioTracker 的
cash / total_value / position，验证资金计算正确性。

注意: backtrader set_coc(True) 下，bar N 的订单在 next() 返回后成交，
bt_log[N+1] 才能看到成交后的持仓。因此 prices 数组需要比 records
多一个 bar，对比时 tracker[idx] ↔ bt[idx+1]。
"""

import numpy as np
import pytest

from src.evaluation.bt_verifier import (
    BacktraderVerifier,
    FixedRateCommission,
)
from src.evaluation.portfolio_tracker import PortfolioTracker


# ------------------------------------------------------------------
# 佣金计算
# ------------------------------------------------------------------

class TestFixedRateCommission:

    def test_commission_formula(self):
        comm = FixedRateCommission(commission=0.0002)
        # 0.0002 * 100 * 2000 = 40.0
        assert comm._getcommission(100, 2000.0, False) == pytest.approx(40.0)

    def test_commission_sell(self):
        comm = FixedRateCommission(commission=0.0002)
        assert comm._getcommission(-100, 2000.0, False) == pytest.approx(40.0)

    def test_commission_rounds_to_2(self):
        comm = FixedRateCommission(commission=0.0002)
        result = comm._getcommission(3, 1873.26, False)
        assert result == round(0.0002 * 3 * 1873.26, 2)


# ------------------------------------------------------------------
# 辅助函数
# ------------------------------------------------------------------

def _build_tracker_and_prices(
    T: int, m: int, rate: float, action_schedule: dict,
    price_start: float = 2000.0, price_step: float = 1.0,
):
    """构造 tracker records 和 prices 数组。

    action_schedule: {bar_idx: target_position}，未指定的 bar 保持当前持仓。
    prices 比 records 多 1 个 bar（用于 bt 偏移对比）。
    """
    prices = np.arange(price_start, price_start + (T + 1) * price_step, price_step)[:T + 1]
    cap = float(m * prices[0])
    tracker = PortfolioTracker(cap)
    pos = 0

    for t in range(T):
        new = action_schedule.get(t, pos)
        delta = new - pos
        price = float(prices[t])
        comm = round(rate * abs(delta) * price, 2) if delta else 0.0
        tracker.update_cash_for_trade(pos, new, price, t)
        action = 2 if new > 0 else (0 if new < 0 else 1)
        tracker.record_step(t, action, price, pos, new, comm, 0.0)
        pos = new

    return tracker, prices, cap


# ------------------------------------------------------------------
# 场景 1: 全 flat
# ------------------------------------------------------------------

class TestVerifierFlat:

    def test_flat_no_trade(self):
        T = 12
        tracker, prices, cap = _build_tracker_and_prices(
            T, m=100, rate=0.0002, action_schedule={},
        )
        report = BacktraderVerifier(
            tracker.records, prices, cap, 100, tolerance=0.01,
        ).run()
        assert report["match"] is True


# ------------------------------------------------------------------
# 场景 2: 开多 → 持有 → 平仓
# ------------------------------------------------------------------

class TestVerifierLong:

    def test_long_then_flat(self):
        T = 12
        tracker, prices, cap = _build_tracker_and_prices(
            T, m=100, rate=0.0002,
            action_schedule={0: 100, 11: 0},
        )
        report = BacktraderVerifier(
            tracker.records, prices, cap, 100, 0.0002, tolerance=0.01,
        ).run()
        assert report["match"] is True, report["mismatches"][:3]


# ------------------------------------------------------------------
# 场景 3: 开空 → 持有 → 平仓
# ------------------------------------------------------------------

class TestVerifierShort:

    def test_short_then_flat(self):
        T = 12
        tracker, prices, cap = _build_tracker_and_prices(
            T, m=100, rate=0.0002,
            action_schedule={0: -100, 11: 0},
            price_start=2000.0, price_step=-1.0,
        )
        report = BacktraderVerifier(
            tracker.records, prices, cap, 100, 0.0002, tolerance=0.01,
        ).run()
        assert report["match"] is True, report["mismatches"][:3]


# ------------------------------------------------------------------
# 场景 4: 换仓 多→空→多
# ------------------------------------------------------------------

class TestVerifierFlip:

    def test_long_short_long(self):
        T = 18
        tracker, prices, cap = _build_tracker_and_prices(
            T, m=100, rate=0.0002,
            action_schedule={0: 100, 6: -100, 12: 100},
        )
        report = BacktraderVerifier(
            tracker.records, prices, cap, 100, 0.0002, tolerance=0.01,
        ).run()
        for mi in report["mismatches"]:
            assert mi["pos_match"] is True, (
                f"idx={mi['state_index']} bt={mi['bt_position']} "
                f"trk={mi['tracker_position']}"
            )


# ------------------------------------------------------------------
# 报告结构
# ------------------------------------------------------------------

class TestVerifierReport:

    def test_report_fields(self):
        prices = np.array([100.0] * 11)  # 10 bars + 1 trailing
        tracker = PortfolioTracker(10000.0)
        for t in range(10):
            tracker.record_step(t, 1, 100.0, 0, 0, 0.0, 0.0)

        report = BacktraderVerifier(
            tracker.records, prices, 10000.0, 100,
        ).run()
        for key in ("match", "total_bars", "mismatches",
                     "bt_final_value", "tracker_final_value", "summary"):
            assert key in report
