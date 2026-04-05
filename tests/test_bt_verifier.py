"""Backtrader 交叉验证测试

验证 bt 与 tracker 的分项数据逐项一致:
  持仓、总手续费、交易次数、各类型交易次数、最终现金/总值。
"""

import numpy as np
import pytest

from src.evaluation.bt_verifier import BacktraderVerifier, FixedRateCommission
from src.evaluation.portfolio_tracker import PortfolioTracker
from src.evaluation.trade_auditor import TradeAuditor


# ------------------------------------------------------------------
# 辅助
# ------------------------------------------------------------------

def _build(schedule, T=12, m=100, rate=0.0002, price_start=2000.0, step=1.0):
    """构造 tracker records + audit stats + prices (多 1 bar)。"""
    prices = np.arange(price_start, price_start + (T + 1) * step, step)[:T + 1]
    cap = float(m * prices[0])
    tracker = PortfolioTracker(cap)
    pos = 0
    for t in range(T):
        new = schedule.get(t, pos)
        delta = new - pos
        price = float(prices[t])
        comm = round(rate * abs(delta) * price, 2) if delta else 0.0
        tracker.update_cash_for_trade(pos, new, price, t)
        action = 2 if new > 0 else (0 if new < 0 else 1)
        tracker.record_step(t, action, price, pos, new, comm, 0.0)
        pos = new
    audit = TradeAuditor(tracker.records, cap).audit()
    return tracker.records, prices, cap, audit


# ------------------------------------------------------------------
# 佣金
# ------------------------------------------------------------------

class TestFixedRateCommission:
    def test_formula(self):
        c = FixedRateCommission(commission=0.0002)
        assert c._getcommission(100, 2000.0, False) == pytest.approx(40.0)

    def test_rounds(self):
        c = FixedRateCommission(commission=0.0002)
        assert c._getcommission(3, 1873.26, False) == round(0.0002 * 3 * 1873.26, 2)


# ------------------------------------------------------------------
# 全 flat
# ------------------------------------------------------------------

class TestFlat:
    def test_all_pass(self):
        recs, prices, cap, audit = _build({})
        report = BacktraderVerifier(recs, prices, cap, 100, tolerance=0.01).run(
            tracker_stats=audit["statistics"],
        )
        assert report["all_pass"] is True
        assert report["position_mismatches"] == 0


# ------------------------------------------------------------------
# 开多 → 平仓
# ------------------------------------------------------------------

class TestLong:
    def test_commission_and_counts(self):
        recs, prices, cap, audit = _build({0: 100, 11: 0})
        report = BacktraderVerifier(recs, prices, cap, 100, 0.0002, 0.01).run(
            tracker_stats=audit["statistics"],
        )
        # 手续费一致
        comm_check = next(c for c in report["comparisons"] if c["name"] == "total_commission")
        assert comm_check["pass"] is True

        # 交易类型一致
        ftl = next(c for c in report["comparisons"] if c["name"] == "trade_count_flat_to_long")
        assert ftl["bt"] == 1 and ftl["tracker"] == 1 and ftl["pass"]

        ltf = next(c for c in report["comparisons"] if c["name"] == "trade_count_long_to_flat")
        assert ltf["bt"] == 1 and ltf["tracker"] == 1 and ltf["pass"]

        assert report["position_mismatches"] == 0


# ------------------------------------------------------------------
# 开空 → 平仓
# ------------------------------------------------------------------

class TestShort:
    def test_commission_and_counts(self):
        recs, prices, cap, audit = _build({0: -100, 11: 0}, step=-1.0)
        report = BacktraderVerifier(recs, prices, cap, 100, 0.0002, 0.01).run(
            tracker_stats=audit["statistics"],
        )
        comm_check = next(c for c in report["comparisons"] if c["name"] == "total_commission")
        assert comm_check["pass"] is True

        fts = next(c for c in report["comparisons"] if c["name"] == "trade_count_flat_to_short")
        assert fts["bt"] == 1 and fts["pass"]

        stf = next(c for c in report["comparisons"] if c["name"] == "trade_count_short_to_flat")
        assert stf["bt"] == 1 and stf["pass"]


# ------------------------------------------------------------------
# 换仓 多→空→多
# ------------------------------------------------------------------

class TestFlip:
    def test_flip_counts(self):
        recs, prices, cap, audit = _build({0: 100, 6: -100, 11: 0}, T=12)
        report = BacktraderVerifier(recs, prices, cap, 100, 0.0002, 0.01).run(
            tracker_stats=audit["statistics"],
        )
        lts = next(c for c in report["comparisons"] if c["name"] == "trade_count_long_to_short")
        assert lts["bt"] == 1 and lts["pass"]

        stf = next(c for c in report["comparisons"] if c["name"] == "trade_count_short_to_flat")
        assert stf["bt"] == 1 and stf["pass"]

        assert report["position_mismatches"] == 0


# ------------------------------------------------------------------
# 最终总值 (无滑点时应完全一致)
# ------------------------------------------------------------------

class TestFinalValue:
    def test_no_slippage_exact_match(self):
        recs, prices, cap, audit = _build({0: 100, 11: 0})
        report = BacktraderVerifier(recs, prices, cap, 100, 0.0002, 0.01).run(
            tracker_stats=audit["statistics"],
        )
        val_check = next(c for c in report["comparisons"] if c["name"] == "final_value")
        assert val_check["pass"] is True
        assert val_check["residual"] <= 0.01


# ------------------------------------------------------------------
# 报告结构
# ------------------------------------------------------------------

class TestReportStructure:
    def test_has_required_keys(self):
        recs, prices, cap, audit = _build({})
        report = BacktraderVerifier(recs, prices, cap, 100).run(
            tracker_stats=audit["statistics"],
        )
        assert "all_pass" in report
        assert "bt_stats" in report
        assert "tracker_stats" in report
        assert "comparisons" in report
        assert "position_mismatches" in report
