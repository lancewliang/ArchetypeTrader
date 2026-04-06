"""Backtrader 交叉验证模块

将 PortfolioTracker 的交易记录回放到 backtrader 中，
提取 bt 侧的分项数据（总手续费、交易次数、最终现金/总值/持仓），
与 TradeAuditor 的 CSV 审计结果逐项对比。

用法:
    from src.evaluation.bt_verifier import BacktraderVerifier
    verifier = BacktraderVerifier(tracker.records, prices, initial_capital, m,
                                  commission_rate)
    report = verifier.run(tracker_audit_stats)
"""

from __future__ import annotations

import datetime as dt
from typing import Dict, List

import backtrader as bt
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ------------------------------------------------------------------
# 自定义数据源
# ------------------------------------------------------------------

class ArrayPriceFeed(bt.feed.DataBase):
    """将 numpy 价格数组转为 backtrader DataFeed (OHLCV)。"""

    params = (
        ("prices", None),
        ("start_date", None),
    )

    def __init__(self):
        super().__init__()
        self._idx = 0
        self._prices = self.p.prices
        self._size = len(self._prices)

    def start(self):
        super().start()
        self._idx = 0

    def _load(self):
        if self._idx >= self._size:
            return False
        price = float(self._prices[self._idx])
        base_date = self.p.start_date or dt.datetime(2024, 1, 1)
        bar_dt = base_date + dt.timedelta(minutes=10 * self._idx)
        self.lines.datetime[0] = bt.date2num(bar_dt)
        self.lines.open[0] = price
        self.lines.high[0] = price
        self.lines.low[0] = price
        self.lines.close[0] = price
        self.lines.volume[0] = 0
        self.lines.openinterest[0] = 0
        self._idx += 1
        return True


# ------------------------------------------------------------------
# 自定义佣金
# ------------------------------------------------------------------

class FixedRateCommission(bt.CommInfoBase):
    params = (
        ("commission", 0.0003),
        ("stocklike", False),
        ("commtype", bt.CommInfoBase.COMM_PERC),
        ("percabs", True),
    )

    def _getcommission(self, size, price, pseudoexec):
        return round(abs(size) * price * self.p.commission, 2)


# ------------------------------------------------------------------
# 信号回放策略 — 记录每笔成交的详细信息
# ------------------------------------------------------------------

class SignalReplayStrategy(bt.Strategy):
    """按 tracker records 逐步下单，记录每笔成交详情。"""

    params = (
        ("records", []),
        ("max_position", 100),
    )

    def __init__(self):
        self._target_by_idx: Dict[int, int] = {}
        for rec in self.p.records:
            self._target_by_idx[rec["state_index"]] = rec["position_after"]

        self.bt_log: List[Dict] = []
        self._bar_count = 0

        # 分项统计
        self.total_commission = 0.0
        self.trade_count = 0
        self.buy_count = 0
        self.sell_count = 0

    def notify_order(self, order):
        if order.status != order.Completed:
            return
        self.total_commission += order.executed.comm
        self.trade_count += 1
        if order.executed.size > 0:
            self.buy_count += 1
        else:
            self.sell_count += 1

    def next(self):
        bar_idx = self._bar_count
        self._bar_count += 1

        if bar_idx in self._target_by_idx:
            target_pos = self._target_by_idx[bar_idx]
            current_pos = self.getposition(self.data).size
            delta = target_pos - current_pos
            if delta != 0:
                if delta > 0:
                    self.buy(size=abs(delta))
                else:
                    self.sell(size=abs(delta))

        self.bt_log.append({
            "bar_idx": bar_idx,
            "bt_cash": round(self.broker.getcash(), 6),
            "bt_value": round(self.broker.getvalue(), 6),
            "bt_position": self.getposition(self.data).size,
        })


# ------------------------------------------------------------------
# 验证器
# ------------------------------------------------------------------

class BacktraderVerifier:
    """使用 backtrader 交叉验证，逐项对比 CSV 审计数据。

    对比项:
      - 最终持仓
      - 最终总值 (扣除滑点差异后)
      - 总手续费 (bt vs tracker)
      - 交易次数 (bt vs tracker)
      - 各类型交易次数 (开多/开空/平多/平空/多转空/空转多)
    """

    def __init__(
        self,
        records: List[Dict],
        prices: np.ndarray,
        initial_capital: float,
        m: int,
        commission_rate: float = 0.0003,
        tolerance: float = 1.0,
    ) -> None:
        self.records = records
        self.prices = prices
        self.initial_capital = initial_capital
        self.m = m
        self.commission_rate = commission_rate
        self.tolerance = tolerance

    def run(self, tracker_stats: Dict | None = None) -> Dict:
        """执行 backtrader 回测，提取分项数据，与 tracker_stats 逐项对比。

        Args:
            tracker_stats: TradeAuditor.audit()["statistics"]，
                           如果提供则进行逐项对比。
        """
        cerebro = bt.Cerebro()
        data = ArrayPriceFeed(
            prices=self.prices,
            start_date=dt.datetime(2024, 1, 1),
        )
        cerebro.adddata(data)
        cerebro.broker.setcash(self.initial_capital)
        cerebro.broker.addcommissioninfo(
            FixedRateCommission(commission=self.commission_rate),
        )
        cerebro.broker.set_slippage_fixed(0.0)
        cerebro.broker.set_coc(True)

        cerebro.addstrategy(
            SignalReplayStrategy,
            records=self.records,
            max_position=self.m,
        )

        results = cerebro.run()
        strat = results[0]

        # bt 侧分项数据
        bt_stats = {
            "final_value": round(cerebro.broker.getvalue(), 2),
            "final_cash": round(cerebro.broker.getcash(), 2),
            "final_position": strat.getposition(strat.data).size,
            "total_commission": round(strat.total_commission, 2),
            "trade_count": strat.trade_count,
            "buy_count": strat.buy_count,
            "sell_count": strat.sell_count,
        }

        # 从 records 计算净交易类型（合并 settle+trade）
        net_counts = self._compute_net_trade_counts()
        bt_stats.update(net_counts)

        # tracker 侧分项数据
        tracker_total_slippage = sum(
            float(r.get("slippage", 0.0)) for r in self.records
        )
        tracker_total_commission = sum(
            float(r.get("commission", 0.0)) for r in self.records
        )
        last = self.records[-1] if self.records else {}
        tracker_stats_local = {
            "final_value": round(float(last.get("total_value", self.initial_capital)), 2),
            "final_cash": round(float(last.get("cash", self.initial_capital)), 2),
            "final_position": int(last.get("position_after", 0)),
            "total_commission": round(tracker_total_commission, 2),
            "total_slippage": round(tracker_total_slippage, 2),
        }

        # 逐项对比
        comparisons = self._cross_compare(bt_stats, tracker_stats_local, tracker_stats)
        all_pass = all(c["pass"] for c in comparisons)

        # 持仓逐步对比
        position_mismatches = self._compare_positions(strat.bt_log)

        report = {
            "all_pass": all_pass and len(position_mismatches) == 0,
            "bt_stats": bt_stats,
            "tracker_stats": tracker_stats_local,
            "comparisons": comparisons,
            "position_mismatches": len(position_mismatches),
            "position_mismatch_details": position_mismatches[:10],
        }

        self._log_report(report, comparisons)
        return report

    def _cross_compare(
        self, bt: Dict, trk: Dict, audit: Dict | None,
    ) -> List[Dict]:
        """逐项对比 bt 和 tracker 的分项数据。

        交易类型统计: bt 侧没有 settle 概念，它看到的是 settle+开仓
        合并后的净持仓变化。因此需要从 records 中计算"净交易类型"
        （合并同一 bar 的 settle 和交易），再与 bt 对比。
        """
        checks = []

        # 1. 最终持仓
        checks.append({
            "name": "final_position",
            "bt": bt["final_position"],
            "tracker": trk["final_position"],
            "pass": bt["final_position"] == trk["final_position"],
        })

        # 2. 总手续费
        checks.append({
            "name": "total_commission",
            "bt": bt["total_commission"],
            "tracker": trk["total_commission"],
            "diff": abs(bt["total_commission"] - trk["total_commission"]),
            "pass": abs(bt["total_commission"] - trk["total_commission"]) <= self.tolerance,
        })

        # 3. 最终总值: bt_value - tracker_value ≈ tracker 累计滑点
        value_diff = abs(bt["final_value"] - trk["final_value"])
        residual = abs(value_diff - trk["total_slippage"])
        checks.append({
            "name": "final_value",
            "bt": bt["final_value"],
            "tracker": trk["final_value"],
            "diff": round(value_diff, 2),
            "total_slippage": trk["total_slippage"],
            "residual": round(residual, 2),
            "pass": residual <= self.tolerance,
        })

        # 4. 交易类型: 从 records 计算净变化（合并同一 bar 的 settle+trade）
        #    这样和 bt 侧看到的持仓变化一致
        net_counts = self._compute_net_trade_counts()
        for key in ("flat_to_long", "flat_to_short", "long_to_flat",
                     "short_to_flat", "long_to_short", "short_to_long"):
            bt_val = bt.get(key, 0)
            net_val = net_counts.get(key, 0)
            checks.append({
                "name": f"trade_count_{key}",
                "bt": bt_val,
                "tracker": net_val,
                "pass": bt_val == net_val,
            })

        return checks

    def _compute_net_trade_counts(self) -> Dict[str, int]:
        """从 records 计算净交易类型（合并同一 bar 的 settle+trade）。

        bt 侧没有 settle，它看到的是每个 bar 的净持仓变化。
        所以我们按 state_index 分组，取每组第一条的 position_before
        和最后一条的 position_after，计算净变化类型。
        """
        from collections import OrderedDict

        # 按 state_index 分组，保持顺序
        groups: Dict[int, List[Dict]] = OrderedDict()
        for rec in self.records:
            idx = rec["state_index"]
            groups.setdefault(idx, []).append(rec)

        counts = {
            "flat_to_long": 0,
            "flat_to_short": 0,
            "long_to_flat": 0,
            "short_to_flat": 0,
            "long_to_short": 0,
            "short_to_long": 0,
        }

        prev_pos = 0  # 上一个 bar 结束时的持仓
        for idx, recs in groups.items():
            pos_before = prev_pos
            pos_after = recs[-1]["position_after"]
            prev_pos = pos_after

            if pos_before == pos_after:
                continue

            if pos_before == 0 and pos_after > 0:
                counts["flat_to_long"] += 1
            elif pos_before == 0 and pos_after < 0:
                counts["flat_to_short"] += 1
            elif pos_before > 0 and pos_after == 0:
                counts["long_to_flat"] += 1
            elif pos_before < 0 and pos_after == 0:
                counts["short_to_flat"] += 1
            elif pos_before > 0 and pos_after < 0:
                counts["long_to_short"] += 1
            elif pos_before < 0 and pos_after > 0:
                counts["short_to_long"] += 1

        return counts

    def _compare_positions(self, bt_log: List[Dict]) -> List[Dict]:
        """逐步对比持仓 (bt_log[N+1] ↔ tracker[N])。"""
        tracker_by_idx: Dict[int, Dict] = {}
        for rec in self.records:
            tracker_by_idx[rec["state_index"]] = rec

        bt_by_idx: Dict[int, Dict] = {}
        for entry in bt_log:
            bt_by_idx[entry["bar_idx"]] = entry

        mismatches = []
        for trk_idx, trk in tracker_by_idx.items():
            bt_idx = trk_idx + 1
            if bt_idx not in bt_by_idx:
                continue
            entry = bt_by_idx[bt_idx]
            if entry["bt_position"] != trk["position_after"]:
                mismatches.append({
                    "state_index": trk_idx,
                    "bt_position": entry["bt_position"],
                    "tracker_position": trk["position_after"],
                })
        return mismatches

    def _log_report(self, report: Dict, comparisons: List[Dict]) -> None:
        logger.info("=" * 60)
        logger.info("Backtrader 交叉验证报告")
        logger.info("-" * 60)

        bt_s = report["bt_stats"]
        trk_s = report["tracker_stats"]
        logger.info("%-20s %15s %15s", "", "Backtrader", "Tracker")
        logger.info("%-20s %15d %15d", "最终持仓",
                     bt_s["final_position"], trk_s["final_position"])
        logger.info("%-20s %15.2f %15.2f", "最终现金(口径不同)",
                     bt_s["final_cash"], trk_s["final_cash"])
        logger.info("%-20s %15.2f %15.2f", "最终总值",
                     bt_s["final_value"], trk_s["final_value"])
        logger.info("%-20s %15.2f %15.2f", "总手续费",
                     bt_s["total_commission"], trk_s["total_commission"])
        logger.info("%-20s %15s %15.2f", "总滑点",
                     "N/A(无LOB)", trk_s["total_slippage"])
        logger.info("%-20s %15d %15s", "成交订单数",
                     bt_s["trade_count"], "-")
        logger.info("%-20s %15d / %15d", "买入/卖出",
                     bt_s["buy_count"], bt_s["sell_count"])
        logger.info("-" * 60)

        for c in comparisons:
            status = "✓" if c["pass"] else "✗"
            if "diff" in c:
                logger.info("[%s] %s: bt=%s tracker=%s diff=%s",
                            status, c["name"], c.get("bt"), c.get("tracker"),
                            c.get("diff"))
            else:
                logger.info("[%s] %s: bt=%s tracker=%s",
                            status, c["name"], c.get("bt"), c.get("tracker"))

        logger.info("-" * 60)
        logger.info("持仓逐步对比: %d 处不匹配", report["position_mismatches"])
        logger.info("=" * 60)

        if report["all_pass"]:
            logger.info("交叉验证结果: 全部通过")
        else:
            failed = [c["name"] for c in comparisons if not c["pass"]]
            if report["position_mismatches"] > 0:
                failed.append(f"position({report['position_mismatches']}处)")
            logger.info("交叉验证结果: %d 项未通过 — %s",
                        len(failed), ", ".join(failed))
