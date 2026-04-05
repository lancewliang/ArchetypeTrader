"""交易记录审计模块

从 PortfolioTracker.records 中统计详细的交易摘要，
用于自验证资金计算的正确性。不依赖外部回测框架。

用法:
    from src.evaluation.trade_auditor import TradeAuditor
    auditor = TradeAuditor(tracker.records, initial_capital)
    report = auditor.audit()
"""

from __future__ import annotations

from typing import Dict, List

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TradeAuditor:
    """从 tracker records 统计详细交易摘要并验证资金一致性。

    Args:
        records: PortfolioTracker.records
        initial_capital: 初始资金
    """

    def __init__(self, records: List[Dict], initial_capital: float) -> None:
        self.records = records
        self.initial_capital = initial_capital

    def audit(self) -> Dict:
        """执行完整审计，返回详细报告。"""
        stats = self._compute_statistics()
        checks = self._run_checks(stats)
        self._log_report(stats, checks)
        return {"statistics": stats, "checks": checks}

    # ------------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------------

    def _compute_statistics(self) -> Dict:
        """从 records 中统计所有交易细节。"""
        total_commission = 0.0
        total_slippage = 0.0
        total_records = len(self.records)

        # 交易次数统计
        trade_counts = {
            "flat_to_long": 0,      # 空仓→多头
            "flat_to_short": 0,     # 空仓→空头
            "long_to_flat": 0,      # 多头→空仓
            "short_to_flat": 0,     # 空头→空仓
            "long_to_short": 0,     # 多→空 (换仓)
            "short_to_long": 0,     # 空→多 (换仓)
            "hold_long": 0,         # 持多不变
            "hold_short": 0,        # 持空不变
            "hold_flat": 0,         # 持空仓不变
            "settle_long": 0,       # horizon 平仓 (多头)
            "settle_short": 0,      # horizon 平仓 (空头)
        }

        # 盈亏统计
        realized_pnl = 0.0
        winning_trades = 0
        losing_trades = 0

        for rec in self.records:
            commission = float(rec.get("commission", 0.0))
            slippage = float(rec.get("slippage", 0.0))
            total_commission += commission
            total_slippage += slippage

            pnl = float(rec.get("position_change_pnl", 0.0))
            realized_pnl += pnl
            if pnl > 0:
                winning_trades += 1
            elif pnl < 0:
                losing_trades += 1

            action = rec.get("action")
            qty = int(rec.get("trade_quantity", 0))
            pos_after = int(rec.get("position_after", 0))

            if action == "settle":
                if qty < 0:
                    trade_counts["settle_long"] += 1
                elif qty > 0:
                    trade_counts["settle_short"] += 1
                continue

            # 根据 trade_quantity 和 position_after 推断交易类型
            if qty == 0:
                if pos_after > 0:
                    trade_counts["hold_long"] += 1
                elif pos_after < 0:
                    trade_counts["hold_short"] += 1
                else:
                    trade_counts["hold_flat"] += 1
            else:
                pos_before = pos_after - qty
                if pos_before == 0 and pos_after > 0:
                    trade_counts["flat_to_long"] += 1
                elif pos_before == 0 and pos_after < 0:
                    trade_counts["flat_to_short"] += 1
                elif pos_before > 0 and pos_after == 0:
                    trade_counts["long_to_flat"] += 1
                elif pos_before < 0 and pos_after == 0:
                    trade_counts["short_to_flat"] += 1
                elif pos_before > 0 and pos_after < 0:
                    trade_counts["long_to_short"] += 1
                elif pos_before < 0 and pos_after > 0:
                    trade_counts["short_to_long"] += 1

        # 最终状态
        last = self.records[-1] if self.records else {}
        final_cash = float(last.get("cash", self.initial_capital))
        final_value = float(last.get("total_value", self.initial_capital))
        final_position = int(last.get("position_after", 0))
        final_profit = float(last.get("profit", 0.0))

        return {
            "total_records": total_records,
            "total_commission": round(total_commission, 2),
            "total_slippage": round(total_slippage, 2),
            "total_cost": round(total_commission + total_slippage, 2),
            "realized_pnl": round(realized_pnl, 2),
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "trade_counts": trade_counts,
            "final_cash": round(final_cash, 2),
            "final_value": round(final_value, 2),
            "final_position": final_position,
            "final_profit": round(final_profit, 2),
            "initial_capital": round(self.initial_capital, 2),
        }

    # ------------------------------------------------------------------
    # 一致性检查
    # ------------------------------------------------------------------

    def _run_checks(self, stats: Dict) -> List[Dict]:
        """执行一系列一致性检查。"""
        checks = []

        # 1. profit = final_value - initial_capital
        expected_profit = round(stats["final_value"] - stats["initial_capital"], 2)
        checks.append({
            "name": "profit_consistency",
            "description": "profit == final_value - initial_capital",
            "expected": expected_profit,
            "actual": stats["final_profit"],
            "pass": abs(expected_profit - stats["final_profit"]) < 0.01,
        })

        # 2. 交易次数平衡: 开仓次数 ≈ 平仓次数 (允许最后一笔未平仓)
        tc = stats["trade_counts"]
        total_opens = (tc["flat_to_long"] + tc["flat_to_short"]
                       + tc["long_to_short"] + tc["short_to_long"])
        total_closes = (tc["long_to_flat"] + tc["short_to_flat"]
                        + tc["long_to_short"] + tc["short_to_long"]
                        + tc["settle_long"] + tc["settle_short"])
        # 换仓同时算开仓和平仓，所以 opens 和 closes 都包含换仓
        checks.append({
            "name": "trade_balance",
            "description": "开仓次数与平仓次数平衡 (差≤1)",
            "total_opens": total_opens,
            "total_closes": total_closes,
            "diff": abs(total_opens - total_closes),
            "pass": abs(total_opens - total_closes) <= 1,
        })

        # 3. 记录总数 = 持仓不变 + 有交易 + settle
        total_holds = tc["hold_long"] + tc["hold_short"] + tc["hold_flat"]
        total_trades = (tc["flat_to_long"] + tc["flat_to_short"]
                        + tc["long_to_flat"] + tc["short_to_flat"]
                        + tc["long_to_short"] + tc["short_to_long"])
        total_settles = tc["settle_long"] + tc["settle_short"]
        expected_total = total_holds + total_trades + total_settles
        checks.append({
            "name": "record_count",
            "description": "记录总数 = 持仓不变 + 交易 + settle",
            "expected": expected_total,
            "actual": stats["total_records"],
            "pass": expected_total == stats["total_records"],
        })

        # 4. 最终持仓为 0 时，cash == total_value
        if stats["final_position"] == 0:
            checks.append({
                "name": "flat_cash_equals_value",
                "description": "最终空仓时 cash == total_value",
                "cash": stats["final_cash"],
                "value": stats["final_value"],
                "pass": abs(stats["final_cash"] - stats["final_value"]) < 0.01,
            })

        # 5. commission 和 slippage 非负
        checks.append({
            "name": "non_negative_costs",
            "description": "总手续费和总滑点均 ≥ 0",
            "total_commission": stats["total_commission"],
            "total_slippage": stats["total_slippage"],
            "pass": stats["total_commission"] >= 0 and stats["total_slippage"] >= 0,
        })

        return checks

    # ------------------------------------------------------------------
    # 日志输出
    # ------------------------------------------------------------------

    def _log_report(self, stats: Dict, checks: List[Dict]) -> None:
        tc = stats["trade_counts"]
        logger.info("=" * 60)
        logger.info("交易审计报告")
        logger.info("-" * 60)
        logger.info("总记录数:       %d", stats["total_records"])
        logger.info("初始资金:       %.2f", stats["initial_capital"])
        logger.info("最终现金:       %.2f", stats["final_cash"])
        logger.info("最终总值:       %.2f", stats["final_value"])
        logger.info("最终持仓:       %d", stats["final_position"])
        logger.info("最终利润:       %.2f", stats["final_profit"])
        logger.info("-" * 60)
        logger.info("总手续费:       %.2f", stats["total_commission"])
        logger.info("总滑点:         %.2f", stats["total_slippage"])
        logger.info("总费用:         %.2f", stats["total_cost"])
        logger.info("已实现盈亏:     %.2f", stats["realized_pnl"])
        logger.info("盈利交易:       %d", stats["winning_trades"])
        logger.info("亏损交易:       %d", stats["losing_trades"])
        logger.info("-" * 60)
        logger.info("空仓→多头:      %d", tc["flat_to_long"])
        logger.info("空仓→空头:      %d", tc["flat_to_short"])
        logger.info("多头→空仓:      %d", tc["long_to_flat"])
        logger.info("空头→空仓:      %d", tc["short_to_flat"])
        logger.info("多头→空头:      %d", tc["long_to_short"])
        logger.info("空头→多头:      %d", tc["short_to_long"])
        logger.info("持多不变:       %d", tc["hold_long"])
        logger.info("持空不变:       %d", tc["hold_short"])
        logger.info("持空仓不变:     %d", tc["hold_flat"])
        logger.info("horizon平仓多:  %d", tc["settle_long"])
        logger.info("horizon平仓空:  %d", tc["settle_short"])
        logger.info("-" * 60)

        all_pass = all(c["pass"] for c in checks)
        for c in checks:
            status = "✓" if c["pass"] else "✗"
            logger.info("[%s] %s: %s", status, c["name"], c["description"])

        logger.info("=" * 60)
        if all_pass:
            logger.info("审计结果: 全部通过")
        else:
            failed = [c["name"] for c in checks if not c["pass"]]
            logger.info("审计结果: %d 项未通过 — %s", len(failed), ", ".join(failed))
