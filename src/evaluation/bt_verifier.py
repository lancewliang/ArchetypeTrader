"""Backtrader 交叉验证模块

将 PortfolioTracker 的交易记录回放到 backtrader 中，
逐步对比 cash / total_value / position，验证资金计算正确性。

backtrader 的订单默认在下一个 bar 成交。为了实现当 bar 成交，
使用 cerebro.broker.set_coc(True) (cheat-on-close) 让 Market 订单
在当前 bar 的 close 价成交。由于我们的 DataFeed 中 open=close=price，
这等价于在当前价格成交。

用法:
    from src.evaluation.bt_verifier import BacktraderVerifier
    verifier = BacktraderVerifier(tracker.records, prices, initial_capital, m,
                                  commission_rate)
    report = verifier.run()
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
# 自定义佣金: rate × |size| × price，保留 2 位小数
# ------------------------------------------------------------------

class FixedRateCommission(bt.CommInfoBase):
    params = (
        ("commission", 0.0002),
        ("stocklike", False),
        ("commtype", bt.CommInfoBase.COMM_PERC),
        ("percabs", True),
    )

    def _getcommission(self, size, price, pseudoexec):
        return round(abs(size) * price * self.p.commission, 2)


# ------------------------------------------------------------------
# 信号回放策略
# ------------------------------------------------------------------

class SignalReplayStrategy(bt.Strategy):
    """按 tracker records 逐步下单。

    使用 set_coc(True) 让订单在当前 bar 的 close 价成交。
    由于 open=high=low=close=price，等价于在当前价格成交。
    """

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

    def next(self):
        bar_idx = self._bar_count
        self._bar_count += 1

        # 下单: set_coc 让订单在当前 bar close 价成交
        if bar_idx in self._target_by_idx:
            target_pos = self._target_by_idx[bar_idx]
            current_pos = self.getposition(self.data).size
            delta = target_pos - current_pos
            if delta != 0:
                if delta > 0:
                    self.buy(size=abs(delta))
                else:
                    self.sell(size=abs(delta))

        # 记录（订单已在当前 bar 成交，因为 set_coc=True）
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
    """使用 backtrader 交叉验证 PortfolioTracker 的资金计算。

    Args:
        records: PortfolioTracker.records
        prices: 完整价格序列 np.ndarray
        initial_capital: 初始资金
        m: 最大持仓量
        commission_rate: 佣金率
        tolerance: 允许的 total_value 误差（默认 1.0）
    """

    def __init__(
        self,
        records: List[Dict],
        prices: np.ndarray,
        initial_capital: float,
        m: int,
        commission_rate: float = 0.0002,
        tolerance: float = 1.0,
    ) -> None:
        self.records = records
        self.prices = prices
        self.initial_capital = initial_capital
        self.m = m
        self.commission_rate = commission_rate
        self.tolerance = tolerance

    def run(self) -> Dict:
        """执行 backtrader 回测并与 tracker records 逐步对比。"""
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
        cerebro.broker.set_coc(True)  # cheat-on-close: 当 bar 成交

        cerebro.addstrategy(
            SignalReplayStrategy,
            records=self.records,
            max_position=self.m,
        )

        results = cerebro.run()
        strategy = results[0]

        bt_log = strategy.bt_log
        mismatches = self._compare(bt_log)

        bt_final = round(cerebro.broker.getvalue(), 6)
        tracker_final = (
            self.records[-1]["total_value"] if self.records else self.initial_capital
        )

        match = len(mismatches) == 0
        # 最终 total_value 对比（此时持仓为 0 或相同价格，应一致）
        final_value_diff = abs(bt_final - tracker_final)
        if final_value_diff > self.tolerance:
            match = False

        summary = (
            f"验证通过: {len(bt_log)} bars, 持仓一致, "
            f"最终价值差={final_value_diff:.2f}"
            if match
            else f"发现 {len(mismatches)} 处持仓差异, "
            f"最终价值差={final_value_diff:.2f} (共 {len(bt_log)} bars)"
        )

        report = {
            "match": match,
            "total_bars": len(bt_log),
            "mismatches": mismatches[:20],
            "bt_final_value": bt_final,
            "tracker_final_value": tracker_final,
            "summary": summary,
        }

        logger.info("[BT验证] %s", summary)
        logger.info(
            "[BT验证] bt_final=%.2f tracker_final=%.2f diff=%.2f",
            bt_final, tracker_final, abs(bt_final - tracker_final),
        )

        return report

    def _compare(self, bt_log: List[Dict]) -> List[Dict]:
        """将 bt_log 与 tracker records 按 state_index 对比。

        set_coc(True) 下，bar N 的订单在 next() 返回后成交，
        bt_log[N+1] 才能看到成交后的持仓。
        对比: tracker records[state_index=X] ↔ bt_log[bar_idx=X+1]

        注意: backtrader 期货模式每 bar 做 mark-to-market，cash 随价格
        变动。而 tracker 的 cash 只在交易时变动。因此中间步骤只对比持仓，
        最终 total_value 在 run() 中单独对比。
        """
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
            pos_match = entry["bt_position"] == trk["position_after"]

            if not pos_match:
                mismatches.append({
                    "state_index": trk_idx,
                    "bt_bar": bt_idx,
                    "bt_cash": entry["bt_cash"],
                    "bt_value": entry["bt_value"],
                    "bt_position": entry["bt_position"],
                    "tracker_cash": trk["cash"],
                    "tracker_value": trk["total_value"],
                    "tracker_position": trk["position_after"],
                    "value_diff": abs(entry["bt_value"] - trk["total_value"]),
                    "pos_match": pos_match,
                })

        return mismatches
