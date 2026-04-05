"""PortfolioTracker — 跨 horizon 资金与持仓管理

将 evaluate.py 中 run_horizon_inference 里的现金管理、持仓跟踪、
盈亏计算、操作记录生成等逻辑抽取为独立可测试的类。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from src.env.trading_env import TradingEnv
from src.utils.logger import get_logger

logger = get_logger(__name__)

ACTION_LABELS: Dict[int, str] = {0: "short", 1: "flat", 2: "long"}


@dataclass
class PortfolioState:
    """跨 horizon 传递的组合状态快照。"""

    cash: float = 0.0
    position: int = 0
    short_open_value: float = 0.0
    avg_hold_price: float = 0.0


@dataclass
class StepRecord:
    """单步操作记录，用于 CSV 导出。"""

    state_index: int = 0
    action: object = None  # int 或 "settle"
    action_label: str = ""
    execution_price: float = 0.0
    trade_quantity: int = 0
    position_after: int = 0
    avg_hold_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    position_change_pnl: float = 0.0
    cash: float = 0.0
    holding_value: float = 0.0
    short_debt: float = 0.0
    total_value: float = 0.0
    profit: float = 0.0
    side: str = ""

    def to_dict(self) -> Dict:
        return {
            "state_index": self.state_index,
            "action": self.action,
            "action_label": self.action_label,
            "execution_price": round(self.execution_price, 6),
            "trade_quantity": self.trade_quantity,
            "position_after": self.position_after,
            "avg_hold_price": round(self.avg_hold_price, 6),
            "commission": abs(round(self.commission, 6)),
            "slippage": round(self.slippage, 6),
            "position_change_pnl": round(self.position_change_pnl, 6),
            "cash": round(self.cash, 6),
            "holding_value": round(self.holding_value, 6),
            "short_debt": round(self.short_debt, 6),
            "total_value": round(self.total_value, 6),
            "profit": round(self.profit, 6),
            "side": self.side,
        }


class PortfolioTracker:
    """管理跨 horizon 的资金、持仓、盈亏计算和操作记录。

    将原先散落在 run_horizon_inference 中的 ~150 行逻辑集中管理，
    每个方法职责单一，便于单元测试。

    Args:
        initial_capital: 初始资金
    """

    def __init__(self, initial_capital: float) -> None:
        self.initial_capital = initial_capital
        self.state = PortfolioState(cash=initial_capital)
        self.records: List[Dict] = []

    # ------------------------------------------------------------------
    # 跨 horizon 平仓清算
    # ------------------------------------------------------------------

    def settle_previous_horizon(
        self, price: float, t_start: int,
        new_first_action: int | None = None,
        m: int = 0,
        commission_rate: float = 0.0,
        slippage: float = 0.0,
    ) -> None:
        """在新 horizon 开始时，根据新 horizon 首个动作决定是否平仓。

        优化逻辑:
        - 同方向延续 (多→多, 空→空): 跳过平仓，持仓直接延续到新 horizon
        - 方向改变 (多→空, 空→多, 多→空仓, 空→空仓): 执行平仓并收取手续费+滑点
        - 无持仓: 无操作

        Args:
            price: 新 horizon 的开盘价
            t_start: 新 horizon 的起始全局索引
            new_first_action: 新 horizon 的第一个动作 {0: short, 1: flat, 2: long}，
                              None 时退化为强制平仓（向后兼容）
            m: 最大持仓量，用于判断新方向
            commission_rate: 佣金率，用于计算平仓手续费
            slippage: 平仓滑点成本（由调用方通过 LOB 计算）
        """
        s = self.state
        prev_position = s.position

        if prev_position == 0:
            return

        # 判断新 horizon 首个动作的目标方向
        if new_first_action is not None:
            direction_map = {0: -1, 1: 0, 2: 1}
            new_direction = direction_map[new_first_action]
            prev_direction = 1 if prev_position > 0 else -1

            # 同方向延续: 跳过平仓，持仓直接 carry forward
            if new_direction == prev_direction:
                logger.debug(
                    "[SETTLE] t=%d 同方向延续 (pos=%d, new_action=%d)，跳过平仓",
                    t_start, prev_position, new_first_action,
                )
                return

        # 需要平仓: 计算手续费
        abs_position = abs(prev_position)
        commission = round(commission_rate * abs_position * price, 2)
        total_cost = commission + slippage

        if prev_position > 0:
            # 遗留多头: 按开盘价卖出，收回现金，扣除手续费+滑点
            s.cash += prev_position * price - total_cost
            pnl = (price - s.avg_hold_price) * abs_position if s.avg_hold_price > 0 else 0.0
            total_value = s.cash
            self.records.append(StepRecord(
                state_index=t_start,
                action="settle",
                action_label="horizon平仓",
                execution_price=price,
                trade_quantity=-prev_position,
                position_after=0,
                avg_hold_price=0.0,
                commission=commission,
                slippage=slippage,
                position_change_pnl=pnl,
                cash=s.cash,
                total_value=total_value,
                profit=total_value - self.initial_capital,
                side="空仓",
            ).to_dict())
        else:
            # 遗留空头: 按开盘价买回平仓，盈亏结算到 cash，扣除手续费+滑点
            close_debt = abs_position * price
            pnl = s.short_open_value - close_debt
            logger.debug(
                "[SETTLE] 空头平仓前: cash=%.2f sov=%.2f close_debt=%.2f pnl=%.2f cost=%.2f",
                s.cash, s.short_open_value, close_debt, pnl, total_cost,
            )
            s.cash += pnl - total_cost
            s.short_open_value = 0.0
            logger.debug("[SETTLE] 空头平仓后: cash=%.2f", s.cash)
            total_value = s.cash
            self.records.append(StepRecord(
                state_index=t_start,
                action="settle",
                action_label="horizon平仓",
                execution_price=price,
                trade_quantity=abs_position,
                position_after=0,
                avg_hold_price=0.0,
                commission=commission,
                slippage=slippage,
                position_change_pnl=pnl,
                cash=s.cash,
                total_value=total_value,
                profit=total_value - self.initial_capital,
                side="空仓",
            ).to_dict())

        s.position = 0
        s.avg_hold_price = 0.0

    # ------------------------------------------------------------------
    # 单步资金更新
    # ------------------------------------------------------------------

    def update_cash_for_trade(
        self, old_position: int, new_position: int, exec_price: float, t: int,
    ) -> None:
        """根据持仓变动更新 cash 和 short_open_value。"""
        s = self.state
        delta = new_position - old_position

        if delta == 0:
            return

        if old_position >= 0 and new_position >= 0:
            s.cash -= delta * exec_price
            logger.debug(
                "[CASH] t=%d 多头侧: old=%d new=%d delta=%d cash=%.2f",
                t, old_position, new_position, delta, s.cash,
            )
        elif old_position > 0 and new_position < 0:
            s.cash += old_position * exec_price
            s.short_open_value = abs(new_position) * exec_price
            logger.debug(
                "[CASH] t=%d 多头→空头: old=%d new=%d cash=%.2f sov=%.2f",
                t, old_position, new_position, s.cash, s.short_open_value,
            )
        elif old_position < 0 and new_position > 0:
            close_debt = abs(old_position) * exec_price
            s.cash += s.short_open_value - close_debt
            s.short_open_value = 0.0
            s.cash -= new_position * exec_price
            logger.debug(
                "[CASH] t=%d 空头→多头: old=%d new=%d cash=%.2f",
                t, old_position, new_position, s.cash,
            )
        elif old_position < 0 and new_position == 0:
            close_debt = abs(old_position) * exec_price
            s.cash += s.short_open_value - close_debt
            s.short_open_value = 0.0
            logger.debug(
                "[CASH] t=%d 空头→空仓: old=%d new=%d cash=%.2f",
                t, old_position, new_position, s.cash,
            )
        elif old_position == 0 and new_position < 0:
            s.short_open_value = abs(new_position) * exec_price
            logger.debug(
                "[CASH] t=%d 空仓→空头: old=%d new=%d cash=%.2f sov=%.2f",
                t, old_position, new_position, s.cash, s.short_open_value,
            )
        elif old_position < 0 and new_position < 0:
            if abs(new_position) > abs(old_position):
                added_qty = abs(new_position) - abs(old_position)
                s.short_open_value += added_qty * exec_price
            elif abs(new_position) < abs(old_position):
                closed_qty = abs(old_position) - abs(new_position)
                avg_open = s.short_open_value / abs(old_position)
                s.cash += closed_qty * avg_open - closed_qty * exec_price
                s.short_open_value -= closed_qty * avg_open
            logger.debug(
                "[CASH] t=%d 空头内部: old=%d new=%d cash=%.2f sov=%.2f",
                t, old_position, new_position, s.cash, s.short_open_value,
            )

    # ------------------------------------------------------------------
    # 盈亏与均价
    # ------------------------------------------------------------------

    @staticmethod
    def compute_position_change_pnl(
        old_position: int,
        new_position: int,
        exec_price: float,
        avg_hold_price: float,
    ) -> float:
        """计算换仓时平仓部分的盈亏。"""
        if old_position > 0 and new_position <= 0:
            return (exec_price - avg_hold_price) * abs(old_position)
        if old_position < 0 and new_position >= 0:
            return (avg_hold_price - exec_price) * abs(old_position)
        if old_position > 0 and new_position > 0 and abs(new_position) < abs(old_position):
            closed_qty = abs(old_position) - abs(new_position)
            return (exec_price - avg_hold_price) * closed_qty
        if old_position < 0 and new_position < 0 and abs(new_position) < abs(old_position):
            closed_qty = abs(old_position) - abs(new_position)
            return (avg_hold_price - exec_price) * closed_qty
        return 0.0

    @staticmethod
    def update_avg_hold_price(
        old_position: int,
        new_position: int,
        exec_price: float,
        avg_hold_price: float,
    ) -> float:
        """根据持仓变动更新平均持仓价格。"""
        if new_position == 0:
            return 0.0
        if old_position == 0:
            return exec_price
        if np.sign(new_position) != np.sign(old_position):
            return exec_price
        delta = abs(new_position) - abs(old_position)
        if delta > 0:
            total_cost = avg_hold_price * abs(old_position) + exec_price * delta
            return total_cost / abs(new_position)
        # 减仓: 平均价格不变
        return avg_hold_price

    # ------------------------------------------------------------------
    # 总资产计算
    # ------------------------------------------------------------------

    def compute_total_value(
        self, new_position: int, exec_price: float,
    ) -> tuple[float, float, float, float]:
        """计算持仓价值、做空负债、总资产、累计利润。

        Returns:
            (holding_value, short_debt, total_value, profit)
        """
        s = self.state
        holding_value = max(new_position, 0) * exec_price
        short_debt = abs(min(new_position, 0)) * exec_price
        short_unrealized_pnl = (
            s.short_open_value - short_debt if new_position < 0 else 0.0
        )
        total_value = s.cash + holding_value + short_unrealized_pnl
        profit = total_value - self.initial_capital
        return holding_value, short_debt, total_value, profit

    # ------------------------------------------------------------------
    # 记录生成
    # ------------------------------------------------------------------

    def record_step(
        self,
        t: int,
        a_final: int,
        exec_price: float,
        old_position: int,
        new_position: int,
        commission: float,
        slippage: float,
    ) -> None:
        """处理单步的盈亏、均价更新，并生成操作记录。

        手续费 (commission) 和滑点 (slippage) 会从 cash 中扣除，
        确保 cash / total_value / profit 正确反映交易成本。
        """
        s = self.state

        pnl = self.compute_position_change_pnl(
            old_position, new_position, exec_price, s.avg_hold_price,
        )
        s.avg_hold_price = self.update_avg_hold_price(
            old_position, new_position, exec_price, s.avg_hold_price,
        )
        s.position = new_position

        # 从 cash 中扣除交易费用
        total_cost = commission + slippage
        if total_cost > 0:
            s.cash -= total_cost
            logger.debug(
                "[FEE] t=%d commission=%.6f slippage=%.6f total=%.6f cash=%.2f",
                t, commission, slippage, total_cost, s.cash,
            )

        if new_position > 0:
            side = "多头"
        elif new_position < 0:
            side = "空头"
        else:
            side = "空仓"

        holding_value, short_debt, total_value, profit = self.compute_total_value(
            new_position, exec_price,
        )

        self.records.append(StepRecord(
            state_index=t,
            action=a_final,
            action_label=ACTION_LABELS[a_final],
            execution_price=exec_price,
            trade_quantity=new_position - old_position,
            position_after=new_position,
            avg_hold_price=s.avg_hold_price,
            commission=commission,
            slippage=slippage,
            position_change_pnl=pnl,
            cash=s.cash,
            holding_value=holding_value,
            short_debt=short_debt,
            total_value=total_value,
            profit=profit,
            side=side,
        ).to_dict())
