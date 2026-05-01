"""盘口逐档成交、手续费、滑点与未成交（reject_transition）处理.

设计文档锚点: §4.10 与 §5.3。

实现要点:
- 当 ``delta=target-prev > 0``: 走 ask 档（买入或减空仓需要在卖盘吃单）。
- 当 ``delta < 0``: 走 bid 档（卖出或减多仓需要在买盘吃单）。
- 五档累计深度不足时按 ``reject_transition`` 处理: 上层（DP/replay）保持 ``prev_position``。
- ``mark_price`` 默认是 ``(ask1 + bid1) / 2``，避免选择一侧带方向偏。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass(frozen=True)
class ExecutionResult:
    """单次成交结果。

    Attributes
    ----------
    prev_position : 成交前持仓（论文动作映射后的整数仓位）。
    target_position : 成交后目标持仓。
    filled_position : 实际成交后持仓；当深度不足且 ``reject_transition`` 时
                      等于 ``prev_position``。
    fill_price : 加权成交均价。``None`` 表示无换仓（``prev == target``）或被 reject。
    fee : 手续费（绝对值，单位与 mark price 一致）。
    slippage : 滑点（绝对值）。
    cost : ``fee + slippage``，等价于论文中的 ``O_t``。
    rejected : 是否触发 ``insufficient_depth_policy=reject_transition``。
    reject_reason : 被拒原因；用于 ``phase1_report.json`` 与 failure case 诊断。
    """
    prev_position: int
    target_position: int
    filled_position: int
    fill_price: Optional[float]
    fee: float
    slippage: float
    cost: float
    rejected: bool
    reject_reason: Optional[str] = None


@dataclass(frozen=True)
class ExecutionBook:
    """单步盘口快照。

    Attributes
    ----------
    ask_prices : 升序的 ask 五档价格（``ask1 < ask2 < ...``）。
    ask_sizes : 与 ask_prices 对齐的可成交数量。
    bid_prices : 降序的 bid 五档价格（``bid1 > bid2 > ...``）。
    bid_sizes : 与 bid_prices 对齐的可成交数量。
    mark_price : 中价（mid）；用于 reward markout 与 slippage 估算。

    Notes
    -----
    数据由 ``HorizonBuilder`` 切出，本模块不直接读文件。
    某档 size <= 0 视为缺失，会被逐档累加流程跳过。
    """
    ask_prices: Sequence[float]
    ask_sizes: Sequence[float]
    bid_prices: Sequence[float]
    bid_sizes: Sequence[float]
    mark_price: float


class LobDepthCostModel:
    """基于 limit order book 五档逐档成交的成本模型。

    职责:
    - 根据 ``prev_position -> target_position`` 决定走 ask 还是 bid。
    - 逐档累加直到填满目标量；不足则按 ``reject_transition`` 处理。
    - 计算 fee（按论文 ``δ=0.02%``）与 slippage（``|p_fill - p_mark|``）。

    边界:
    - 不知道 reward 公式（由调用方组合 ``markout - exec`` 价差）。
    - 不引用未来行；调用方负责传入对应 ``execution_row`` 的 book。
    """

    def __init__(
        self,
        commission_rate: float,
        book_levels: int = 5,
        insufficient_depth_policy: str = "reject_transition",
        slippage_multiplier: float = 1.0,
    ) -> None:
        self.commission_rate = commission_rate
        self.book_levels = book_levels
        self.slippage_multiplier = float(slippage_multiplier)
        if insufficient_depth_policy not in ("reject_transition",):
            raise ValueError(
                f"目前只支持 reject_transition; got {insufficient_depth_policy!r}"
            )
        self.insufficient_depth_policy = insufficient_depth_policy

    def execute(
        self,
        *,
        prev_position: int,
        target_position: int,
        execution_book: ExecutionBook,
    ) -> ExecutionResult:
        """根据目标仓位与盘口估算成交。

        实现要点
        --------
        1. ``delta = target - prev``；``delta == 0`` 时直接返回零成本结果（无换仓）。
        2. ``delta > 0``（增多/减空 → 在卖盘吃单）走 ask 档；``delta < 0`` 走 bid 档。
        3. 逐档累加可成交量直到 ``filled_qty >= |delta|``；
           若五档总深度仍不足 → 按 ``reject_transition`` 返回，``filled_position = prev``。
        4. ``fill_price = Σ q_l * p_l / Σ q_l``。
        5. ``fee = commission_rate * |Δ| * mark_price``；
           ``slippage = |Δ| * |fill_price - mark_price|``。

        Parameters
        ----------
        prev_position, target_position : 整数仓位（已乘 ``max_position``）。
        execution_book : 当前 ``execution_row`` 的盘口快照。

        Returns
        -------
        ExecutionResult
            ``rejected=True`` 时上层（DP / replay）应跳过本次换仓，
            保持 ``prev_position``，并通过 ``reject_event`` 记录到统计中。
        """
        delta = target_position - prev_position
        # 不需要换仓 → 零成本路径，避免对盘口做无效遍历。
        if delta == 0:
            return ExecutionResult(
                prev_position=prev_position,
                target_position=target_position,
                filled_position=prev_position,
                fill_price=None,
                fee=0.0,
                slippage=0.0,
                cost=0.0,
                rejected=False,
            )

        # 选择档位侧。
        if delta > 0:
            prices = list(execution_book.ask_prices)[: self.book_levels]
            sizes = list(execution_book.ask_sizes)[: self.book_levels]
            side = "ask"
        else:
            prices = list(execution_book.bid_prices)[: self.book_levels]
            sizes = list(execution_book.bid_sizes)[: self.book_levels]
            side = "bid"

        need = abs(delta)
        # 逐档累加，得到加权均价。
        filled_qty = 0.0
        weighted_price_sum = 0.0
        for level_price, level_size in zip(prices, sizes):
            if level_size <= 0:
                continue
            take = min(need - filled_qty, float(level_size))
            if take <= 0:
                break
            weighted_price_sum += take * float(level_price)
            filled_qty += take
            if filled_qty >= need:
                break

        # 五档总深度不够 → reject。
        if filled_qty + 1e-9 < need:
            return ExecutionResult(
                prev_position=prev_position,
                target_position=target_position,
                filled_position=prev_position,
                fill_price=None,
                fee=0.0,
                slippage=0.0,
                cost=0.0,
                rejected=True,
                reject_reason=f"insufficient_{side}_depth: need={need} got={filled_qty:.4f}",
            )

        fill_price = weighted_price_sum / filled_qty
        mark = float(execution_book.mark_price)
        # fee = δ * |Δ| * mark_price（按论文）。
        fee = self.commission_rate * need * mark
        # slippage = |Δ| * |fill - mark|（fill 与 mark 越远滑点越大）。
        slippage = need * abs(fill_price - mark) * self.slippage_multiplier
        cost = fee + slippage
        return ExecutionResult(
            prev_position=prev_position,
            target_position=target_position,
            filled_position=target_position,
            fill_price=fill_price,
            fee=fee,
            slippage=slippage,
            cost=cost,
            rejected=False,
        )
