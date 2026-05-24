"""交易动作执行收益计算工具。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LOB_ASK_PRICE_COLS = ["ask1_price", "ask2_price", "ask3_price", "ask4_price", "ask5_price"]
LOB_ASK_SIZE_COLS = ["ask1_size", "ask2_size", "ask3_size", "ask4_size", "ask5_size"]
LOB_BID_PRICE_COLS = ["bid1_price", "bid2_price", "bid3_price", "bid4_price", "bid5_price"]
LOB_BID_SIZE_COLS = ["bid1_size", "bid2_size", "bid3_size", "bid4_size", "bid5_size"]
LOB_DEPTH_WIDTH = (
    len(LOB_ASK_PRICE_COLS)
    + len(LOB_ASK_SIZE_COLS)
    + len(LOB_BID_PRICE_COLS)
    + len(LOB_BID_SIZE_COLS)
)


@dataclass(frozen=True)
class ActionExecutionResult:
    """动作执行后的逐 horizon 收益结果。

    字段:
        returns: 扣除手续费和 LOB 滑点后的净收益。
        gross_returns: 未扣交易成本的持仓价差收益。
        fees: 手续费成本。
        turnover: 换手量，初始持仓按 flat 计算。
    """

    returns: np.ndarray
    gross_returns: np.ndarray
    fees: np.ndarray
    turnover: np.ndarray


class ActionExecutionCalculator:
    """根据价格序列和动作序列计算执行收益。

    统一动作语义:
        ``0=short``、``1=flat``、``2=long``，对应持仓 ``-1/0/1``。

    统一收益语义:
        ``action_t`` 先在 ``price_t`` 附近执行并切到 ``position_t``，
        ``position_t`` 再持有 ``price_t -> price_{t+1}`` 这一根 bar；
        horizon 最后一个 action 没有下一根 bar，因此不贡献收益；
        初始持仓为 flat，手续费按单边换手 ``abs(position_t-position_{t-1})``
        乘以 ``price_t`` 和 ``fee_rate`` 计算；提供 LOB 深度行情时，滑点从
        ``depthprices_t`` 的五档 ask/bid price 和 size 逐档撮合得到。
    """

    DEFAULT_EPS = 1e-12

    def __init__(self, *, fee_rate: float = 0.0004, eps: float = DEFAULT_EPS) -> None:
        if fee_rate < 0.0:
            raise ValueError("fee_rate must be non-negative")
        if eps <= 0.0:
            raise ValueError("eps must be positive")

        self.fee_rate = float(fee_rate)
        self.eps = float(eps)

    def execute(
        self,
        prices: np.ndarray | None,
        actions: np.ndarray,
        depthprices: np.ndarray | None = None,
    ) -> ActionExecutionResult:
        """执行动作并返回逐样本收益。

        参数:
            prices: 价格数组，支持 ``[N, H]`` 或 ``[N, H, 1]``；缺失或无效时
                返回 NaN 收益。
            actions: 动作数组，形状 ``[N, H]``，取值语义为 ``0/1/2``。
            depthprices: 可选 LOB 深度行情，形状 ``[N, H, 20]``。列顺序为
                ask price、ask size、bid price、bid size 各五档。
        """

        action_values = self._actions_2d(actions)
        price_values = self._prices_2d(prices)
        if price_values is None:
            return self._nan_result(action_values.shape[0])
        if price_values.shape[0] != action_values.shape[0]:
            raise ValueError("prices and actions must have the same sample count")
        depth_values = self._depthprices_3d(depthprices)
        if depth_values is not None and depth_values.shape[0] != action_values.shape[0]:
            raise ValueError("depthprices and actions must have the same sample count")

        positions = self.actions_to_positions(action_values)
        horizon = min(price_values.shape[1], positions.shape[1])
        if depth_values is not None:
            horizon = min(horizon, depth_values.shape[1])
        if horizon < 2:
            return self._nan_result(positions.shape[0])

        price_values = price_values[:, :horizon]
        positions = positions[:, :horizon]
        if depth_values is not None:
            depth_values = depth_values[:, :horizon]

        executable_positions = positions[:, :-1]
        previous_positions = np.concatenate(
            [
                np.zeros((positions.shape[0], 1), dtype=np.float64),
                positions[:, :-2],
            ],
            axis=1,
        )
        delta_positions = executable_positions - previous_positions
        gross_path = executable_positions * (price_values[:, 1:] - price_values[:, :-1])
        gross_returns = np.sum(gross_path, axis=1)
        turnover_path = np.abs(delta_positions)
        turnover = np.sum(turnover_path, axis=1)
        fee_path = turnover_path * price_values[:, :-1] * self.fee_rate
        fees = np.sum(fee_path, axis=1)
        slippages = self._slippage_path(
            delta_positions=delta_positions,
            depthprices=depth_values,
            mark_prices=price_values[:, :-1],
        )
        return ActionExecutionResult(
            returns=gross_returns - fees - slippages,
            gross_returns=gross_returns,
            fees=fees,
            turnover=turnover,
        )

    @classmethod
    def execute_actions(
        cls,
        prices: np.ndarray | None,
        actions: np.ndarray,
        fee_rate: float,
        depthprices: np.ndarray | None = None,
    ) -> ActionExecutionResult:
        """无状态便捷入口，适合已有函数式调用方。"""

        return cls(fee_rate=fee_rate).execute(prices, actions, depthprices)

    @staticmethod
    def actions_to_positions(actions: np.ndarray) -> np.ndarray:
        """将动作 id 映射为持仓值。"""

        return np.asarray(actions, dtype=np.float64) - 1.0

    @staticmethod
    def _actions_2d(actions: np.ndarray) -> np.ndarray:
        values = np.asarray(actions)
        if values.ndim != 2:
            raise ValueError("actions must have shape [sample, horizon]")
        return values

    @staticmethod
    def _prices_2d(prices: np.ndarray | None) -> np.ndarray | None:
        if prices is None:
            return None
        values = np.asarray(prices, dtype=np.float64)
        if values.ndim == 3 and values.shape[-1] == 1:
            values = values[..., 0]
        if values.ndim != 2 or values.shape[1] < 2:
            return None
        return values

    @staticmethod
    def _depthprices_3d(depthprices: np.ndarray | None) -> np.ndarray | None:
        if depthprices is None:
            return None
        values = np.asarray(depthprices, dtype=np.float64)
        if values.ndim != 3 or values.shape[-1] != LOB_DEPTH_WIDTH:
            raise ValueError(
                f"depthprices must have shape [sample, horizon, {LOB_DEPTH_WIDTH}]"
            )
        if values.shape[1] < 2:
            return None
        return values

    @staticmethod
    def _nan_result(sample_count: int) -> ActionExecutionResult:
        empty = np.full(sample_count, float("nan"), dtype=np.float64)
        return ActionExecutionResult(empty, empty.copy(), empty.copy(), empty.copy())

    @classmethod
    def _slippage_path(
        cls,
        *,
        delta_positions: np.ndarray,
        depthprices: np.ndarray | None,
        mark_prices: np.ndarray,
    ) -> np.ndarray:
        if depthprices is None:
            return np.zeros(delta_positions.shape[0], dtype=np.float64)

        return np.sum(
            cls.compute_lob_slippage_from_depthprices(
                delta_positions=delta_positions,
                depthprices=depthprices[:, : delta_positions.shape[1]],
                mark_prices=mark_prices,
            ),
            axis=1,
        )

    @staticmethod
    def compute_lob_slippage_from_depthprices(
        *,
        delta_positions: np.ndarray,
        depthprices: np.ndarray,
        mark_prices: np.ndarray,
    ) -> np.ndarray:
        """批量使用 ``[ask_price, ask_size, bid_price, bid_size]`` LOB 向量计算滑点。"""

        delta_values = np.asarray(delta_positions, dtype=np.float64)
        depth_values = np.asarray(depthprices, dtype=np.float64)
        mark_values = np.asarray(mark_prices, dtype=np.float64)
        if depth_values.ndim != 3 or depth_values.shape[-1] != LOB_DEPTH_WIDTH:
            raise ValueError(
                f"depthprices must have shape [sample, horizon, {LOB_DEPTH_WIDTH}]"
            )
        if delta_values.shape != depth_values.shape[:2]:
            raise ValueError(
                "delta_positions must share [sample, horizon] with depthprices"
            )
        if mark_values.shape != delta_values.shape:
            raise ValueError("mark_prices must share shape with delta_positions")

        level_count = len(LOB_ASK_PRICE_COLS)
        ask_prices = depth_values[..., :level_count]
        ask_sizes = depth_values[..., level_count : level_count * 2]
        bid_prices = depth_values[..., level_count * 2 : level_count * 3]
        bid_sizes = depth_values[..., level_count * 3 : level_count * 4]

        buy_mask = delta_values > 0.0
        sell_mask = delta_values < 0.0
        side_prices = np.where(buy_mask[..., None], ask_prices, bid_prices)
        side_sizes = np.where(buy_mask[..., None], ask_sizes, bid_sizes)

        remaining = np.abs(delta_values)
        fill_cash = np.zeros_like(delta_values, dtype=np.float64)
        last_prices = mark_values.copy()
        for level_index in range(level_count):
            level_prices = side_prices[..., level_index]
            level_sizes = side_sizes[..., level_index]
            active = (remaining > 0.0) & (level_prices > 0.0) & (level_sizes > 0.0)
            last_prices = np.where(active, level_prices, last_prices)
            fill_qty = np.minimum(remaining, np.where(active, level_sizes, 0.0))
            fill_cash += fill_qty * np.where(active, level_prices, 0.0)
            remaining -= fill_qty

        fill_cash += remaining * last_prices
        mark_cash = np.abs(delta_values) * mark_values
        slippage = np.where(buy_mask, fill_cash - mark_cash, mark_cash - fill_cash)
        slippage = np.where(buy_mask | sell_mask, slippage, 0.0)
        return np.maximum(slippage, 0.0)

    # @staticmethod
    # def compute_lob_slippage(
    #     delta_position: float, state: dict, mark_price: float,
    # ) -> float:
    #     """Walk the 5-level LOB to compute slippage cost.

    #     # Section 3.1: C(|ΔP|) - |ΔP| × p_mark
    #     # For buys (ΔP > 0): walk ask side, slippage = fill_cash - |ΔP| × mark
    #     # For sells (ΔP < 0): walk bid side, slippage = |ΔP| × mark - fill_cash
    #     #
    #     # If the 5-level book cannot fill the entire order, remaining
    #     # quantity is filled at the worst available level.

    #     Args:
    #         delta_position: signed position change (>0 buy, <0 sell)
    #         state: polars row dict containing LOB features
    #         mark_price: mark price p_mark

    #     Returns:
    #         slippage cost (non-negative)
    #     """
    #     if delta_position == 0:
    #         return 0.0

    #     abs_delta = float(abs(delta_position))

    #     if delta_position > 0:
    #         price_cols = LOB_ASK_PRICE_COLS
    #         size_cols = LOB_ASK_SIZE_COLS
    #     else:
    #         price_cols = LOB_BID_PRICE_COLS
    #         size_cols = LOB_BID_SIZE_COLS

    #     qty_remaining = abs_delta
    #     fill_cash = 0.0
    #     last_price = mark_price

    #     for p_col, s_col in zip(price_cols, size_cols):
    #         level_price = float(state[p_col])
    #         level_size = float(state[s_col])
    #         if level_price <= 0 or level_size <= 0:
    #             continue
    #         last_price = level_price
    #         fill_qty = min(qty_remaining, level_size)
    #         fill_cash += fill_qty * level_price
    #         qty_remaining -= fill_qty
    #         if qty_remaining <= 0:
    #             break

    #     if qty_remaining > 0:
    #         fill_cash += qty_remaining * last_price
    #     # 价格举例：当前持仓量为 0，目标持仓量为 1，当前价格为 10，如果 通过委托价格9 购买 fillcash 9 slippage 为 -1
    #     # 价格举例：当前持仓量为 0，目标持仓量为 1，当前价格为 10，如果 通过委托价格11购买 fillcash 11 slippage 为 1
    #     if delta_position > 0:
    #         slippage = fill_cash - abs_delta * mark_price
    #     else:
    #         slippage = abs_delta * mark_price - fill_cash

    #     return max(slippage, 0.0)

    @staticmethod
    def compute_lob_slippage_from_depthprice(
        *,
        delta_position: float,
        depthprice: np.ndarray,
        mark_price: float,
    ) -> float:
        """使用 ``[ask_price, ask_size, bid_price, bid_size]`` LOB 向量计算滑点。"""

        if delta_position == 0.0:
            return 0.0

        values = np.asarray(depthprice, dtype=np.float64).reshape(-1)
        if values.shape[0] != LOB_DEPTH_WIDTH:
            raise ValueError(f"depthprice must have {LOB_DEPTH_WIDTH} values")

        level_count = len(LOB_ASK_PRICE_COLS)
        ask_prices = values[:level_count]
        ask_sizes = values[level_count : level_count * 2]
        bid_prices = values[level_count * 2 : level_count * 3]
        bid_sizes = values[level_count * 3 : level_count * 4]
        if delta_position > 0.0:
            level_prices = ask_prices
            level_sizes = ask_sizes
        else:
            level_prices = bid_prices
            level_sizes = bid_sizes

        qty_remaining = abs(float(delta_position))
        fill_cash = 0.0
        last_price = float(mark_price)
        for level_price, level_size in zip(level_prices, level_sizes):
            price = float(level_price)
            size = float(level_size)
            if price <= 0.0 or size <= 0.0:
                continue
            last_price = price
            fill_qty = min(qty_remaining, size)
            fill_cash += fill_qty * price
            qty_remaining -= fill_qty
            if qty_remaining <= 0.0:
                break

        if qty_remaining > 0.0:
            fill_cash += qty_remaining * last_price

        mark_cash = abs(float(delta_position)) * float(mark_price)
        if delta_position > 0.0:
            slippage = fill_cash - mark_cash
        else:
            slippage = mark_cash - fill_cash
        return max(float(slippage), 0.0)

__all__ = ["ActionExecutionCalculator", "ActionExecutionResult"]
