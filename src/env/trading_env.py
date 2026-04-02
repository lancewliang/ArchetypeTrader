"""MDP 交易环境

# Section 3.1: MDP 定义
# 状态空间: LOB 数据 + OHLCV + 技术指标 (45 维)
# 动作空间: a_t ∈ {0, 1, 2} → short / flat / long
# 持仓: P_t ∈ {-m, 0, m}，m 为交易对最大持仓量
# 奖励: r_step_t = P_t × (p_{t+1} - p_t) - O_t  (论文 Eq. 1)
"""

from typing import Dict, Tuple

import numpy as np
import polars as pl


class TradingEnv:
    """MDP 交易环境

    符合论文 Section 3.1 定义的交易环境，支持 DP 规划和 RL 训练。
    """

    # Section 3.1: 动作 → 持仓方向映射
    POSITION_MAP: Dict[int, int] = {0: -1, 1: 0, 2: 1}  # short / flat / long

    # 各交易对最大持仓量 m
    MAX_POSITIONS: Dict[str, int] = {
        "BTC": 8,
        "ETH": 100,
        "DOT": 2500,
        "BNB": 200,
    }

    # Section 3.1: 佣金率 δ = 0.02%
    COMMISSION_RATE: float = 0.0002

    # Section 3.1: LOB feature indices in state vector
    # (matching SINGLE_FEATURES order in feature_pipeline.py)
    LOB_ASK_PRICE_COLS = ["ask1_price", "ask2_price", "ask3_price", "ask4_price", "ask5_price"]
    LOB_ASK_SIZE_COLS = ["ask1_size", "ask2_size", "ask3_size", "ask4_size", "ask5_size"]
    LOB_BID_PRICE_COLS = ["bid1_price", "bid2_price", "bid3_price", "bid4_price", "bid5_price"]
    LOB_BID_SIZE_COLS = ["bid1_size", "bid2_size", "bid3_size", "bid4_size", "bid5_size"]

    def __init__(
        self,
        states: np.ndarray,
        prices: np.ndarray,
        pair: str,
        horizon: int = 72,
        states_dataframe: pl.DataFrame = None,
        max_positions: Dict[str, int] | None = None,
        commission_rate: float | None = None,
    ):
        """
        Args:
            states: 状态序列 shape (T, state_dim)
            prices: 价格序列 shape (T,)，mark prices 用于奖励计算
            pair: 交易对名称，如 'BTC', 'ETH', 'DOT', 'BNB'
            horizon: 交易周期长度 h，默认 72
            states_dataframe: 对应的 polars DataFrame，可选
            max_positions: 各交易对最大持仓量映射，可选。未提供时使用论文默认值。
            commission_rate: 佣金率，可选。未提供时使用论文默认值 0.02%。
        """
        position_config = max_positions or self.MAX_POSITIONS
        if pair not in position_config:
            raise ValueError(
                f"不支持的交易对 '{pair}'，支持: {list(position_config.keys())}"
            )
        if states.ndim != 2:
            raise ValueError(f"states 应为 2 维数组，收到 {states.ndim} 维")
        if prices.ndim != 1:
            raise ValueError(f"prices 应为 1 维数组，收到 {prices.ndim} 维")
        if states.shape[0] != prices.shape[0]:
            raise ValueError(
                f"states 长度 ({states.shape[0]}) 与 prices 长度 ({prices.shape[0]}) 不一致"
            )
        if horizon <= 0:
            raise ValueError(f"horizon 必须为正整数，收到 {horizon}")

        self.states = states
        self.prices = prices
        self.pair = pair
        self.horizon = horizon
        self.state_dim = states.shape[1]
        self.states_dataframe = states_dataframe
        self.max_positions = dict(position_config)
        self.commission_rate = (
            float(commission_rate)
            if commission_rate is not None
            else float(self.COMMISSION_RATE)
        )

        # Section 3.1: 最大持仓量 m
        self.m: int = int(self.max_positions[pair])
        if self.m <= 0:
            raise ValueError(f"最大持仓量 m 必须为正整数，收到 m={self.m}")

        # 计算可用 horizon 数量
        self.num_horizons = len(states) // horizon

        # Episode 状态（由 reset 初始化）
        self._current_step: int = 0
        self._horizon_start: int = 0
        self._position: int = 0  # 当前持仓量: -m, 0, 或 m
        self._done: bool = True

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def reset(self, horizon_idx: int) -> np.ndarray:
        """重置环境到指定 horizon 的起始状态。

        Args:
            horizon_idx: horizon 索引，范围 [0, num_horizons)

        Returns:
            初始状态向量 shape (state_dim,)

        Raises:
            IndexError: horizon_idx 越界
        """
        if horizon_idx < 0 or horizon_idx >= self.num_horizons:
            raise IndexError(
                f"horizon_idx={horizon_idx} 越界，有效范围 [0, {self.num_horizons})"
            )

        self._horizon_start = horizon_idx * self.horizon
        self._current_step = 0
        # Section 3.1: episode 开始时持仓为 flat
        self._position = 0
        self._done = False

        return self.states[self._horizon_start].copy()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """执行一步交易动作。

        # 论文 Eq. 1: r_step_t = P_t × (p_mark_{t+1} - p_mark_t) - O_t

        Args:
            action: 交易动作 a_t ∈ {0, 1, 2}

        Returns:
            (next_state, reward, done, info)

        Raises:
            ValueError: 无效动作
            RuntimeError: episode 已结束但仍调用 step
        """
        if self._done:
            raise RuntimeError("Episode 已结束，请先调用 reset()")
        if action not in self.POSITION_MAP:
            raise ValueError(
                f"无效动作 {action}，有效动作: {list(self.POSITION_MAP.keys())}"
            )

        # 当前全局索引
        t = self._horizon_start + self._current_step

        # Section 3.1: 根据动作计算目标持仓
        old_position = self._position
        target_direction = self.POSITION_MAP[action]
        new_position = target_direction * self.m

        # Section 3.1: 计算执行损失 O_t（仅在持仓变化时产生）
        if self.states_dataframe is not None:
            state_dict = self.states_dataframe.row(t, named=True)
        else:
            state_dict = None
        execution_cost = self.compute_execution_cost(
            action, old_position, self.prices[t], state_dict
        )

        # 更新持仓
        self._position = new_position

        # Section 3.1 / Eq. 1: 计算逐步奖励
        # r_step_t = P_t × (p_{t+1} - p_t) - O_t
        # P_t 是当前持仓（更新后），用于计算本步持有收益
        t_next = t + 1
        if t_next < len(self.prices):
            price_diff = self.prices[t_next] - self.prices[t]
        else:
            # 最后一步无下一个价格，价差为 0
            price_diff = 0.0

        reward = float(self._position * price_diff - execution_cost)

        # 推进步数
        self._current_step += 1

        # Section 3.1: episode 在 h 步后终止
        done = self._current_step >= self.horizon

        self._done = done

        # 下一个状态
        if done:
            # episode 结束，返回最后一个可用状态
            next_t = min(t + 1, len(self.states) - 1)
            next_state = self.states[next_t].copy()
        else:
            next_state = self.states[t + 1].copy()

        info = {
            "position": self._position,
            "old_position": old_position,
            "execution_cost": execution_cost,
            "price": self.prices[t],
            "step_in_horizon": self._current_step,
        }

        return next_state, reward, done, info

    @staticmethod
    def compute_lob_slippage(
        delta_position: int, state: dict, mark_price: float,
    ) -> float:
        """Walk the 5-level LOB to compute slippage cost.

        # Section 3.1: C(|ΔP|) - |ΔP| × p_mark
        # For buys (ΔP > 0): walk ask side, slippage = fill_cash - |ΔP| × mark
        # For sells (ΔP < 0): walk bid side, slippage = |ΔP| × mark - fill_cash
        #
        # If the 5-level book cannot fill the entire order, remaining
        # quantity is filled at the worst available level.

        Args:
            delta_position: signed position change (>0 buy, <0 sell)
            state: polars row dict containing LOB features
            mark_price: mark price p_mark

        Returns:
            slippage cost (non-negative)
        """
        if delta_position == 0:
            return 0.0

        abs_delta = float(abs(delta_position))

        if delta_position > 0:
            price_cols = TradingEnv.LOB_ASK_PRICE_COLS
            size_cols = TradingEnv.LOB_ASK_SIZE_COLS
        else:
            price_cols = TradingEnv.LOB_BID_PRICE_COLS
            size_cols = TradingEnv.LOB_BID_SIZE_COLS

        qty_remaining = abs_delta
        fill_cash = 0.0
        last_price = mark_price

        for p_col, s_col in zip(price_cols, size_cols):
            level_price = float(state[p_col])
            level_size = float(state[s_col])
            if level_price <= 0 or level_size <= 0:
                continue
            last_price = level_price
            fill_qty = min(qty_remaining, level_size)
            fill_cash += fill_qty * level_price
            qty_remaining -= fill_qty
            if qty_remaining <= 0:
                break

        if qty_remaining > 0:
            fill_cash += qty_remaining * last_price
        # 价格举例：当前持仓量为 0，目标持仓量为 1，当前价格为 10，如果 通过委托价格9 购买 fillcash 9 slippage 为 -1
        # 价格举例：当前持仓量为 0，目标持仓量为 1，当前价格为 10，如果 通过委托价格11购买 fillcash 11 slippage 为 1
        if delta_position > 0:
            slippage = fill_cash - abs_delta * mark_price
        else:
            slippage = abs_delta * mark_price - fill_cash

        return max(slippage, 0.0)

    def compute_fill_cost(
        self, delta_position: int, state: dict, mark_price: float,
    ) -> float:
        """Compute LOB fill cost (slippage component of execution loss).

        # Section 3.1: C(|ΔP|) - |ΔP| × p_mark
        # Delegates to compute_lob_slippage.

        Args:
            delta_position: signed position change
            state: polars row dict containing LOB features
            mark_price: mark price

        Returns:
            slippage cost (non-negative)
        """
        return self.compute_lob_slippage(delta_position, state, mark_price)

    def compute_execution_cost(
        self, action: int, current_position: int, price: float,
        state: dict | None = None,
    ) -> float:
        """计算总执行损失 = slippage + 佣金。

        # Section 3.1: O_t = C(|ΔP|) - |ΔP| × p_mark + δ × |ΔP| × p_mark
        #                   = slippage + commission
        # C(·) 通过 walk 5-level LOB 计算。
        # 若 state 未提供，退化为仅佣金（向后兼容）。

        Args:
            action: 交易动作 a_t ∈ {0, 1, 2}
            current_position: 当前持仓量
            price: 当前 mark price
            state: polars row dict（含 LOB 特征），可选

        Returns:
            总执行损失（非负值）
        """
        target_direction = self.POSITION_MAP[action]
        new_position = target_direction * self.m

        # 持仓变化量（有符号）
        delta_position = new_position - current_position

        if delta_position == 0:
            return 0.0

        abs_delta = abs(delta_position)

        # C(|ΔP|) - |ΔP| × p_mark: LOB slippage
        if state is not None:
            slippage = self.compute_lob_slippage(delta_position, state, price)
        else:
            slippage = 0.0

        # Section 3.1: 佣金 = δ × |ΔP| × price
        commission = self.commission_rate * abs_delta * price

        return slippage + commission
