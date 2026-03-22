"""MDP 交易环境

# Section 3.1: MDP 定义
# 状态空间: LOB 数据 + OHLCV + 技术指标 (45 维)
# 动作空间: a_t ∈ {0, 1, 2} → short / flat / long
# 持仓: P_t ∈ {-m, 0, m}，m 为交易对最大持仓量
# 奖励: r_step_t = P_t × (p_{t+1} - p_t) - O_t  (论文 Eq. 1)
"""

from typing import Dict, Tuple

import numpy as np


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

    def __init__(
        self,
        states: np.ndarray,
        prices: np.ndarray,
        pair: str,
        horizon: int = 72,
    ):
        """
        Args:
            states: 状态序列 shape (T, state_dim)
            prices: 价格序列 shape (T,)，mark prices 用于奖励计算
            pair: 交易对名称，如 'BTC', 'ETH', 'DOT', 'BNB'
            horizon: 交易周期长度 h，默认 72
        """
        if pair not in self.MAX_POSITIONS:
            raise ValueError(
                f"不支持的交易对 '{pair}'，支持: {list(self.MAX_POSITIONS.keys())}"
            )
        if states.ndim != 2:
            raise ValueError(f"states 应为 2 维数组，收到 {states.ndim} 维")
        if prices.ndim != 1:
            raise ValueError(f"prices 应为 1 维数组，收到 {prices.ndim} 维")
        if states.shape[0] != prices.shape[0]:
            raise ValueError(
                f"states 长度 ({states.shape[0]}) 与 prices 长度 ({prices.shape[0]}) 不一致"
            )

        self.states = states
        self.prices = prices
        self.pair = pair
        self.horizon = horizon
        self.state_dim = states.shape[1]

        # Section 3.1: 最大持仓量 m
        self.m: int = self.MAX_POSITIONS[pair]

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
        execution_cost = self.compute_execution_cost(
            action, old_position, self.prices[t]
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

    def compute_fill_cost(self, action: int, current_position: int) -> float:
        """计算 LOB fill cost（成交价差）。

        # Section 3.1: LOB fill cost 计算
        # [NOTE: 论文未明确描述从特征向量计算 fill cost 的具体方法]
        # 由于 LOB 信息已嵌入状态特征（bid/ask sizes），此处使用简化模型：
        # fill cost = 半个 spread × 交易量变化
        # 当持仓不变时 fill cost 为 0

        Args:
            action: 交易动作 a_t ∈ {0, 1, 2}
            current_position: 当前持仓量

        Returns:
            fill cost（非负值）
        """
        target_direction = self.POSITION_MAP[action]
        new_position = target_direction * self.m

        # 持仓未变化，无 fill cost
        if new_position == current_position:
            return 0.0

        # [TODO: 论文未提供从特征向量精确计算 LOB fill cost 的方法]
        # 简化模型：使用固定的 spread 比例作为 fill cost 近似
        # 实际实现应根据 LOB 深度（bid/ask sizes in state features）计算
        # 此处 fill cost 设为 0，佣金已在 compute_execution_cost 中计算
        return 0.0

    def compute_execution_cost(
        self, action: int, current_position: int, price: float
    ) -> float:
        """计算总执行损失 = fill cost + 佣金。

        # Section 3.1: O_t = fill_cost + commission
        # 佣金 = δ × |ΔP| × price，δ = 0.02%

        Args:
            action: 交易动作 a_t ∈ {0, 1, 2}
            current_position: 当前持仓量
            price: 当前价格

        Returns:
            总执行损失（非负值）
        """
        target_direction = self.POSITION_MAP[action]
        new_position = target_direction * self.m

        # 持仓变化量
        delta_position = abs(new_position - current_position)

        if delta_position == 0:
            return 0.0

        # Fill cost（基于 LOB 深度）
        fill_cost = self.compute_fill_cost(action, current_position)

        # Section 3.1: 佣金 = δ × |ΔP| × price
        commission = self.COMMISSION_RATE * delta_position * price

        return fill_cost + commission
