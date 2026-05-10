"""交易动作执行收益计算工具。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ActionExecutionResult:
    """动作执行后的逐 horizon 收益结果。

    字段:
        returns: 扣除手续费后的净收益。
        gross_returns: 未扣手续费的收益。
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
        ``position_t`` 执行在 ``price_t -> price_{t+1}`` 这一根 bar 上；
        horizon 最后一个 action 没有下一根 bar，因此不贡献收益；
        初始持仓为 flat，手续费按单边换手 ``abs(position_t-position_{t-1})``
        乘以 ``fee_rate`` 计算。
    """

    DEFAULT_EPS = 1e-12

    def __init__(self, *, fee_rate: float = 0.0, eps: float = DEFAULT_EPS) -> None:
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
    ) -> ActionExecutionResult:
        """执行动作并返回逐样本收益。

        参数:
            prices: 价格数组，支持 ``[N, H]`` 或 ``[N, H, 1]``；缺失或无效时
                返回 NaN 收益。
            actions: 动作数组，形状 ``[N, H]``，取值语义为 ``0/1/2``。
        """

        action_values = self._actions_2d(actions)
        price_values = self._prices_2d(prices)
        if price_values is None:
            return self._nan_result(action_values.shape[0])
        if price_values.shape[0] != action_values.shape[0]:
            raise ValueError("prices and actions must have the same sample count")

        positions = self.actions_to_positions(action_values)
        horizon = min(price_values.shape[1], positions.shape[1])
        if horizon < 2:
            return self._nan_result(positions.shape[0])

        price_values = price_values[:, :horizon]
        positions = positions[:, :horizon]
        bar_returns = (
            price_values[:, 1:] / np.maximum(price_values[:, :-1], self.eps) - 1.0
        )
        gross_path = positions[:, :-1] * bar_returns
        gross_returns = np.sum(gross_path, axis=1)
        position_path = np.concatenate(
            [np.zeros((positions.shape[0], 1), dtype=np.float64), positions],
            axis=1,
        )
        turnover = np.sum(np.abs(np.diff(position_path, axis=1)), axis=1)
        fees = turnover * self.fee_rate
        return ActionExecutionResult(
            returns=gross_returns - fees,
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
    ) -> ActionExecutionResult:
        """无状态便捷入口，适合已有函数式调用方。"""

        return cls(fee_rate=fee_rate).execute(prices, actions)

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
    def _nan_result(sample_count: int) -> ActionExecutionResult:
        empty = np.full(sample_count, float("nan"), dtype=np.float64)
        return ActionExecutionResult(empty, empty.copy(), empty.copy(), empty.copy())


__all__ = ["ActionExecutionCalculator", "ActionExecutionResult"]
