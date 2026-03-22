"""评估指标模块 — ArchetypeTrader

实现论文中定义的所有评估指标：
- Total Return (TR)
- Annual Volatility (AVOL)
- Maximum Drawdown (MDD)
- Annual Sharpe Ratio (ASR)
- Annual Calmar Ratio (ACR)
- Annual Sortino Ratio (ASoR)

需求: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6
"""

import math
from typing import Dict

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class EvaluationEngine:
    """评估引擎，计算论文中定义的所有指标。

    Args:
        annualization_factor: 年化因子 m，默认 52560（10 分钟级别）。
    """

    def __init__(self, annualization_factor: int = 52560) -> None:
        self.annualization_factor = annualization_factor

    def compute_total_return(self, returns: np.ndarray) -> float:
        """计算总收益率。

        TR = Π(1 + r_t) - 1

        Args:
            returns: 1D 逐步收益序列。

        Returns:
            总收益率。
        """
        return float(np.prod(1.0 + returns) - 1.0)

    def compute_annual_volatility(self, returns: np.ndarray) -> float:
        """计算年化波动率。

        AVOL = σ(r) × √m

        Args:
            returns: 1D 逐步收益序列。

        Returns:
            年化波动率。
        """
        return float(np.std(returns) * math.sqrt(self.annualization_factor))

    def compute_max_drawdown(self, returns: np.ndarray) -> float:
        """计算最大回撤。

        基于累积财富曲线计算：MDD = max((peak - trough) / peak)

        Args:
            returns: 1D 逐步收益序列。

        Returns:
            最大回撤（非负值）。
        """
        # 累积财富曲线
        wealth = np.cumprod(1.0 + returns)
        # 滚动峰值
        running_max = np.maximum.accumulate(wealth)
        # 回撤序列
        drawdowns = (running_max - wealth) / running_max
        return float(np.max(drawdowns))

    def compute_annual_sharpe_ratio(self, returns: np.ndarray) -> float:
        """计算年化夏普比率。

        ASR = mean(r) / σ(r) × √m

        当 σ(r) = 0 时返回 0.0 并记录 WARNING。

        Args:
            returns: 1D 逐步收益序列。

        Returns:
            年化夏普比率。
        """
        std = float(np.std(returns))
        if std == 0.0:
            logger.warning("Sharpe ratio: std of returns is 0, returning 0.0")
            return 0.0
        mean_r = float(np.mean(returns))
        return mean_r / std * math.sqrt(self.annualization_factor)

    def compute_annual_calmar_ratio(self, returns: np.ndarray) -> float:
        """计算年化卡尔玛比率。

        ACR = mean(r) / MDD × m

        当 MDD = 0 时返回 0.0 并记录 WARNING。

        Args:
            returns: 1D 逐步收益序列。

        Returns:
            年化卡尔玛比率。
        """
        mdd = self.compute_max_drawdown(returns)
        if mdd == 0.0:
            logger.warning("Calmar ratio: MDD is 0, returning 0.0")
            return 0.0
        mean_r = float(np.mean(returns))
        return mean_r / mdd * self.annualization_factor

    def compute_annual_sortino_ratio(self, returns: np.ndarray) -> float:
        """计算年化索提诺比率。

        ASoR = mean(r) / DD × √m

        DD 为负收益的标准差（下行偏差）。当 DD = 0 时返回 0.0 并记录 WARNING。

        Args:
            returns: 1D 逐步收益序列。

        Returns:
            年化索提诺比率。
        """
        negative_returns = returns[returns < 0]
        if len(negative_returns) == 0:
            logger.warning("Sortino ratio: no negative returns (DD=0), returning 0.0")
            return 0.0
        dd = float(np.std(negative_returns))
        if dd == 0.0:
            logger.warning("Sortino ratio: downside deviation is 0, returning 0.0")
            return 0.0
        mean_r = float(np.mean(returns))
        return mean_r / dd * math.sqrt(self.annualization_factor)

    def evaluate(self, returns: np.ndarray) -> Dict[str, float]:
        """计算所有评估指标并返回字典。

        Args:
            returns: 1D 逐步收益序列。

        Returns:
            包含所有指标的字典。
        """
        return {
            "total_return": self.compute_total_return(returns),
            "annual_volatility": self.compute_annual_volatility(returns),
            "max_drawdown": self.compute_max_drawdown(returns),
            "annual_sharpe_ratio": self.compute_annual_sharpe_ratio(returns),
            "annual_calmar_ratio": self.compute_annual_calmar_ratio(returns),
            "annual_sortino_ratio": self.compute_annual_sortino_ratio(returns),
        }
