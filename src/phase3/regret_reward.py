"""Regret-aware Reward 计算模块

# Section 4.3: Regret-aware reward for refinement agent
# r_ref = (R - R_base) + β_1 × (R - R_1_opt)  if a_ref != 0
# r_ref = 0                                      if a_ref == 0
#
# R       : 实际收益（使用精炼后动作的 horizon 收益）
# R_base  : 基线收益（使用原始原型动作的 horizon 收益）
# R_1_opt : top-1 hindsight-optimal adaptation 的收益
# β_1     : regret 系数，可选 {0.3, 0.5, 0.7}
#
# Top-5 hindsight-optimal adaptations (Eq. 7):
#   O_top5 = {(τ_opt^n, a_opt^n, R_opt^n)}_{n=1}^{5}
#   对 horizon 内每个可能的调整点和调整动作，
#   计算假设执行该调整后的 horizon 收益，
#   返回收益最高的 5 个 (τ_opt, adaptation_action, resulting_return)。
"""

from typing import List, Tuple

import numpy as np

from src.env.trading_env import TradingEnv


def compute_regret_reward(
    R: float,
    R_base: float,
    R_1_opt: float,
    a_ref: int,
    beta1: float = 0.5,
) -> float:
    """Compute regret-aware reward for refinement agent.

    # Section 4.3: Regret-aware reward
    # r_ref = (R - R_base) + β_1 × (R - R_1_opt)  if a_ref != 0
    # r_ref = 0                                      if a_ref == 0

    Args:
        R: 实际收益（精炼后动作的 horizon 收益）
        R_base: 基线收益（原始原型动作的 horizon 收益）
        R_1_opt: top-1 hindsight-optimal adaptation 的收益
        a_ref: refinement agent 的调整信号 ∈ {-1, 0, 1}
        beta1: regret 系数 β_1，可选 {0.3, 0.5, 0.7}，默认 0.5

    Returns:
        r_ref: regret-aware reward 值
    """
    if a_ref == 0:
        return 0.0

    # r_ref = (R - R_base) + β_1 × (R - R_1_opt)
    improvement = R - R_base
    regret = R - R_1_opt
    return improvement + beta1 * regret


def compute_top5_hindsight_optimal(
    prices: np.ndarray,
    base_actions: np.ndarray,
    step_idx: int,
    env: TradingEnv,
) -> List[Tuple[int, int, float]]:
    """Compute top-5 hindsight-optimal adaptations at a given step.

    # Section 4.3 / Eq. 7: Hindsight-optimal adaptations
    # O_top5 = {(τ_opt^n, a_opt^n, R_opt^n)}_{n=1}^{5}
    # 对 horizon 内从 step_idx 开始的每个可能调整点，
    # 尝试每种调整动作 {-1, 1}，计算假设执行该调整后的总收益。
    # 返回收益最高的 5 个结果。
    #
    # 调整动作含义（对应 Eq. 6）:
    #   a_ref = -1 → a_final = 0 (flat/short)
    #   a_ref =  1 → a_final = 2 (long)

    Args:
        prices: 当前 horizon 的价格序列 shape (h,)
        base_actions: 基础动作序列 shape (h,)，值域 {0, 1, 2}
        step_idx: 当前步索引（从此步开始考虑调整）
        env: TradingEnv 实例，用于获取持仓量和佣金率等参数

    Returns:
        List of (τ_opt, adaptation_action, resulting_return) tuples,
        sorted descending by return. 最多 5 个。
        τ_opt: 调整时间步
        adaptation_action ∈ {-1, 1}
    """
    h = len(prices)
    m = env.m
    commission_rate = env.COMMISSION_RATE

    candidates: List[Tuple[int, int, float]] = []

    # 对从 step_idx 到 horizon 末尾的每个可能调整点
    for adapt_step in range(step_idx, h):
        # 对每种非零调整动作 {-1, 1}（a_ref=0 不产生调整）
        for a_ref in [-1, 1]:
            total_return = _simulate_adaptation(
                prices=prices,
                base_actions=base_actions,
                adapt_step=adapt_step,
                a_ref=a_ref,
                m=m,
                commission_rate=commission_rate,
            )
            candidates.append((adapt_step, a_ref, total_return))

    # 按收益降序排列，取 top-5
    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates[:5]


def _simulate_adaptation(
    prices: np.ndarray,
    base_actions: np.ndarray,
    adapt_step: int,
    a_ref: int,
    m: int,
    commission_rate: float,
) -> float:
    """模拟在指定步执行一次调整后的 horizon 总收益。

    # 按 Eq. 6 规则：
    #   在 adapt_step 之前使用 base_actions
    #   在 adapt_step 处应用调整: a_ref=-1 → action=0, a_ref=1 → action=2
    #   在 adapt_step 之后恢复 base_actions
    # 每个 horizon 最多一次调整（已由单次 adapt_step 保证）

    Args:
        prices: horizon 价格序列 shape (h,)
        base_actions: 基础动作序列 shape (h,)
        adapt_step: 执行调整的步索引
        a_ref: 调整信号 ∈ {-1, 1}
        m: 最大持仓量
        commission_rate: 佣金率

    Returns:
        horizon 总收益（所有步奖励之和）
    """
    h = len(prices)
    position_map = {0: -1, 1: 0, 2: 1}  # action → direction

    total_reward = 0.0
    position = 0  # 初始持仓为 flat

    for t in range(h):
        # 确定本步实际动作
        if t == adapt_step:
            # Eq. 6: 应用调整
            if a_ref == -1:
                action = 0  # flat/short
            else:  # a_ref == 1
                action = 2  # long
        else:
            action = int(base_actions[t])

        # 计算目标持仓
        direction = position_map[action]
        new_position = direction * m

        # 计算执行损失（佣金）
        delta = abs(new_position - position)
        execution_cost = commission_rate * delta * prices[t] if delta > 0 else 0.0

        # 更新持仓
        position = new_position

        # 计算逐步奖励: r = P_t × (p_{t+1} - p_t) - O_t
        if t + 1 < h:
            price_diff = prices[t + 1] - prices[t]
        else:
            price_diff = 0.0

        reward = position * price_diff - execution_cost
        total_reward += reward

    return total_reward
