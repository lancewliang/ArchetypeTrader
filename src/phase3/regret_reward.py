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
#
# 向量化实现说明:
#   原始实现对每个 (adapt_step, a_ref) 组合独立模拟 h 步，
#   复杂度 O(C × h)，其中 C = 可调整步数 × 2。
#   向量化版本将所有候选组合的动作序列堆叠为 (C, h) 矩阵，
#   通过 NumPy 广播一次性计算所有候选的持仓、交易成本和收益。
#   LOB slippage 也从逐条 dict 查找改为预提取的 (h, 5) 数组批量计算。
"""

from typing import List, Tuple

import numpy as np

from src.env.trading_env import TradingEnv

# action → 持仓方向映射表，用于向量化索引
_ACTION_TO_DIRECTION = np.array([-1, 0, 1])  # index 0→-1, 1→0, 2→1


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


# ---------------------------------------------------------------------------
# LOB 数据预提取: dict list → NumPy 数组
# ---------------------------------------------------------------------------

def _extract_lob_arrays(
    states: list[dict],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """将 polars row dicts 中的 LOB 列批量提取为 NumPy 数组。

    避免在模拟循环中反复做 dict[key] 查找。
    一次性提取后，后续 slippage 计算全部走数组索引。

    Args:
        states: horizon 状态序列 list[dict]，长度 h

    Returns:
        ask_prices: shape (h, 5) — 5 档卖价
        ask_sizes:  shape (h, 5) — 5 档卖量
        bid_prices: shape (h, 5) — 5 档买价
        bid_sizes:  shape (h, 5) — 5 档买量
    """
    h = len(states)
    ask_prices = np.empty((h, 5), dtype=np.float64)
    ask_sizes = np.empty((h, 5), dtype=np.float64)
    bid_prices = np.empty((h, 5), dtype=np.float64)
    bid_sizes = np.empty((h, 5), dtype=np.float64)

    ap_cols = TradingEnv.LOB_ASK_PRICE_COLS
    as_cols = TradingEnv.LOB_ASK_SIZE_COLS
    bp_cols = TradingEnv.LOB_BID_PRICE_COLS
    bs_cols = TradingEnv.LOB_BID_SIZE_COLS

    for t, row in enumerate(states):
        for lv in range(5):
            ask_prices[t, lv] = row[ap_cols[lv]]
            ask_sizes[t, lv] = row[as_cols[lv]]
            bid_prices[t, lv] = row[bp_cols[lv]]
            bid_sizes[t, lv] = row[bs_cols[lv]]

    return ask_prices, ask_sizes, bid_prices, bid_sizes


# ---------------------------------------------------------------------------
# 向量化 LOB slippage: 对 (C, h) 批量持仓变化一次性计算
# ---------------------------------------------------------------------------

def _vectorized_lob_slippage(
    delta_positions: np.ndarray,
    mark_prices: np.ndarray,
    ask_prices: np.ndarray,
    ask_sizes: np.ndarray,
    bid_prices: np.ndarray,
    bid_sizes: np.ndarray,
) -> np.ndarray:
    """批量计算 LOB slippage，替代逐条调用 TradingEnv.compute_lob_slippage。

    对每个 (candidate, timestep) 位置，根据 delta_position 的符号
    选择 ask/bid 侧，执行 5-level walk 计算 fill_cash，
    再得到 slippage = max(fill_cash - |Δ|×mark, 0) 或对称形式。

    虽然 5-level walk 本身难以完全消除循环（因为有 early-break 逻辑），
    但此函数将外层 (candidate × timestep) 循环限制在有持仓变化的位置，
    并使用预提取的数组而非 dict 查找，大幅减少 Python 开销。

    Args:
        delta_positions: shape (C, h) — 每个候选在每步的持仓变化量
        mark_prices:     shape (h,)   — 每步的 mark price
        ask_prices:      shape (h, 5) — 预提取的 5 档卖价
        ask_sizes:       shape (h, 5) — 预提取的 5 档卖量
        bid_prices:      shape (h, 5) — 预提取的 5 档买价
        bid_sizes:       shape (h, 5) — 预提取的 5 档买量

    Returns:
        slippage: shape (C, h) — 每个位置的 slippage cost (≥ 0)
    """
    C, h = delta_positions.shape
    slippage = np.zeros((C, h), dtype=np.float64)

    # 只处理有持仓变化的位置
    nonzero_mask = delta_positions != 0
    cand_indices, step_indices = np.nonzero(nonzero_mask)

    for idx in range(len(cand_indices)):
        c = cand_indices[idx]
        t = step_indices[idx]
        dp = delta_positions[c, t]
        abs_delta = abs(dp)
        mark = mark_prices[t]

        # 选择 ask/bid 侧
        if dp > 0:
            lvl_prices = ask_prices[t]  # shape (5,)
            lvl_sizes = ask_sizes[t]
        else:
            lvl_prices = bid_prices[t]
            lvl_sizes = bid_sizes[t]

        # 5-level walk
        qty_remaining = abs_delta
        fill_cash = 0.0
        last_price = mark

        for lv in range(5):
            lp = lvl_prices[lv]
            ls = lvl_sizes[lv]
            if lp <= 0 or ls <= 0:
                continue
            last_price = lp
            fill_qty = min(qty_remaining, ls)
            fill_cash += fill_qty * lp
            qty_remaining -= fill_qty
            if qty_remaining <= 0:
                break

        if qty_remaining > 0:
            fill_cash += qty_remaining * last_price

        if dp > 0:
            s = fill_cash - abs_delta * mark
        else:
            s = abs_delta * mark - fill_cash

        slippage[c, t] = max(s, 0.0)

    return slippage


# ---------------------------------------------------------------------------
# 核心: 向量化 hindsight-optimal 计算
# ---------------------------------------------------------------------------

def compute_top5_hindsight_optimal(
    prices: np.ndarray,
    base_actions: np.ndarray,
    step_idx: int,
    env: TradingEnv,
    states: list[dict] | None = None,
) -> List[Tuple[int, int, float]]:
    """Compute top-5 hindsight-optimal adaptations at a given step.

    # Section 4.3 / Eq. 7: Hindsight-optimal adaptations
    # O_top5 = {(τ_opt^n, a_opt^n, R_opt^n)}_{n=1}^{5}
    #
    # 向量化策略:
    #   1. 筛选可调整步（排除 base_action 发生变化的步）
    #   2. 为每个 (adapt_step, a_ref) 构建动作序列，堆叠为 (C, h) 矩阵
    #   3. 通过查表将动作矩阵映射为持仓矩阵 (C, h)
    #   4. 计算持仓变化 delta_positions (C, h)
    #   5. 批量计算交易成本（佣金 + LOB slippage）
    #   6. 批量计算逐步收益并求和，得到每个候选的总收益
    #   7. argpartition 取 top-5

    Args:
        prices: 当前 horizon 的价格序列 shape (h,)
        base_actions: 基础动作序列 shape (h,)，值域 {0, 1, 2}
        step_idx: 当前步索引（从此步开始考虑调整）
        env: TradingEnv 实例，用于获取持仓量和佣金率等参数
        states: 当前 horizon 的状态序列 list[dict]（polars row dicts），
                用于计算 LOB slippage。可选，为 None 时退化为仅佣金。

    Returns:
        List of (τ_opt, adaptation_action, resulting_return) tuples,
        sorted descending by return. 最多 5 个。
        τ_opt: 调整时间步
        adaptation_action ∈ {-1, 1}
    """
    h = len(prices)
    m = env.m
    commission_rate = env.COMMISSION_RATE

    # ------------------------------------------------------------------
    # Step 1: 筛选可调整步 — Eq. 6 约束
    # 若 adapt_step > 0 且 base_actions[t] != base_actions[t-1]，不可调整
    # ------------------------------------------------------------------
    base_int = base_actions.astype(np.intp)
    adjustable = np.ones(h, dtype=bool)
    if step_idx > 0:
        adjustable[:step_idx] = False
    else:
        # step_idx == 0 时，第 0 步始终可调整（无前一步比较）
        adjustable[0] = True
        # 从第 1 步开始检查
        adjustable[1:] = base_int[1:] == base_int[:-1]
    if step_idx > 0:
        adjustable[step_idx:] = True
        # 对 step_idx 之后的步，检查 base_action 变化
        change_mask = np.zeros(h, dtype=bool)
        change_mask[1:] = base_int[1:] != base_int[:-1]
        adjustable[step_idx + 1:] &= ~change_mask[step_idx + 1:]
        # step_idx 本身: 若 step_idx > 0，需检查与前一步是否不同
        if change_mask[step_idx]:
            adjustable[step_idx] = False

    adapt_steps = np.nonzero(adjustable)[0]  # 可调整步的索引数组

    if len(adapt_steps) == 0:
        return []

    # ------------------------------------------------------------------
    # Step 2: 构建候选动作矩阵 (C, h)
    # 每个可调整步 × 2 种 a_ref → C = len(adapt_steps) * 2
    # ------------------------------------------------------------------
    num_adapt = len(adapt_steps)
    C = num_adapt * 2  # 候选总数

    # 记录每个候选对应的 (adapt_step, a_ref)
    # 前 num_adapt 个: a_ref = -1 (action=0)
    # 后 num_adapt 个: a_ref = +1 (action=2)
    candidate_adapt_steps = np.concatenate([adapt_steps, adapt_steps])
    candidate_a_refs = np.empty(C, dtype=np.intp)
    candidate_a_refs[:num_adapt] = -1
    candidate_a_refs[num_adapt:] = 1

    # 用 base_actions 广播填充 (C, h)，然后在各候选的 adapt_step 处覆写
    actions_matrix = np.tile(base_int, (C, 1))  # (C, h)
    row_indices = np.arange(C)
    adapted_actions = np.empty(C, dtype=np.intp)
    adapted_actions[:num_adapt] = 0   # a_ref=-1 → action=0
    adapted_actions[num_adapt:] = 2   # a_ref=+1 → action=2
    actions_matrix[row_indices, candidate_adapt_steps] = adapted_actions

    # ------------------------------------------------------------------
    # Step 3: 动作 → 持仓方向 → 持仓量 (C, h)
    # ------------------------------------------------------------------
    directions = _ACTION_TO_DIRECTION[actions_matrix]  # (C, h), 值域 {-1, 0, 1}
    positions = directions * m                          # (C, h), 值域 {-m, 0, m}

    # ------------------------------------------------------------------
    # Step 4: 持仓变化 delta_positions (C, h)
    # position[t=0] 的前一步持仓为 0 (flat)
    # ------------------------------------------------------------------
    prev_positions = np.zeros((C, h), dtype=positions.dtype)
    prev_positions[:, 1:] = positions[:, :-1]
    delta_positions = positions - prev_positions  # (C, h)

    # ------------------------------------------------------------------
    # Step 5: 交易成本 = 佣金 + LOB slippage
    # ------------------------------------------------------------------
    abs_delta = np.abs(delta_positions).astype(np.float64)  # (C, h)
    prices_f = prices.astype(np.float64)

    # 佣金: δ × |ΔP| × p_mark
    commission_costs = commission_rate * abs_delta * prices_f[np.newaxis, :]  # (C, h)

    # LOB slippage
    if states is not None:
        ask_p, ask_s, bid_p, bid_s = _extract_lob_arrays(states)
        slippage_costs = _vectorized_lob_slippage(
            delta_positions.astype(np.float64), prices_f,
            ask_p, ask_s, bid_p, bid_s,
        )
    else:
        slippage_costs = np.zeros_like(commission_costs)

    execution_costs = commission_costs + slippage_costs  # (C, h)

    # 仅在有持仓变化时才有成本（与原始逻辑一致）
    no_change_mask = delta_positions == 0
    execution_costs[no_change_mask] = 0.0

    # ------------------------------------------------------------------
    # Step 6: 逐步收益 & 总收益
    # r_t = position[t] × (price[t+1] - price[t]) - execution_cost[t]
    # 最后一步 price_diff = 0
    # ------------------------------------------------------------------
    price_diffs = np.zeros(h, dtype=np.float64)
    price_diffs[:-1] = prices_f[1:] - prices_f[:-1]

    positions_f = positions.astype(np.float64)
    step_rewards = positions_f * price_diffs[np.newaxis, :] - execution_costs  # (C, h)
    total_returns = step_rewards.sum(axis=1)  # (C,)

    # ------------------------------------------------------------------
    # Step 7: 取 top-5 并排序
    # ------------------------------------------------------------------
    k = min(5, C)
    if k < C:
        # argpartition 比完整排序更快: O(C) vs O(C log C)
        top_k_indices = np.argpartition(total_returns, -k)[-k:]
    else:
        top_k_indices = np.arange(C)

    # 对 top-k 按收益降序排列
    sorted_within = np.argsort(total_returns[top_k_indices])[::-1]
    top_k_sorted = top_k_indices[sorted_within]

    result: List[Tuple[int, int, float]] = []
    for idx in top_k_sorted:
        result.append((
            int(candidate_adapt_steps[idx]),
            int(candidate_a_refs[idx]),
            float(total_returns[idx]),
        ))

    return result


# ---------------------------------------------------------------------------
# 保留原始标量版本供测试对照和向后兼容
# ---------------------------------------------------------------------------

def _simulate_adaptation(
    prices: np.ndarray,
    base_actions: np.ndarray,
    adapt_step: int,
    a_ref: int,
    m: int,
    commission_rate: float,
    states: list[dict] | None = None,
) -> float:
    """模拟在指定步执行一次调整后的 horizon 总收益（标量版本）。

    保留此函数用于:
    - 单元测试中的逐条验证
    - 向后兼容已有调用方

    # 按 Eq. 6 规则：
    #   在 adapt_step 之前使用 base_actions
    #   在 adapt_step 处应用调整: a_ref=-1 → action=0, a_ref=1 → action=2
    #   在 adapt_step 之后恢复 base_actions
    #
    # Section 3.1: O_t = C(|ΔP|) - |ΔP| × p_mark + δ × |ΔP| × p_mark

    Args:
        prices: horizon 价格序列 shape (h,)
        base_actions: 基础动作序列 shape (h,)
        adapt_step: 执行调整的步索引
        a_ref: 调整信号 ∈ {-1, 1}
        m: 最大持仓量
        commission_rate: 佣金率
        states: horizon 状态序列 list[dict]（polars row dicts），可选

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

        # Section 3.1: O_t = slippage + commission
        delta_pos = new_position - position
        if delta_pos != 0:
            abs_delta = abs(delta_pos)
            if states is not None:
                slippage = TradingEnv.compute_lob_slippage(
                    delta_pos, states[t], prices[t]
                )
            else:
                slippage = 0.0
            commission = commission_rate * abs_delta * prices[t]
            execution_cost = slippage + commission
        else:
            execution_cost = 0.0

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
