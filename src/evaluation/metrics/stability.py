"""Epoch 间 code 分配稳定性 + horizon 边界衔接指标."""
from __future__ import annotations

import math
from typing import Dict, List, Sequence


def epoch_code_stability(best_code_ids, last_code_ids) -> float:
    """raw 一致率: ``mean(best == last)``。

    与 ``matched_epoch_code_stability`` 相比，此版本对纯 code-id 交换敏感
    （会判定为不稳定），适合检测 codebook 漂移。
    """
    if not best_code_ids:
        return 0.0
    matches = sum(1 for a, b in zip(best_code_ids, last_code_ids) if a == b)
    return matches / len(best_code_ids)


def matched_epoch_code_stability(
    best_code_ids: Sequence[int],
    last_code_ids: Sequence[int],
    best_codebook: Sequence[Sequence[float]],
    last_codebook: Sequence[Sequence[float]],
) -> float:
    """先用 codebook 距离做 Hungarian matching，再算一致率。

    用途
    ----
    区分两种"不稳定": 真正的标签漂移 vs 纯 code-id 交换。
    后者其实是 stable 的（语义没变，只是 K 个 cluster 重新编号）。

    实现
    ----
    使用最小成本二分图匹配求全局最优 code-id 对齐；K 通常较小，不引入
    scipy 依赖。
    """
    K = min(len(best_codebook), len(last_codebook))
    if K == 0 or not best_code_ids:
        return 0.0
    # cost[i][j] = ||best[i] - last[j]||
    cost = [[_l2(best_codebook[i], last_codebook[j]) for j in range(K)] for i in range(K)]
    mapping = _hungarian_assignment(cost)

    # 应用 mapping: best 的 code i → last 的 code mapping[i]
    matches = sum(
        1
        for a, b in zip(best_code_ids, last_code_ids)
        if mapping.get(int(a), -1) == int(b)
    )
    return matches / len(best_code_ids)


def _hungarian_assignment(cost: Sequence[Sequence[float]]) -> Dict[int, int]:
    """返回 ``row -> col`` 的最小成本一对一匹配。"""
    n = len(cost)
    if n == 0:
        return {}

    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (n + 1)
        used = [False] * (n + 1)
        way = [0] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break

        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment = [0] * n
    for j in range(1, n + 1):
        if p[j] != 0:
            assignment[p[j] - 1] = j - 1
    return {i: assignment[i] for i in range(n)}


def codebook_displacement(
    current_codebook: Sequence[Sequence[float]],
    previous_codebook: Sequence[Sequence[float]],
) -> Dict[int, float]:
    """每个 code embedding 在相邻 epoch 之间的位移；返回 ``Dict[code_id, l2_distance]``。

    全 0 输出表示 codebook 未动；单测可用此作为 sanity check。
    用于 ``codebook_displacement_by_epoch`` 诊断字段。
    """
    out: Dict[int, float] = {}
    for i, (a, b) in enumerate(zip(current_codebook, previous_codebook)):
        out[i] = _l2(a, b)
    return out


def horizon_boundary_metrics(
    boundary_positions: List[tuple],
    boundary_books: List,
    cost_model,
):
    """计算 ``horizon_boundary_turnover_cost`` 与 ``horizon_boundary_position_consistency``。

    Parameters
    ----------
    boundary_positions : ``[(prev_terminal_position, next_initial_target_position), ...]``。
    boundary_books : 与 ``boundary_positions`` 顺序一致的成交盘口（用于 cost_model.execute）。
    cost_model : ``LobDepthCostModel`` 实例；与 DP / replay 共用以保证可比。

    Returns
    -------
    ``{"horizon_boundary_turnover_cost": float, "horizon_boundary_position_consistency": float}``

    Notes
    -----
    - turnover_cost: 平均每个边界的 fee + slippage。
    - position_consistency: ``prev == next_target`` 的边界占比；越高越好。
    设计 §4.7: 该指标用于估算 Phase II selector 独立选 horizon 时的边界成本。
    """
    total_cost = 0.0
    consistent = 0
    n = len(boundary_positions)
    for (prev, target), book in zip(boundary_positions, boundary_books):
        if prev == target:
            consistent += 1
            continue
        result = cost_model.execute(
            prev_position=prev, target_position=target, execution_book=book
        )
        total_cost += result.cost
    return {
        "horizon_boundary_turnover_cost": total_cost / max(n, 1),
        "horizon_boundary_position_consistency": consistent / max(n, 1),
    }


def _l2(a: Sequence[float], b: Sequence[float]) -> float:
    s = 0.0
    for x, y in zip(a, b):
        s += (x - y) ** 2
    return math.sqrt(s)
