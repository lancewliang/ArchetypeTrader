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
    """先用 codebook 距离做 Hungarian-like matching，再算一致率。

    用途
    ----
    区分两种"不稳定": 真正的标签漂移 vs 纯 code-id 交换。
    后者其实是 stable 的（语义没变，只是 K 个 cluster 重新编号）。

    实现
    ----
    简化版贪心匹配: 反复挑选 cost 最小的 (best, last) 对，直到匹配满 K。
    K 通常 ≤ 16，性能完全够用；不用 scipy linear_sum_assignment 减少依赖。
    """
    K = len(best_codebook)
    if K == 0 or not best_code_ids:
        return 0.0
    # cost[i][j] = ||best[i] - last[j]||
    cost = [[_l2(best_codebook[i], last_codebook[j]) for j in range(K)] for i in range(K)]
    # 贪心匹配
    used_last = set()
    mapping: Dict[int, int] = {}
    for _ in range(K):
        best_pair = None
        best_val = float("inf")
        for i in range(K):
            if i in mapping:
                continue
            for j in range(K):
                if j in used_last:
                    continue
                if cost[i][j] < best_val:
                    best_val = cost[i][j]
                    best_pair = (i, j)
        if best_pair is None:
            break
        mapping[best_pair[0]] = best_pair[1]
        used_last.add(best_pair[1])

    # 应用 mapping: best 的 code i → last 的 code mapping[i]
    matches = sum(
        1
        for a, b in zip(best_code_ids, last_code_ids)
        if mapping.get(int(a), -1) == int(b)
    )
    return matches / len(best_code_ids)


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
