"""Selector 行为指标: action dominance / active archetype ratio / dead code usage。

设计文档锚点: Phase II 执行计划 §Step 6。
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List


def action_dominance_ratio(actions: List[int], num_codes: int) -> float:
    """计算 action dominance ratio: 最常选择的 archetype 占比。

    Parameters
    ----------
    actions : 所有 horizon 选择的 archetype id 列表。
    num_codes : archetype 总数 K。

    Returns
    -------
    float : 最高频 archetype 的占比。
    """
    if not actions:
        return 0.0
    counts = Counter(actions)
    max_count = max(counts.values())
    return max_count / len(actions)


def active_archetype_ratio(actions: List[int], num_codes: int) -> float:
    """计算 active archetype ratio: 被使用的 archetype 占比。

    Parameters
    ----------
    actions : 所有 horizon 选择的 archetype id 列表。
    num_codes : archetype 总数 K。

    Returns
    -------
    float : 被使用的 archetype 数 / K。
    """
    if not actions or num_codes == 0:
        return 0.0
    unique = len(set(actions))
    return unique / num_codes


def dead_code_usage_check(
    actions: List[int],
    dead_code_mask: List[bool],
) -> Dict[str, Any]:
    """检查 selector 是否选择了 dead code。

    Returns
    -------
    dict : 包含 dead_code_selected_count / dead_code_selected_ratio。
    """
    if not actions or not dead_code_mask:
        return {"dead_code_selected_count": 0, "dead_code_selected_ratio": 0.0}

    dead_set = {i for i, m in enumerate(dead_code_mask) if m}
    dead_count = sum(1 for a in actions if a in dead_set)
    return {
        "dead_code_selected_count": dead_count,
        "dead_code_selected_ratio": dead_count / max(len(actions), 1),
    }
