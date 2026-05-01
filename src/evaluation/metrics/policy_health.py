"""PPO 策略健康诊断指标: approx_kl / clip_fraction / explained_variance / kl_demo_dominance。

设计文档锚点: Phase II 执行计划 §Step 6。
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

import math


def compute_approx_kl(old_log_probs: List[float], new_log_probs: List[float]) -> float:
    """计算近似 KL 散度。"""
    if not old_log_probs or not new_log_probs:
        return 0.0
    total = 0.0
    for old_lp, new_lp in zip(old_log_probs, new_log_probs):
        log_ratio = new_lp - old_lp
        total += (math.exp(log_ratio) - 1) - log_ratio
    return total / len(old_log_probs)


def compute_clip_fraction(
    old_log_probs: List[float],
    new_log_probs: List[float],
    clip_ratio: float,
) -> float:
    """计算 clip fraction。"""
    if not old_log_probs or not new_log_probs:
        return 0.0
    clipped = 0
    for old_lp, new_lp in zip(old_log_probs, new_log_probs):
        ratio = math.exp(new_lp - old_lp)
        if abs(ratio - 1.0) > clip_ratio:
            clipped += 1
    return clipped / len(old_log_probs)


def compute_explained_variance(
    values: List[float],
    returns: List[float],
) -> float:
    """计算 explained variance。"""
    if len(values) < 2 or len(returns) < 2:
        return 0.0
    mean_ret = sum(returns) / len(returns)
    var_ret = sum((r - mean_ret) ** 2 for r in returns) / len(returns)
    if var_ret < 1e-10:
        return 0.0
    residuals = [v - r for v, r in zip(values, returns)]
    var_res = sum(r ** 2 for r in residuals) / len(residuals)
    return 1.0 - var_res / var_ret


def compute_kl_demo_dominance_ratio(
    kl_demo_loss: float,
    policy_loss: float,
) -> float:
    """计算 kl_demo_loss 占总 policy loss 的比例。"""
    total = abs(policy_loss) + abs(kl_demo_loss)
    if total < 1e-10:
        return 0.0
    return abs(kl_demo_loss) / total


def per_archetype_reward_stats(
    actions: List[int],
    rewards: List[float],
    num_codes: int,
) -> Dict[int, Dict[str, float]]:
    """按 archetype 聚合 reward 的 mean 和 std。"""
    buckets: Dict[int, List[float]] = defaultdict(list)
    for a, r in zip(actions, rewards):
        buckets[a].append(r)

    result: Dict[int, Dict[str, float]] = {}
    for code_id in range(num_codes):
        vals = buckets.get(code_id, [])
        if vals:
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / max(len(vals) - 1, 1)
            std = math.sqrt(max(var, 0.0))
            result[code_id] = {"mean": mean, "std": std, "count": len(vals)}
        else:
            result[code_id] = {"mean": 0.0, "std": 0.0, "count": 0}
    return result
