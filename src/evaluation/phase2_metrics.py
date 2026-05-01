"""Phase II 指标门面: 聚合 selection / portfolio / policy_health 子模块。

设计文档锚点: Phase II 执行计划 §Step 6。

类似 Phase I 的 phase1_metrics.py，为 Phase II evaluator 提供统一 API。
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List

from .metrics.policy_health import (
    compute_approx_kl,
    compute_clip_fraction,
    compute_explained_variance,
    compute_kl_demo_dominance_ratio,
    per_archetype_reward_stats,
)
from .metrics.portfolio import (
    build_equity_curve_summary,
    compute_boundary_cost,
    compute_calmar,
    compute_max_drawdown,
    compute_net_return,
    compute_sharpe,
    compute_sortino,
    compute_turnover,
)
from .metrics.selection import (
    action_dominance_ratio,
    active_archetype_ratio,
    dead_code_usage_check,
)


def phase2_composite_metrics(
    horizon_records: List[Dict[str, Any]],
    ppo_stats: Dict[str, float],
    num_codes: int,
    dead_code_mask: List[bool],
    annualization_factor: int = 525600,
) -> Dict[str, Any]:
    """计算 Phase II 的完整指标集。

    Parameters
    ----------
    horizon_records : per-horizon replay 记录列表。
    ppo_stats : PPO update 统计。
    num_codes : archetype 数量 K。
    dead_code_mask : dead code mask。
    annualization_factor : 年化因子。

    Returns
    -------
    dict : 包含所有 Phase II 指标的字典。
    """
    actions = [r.get("chosen_code", 0) for r in horizon_records]
    rewards = [r.get("reward_raw", 0.0) for r in horizon_records]
    boundary_costs = [r.get("boundary_cost", 0.0) for r in horizon_records]
    step_returns = []
    for r in horizon_records:
        step_returns.extend(r.get("step_returns", []))

    # Portfolio metrics
    net_return = compute_net_return(rewards)
    sharpe = compute_sharpe(step_returns, annualization_factor) if step_returns else 0.0
    sortino = compute_sortino(step_returns, annualization_factor) if step_returns else 0.0
    equity = build_equity_curve_summary(rewards)
    mdd = compute_max_drawdown(equity.per_horizon_cumulative_pnl)
    annual_ret = net_return * (annualization_factor / max(len(rewards), 1))
    calmar = compute_calmar(annual_ret, mdd)
    turnover = compute_turnover(actions)
    total_boundary_cost = compute_boundary_cost(boundary_costs)
    total_cost = sum(r.get("cost_paid", 0.0) for r in horizon_records)

    # Selection metrics
    dominance = action_dominance_ratio(actions, num_codes)
    active_ratio = active_archetype_ratio(actions, num_codes)
    dead_usage = dead_code_usage_check(actions, dead_code_mask)

    # Policy health
    per_arch_stats = per_archetype_reward_stats(actions, rewards, num_codes)

    return {
        "net_return": net_return,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": mdd,
        "calmar_ratio": calmar,
        "turnover": turnover,
        "total_boundary_cost": total_boundary_cost,
        "total_cost_paid": total_cost,
        "action_dominance_ratio": dominance,
        "active_archetype_ratio": active_ratio,
        "dead_code_usage": dead_usage,
        "per_archetype_reward_stats": per_arch_stats,
        "equity_curve_summary": asdict(equity),
        "num_horizons": len(horizon_records),
        # PPO stats passthrough
        "ppo_policy_loss": ppo_stats.get("policy_loss", 0.0),
        "ppo_value_loss": ppo_stats.get("value_loss", 0.0),
        "ppo_entropy_loss": ppo_stats.get("entropy_loss", 0.0),
        "ppo_kl_demo_loss": ppo_stats.get("kl_demo_loss", 0.0),
        "ppo_approx_kl": ppo_stats.get("approx_kl", 0.0),
        "ppo_clip_fraction": ppo_stats.get("clip_fraction", 0.0),
    }
