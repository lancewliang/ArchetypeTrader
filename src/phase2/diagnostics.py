"""Phase II 诊断工具模块

本模块提供训练过程中的诊断和统计工具函数，用于分析执行细节和性能指标。

Functions:
    histogram_counts: 把离散标签序列转成固定长度直方图计数
    format_histogram_from_counts: 把直方图计数格式化成紧凑日志字符串
    aggregate_execution_diagnostics: 汇总一批 horizon 执行诊断指标
"""

from typing import Any

import numpy as np


def histogram_counts(values: np.ndarray | list[int], num_bins: int) -> np.ndarray:
    """把离散标签序列转成固定长度直方图计数。

    Args:
        values: 离散标签序列
        num_bins: 直方图 bin 数量

    Returns:
        长度为 num_bins 的计数数组
    """
    values_np = np.asarray(values, dtype=np.int64).reshape(-1)
    if values_np.size == 0:
        return np.zeros(num_bins, dtype=np.int64)
    valid = values_np[(values_np >= 0) & (values_np < num_bins)]
    if valid.size == 0:
        return np.zeros(num_bins, dtype=np.int64)
    return np.bincount(valid, minlength=num_bins).astype(np.int64)


def format_histogram_from_counts(counts: np.ndarray | list[int]) -> str:
    """把直方图计数格式化成紧凑日志字符串。

    Args:
        counts: 直方图计数数组

    Returns:
        格式化的字符串，如 "[0:10, 1:20, 2:15]"
    """
    counts_np = np.asarray(counts, dtype=np.int64).reshape(-1)
    return "[" + ", ".join(f"{idx}:{int(v)}" for idx, v in enumerate(counts_np.tolist())) + "]"


def aggregate_execution_diagnostics(horizon_details: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总一批 horizon 执行诊断指标。

    功能说明:
        把 decoder 在环境中的 horizon 级执行结果拆成 gross pnl / cost / turnover
        / 换仓次数等指标，避免只看最终 reward 而无法定位负收益来源。

    Args:
        horizon_details: 每个 horizon 的执行细节字典列表，包含：
            - horizon_return: horizon 总收益
            - gross_pnl: 毛利润
            - execution_cost_total: 总执行成本
            - commission_total: 总佣金
            - slippage_total: 总滑点
            - num_position_changes: 持仓变化次数
            - num_direct_flips: 直接翻转次数
            - turnover_total: 总换手量
            - decoder_action_histogram: decoder action 直方图

    Returns:
        汇总的诊断指标字典，包含所有指标的平均值和 action 直方图
    """
    if not horizon_details:
        return {
            "avg_return": 0.0,
            "avg_gross_pnl": 0.0,
            "avg_execution_cost": 0.0,
            "avg_commission": 0.0,
            "avg_slippage": 0.0,
            "avg_position_changes": 0.0,
            "avg_direct_flips": 0.0,
            "avg_turnover": 0.0,
            "decoder_action_histogram": format_histogram_from_counts(np.zeros(3, dtype=np.int64)),
        }

    decoder_hist = np.sum(
        [np.asarray(item["decoder_action_histogram"], dtype=np.int64) for item in horizon_details],
        axis=0,
    )

    return {
        "avg_return": float(np.mean([item["horizon_return"] for item in horizon_details])),
        "avg_gross_pnl": float(np.mean([item["gross_pnl"] for item in horizon_details])),
        "avg_execution_cost": float(np.mean([item["execution_cost_total"] for item in horizon_details])),
        "avg_commission": float(np.mean([item["commission_total"] for item in horizon_details])),
        "avg_slippage": float(np.mean([item["slippage_total"] for item in horizon_details])),
        "avg_position_changes": float(np.mean([item["num_position_changes"] for item in horizon_details])),
        "avg_direct_flips": float(np.mean([item["num_direct_flips"] for item in horizon_details])),
        "avg_turnover": float(np.mean([item["turnover_total"] for item in horizon_details])),
        "decoder_action_histogram": format_histogram_from_counts(decoder_hist),
    }
