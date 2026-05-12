"""Phase I validation Layer 0: DP teacher quality raw metrics。

文件功能说明:
    本文件负责计算第零层 DP teacher 数据质量的 raw metrics，包括 teacher 相对
    flat baseline 的优势、胜率、弱机会比例、手续费敏感性、市场形态覆盖率以及
    头部收益剔除后的剩余优势。

设计边界:
    - 只计算 raw metric，不判断 pass/fail；
    - 不访问模型、DataLoader 或文件系统；
    - hard gate 判定统一交给 ``phase1_validation_rules.py``；
    - 缺失价格等不可计算输入统一返回 ``nan``，由 rules 层按 hard gate 失败处理。

使用场景:
    ``Phase1CodebookEvaluator`` 收集 train/validation snapshot 后，调用
    ``compute_teacher_quality_metrics()`` 生成 ``Phase1TeacherQualityMetrics``，
    再交给 rules、score、report 和 checkpoint selector 消费。
"""

from __future__ import annotations

import numpy as np

from src.utils import ActionExecutionCalculator, nan_value as _nan

from ...metrics import (
    Phase1EvaluationSnapshot,
    Phase1LayerComputation,
    Phase1TeacherQualityMetrics,
    Phase1ValidationRuntimeConfig,
)
from .layer2_behavior_quality import classify_market_morphology


_EPS = 1e-12


def _prices_2d(prices: np.ndarray | None) -> np.ndarray | None:
    """把价格数组标准化为二维 ``[sample, horizon]``。

    输入参数:
        prices: 原始价格数组，可为 ``[N, H]``、``[N, H, 1]`` 或 ``None``。

    输出:
        二维 ``np.ndarray``；输入缺失、维度不合法或 horizon 不足时返回 ``None``。

    使用场景:
        收益执行、手续费敏感性和 market morphology 计算前统一价格形状。
    """

    if prices is None:
        return None
    values = np.asarray(prices, dtype=np.float64)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim != 2 or values.shape[1] < 2:
        return None
    return values


def _demo_returns(
    snapshot: Phase1EvaluationSnapshot,
    *,
    fee_rate: float,
) -> np.ndarray:
    """计算 DP teacher 每条 horizon 的收益。

    输入参数:
        snapshot: validation snapshot，读取 ``demo_rewards``、``prices`` 和
            ``demo_actions``。
        fee_rate: 当需要根据价格和动作重算收益时使用的手续费率。

    输出:
        ``[N]`` 形状的 teacher return 数组。优先使用 snapshot 中已有 reward；
        reward 不可用时回退到价格执行口径。

    使用场景:
        计算 teacher advantage、win rate 和收益集中度。
    """

    rewards = np.asarray(snapshot.demo_rewards, dtype=np.float64)
    if rewards.ndim == 3 and rewards.shape[-1] == 1:
        rewards = rewards[..., 0]
    if rewards.ndim == 2 and np.all(np.isfinite(rewards)):
        return np.sum(rewards, axis=1)

    price_values = _prices_2d(snapshot.prices)
    if price_values is None:
        return np.full(np.asarray(snapshot.demo_actions).shape[0], _nan())
    return ActionExecutionCalculator.execute_actions(
        price_values,
        snapshot.demo_actions,
        fee_rate,
    ).returns


def _top_removed_total_advantage(advantages: np.ndarray, top_ratio: float) -> float:
    """剔除头部优势样本后计算剩余总优势。

    输入参数:
        advantages: 每条 horizon 相对 flat baseline 的收益优势。
        top_ratio: 需要剔除的头部样本比例，例如 0.05 表示剔除最高 5%。

    输出:
        剩余样本的优势总和；无有效样本时返回 NaN。

    使用场景:
        检查 teacher 收益是否过度依赖少数尾部样本。
    """

    values = advantages[np.isfinite(advantages)]
    if values.size == 0:
        return _nan()
    remove_count = int(np.ceil(values.size * max(0.0, min(1.0, top_ratio))))
    if remove_count <= 0:
        return float(np.sum(values))
    order = np.argsort(values)
    kept = values[order[:-remove_count]]
    if kept.size == 0:
        return _nan()
    return float(np.sum(kept))


def compute_teacher_quality_metrics(
    *,
    train_snapshot: Phase1EvaluationSnapshot,
    val_snapshot: Phase1EvaluationSnapshot,
    runtime_config: Phase1ValidationRuntimeConfig,
) -> Phase1LayerComputation:
    """计算 Layer 0 DP teacher 质量 raw metrics。

    功能说明:
        基于 validation snapshot 计算 teacher 是否值得被 VQ codebook 学习。
        ``train_snapshot`` 仅用于保持五层 calculator 入口风格一致，当前实现
        使用 validation 数据表达 out-of-sample teacher 质量。

    输入参数:
        train_snapshot: 训练集 snapshot，当前不参与计算。
        val_snapshot: 验证集 snapshot，读取 teacher action/reward/price。
        runtime_config: validation 运行参数，提供手续费率和头部收益剔除比例。

    输出:
        ``Phase1LayerComputation``，其中 ``metrics`` 为
        ``Phase1TeacherQualityMetrics``。

    使用场景:
        full checkpoint validation 的第一层 raw metric 计算；结果交给
        ``evaluate_teacher_quality_rules()`` 判定 hard gate。
    """

    del train_snapshot
    dp_returns = _demo_returns(val_snapshot, fee_rate=runtime_config.fee_rate)
    flat_returns = np.zeros_like(dp_returns)
    advantages = dp_returns - flat_returns

    price_values = _prices_2d(val_snapshot.prices)
    if price_values is None:
        fee_sensitivity = _nan()
        morphology_coverage = _nan()
    else:
        doubled_fee_returns = ActionExecutionCalculator.execute_actions(
            price_values,
            val_snapshot.demo_actions,
            runtime_config.fee_rate * 2.0,
        ).returns
        fee_sensitivity = float(
            np.nansum(doubled_fee_returns - flat_returns)
            / (np.nansum(advantages) + _EPS)
        )
        labels = classify_market_morphology(
            price_values,
            fee_rate=runtime_config.fee_rate,
        )
        morphology_coverage = float(np.mean(labels != "neutral")) if labels.size else _nan()

    metrics = Phase1TeacherQualityMetrics(
        dp_advantage_vs_flat=float(np.nanmean(advantages)),
        dp_win_rate_vs_flat=float(np.nanmean(dp_returns > flat_returns)),
        near_zero_opportunity_ratio=float(
            np.nanmean(np.abs(advantages) < runtime_config.fee_rate)
        ),
        fee_sensitivity=fee_sensitivity,
        morphology_coverage=morphology_coverage,
        dp_return_concentration_after_top5_removed=_top_removed_total_advantage(
            advantages,
            runtime_config.top_contribution_ratio,
        ),
    )
    return Phase1LayerComputation(
        layer_id=0,
        layer_name="teacher_quality",
        metrics=metrics,
    )


__all__ = ["compute_teacher_quality_metrics"]
