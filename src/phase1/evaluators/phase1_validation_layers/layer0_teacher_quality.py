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
    Phase1TeacherQualityComputation,
    Phase1TeacherQualityMetrics,
    Phase1TeacherQualityPayload,
    Phase1ValidationRuntimeConfig,
)
from .layer2_behavior_quality import (
    classify_market_morphology as _classify_market_morphology,
)


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


def compute_flat_returns(
    prices: np.ndarray | None,
    *,
    sample_count: int | None = None,
) -> np.ndarray:
    """计算 flat baseline 的逐 horizon 收益。

    输入参数:
        prices: 价格数组，可为 ``[N, H]``、``[N, H, 1]`` 或 ``None``。
        sample_count: 当价格缺失但调用方已知样本数时，用于返回对齐长度的
            flat baseline。

    输出:
        ``[N]`` 形状的全 0 收益数组。flat baseline 表示全程空仓、不交易；
        第一版不计资金利息或持仓成本。

    使用场景:
        作为 DP teacher return 的对照基准 ``R_flat``。
    """

    price_values = _prices_2d(prices)
    if price_values is not None:
        return np.zeros(price_values.shape[0], dtype=np.float64)
    if sample_count is not None:
        return np.zeros(int(sample_count), dtype=np.float64)
    return np.asarray([], dtype=np.float64)


def compute_demo_returns(
    snapshot: Phase1EvaluationSnapshot,
    *,
    fee_rate: float = 0.0,
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
        snapshot.depthprices,
    ).returns


def compute_fee_sensitivity(
    prices: np.ndarray | None,
    actions: np.ndarray,
    fee_rate: float,
    *,
    depthprices: np.ndarray | None = None,
    original_advantages: np.ndarray | None = None,
) -> float:
    """计算 teacher 策略对手续费的敏感性。

    输入参数:
        prices: 价格数组，可为 ``[N, H]``、``[N, H, 1]`` 或 ``None``。
        actions: DP teacher 动作数组，形状为 ``[N, H]``。
        fee_rate: 原始手续费率；函数使用 ``fee_rate * 2`` 执行敏感性检查。
        original_advantages: 可选原始优势数组。Layer 0 主流程传入
            ``R_DP - R_flat``，保证分母和 teacher return 口径一致；缺失时函数
            使用原始 ``fee_rate`` 下的 execution return 作为分母。

    输出:
        翻倍手续费后总优势保留比例。价格缺失或不可计算时返回 NaN。

    使用场景:
        过滤对手续费极度敏感、优势容易被交易成本侵蚀的 teacher 数据。
    """

    price_values = _prices_2d(prices)
    if price_values is None:
        return _nan()

    if original_advantages is None:
        original_returns = ActionExecutionCalculator.execute_actions(
            price_values,
            actions,
            fee_rate,
            depthprices,
        ).returns
        denominator_values = original_returns
    else:
        denominator_values = np.asarray(original_advantages, dtype=np.float64)

    doubled_fee_returns = ActionExecutionCalculator.execute_actions(
        price_values,
        actions,
        fee_rate * 2.0,
        depthprices,
    ).returns
    numerator = np.nansum(doubled_fee_returns)
    denominator = np.nansum(denominator_values)
    return float(numerator / (denominator + _EPS))


def compute_top_removed_total_advantage(
    advantages: np.ndarray,
    top_ratio: float,
) -> float:
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
    if values.size == 1:
        remove_count = 0
    else:
        remove_count = min(remove_count, values.size - 1)
    if remove_count <= 0:
        return float(np.sum(values))
    order = np.argsort(values)
    kept = values[order[:-remove_count]]
    if kept.size == 0:
        return _nan()
    return float(np.sum(kept))


def classify_market_morphology(
    prices: np.ndarray | None,
    *,
    fee_rate: float = 0.0002,
) -> np.ndarray:
    """根据价格路径分类市场形态。

    输入参数:
        prices: 价格数组，可为 ``[N, H]``、``[N, H, 1]`` 或 ``None``。
        fee_rate: 手续费率，用于底层 morphology helper 设置 neutral band。

    输出:
        ``[N]`` 形状的标签数组；价格缺失或不可用时返回空数组。标签包括
        ``uptrend``、``downtrend``、``reversal-up``、``reversal-down``、
        ``range-high-vol``、``range-low-vol``、``volatile-mixed`` 和
        ``neutral``。

    使用场景:
        Layer 0 用于计算非 neutral 市场覆盖率，Layer 2/3/4 可复用同一分类口径。
    """

    return _classify_market_morphology(prices, fee_rate=fee_rate)


def compute_teacher_quality_metrics(
    *,
    train_snapshot: Phase1EvaluationSnapshot,
    val_snapshot: Phase1EvaluationSnapshot,
    runtime_config: Phase1ValidationRuntimeConfig,
) -> Phase1TeacherQualityComputation:
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
        ``Phase1TeacherQualityComputation``，其中 ``metrics`` 为
        ``Phase1TeacherQualityMetrics``。

    使用场景:
        full checkpoint validation 的第一层 raw metric 计算；结果交给
        ``evaluate_teacher_quality_rules()`` 判定 hard gate。
    """

    del train_snapshot
    dp_returns = compute_demo_returns(val_snapshot, fee_rate=runtime_config.fee_rate)
    flat_returns = compute_flat_returns(
        val_snapshot.prices,
        sample_count=dp_returns.shape[0],
    )
    advantages = dp_returns - flat_returns

    price_values = _prices_2d(val_snapshot.prices)
    if price_values is None:
        fee_sensitivity = _nan()
        morphology_coverage = _nan()
        missing_reason = "missing_prices"
    else:
        fee_sensitivity = compute_fee_sensitivity(
            price_values,
            val_snapshot.demo_actions,
            runtime_config.fee_rate,
            depthprices=val_snapshot.depthprices,
            original_advantages=advantages,
        )
        labels = classify_market_morphology(
            price_values,
            fee_rate=runtime_config.fee_rate,
        )
        morphology_coverage = float(np.mean(labels != "neutral")) if labels.size else _nan()
        missing_reason = None

    metrics = Phase1TeacherQualityMetrics(
        dp_advantage_vs_flat=float(np.nanmean(advantages)),
        dp_win_rate_vs_flat=float(np.nanmean(dp_returns > flat_returns)),
        near_zero_opportunity_ratio=float(
            np.nanmean(np.abs(advantages) < runtime_config.fee_rate)
        ),
        fee_sensitivity=fee_sensitivity,
        morphology_coverage=morphology_coverage,
        dp_return_concentration_after_top5_removed=compute_top_removed_total_advantage(
            advantages,
            runtime_config.top_contribution_ratio,
        ),
    )
    return Phase1TeacherQualityComputation(
        metrics=metrics,
        teacher_quality_payload=Phase1TeacherQualityPayload(
            dp_returns=dp_returns,
            flat_returns=flat_returns,
            advantages=advantages,
            missing_reason=missing_reason,
        ),
    )


__all__ = [
    "classify_market_morphology",
    "compute_demo_returns",
    "compute_fee_sensitivity",
    "compute_flat_returns",
    "compute_teacher_quality_metrics",
    "compute_top_removed_total_advantage",
]
