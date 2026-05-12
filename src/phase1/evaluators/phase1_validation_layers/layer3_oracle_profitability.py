"""Phase I validation Layer 3: oracle assigned-label profitability raw metrics。

文件功能说明:
    本文件负责计算 oracle assigned-label 条件下 decoder 策略的盈利性 raw metrics，
    包括 decoded advantage、win rate、random label baseline lift、teacher
    retention、downside control、risk-adjusted return、收益集中度、fee drag、
    turnover-return correlation、bad code ratio 和 dominant pair positive ratio。

设计边界:
    - 本层只证明 encoder 已分配 label 的 oracle 执行价值；
    - random label baseline 和 probe label 执行共享本文件的收益执行口径；
    - 不判断 hard gate pass/fail；
    - 不训练模型，不访问 DataLoader 或文件系统；
    - 缺失 prices 时收益类指标返回 NaN，由 rules 层失败。

使用场景:
    ``Phase1CodebookEvaluator`` 在 validation snapshot 收集完成后调用
    ``compute_oracle_profitability_metrics()``。其 ``extra_payload`` 中的
    ``per_code_profitability`` 会传给 Layer 2 复用。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.utils import ActionExecutionCalculator, ActionExecutionResult, nan_value as _nan

from ...metrics import (
    Phase1EvaluationSnapshot,
    Phase1LayerComputation,
    Phase1OracleProfitabilityMetrics,
    Phase1OracleProfitabilityThresholds,
    Phase1PerCodeProfitability,
    Phase1ValidationRuntimeConfig,
)
from .layer2_behavior_quality import classify_action_motif, classify_market_morphology


_EPS = 1e-12


def _demo_returns(
    snapshot: Phase1EvaluationSnapshot,
    runtime_config: Phase1ValidationRuntimeConfig,
) -> ActionExecutionResult:
    """计算 DP teacher 的执行收益结果。

    输入参数:
        snapshot: validation snapshot，读取 ``demo_rewards``、``prices`` 和
            ``demo_actions``。
        runtime_config: 提供手续费率。

    输出:
        ``ActionExecutionResult``。若 ``demo_rewards`` 可用，则 ``returns`` 优先使用
        reward 求和，gross/fee/turnover 仍由统一执行口径补充。

    使用场景:
        计算 decoded profitability 相对 DP teacher 的 retention ratio 和 downside。
    """

    rewards = np.asarray(snapshot.demo_rewards, dtype=np.float64)
    if rewards.ndim == 3 and rewards.shape[-1] == 1:
        rewards = rewards[..., 0]
    execution = ActionExecutionCalculator.execute_actions(
        snapshot.prices,
        snapshot.demo_actions,
        runtime_config.fee_rate,
    )
    if rewards.ndim == 2 and np.all(np.isfinite(rewards)):
        return ActionExecutionResult(
            returns=np.sum(rewards, axis=1),
            gross_returns=execution.gross_returns,
            fees=execution.fees,
            turnover=execution.turnover,
        )
    return execution


def _decode_labels(
    *,
    model: Any,
    states: np.ndarray,
    code_ids: np.ndarray,
    device: torch.device | str,
) -> np.ndarray:
    """用指定 code label 调用 decoder 生成动作。

    输入参数:
        model: ``ArchetypeVQModel`` 或兼容对象，需要提供 ``quantizer`` 和 ``decoder``。
        states: 状态序列数组，形状为 ``[N, H, state_dim]``。
        code_ids: 每个样本要使用的 code id，形状为 ``[N]``。
        device: decoder 推理设备。

    输出:
        ``[N, H]`` 形状的 argmax decoded action 数组。

    使用场景:
        random label baseline 需要用随机 code 重新 decode；Layer 4 probe label
        执行也使用相同逻辑。
    """

    if model is None:
        raise ValueError("model is required to decode label-conditioned actions")

    torch_device = torch.device(device)
    model = model.to(torch_device)
    model.eval()
    with torch.no_grad():
        state_tensor = torch.as_tensor(states, dtype=torch.float32, device=torch_device)
        label_tensor = torch.as_tensor(code_ids, dtype=torch.long, device=torch_device)
        z_q = model.quantizer.embedding_from_code(label_tensor)
        logits = model.decoder(state_tensor, z_q)
        return logits.argmax(dim=-1).cpu().numpy()


def _random_label_returns(
    *,
    model: Any,
    snapshot: Phase1EvaluationSnapshot,
    runtime_config: Phase1ValidationRuntimeConfig,
    device: torch.device | str,
) -> np.ndarray:
    """计算 random label baseline 的平均收益。

    输入参数:
        model: VQ 模型或兼容对象。
        snapshot: validation snapshot，提供 states、prices 和样本数量。
        runtime_config: 提供随机 trial 数、随机种子和手续费率。
        device: decoder 推理设备。

    输出:
        ``[N]`` 形状的 random label 平均收益数组。

    使用场景:
        计算 assigned label 相对随机 label 的 mean advantage 和 relative lift。
    """

    rng = np.random.default_rng(runtime_config.random_seed)
    num_codes = int(getattr(model, "num_archetypes", 0) or (np.max(snapshot.code_ids) + 1))
    trial_returns: list[np.ndarray] = []
    for _ in range(max(1, runtime_config.random_label_trials)):
        random_labels = rng.integers(
            low=0,
            high=max(1, num_codes),
            size=np.asarray(snapshot.code_ids).shape[0],
            dtype=np.int64,
        )
        random_actions = _decode_labels(
            model=model,
            states=snapshot.states,
            code_ids=random_labels,
            device=device,
        )
        trial_returns.append(
            ActionExecutionCalculator.execute_actions(
                snapshot.prices,
                random_actions,
                runtime_config.fee_rate,
            ).returns
        )
    return np.mean(np.stack(trial_returns, axis=0), axis=0)


def _risk_adjusted_return(returns: np.ndarray) -> float:
    """计算风险调整收益。

    输入参数:
        returns: 每条 horizon 的收益数组。

    输出:
        ``mean(returns) / std(returns)``；无有效样本时返回 NaN。

    使用场景:
        Layer 3 scoring 和 checkpoint tie-breaker。
    """

    values = returns[np.isfinite(returns)]
    if values.size == 0:
        return _nan()
    return float(np.mean(values) / (np.std(values) + _EPS))


def _max_drawdown(cumulative_returns: np.ndarray) -> float:
    """计算累计收益序列最大回撤。

    输入参数:
        cumulative_returns: 累计收益序列。

    输出:
        最大回撤值；无有效样本时返回 NaN。

    使用场景:
        计算 decoded downside control 相对 DP teacher 的比例。
    """

    values = cumulative_returns[np.isfinite(cumulative_returns)]
    if values.size == 0:
        return _nan()
    peak = np.maximum.accumulate(values)
    return float(np.max(peak - values))


def _top_contribution_ratio(returns: np.ndarray, top_ratio: float) -> float:
    """计算正收益中头部样本贡献比例。

    输入参数:
        returns: 每条 horizon 的收益或优势数组。
        top_ratio: 头部样本比例，例如 0.05。

    输出:
        头部正收益占总正收益的比例；无正收益时返回 NaN。

    使用场景:
        诊断 decoded profit 是否过度依赖少数尾部 horizon。
    """

    positive = returns[np.isfinite(returns) & (returns > 0)]
    if positive.size == 0:
        return _nan()
    count = max(1, int(np.ceil(positive.size * max(0.0, min(1.0, top_ratio)))))
    top = np.sort(positive)[-count:]
    return float(np.sum(top) / (np.sum(positive) + _EPS))


def _trimmed_mean(values: np.ndarray, trim_ratio: float) -> float:
    """计算双侧截尾均值。

    输入参数:
        values: 原始指标数组。
        trim_ratio: 两端各自剔除的比例，上限保护为 0.49。

    输出:
        截尾后的均值；无有效样本时返回 NaN。

    使用场景:
        计算去除尾部样本后的 decoded advantage。
    """

    finite = np.sort(values[np.isfinite(values)])
    if finite.size == 0:
        return _nan()
    trim = int(np.floor(finite.size * max(0.0, min(0.49, trim_ratio))))
    if trim > 0 and finite.size > 2 * trim:
        finite = finite[trim:-trim]
    return float(np.mean(finite)) if finite.size else _nan()


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    """计算带防御逻辑的 Pearson correlation。

    输入参数:
        left: 左侧数值数组。
        right: 右侧数值数组。

    输出:
        correlation；有效样本不足时返回 NaN，任一侧方差接近 0 时返回 0。

    使用场景:
        计算 turnover 与 decoded return 的相关性。
    """

    mask = np.isfinite(left) & np.isfinite(right)
    if np.sum(mask) < 2:
        return _nan()
    if np.std(left[mask]) <= _EPS or np.std(right[mask]) <= _EPS:
        return 0.0
    return float(np.corrcoef(left[mask], right[mask])[0, 1])


def _fee_drag(fees: np.ndarray, gross_returns: np.ndarray) -> float:
    """按设计口径计算 fee drag: total_fee / gross_profit。"""

    fee_values = fees[np.isfinite(fees)]
    gross_profit = gross_returns[np.isfinite(gross_returns) & (gross_returns > 0.0)]
    if fee_values.size == 0:
        return _nan()
    denominator = float(np.sum(gross_profit))
    if denominator <= 0.0:
        return float("inf")
    return float(np.sum(fee_values) / (denominator + _EPS))


def _per_code_profitability(
    *,
    code_ids: np.ndarray,
    decoded_advantage: np.ndarray,
    decoded_returns: np.ndarray,
    dp_advantage: np.ndarray,
    decoded_gross_returns: np.ndarray,
    decoded_fees: np.ndarray,
    thresholds: Phase1OracleProfitabilityThresholds,
) -> tuple[Phase1PerCodeProfitability, ...]:
    """计算每个 code 的 profitability 摘要。

    输入参数:
        code_ids: 每个样本的 assigned code id。
        decoded_advantage: decoded return 相对 flat 的优势。
        decoded_returns: decoded 策略净收益。
        dp_advantage: DP teacher 相对 flat 的优势。
        decoded_gross_returns: 每个样本的 decoded gross return。
        decoded_fees: 每个样本的手续费。

    输出:
        ``Phase1PerCodeProfitability`` tuple。

    使用场景:
        Layer 3 计算 bad code ratio；Layer 2 复用该结果计算 profitable code coverage。
    """

    output: list[Phase1PerCodeProfitability] = []
    for code_id in np.unique(code_ids):
        mask = code_ids == code_id
        if not np.any(mask):
            continue
        mean_advantage = float(np.nanmean(decoded_advantage[mask]))
        win_rate = float(np.nanmean(decoded_returns[mask] > 0.0))
        retention_ratio = float(
            np.nansum(decoded_advantage[mask]) / (np.nansum(dp_advantage[mask]) + _EPS)
        )
        fee_drag = _fee_drag(decoded_fees[mask], decoded_gross_returns[mask])
        output.append(
            Phase1PerCodeProfitability(
                code_id=int(code_id),
                mean_advantage=mean_advantage,
                win_rate=win_rate,
                retention_ratio=retention_ratio,
                fee_drag=fee_drag,
                passed=(
                    mean_advantage > 0.0
                    and win_rate >= thresholds.per_code_win_rate_min
                    and retention_ratio >= thresholds.per_code_retention_ratio_min
                    and fee_drag <= thresholds.per_code_fee_drag_max
                ),
            )
        )
    return tuple(output)


def _dominant_pair_positive_ratio(
    snapshot: Phase1EvaluationSnapshot,
    decoded_advantage: np.ndarray,
) -> float:
    """计算 active code dominant morphology-motif pair 的正优势比例。

    输入参数:
        snapshot: validation snapshot，读取 prices、decoded_actions 和 code_ids。
        decoded_advantage: 每条 horizon 的 decoded advantage。

    输出:
        dominant pair 正 mean advantage 的 active code 数量 / active code 数量；
        缺少 morphology 时返回 NaN。

    使用场景:
        Layer 3 hard gate 检查主要市场-行为组合是否具有正价值。
    """

    morphologies = classify_market_morphology(snapshot.prices)
    if morphologies.size != np.asarray(snapshot.code_ids).shape[0]:
        return _nan()
    motifs = classify_action_motif(snapshot.decoded_actions, snapshot.prices)
    pairs = np.asarray(
        [
            f"{morphology}:{motif}"
            for morphology, motif in zip(morphologies, motifs, strict=False)
        ],
        dtype=object,
    )
    code_ids = np.asarray(snapshot.code_ids, dtype=np.int64)
    positive = 0
    total = 0
    for code_id in np.unique(code_ids):
        code_mask = code_ids == code_id
        if not np.any(code_mask):
            continue
        pair_values, pair_counts = np.unique(pairs[code_mask], return_counts=True)
        if pair_values.size == 0:
            continue
        dominant_pair = pair_values[int(np.argmax(pair_counts))]
        mask = code_mask & (pairs == dominant_pair)
        total += 1
        positive += int(float(np.nanmean(decoded_advantage[mask])) > 0.0)
    return positive / total if total else _nan()


def compute_oracle_profitability_metrics(
    *,
    model: Any,
    val_snapshot: Phase1EvaluationSnapshot,
    runtime_config: Phase1ValidationRuntimeConfig,
    device: torch.device | str,
    thresholds: Phase1OracleProfitabilityThresholds | None = None,
) -> Phase1LayerComputation:
    """计算 Layer 3 oracle assigned-label 盈利性 raw metrics。

    功能说明:
        用 validation snapshot 中 encoder 已分配的 code 对 decoded actions 执行收益
        评估，并构造 random label baseline、per-code profitability 和 report payload。

    输入参数:
        model: VQ 模型或兼容对象，用于 random label baseline 重新 decode。
        val_snapshot: 验证集 snapshot，读取 prices、demo_actions、decoded_actions、
            states 和 code_ids。
        runtime_config: validation 运行参数，提供手续费率、random trials、seed 和
            top contribution ratio。
        device: decoder 推理设备。
        thresholds: oracle profitability 阈值配置，用于构造 per-code profitability
            的 ``passed`` 字段；不传则使用默认阈值。

    输出:
        ``Phase1LayerComputation``，其中 ``metrics`` 为
        ``Phase1OracleProfitabilityMetrics``，``extra_payload`` 包含 per-code
        profitability、decoded/DP/flat/random returns 和 random seed。

    使用场景:
        full checkpoint validation 的第三层盈利性 raw metric 计算；结果交给
        ``evaluate_oracle_profitability_rules()`` 判定 hard gate，并把 per-code
        profitability 传给 Layer 2。
    """

    thresholds = thresholds or Phase1OracleProfitabilityThresholds()
    dp_execution = _demo_returns(val_snapshot, runtime_config)
    decoded_execution = ActionExecutionCalculator.execute_actions(
        val_snapshot.prices,
        val_snapshot.decoded_actions,
        runtime_config.fee_rate,
    )
    flat_returns = np.zeros_like(decoded_execution.returns)
    random_returns = _random_label_returns(
        model=model,
        snapshot=val_snapshot,
        runtime_config=runtime_config,
        device=device,
    )

    decoded_advantage = decoded_execution.returns - flat_returns
    dp_advantage = dp_execution.returns - flat_returns
    random_advantage = random_returns - flat_returns
    per_code = _per_code_profitability(
        code_ids=np.asarray(val_snapshot.code_ids, dtype=np.int64),
        decoded_advantage=decoded_advantage,
        decoded_returns=decoded_execution.returns,
        dp_advantage=dp_advantage,
        decoded_gross_returns=decoded_execution.gross_returns,
        decoded_fees=decoded_execution.fees,
        thresholds=thresholds,
    )

    random_label_risk_adjusted_return = _risk_adjusted_return(random_returns)
    risk_adjusted_return = _risk_adjusted_return(decoded_execution.returns)

    metrics = Phase1OracleProfitabilityMetrics(
        mean_decoded_advantage_vs_flat=float(np.nanmean(decoded_advantage)),
        decoded_win_rate_vs_flat=float(
            np.nanmean(decoded_execution.returns > flat_returns)
        ),
        mean_advantage_vs_random_label=float(
            np.nanmean(decoded_execution.returns - random_returns)
        ),
        random_label_relative_lift=float(
            np.nanmean(decoded_execution.returns - random_returns)
            / (abs(np.nanmean(random_advantage)) + _EPS)
        ),
        retention_ratio=float(np.nansum(decoded_advantage) / (np.nansum(dp_advantage) + _EPS)),
        downside_control=float(
            _max_drawdown(np.cumsum(decoded_execution.returns))
            / (_max_drawdown(np.cumsum(dp_execution.returns)) + _EPS)
        ),
        risk_adjusted_return=risk_adjusted_return,
        top_5_contribution=_top_contribution_ratio(
            decoded_advantage,
            runtime_config.top_contribution_ratio,
        ),
        trimmed_decoded_advantage=_trimmed_mean(
            decoded_advantage,
            runtime_config.top_contribution_ratio,
        ),
        fee_drag=_fee_drag(decoded_execution.fees, decoded_execution.gross_returns),
        turnover_return_correlation=_safe_corr(
            decoded_execution.turnover,
            decoded_execution.returns,
        ),
        bad_code_ratio=float(
            np.mean([item.mean_advantage < 0.0 for item in per_code])
        )
        if per_code
        else _nan(),
        dominant_pair_positive_ratio=_dominant_pair_positive_ratio(
            val_snapshot,
            decoded_advantage,
        ),
        random_label_risk_adjusted_return=random_label_risk_adjusted_return,
        risk_adjusted_return_vs_random=(
            risk_adjusted_return - random_label_risk_adjusted_return
            if np.isfinite(risk_adjusted_return)
            and np.isfinite(random_label_risk_adjusted_return)
            else _nan()
        ),
    )
    return Phase1LayerComputation(
        layer_id=3,
        layer_name="oracle_profitability",
        metrics=metrics,
        extra_payload={
            "per_code_profitability": per_code,
            "decoded_returns": decoded_execution.returns,
            "dp_returns": dp_execution.returns,
            "flat_returns": flat_returns,
            "random_label_returns": random_returns,
            "random_seed": runtime_config.random_seed,
        },
    )


__all__ = [
    "compute_oracle_profitability_metrics",
]
