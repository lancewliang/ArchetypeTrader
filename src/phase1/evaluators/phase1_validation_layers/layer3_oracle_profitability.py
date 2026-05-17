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

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from src.utils import ActionExecutionCalculator, ActionExecutionResult, nan_value as _nan

from ...metrics import (
    Phase1EvaluationSnapshot,
    Phase1LayerComputation,
    Phase1OracleProfitabilityMetrics,
    Phase1OracleProfitabilityPayload,
    Phase1OracleProfitabilityThresholds,
    Phase1PairProfitabilityCell,
    Phase1PerCodeProfitability,
    Phase1ValidationRuntimeConfig,
)
from .layer2_behavior_quality import classify_action_motif, classify_market_morphology


_EPS = 1e-12


def _nan_array_like(reference: np.ndarray, *, dtype: np.dtype = np.float64) -> np.ndarray:
    """返回与 reference 第一维一致的 NaN 数组。"""

    return np.full(np.asarray(reference).shape[0], _nan(), dtype=dtype)


def _has_valid_prices(prices: np.ndarray | None) -> bool:
    """判断 Layer 3 是否具备可执行收益计算的价格输入。"""

    values = ActionExecutionCalculator._prices_2d(prices)
    return values is not None and np.all(np.isfinite(values))


def _safe_retention_ratio(numerator: np.ndarray, denominator: np.ndarray) -> float:
    """计算 retention ratio，并在 DP teacher 无正优势时返回 NaN。"""

    numerator_values = np.asarray(numerator, dtype=np.float64)
    denominator_values = np.asarray(denominator, dtype=np.float64)
    if not np.any(np.isfinite(numerator_values)) or not np.any(
        np.isfinite(denominator_values)
    ):
        return _nan()
    numerator_sum = float(np.nansum(numerator_values))
    denominator_sum = float(np.nansum(denominator_values))
    if not np.isfinite(denominator_sum) or denominator_sum <= _EPS:
        return _nan()
    return float(numerator_sum / denominator_sum)


def _safe_positive_denominator_ratio(numerator: float, denominator: float) -> float:
    """计算正分母比例；分母缺失、非正或接近 0 时返回 NaN。"""

    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= _EPS:
        return _nan()
    return float(numerator / denominator)


def _missing_prices_computation(
    val_snapshot: Phase1EvaluationSnapshot,
    *,
    random_seed: int,
) -> Phase1LayerComputation:
    """构造缺失价格时的 skip-as-fail Layer 3 结果。"""

    missing_returns = _nan_array_like(val_snapshot.code_ids)
    metrics = Phase1OracleProfitabilityMetrics(
        mean_decoded_advantage_vs_flat=_nan(),
        decoded_win_rate_vs_flat=_nan(),
        mean_advantage_vs_random_label=_nan(),
        random_label_relative_lift=_nan(),
        retention_ratio=_nan(),
        downside_control=_nan(),
        risk_adjusted_return=_nan(),
        top_5_contribution=_nan(),
        trimmed_decoded_advantage=_nan(),
        fee_drag=_nan(),
        turnover_return_correlation=_nan(),
        bad_code_ratio=_nan(),
        dominant_pair_positive_ratio=_nan(),
        random_label_risk_adjusted_return=_nan(),
        risk_adjusted_return_vs_random=_nan(),
    )
    return Phase1LayerComputation(
        layer_id=3,
        layer_name="oracle_profitability",
        metrics=metrics,
        extra_payload=Phase1OracleProfitabilityPayload(
            per_code_profitability=tuple(),
            decoded_returns=missing_returns,
            dp_returns=missing_returns.copy(),
            flat_returns=missing_returns.copy(),
            random_label_returns=missing_returns.copy(),
            random_seed=random_seed,
        ),
    )


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
        snapshot.depthprices,
    )
    if rewards.ndim == 2 and np.all(np.isfinite(rewards)):
        return ActionExecutionResult(
            returns=np.sum(rewards, axis=1),
            gross_returns=execution.gross_returns,
            fees=execution.fees,
            turnover=execution.turnover,
        )
    return execution


def decode_labels(
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


def decode_random_labels(
    *,
    model: Any,
    states: np.ndarray,
    num_archetypes: int,
    trials: int,
    seed: int,
    device: torch.device | str,
) -> np.ndarray:
    """用随机 code label 调用 decoder 生成 baseline actions。

    输入参数:
        model: ``ArchetypeVQModel`` 或兼容对象。
        states: 状态序列数组，形状为 ``[N, H, state_dim]``。
        num_archetypes: random label 采样空间大小。
        trials: 随机采样 trial 数，最小按 1 处理。
        seed: deterministic random seed。
        device: decoder 推理设备。

    输出:
        ``[trials, N, H]`` 形状的 random-label decoded action 数组。

    使用场景:
        random label baseline；调用方必须继续使用
        ``ActionExecutionCalculator.execute_actions()`` 计算收益。
    """

    rng = np.random.default_rng(seed)
    sample_count = np.asarray(states).shape[0]
    trial_actions: list[np.ndarray] = []
    for _ in range(max(1, int(trials))):
        random_labels = rng.integers(
            low=0,
            high=max(1, int(num_archetypes)),
            size=sample_count,
            dtype=np.int64,
        )
        trial_actions.append(
            decode_labels(
                model=model,
                states=states,
                code_ids=random_labels,
                device=device,
            )
        )
    return np.stack(trial_actions, axis=0)


def compute_random_label_returns(
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

    if not _has_valid_prices(snapshot.prices):
        return _nan_array_like(snapshot.code_ids)

    num_codes = int(getattr(model, "num_archetypes", 0) or (np.max(snapshot.code_ids) + 1))
    random_action_trials = decode_random_labels(
        model=model,
        states=snapshot.states,
        num_archetypes=num_codes,
        trials=runtime_config.random_label_trials,
        seed=runtime_config.random_seed,
        device=device,
    )
    trial_returns: list[np.ndarray] = []
    for random_actions in random_action_trials:
        trial_returns.append(
            ActionExecutionCalculator.execute_actions(
                snapshot.prices,
                random_actions,
                runtime_config.fee_rate,
                snapshot.depthprices,
            ).returns
        )
    return np.mean(np.stack(trial_returns, axis=0), axis=0)


def compute_risk_adjusted_return(returns: np.ndarray) -> float:
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


def compute_max_drawdown(cumulative_returns: np.ndarray) -> float:
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


def compute_top_contribution_ratio(returns: np.ndarray, top_ratio: float) -> float:
    """计算正收益中头部样本贡献比例。

    输入参数:
        returns: 每条 horizon 的收益或优势数组。
        top_ratio: 头部样本比例，例如 0.05。

    输出:
        头部正收益占总正收益的比例；无正收益时返回 NaN。

    使用场景:
        诊断 decoded profit 是否过度依赖少数尾部 horizon。
    """

    finite = returns[np.isfinite(returns)]
    positive = finite[finite > 0]
    if positive.size == 0:
        return _nan()
    count = max(1, int(np.ceil(finite.size * max(0.0, min(1.0, top_ratio)))))
    top = np.sort(finite)[-count:]
    return float(np.sum(top) / (np.sum(positive) + _EPS))


def compute_active_codes(
    code_ids: np.ndarray,
    *,
    active_code_min_occupancy: float,
) -> tuple[int, ...]:
    """按 occupancy 过滤 active code。"""

    if code_ids.size == 0:
        return ()
    counts = np.bincount(code_ids.astype(np.int64))
    occupancy = counts / max(1, code_ids.size)
    return tuple(
        int(code_id)
        for code_id, ratio in enumerate(occupancy)
        if ratio >= active_code_min_occupancy
    )


def compute_trimmed_mean(values: np.ndarray, trim_ratio: float) -> float:
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


def compute_safe_corr(left: np.ndarray, right: np.ndarray) -> float:
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


def compute_fee_drag(fees: np.ndarray, gross_returns: np.ndarray) -> float:
    """按设计口径计算 fee drag: total_fee / gross_profit。"""

    fee_values = fees[np.isfinite(fees)]
    gross_profit = gross_returns[np.isfinite(gross_returns) & (gross_returns > 0.0)]
    if fee_values.size == 0:
        return _nan()
    denominator = float(np.sum(gross_profit))
    if denominator <= 0.0:
        return float("inf")
    return float(np.sum(fee_values) / (denominator + _EPS))


def compute_per_code_profitability(
    *,
    code_ids: np.ndarray,
    decoded_advantage: np.ndarray,
    decoded_returns: np.ndarray,
    dp_advantage: np.ndarray,
    decoded_gross_returns: np.ndarray,
    decoded_fees: np.ndarray,
    thresholds: Phase1OracleProfitabilityThresholds,
    active_codes: Sequence[int] | None = None,
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
    code_iterable = active_codes if active_codes is not None else np.unique(code_ids)
    for code_id in code_iterable:
        mask = code_ids == code_id
        if not np.any(mask):
            continue
        mean_advantage = float(np.nanmean(decoded_advantage[mask]))
        win_rate = float(np.nanmean(decoded_returns[mask] > 0.0))
        retention_ratio = _safe_retention_ratio(
            decoded_advantage[mask],
            dp_advantage[mask],
        )
        fee_drag = compute_fee_drag(decoded_fees[mask], decoded_gross_returns[mask])
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


def compute_pair_profitability_matrix(
    *,
    morphologies: np.ndarray,
    motifs: np.ndarray,
    decoded_advantage: np.ndarray,
    decoded_returns: np.ndarray,
    flat_returns: np.ndarray,
    dp_advantage: np.ndarray,
    decoded_gross_returns: np.ndarray,
    decoded_fees: np.ndarray,
) -> tuple[Phase1PairProfitabilityCell, ...]:
    """计算 morphology x motif 的 decoded advantage 矩阵 cell。

    输入参数:
        morphologies: 每条 horizon 的市场形态标签。
        motifs: 每条 horizon 的行为 motif 标签。
        decoded_advantage: decoded return 相对 flat 的优势。
        decoded_returns: decoded 策略净收益。
        flat_returns: flat baseline 收益。
        dp_advantage: DP teacher 相对 flat 的优势。
        decoded_gross_returns: decoded gross return。
        decoded_fees: decoded 手续费。

    输出:
        按 morphology、motif 排序的 pair profitability cell。
    """

    morphology_values = np.asarray(morphologies, dtype=object).reshape(-1)
    motif_values = np.asarray(motifs, dtype=object).reshape(-1)
    sample_count = morphology_values.shape[0]
    if sample_count == 0 or motif_values.shape[0] != sample_count:
        return ()

    decoded_advantage_values = np.asarray(decoded_advantage, dtype=np.float64)
    decoded_return_values = np.asarray(decoded_returns, dtype=np.float64)
    flat_return_values = np.asarray(flat_returns, dtype=np.float64)
    dp_advantage_values = np.asarray(dp_advantage, dtype=np.float64)
    gross_values = np.asarray(decoded_gross_returns, dtype=np.float64)
    fee_values = np.asarray(decoded_fees, dtype=np.float64)
    if any(
        values.shape[0] != sample_count
        for values in (
            decoded_advantage_values,
            decoded_return_values,
            flat_return_values,
            dp_advantage_values,
            gross_values,
            fee_values,
        )
    ):
        return ()

    output: list[Phase1PairProfitabilityCell] = []
    pairs = sorted(
        {
            (str(morphology), str(motif))
            for morphology, motif in zip(
                morphology_values,
                motif_values,
                strict=False,
            )
        }
    )
    for morphology, motif in pairs:
        mask = (morphology_values == morphology) & (motif_values == motif)
        support = int(np.sum(mask))
        if support <= 0:
            continue
        output.append(
            Phase1PairProfitabilityCell(
                morphology=morphology,
                motif=motif,
                support=support,
                mean_decoded_advantage=float(
                    np.nanmean(decoded_advantage_values[mask])
                ),
                decoded_win_rate=float(
                    np.nanmean(decoded_return_values[mask] > flat_return_values[mask])
                ),
                retention_ratio=_safe_retention_ratio(
                    decoded_advantage_values[mask],
                    dp_advantage_values[mask],
                ),
                fee_drag=compute_fee_drag(fee_values[mask], gross_values[mask]),
            )
        )
    return tuple(output)


def compute_dominant_pair_positive_ratio(
    snapshot: Phase1EvaluationSnapshot,
    decoded_advantage: np.ndarray,
    *,
    active_code_min_occupancy: float = 0.0,
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
    active_codes = compute_active_codes(
        code_ids,
        active_code_min_occupancy=active_code_min_occupancy,
    )
    positive = 0
    total = 0
    for code_id in active_codes:
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
    if not _has_valid_prices(val_snapshot.prices):
        return _missing_prices_computation(
            val_snapshot,
            random_seed=runtime_config.random_seed,
        )

    dp_execution = _demo_returns(val_snapshot, runtime_config)
    decoded_execution = ActionExecutionCalculator.execute_actions(
        val_snapshot.prices,
        val_snapshot.decoded_actions,
        runtime_config.fee_rate,
        val_snapshot.depthprices,
    )
    flat_returns = np.zeros_like(decoded_execution.returns)
    random_returns = compute_random_label_returns(
        model=model,
        snapshot=val_snapshot,
        runtime_config=runtime_config,
        device=device,
    )

    decoded_advantage = decoded_execution.returns - flat_returns
    dp_advantage = dp_execution.returns - flat_returns
    random_advantage = random_returns - flat_returns
    retention_ratio = _safe_retention_ratio(decoded_advantage, dp_advantage)
    active_codes = compute_active_codes(
        np.asarray(val_snapshot.code_ids, dtype=np.int64),
        active_code_min_occupancy=runtime_config.active_code_min_occupancy,
    )
    per_code = compute_per_code_profitability(
        code_ids=np.asarray(val_snapshot.code_ids, dtype=np.int64),
        decoded_advantage=decoded_advantage,
        decoded_returns=decoded_execution.returns,
        dp_advantage=dp_advantage,
        decoded_gross_returns=decoded_execution.gross_returns,
        decoded_fees=decoded_execution.fees,
        thresholds=thresholds,
        active_codes=active_codes,
    )
    morphologies = classify_market_morphology(val_snapshot.prices)
    motifs = classify_action_motif(val_snapshot.decoded_actions, val_snapshot.prices)
    pair_profitability_matrix = compute_pair_profitability_matrix(
        morphologies=morphologies,
        motifs=motifs,
        decoded_advantage=decoded_advantage,
        decoded_returns=decoded_execution.returns,
        flat_returns=flat_returns,
        dp_advantage=dp_advantage,
        decoded_gross_returns=decoded_execution.gross_returns,
        decoded_fees=decoded_execution.fees,
    )

    random_label_risk_adjusted_return = compute_risk_adjusted_return(random_returns)
    risk_adjusted_return = compute_risk_adjusted_return(decoded_execution.returns)
    decoded_drawdown = compute_max_drawdown(np.cumsum(decoded_execution.returns))
    dp_drawdown = compute_max_drawdown(np.cumsum(dp_execution.returns))

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
        retention_ratio=retention_ratio,
        downside_control=_safe_positive_denominator_ratio(
            decoded_drawdown,
            dp_drawdown,
        ),
        risk_adjusted_return=risk_adjusted_return,
        top_5_contribution=compute_top_contribution_ratio(
            decoded_advantage,
            runtime_config.top_contribution_ratio,
        ),
        trimmed_decoded_advantage=compute_trimmed_mean(
            decoded_advantage,
            runtime_config.top_contribution_ratio,
        ),
        fee_drag=compute_fee_drag(decoded_execution.fees, decoded_execution.gross_returns),
        turnover_return_correlation=compute_safe_corr(
            decoded_execution.turnover,
            decoded_execution.returns,
        ),
        bad_code_ratio=float(
            np.mean([item.mean_advantage < 0.0 for item in per_code])
        )
        if per_code
        else _nan(),
        dominant_pair_positive_ratio=compute_dominant_pair_positive_ratio(
            val_snapshot,
            decoded_advantage,
            active_code_min_occupancy=runtime_config.active_code_min_occupancy,
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
        extra_payload=Phase1OracleProfitabilityPayload(
            per_code_profitability=per_code,
            decoded_returns=decoded_execution.returns,
            dp_returns=dp_execution.returns,
            flat_returns=flat_returns,
            random_label_returns=random_returns,
            random_seed=runtime_config.random_seed,
            pair_profitability_matrix=pair_profitability_matrix,
        ),
    )

__all__ = [
    "compute_active_codes",
    "compute_dominant_pair_positive_ratio",
    "compute_fee_drag",
    "compute_max_drawdown",
    "compute_oracle_profitability_metrics",
    "compute_pair_profitability_matrix",
    "compute_per_code_profitability",
    "compute_random_label_returns",
    "compute_risk_adjusted_return",
    "compute_safe_corr",
    "compute_top_contribution_ratio",
    "compute_trimmed_mean",
    "decode_labels",
    "decode_random_labels",
]
