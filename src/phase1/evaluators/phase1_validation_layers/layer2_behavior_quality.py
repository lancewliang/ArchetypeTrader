"""Phase I validation Layer 2: archetype behavior quality raw metrics。

文件功能说明:
    本文件负责计算每个 active code 的行为结构质量 raw metrics，包括 support、
    morphology/motif/pair 纯度、morphology lift、code 内行为一致性、code 间
    分离度、latent silhouette、重复 code pair 数量和盈利 code 覆盖率。

设计边界:
    - 只计算 archetype 行为结构和 code-level diagnostics；
    - 不直接计算交易收益，盈利性信息通过 Layer 3 的 per-code profitability 输入；
    - 不做 hard gate pass/fail 判定；
    - 缺失 prices 时 morphology 相关指标按 weak/missing 处理，最终由 rules 层失败。

使用场景:
    ``Phase1CodebookEvaluator`` 通常先计算 Layer 3，再把
    ``per_code_profitability`` 传入 ``compute_behavior_quality_metrics()``，
    生成 ``Phase1BehaviorQualityMetrics`` 和 code diagnostics。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

import numpy as np

from src.utils import nan_value as _nan

from ...metrics import (
    Phase1BehaviorQualityComputation,
    Phase1BehaviorQualityMetrics,
    Phase1BehaviorQualityPayload,
    Phase1BehaviorQualityThresholds,
    Phase1CodeDiagnostic,
    Phase1EvaluationSnapshot,
    Phase1PerCodeProfitability,
    Phase1ValidationRuntimeConfig,
)


_EPS = 1e-12


def _positions(actions: np.ndarray) -> np.ndarray:
    """将动作 id 映射为持仓值。

    输入参数:
        actions: 动作数组，约定 ``0=short``、``1=flat``、``2=long``。

    输出:
        同形状持仓数组，取值为 ``-1/0/1``。

    使用场景:
        action motif、prototype similarity、intra similarity 和 separation 计算。
    """

    return np.asarray(actions, dtype=np.float64) - 1.0


def _prices_2d(prices: np.ndarray | None) -> np.ndarray | None:
    """把价格数组标准化为二维 ``[sample, horizon]``。

    输入参数:
        prices: 原始价格数组，可为 ``[N, H]``、``[N, H, 1]`` 或 ``None``。

    输出:
        二维价格数组；缺失、维度不合法或 horizon 不足时返回 ``None``。

    使用场景:
        morphology 和部分 motif 计算前统一价格形状。
    """

    if prices is None:
        return None
    values = np.asarray(prices, dtype=np.float64)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim != 2 or values.shape[1] < 2:
        return None
    return values


def classify_market_morphology(
    prices: np.ndarray | None,
    *,
    fee_rate: float = 0.0002,
) -> np.ndarray:
    """根据价格路径分类市场形态。

    输入参数:
        prices: 价格数组，可为 ``[N, H]``、``[N, H, 1]`` 或 ``None``。
        fee_rate: 手续费率，用于设定 neutral band。

    输出:
        ``[N]`` 形状的 morphology 标签数组；价格不可用时返回空数组。标签遵循
        validation criteria 文档定义:
        ``uptrend``、``downtrend``、``reversal-up``、``reversal-down``、
        ``range-high-vol``、``range-low-vol``、``volatile-mixed``、``neutral``。

    使用场景:
        统计每个 code 的 dominant morphology、morphology purity 和 lift。
    """

    price_values = _prices_2d(prices)
    if price_values is None:
        return np.asarray([], dtype=object)
    horizon = price_values.shape[1]
    mid = max(1, horizon // 2)
    log_returns = np.diff(np.log(np.maximum(price_values, _EPS)), axis=1)
    total_return = price_values[:, -1] / np.maximum(price_values[:, 0], _EPS) - 1.0
    ret_first = price_values[:, mid] / np.maximum(price_values[:, 0], _EPS) - 1.0
    ret_second = price_values[:, -1] / np.maximum(price_values[:, mid], _EPS) - 1.0
    realized_vol = np.std(log_returns, axis=1) * np.sqrt(horizon)
    range_ratio = (
        np.max(price_values, axis=1) - np.min(price_values, axis=1)
    ) / np.maximum(price_values[:, 0], _EPS)
    path_length = np.sum(np.abs(np.diff(price_values, axis=1)), axis=1)
    trend_efficiency = np.abs(price_values[:, -1] - price_values[:, 0]) / (
        path_length + _EPS
    )

    vol_high = float(np.quantile(realized_vol, 0.70))
    vol_low = float(np.quantile(realized_vol, 0.30))
    range_high = float(np.quantile(range_ratio, 0.70))
    trend_ret_threshold = max(
        float(np.quantile(np.abs(total_return), 0.60)),
        3.0 * fee_rate,
    )
    reversal_leg_threshold = max(
        float(np.quantile(np.abs(np.concatenate([ret_first, ret_second])), 0.60)),
        2.0 * fee_rate,
    )

    labels = np.full(price_values.shape[0], "neutral", dtype=object)

    reversal_up = (
        (ret_first < -reversal_leg_threshold)
        & (ret_second > reversal_leg_threshold)
    )
    reversal_down = (
        (ret_first > reversal_leg_threshold)
        & (ret_second < -reversal_leg_threshold)
    )
    labels[reversal_up] = "reversal-up"
    labels[reversal_down] = "reversal-down"

    unassigned = labels == "neutral"
    labels[
        unassigned
        & (total_return > trend_ret_threshold)
        & (trend_efficiency >= 0.35)
    ] = "uptrend"
    labels[
        unassigned
        & (total_return < -trend_ret_threshold)
        & (trend_efficiency >= 0.35)
    ] = "downtrend"

    unassigned = labels == "neutral"
    labels[
        unassigned
        & (np.abs(total_return) <= trend_ret_threshold)
        & (range_ratio >= range_high)
    ] = "range-high-vol"
    labels[
        unassigned
        & (np.abs(total_return) <= trend_ret_threshold)
        & (realized_vol <= vol_low)
    ] = "range-low-vol"

    unassigned = labels == "neutral"
    labels[unassigned & (realized_vol >= vol_high)] = "volatile-mixed"
    return labels


def classify_action_motif(
    actions: np.ndarray,
    prices: np.ndarray | None = None,
) -> np.ndarray:
    """根据 decoded action sequence 分类交易 motif。

    输入参数:
        actions: decoded action 数组，形状为 ``[N, H]``。
        prices: 可选价格路径，用于判断入场方向和入场前 recent move 的关系。

    输出:
        ``[N]`` 形状的 motif 标签数组，格式为
        ``{direction} + {entry_bucket} + {holding_style}``，必要时追加
        ``+ {reversal_type}``。

    使用场景:
        统计每个 code 的 dominant motif 和 morphology-motif pair。
    """

    positions = _positions(actions)
    price_values = _prices_2d(prices)
    motifs: list[str] = []
    horizon = max(1, positions.shape[1])

    for sample_index, position_path in enumerate(positions):
        non_flat = position_path != 0
        non_flat_ratio = float(np.mean(non_flat))
        long_ratio = float(np.mean(position_path > 0))
        short_ratio = float(np.mean(position_path < 0))

        if non_flat_ratio < 0.20:
            motifs.append("flat + none + mostly-flat")
            continue

        if long_ratio >= 0.35 and long_ratio >= short_ratio + 0.15:
            direction = "long"
        elif short_ratio >= 0.35 and short_ratio >= long_ratio + 0.15:
            direction = "short"
        else:
            direction = "mixed"

        first_trade_t = int(np.argmax(non_flat))
        entry_position = float(position_path[first_trade_t])
        entry_ratio = first_trade_t / horizon
        if entry_ratio < 1.0 / 3.0:
            entry_bucket = "early"
        elif entry_ratio < 2.0 / 3.0:
            entry_bucket = "middle"
        else:
            entry_bucket = "late"

        position_with_initial = np.concatenate(
            [np.zeros(1, dtype=np.float64), position_path]
        )
        change_count = int(np.sum(np.diff(position_with_initial) != 0))
        after_entry = position_path[first_trade_t:]
        holding_ratio_after_entry = float(
            np.mean(after_entry == entry_position)
        ) if after_entry.size else 0.0

        if entry_bucket in {"middle", "late"} and holding_ratio_after_entry >= 0.70:
            holding_style = "delayed-hold"
        elif holding_ratio_after_entry >= 0.70 and change_count <= 2:
            holding_style = "hold"
        elif 0.20 <= non_flat_ratio < 0.50 and _has_contiguous_non_flat(non_flat):
            holding_style = "brief-trade"
        elif change_count > 2:
            holding_style = "switching"
        else:
            holding_style = "hold"

        reversal_type = _classify_reversal_type(
            position_path=position_path,
            price_path=(
                price_values[sample_index]
                if price_values is not None and sample_index < price_values.shape[0]
                else None
            ),
            first_trade_t=first_trade_t,
            long_ratio=long_ratio,
            short_ratio=short_ratio,
        )
        motif = f"{direction} + {entry_bucket} + {holding_style}"
        if reversal_type != "none":
            motif = f"{motif} + {reversal_type}"
        motifs.append(motif)
    return np.asarray(motifs, dtype=object)


def _has_contiguous_non_flat(non_flat: np.ndarray) -> bool:
    """判断非空仓片段是否主要为连续单段。"""

    indices = np.flatnonzero(non_flat)
    if indices.size == 0:
        return False
    return int(indices[-1] - indices[0] + 1) == int(indices.size)


def _classify_reversal_type(
    *,
    position_path: np.ndarray,
    price_path: np.ndarray | None,
    first_trade_t: int,
    long_ratio: float,
    short_ratio: float,
) -> str:
    """分类 motif 的反转/近期价格关系附加标签。"""

    non_zero = position_path[position_path != 0]
    if non_zero.size:
        first_direction = float(non_zero[0])
        last_direction = float(non_zero[-1])
        if (
            first_direction > 0
            and last_direction < 0
            and long_ratio >= 0.20
            and short_ratio >= 0.20
        ):
            return "long-to-short"
        if (
            first_direction < 0
            and last_direction > 0
            and long_ratio >= 0.20
            and short_ratio >= 0.20
        ):
            return "short-to-long"

    if price_path is None or first_trade_t <= 0:
        return "none"
    lookback_start = max(0, first_trade_t - 12)
    recent_move = price_path[first_trade_t] - price_path[lookback_start]
    entry_direction = position_path[first_trade_t]
    if abs(recent_move) <= _EPS or entry_direction == 0:
        return "none"
    if np.sign(recent_move) == np.sign(entry_direction):
        return "with-recent-move"
    return "against-recent-move"


def _active_codes(
    code_ids: np.ndarray,
    runtime_config: Phase1ValidationRuntimeConfig,
) -> tuple[int, ...]:
    """根据 occupancy 找出 active code。

    输入参数:
        code_ids: 每个样本的 assigned code id。
        runtime_config: 提供 ``active_code_min_occupancy``。

    输出:
        active code id tuple。

    使用场景:
        只对 active code 计算 support、purity、diagnostics 和结构分离度。
    """

    if code_ids.size == 0:
        return ()
    counts = np.bincount(code_ids.astype(np.int64))
    occupancy = counts / max(1, code_ids.size)
    return tuple(
        int(code_id)
        for code_id, ratio in enumerate(occupancy)
        if ratio >= runtime_config.active_code_min_occupancy
    )


def _num_codes(snapshot: Phase1EvaluationSnapshot, code_ids: np.ndarray) -> int:
    """推断当前 codebook size K。"""

    distances = np.asarray(snapshot.distances)
    if distances.ndim == 2 and distances.shape[1] > 0:
        return int(distances.shape[1])
    return int(np.max(code_ids) + 1) if code_ids.size else 0


def _dominant(values: np.ndarray) -> tuple[str | None, float | None]:
    """计算一组离散标签的 dominant value 和占比。

    输入参数:
        values: 字符串或可转字符串的离散标签数组。

    输出:
        ``(dominant_label, ratio)``；输入为空时返回 ``(None, None)``。

    使用场景:
        per-code dominant morphology、motif 和 pair 统计。
    """

    if values.size == 0:
        return None, None
    counts = Counter(str(value) for value in values)
    label, count = counts.most_common(1)[0]
    return label, count / values.size


def _entropy_purity(values: np.ndarray) -> float | None:
    """计算 1 - normalized entropy purity。

    单一类别视为完全纯净；空输入返回 None。
    """

    if values.size == 0:
        return None
    counts = np.asarray(list(Counter(str(value) for value in values).values()))
    if counts.size <= 1:
        return 1.0
    probabilities = counts / np.sum(counts)
    entropy = -float(np.sum(probabilities * np.log(probabilities + _EPS)))
    return float(1.0 - entropy / (np.log(counts.size) + _EPS))


def _global_distribution(values: np.ndarray) -> dict[str, float]:
    """计算全体验证集离散标签分布。

    输入参数:
        values: 离散标签数组。

    输出:
        ``label -> probability`` 字典。

    使用场景:
        计算 per-code dominant morphology 相对全局分布的 lift。
    """

    counts = Counter(str(value) for value in values)
    total = sum(counts.values())
    return {label: count / max(1, total) for label, count in counts.items()}


def compute_distribution_by_code(
    values: np.ndarray,
    code_ids: np.ndarray,
) -> dict[int, dict[str, float]]:
    """计算每个 code 内离散标签的经验分布。

    输入参数:
        values: ``[N]`` 形状的离散标签数组，例如 morphology、motif 或 pair。
        code_ids: ``[N]`` 形状的 assigned code id。

    输出:
        ``code_id -> label -> probability`` 字典；输入为空时返回空字典。

    使用场景:
        统计 ``P(morphology | code)``、``P(motif | code)`` 和
        ``P(morphology, motif | code)``，再由分布派生 dominant label、purity
        和 lift。
    """

    label_values = np.asarray(values, dtype=object).reshape(-1)
    codes = np.asarray(code_ids, dtype=np.int64).reshape(-1)
    if label_values.shape[0] != codes.shape[0]:
        raise ValueError("values and code_ids must have the same length")
    distributions: dict[int, dict[str, float]] = {}
    for code_id in np.unique(codes):
        labels = label_values[codes == code_id]
        counts = Counter(str(value) for value in labels)
        total = sum(counts.values())
        distributions[int(code_id)] = {
            label: count / max(1, total) for label, count in counts.items()
        }
    return distributions


def compute_lift(
    code_distribution: Mapping[str, float],
    global_distribution: Mapping[str, float],
) -> dict[str, float]:
    """计算 code 内标签分布相对全局分布的 lift。

    输入参数:
        code_distribution: 单个 code 内的 ``label -> probability`` 分布。
        global_distribution: 全体验证集上的 ``label -> probability`` 分布。

    输出:
        ``label -> lift`` 字典，分母带 ``eps`` 保护。

    使用场景:
        判断某个 code 是否真正富集某类市场形态，而不是只反映全局基准分布。
    """

    return {
        str(label): float(probability)
        / (float(global_distribution.get(label, 0.0)) + _EPS)
        for label, probability in code_distribution.items()
    }


def _dominant_from_distribution(
    distribution: Mapping[str, float],
) -> tuple[str | None, float | None]:
    """从 label 分布中取 dominant label 和概率。"""

    if not distribution:
        return None, None
    label, probability = max(distribution.items(), key=lambda item: item[1])
    return str(label), float(probability)


def _prototype_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """计算两个 action prototype 的简化相似度。

    输入参数:
        left: 左侧 code 的 action prototype。
        right: 右侧 code 的 action prototype。

    输出:
        ``[0, 1]`` 附近的相似度值，越高表示两个原型越接近。

    使用场景:
        duplicate code pair 检测。
    """

    return float(1.0 - np.mean(np.abs(left - right)) / 2.0)


def _action_prototypes(
    actions: np.ndarray,
    code_ids: np.ndarray,
    active_codes: Sequence[int],
) -> dict[int, np.ndarray]:
    """计算每个 active code 的 decoded action prototype。

    输入参数:
        actions: decoded action 数组，形状为 ``[N, H]``。
        code_ids: 每个样本的 assigned code id。
        active_codes: 需要统计的 active code 列表。

    输出:
        ``code_id -> prototype`` 字典，prototype 为该 code 内持仓序列均值。

    使用场景:
        intra-code similarity、inter/intra separation 和 duplicate pair 计算。
    """

    positions = _positions(actions)
    return {
        int(code_id): np.mean(positions[code_ids == code_id], axis=0)
        for code_id in active_codes
        if np.any(code_ids == code_id)
    }


def _all_present_codes(code_ids: np.ndarray) -> tuple[int, ...]:
    """返回 code_ids 中出现过的 code，供公开 helper 未显式传 active_codes 时使用。"""

    codes = np.asarray(code_ids, dtype=np.int64).reshape(-1)
    return tuple(int(code_id) for code_id in np.unique(codes)) if codes.size else ()


def compute_intra_code_action_similarity(
    actions: np.ndarray,
    code_ids: np.ndarray,
    active_codes: Sequence[int] | None = None,
) -> float:
    """计算同一 code 内 decoded action 的平均相似度。

    输入参数:
        actions: decoded action 数组。
        code_ids: 每个样本的 assigned code id。
        active_codes: active code 列表；不传时使用 ``code_ids`` 中所有出现过的 code。

    输出:
        active code 内样本到本 code prototype 的平均相似度；无有效样本时返回 NaN。

    使用场景:
        衡量每个 archetype 内部行为是否一致。
    """

    positions = _positions(actions)
    codes = np.asarray(code_ids, dtype=np.int64).reshape(-1)
    if positions.shape[0] != codes.shape[0]:
        raise ValueError("actions and code_ids must have the same number of samples")
    selected_codes = (
        tuple(active_codes) if active_codes is not None else _all_present_codes(codes)
    )
    values: list[float] = []
    for code_id in selected_codes:
        members = positions[codes == code_id]
        if members.shape[0] == 0:
            continue
        prototype = np.mean(members, axis=0)
        similarities = 1.0 - np.mean(np.abs(members - prototype), axis=1) / 2.0
        values.extend(similarities.tolist())
    return float(np.mean(values)) if values else _nan()


def _intra_code_similarity(
    actions: np.ndarray,
    code_ids: np.ndarray,
    active_codes: Sequence[int],
) -> float:
    """兼容旧私有 helper 名称。"""

    return compute_intra_code_action_similarity(actions, code_ids, active_codes)


def compute_inter_intra_separation(
    actions: np.ndarray,
    code_ids: np.ndarray,
    active_codes: Sequence[int] | None = None,
) -> float:
    """计算 code 间距离与 code 内距离的分离度。

    输入参数:
        actions: decoded action 数组。
        code_ids: 每个样本的 assigned code id。
        active_codes: active code 列表；不传时使用 ``code_ids`` 中所有出现过的 code。

    输出:
        ``mean_inter_distance / mean_intra_distance``；active code 少于 2 时返回 NaN。

    使用场景:
        衡量不同 archetype 是否足够可区分。
    """

    codes = np.asarray(code_ids, dtype=np.int64).reshape(-1)
    if np.asarray(actions).shape[0] != codes.shape[0]:
        raise ValueError("actions and code_ids must have the same number of samples")
    selected_codes = (
        tuple(active_codes) if active_codes is not None else _all_present_codes(codes)
    )
    prototypes = _action_prototypes(actions, codes, selected_codes)
    if len(prototypes) < 2:
        return _nan()
    positions = _positions(actions)
    intra_values: list[float] = []
    for code_id, prototype in prototypes.items():
        members = positions[codes == code_id]
        intra_values.extend(np.linalg.norm(members - prototype, axis=1).tolist())
    inter_values: list[float] = []
    proto_items = list(prototypes.items())
    for index, (_, left) in enumerate(proto_items):
        for _, right in proto_items[index + 1 :]:
            inter_values.append(float(np.linalg.norm(left - right)))
    return float(np.mean(inter_values) / (np.mean(intra_values) + _EPS))


def _inter_intra_separation(
    actions: np.ndarray,
    code_ids: np.ndarray,
    active_codes: Sequence[int],
) -> float:
    """兼容旧私有 helper 名称。"""

    return compute_inter_intra_separation(actions, code_ids, active_codes)


def compute_duplicate_code_pair_count(
    actions: np.ndarray,
    code_ids: np.ndarray,
    active_codes: Sequence[int] | None = None,
    *,
    threshold: float,
) -> int:
    """统计相似度超过阈值的重复 code pair 数量。

    输入参数:
        actions: decoded action 数组。
        code_ids: 每个样本的 assigned code id。
        active_codes: active code 列表；不传时使用 ``code_ids`` 中所有出现过的 code。
        threshold: 判定重复 code 的 prototype similarity 阈值。

    输出:
        重复 code pair 数量。

    使用场景:
        检查 codebook 是否把多个 code 学成几乎相同的行为原型。
    """

    codes = np.asarray(code_ids, dtype=np.int64).reshape(-1)
    if np.asarray(actions).shape[0] != codes.shape[0]:
        raise ValueError("actions and code_ids must have the same number of samples")
    selected_codes = (
        tuple(active_codes) if active_codes is not None else _all_present_codes(codes)
    )
    prototypes = list(_action_prototypes(actions, codes, selected_codes).values())
    duplicates = 0
    for index, left in enumerate(prototypes):
        for right in prototypes[index + 1 :]:
            duplicates += int(_prototype_similarity(left, right) > threshold)
    return duplicates


def _duplicate_pair_count(
    actions: np.ndarray,
    code_ids: np.ndarray,
    active_codes: Sequence[int],
    *,
    threshold: float,
) -> int:
    """兼容旧私有 helper 名称。"""

    return compute_duplicate_code_pair_count(
        actions,
        code_ids,
        active_codes,
        threshold=threshold,
    )


def _approx_silhouette(
    z_e: np.ndarray,
    code_ids: np.ndarray,
    active_codes: Sequence[int],
) -> float:
    """计算基于 code centroid 的近似 latent silhouette score。

    输入参数:
        z_e: encoder latent 数组，形状为 ``[N, latent_dim]``。
        code_ids: 每个样本的 assigned code id。
        active_codes: active code 列表。

    输出:
        近似 silhouette score；active code 少于 2 或输入维度不合法时返回 NaN。

    使用场景:
        诊断 latent 空间中 assigned code 是否有聚类分离度。
    """

    if len(active_codes) < 2:
        return _nan()
    values = np.asarray(z_e, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != code_ids.shape[0]:
        return _nan()
    centroids = {
        int(code_id): np.mean(values[code_ids == code_id], axis=0)
        for code_id in active_codes
        if np.any(code_ids == code_id)
    }
    scores: list[float] = []
    for value, code_id in zip(values, code_ids, strict=False):
        if int(code_id) not in centroids:
            continue
        own = np.linalg.norm(value - centroids[int(code_id)])
        other = min(
            np.linalg.norm(value - centroid)
            for other_code, centroid in centroids.items()
            if other_code != int(code_id)
        )
        scores.append((other - own) / max(other, own, _EPS))
    return float(np.mean(scores)) if scores else _nan()


def _per_code_profitability_map(
    per_code_profitability: Sequence[Phase1PerCodeProfitability] | None,
) -> Mapping[int, Phase1PerCodeProfitability]:
    """把 Layer 3 per-code profitability 转成按 code id 查询的映射。

    输入参数:
        per_code_profitability: Layer 3 输出的 per-code 盈利性列表；可为 ``None``。

    输出:
        ``code_id -> Phase1PerCodeProfitability`` 映射；输入为空时返回空 dict。

    使用场景:
        layer2 统计 profitable code coverage 和 weak-lift-but-profitable 诊断。
    """

    if per_code_profitability is None:
        return {}
    return {item.code_id: item for item in per_code_profitability}


def _code_diagnostic_status(
    *,
    weak_support: bool,
    weak_morphology: bool,
    weak_motif: bool,
    weak_pair: bool,
    weak_lift_nonprofitable: bool,
) -> str:
    """把 per-code 支撑度、结构质量和盈利辅助证据聚合为 report 状态。"""

    if weak_support:
        return "bad"
    weak_structure_count = sum(
        (
            weak_morphology,
            weak_motif,
            weak_pair,
            weak_lift_nonprofitable,
        )
    )
    if weak_structure_count == 0:
        return "pass"
    if weak_structure_count >= 3:
        return "bad"
    return "weak"


def compute_behavior_quality_metrics(
    *,
    train_snapshot: Phase1EvaluationSnapshot,
    val_snapshot: Phase1EvaluationSnapshot,
    runtime_config: Phase1ValidationRuntimeConfig,
    per_code_profitability: Sequence[Phase1PerCodeProfitability] | None = None,
    thresholds: Phase1BehaviorQualityThresholds | None = None,
) -> Phase1BehaviorQualityComputation:
    """计算 Layer 2 archetype 行为质量 raw metrics 和 code diagnostics。

    功能说明:
        对 validation split 的 active code 逐个统计 support、dominant morphology、
        dominant motif、dominant pair、lift 和盈利性摘要，再聚合为 layer-level
        behavior quality metrics。

    输入参数:
        train_snapshot: 训练集 snapshot，当前不参与计算，保留用于统一接口。
        val_snapshot: 验证集 snapshot，读取 prices、decoded_actions、code_ids 和 z_e。
        runtime_config: validation 运行参数，提供 active code 占用阈值和手续费率。
        per_code_profitability: Layer 3 输出的 per-code 盈利性摘要；缺失时盈利覆盖率
            返回 NaN。
        thresholds: Layer 2 阈值配置；不传时使用默认 ``Phase1BehaviorQualityThresholds``。

    输出:
        ``Phase1BehaviorQualityComputation``，其中 ``metrics`` 为
        ``Phase1BehaviorQualityMetrics``，``code_diagnostics`` 为 per-code report
        诊断表，``behavior_quality_payload`` 包含 morphology、motif 和 active code 列表。

    使用场景:
        full checkpoint validation 的第二层行为结构 raw metric 计算；结果交给
        ``evaluate_behavior_quality_rules()`` 判定 hard gate。
    """

    del train_snapshot
    thresholds = thresholds or Phase1BehaviorQualityThresholds()
    code_ids = np.asarray(val_snapshot.code_ids, dtype=np.int64)
    num_codes = _num_codes(val_snapshot, code_ids)
    active_codes = _active_codes(code_ids, runtime_config)
    active_count = len(active_codes)

    morphologies = classify_market_morphology(
        val_snapshot.prices,
        fee_rate=runtime_config.fee_rate,
    )
    missing_morphology = morphologies.size != code_ids.size
    if missing_morphology:
        morphologies = np.full(code_ids.size, "missing", dtype=object)
    motifs = classify_action_motif(val_snapshot.decoded_actions, val_snapshot.prices)
    pairs = np.asarray(
        [
            f"{morphology}:{motif}"
            for morphology, motif in zip(morphologies, motifs, strict=False)
        ],
        dtype=object,
    )
    global_morphology = _global_distribution(morphologies)
    morphology_by_code = compute_distribution_by_code(morphologies, code_ids)
    motif_by_code = compute_distribution_by_code(motifs, code_ids)
    pair_by_code = compute_distribution_by_code(pairs, code_ids)
    profitability_map = _per_code_profitability_map(per_code_profitability)

    min_support = max(
        thresholds.min_code_support_abs,
        int(np.ceil(thresholds.min_code_support_ratio * max(1, code_ids.size))),
    )
    weak_support = 0
    weak_morphology = 0
    weak_motif = 0
    weak_pair = 0
    weak_lift_nonprofitable = 0
    profitable_count = 0
    diagnostics: list[Phase1CodeDiagnostic] = []

    for code_id in active_codes:
        mask = code_ids == code_id
        support = int(np.sum(mask))
        occupancy = support / max(1, code_ids.size)
        morphology_distribution = morphology_by_code.get(code_id, {})
        motif_distribution = motif_by_code.get(code_id, {})
        pair_distribution = pair_by_code.get(code_id, {})
        dominant_morphology, dominant_morphology_ratio = _dominant_from_distribution(
            morphology_distribution
        )
        morphology_purity = _entropy_purity(morphologies[mask])
        dominant_motif, dominant_motif_ratio = _dominant_from_distribution(
            motif_distribution
        )
        motif_purity = _entropy_purity(motifs[mask])
        dominant_pair, dominant_pair_ratio = _dominant_from_distribution(
            pair_distribution
        )
        morphology_lift_by_label = compute_lift(
            morphology_distribution,
            global_morphology,
        )
        morphology_lift = (
            morphology_lift_by_label.get(dominant_morphology)
            if dominant_morphology is not None
            else None
        )
        profitability = profitability_map.get(code_id)
        profitable = bool(profitability and profitability.passed)
        profitable_count += int(profitable)

        code_weak_support = support < min_support
        code_weak_morphology = missing_morphology or (
            (
                dominant_morphology_ratio is None
                or dominant_morphology_ratio
                < thresholds.dominant_morphology_ratio_min
            )
            and (
                morphology_purity is None
                or morphology_purity < thresholds.morphology_purity_min
            )
        )
        code_weak_motif = (
            (
                dominant_motif_ratio is None
                or dominant_motif_ratio < thresholds.dominant_motif_ratio_min
            )
            and (
                motif_purity is None
                or motif_purity < thresholds.motif_purity_min
            )
        )
        code_weak_pair = (
            missing_morphology
            or dominant_pair_ratio is None
            or dominant_pair_ratio < thresholds.dominant_pair_ratio_min
        )
        code_weak_lift_nonprofitable = (
            (
                missing_morphology
                or morphology_lift is None
                or morphology_lift < thresholds.morphology_lift_min
            )
            and not profitable
        )

        weak_support += int(code_weak_support)
        weak_morphology += int(code_weak_morphology)
        weak_motif += int(code_weak_motif)
        weak_pair += int(code_weak_pair)
        weak_lift_nonprofitable += int(code_weak_lift_nonprofitable)
        status = _code_diagnostic_status(
            weak_support=code_weak_support,
            weak_morphology=code_weak_morphology,
            weak_motif=code_weak_motif,
            weak_pair=code_weak_pair,
            weak_lift_nonprofitable=code_weak_lift_nonprofitable,
        )

        diagnostics.append(
            Phase1CodeDiagnostic(
                code_id=code_id,
                support=support,
                occupancy=occupancy,
                dominant_morphology=dominant_morphology,
                dominant_morphology_ratio=dominant_morphology_ratio,
                morphology_lift=morphology_lift,
                dominant_motif=dominant_motif,
                dominant_motif_ratio=dominant_motif_ratio,
                dominant_pair=dominant_pair,
                dominant_pair_ratio=dominant_pair_ratio,
                decoded_mean_advantage=(
                    profitability.mean_advantage if profitability else None
                ),
                decoded_win_rate=profitability.win_rate if profitability else None,
                retention_ratio=profitability.retention_ratio if profitability else None,
                fee_drag=profitability.fee_drag if profitability else None,
                status=status,
            )
        )

    denominator = active_count if active_count > 0 else 0
    metrics = Phase1BehaviorQualityMetrics(
        weak_support_code_ratio=weak_support / denominator if denominator else _nan(),
        weak_morphology_code_ratio=(
            weak_morphology / denominator if denominator else _nan()
        ),
        weak_motif_code_ratio=weak_motif / denominator if denominator else _nan(),
        weak_pair_code_ratio=weak_pair / denominator if denominator else _nan(),
        weak_lift_nonprofitable_code_ratio=(
            weak_lift_nonprofitable / denominator if denominator else _nan()
        ),
        intra_code_action_similarity=compute_intra_code_action_similarity(
            val_snapshot.decoded_actions,
            code_ids,
            active_codes,
        ),
        inter_intra_separation=compute_inter_intra_separation(
            val_snapshot.decoded_actions,
            code_ids,
            active_codes,
        ),
        latent_silhouette_score=_approx_silhouette(
            val_snapshot.z_e,
            code_ids,
            active_codes,
        ),
        duplicate_code_pair_count=compute_duplicate_code_pair_count(
            val_snapshot.decoded_actions,
            code_ids,
            active_codes,
            threshold=thresholds.duplicate_code_similarity_max,
        ),
        profitable_code_coverage=(
            profitable_count / denominator
            if denominator and per_code_profitability is not None
            else _nan()
        ),
        num_codes=num_codes,
    )
    return Phase1BehaviorQualityComputation(
        metrics=metrics,
        code_diagnostics=tuple(diagnostics),
        behavior_quality_payload=Phase1BehaviorQualityPayload(
            morphology_labels=morphologies,
            motif_labels=motifs,
            active_codes=active_codes,
        ),
    )


__all__ = [
    "classify_action_motif",
    "classify_market_morphology",
    "compute_distribution_by_code",
    "compute_duplicate_code_pair_count",
    "compute_intra_code_action_similarity",
    "compute_inter_intra_separation",
    "compute_lift",
    "compute_behavior_quality_metrics",
]
