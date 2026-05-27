from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from src.utils import PydanticBaseModel


class VQCodeDistributionPayload(PydanticBaseModel):
    """ 
    validation split 的
    """

    # code 使用分布，索引为 code id，值为 occupancy probability。
    # validation split /code_distribution_total_sample_count
    code_distribution: tuple[float, ...]
    
     # code 使用分布，索引为 code id，值为 assigned 样本数。
    code_distribution_sample_count: tuple[int, ...]
    
    # validation split 中达到 active occupancy 阈值的 code id。
    active_codes: tuple[int, ...] 
    
    # 参与 code distribution 统计的样本数。
    code_distribution_total_sample_count: int


class CodeDistribution(PydanticBaseModel):
    """代码使用情况。"""

    code_id: str
    # 通过 phase1 vpmodel 生成的 demo 数量。
    vp_demo_sample_count: int
    # vp_demo_sample_count/totals_sample_count*100
    vp_demo_sample_ratio: int
    # 通过 phase1 vpmodel 生成的 demo 中，最佳 code 数量。
    best_code_sample_count: int
    # best_code_sample_count/totals_sample_count*100
    best_code_sample_ratio: int
    # 通过 phase2 selector 选择的 code 的样本数量。
    selector_sample_count: int
    # selector_sample_count/totals_sample_count*100
    selector_sample_ratio: int
    # HTML/CSS 条形图宽度。
    bar_width: int
    # 该 code 是否 active。
    active: bool


class CodeDistributionView(PydanticBaseModel):
    # 总代码数量。
    total_code_count: int
    # 总样本数量。
    totals_sample_count: int
    # 代码使用情况行。
    code_usage_rows: tuple[CodeDistribution, ...]


def buildVQCodeDistributionPayload(
    *,
    code_ids: Sequence[int],
    num_codes: int,
    active_code_min_occupancy: float,
) -> VQCodeDistributionPayload:
    """构建 VQ code 使用分布 payload。"""

    code_id_values = np.asarray(code_ids, dtype=np.int64).reshape(-1)
    code_distribution = compute_code_distribution(code_id_values, num_codes)
    code_distribution_sample_count = compute_code_sample_counts(
        code_id_values,
        num_codes,
    )
    active_codes = tuple(
        int(code_id)
        for code_id, occupancy in enumerate(code_distribution)
        if occupancy >= active_code_min_occupancy
    )
    return VQCodeDistributionPayload(
        code_distribution=tuple(float(value) for value in code_distribution),
        code_distribution_sample_count=tuple(
            int(sample_count) for sample_count in code_distribution_sample_count
        ),
        active_codes=active_codes,
        code_distribution_total_sample_count=int(code_id_values.size),
    )


def compute_code_distribution(code_ids: Sequence[int], k: int) -> np.ndarray:
    """计算 code occupancy 分布。"""

    if k <= 0:
        return np.asarray([], dtype=np.float64)
    values = np.asarray(code_ids, dtype=np.int64).reshape(-1)
    if np.any((values < 0) | (values >= k)):
        raise ValueError("code_ids must be in [0, k)")
    counts = np.bincount(values, minlength=k)
    total = np.sum(counts)
    if total <= 0:
        return np.zeros(k, dtype=np.float64)
    return counts.astype(np.float64) / float(total)


def compute_code_sample_counts(code_ids: Sequence[int], k: int) -> np.ndarray:
    """计算每个 code 获得的样本数。"""

    if k <= 0:
        return np.asarray([], dtype=np.int64)
    values = np.asarray(code_ids, dtype=np.int64).reshape(-1)
    if np.any((values < 0) | (values >= k)):
        raise ValueError("code_ids must be in [0, k)")
    return np.bincount(values, minlength=k).astype(np.int64)


def build_code_distribution_context(
    payload: VQCodeDistributionPayload,
) -> CodeDistributionView:
    """构建 codebook 使用分布的模板上下文。"""

    distribution = payload.code_distribution
    active_codes = {int(code_id) for code_id in payload.active_codes}
    total_sample_count = payload.code_distribution_total_sample_count
    rows: list[CodeDistribution] = []
    for code_id, occupancy in enumerate(distribution):
        vp_demo_count = payload.code_distribution_sample_count[code_id]
        rows.append(
            CodeDistribution(
                code_id=str(code_id),
                vp_demo_sample_count=vp_demo_count,
                vp_demo_sample_ratio=_occupancy_bar_width(occupancy),
                best_code_sample_count=0,
                best_code_sample_ratio=0,
                selector_sample_count=0,
                selector_sample_ratio=0,
                bar_width=_occupancy_bar_width(occupancy),
                active=code_id in active_codes,
            )
        )
    return CodeDistributionView(
        total_code_count=len(distribution),
        totals_sample_count=total_sample_count,
        code_usage_rows=tuple(rows),
    )


def _occupancy_bar_width(occupancy: float) -> int:
    """把 occupancy 转换成 0-100 的条形宽度整数。"""

    if not math.isfinite(occupancy):
        return 0
    return int(round(max(0.0, min(100.0, occupancy * 100.0))))
