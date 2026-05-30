from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from model.data_types import DemonstrationHorizonLabelDataset, HorizonDataset, HorizonLabelDataset, VisibleStatesDataset, VisibleStatesLabelDataset
from src.utils import PydanticBaseModel

import torch

from model.vq_archetype import ArchetypeVQModel
from src.phase1.evaluators.phase1_validation_layers.layer3_oracle_profitability import decode_labels
from src.utils import ActionExecutionCalculator

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
    
def _buildCodeDistributionPayload(
    *,
    code_id_values: Sequence[int],
    num_codes: int,
    active_code_min_occupancy: float,
) -> VQCodeDistributionPayload:
    code_distribution = _compute_code_distribution(code_id_values, num_codes)
    code_distribution_sample_count = _compute_code_sample_counts(
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

def buildDemonstrationCodeDistributionPayload(
    *,
    dataset: DemonstrationHorizonLabelDataset,
    num_codes: int,
    active_code_min_occupancy: float,
) -> VQCodeDistributionPayload:
    """构建 Demonstration VQ code 使用分布 payload。"""

    _, code_labels = dataset
    code_id_values = np.asarray(code_labels, dtype=np.int64).reshape(-1)
    return _buildCodeDistributionPayload(
        code_id_values=code_id_values,
        num_codes=num_codes,
        active_code_min_occupancy=active_code_min_occupancy,
    )
    
def buildBestRewardCodeDistributionPayload(
    *,
    model: ArchetypeVQModel,
    dataset: HorizonDataset,
    num_codes: int,
    active_code_min_occupancy: float,
    device: torch.device | str = "cpu",
    fee_rate: float = 0.0004,
) -> VQCodeDistributionPayload:
    """对每个样本枚举所有 code label，decode 动作并计算 reward，统计最优 code 分布。"""

    states, relative_states, trend_states, prices, depthprices = dataset
    sample_count = int(np.asarray(states).shape[0])
    if sample_count == 0:
        return _buildCodeDistributionPayload(
            code_id_values=np.asarray([], dtype=np.int64),
            num_codes=num_codes,
            active_code_min_occupancy=active_code_min_occupancy,
        )

    best_code_ids = np.zeros(sample_count, dtype=np.int64)
    best_returns = np.full(sample_count, -np.inf, dtype=np.float64)

    torch_device = torch.device(device)
    model = model.to(torch_device)
    model.eval()

    with torch.no_grad():
        for code_id in range(num_codes):
            code_ids = np.full(sample_count, code_id, dtype=np.int64)

            actions = decode_labels(
                model=model,
                states=states,
                relative_states=relative_states,
                trend_states=trend_states,
                code_ids=code_ids,
                device=torch_device,
            )

            returns = ActionExecutionCalculator.execute_actions(
                prices=prices,
                actions=actions,
                fee_rate=fee_rate,
                depthprices=depthprices,
            ).returns

            improved = returns > best_returns
            best_returns[improved] = returns[improved]
            best_code_ids[improved] = code_id

    return _buildCodeDistributionPayload(
        code_id_values=best_code_ids,
        num_codes=num_codes,
        active_code_min_occupancy=active_code_min_occupancy,
    )
    
def buildSelectorCodeDistributionPayload(
    dataset: VisibleStatesLabelDataset,
    num_codes: int,
    active_code_min_occupancy: float    
) -> VQCodeDistributionPayload:
    """对每个样本选择器选中的 code，使用分布 payload。"""
    _, code_labels = dataset
    code_id_values = np.asarray(code_labels, dtype=np.int64).reshape(-1)
    return _buildCodeDistributionPayload(
        code_id_values=code_id_values,
        num_codes=num_codes,
        active_code_min_occupancy=active_code_min_occupancy,
    )

def _compute_code_distribution(code_ids: Sequence[int], k: int) -> np.ndarray:
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


def _compute_code_sample_counts(code_ids: Sequence[int], k: int) -> np.ndarray:
    """计算每个 code 获得的样本数。"""

    if k <= 0:
        return np.asarray([], dtype=np.int64)
    values = np.asarray(code_ids, dtype=np.int64).reshape(-1)
    if np.any((values < 0) | (values >= k)):
        raise ValueError("code_ids must be in [0, k)")
    return np.bincount(values, minlength=k).astype(np.int64)


