from __future__ import annotations

import math
from typing import TYPE_CHECKING

from src.analysis.analysis_code_distribution_model import VQCodeDistributionPayload
from src.utils import PydanticBaseModel

if TYPE_CHECKING:
    from src.phase1.metrics import Phase1VQInternalPayload


class CodeDistribution(PydanticBaseModel):
    """代码使用情况。"""
    code_id: str
    # 通过 phase1 vpmodel 生成的 demo 数量。
    vp_demo_sample_count: int
     #　vp_demo_sample_count/totals_sample_count*100
    vp_demo_sample_ratio: int
    # 通过 phase1 vpmodel 生成的 demo 中，最佳 code 数量。
    best_code_sample_count: int
    #　best_code_sample_count/totals_sample_count*100
    best_code_sample_ratio: int
    # 通过 phase2 selector 选择的 code 的样本数量。
    selector_sample_count: int
    #　selector_sample_count/totals_sample_count*100
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
                vp_demo_sample_ratio=occupancy,
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
