"""Phase II validation layer computation shared schema."""

from __future__ import annotations

from src.utils import PydanticBaseModel


class Phase2LayerComputationBase(PydanticBaseModel):
    """单个 Phase II validation layer computation 的公共字段。"""

    # layer 数字编号，0 到 5。用途：固定 layer 顺序；方向：无好坏方向。
    layer_id: int

    # layer 稳定名称，例如 "selector_profitability"。用途：反序列化和 report
    # 分组；方向：无好坏方向。
    layer_name: str

    # 各层 raw metrics 公共访问入口；具体子类会收窄为本层 metrics 类型。
    metrics: object


__all__ = ["Phase2LayerComputationBase"]
