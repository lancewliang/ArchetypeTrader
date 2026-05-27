"""Phase I validation layer computation shared schema."""

from __future__ import annotations

from src.utils import PydanticBaseModel


class Phase1LayerComputationBase(PydanticBaseModel):
    """单个 validation layer raw metric 计算结果的公共字段。"""

    # layer 数字编号，0 到 4。
    layer_id: int

    # layer 稳定名称，例如 "teacher_quality"。
    layer_name: str


__all__ = ["Phase1LayerComputationBase"]
