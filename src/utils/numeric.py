"""通用数值工具。"""

from __future__ import annotations


def nan_value() -> float:
    """返回标准 NaN 标记。"""

    return float("nan")


__all__ = ["nan_value"]
