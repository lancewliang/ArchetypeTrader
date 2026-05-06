"""读取 train/val/test 三个 ``.feather`` 数据文件.

设计文档锚点: §3.1 与 §4.3。

职责:
- 用 ``polars.read_ipc`` 读取 Feather/Arrow IPC。
- 调试 fixture 允许 ``polars.read_csv``。
- 不做 schema 校验、不做特征工程；仅返回原始 DataFrame。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Literal

from src.utils import feather_io


SplitName = Literal["train", "val", "test"]


class MarketFileReader:
    """单文件读取器。

    使用方式::

        reader = MarketFileReader()
        train_frame = reader.read("data/AL/train.feather")

    主路径只走 Feather/Arrow IPC；CSV 仅作为本地 fixture/调试通道暴露，
    避免生产代码误用 CSV 造成类型推断不稳。
    """

    SUPPORTED_SUFFIXES = (".feather", ".arrow", ".csv")

    def read(self, path):
        """根据扩展名分发读取函数。

        Returns
        -------
        ``polars.DataFrame``

        Raises
        ------
        FileNotFoundError : 文件不存在。
        ValueError : 文件扩展名不在 ``SUPPORTED_SUFFIXES`` 内。
        """
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(f"market 数据文件不存在: {target}")
        suffix = target.suffix.lower()
        if suffix in (".feather", ".arrow"):
            return feather_io.read_ipc(target)
        if suffix == ".csv":
            # 仅供 fixture / 调试；主路径必须使用 Feather。
            return feather_io.read_csv_for_debug(target)
        raise ValueError(
            f"不支持的扩展名: {suffix!r}; 仅允许 {self.SUPPORTED_SUFFIXES}"
        )

    def read_split(self, train_file, val_file, test_file) -> Dict[str, object]:
        """便捷方法: 一次读取三个 split。

        Returns
        -------
        dict : ``{"train": frame, "val": frame, "test": frame}``。
        """
        return {
            "train": self.read(train_file),
            "val": self.read(val_file),
            "test": self.read(test_file),
        }
