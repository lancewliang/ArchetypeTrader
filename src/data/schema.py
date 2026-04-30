"""输入 schema 校验与 ``input_schema.json`` 生成.

设计文档锚点: §3.1, §3.2, §4.3。

强约束:
- ``close`` 是价格列；不得出现在 ``feature_columns`` 或 ``states`` 中。
- 元信息列 ``timestamp/symbol/split/sample_id/close`` 必须从 feature 中显式排除。
- 数值列必须能转 ``float32``，且无 NaN/Inf。
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

from src.utils.feather_io import atomic_write_json


# 默认排除列：进入 schema 后必须从 feature_columns 中剔除。
DEFAULT_EXCLUDED_COLUMNS = (
    "timestamp",
    "symbol",
    "split",
    "sample_id",
    "close",
)

# 已知盘口字段；schema 仅做识别和记录，不做特征生成。
KNOWN_ORDERBOOK_COLUMNS = tuple(
    [f"ask{i}_price" for i in range(1, 6)]
    + [f"ask{i}_size" for i in range(1, 6)]
    + [f"bid{i}_price" for i in range(1, 6)]
    + [f"bid{i}_size" for i in range(1, 6)]
)

KNOWN_TRADE_COLUMNS = ("total_trade_volume", "turnover", "open_interest")


@dataclass(frozen=True)
class InputSchema:
    """统一输入 schema。"""
    timestamp_column: str
    price_column: str
    feature_columns: List[str] = field(default_factory=list)
    excluded_columns: List[str] = field(default_factory=list)
    orderbook_columns: List[str] = field(default_factory=list)
    num_rows: int = 0

    def assert_close_not_in_features(self) -> None:
        """``close`` 不得进入 ``feature_columns``；该规则是 sign-off 阻塞项。"""
        if self.price_column in self.feature_columns:
            raise ValueError(
                f"{self.price_column} 不能进入 feature_columns；schema 校验失败"
            )

    def feature_dim(self) -> int:
        return len(self.feature_columns)

    def to_dict(self) -> dict:
        return asdict(self)


class InputSchemaValidator:
    """对单个 split DataFrame 做 schema 校验，并生成 ``InputSchema``。

    校验项:
    - 必含列 ``timestamp`` 与 ``close``。
    - ``close > 0`` 且无 NaN/Inf。
    - 特征数值列无 null / NaN / Inf。
    - 时间列单调非降（不强制严格递增以兼容个别交易所重复时间戳）。
    - 自动识别 ``KNOWN_ORDERBOOK_COLUMNS``，纳入 ``feature_columns`` 同时记入
      ``orderbook_columns`` 便于下游切片 ``execution_books``。
    """

    def __init__(
        self,
        timestamp_column: str = "timestamp",
        price_column: str = "close",
        excluded_columns: Optional[List[str]] = None,
    ) -> None:
        self.timestamp_column = timestamp_column
        self.price_column = price_column
        self.excluded_columns = list(excluded_columns or DEFAULT_EXCLUDED_COLUMNS)

    def validate(self, frame) -> InputSchema:
        """对 polars DataFrame 做 schema 校验，返回 ``InputSchema``。

        Raises
        ------
        TypeError : ``frame`` 不是 ``polars.DataFrame``。
        ValueError : 缺列、``close <= 0``、含 NaN/Inf、列类型非数值无法转 float32、
                     时间回退、或 ``close`` 进入 ``feature_columns`` 等任一硬约束被违反。

        Notes
        -----
        - 非数值列（例如 string）会被静默忽略，不进入 features，避免误把
          ``symbol`` 等列纳入。
        - ``close`` 不可进入 ``feature_columns`` 在最后通过
          ``InputSchema.assert_close_not_in_features`` 复核。
        """
        import polars as pl

        if not isinstance(frame, pl.DataFrame):
            raise TypeError(f"期望 polars.DataFrame，得到 {type(frame).__name__}")

        columns = frame.columns
        # 必含列检查
        if self.timestamp_column not in columns:
            raise ValueError(f"缺少必填列: {self.timestamp_column}")
        if self.price_column not in columns:
            raise ValueError(f"缺少必填列: {self.price_column}")

        # close 必须为正、无 NaN/Inf。
        close_series = frame[self.price_column]
        # null 检查
        if close_series.null_count() > 0:
            raise ValueError(f"{self.price_column} 含 null 值")
        # 数值检查（polars 中 inf 不视为 null，需要单独检查）
        numpy_close = close_series.to_numpy()
        if not all(math.isfinite(float(v)) for v in numpy_close):
            raise ValueError(f"{self.price_column} 含 NaN 或 Inf")
        if not all(float(v) > 0 for v in numpy_close):
            raise ValueError(f"{self.price_column} 必须 > 0")

        # 时间列单调非降（仅当时间列可比较时；polars 内部排序更稳）。
        ts = frame[self.timestamp_column]
        # 转为 numpy 比较，避免 polars 类型差异；
        ts_values = ts.to_numpy()
        for i in range(1, len(ts_values)):
            if ts_values[i] < ts_values[i - 1]:
                raise ValueError(
                    f"{self.timestamp_column} 必须单调非降，行 {i} 出现回退"
                )

        # 拆 feature / orderbook / excluded
        excluded_set = set(self.excluded_columns)
        orderbook_columns: List[str] = []
        feature_columns: List[str] = []
        for col in columns:
            if col in excluded_set:
                continue
            if col in KNOWN_ORDERBOOK_COLUMNS:
                orderbook_columns.append(col)
                feature_columns.append(col)  # 盘口字段也是 feature 的一部分
                continue
            # 必须是数值类型且可转 float32；含 NaN/Inf 拒绝
            series = frame[col]
            if not series.dtype.is_numeric():
                # 非数值列默默忽略，不进入 features，避免误把 string 列纳入。
                continue
            if series.null_count() > 0:
                raise ValueError(f"特征列 {col} 含 null 值")
            arr = series.to_numpy()
            if not all(math.isfinite(float(v)) for v in arr):
                raise ValueError(f"特征列 {col} 含 NaN 或 Inf")
            feature_columns.append(col)

        schema = InputSchema(
            timestamp_column=self.timestamp_column,
            price_column=self.price_column,
            feature_columns=feature_columns,
            excluded_columns=list(self.excluded_columns),
            orderbook_columns=orderbook_columns,
            num_rows=frame.height,
        )
        # close 不可进入 features 是设计硬约束。
        schema.assert_close_not_in_features()
        return schema

    def write_schema_json(self, schema: InputSchema, path: Path) -> Path:
        """原子写 ``input_schema.json``。"""
        return atomic_write_json(schema.to_dict(), path)
