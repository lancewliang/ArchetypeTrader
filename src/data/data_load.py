"""Feature 数据文件读取类的接口骨架。"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import polars as pl


def _dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


class DataLoad:
    """负责读取外部 feature 输入文件。

    为什么需要这个类:
        文件读取属于 I/O 责任，不应该混在 ``DataPreparer`` 的流程编排里。
        单独拆出后，后续可以独立支持 feather、parquet、csv 等格式，
        而不影响 horizon 构建和 trajectory 生成逻辑。
    """

    def __init__(
        self,
        feature_columns: Sequence[str] | None = None,
    ) -> None:
        """初始化 feature 读取器。

        参数:
            feature_columns: 可选列清单。提供时只保留这些列，通常由
                ``src.data.resolve_factor.build_feature_columns`` 生成。
        """

        self.feature_columns = (
            _dedupe_preserve_order(feature_columns)
            if feature_columns is not None
            else None
        )

    def load_feature_file(
        self,
        path: str | Path,
        feature_columns: Sequence[str] | None = None,
    ) -> pl.DataFrame:
        """读取 feature 输入文件。

        参数:
            path: feature 输入文件路径。文件内容读取后应转换成 ``pl.DataFrame``。
            feature_columns: 可选列清单。提供时覆盖初始化时的列清单。

        输出:
            返回 ``pl.DataFrame``。
            其中 ``close`` 列是价格列，其它列作为状态特征候选。

        方法作用:
            将磁盘上的 feature 文件读取到内存表结构中，
            供 ``HorizonBuilder`` 切分为 horizon 数据。

        为什么:
            ``DataPreparer`` 的公开入口只接收文件路径。
            在进入数据准备主流程前，需要先把文件统一读取为 ``pl.DataFrame``。
        """
        input_path = Path(path)
        if not input_path.exists():
            raise FileNotFoundError(f"feature file not found: {input_path}")

        suffix = input_path.suffix.lower()
        if suffix in {".feather", ".ipc", ".arrow"}:
            dataframe = pl.read_ipc(input_path)
        elif suffix == ".parquet":
            dataframe = pl.read_parquet(input_path)
        elif suffix == ".csv":
            dataframe = pl.read_csv(input_path)
        else:
            raise ValueError(
                "unsupported feature file format: "
                f"{suffix or '<no suffix>'}; expected feather/ipc/parquet/csv"
            )

        dataframe = self._normalize_close_column(dataframe)
        selected_columns = self._resolve_selected_columns(feature_columns)
        if selected_columns is not None:
            dataframe = self._select_feature_columns(dataframe, selected_columns)

        if "close" not in dataframe.columns:
            raise ValueError("feature dataframe must contain a 'close' column")
        if dataframe.is_empty():
            raise ValueError("feature dataframe is empty")
        return dataframe

    @staticmethod
    def _normalize_close_column(dataframe: pl.DataFrame) -> pl.DataFrame:
        """兼容原始行情文件中的 ``close_price`` 列名。"""

        if "close" not in dataframe.columns and "close_price" in dataframe.columns:
            return dataframe.rename({"close_price": "close"})
        return dataframe

    def _resolve_selected_columns(
        self,
        feature_columns: Sequence[str] | None,
    ) -> list[str] | None:
        selected = (
            feature_columns if feature_columns is not None else self.feature_columns
        )
        if selected is None:
            return None
        columns = _dedupe_preserve_order(selected)
        if "close" not in columns:
            columns.insert(0, "close")
        return columns

    @staticmethod
    def _select_feature_columns(
        dataframe: pl.DataFrame,
        selected_columns: Sequence[str],
    ) -> pl.DataFrame:
        missing_columns = [
            column for column in selected_columns if column not in dataframe.columns
        ]
        if missing_columns:
            missing = ", ".join(missing_columns)
            raise ValueError(
                f"feature dataframe is missing configured columns: {missing}"
            )

        non_numeric_columns = [
            column
            for column in selected_columns
            if column != "close" and not dataframe.schema[column].is_numeric()
        ]
        if non_numeric_columns:
            non_numeric = ", ".join(non_numeric_columns)
            raise ValueError(
                f"configured feature columns must be numeric: {non_numeric}"
            )

        if not dataframe.schema["close"].is_numeric():
            raise ValueError("close column must be numeric")
        return dataframe.select(selected_columns)
