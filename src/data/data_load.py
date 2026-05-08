"""Feature 数据文件读取类的接口骨架。"""

from __future__ import annotations

from pathlib import Path

import polars as pl


class DataLoad:
    """负责读取外部 feature 输入文件。

    为什么需要这个类:
        文件读取属于 I/O 责任，不应该混在 ``DataPreparer`` 的流程编排里。
        单独拆出后，后续可以独立支持 feather、parquet、csv 等格式，
        而不影响 horizon 构建和 trajectory 生成逻辑。
    """

    def load_feature_file(
        self,
        path: str | Path,
    ) -> pl.DataFrame:
        """读取 feature 输入文件。

        参数:
            path: feature 输入文件路径。文件内容读取后应转换成 ``pl.DataFrame``。

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

        if "close" not in dataframe.columns:
            raise ValueError("feature dataframe must contain a 'close' column")
        if dataframe.is_empty():
            raise ValueError("feature dataframe is empty")
        return dataframe
