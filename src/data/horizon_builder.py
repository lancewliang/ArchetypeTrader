"""从时间序列数据构建固定 horizon 的状态与价格数据立方体。"""

from __future__ import annotations

import numpy as np
import polars as pl

from .data_types import HorizonDataset


class HorizonBuilder:
    """将时间序列 DataFrame 转换为 ``numpy.ndarray`` 张量。

    输出 ``states`` 张量的 ``shape`` 为 ``[x, h, len(states)]``。
    输出 ``prices`` 张量的 ``shape`` 为 ``[x, h, 1]``。
    ``h`` 默认值为 72，``x`` 预期为 ``len(dataframe) / h``。
    其中 ``dataframe`` 的 ``close`` 列就是价格列。
    ``states`` 是输入 ``dataframe`` 中排除 ``close`` 后的所有列，
    ``prices`` 是单独取出的 ``close`` 列。
    """

    def __init__(self, horizon: int = 72) -> None:
        """初始化 HorizonBuilder。

        参数:
            horizon: 时间窗口长度 ``h``，默认 72。用于将时间序列按固定步长切分。

        输出:
            无返回值。

        方法作用:
            保存 horizon 长度，供 ``build`` 统一切分状态和价格。

        为什么:
            Phase I 的 demonstration trajectory 需要 ``s_demo`` 作为模型输入，
            同时 DP teacher 需要 ``close`` 价格序列计算 ``a_demo`` 和 ``r_demo``。
            将 ``close`` 单独返回可以避免把结算价格误混入状态特征。
        """
        ...

    def build(
        self,
        dataframe: pl.DataFrame,
    ) -> HorizonDataset:
        """从 ``dataframe`` 构建 horizon 状态张量和价格张量。

        参数:
            dataframe: 输入的时间序列数据表，每一行代表一个时间步。
                其中 ``close`` 列就是价格列，会被单独返回为 ``prices``。

        输出:
            返回 ``HorizonDataset``，即 ``(states, prices)``。
            ``states`` 的 ``shape`` 为 ``[x, h, len(states)]``。
            ``prices`` 的 ``shape`` 为 ``[x, h, 1]``。
            其中 ``h = horizon``，``x = len(dataframe) / h``。
            ``states`` 不包含 ``close`` 列，``prices`` 只包含 ``close`` 列。

        方法作用:
            将连续时间序列按固定 horizon 切成多个样本。
            ``states`` 用于模型输入，``prices`` 使用 ``close`` 列，
            用于 DP planner 和 reward 计算。

        为什么:
            论文中的 ``tau = (s_demo, a_demo, r_demo)`` 需要状态序列，
            但生成 ``a_demo`` 和 ``r_demo`` 还必须有价格序列。
            同时返回价格可以保证 HorizonBuilder 的输出直接供后续
            Single-trade DP planner 使用。
        """
        ...
