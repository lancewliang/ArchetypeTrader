"""从时间序列数据构建固定 horizon 的状态与价格数据立方体。"""

from __future__ import annotations

import numpy as np
import polars as pl

from ..model.data_types import HorizonDataset
from .feature_spec import FeatureBlock, FeatureInputSpec
from .state_normalizer import StateNormalizer
from ..utils.trade_execution import (
    LOB_ASK_PRICE_COLS,
    LOB_ASK_SIZE_COLS,
    LOB_BID_PRICE_COLS,
    LOB_BID_SIZE_COLS,
)


LOB_DEPTH_COLS = (
    LOB_ASK_PRICE_COLS
    + LOB_ASK_SIZE_COLS
    + LOB_BID_PRICE_COLS
    + LOB_BID_SIZE_COLS
)


class HorizonBuilder:
    """将时间序列 DataFrame 转换为 ``numpy.ndarray`` 张量。

    输出 ``states`` 张量的 ``shape`` 为 ``[x, h, len(states)]``。
    输出 ``prices`` 张量的 ``shape`` 为 ``[x, h, 1]``。
    输出 ``depthprices`` 张量的 ``shape`` 为 ``[x, h, 20]``。
    ``h`` 默认值为 72，``x`` 预期为 ``len(dataframe) / h``。
    其中 ``dataframe`` 的 ``close`` 列就是价格列。
    ``states`` 是输入 ``dataframe`` 中排除 ``close`` 后的所有列，
    ``prices`` 是单独取出的 ``close`` 列。
    ``depthprices`` 从 ``states`` 中切出五档 ask/bid price 和 size 特征。
    如果配置了 state normalizer，``states`` 会在这里被标准化，但
    ``prices`` 和 ``depthprices`` 始终保持原始尺度，供 DP 和成本计算使用。
    """

    def __init__(
        self,
        horizon: int = 72,
        feature_spec: FeatureInputSpec | None = None,
    ) -> None:
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
        if horizon <= 1:
            raise ValueError("horizon must be greater than 1")
        self.horizon = horizon
        self.feature_spec = feature_spec
        self.state_normalizer: StateNormalizer | None = None
        self.feature_normalizers: dict[str, StateNormalizer] = {}

    def set_state_normalizer(self, state_normalizer: StateNormalizer | None) -> None:
        """设置 state 归一化器。"""

        self.state_normalizer = state_normalizer

    def set_feature_normalizers(
        self,
        feature_normalizers: dict[str, StateNormalizer] | None,
    ) -> None:
        """设置 feature block 归一化器。"""

        self.feature_normalizers = dict(feature_normalizers or {})

    def build(
        self,
        dataframe: pl.DataFrame,
    ) -> HorizonDataset:
        """从 ``dataframe`` 构建 horizon 状态张量、价格张量和深度行情张量。

        参数:
            dataframe: 输入的时间序列数据表，每一行代表一个时间步。
                其中 ``close`` 列就是价格列，会被单独返回为 ``prices``。

        输出:
            返回 ``HorizonDataset``，即 ``(states, prices, depthprices)``。
            ``states`` 的 ``shape`` 为 ``[x, h, len(states)]``。
            ``prices`` 的 ``shape`` 为 ``[x, h, 1]``。
            ``depthprices`` 的 ``shape`` 为 ``[x, h, 20]``。
            其中 ``h = horizon``，``x = len(dataframe) / h``。
            ``states`` 不包含 ``close`` 列，``prices`` 只包含 ``close`` 列。
            ``depthprices`` 从 ``states`` 中的 LOB price/size 特征切出。

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
        if "close" not in dataframe.columns:
            raise ValueError("dataframe must contain a 'close' column")

        if self.feature_spec is not None:
            return self._build_from_feature_spec(dataframe)

        return self._build_legacy(dataframe)

    def _build_legacy(self, dataframe: pl.DataFrame) -> HorizonDataset:
        """按旧单路 states 逻辑构建数据，兼容迁移期调用。"""

        state_columns = [
            column
            for column, dtype in dataframe.schema.items()
            if column != "close" and dtype.is_numeric()
        ]
        if not state_columns:
            raise ValueError("dataframe must contain at least one state feature column")
        missing_depth_columns = [
            column for column in LOB_DEPTH_COLS if column not in state_columns
        ]
        if missing_depth_columns:
            missing = ", ".join(missing_depth_columns)
            raise ValueError(
                f"dataframe must contain numeric LOB depth columns: {missing}"
            )

        usable_rows = len(dataframe) // self.horizon * self.horizon
        if usable_rows == 0:
            raise ValueError(
                f"dataframe has {len(dataframe)} rows, fewer than horizon={self.horizon}"
            )
        dataframe = dataframe.head(usable_rows)

        state_values = dataframe.select(state_columns).to_numpy().astype(
            np.float32,
            copy=False,
        )
        raw_states = state_values.reshape(-1, self.horizon, len(state_columns))
        depth_indices = [state_columns.index(column) for column in LOB_DEPTH_COLS]
        depthprices = raw_states[..., depth_indices].copy()
        if self.state_normalizer is not None:
            if tuple(state_columns) != self.state_normalizer.feature_columns:
                raise ValueError(
                    "dataframe state columns do not match the configured normalizer"
                )
            state_values = self.state_normalizer.transform(state_values)
        states = state_values.reshape(-1, self.horizon, len(state_columns))
        prices = (
            dataframe.select("close")
            .to_numpy()
            .astype(np.float32, copy=False)
            .reshape(-1, self.horizon, 1)
        )
        if not np.isfinite(states).all():
            raise ValueError("state features contain non-finite values")
        if not np.isfinite(prices).all():
            raise ValueError("close prices contain non-finite values")
        if not np.isfinite(depthprices).all():
            raise ValueError("LOB depth features contain non-finite values")
        empty = np.empty((states.shape[0], states.shape[1], 0), dtype=np.float32)
        return states, empty, empty, prices, depthprices

    def _build_from_feature_spec(self, dataframe: pl.DataFrame) -> HorizonDataset:
        """按三路 feature spec 构建 ``states/relative_states/trend_states``。"""

        if self.feature_spec is None:
            raise ValueError("feature_spec is required")

        self._validate_required_columns(dataframe, self.feature_spec.required_columns)

        usable_rows = len(dataframe) // self.horizon * self.horizon
        if usable_rows == 0:
            raise ValueError(
                f"dataframe has {len(dataframe)} rows, fewer than horizon={self.horizon}"
            )
        dataframe = dataframe.head(usable_rows)

        states = self._build_state_tensor(dataframe, self.feature_spec.state_blocks)
        relative_states = self._build_state_tensor(
            dataframe,
            self.feature_spec.relative_state_blocks,
        )
        trend_states = self._build_state_tensor(
            dataframe,
            self.feature_spec.trend_state_blocks,
        )
        prices = (
            dataframe.select("close")
            .to_numpy()
            .astype(np.float32, copy=False)
            .reshape(-1, self.horizon, 1)
        )
        depthprices = self._build_depthprices(dataframe)

        for name, values in (
            ("states", states),
            ("relative_states", relative_states),
            ("trend_states", trend_states),
            ("prices", prices),
            ("depthprices", depthprices),
        ):
            if not np.isfinite(values).all():
                raise ValueError(f"{name} contain non-finite values")
        return states, relative_states, trend_states, prices, depthprices

    def _build_state_tensor(
        self,
        dataframe: pl.DataFrame,
        blocks: tuple[FeatureBlock, ...],
    ) -> np.ndarray:
        """Build one state tensor from ordered feature blocks."""

        values = [
            self._build_block_values(dataframe, block)
            for block in blocks
        ]
        matrix = np.concatenate(values, axis=1)
        return matrix.reshape(-1, self.horizon, matrix.shape[1])

    def _build_block_values(
        self,
        dataframe: pl.DataFrame,
        block: FeatureBlock,
    ) -> np.ndarray:
        """Read and optionally normalize one feature block."""

        self._validate_required_columns(dataframe, block.columns)
        values = dataframe.select(list(block.columns)).to_numpy().astype(
            np.float32,
            copy=False,
        )
        if not block.normalize:
            return values

        key = block.effective_normalizer_key
        normalizer = self.feature_normalizers.get(key)
        if normalizer is None:
            raise ValueError(f"missing feature normalizer for block {key!r}")
        if tuple(block.columns) != normalizer.feature_columns:
            raise ValueError(
                f"feature normalizer {key!r} columns do not match block columns"
            )
        return normalizer.transform(values)

    def _build_depthprices(self, dataframe: pl.DataFrame) -> np.ndarray:
        """Build unnormalized LOB depth tensor from raw dataframe columns."""

        self._validate_required_columns(dataframe, LOB_DEPTH_COLS)
        return (
            dataframe.select(list(LOB_DEPTH_COLS))
            .to_numpy()
            .astype(np.float32, copy=False)
            .reshape(-1, self.horizon, len(LOB_DEPTH_COLS))
        )

    @staticmethod
    def _validate_required_columns(
        dataframe: pl.DataFrame,
        columns: tuple[str, ...],
    ) -> None:
        """Validate that required columns exist and are numeric."""

        missing = [column for column in columns if column not in dataframe.columns]
        if missing:
            raise ValueError(
                "dataframe is missing required columns: " + ", ".join(missing)
            )
        non_numeric = [
            column
            for column in columns
            if not dataframe.schema[column].is_numeric()
        ]
        if non_numeric:
            raise ValueError(
                "dataframe required columns must be numeric: "
                + ", ".join(non_numeric)
            )
