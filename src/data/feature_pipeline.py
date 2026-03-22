"""特征加载与处理管道

负责加载 train/val/test 三个 feather 文件的数据，使用 polars 高效处理。
"""

from typing import Tuple

import polars as pl
import pyarrow.feather as pa_feather


SINGLE_FEATURES = [
    "ask1_price", "ask1_size", "bid1_price", "bid1_size",
    "ask2_price", "ask2_size", "bid2_price", "bid2_size",
    "ask3_price", "ask3_size", "bid3_price", "bid3_size",
    "ask4_price", "ask4_size", "bid4_price", "bid4_size",
    "ask5_price", "ask5_size", "bid5_price", "bid5_size",
    "volume",
    "bid1_size_n", "bid2_size_n", "bid3_size_n", "bid4_size_n", "bid5_size_n",
    "ask1_size_n", "ask2_size_n", "ask3_size_n", "ask4_size_n", "ask5_size_n",
    "wap_1", "wap_2", "wap_balance",
    "buy_spread", "sell_spread",
]

TREND_FEATURES = [
    "ask1_price_trend_60", "bid1_price_trend_60",
    "buy_spread_trend_60", "sell_spread_trend_60",
    "wap_1_trend_60", "wap_2_trend_60",
    "buy_vwap_trend_60", "sell_vwap_trend_60",
    "volume_trend_60",
]


class FeaturePipeline:
    """特征加载与处理管道

    直接从 feather 文件加载 train/val/test 数据集，
    并筛选出与 npy 文件对应的 single_features (36维) 和 trend_features (9维) 特征。
    """

    def __init__(self, data_dir: str, pair: str):
        """
        Args:
            data_dir: 数据根目录路径（包含 df_train.feather, df_val.feather, df_test.feather）
            pair: 交易对名称，如 'BTC', 'ETH', 'DOT', 'BNB'
        """
        self.data_dir = data_dir
        self.pair = pair
        self._loaded = False
        self._raw_train: pl.DataFrame | None = None
        self._raw_val: pl.DataFrame | None = None
        self._raw_test: pl.DataFrame | None = None

    def _load_data(self) -> None:
        """加载原始数据，只执行一次"""
        if self._loaded:
            return

        train_path = f"{self.data_dir}/df_train.feather"
        val_path = f"{self.data_dir}/df_val.feather"
        test_path = f"{self.data_dir}/df_test.feather"

        self._raw_train = pl.DataFrame._from_arrow(pa_feather.read_table(train_path))
        self._raw_val = pl.DataFrame._from_arrow(pa_feather.read_table(val_path))
        self._raw_test = pl.DataFrame._from_arrow(pa_feather.read_table(test_path))

        self._loaded = True

    def get_state_vector(self) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """返回 train/val/test 三个数据集，筛选出对应的特征列。

        特征顺序: single_features (36维) + trend_features (9维) = 45维状态向量

        Returns:
            (train_df, val_df, test_df) 三元组，每个都是 polars DataFrame
        """
        self._load_data()

        feature_cols = SINGLE_FEATURES + TREND_FEATURES
        missing = [c for c in feature_cols if c not in self._raw_train.columns]
        if missing:
            raise ValueError(f"缺少特征列: {missing}")

        return (
            self._raw_train.select(feature_cols),
            self._raw_val.select(feature_cols),
            self._raw_test.select(feature_cols),
        )

    def get_prices(self) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """返回 train/val/test 三个数据集的价格列。

        Returns:
            (train_prices, val_prices, test_prices) 三元组，每个都是包含 'close' 列的 polars DataFrame
        """
        self._load_data()

        if "close" not in self._raw_train.columns:
            raise ValueError(f"缺少 'close' 列，可用列: {self._raw_train.columns}")

        return (
            self._raw_train.select("close"),
            self._raw_val.select("close"),
            self._raw_test.select("close"),
        )

    def split_into_horizons(
        self, df: pl.DataFrame, h: int = 72
    ) -> list[pl.DataFrame]:
        """按 horizon 长度切分数据。

        Args:
            df: 待切分的 DataFrame
            h: horizon 长度，默认 72

        Returns:
            切分后的 DataFrame 列表
        """
        if h <= 0:
            raise ValueError(f"horizon 长度必须为正整数，收到 h={h}")

        T = df.height
        horizons = []
        for start in range(0, T, h):
            end = min(start + h, T)
            horizons.append(df[start:end])
        return horizons
