"""特征加载与处理管道。

负责加载 train/val/test 三个 feather 文件的数据，并按
`fixed_features + cycle_features` 组装状态向量。
"""

from typing import Iterable, Tuple

import polars as pl
import pyarrow.feather as pa_feather


FIXED_FEATURES = [
    "close",
    "ask1_price", "ask1_size", "bid1_price", "bid1_size",
    "ask2_price", "ask2_size", "bid2_price", "bid2_size",
    "ask3_price", "ask3_size", "bid3_price", "bid3_size",
    "ask4_price", "ask4_size", "bid4_price", "bid4_size",
    "ask5_price", "ask5_size", "bid5_price", "bid5_size",
    "total_trade_volume", "turnover", "open_interest" 
]

SHORT_CYCLE_FEATURES = [
    "ksft2",
    "kup2",
    "klow2",
    "kmid2",
    "klow",
    "kup",
    "max_gap_diff",
    "max_ask_gap",
    "bid_gap_near_far_ratio",
    "sell_vwap_trend_60",
    "sell_vwap_trend_180",
    "weighted_imbalance_inv",
    "imbalance_top3",
    "depth_replenishment_diff",
    "turnover_delta_slope_60",
    "signed_trade_pressure_60",
]

MIDDLE_CYCLE_FEATURES = [
    "ksft2",
    "klow",
    "kmid2",
    "kup2",
    "log_return_wap_1_vol_60",
    "open_interest_slope_360",
    "klow2",
    "turnover_delta_vol_360",
    "bid_gap_near_far_ratio",
    "volume_trend_60",
    "ask_gap_2_3",
    "kup",
    "ofi_vol_360",
    "ask_gap_count",
    "buy_volume",
    "sell_vwap_trend_360",
    "wap_1",
    "sell_vwap",
    "ofi_vol_180",
    "volume",
]

LONG_CYCLE_FEATURES = [
    "open_interest_slope_360",
    "klow",
    "ksft2",
    "klen",
    "log_return_wap_1_vol_60",
    "ask_gap_count",
    "kup2",
    "turnover_delta_vol_360",
    "volume",
    "buy_volume",
    "kmid2",
    "volume_trend_360",
    "ask_gap_2_3",
    "sell_volume",
    "ofi_vol_360",
    "gap_count_diff",
    "sell_spread",
    "ofi_vol_180",
    "wap_1",
    "buy_vwap_trend_180",
    "sell_vwap",
]

CYCLE_FEATURES = {
    "short": SHORT_CYCLE_FEATURES,
    "middle": MIDDLE_CYCLE_FEATURES,
    "long": LONG_CYCLE_FEATURES,
}

# Backward-compatible alias retained for tests/imports that still reference it.
SINGLE_FEATURES = FIXED_FEATURES
TREND_FEATURES: list[str] = []


def _dedupe_preserve_order(columns: Iterable[str]) -> list[str]:
    """去重并保留第一次出现的顺序。"""
    seen: set[str] = set()
    ordered: list[str] = []
    for col in columns:
        if col in seen:
            continue
        seen.add(col)
        ordered.append(col)
    return ordered


def resolve_cycle_features(cycle_feature_sets: Iterable[str]) -> list[str]:
    """把 short/middle/long 组合解析为去重后的 cycle 特征列表。"""
    merged: list[str] = []
    for name in cycle_feature_sets:
        if name not in CYCLE_FEATURES:
            raise ValueError(
                f"未知 cycle feature set: {name}，可选: {sorted(CYCLE_FEATURES.keys())}"
            )
        merged.extend(CYCLE_FEATURES[name])
    return _dedupe_preserve_order(merged)


class FeaturePipeline:
    """特征加载与处理管道。

    直接从 feather 文件加载 train/val/test 数据集，
    并筛选出 `fixed_features + cycle_features` 组成状态向量。
    """

    def __init__(
        self,
        data_dir: str,
        pair: str,
        cycle_features: Iterable[str] | None = None,
    ):
        """
        Args:
            data_dir: 数据根目录路径（包含 df_train.feather, df_val.feather, df_test.feather）
            pair: 交易对名称，如 'BTC', 'ETH', 'DOT', 'BNB'
            cycle_features: 额外启用的周期因子列
        """
        self.data_dir = data_dir
        self.pair = pair
        self.cycle_features = _dedupe_preserve_order(cycle_features or [])
        self._loaded = False
        self._raw_train: pl.DataFrame | None = None
        self._raw_val: pl.DataFrame | None = None
        self._raw_test: pl.DataFrame | None = None

    def _load_data(self) -> None:
        """加载原始数据，只执行一次"""
        if self._loaded:
            return

        train_path = f"{self.data_dir}/{self.pair}/df_train.feather"
        val_path = f"{self.data_dir}/{self.pair}/df_val.feather"
        test_path = f"{self.data_dir}/{self.pair}/df_test.feather"

        self._raw_train = pl.DataFrame._from_arrow(pa_feather.read_table(train_path))
        self._raw_val = pl.DataFrame._from_arrow(pa_feather.read_table(val_path))
        self._raw_test = pl.DataFrame._from_arrow(pa_feather.read_table(test_path))

        self._loaded = True

    def get_state_vector(self) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """返回 train/val/test 三个数据集，筛选出对应的特征列。

        特征顺序: fixed_features + cycle_features

        Returns:
            (train_df, val_df, test_df) 三元组，每个都是 polars DataFrame
        """
        self._load_data()

        feature_cols = FIXED_FEATURES + self.cycle_features
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
