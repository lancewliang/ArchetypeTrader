"""特征加载与处理管道。

负责加载 train/val/test 三个 feather 文件的数据，并按
`fixed_features + cycle_features` 组装状态向量。
"""

from pathlib import Path
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

FACTORS_ROOT = Path(__file__).resolve().parents[1] / "factors"


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


def _resolve_pair_factor_dir(pair: str, factors_root: Path | None = None) -> Path:
    """解析并返回某个品种对应的因子目录（大小写不敏感）。"""
    root = factors_root or FACTORS_ROOT
    if not root.exists():
        raise ValueError(f"因子目录不存在: {root}")

    normalized_pair = pair.strip()
    if not normalized_pair:
        raise ValueError("pair 不能为空")

    for candidate in (normalized_pair, normalized_pair.upper(), normalized_pair.lower()):
        candidate_path = root / candidate
        if candidate_path.is_dir():
            return candidate_path

    lower_name_to_path = {
        child.name.lower(): child
        for child in root.iterdir()
        if child.is_dir()
    }
    matched = lower_name_to_path.get(normalized_pair.lower())
    if matched is not None:
        return matched

    available_pairs = sorted(path.name for path in lower_name_to_path.values())
    raise ValueError(
        f"未找到品种 {pair!r} 的因子目录，当前可用品种: {available_pairs}"
    )


def list_cycle_feature_sets(pair: str, factors_root: Path | None = None) -> list[str]:
    """列出某个品种可用的 cycle 集合名（来自 *.txt 文件名）。"""
    pair_dir = _resolve_pair_factor_dir(pair, factors_root=factors_root)
    return sorted(
        path.stem.strip().lower()
        for path in pair_dir.glob("*.txt")
        if path.is_file()
    )


def load_cycle_features_from_file(
    pair: str,
    cycle_feature_set: str,
    factors_root: Path | None = None,
) -> list[str]:
    """从 src/factors/<pair>/<cycle>.txt 读取特征列表。"""
    pair_dir = _resolve_pair_factor_dir(pair, factors_root=factors_root)
    normalized_cycle = cycle_feature_set.strip().lower()
    if not normalized_cycle:
        raise ValueError("cycle feature set 名称不能为空")

    feature_path = pair_dir / f"{normalized_cycle}.txt"
    if not feature_path.exists():
        available_sets = list_cycle_feature_sets(pair, factors_root=factors_root)
        raise ValueError(
            f"未找到 cycle feature set: {normalized_cycle!r} (pair={pair})，"
            f"可选: {available_sets}"
        )

    raw_lines = feature_path.read_text(encoding="utf-8").splitlines()
    features: list[str] = []
    for raw_line in raw_lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        feature_name = line.split("#", 1)[0].strip()
        if feature_name:
            features.append(feature_name)

    if not features:
        raise ValueError(f"因子文件为空: {feature_path}")

    return _dedupe_preserve_order(features)


def resolve_cycle_features(cycle_feature_sets: Iterable[str], pair: str) -> list[str]:
    """把 short/middle/long 组合解析为去重后的 cycle 特征列表。"""
    merged: list[str] = []
    for name in cycle_feature_sets:
        normalized_name = name.strip().lower()
        if not normalized_name:
            continue
        merged.extend(load_cycle_features_from_file(pair, normalized_name))
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
