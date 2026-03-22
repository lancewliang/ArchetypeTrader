"""特征加载与处理管道

负责加载、验证和组织市场特征数据，支持按交易对和时间范围划分。
"""

import os
from datetime import datetime
from typing import List, Tuple

import numpy as np

from src.config import Config


# 交易对索引映射（用于多维数据中按 pair 选择）
PAIR_INDEX = {"BTC": 0, "ETH": 1, "DOT": 2, "BNB": 3}


class FeaturePipeline:
    """特征加载与处理管道

    加载 single_features (36维) 和 trend_features (9维)，
    拼接为 45 维状态向量，并支持按时间和 horizon 切分。
    """

    def __init__(self, data_dir: str, pair: str, config: Config | None = None):
        """
        Args:
            data_dir: 数据根目录路径（包含 single_features.npy 和 trend_features.npy）
            pair: 交易对名称，如 'BTC', 'ETH', 'DOT', 'BNB'
            config: 可选的全局配置，用于获取默认参数
        """
        self.data_dir = data_dir
        if pair not in PAIR_INDEX:
            raise ValueError(
                f"不支持的交易对 '{pair}'，支持的交易对: {list(PAIR_INDEX.keys())}"
            )
        self.pair = pair
        self.config = config or Config()

        self._single_features: np.ndarray | None = None
        self._trend_features: np.ndarray | None = None

    def load_single_features(self) -> np.ndarray:
        """加载 36 维单步特征，返回 shape (T, 36)。

        支持多种输入 shape：
        - (36,): 单个时间步 → reshape 为 (1, 36)
        - (T, 36): 单交易对时间序列
        - (num_pairs, T, 36): 多交易对，按 pair 索引选择

        Raises:
            FileNotFoundError: 特征文件不存在
            ValueError: 特征最后一维不为 36
        """
        path = os.path.join(self.data_dir, "single_features.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"特征文件不存在: {path}")

        raw = np.load(path)
        data = self._extract_pair_data(raw, "single_features")

        expected_dim = self.config.single_feature_dim
        if data.shape[-1] != expected_dim:
            raise ValueError(
                f"single_features 最后一维为 {data.shape[-1]}，预期为 {expected_dim}"
            )

        self._single_features = data
        return data

    def load_trend_features(self) -> np.ndarray:
        """加载 9 维趋势特征，返回 shape (T, 9)。

        支持多种输入 shape：
        - (9,): 单个时间步 → reshape 为 (1, 9)
        - (T, 9): 单交易对时间序列
        - (num_pairs, T, 9): 多交易对，按 pair 索引选择

        Raises:
            FileNotFoundError: 特征文件不存在
            ValueError: 特征最后一维不为 9
        """
        path = os.path.join(self.data_dir, "trend_features.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"特征文件不存在: {path}")

        raw = np.load(path)
        data = self._extract_pair_data(raw, "trend_features")

        expected_dim = self.config.trend_feature_dim
        if data.shape[-1] != expected_dim:
            raise ValueError(
                f"trend_features 最后一维为 {data.shape[-1]}，预期为 {expected_dim}"
            )

        self._trend_features = data
        return data

    def get_state_vector(self) -> np.ndarray:
        """拼接单步和趋势特征，返回 shape (T, 45)。

        如果尚未加载特征，会自动调用 load 方法。
        两个特征的时间维度 T 必须一致。

        Raises:
            ValueError: 两个特征的时间维度不一致
        """
        if self._single_features is None:
            self.load_single_features()
        if self._trend_features is None:
            self.load_trend_features()

        single = self._single_features
        trend = self._trend_features

        if single.shape[0] != trend.shape[0]:
            raise ValueError(
                f"single_features 时间维度 ({single.shape[0]}) "
                f"与 trend_features 时间维度 ({trend.shape[0]}) 不一致"
            )

        return np.concatenate([single, trend], axis=-1)

    def split_by_date(
        self,
        data: np.ndarray | None = None,
        train_ratio: float | None = None,
        val_ratio: float | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """按时间范围划分训练/验证/测试集。

        由于 .npy 文件不含时间戳，使用基于索引的划分。
        根据 Config 中的日期范围计算各集合占总时间的比例，
        然后按比例划分数据索引。

        Args:
            data: 待划分的数据，shape (T, ...)。若为 None 则使用 get_state_vector()。
            train_ratio: 可选，直接指定训练集比例（覆盖日期计算）
            val_ratio: 可选，直接指定验证集比例（覆盖日期计算）

        Returns:
            (train_data, val_data, test_data) 三元组
        """
        if data is None:
            data = self.get_state_vector()

        T = data.shape[0]

        if train_ratio is not None and val_ratio is not None:
            # 直接使用指定比例
            train_end_idx = int(T * train_ratio)
            val_end_idx = int(T * (train_ratio + val_ratio))
        else:
            # 根据 Config 日期范围计算比例
            train_end_idx, val_end_idx = self._compute_split_indices(T)

        train_data = data[:train_end_idx]
        val_data = data[train_end_idx:val_end_idx]
        test_data = data[val_end_idx:]

        return train_data, val_data, test_data

    def split_into_horizons(
        self, data: np.ndarray, h: int = 72
    ) -> List[np.ndarray]:
        """按 horizon 长度切分数据。

        Args:
            data: 待切分的数据，shape (T, ...)
            h: horizon 长度，默认 72

        Returns:
            切分后的片段列表，每个片段 shape (h, ...) 或最后一个可能更短
        """
        if h <= 0:
            raise ValueError(f"horizon 长度必须为正整数，收到 h={h}")

        T = data.shape[0]
        horizons = []
        for start in range(0, T, h):
            end = min(start + h, T)
            horizons.append(data[start:end])
        return horizons

    # ---- 内部方法 ----

    def _extract_pair_data(self, raw: np.ndarray, name: str) -> np.ndarray:
        """从原始数据中提取指定交易对的数据。

        处理多种 shape：
        - 1D (feature_dim,): 单时间步 → (1, feature_dim)
        - 2D (T, feature_dim): 单交易对 → 直接返回
        - 3D (num_pairs, T, feature_dim): 多交易对 → 按索引选择
        """
        if raw.ndim == 1:
            # 单个时间步，reshape 为 (1, feature_dim)
            return raw.reshape(1, -1)
        elif raw.ndim == 2:
            # (T, feature_dim) — 单交易对或共享数据
            return raw
        elif raw.ndim == 3:
            # (num_pairs, T, feature_dim) — 按交易对索引选择
            pair_idx = PAIR_INDEX[self.pair]
            if pair_idx >= raw.shape[0]:
                raise ValueError(
                    f"{name} 只有 {raw.shape[0]} 个交易对的数据，"
                    f"无法选择索引 {pair_idx} ({self.pair})"
                )
            return raw[pair_idx]
        else:
            raise ValueError(
                f"{name} 的维度为 {raw.ndim}，预期为 1、2 或 3 维"
            )

    def _compute_split_indices(self, T: int) -> Tuple[int, int]:
        """根据 Config 日期范围计算划分索引。

        假设数据按时间均匀分布，根据各阶段占总时间跨度的比例计算索引。

        Returns:
            (train_end_idx, val_end_idx)
        """
        fmt = "%Y-%m-%d"
        total_start = datetime.strptime(self.config.train_start, fmt)
        total_end = datetime.strptime(self.config.test_end, fmt)
        train_end = datetime.strptime(self.config.train_end, fmt)
        val_end = datetime.strptime(self.config.val_end, fmt)

        total_days = (total_end - total_start).days
        if total_days <= 0:
            raise ValueError("配置的日期范围无效：总时间跨度 ≤ 0")

        train_days = (train_end - total_start).days
        val_days = (val_end - total_start).days

        train_end_idx = int(T * train_days / total_days)
        val_end_idx = int(T * val_days / total_days)

        return train_end_idx, val_end_idx
