"""状态特征归一化器。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import polars as pl


_EPS = 1e-12


@dataclass(frozen=True)
class StateNormalizer:
    """对 horizon state 特征做逐列标准化。"""

    feature_columns: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(
        cls,
        dataframe: pl.DataFrame,
        state_columns: Sequence[str],
    ) -> StateNormalizer:
        """基于训练集拟合均值和标准差。"""

        columns = tuple(state_columns)
        if not columns:
            raise ValueError("state_columns must not be empty")

        values = dataframe.select(list(columns)).to_numpy().astype(np.float64, copy=False)
        if values.ndim != 2:
            raise ValueError("state features must be a 2D matrix")
        if values.shape[0] == 0:
            raise ValueError("state features must contain at least one row")

        mean = np.mean(values, axis=0)
        std = np.std(values, axis=0)
        std[std <= _EPS] = 1.0
        return cls(
            feature_columns=columns,
            mean=mean.astype(np.float32, copy=False),
            std=std.astype(np.float32, copy=False),
        )

    def transform(self, values: np.ndarray) -> np.ndarray:
        """对二维 state 特征矩阵做标准化。"""

        array = np.asarray(values, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError("state values must have shape [n, feature_dim]")
        if array.shape[1] != self.mean.shape[0]:
            raise ValueError(
                "state values feature dim does not match normalizer, "
                f"got {array.shape[1]} and {self.mean.shape[0]}"
            )
        return ((array - self.mean) / self.std).astype(np.float32, copy=False)

    def to_dict(self) -> dict[str, object]:
        """转成 JSON 友好结构。"""

        return {
            "feature_columns": list(self.feature_columns),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> StateNormalizer:
        """从 JSON payload 恢复 normalizer。"""

        feature_columns = payload.get("feature_columns")
        mean = payload.get("mean")
        std = payload.get("std")
        if not isinstance(feature_columns, list):
            raise ValueError("normalizer feature_columns must be a list")
        if mean is None or std is None:
            raise ValueError("normalizer payload must contain mean and std")

        feature_names = tuple(str(column) for column in feature_columns)
        mean_values = np.asarray(mean, dtype=np.float32)
        std_values = np.asarray(std, dtype=np.float32)
        if mean_values.ndim != 1 or std_values.ndim != 1:
            raise ValueError("normalizer mean/std must be 1D arrays")
        if mean_values.shape != std_values.shape:
            raise ValueError("normalizer mean/std must have the same shape")
        return cls(
            feature_columns=feature_names,
            mean=mean_values,
            std=std_values,
        )


__all__ = ["StateNormalizer"]
