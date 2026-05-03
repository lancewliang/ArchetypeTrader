"""Train-only state feature normalization for Phase I/II.

The normalizer is fitted on Phase I train horizons and persisted as
``state_normalizer.json`` so Phase II can apply the exact same transform to raw
market frames. Prices, execution books, and rewards stay raw; only model state
features are transformed.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List, Mapping, Sequence

import numpy as np


_EPS = 1.0e-6


@dataclass(frozen=True)
class StateNormalizerStats:
    method: str
    feature_columns: List[str]
    transform_kinds: List[str]
    center: List[float]
    scale: List[float]
    clip_value: float
    scale_floor: float
    max_abs_before: float
    max_abs_after_fit: float
    fallback_to_standard_count: int

    def to_dict(self) -> dict:
        return asdict(self)


def _is_log_feature(name: str) -> bool:
    lower = name.lower()
    if "ratio" in lower or "zscore" in lower or lower.startswith("log_return"):
        return False
    if lower in {"turnover", "total_trade_volume", "open_interest"}:
        return True
    return lower.endswith("_size") or lower.endswith("_volume")


def _signed_log1p(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.log1p(np.abs(values))


def _median(values: np.ndarray, axis: int = 0) -> np.ndarray:
    return np.nanmedian(values, axis=axis)


def _mad(values: np.ndarray, center: np.ndarray) -> np.ndarray:
    return np.nanmedian(np.abs(values - center), axis=0)


class StateNormalizer:
    """Robust feature normalizer shared by Phase I and Phase II.

    The default policy is intentionally conservative:
    - heavy magnitude count/notional fields get signed ``log1p`` first;
    - every feature then gets train-only robust z-score;
    - near-constant features fall back to standard deviation, then scale=1.
    """

    def __init__(self, stats: StateNormalizerStats) -> None:
        self.stats = stats

    @classmethod
    def fit_records(
        cls,
        records: Sequence[Any],
        *,
        feature_columns: Sequence[str],
        clip_value: float = 8.0,
        scale_floor: float = _EPS,
    ) -> "StateNormalizer":
        matrix = cls._records_to_matrix(records, expected_dim=len(feature_columns))
        return cls.fit_matrix(
            matrix,
            feature_columns=feature_columns,
            clip_value=clip_value,
            scale_floor=scale_floor,
        )

    @classmethod
    def fit_matrix(
        cls,
        matrix: np.ndarray,
        *,
        feature_columns: Sequence[str],
        clip_value: float = 8.0,
        scale_floor: float = _EPS,
    ) -> "StateNormalizer":
        arr = np.asarray(matrix, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"state normalizer fit expects [N, F], got shape={arr.shape}")
        if arr.shape[1] != len(feature_columns):
            raise ValueError(
                f"feature dimension mismatch: matrix={arr.shape[1]} columns={len(feature_columns)}"
            )
        if arr.shape[0] == 0:
            raise ValueError("state normalizer requires at least one train row")
        if not np.isfinite(arr).all():
            raise ValueError("state normalizer fit received non-finite values")

        transform_kinds = [
            "signed_log1p" if _is_log_feature(col) else "identity"
            for col in feature_columns
        ]
        prepared = cls._prepare(arr, transform_kinds)
        center = _median(prepared, axis=0)
        robust_scale = 1.4826 * _mad(prepared, center)

        std = prepared.std(axis=0, ddof=1) if prepared.shape[0] > 1 else np.zeros(arr.shape[1])
        use_std = robust_scale < scale_floor
        scale = np.where(use_std & (std >= scale_floor), std, robust_scale)
        scale = np.where(scale < scale_floor, 1.0, scale)
        normalized = np.clip((prepared - center) / scale, -clip_value, clip_value)

        stats = StateNormalizerStats(
            method="train_state_robust_v1",
            feature_columns=list(feature_columns),
            transform_kinds=transform_kinds,
            center=[float(v) for v in center.tolist()],
            scale=[float(v) for v in scale.tolist()],
            clip_value=float(clip_value),
            scale_floor=float(scale_floor),
            max_abs_before=float(np.max(np.abs(arr))) if arr.size else 0.0,
            max_abs_after_fit=float(np.max(np.abs(normalized))) if normalized.size else 0.0,
            fallback_to_standard_count=int(np.count_nonzero(use_std)),
        )
        return cls(stats)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StateNormalizer":
        return cls(
            StateNormalizerStats(
                method=str(payload.get("method", "train_state_robust_v1")),
                feature_columns=list(payload.get("feature_columns", [])),
                transform_kinds=list(payload.get("transform_kinds", [])),
                center=[float(v) for v in payload.get("center", [])],
                scale=[float(v) for v in payload.get("scale", [])],
                clip_value=float(payload.get("clip_value", 8.0)),
                scale_floor=float(payload.get("scale_floor", _EPS)),
                max_abs_before=float(payload.get("max_abs_before", 0.0)),
                max_abs_after_fit=float(payload.get("max_abs_after_fit", 0.0)),
                fallback_to_standard_count=int(payload.get("fallback_to_standard_count", 0)),
            )
        )

    @classmethod
    def load_json(cls, path: Path | str) -> "StateNormalizer":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> dict:
        return self.stats.to_dict()

    def transform_array(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64)
        original_shape = arr.shape
        if arr.ndim == 1:
            flat = arr.reshape(1, -1)
        elif arr.ndim == 2:
            flat = arr
        elif arr.ndim == 3:
            flat = arr.reshape(-1, arr.shape[-1])
        else:
            raise ValueError(f"state normalizer transform expects rank 1/2/3, got {arr.ndim}")
        self._validate_dim(flat.shape[1])
        if not np.isfinite(flat).all():
            raise ValueError("state normalizer transform received non-finite values")

        prepared = self._prepare(flat, self.stats.transform_kinds)
        center = np.asarray(self.stats.center, dtype=np.float64)
        scale = np.asarray(self.stats.scale, dtype=np.float64)
        out = np.clip(
            (prepared - center) / np.maximum(scale, self.stats.scale_floor),
            -self.stats.clip_value,
            self.stats.clip_value,
        ).astype("float32")
        return out.reshape(original_shape)

    def transform_records(self, records: Sequence[Any]) -> dict:
        max_before = 0.0
        max_after = 0.0
        for rec in records:
            arr = np.asarray(rec.states, dtype=np.float64)
            if arr.size:
                max_before = max(max_before, float(np.max(np.abs(arr))))
            out = self.transform_array(arr)
            if out.size:
                max_after = max(max_after, float(np.max(np.abs(out))))
            rec.states = out.tolist()
        return {
            "count": len(records),
            "max_abs_before": max_before,
            "max_abs_after": max_after,
        }

    def feature_count(self) -> int:
        return len(self.stats.feature_columns)

    def _validate_dim(self, dim: int) -> None:
        expected = self.feature_count()
        if dim != expected:
            raise ValueError(f"state feature dimension mismatch: got={dim} expected={expected}")

    @staticmethod
    def _prepare(matrix: np.ndarray, transform_kinds: Sequence[str]) -> np.ndarray:
        out = np.asarray(matrix, dtype=np.float64).copy()
        if out.shape[1] != len(transform_kinds):
            raise ValueError(
                f"transform kind count mismatch: matrix={out.shape[1]} kinds={len(transform_kinds)}"
            )
        for idx, kind in enumerate(transform_kinds):
            if kind == "signed_log1p":
                out[:, idx] = _signed_log1p(out[:, idx])
            elif kind == "identity":
                continue
            else:
                raise ValueError(f"unknown state transform kind: {kind}")
        return out

    @staticmethod
    def _records_to_matrix(records: Sequence[Any], *, expected_dim: int) -> np.ndarray:
        chunks: List[np.ndarray] = []
        for rec in records:
            arr = np.asarray(rec.states, dtype=np.float64)
            if arr.ndim != 2:
                raise ValueError(f"record states must be [h, F], got shape={arr.shape}")
            if arr.shape[1] != expected_dim:
                raise ValueError(
                    f"record state dim mismatch: got={arr.shape[1]} expected={expected_dim}"
                )
            chunks.append(arr)
        if not chunks:
            return np.empty((0, expected_dim), dtype=np.float64)
        return np.concatenate(chunks, axis=0)
