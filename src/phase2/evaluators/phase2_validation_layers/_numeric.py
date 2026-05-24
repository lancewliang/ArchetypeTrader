"""Small numeric helpers for Phase II validation layer calculators."""

from __future__ import annotations

import math

import numpy as np


def nan_value() -> float:
    """Return a quiet NaN."""

    return float("nan")


def as_float_array(values: tuple[float, ...] | list[float] | np.ndarray) -> np.ndarray:
    """Convert a sequence to a 1D float64 array."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        return array.reshape(-1)
    return array


def finite_values(values: np.ndarray) -> np.ndarray:
    """Return finite values only."""

    return values[np.isfinite(values)]


def safe_mean(values: np.ndarray) -> float:
    """Mean over finite values, NaN for empty input."""

    finite = finite_values(values)
    if finite.size == 0:
        return nan_value()
    return float(np.mean(finite))


def safe_median(values: np.ndarray) -> float:
    """Median over finite values, NaN for empty input."""

    finite = finite_values(values)
    if finite.size == 0:
        return nan_value()
    return float(np.median(finite))


def safe_sum(values: np.ndarray) -> float:
    """Sum over finite values, NaN for empty input."""

    finite = finite_values(values)
    if finite.size == 0:
        return nan_value()
    return float(np.sum(finite))


def safe_percentile(values: np.ndarray, percentile: float) -> float:
    """Percentile over finite values, NaN for empty input."""

    finite = finite_values(values)
    if finite.size == 0:
        return nan_value()
    return float(np.percentile(finite, percentile))


def safe_std(values: np.ndarray) -> float:
    """Population std over finite values, NaN for empty input."""

    finite = finite_values(values)
    if finite.size == 0:
        return nan_value()
    return float(np.std(finite))


def safe_ratio(numerator: float, denominator: float) -> float:
    """Return numerator / denominator with NaN for invalid denominator."""

    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return nan_value()
    if abs(denominator) <= 1e-12:
        return nan_value()
    return float(numerator / denominator)
