"""滑动窗口枚举.

设计文档锚点: §3.4 与 §4.3。

候选窗口由 ``reward_alignment`` 决定:
- ``paper_formula``     : ``num_rows - h``  个候选 (需要第 h 行作为最后一步 markout)
- ``next_row_execution``: ``num_rows - h - 1`` 个候选 (再多预留 1 行)
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import List, Literal, Optional, Sequence

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


@dataclass(frozen=True)
class WindowIndexEntry:
    """单个候选窗口的索引项。"""
    window_start: int
    window_end: int
    last_execution_row: int
    last_markout_row: int
    horizon_return: float
    realized_volatility: float
    draw_pattern: str  # "upward" | "downward" | "mixed"
    past_return: float
    past_realized_volatility: float
    past_draw_pattern: str

    def to_dict(self) -> dict:
        return asdict(self)


def _classify_draw(max_up: float, max_down: float) -> str:
    """根据最大涨/跌相对强弱给出 ``upward`` / ``downward`` / ``mixed``。"""
    if max_up <= 0 and max_down <= 0:
        return "mixed"
    if max_up > 1.5 * max_down:
        return "upward"
    if max_down > 1.5 * max_up:
        return "downward"
    return "mixed"


def _compute_window_stats(close: Sequence[float], start: int, length: int):
    """horizon 内 horizon_return / realized_vol / draw_pattern。"""
    closes = [float(close[start + i]) for i in range(length + 1)]
    if closes[0] <= 0 or any(c <= 0 for c in closes):
        return float("nan"), float("nan"), "mixed", float("-inf"), float("inf")
    horizon_return = (closes[-1] - closes[0]) / closes[0]
    # 1-bar 收益率
    rets = [(closes[i] - closes[i - 1]) / closes[i - 1] for i in range(1, length)]
    if rets:
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / max(len(rets) - 1, 1)
        realized_vol = math.sqrt(max(var, 0.0))
    else:
        realized_vol = 0.0
    # max draw up / down
    peak = closes[0]
    trough = closes[0]
    max_up = 0.0
    max_down = 0.0
    for c in closes:
        peak = max(peak, c)
        trough = min(trough, c)
        max_up = max(max_up, (c - trough) / max(trough, 1e-12))
        max_down = max(max_down, (peak - c) / max(peak, 1e-12))
    draw_pattern = _classify_draw(max_up, max_down)
    return horizon_return, realized_vol, draw_pattern, max_up, max_down


def _compute_past_stats(close: Sequence[float], start: int, lookback: int):
    """前瞻性诊断模式: 只用 ``[start-lookback, start]`` 区间。"""
    if start - lookback < 0:
        return float("nan"), float("nan"), "mixed"
    closes = [float(close[start - lookback + i]) for i in range(lookback + 1)]
    if closes[0] <= 0 or any(c <= 0 for c in closes):
        return float("nan"), float("nan"), "mixed"
    past_ret = (closes[-1] - closes[0]) / closes[0]
    rets = [(closes[i] - closes[i - 1]) / closes[i - 1] for i in range(1, lookback + 1)]
    if rets:
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / max(len(rets) - 1, 1)
        past_vol = math.sqrt(max(var, 0.0))
    else:
        past_vol = 0.0
    peak = closes[0]
    trough = closes[0]
    max_up = 0.0
    max_down = 0.0
    for c in closes:
        peak = max(peak, c)
        trough = min(trough, c)
        max_up = max(max_up, (c - trough) / max(trough, 1e-12))
        max_down = max(max_down, (peak - c) / max(peak, 1e-12))
    return past_ret, past_vol, _classify_draw(max_up, max_down)


def _rolling_all_positive(close: np.ndarray, window_size: int) -> np.ndarray:
    positive = (close > 0).astype(np.int64)
    prefix = np.concatenate(([0], np.cumsum(positive, dtype=np.int64)))
    counts = prefix[window_size:] - prefix[:-window_size]
    return counts == window_size


def _windowed_sample_volatility(
    returns: np.ndarray,
    *,
    window_size: int,
    n_windows: int,
) -> np.ndarray:
    if n_windows <= 0:
        return np.asarray([], dtype=np.float64)
    if window_size <= 1:
        return np.zeros(n_windows, dtype=np.float64)

    prefix_sum = np.concatenate(([0.0], np.cumsum(returns, dtype=np.float64)))
    prefix_sq = np.concatenate(([0.0], np.cumsum(returns * returns, dtype=np.float64)))
    sums = prefix_sum[window_size : window_size + n_windows] - prefix_sum[:n_windows]
    sums_sq = prefix_sq[window_size : window_size + n_windows] - prefix_sq[:n_windows]
    n = float(window_size)
    denom = max(window_size - 1, 1)
    var = (sums_sq - (sums * sums) / n) / denom
    return np.sqrt(np.maximum(var, 0.0))


def _classify_draw_arrays(max_up: np.ndarray, max_down: np.ndarray) -> np.ndarray:
    labels = np.full(max_up.shape, "mixed", dtype="<U8")
    labels[max_up > 1.5 * max_down] = "upward"
    labels[max_down > 1.5 * max_up] = "downward"
    flat = (max_up <= 0.0) & (max_down <= 0.0)
    labels[flat] = "mixed"
    return labels


def _rolling_draw_patterns(
    close: np.ndarray,
    *,
    window_size: int,
    n_windows: int,
    valid_windows: np.ndarray,
) -> np.ndarray:
    labels = np.full(n_windows, "mixed", dtype="<U8")
    if n_windows <= 0 or window_size <= 0 or not bool(np.any(valid_windows)):
        return labels

    windows_view = sliding_window_view(close, window_shape=window_size)[:n_windows]
    # Keep temporary arrays bounded for large prospective lookbacks.
    chunk_size = max(512, min(n_windows, max(1, 8_000_000 // window_size)))
    for chunk_start in range(0, n_windows, chunk_size):
        chunk_end = min(n_windows, chunk_start + chunk_size)
        valid = valid_windows[chunk_start:chunk_end]
        if not bool(np.any(valid)):
            continue
        windows = windows_view[chunk_start:chunk_end][valid].astype(
            np.float64, copy=False
        )
        trough = np.minimum.accumulate(windows, axis=1)
        peak = np.maximum.accumulate(windows, axis=1)
        max_up = np.max(
            (windows - trough) / np.maximum(trough, 1.0e-12),
            axis=1,
        )
        max_down = np.max(
            (peak - windows) / np.maximum(peak, 1.0e-12),
            axis=1,
        )
        labels_chunk = labels[chunk_start:chunk_end]
        labels_chunk[valid] = _classify_draw_arrays(max_up, max_down)
        labels[chunk_start:chunk_end] = labels_chunk
    return labels


def _vectorized_window_stats(
    close: np.ndarray,
    *,
    horizon: int,
    n_candidates: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_candidates <= 0:
        empty_f = np.asarray([], dtype=np.float64)
        return empty_f, empty_f, np.asarray([], dtype="<U8")

    window_size = horizon + 1
    valid = _rolling_all_positive(close, window_size)[:n_candidates]
    start_prices = close[:n_candidates]
    end_prices = close[horizon : horizon + n_candidates]
    horizon_return = np.full(n_candidates, np.nan, dtype=np.float64)
    horizon_return[valid] = (end_prices[valid] - start_prices[valid]) / start_prices[valid]

    if horizon > 1:
        returns = np.divide(
            close[1:] - close[:-1],
            close[:-1],
            out=np.zeros(len(close) - 1, dtype=np.float64),
            where=close[:-1] > 0,
        )
        realized_vol = _windowed_sample_volatility(
            returns,
            window_size=horizon - 1,
            n_windows=n_candidates,
        )
    else:
        realized_vol = np.zeros(n_candidates, dtype=np.float64)
    realized_vol = np.where(valid, realized_vol, np.nan)

    draw_pattern = _rolling_draw_patterns(
        close,
        window_size=window_size,
        n_windows=n_candidates,
        valid_windows=valid,
    )
    return horizon_return, realized_vol, draw_pattern


def _vectorized_past_stats(
    close: np.ndarray,
    *,
    lookback: int,
    n_candidates: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    past_ret = np.full(n_candidates, np.nan, dtype=np.float64)
    past_vol = np.full(n_candidates, np.nan, dtype=np.float64)
    past_draw = np.full(n_candidates, "mixed", dtype="<U8")
    if n_candidates <= 0:
        return past_ret, past_vol, past_draw
    if lookback < 0:
        raise ValueError(f"prospective_lookback_minutes 必须 >= 0, got {lookback}")

    window_size = lookback + 1
    n_past_windows = max(0, n_candidates - lookback)
    if n_past_windows <= 0:
        return past_ret, past_vol, past_draw

    valid = _rolling_all_positive(close, window_size)[:n_past_windows]
    starts = np.arange(lookback, n_candidates, dtype=np.int64)
    start_prices = close[:n_past_windows]
    end_prices = close[lookback:n_candidates]
    values = np.full(n_past_windows, np.nan, dtype=np.float64)
    values[valid] = (end_prices[valid] - start_prices[valid]) / start_prices[valid]
    past_ret[starts] = values

    if lookback > 0:
        returns = np.divide(
            close[1:] - close[:-1],
            close[:-1],
            out=np.zeros(len(close) - 1, dtype=np.float64),
            where=close[:-1] > 0,
        )
        vol_values = _windowed_sample_volatility(
            returns,
            window_size=lookback,
            n_windows=n_past_windows,
        )
    else:
        vol_values = np.zeros(n_past_windows, dtype=np.float64)
    past_vol[starts] = np.where(valid, vol_values, np.nan)

    draw_values = _rolling_draw_patterns(
        close,
        window_size=window_size,
        n_windows=n_past_windows,
        valid_windows=valid,
    )
    past_draw[starts] = draw_values
    return past_ret, past_vol, past_draw


class SlidingWindowIndexer:
    """枚举所有候选 horizon。

    使用方式::

        indexer = SlidingWindowIndexer(horizon=72, reward_alignment="paper_formula")
        entries = indexer.enumerate(train_frame, stratification_mode="hindsight_horizon")
    """

    def __init__(
        self,
        horizon: int,
        reward_alignment: Literal["paper_formula", "next_row_execution"],
        prospective_lookback_minutes: int = 1440,
    ) -> None:
        if horizon <= 0:
            raise ValueError(f"horizon 必须 > 0, got {horizon}")
        if reward_alignment not in ("paper_formula", "next_row_execution"):
            raise ValueError(f"非法 reward_alignment: {reward_alignment}")
        self.horizon = horizon
        self.reward_alignment = reward_alignment
        self.prospective_lookback_minutes = prospective_lookback_minutes

    def num_candidates(self, num_rows: int) -> int:
        """根据 reward_alignment 计算候选窗口总数。

        - ``paper_formula``: ``num_rows - h``（需要保留第 ``h`` 行作为最后一步 markout）。
        - ``next_row_execution``: ``num_rows - h - 1``（再多预留 1 行）。
        """
        if self.reward_alignment == "paper_formula":
            return max(0, num_rows - self.horizon)
        return max(0, num_rows - self.horizon - 1)

    def enumerate(
        self,
        frame,
        stratification_mode: Literal["hindsight_horizon", "prospective_past"],
    ) -> List[WindowIndexEntry]:
        """生成 ``WindowIndexEntry`` 列表。

        实现要点
        --------
        - 仅按 stride=1 枚举（设计中 ``window_stride`` 默认 1）。
        - 计算分层统计:
          * ``hindsight_horizon`` 用 horizon 内部统计（``horizon_return``、
            ``realized_volatility``、``draw_pattern``），仅供离线 demonstration
            curation；不可解读为预测能力。
          * ``prospective_past`` 只用 ``[t - lookback, t)`` 区间统计；
            lookback 不足时 ``past_*`` 字段写 NaN，采样阶段会丢弃这些桶。
        - 任一模式都不允许在 ``hindsight`` 路径下读 ``t`` 之后的数据，
          也不允许在 ``prospective_past`` 路径下读 ``t`` 之后的数据。

        Raises
        ------
        ValueError : ``stratification_mode`` 非法 / frame 缺 ``close`` 列。
        TypeError : frame 非 polars.DataFrame。
        """
        import polars as pl

        if stratification_mode not in ("hindsight_horizon", "prospective_past"):
            raise ValueError(f"非法 stratification_mode: {stratification_mode}")
        if not isinstance(frame, pl.DataFrame):
            raise TypeError("frame 必须是 polars.DataFrame")
        if "close" not in frame.columns:
            raise ValueError("frame 必须包含 close 列")

        close = frame["close"].to_numpy().astype(np.float64, copy=False)
        num_rows = len(close)
        n_candidates = self.num_candidates(num_rows)
        h = self.horizon
        # 超 lookback 时 past 统计为 NaN，但仍保留窗口（采样阶段会丢弃 NaN strata）。
        lookback = self.prospective_lookback_minutes
        horizon_returns, realized_vols, draw_patterns = _vectorized_window_stats(
            close,
            horizon=h,
            n_candidates=n_candidates,
        )
        if stratification_mode == "prospective_past":
            past_returns, past_vols, past_draw_patterns = _vectorized_past_stats(
                close,
                lookback=lookback,
                n_candidates=n_candidates,
            )
        else:
            past_returns = np.full(n_candidates, np.nan, dtype=np.float64)
            past_vols = np.full(n_candidates, np.nan, dtype=np.float64)
            past_draw_patterns = np.full(n_candidates, "mixed", dtype="<U8")

        entries: List[WindowIndexEntry] = []
        for start in range(n_candidates):
            window_end = start + h - 1
            if self.reward_alignment == "paper_formula":
                last_exec = window_end
                last_markout = window_end + 1
            else:
                last_exec = window_end + 1
                last_markout = window_end + 2

            entries.append(
                WindowIndexEntry(
                    window_start=start,
                    window_end=window_end,
                    last_execution_row=last_exec,
                    last_markout_row=last_markout,
                    horizon_return=float(horizon_returns[start]),
                    realized_volatility=float(realized_vols[start]),
                    draw_pattern=str(draw_patterns[start]),
                    past_return=float(past_returns[start]),
                    past_realized_volatility=float(past_vols[start]),
                    past_draw_pattern=str(past_draw_patterns[start]),
                )
            )
        return entries

    def to_frame(self, entries: List[WindowIndexEntry]):
        """把索引项转为 polars DataFrame，便于写 ``window_index_*.feather``。"""
        import polars as pl

        return pl.DataFrame(
            {
                "window_start": [e.window_start for e in entries],
                "window_end": [e.window_end for e in entries],
                "last_execution_row": [e.last_execution_row for e in entries],
                "last_markout_row": [e.last_markout_row for e in entries],
                "horizon_return": [e.horizon_return for e in entries],
                "realized_volatility": [e.realized_volatility for e in entries],
                "draw_pattern": [e.draw_pattern for e in entries],
                "past_return": [e.past_return for e in entries],
                "past_realized_volatility": [
                    e.past_realized_volatility for e in entries
                ],
                "past_draw_pattern": [e.past_draw_pattern for e in entries],
            }
        )
