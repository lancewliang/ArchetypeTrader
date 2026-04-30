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
    closes = [float(close[start + i]) for i in range(length)]
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
    """前瞻性诊断模式: 只用 ``[start-lookback, start)`` 区间。"""
    if start - lookback < 0:
        return float("nan"), float("nan"), "mixed"
    closes = [float(close[start - lookback + i]) for i in range(lookback)]
    if closes[0] <= 0:
        return float("nan"), float("nan"), "mixed"
    past_ret = (closes[-1] - closes[0]) / closes[0]
    rets = [(closes[i] - closes[i - 1]) / closes[i - 1] for i in range(1, lookback)]
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

        close = frame["close"].to_numpy()
        num_rows = len(close)
        n_candidates = self.num_candidates(num_rows)
        h = self.horizon
        # 超 lookback 时 past 统计为 NaN，但仍保留窗口（采样阶段会丢弃 NaN strata）。
        lookback = self.prospective_lookback_minutes

        entries: List[WindowIndexEntry] = []
        for start in range(n_candidates):
            window_end = start + h - 1
            if self.reward_alignment == "paper_formula":
                last_exec = window_end
                last_markout = window_end + 1
            else:
                last_exec = window_end + 1
                last_markout = window_end + 2

            horizon_return, realized_vol, draw_pattern, _, _ = _compute_window_stats(
                close, start, h
            )
            if stratification_mode == "prospective_past":
                past_ret, past_vol, past_draw = _compute_past_stats(close, start, lookback)
            else:
                past_ret = float("nan")
                past_vol = float("nan")
                past_draw = "mixed"

            entries.append(
                WindowIndexEntry(
                    window_start=start,
                    window_end=window_end,
                    last_execution_row=last_exec,
                    last_markout_row=last_markout,
                    horizon_return=horizon_return,
                    realized_volatility=realized_vol,
                    draw_pattern=draw_pattern,
                    past_return=past_ret,
                    past_realized_volatility=past_vol,
                    past_draw_pattern=past_draw,
                )
            )
        return entries

    def to_frame(self, entries: List[WindowIndexEntry]):
        """把索引项转为 polars DataFrame，便于写 ``window_index_*.feather``。"""
        import polars as pl

        return pl.DataFrame([e.to_dict() for e in entries])
