"""根据采样后的 window 索引切出 horizon 张量样本.

设计文档锚点: §3.3 与 §4.3。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from src.trading.cost_model import ExecutionBook
from src.trading.reward_alignment import RewardAlignment

from .schema import KNOWN_ORDERBOOK_COLUMNS, InputSchema
from .stratified_sampler import SampledHorizon


@dataclass
class HorizonRecord:
    """单个 horizon 记录。"""
    sample_id: str
    start_index: int
    end_index: int
    pair: str
    split: str
    strata_label: str
    states: list                       # [h, feature_dim]
    prices: list                       # [h+1] 或 [h+2]
    execution_books: List[ExecutionBook]  # 长度 = h
    last_execution_row: Optional[int] = None
    last_markout_row: Optional[int] = None
    actions: Optional[list] = None     # [h]，DP 后填充
    rewards: Optional[list] = None     # [h]，DP 后填充
    is_augmented: bool = False
    augmentation_type: str = "none"


# 五档键名常量（与 schema 模块一致；这里复用避免散落）。
_ASK_PRICE_COLS = [f"ask{i}_price" for i in range(1, 6)]
_ASK_SIZE_COLS = [f"ask{i}_size" for i in range(1, 6)]
_BID_PRICE_COLS = [f"bid{i}_price" for i in range(1, 6)]
_BID_SIZE_COLS = [f"bid{i}_size" for i in range(1, 6)]


class HorizonBuilder:
    """切片器。

    实现要点
    --------
    - states 从 ``schema.feature_columns`` 切出；不包含 ``close``。
    - prices 长度由 ``RewardAlignment.required_lookahead_rows`` 决定:
      paper_formula → ``h+1``；next_row_execution → ``h+2``。
    - execution_books 行号由 ``reward_alignment`` 决定（paper: row=t；
      next_row: row=t+1），切片时直接对齐到对应数据行，使下游 env 不需要
      二次偏移。
    - 不进行任何标准化（外部数据视为已对齐）。
    - 缺失盘口列时填 0；env / cost_model 会按 ``reject_transition`` 处理。
    """

    def __init__(self, horizon: int, schema: InputSchema, reward_alignment: str) -> None:
        if horizon <= 0:
            raise ValueError(f"horizon 必须 > 0, got {horizon}")
        self.horizon = horizon
        self.schema = schema
        self.alignment = RewardAlignment(reward_alignment)

    def build(
        self,
        frame,
        sampled_horizons: List[SampledHorizon],
        pair: str,
        split: str,
    ) -> List[HorizonRecord]:
        """切出 horizon。

        实现注意
        --------
        - 必须避免在 Python 层逐元素读取；先用 polars 一次性 ``select(...).to_numpy``
          再做 slice，性能与 deterministic 兼顾。
        - prices 必须从 ``close`` 列切出，绝不从 ``feature_columns`` 中复制
          （否则会出现 close 通过 features 间接进入模型）。
        - mark price 默认用 ``(ask1 + bid1) / 2``；ask1/bid1 缺失（=0）时回退到 close。

        Raises
        ------
        TypeError : frame 非 polars.DataFrame。
        """
        import numpy as np
        import polars as pl

        if not isinstance(frame, pl.DataFrame):
            raise TypeError("frame 必须是 polars.DataFrame")
        # 预提取列到 numpy，避免 Python 层逐元素读取。
        feature_cols = self.schema.feature_columns
        feature_matrix = frame.select(feature_cols).to_numpy().astype("float32")
        close = frame[self.schema.price_column].to_numpy().astype("float32")
        # 盘口五档；缺失列填 0（视为深度为 0，cost_model 会 reject）。
        def _safe_col(col: str):
            if col in frame.columns:
                return frame[col].to_numpy().astype("float32")
            return np.zeros(frame.height, dtype="float32")

        ask_p = [_safe_col(c) for c in _ASK_PRICE_COLS]
        ask_s = [_safe_col(c) for c in _ASK_SIZE_COLS]
        bid_p = [_safe_col(c) for c in _BID_PRICE_COLS]
        bid_s = [_safe_col(c) for c in _BID_SIZE_COLS]
        # mark price: (ask1 + bid1) / 2，若缺失则回退到 close。
        if "ask1_price" in frame.columns and "bid1_price" in frame.columns:
            mark = (ask_p[0] + bid_p[0]) / 2.0
            # 当某一行 ask1 或 bid1 为 0（缺失）时回退到 close
            invalid = (ask_p[0] <= 0) | (bid_p[0] <= 0)
            mark = np.where(invalid, close, mark)
        else:
            mark = close

        h = self.horizon
        lookahead = self.alignment.required_lookahead_rows()
        records: List[HorizonRecord] = []
        for sh in sampled_horizons:
            start = sh.window_start
            end_idx = start + h - 1  # inclusive
            # states: [h, feature_dim]
            states_slice = feature_matrix[start : start + h, :]
            # prices: 长度 = h + lookahead
            prices_slice = mark[start : start + h + lookahead]
            # execution_books: 每步 t 取 alignment.execution_row(t) = t (paper) 或 t+1 (next_row)
            books: List[ExecutionBook] = []
            for t in range(h):
                rows = self.alignment.rows(t)
                row = start + rows.execution_row
                book = ExecutionBook(
                    ask_prices=tuple(float(ask_p[k][row]) for k in range(5)),
                    ask_sizes=tuple(float(ask_s[k][row]) for k in range(5)),
                    bid_prices=tuple(float(bid_p[k][row]) for k in range(5)),
                    bid_sizes=tuple(float(bid_s[k][row]) for k in range(5)),
                    mark_price=float(mark[row]),
                )
                books.append(book)
            records.append(
                HorizonRecord(
                    sample_id=sh.sample_id,
                    start_index=start,
                    end_index=end_idx,
                    pair=pair,
                    split=split,
                    strata_label=sh.strata_label,
                    states=states_slice.tolist(),
                    prices=prices_slice.tolist(),
                    execution_books=books,
                    last_execution_row=sh.last_execution_row,
                    last_markout_row=sh.last_markout_row,
                )
            )
        return records
