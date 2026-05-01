"""Phase II dataset: horizon index + market frame → selector state 读取接口。

设计文档锚点: Phase II 执行计划 §Step 2。

职责:
- 将 horizon index + 原始 market frame 适配为 state 读取接口。
- 不重写 HorizonBuilder 逻辑；若需要 HorizonInputs，必须函数级复用 Phase I 切片协议。
- 不调用 DP。

关键约束:
- state 维度与 feature_columns + position_encoding + optional extensions 一致。
- position_continuity=true 时 prev_terminal_position 必须进入状态。
- close 不进入 selector state。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.config.phase2_config import Phase2Config
from src.data.phase2_horizon_index import Phase2HorizonEntry
from src.trading.cost_model import ExecutionBook
from src.trading.env import HorizonInputs
from src.trading.reward_alignment import RewardAlignment


@dataclass
class Phase2StateSpec:
    """Selector 状态维度规范。

    记录 feature_columns / position_encoding / extensions 的维度分解，
    用于 state_dim_breakdown 校验。
    """
    feature_dim: int
    position_dim: int
    total_dim: int
    feature_columns: List[str]
    has_prev_terminal_position: bool


class Phase2Dataset:
    """Phase II 数据集: 提供 selector state 读取接口。

    不继承 torch.utils.data.Dataset，因为 HorizonEnv 直接按 index 读取，
    不走 DataLoader。

    边界:
    - 不读取原始文件（由调用方传入 frame）。
    - 不调用 DP。
    - 不重写 Phase I horizon slicing 语义。
    """

    def __init__(
        self,
        frame,
        horizon_entries: List[Phase2HorizonEntry],
        input_schema: Dict[str, Any],
        config: Phase2Config,
        reward_alignment: Optional[str] = None,
    ) -> None:
        self.horizon_entries = horizon_entries
        self.config = config
        self._feature_columns: List[str] = input_schema.get("feature_columns", [])
        self._price_column: str = input_schema.get("price_column", "close")
        self._horizon = config.horizon
        # reward_alignment 由 trainer/validator 解析后传入；保留 fallback 兼容单测。
        self._alignment = RewardAlignment(
            reward_alignment or self._resolve_reward_alignment()
        )

        # 预提取 numpy 数组以避免逐行 polars 查询
        import polars as pl
        if isinstance(frame, pl.DataFrame):
            self._validate_inputs(frame)
            self._feature_matrix = frame.select(self._feature_columns).to_numpy().astype("float32")
            self._close = frame[self._price_column].to_numpy().astype("float32")
            self._num_rows = frame.height
            # 盘口
            self._ask_p = [
                frame[f"ask{i}_price"].to_numpy().astype("float32")
                if f"ask{i}_price" in frame.columns
                else np.zeros(frame.height, dtype="float32")
                for i in range(1, 6)
            ]
            self._ask_s = [
                frame[f"ask{i}_size"].to_numpy().astype("float32")
                if f"ask{i}_size" in frame.columns
                else np.zeros(frame.height, dtype="float32")
                for i in range(1, 6)
            ]
            self._bid_p = [
                frame[f"bid{i}_price"].to_numpy().astype("float32")
                if f"bid{i}_price" in frame.columns
                else np.zeros(frame.height, dtype="float32")
                for i in range(1, 6)
            ]
            self._bid_s = [
                frame[f"bid{i}_size"].to_numpy().astype("float32")
                if f"bid{i}_size" in frame.columns
                else np.zeros(frame.height, dtype="float32")
                for i in range(1, 6)
            ]
            # mark price
            if "ask1_price" in frame.columns and "bid1_price" in frame.columns:
                mark = (self._ask_p[0] + self._bid_p[0]) / 2.0
                invalid = (self._ask_p[0] <= 0) | (self._bid_p[0] <= 0)
                self._mark = np.where(invalid, self._close, mark)
            else:
                self._mark = self._close.copy()
        else:
            raise TypeError("frame 必须是 polars.DataFrame")

    def _validate_inputs(self, frame) -> None:
        """构造阶段做防御性校验，避免训练中途暴露难定位错误。"""
        missing_features = [c for c in self._feature_columns if c not in frame.columns]
        if missing_features:
            raise ValueError(f"input_schema feature_columns 在 frame 中缺失: {missing_features}")
        if self._price_column not in frame.columns:
            raise ValueError(f"price_column={self._price_column!r} 在 frame 中缺失")

        if "timestamp" in frame.columns and frame.height > 1:
            ts_values = frame["timestamp"].to_list()
            for i in range(1, len(ts_values)):
                if ts_values[i] < ts_values[i - 1]:
                    raise ValueError("frame.timestamp 必须单调非递减")

        if self.horizon_entries:
            splits = {e.split for e in self.horizon_entries}
            if len(splits) != 1:
                raise ValueError(f"horizon_entries split 不一致: {sorted(splits)}")
            if "split" in frame.columns:
                frame_splits = set(str(v) for v in frame["split"].unique().to_list())
                if frame_splits and not splits.issubset(frame_splits):
                    raise ValueError(
                        f"horizon_entries split={sorted(splits)} 与 frame split={sorted(frame_splits)} 不匹配"
                    )
            for e in self.horizon_entries:
                if e.horizon_start < 0 or e.horizon_end < e.horizon_start:
                    raise ValueError(f"非法 horizon 边界: {e}")
                if e.horizon_end >= frame.height:
                    raise ValueError(
                        f"horizon_end={e.horizon_end} 超过 frame 行数 {frame.height}"
                    )

    def _resolve_reward_alignment(self) -> str:
        """从 Phase I config 获取 reward_alignment，带文件存在性检查。"""
        phase1_dir = self.config.phase1_dir()
        config_path = phase1_dir / "phase1_config.yaml"
        if config_path.exists():
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                p1cfg = yaml.safe_load(f) or {}
            return p1cfg.get("dp", {}).get("cost_config", {}).get(
                "reward_alignment", "paper_formula"
            )
        return "paper_formula"

    def get_selector_state(
        self, idx: int, prev_terminal_position: int = 0
    ) -> np.ndarray:
        """获取第 idx 个 horizon 的 selector 状态 s^sel。

        返回 feature vector，可选拼接 prev_terminal_position 编码。
        """
        entry = self.horizon_entries[idx]
        start = entry.horizon_start
        # selector state = horizon 起点的 feature vector
        features = self._feature_matrix[start, :].copy()

        if self.config.horizon_schedule.position_continuity:
            # 拼接 prev_terminal_position 的 scaled encoding
            pos_enc = np.array(
                [float(prev_terminal_position) / max(self.config.max_position, 1)],
                dtype="float32",
            )
            return np.concatenate([features, pos_enc])
        return features

    def get_horizon_inputs(self, idx: int) -> HorizonInputs:
        """获取第 idx 个 horizon 的 HorizonInputs（prices + execution_books）。

        复用 Phase I 的 HorizonBuilder 切片协议。
        """
        entry = self.horizon_entries[idx]
        start = entry.horizon_start
        h = self._horizon
        lookahead = self._alignment.required_lookahead_rows()

        prices = self._mark[start: start + h + lookahead].tolist()

        books: List[ExecutionBook] = []
        for t in range(h):
            rows = self._alignment.rows(t)
            row = start + rows.execution_row
            if row >= self._num_rows:
                raise IndexError(
                    f"horizon {entry.sample_id} execution_row={row} 越界; "
                    f"num_rows={self._num_rows}"
                )
            book = ExecutionBook(
                ask_prices=tuple(float(self._ask_p[k][row]) for k in range(5)),
                ask_sizes=tuple(float(self._ask_s[k][row]) for k in range(5)),
                bid_prices=tuple(float(self._bid_p[k][row]) for k in range(5)),
                bid_sizes=tuple(float(self._bid_s[k][row]) for k in range(5)),
                mark_price=float(self._mark[row]),
            )
            books.append(book)

        return HorizonInputs(prices=prices, execution_books=books)

    def get_horizon_states(self, idx: int) -> np.ndarray:
        """获取第 idx 个 horizon 的完整 states [h, feature_dim]。

        用于 Phase1FrozenPolicy.decode_step() 的逐步输入。
        """
        entry = self.horizon_entries[idx]
        start = entry.horizon_start
        h = self._horizon
        return self._feature_matrix[start: start + h, :].copy()

    def state_spec(self) -> Phase2StateSpec:
        """返回 state 维度规范，用于 selector 网络初始化和校验。"""
        feature_dim = len(self._feature_columns)
        position_dim = 1 if self.config.horizon_schedule.position_continuity else 0
        return Phase2StateSpec(
            feature_dim=feature_dim,
            position_dim=position_dim,
            total_dim=feature_dim + position_dim,
            feature_columns=list(self._feature_columns),
            has_prev_terminal_position=self.config.horizon_schedule.position_continuity,
        )

    def __len__(self) -> int:
        return len(self.horizon_entries)
