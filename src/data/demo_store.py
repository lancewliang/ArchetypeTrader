"""DP demonstration / horizon labels 持久化.

设计文档锚点: §4.3 与 §8 (输出产物)。

第一版用 polars feather 写出，metadata 通过附加列携带（feather 本身不写文件级 metadata）。
关键审计字段（config_hash / schema_hash）作为列前缀写入，下游读取时校验。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

from src.utils import feather_io

from .horizon_builder import HorizonRecord


@dataclass(frozen=True)
class HorizonLabel:
    sample_id: str
    start_index: int
    end_index: int
    last_execution_row: int
    last_markout_row: int
    strata_label: str
    stratification_mode: str
    is_augmented: bool
    augmentation_type: str
    code_label: int
    demo_return: float
    num_switches: int
    is_no_trade: bool


class Phase1DemoStore:
    """统一的 demo / label 持久化对象。

    边界
    ----
    - 不调用 DP；调用方负责把 ``actions/rewards`` 填进 ``HorizonRecord``。
    - 不计算 metrics；只做序列化。
    - 通过 ``_config_hash`` / ``_schema_hash`` 列做 cache 失效检测。
      下游 ``load_demos`` 会校验，配置变化必须重新生成。
    """

    def __init__(self, artifacts_dir, config_hash: str, schema_hash: str) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.config_hash = config_hash
        self.schema_hash = schema_hash

    # ---------- demos ----------

    def save_demos(self, demos: List[HorizonRecord]) -> Path:
        """写 ``demos_train.feather``，包含 ``states / prices / actions / rewards / meta``。

        实现要点
        --------
        - 大数组按列写出（states/prices）；polars 会保留嵌套 list dtype。
        - ``execution_books`` 不写入（盘口体积大且 cache 命中后不需要 replay）。
        - ``_config_hash`` / ``_schema_hash`` 作为列存入，避免依赖文件级 metadata。
        """
        import polars as pl

        rows = []
        for d in demos:
            rows.append(
                {
                    "sample_id": d.sample_id,
                    "pair": d.pair,
                    "split": d.split,
                    "start_index": d.start_index,
                    "end_index": d.end_index,
                    "strata_label": d.strata_label,
                    "is_augmented": d.is_augmented,
                    "augmentation_type": d.augmentation_type,
                    "states": d.states,
                    "prices": d.prices,
                    "actions": d.actions if d.actions is not None else [],
                    "rewards": d.rewards if d.rewards is not None else [],
                    "_config_hash": self.config_hash,
                    "_schema_hash": self.schema_hash,
                }
            )
        frame = pl.DataFrame(rows)
        target = self.artifacts_dir / "demos_train.feather"
        return feather_io.write_ipc(frame, target)

    def load_demos(self, path: Optional[Path] = None) -> List[HorizonRecord]:
        """读取 ``demos_train.feather`` 并校验 hash 一致。

        Raises
        ------
        ValueError : 文件中的 ``_config_hash`` / ``_schema_hash`` 与当前不一致；
                     此时必须重新生成 demos，禁止误用旧 cache。

        Notes
        -----
        反序列化时 ``execution_books`` 不还原（保存时也未写入）；
        cache 主要供 trainer 内部 tensor 准备使用，命中后不需要再 replay 老师轨迹。
        """
        target = Path(path) if path else self.artifacts_dir / "demos_train.feather"
        frame = feather_io.read_ipc(target)
        if frame.height == 0:
            return []
        # 校验前两个字段（够代表 cache key）
        first_row = frame.row(0, named=True)
        if first_row.get("_config_hash") != self.config_hash:
            raise ValueError(
                f"demos cache 的 config_hash 与当前不一致; 应重新生成。"
                f" disk={first_row.get('_config_hash')} now={self.config_hash}"
            )
        if first_row.get("_schema_hash") != self.schema_hash:
            raise ValueError(
                f"demos cache 的 schema_hash 与当前不一致; 应重新生成。"
            )
        # 注意: 反序列化时 execution_books 不还原（保存时未写入）；
        # demo cache 主要用于 trainer 内部 tensor 准备，cache 命中后不需要重 replay。
        records: List[HorizonRecord] = []
        for row in frame.iter_rows(named=True):
            records.append(
                HorizonRecord(
                    sample_id=row["sample_id"],
                    start_index=row["start_index"],
                    end_index=row["end_index"],
                    pair=row["pair"],
                    split=row["split"],
                    strata_label=row["strata_label"],
                    states=row["states"],
                    prices=row["prices"],
                    execution_books=[],  # cache 不还原盘口
                    actions=row["actions"] or None,
                    rewards=row["rewards"] or None,
                    is_augmented=row["is_augmented"],
                    augmentation_type=row["augmentation_type"],
                )
            )
        return records

    # ---------- labels ----------

    def save_labels(self, labels: List[HorizonLabel], split: str) -> Path:
        import polars as pl

        rows = [asdict(lab) for lab in labels]
        # 添加 hash 字段便于审计
        for r in rows:
            r["_config_hash"] = self.config_hash
            r["_schema_hash"] = self.schema_hash
        frame = pl.DataFrame(rows)
        target = self.artifacts_dir / f"horizon_labels_{split}.feather"
        return feather_io.write_ipc(frame, target)

    def load_labels(self, split: str) -> List[HorizonLabel]:
        target = self.artifacts_dir / f"horizon_labels_{split}.feather"
        frame = feather_io.read_ipc(target)
        out: List[HorizonLabel] = []
        for row in frame.iter_rows(named=True):
            out.append(
                HorizonLabel(
                    sample_id=row["sample_id"],
                    start_index=row["start_index"],
                    end_index=row["end_index"],
                    last_execution_row=row["last_execution_row"],
                    last_markout_row=row["last_markout_row"],
                    strata_label=row["strata_label"],
                    stratification_mode=row["stratification_mode"],
                    is_augmented=row["is_augmented"],
                    augmentation_type=row["augmentation_type"],
                    code_label=row["code_label"],
                    demo_return=row["demo_return"],
                    num_switches=row["num_switches"],
                    is_no_trade=row["is_no_trade"],
                )
            )
        return out
