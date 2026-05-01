"""Phase II label loader: join train/val 的 code_label 与 is_labeled。

设计文档锚点: Phase II 执行计划 §Step 2。

职责:
- 仅 join train/val 的 code_label 与 is_labeled。
- 输出 kl_label_temporal_coverage 的原始统计。
- test labels 被请求时抛错。

关键约束:
- test split 的 code_label 只能用于 posthoc baseline，不可进入训练/决策路径。
- label 时间分布熵过低时写入 warning。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl

from src.config.phase2_config import Phase2Config
from src.data.phase2_horizon_index import Phase2HorizonEntry
from src.utils.feather_io import read_ipc


class Phase2TestLabelRequestError(ValueError):
    """试图加载 test split 的 code_label 进入训练/决策路径。"""


@dataclass
class LabelCoverageStats:
    """KL label 时间覆盖统计。"""
    total_horizons: int = 0
    labeled_horizons: int = 0
    coverage_ratio: float = 0.0
    temporal_coverage_sequence: List[float] = field(default_factory=list)
    entropy: float = 0.0
    warnings: List[str] = field(default_factory=list)


class Phase2LabelLoader:
    """加载并 join Phase I 的 code_label 到 Phase II horizon index。

    边界:
    - 只 join train/val 的 code_label。
    - test labels 被请求时抛 Phase2TestLabelRequestError。
    - 未标注 horizon 的 is_labeled=false。
    """

    def __init__(self, config: Phase2Config) -> None:
        self.config = config

    def load_and_join(
        self,
        horizon_entries: List[Phase2HorizonEntry],
        split: str,
        phase1_labels_path: Optional[Path] = None,
    ) -> List[Phase2HorizonEntry]:
        """将 Phase I code_label join 到 horizon entries。

        Parameters
        ----------
        horizon_entries : Phase II horizon index。
        split : "train" / "val" / "test"。
        phase1_labels_path : Phase I horizon labels feather 路径。

        Returns
        -------
        更新后的 horizon entries（code_label / is_labeled 已填充）。

        Raises
        ------
        Phase2TestLabelRequestError : split="test" 时抛出。
        """
        if split == "test":
            raise Phase2TestLabelRequestError(
                "test split 的 code_label 不可进入训练/决策路径。"
                "仅允许 posthoc baseline 使用。"
            )

        if phase1_labels_path is None or not Path(phase1_labels_path).exists():
            return horizon_entries

        labels_df = read_ipc(phase1_labels_path)
        if "start_index" not in labels_df.columns or "code_label" not in labels_df.columns:
            return horizon_entries

        label_map: Dict[int, int] = {}
        starts = labels_df["start_index"].to_list()
        codes = labels_df["code_label"].to_list()
        for s, c in zip(starts, codes):
            if c is not None:
                label_map[int(s)] = int(c)

        for entry in horizon_entries:
            if entry.horizon_start in label_map:
                entry.code_label = label_map[entry.horizon_start]
                entry.is_labeled = True
            else:
                entry.code_label = None
                entry.is_labeled = False

        return horizon_entries

    def compute_coverage_stats(
        self,
        horizon_entries: List[Phase2HorizonEntry],
        split: str,
    ) -> LabelCoverageStats:
        """计算 kl_label_temporal_coverage 统计。

        包含覆盖率、时间序列覆盖、分布熵和 warnings。
        """
        total = len(horizon_entries)
        labeled = sum(1 for e in horizon_entries if e.is_labeled)
        coverage_ratio = labeled / max(total, 1)

        # 按 10 个 bucket 计算时间覆盖序列
        num_buckets = min(10, max(total, 1))
        bucket_size = max(total // num_buckets, 1)
        temporal_seq: List[float] = []
        for b in range(num_buckets):
            start_idx = b * bucket_size
            end_idx = min((b + 1) * bucket_size, total)
            bucket_entries = horizon_entries[start_idx:end_idx]
            if bucket_entries:
                bucket_labeled = sum(1 for e in bucket_entries if e.is_labeled)
                temporal_seq.append(bucket_labeled / len(bucket_entries))
            else:
                temporal_seq.append(0.0)

        # 计算分布熵
        entropy = 0.0
        if temporal_seq:
            total_coverage = sum(temporal_seq)
            if total_coverage > 0:
                for v in temporal_seq:
                    p = v / total_coverage
                    if p > 0:
                        entropy -= p * math.log(p)

        warnings: List[str] = []
        if entropy < 1.0 and labeled > 0:
            warnings.append(
                f"label 时间分布熵过低 ({entropy:.3f})，"
                "可能存在 KL label 时间偏置"
            )
        if coverage_ratio < 0.1:
            warnings.append(
                f"label 覆盖率过低 ({coverage_ratio:.3f})，"
                "KL/demo regularization 效果可能有限"
            )

        return LabelCoverageStats(
            total_horizons=total,
            labeled_horizons=labeled,
            coverage_ratio=coverage_ratio,
            temporal_coverage_sequence=temporal_seq,
            entropy=entropy,
            warnings=warnings,
        )
