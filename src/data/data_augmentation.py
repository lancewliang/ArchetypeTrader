"""数据增强（仅 train）.

设计文档锚点: §3.4.4 / §5.4 / §9.3。

第一版只实现 ``temporal_contrastive`` 的 shifted horizon 生成；
synthetic 拼接增强占位返回空，留给后续 ablation 实验填充。
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional

from .horizon_builder import HorizonBuilder, HorizonRecord
from .stratified_sampler import SampledHorizon


@dataclass(frozen=True)
class ContrastivePair:
    pair_id: str
    sample_id_original: str
    sample_id_shifted: str
    shift_bars: int


@dataclass(frozen=True)
class SyntheticHorizonMeta:
    sample_id: str
    source_sample_id_a: str
    source_sample_id_b: str
    splice_index: int
    blend_window: int
    synthetic_method: str


class TemporalContrastiveBuilder:
    """对 train horizon 做 ±1/±2 bar 偏移生成 positive pair。"""

    def __init__(
        self,
        shift_bars: List[int],
        pair_ratio: float,
        max_pairs: int,
        require_same_strata: bool = False,
        seed: int = 2026,
    ) -> None:
        self.shift_bars = list(shift_bars)
        self.pair_ratio = pair_ratio
        self.max_pairs = max_pairs
        self.require_same_strata = require_same_strata
        self.seed = seed

    def build_pairs(
        self,
        train_horizons: List[HorizonRecord],
        train_frame,
        builder: HorizonBuilder,
        pair: str,
    ) -> tuple[List[HorizonRecord], List[ContrastivePair]]:
        """生成 shifted horizons 与 pair index。

        实现策略:
        - 按 ``pair_ratio * len(train_horizons)`` 限制配对总数。
        - 每个原始 horizon 随机选 1 个 ``shift``。
        - 边界检查: shifted window 必须落在 [0, num_rows-h-lookahead) 内。
        """
        rng = random.Random(self.seed)
        num_rows = train_frame.height
        h = builder.horizon
        lookahead = builder.alignment.required_lookahead_rows()

        target_count = min(self.max_pairs, int(len(train_horizons) * self.pair_ratio))
        candidates = list(train_horizons)
        rng.shuffle(candidates)

        new_records: List[HorizonRecord] = []
        pairs: List[ContrastivePair] = []
        for original in candidates:
            if len(pairs) >= target_count:
                break
            shift = rng.choice(self.shift_bars)
            new_start = original.start_index + shift
            if new_start < 0 or new_start + h + lookahead > num_rows:
                continue  # 越界
            shifted_id = f"{original.sample_id}_sh{shift:+d}"
            sh_record = SampledHorizon(
                sample_id=shifted_id,
                window_start=new_start,
                window_end=new_start + h - 1,
                last_execution_row=new_start + h - 1 + (lookahead - 1),
                last_markout_row=new_start + h - 1 + lookahead,
                strata_label=original.strata_label,
            )
            built = builder.build(train_frame, [sh_record], pair=pair, split="train")
            if not built:
                continue
            shifted_rec = built[0]
            shifted_rec.is_augmented = True
            shifted_rec.augmentation_type = "temporal_shift"
            new_records.append(shifted_rec)
            pairs.append(
                ContrastivePair(
                    pair_id=f"p_{original.sample_id}_{shift:+d}",
                    sample_id_original=original.sample_id,
                    sample_id_shifted=shifted_id,
                    shift_bars=shift,
                )
            )
        return new_records, pairs


class SyntheticHorizonBuilder:
    """合成 horizon: 第一版返回空列表，作为占位让上层流程不中断。

    完整实现见设计 §5.4.2; 在 ablation 实验时再启用。
    """

    def __init__(
        self,
        synthetic_ratio: float,
        max_synthetic: int,
        blend_window: int,
        min_source_distance: int,
        require_orderbook_consistency: bool = True,
    ) -> None:
        self.synthetic_ratio = synthetic_ratio
        self.max_synthetic = max_synthetic
        self.blend_window = blend_window
        self.min_source_distance = min_source_distance
        self.require_orderbook_consistency = require_orderbook_consistency

    def build_synthetic(
        self,
        train_horizons: List[HorizonRecord],
        train_frame,
    ) -> tuple[List[HorizonRecord], List[SyntheticHorizonMeta]]:
        # 占位: 默认关闭路径不应被调用；真启用时再补完整拼接逻辑。
        return [], []
