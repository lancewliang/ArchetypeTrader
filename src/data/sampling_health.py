"""采样健康检查.

设计文档锚点: §3.4, §9.2 与 §10。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from .stratified_sampler import SampledHorizon


class SamplingHealthError(RuntimeError):
    """采样健康检查阻塞错误。"""


@dataclass
class SamplingHealthReport:
    """采样健康汇总；写入 ``phase1_report.json``。"""
    window_overlap_ratio: float = 0.0
    min_sample_gap: int = 0
    mean_sample_gap: float = 0.0
    flat_low_vol_sample_ratio: float = 0.0
    split_boundary_gap: int = 0
    effective_min_gap_between_samples: int = 0
    overlap_relaxation_applied: bool = False
    sampling_health_warnings: List[str] = field(default_factory=list)


class SamplingHealthChecker:
    """对最终采样结果做健康检查。

    输出指标
    --------
    - ``window_overlap_ratio`` : 相邻采样窗口的平均重叠比例
      ``mean(max(0, h - gap) / h)``；过高 → 训练样本时间自相关高、validation 虚高。
    - ``min_sample_gap`` / ``mean_sample_gap``: 相邻 ``window_start`` 间距统计。
    - ``flat_low_vol_sample_ratio`` : 占比；过高 → DP 多落入全 flat。
    - ``split_boundary_gap`` : 任一采样的 ``last_markout_row`` 距 ``train_end_row``
      的最近距离；必须 ≥ ``split_boundary_embargo``，否则 markout 行可能越过
      train/val/test 边界。

    行为
    ----
    - ``warn_only=False`` 且任一阈值超出 → 抛 ``SamplingHealthError``。
    - ``warn_only=True`` 时只把警告写入 ``report.sampling_health_warnings``。
    """

    def __init__(
        self,
        horizon: int,
        max_overlap_ratio: float,
        min_gap_between_samples: int,
        split_boundary_embargo: int,
        flat_low_vol_max_ratio: float,
        warn_only: bool = False,
        effective_min_gap_between_samples: int | None = None,
        overlap_relaxation_applied: bool = False,
    ) -> None:
        self.horizon = horizon
        self.max_overlap_ratio = max_overlap_ratio
        self.min_gap_between_samples = min_gap_between_samples
        self.split_boundary_embargo = split_boundary_embargo
        self.flat_low_vol_max_ratio = flat_low_vol_max_ratio
        self.warn_only = warn_only
        self.effective_min_gap_between_samples = (
            effective_min_gap_between_samples
            if effective_min_gap_between_samples is not None
            else min_gap_between_samples
        )
        self.overlap_relaxation_applied = overlap_relaxation_applied

    def check(
        self,
        sampled: Sequence[SampledHorizon],
        split_boundaries: Dict[str, int],
        strata_labels: Sequence[str],
    ) -> SamplingHealthReport:
        """计算指标 + 校验阈值。

        Parameters
        ----------
        sampled : 已经采样的 horizon 列表（顺序无所谓；内部会按 window_start 排序）。
        split_boundaries : ``{"train_end_row": int, ...}``；可选 ``val_start_row`` 等。
        strata_labels : 与 ``sampled`` 顺序一致的 strata label 列表，用于 flat_low 占比。

        Returns
        -------
        SamplingHealthReport

        Raises
        ------
        SamplingHealthError : ``warn_only=False`` 且任一阈值超出。
        """
        if not sampled:
            raise SamplingHealthError("sampled 为空，无法做健康检查")

        report = SamplingHealthReport(
            effective_min_gap_between_samples=self.effective_min_gap_between_samples,
            overlap_relaxation_applied=self.overlap_relaxation_applied,
        )

        sorted_samples = sorted(sampled, key=lambda s: s.window_start)
        starts = [s.window_start for s in sorted_samples]

        # 相邻 gap（按 sorted）。
        gaps = [starts[i + 1] - starts[i] for i in range(len(starts) - 1)]
        if gaps:
            report.min_sample_gap = min(gaps)
            report.mean_sample_gap = sum(gaps) / len(gaps)
            # window_overlap_ratio: 相邻窗口平均重叠比例。
            overlaps = [max(0, self.horizon - g) / self.horizon for g in gaps]
            report.window_overlap_ratio = sum(overlaps) / len(overlaps)
        else:
            report.min_sample_gap = 0
            report.mean_sample_gap = 0.0
            report.window_overlap_ratio = 0.0

        # flat-low 占比
        if strata_labels:
            flat_low = sum(1 for lab in strata_labels if lab.startswith("flat|low|"))
            report.flat_low_vol_sample_ratio = flat_low / len(strata_labels)

        # split boundary: 计算 sampled 中 last_markout_row 与 train_end_row 的距离。
        train_end = split_boundaries.get("train_end_row")
        if train_end is not None:
            max_markout = max(s.last_markout_row for s in sorted_samples)
            report.split_boundary_gap = train_end - max_markout
        else:
            report.split_boundary_gap = 0

        # 阈值校验
        if report.window_overlap_ratio > self.max_overlap_ratio:
            report.sampling_health_warnings.append(
                f"window_overlap_ratio={report.window_overlap_ratio:.3f} > "
                f"max={self.max_overlap_ratio}"
            )
        if (
            report.min_sample_gap > 0
            and report.min_sample_gap < report.effective_min_gap_between_samples
        ):
            report.sampling_health_warnings.append(
                f"min_sample_gap={report.min_sample_gap} < "
                f"effective_min_gap_between_samples={report.effective_min_gap_between_samples}"
            )
        if report.flat_low_vol_sample_ratio > self.flat_low_vol_max_ratio:
            report.sampling_health_warnings.append(
                f"flat_low_vol_sample_ratio={report.flat_low_vol_sample_ratio:.3f} > "
                f"max={self.flat_low_vol_max_ratio}"
            )
        if train_end is not None and report.split_boundary_gap < self.split_boundary_embargo:
            report.sampling_health_warnings.append(
                f"split_boundary_gap={report.split_boundary_gap} < "
                f"embargo={self.split_boundary_embargo}; "
                f"markout 行 {max(s.last_markout_row for s in sorted_samples)} "
                f"距 train_end={train_end} 过近"
            )

        if not self.warn_only and report.sampling_health_warnings:
            raise SamplingHealthError(
                "采样健康检查未通过: " + "; ".join(report.sampling_health_warnings)
            )
        return report
