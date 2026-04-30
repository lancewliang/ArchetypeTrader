"""分层采样.

设计文档锚点: §3.4 与 §4.3。

约束:
- 不读原始文件，只接收 ``WindowIndexEntry`` 列表与 strata 标签。
- 同一 ``seed`` + 同一输入 → 相同 ``window_start`` 集合（reproducibility 验收项）。
- ``min_gap_between_samples`` 在 strata 内强制；不得用 ``allow_overlap_relaxation=True`` 默认放宽。
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Literal, Sequence

from .window_indexer import WindowIndexEntry


@dataclass(frozen=True)
class SampledHorizon:
    """采样后的 horizon。"""
    sample_id: str
    window_start: int
    window_end: int
    last_execution_row: int
    last_markout_row: int
    strata_label: str


SamplingStrategy = Literal["stratified_uniform", "stratified_proportional"]


def _bin_return(value: float) -> str:
    """horizon_return 分桶。阈值与设计 §3.4 默认一致。"""
    if math.isnan(value):
        return "unknown"
    if value > 0.002:
        return "up"
    if value < -0.002:
        return "down"
    return "flat"


def _bin_vol(value: float) -> str:
    """波动率分桶。阈值参考分钟级 returns std 经验值。"""
    if math.isnan(value):
        return "unknown"
    if value > 0.0015:
        return "high"
    if value < 0.0005:
        return "low"
    return "mid"


def _strata_from_entry(entry: WindowIndexEntry, prospective: bool) -> str:
    """构造 strata label。"""
    if prospective:
        ret = entry.past_return
        vol = entry.past_realized_volatility
        pattern = entry.past_draw_pattern
    else:
        ret = entry.horizon_return
        vol = entry.realized_volatility
        pattern = entry.draw_pattern
    return f"{_bin_return(ret)}|{_bin_vol(vol)}|{pattern}"


class StratifiedWindowSampler:
    """按 strata 做去相关采样。

    实现策略
    --------
    1. 按 ``strata_label`` 分桶，桶内 deterministic 洗牌（固定 seed）。
    2. 按 ``stratified_uniform`` 或 ``stratified_proportional`` 计算各桶配额。
    3. 抑制 ``flat|low|*`` 类 strata 的总占比不超过 ``flat_low_vol_max_ratio``，
       多出的额度补到非 flat-low 类候选最多的 strata。
    4. 桶内贪心采样: 已选窗口起点距离 < ``min_gap_between_samples`` 跳过。
    5. 因 gap 约束采不满时:
       - ``allow_overlap_relaxation=False``（默认）: 抛错。
       - ``allow_overlap_relaxation=True``: 把 gap 折半再尝试一次，
         并通过 ``sampling_health_warnings`` 提醒。
    """

    def __init__(
        self,
        strategy: SamplingStrategy,
        min_gap_between_samples: int,
        flat_low_vol_max_ratio: float,
        allow_overlap_relaxation: bool = False,
        seed: int = 42,
    ) -> None:
        if strategy not in ("stratified_uniform", "stratified_proportional"):
            raise ValueError(f"非法 strategy: {strategy}")
        self.strategy = strategy
        self.min_gap_between_samples = min_gap_between_samples
        self.flat_low_vol_max_ratio = flat_low_vol_max_ratio
        self.allow_overlap_relaxation = allow_overlap_relaxation
        self.seed = seed

    def sample(
        self,
        entries: Sequence[WindowIndexEntry],
        num_samples: int,
        strata_labels: Sequence[str],
    ) -> List[SampledHorizon]:
        """按策略做分层采样。

        Parameters
        ----------
        entries : 候选窗口（``SlidingWindowIndexer.enumerate`` 输出）。
        num_samples : 期望采样数；超过候选总数会抛 ``ValueError``。
        strata_labels : 与 ``entries`` 顺序一致；通常由
                        ``StratifiedWindowSampler.assign_strata`` 生成。

        Returns
        -------
        list[SampledHorizon] : 长度严格 ≤ ``num_samples``；
                                正常情况下 == ``num_samples``。

        Raises
        ------
        ValueError : 长度不一致 / num_samples 越界。
        RuntimeError : 因 gap 约束采不满且禁止放宽。
        """
        if len(entries) != len(strata_labels):
            raise ValueError("entries 与 strata_labels 长度必须一致")
        if num_samples <= 0:
            return []
        if num_samples > len(entries):
            raise ValueError(
                f"num_samples={num_samples} 超过候选总数 {len(entries)}; 无法采样"
            )

        rng = random.Random(self.seed)

        # 按 strata 分桶。
        buckets: Dict[str, List[int]] = defaultdict(list)
        for idx, label in enumerate(strata_labels):
            buckets[label].append(idx)
        # 桶内打乱（fixed seed → deterministic）。
        for label in buckets:
            rng.shuffle(buckets[label])

        # 计算每个 strata 的配额。
        non_empty_strata = [k for k, v in buckets.items() if v]
        n_strata = len(non_empty_strata)
        if n_strata == 0:
            raise RuntimeError("没有可用 strata，无法采样")

        if self.strategy == "stratified_uniform":
            # 每个 strata 平均分配；剩余余量按桶大小补齐。
            base = num_samples // n_strata
            quotas: Dict[str, int] = {k: base for k in non_empty_strata}
            remainder = num_samples - base * n_strata
            for k in sorted(non_empty_strata, key=lambda s: -len(buckets[s])):
                if remainder <= 0:
                    break
                quotas[k] += 1
                remainder -= 1
        else:
            # stratified_proportional
            total = sum(len(buckets[k]) for k in non_empty_strata)
            quotas = {
                k: max(1, round(num_samples * len(buckets[k]) / total))
                for k in non_empty_strata
            }
            # 修正凑齐 num_samples
            diff = num_samples - sum(quotas.values())
            sorted_keys = sorted(non_empty_strata, key=lambda s: -len(buckets[s]))
            i = 0
            while diff != 0 and sorted_keys:
                key = sorted_keys[i % len(sorted_keys)]
                if diff > 0:
                    quotas[key] += 1
                    diff -= 1
                else:
                    if quotas[key] > 0:
                        quotas[key] -= 1
                        diff += 1
                i += 1

        # 抑制 flat|low|* strata 比例（"flat-low" 类容易让 DP 全 flat）。
        flat_low_keys = [k for k in non_empty_strata if k.startswith("flat|low|")]
        flat_low_total = sum(quotas.get(k, 0) for k in flat_low_keys)
        cap = int(num_samples * self.flat_low_vol_max_ratio)
        if flat_low_total > cap and flat_low_keys:
            overflow = flat_low_total - cap
            # 从 flat_low 桶按比例减；减去的量补到非 flat_low 中候选最多的 strata。
            non_flat_low = [k for k in non_empty_strata if k not in flat_low_keys]
            non_flat_low.sort(key=lambda s: -len(buckets[s]))
            i = 0
            while overflow > 0 and flat_low_keys:
                # 找出当前 quota 最大的 flat_low strata 减 1
                candidate = max(flat_low_keys, key=lambda k: quotas.get(k, 0))
                if quotas[candidate] <= 0:
                    flat_low_keys = [k for k in flat_low_keys if quotas[k] > 0]
                    if not flat_low_keys:
                        break
                    continue
                quotas[candidate] -= 1
                if non_flat_low:
                    quotas[non_flat_low[i % len(non_flat_low)]] += 1
                    i += 1
                overflow -= 1

        # 每个 strata 内做 min_gap 去相关采样。
        sampled_indices: List[int] = []
        chosen_starts: List[int] = []
        for label, idx_list in buckets.items():
            quota = quotas.get(label, 0)
            if quota <= 0:
                continue
            picked = self._pick_with_gap(
                idx_list,
                entries,
                quota=quota,
                min_gap=self.min_gap_between_samples,
                chosen_starts=chosen_starts,
            )
            sampled_indices.extend(picked)
            chosen_starts.extend(entries[i].window_start for i in picked)

        # 如果由于 gap 约束 + 严禁放宽导致采不满，从其他 strata 补；
        # 仍不够则抛错。
        if len(sampled_indices) < num_samples:
            shortfall = num_samples - len(sampled_indices)
            already = set(sampled_indices)
            spare_pool: List[int] = []
            for label, idx_list in buckets.items():
                for i in idx_list:
                    if i in already:
                        continue
                    spare_pool.append(i)
            spare_pool.sort()
            picked_extra = self._pick_with_gap(
                spare_pool,
                entries,
                quota=shortfall,
                min_gap=self.min_gap_between_samples,
                chosen_starts=chosen_starts,
            )
            sampled_indices.extend(picked_extra)
            chosen_starts.extend(entries[i].window_start for i in picked_extra)
            if len(sampled_indices) < num_samples:
                if not self.allow_overlap_relaxation:
                    raise RuntimeError(
                        f"无法在 min_gap={self.min_gap_between_samples} 下采到 {num_samples} 个样本; "
                        f"实际只能采 {len(sampled_indices)} 个；"
                        "如需放宽请设 allow_overlap_relaxation=True 并记录原因。"
                    )
                # 放宽: 把 gap 折半，重新尝试一次（实操中通常够用）
                relaxed_gap = max(1, self.min_gap_between_samples // 2)
                picked_extra = self._pick_with_gap(
                    spare_pool,
                    entries,
                    quota=num_samples - len(sampled_indices),
                    min_gap=relaxed_gap,
                    chosen_starts=chosen_starts,
                )
                sampled_indices.extend(picked_extra)
                chosen_starts.extend(entries[i].window_start for i in picked_extra)

        sampled_indices = list(dict.fromkeys(sampled_indices))[:num_samples]
        result: List[SampledHorizon] = []
        for idx in sampled_indices:
            entry = entries[idx]
            label = strata_labels[idx]
            sample_id = self._build_sample_id(idx, entry, label)
            result.append(
                SampledHorizon(
                    sample_id=sample_id,
                    window_start=entry.window_start,
                    window_end=entry.window_end,
                    last_execution_row=entry.last_execution_row,
                    last_markout_row=entry.last_markout_row,
                    strata_label=label,
                )
            )
        return result

    @staticmethod
    def assign_strata(entry: WindowIndexEntry, prospective: bool) -> str:
        return _strata_from_entry(entry, prospective)

    # ---------- 私有 ----------

    @staticmethod
    def _pick_with_gap(
        candidate_indices: Sequence[int],
        entries: Sequence[WindowIndexEntry],
        quota: int,
        min_gap: int,
        chosen_starts: Sequence[int] | None = None,
    ) -> List[int]:
        """在 strata 内按 min_gap 去相关采样。

        贪心策略: 候选已被随机洗过；按顺序选择，跳过与已选窗口距离 < min_gap 的。
        """
        picked: List[int] = []
        all_chosen_starts: List[int] = list(chosen_starts or [])
        for idx in candidate_indices:
            if quota <= 0:
                break
            start = entries[idx].window_start
            if all(abs(start - s) >= min_gap for s in all_chosen_starts):
                picked.append(idx)
                all_chosen_starts.append(start)
                quota -= 1
        return picked

    @staticmethod
    def _build_sample_id(idx: int, entry: WindowIndexEntry, label: str) -> str:
        """sample_id 必须在固定 seed + 输入下 deterministic。

        约定: ``s_{window_start:08d}_{idx:06d}``，便于排序与 join。
        """
        return f"s_{entry.window_start:08d}_{idx:06d}"
