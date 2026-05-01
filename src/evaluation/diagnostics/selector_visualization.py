"""Selector 可视化: 时间 vs 累计收益 vs archetype 选择 / action distribution / entropy 曲线。

设计文档锚点: Phase II 执行计划 §Step 7。

职责:
- 时间 vs 累计收益 vs archetype 选择图。
- action distribution 图。
- entropy / KL 曲线图。
- label temporal coverage 可视化。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


class SelectorVisualizationWriter:
    """Selector 可视化输出。"""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_cumulative_return_with_archetype(
        self,
        horizon_records: List[Dict[str, Any]],
        output_path: Optional[Path] = None,
    ) -> Path:
        """绘制时间 vs 累计收益 vs archetype 选择图。"""
        path = output_path or (self.output_dir / "cumulative_return_archetype.png")
        if not HAS_MPL:
            Path(path).touch()
            return Path(path)

        rewards = [r.get("reward_raw", 0.0) for r in horizon_records]
        codes = [r.get("chosen_code", 0) for r in horizon_records]
        cumulative = []
        running = 0.0
        for r in rewards:
            running += r
            cumulative.append(running)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        ax1.plot(cumulative, label="Cumulative Return")
        ax1.set_ylabel("Cumulative Return")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.scatter(range(len(codes)), codes, c=codes, cmap="tab10", s=10, alpha=0.7)
        ax2.set_ylabel("Archetype Code")
        ax2.set_xlabel("Horizon Index")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(path, dpi=100)
        plt.close(fig)
        return Path(path)

    def plot_action_distribution(
        self,
        actions: List[int],
        num_codes: int,
        output_path: Optional[Path] = None,
    ) -> Path:
        """绘制 action distribution 图。"""
        path = output_path or (self.output_dir / "action_distribution.png")
        if not HAS_MPL:
            Path(path).touch()
            return Path(path)

        from collections import Counter
        counts = Counter(actions)
        codes = list(range(num_codes))
        freqs = [counts.get(c, 0) for c in codes]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(codes, freqs)
        ax.set_xlabel("Archetype Code")
        ax.set_ylabel("Count")
        ax.set_title("Action Distribution")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(path, dpi=100)
        plt.close(fig)
        return Path(path)

    def plot_entropy_kl_curves(
        self,
        entropy_history: List[float],
        kl_history: List[float],
        output_path: Optional[Path] = None,
    ) -> Path:
        """绘制 entropy / KL 曲线图。"""
        path = output_path or (self.output_dir / "entropy_kl_curves.png")
        if not HAS_MPL:
            Path(path).touch()
            return Path(path)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        ax1.plot(entropy_history, label="Entropy")
        ax1.set_ylabel("Entropy")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(kl_history, label="Approx KL", color="orange")
        ax2.set_ylabel("Approx KL")
        ax2.set_xlabel("Update")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(path, dpi=100)
        plt.close(fig)
        return Path(path)

    def plot_label_temporal_coverage(
        self,
        coverage_sequence: List[float],
        output_path: Optional[Path] = None,
    ) -> Path:
        """绘制 label temporal coverage 可视化。"""
        path = output_path or (self.output_dir / "label_temporal_coverage.png")
        if not HAS_MPL:
            Path(path).touch()
            return Path(path)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(range(len(coverage_sequence)), coverage_sequence)
        ax.set_xlabel("Time Bucket")
        ax.set_ylabel("Label Coverage Ratio")
        ax.set_title("KL Label Temporal Coverage")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(path, dpi=100)
        plt.close(fig)
        return Path(path)
