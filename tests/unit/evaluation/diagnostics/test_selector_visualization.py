"""Selector visualization 单元测试。"""
from __future__ import annotations

from src.evaluation.diagnostics.selector_visualization import SelectorVisualizationWriter


class TestSelectorVisualization:

    def test_cumulative_return_plot(self, tmp_path):
        """生成时间 vs 累计收益 vs archetype 选择图。"""
        writer = SelectorVisualizationWriter(tmp_path)
        path = writer.plot_cumulative_return_with_archetype(
            [
                {"reward_raw": 1.0, "chosen_code": 0},
                {"reward_raw": -0.5, "chosen_code": 1},
            ]
        )
        assert path.exists()
        assert path.name == "cumulative_return_archetype.png"

    def test_label_temporal_coverage_plot(self, tmp_path):
        """生成 label temporal coverage 图。"""
        writer = SelectorVisualizationWriter(tmp_path)
        path = writer.plot_label_temporal_coverage([0.0, 0.5, 1.0])
        assert path.exists()
        assert path.name == "label_temporal_coverage.png"
