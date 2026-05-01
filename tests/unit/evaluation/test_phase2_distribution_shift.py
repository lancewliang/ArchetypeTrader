"""Phase II distribution shift / OOD 单元测试。

测试用例:
- OOD score 基于指定 state 维度计算。
- 超阈值触发 fallback。
- OOD 维度与 state_dim_breakdown 对齐。
"""
import pytest

from src.config.phase2_config import DistributionShiftConfig
from src.evaluation.phase2_distribution_shift import Phase2DistributionShiftMonitor


class TestPhase2DistributionShift:

    def test_ood_score_on_specified_dims(self):
        monitor = Phase2DistributionShiftMonitor(
            DistributionShiftConfig(threshold=10.0),
            dims=[0, 2],
        )
        stats = monitor.fit([[1.0, 100.0, 1.0], [3.0, 200.0, 5.0]])
        result = monitor.score([5.0, 999.0, 9.0])
        assert stats.dims == [0, 2]
        assert set(result.per_dim_scores) == {0, 2}
        assert 1 not in result.per_dim_scores
        assert result.score > 0

    def test_threshold_triggers_fallback(self):
        monitor = Phase2DistributionShiftMonitor(
            DistributionShiftConfig(threshold=1.0, fallback_action="flat_only"),
            dims=[0],
        )
        monitor.fit([[0.0], [0.1], [-0.1]])
        result = monitor.score([10.0])
        assert result.triggered is True
        assert result.fallback_action == "flat_only"

    def test_ood_dims_align_with_breakdown(self):
        feature_dim = 3
        position_dim = 1
        dims = list(range(feature_dim))
        monitor = Phase2DistributionShiftMonitor(
            DistributionShiftConfig(use_market_features_only=True),
            dims=dims,
        )
        stats = monitor.fit([
            [0.0, 0.1, 0.2, -1.0],
            [0.2, 0.1, 0.0, 1.0],
        ])
        assert stats.dims == [0, 1, 2]
        assert len(stats.dims) == feature_dim
        assert position_dim == 1
