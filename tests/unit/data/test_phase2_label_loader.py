"""Phase II label loader 单元测试。

测试用例:
- 只 join train/val 的 code_label。
- 未标注 horizon is_labeled=false。
- kl_label_temporal_coverage 正确聚合。
- label 时间分布熵过低时写入 warning。
- test labels 被请求时抛错。
"""
import pytest

from src.config.phase2_config import Phase2Config
from src.data.phase2_horizon_index import Phase2HorizonEntry
from src.data.phase2_label_loader import (
    Phase2LabelLoader,
    Phase2TestLabelRequestError,
)


@pytest.fixture
def loader():
    config = Phase2Config(pair="TEST", phase1_batch_id="s", phase2_batch_id="s")
    return Phase2LabelLoader(config)


@pytest.fixture
def sample_entries():
    return [
        Phase2HorizonEntry(
            sample_id=f"p2_train_{i:06d}",
            horizon_start=i * 8,
            horizon_end=i * 8 + 7,
            split="train",
        )
        for i in range(10)
    ]


class TestPhase2LabelLoader:

    def test_test_labels_raise_error(self, loader):
        """test labels 被请求时抛 Phase2TestLabelRequestError。"""
        entries = [
            Phase2HorizonEntry(
                sample_id="test_0", horizon_start=0, horizon_end=7, split="test"
            )
        ]
        with pytest.raises(Phase2TestLabelRequestError):
            loader.load_and_join(entries, "test")

    def test_unlabeled_horizon_is_labeled_false(self, loader, sample_entries):
        """未标注 horizon is_labeled=false（无 label 文件时）。"""
        result = loader.load_and_join(sample_entries, "train", None)
        for e in result:
            assert e.is_labeled is False
            assert e.code_label is None

    def test_temporal_coverage_aggregation(self, loader, sample_entries):
        """kl_label_temporal_coverage 正确聚合。"""
        stats = loader.compute_coverage_stats(sample_entries, "train")
        assert stats.total_horizons == 10
        assert stats.labeled_horizons == 0
        assert stats.coverage_ratio == 0.0

    def test_low_entropy_warning(self, loader):
        """label 时间分布熵过低时写入 warning。"""
        # 创建只有前 2 个有 label 的 entries
        entries = [
            Phase2HorizonEntry(
                sample_id=f"e_{i}",
                horizon_start=i * 8,
                horizon_end=i * 8 + 7,
                split="train",
                is_labeled=(i < 2),
                code_label=0 if i < 2 else None,
            )
            for i in range(20)
        ]
        stats = loader.compute_coverage_stats(entries, "train")
        assert stats.labeled_horizons == 2
        # 覆盖率低应有 warning
        assert len(stats.warnings) > 0
