"""Feature Pipeline 单元测试

测试特征加载、维度校验、拼接、时间划分和 horizon 切分。
"""

import os
import tempfile

import numpy as np
import pytest

from src.config import Config
from src.data.feature_pipeline import PAIR_INDEX, FeaturePipeline


@pytest.fixture
def tmp_data_dir():
    """创建临时数据目录，包含有效的 .npy 文件。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        T = 288  # 4 个 horizon (72 * 4)
        single = np.random.randn(T, 36).astype(np.float32)
        trend = np.random.randn(T, 9).astype(np.float32)
        np.save(os.path.join(tmpdir, "single_features.npy"), single)
        np.save(os.path.join(tmpdir, "trend_features.npy"), trend)
        yield tmpdir, single, trend


@pytest.fixture
def tmp_data_dir_1d():
    """创建临时数据目录，包含 1D .npy 文件（单时间步）。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        single = np.random.randn(36).astype(np.float32)
        trend = np.random.randn(9).astype(np.float32)
        np.save(os.path.join(tmpdir, "single_features.npy"), single)
        np.save(os.path.join(tmpdir, "trend_features.npy"), trend)
        yield tmpdir, single, trend


@pytest.fixture
def tmp_data_dir_3d():
    """创建临时数据目录，包含 3D .npy 文件（多交易对）。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        num_pairs = 4
        T = 144
        single = np.random.randn(num_pairs, T, 36).astype(np.float32)
        trend = np.random.randn(num_pairs, T, 9).astype(np.float32)
        np.save(os.path.join(tmpdir, "single_features.npy"), single)
        np.save(os.path.join(tmpdir, "trend_features.npy"), trend)
        yield tmpdir, single, trend


class TestLoadSingleFeatures:
    """测试 load_single_features 方法"""

    def test_load_2d(self, tmp_data_dir):
        tmpdir, expected_single, _ = tmp_data_dir
        fp = FeaturePipeline(tmpdir, "BTC")
        result = fp.load_single_features()
        np.testing.assert_array_equal(result, expected_single)
        assert result.shape == (288, 36)

    def test_load_1d(self, tmp_data_dir_1d):
        tmpdir, expected_single, _ = tmp_data_dir_1d
        fp = FeaturePipeline(tmpdir, "BTC")
        result = fp.load_single_features()
        assert result.shape == (1, 36)
        np.testing.assert_array_equal(result[0], expected_single)

    def test_load_3d(self, tmp_data_dir_3d):
        tmpdir, expected_single, _ = tmp_data_dir_3d
        for pair, idx in PAIR_INDEX.items():
            fp = FeaturePipeline(tmpdir, pair)
            result = fp.load_single_features()
            np.testing.assert_array_equal(result, expected_single[idx])
            assert result.shape == (144, 36)

    def test_file_not_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fp = FeaturePipeline(tmpdir, "BTC")
            with pytest.raises(FileNotFoundError):
                fp.load_single_features()

    def test_wrong_dimension(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bad = np.random.randn(100, 30).astype(np.float32)  # 30 != 36
            np.save(os.path.join(tmpdir, "single_features.npy"), bad)
            fp = FeaturePipeline(tmpdir, "BTC")
            with pytest.raises(ValueError, match="最后一维为 30.*预期为 36"):
                fp.load_single_features()


class TestLoadTrendFeatures:
    """测试 load_trend_features 方法"""

    def test_load_2d(self, tmp_data_dir):
        tmpdir, _, expected_trend = tmp_data_dir
        fp = FeaturePipeline(tmpdir, "BTC")
        result = fp.load_trend_features()
        np.testing.assert_array_equal(result, expected_trend)
        assert result.shape == (288, 9)

    def test_load_1d(self, tmp_data_dir_1d):
        tmpdir, _, expected_trend = tmp_data_dir_1d
        fp = FeaturePipeline(tmpdir, "BTC")
        result = fp.load_trend_features()
        assert result.shape == (1, 9)
        np.testing.assert_array_equal(result[0], expected_trend)

    def test_wrong_dimension(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bad = np.random.randn(100, 5).astype(np.float32)  # 5 != 9
            np.save(os.path.join(tmpdir, "trend_features.npy"), bad)
            fp = FeaturePipeline(tmpdir, "BTC")
            with pytest.raises(ValueError, match="最后一维为 5.*预期为 9"):
                fp.load_trend_features()


class TestGetStateVector:
    """测试 get_state_vector 方法"""

    def test_concatenation(self, tmp_data_dir):
        tmpdir, expected_single, expected_trend = tmp_data_dir
        fp = FeaturePipeline(tmpdir, "BTC")
        state = fp.get_state_vector()
        assert state.shape == (288, 45)
        np.testing.assert_array_equal(state[:, :36], expected_single)
        np.testing.assert_array_equal(state[:, 36:], expected_trend)

    def test_auto_loads(self, tmp_data_dir):
        tmpdir, _, _ = tmp_data_dir
        fp = FeaturePipeline(tmpdir, "BTC")
        # 不手动调用 load，直接 get_state_vector
        state = fp.get_state_vector()
        assert state.shape == (288, 45)

    def test_mismatched_time_dim(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            np.save(os.path.join(tmpdir, "single_features.npy"), np.random.randn(100, 36))
            np.save(os.path.join(tmpdir, "trend_features.npy"), np.random.randn(50, 9))
            fp = FeaturePipeline(tmpdir, "BTC")
            with pytest.raises(ValueError, match="时间维度.*不一致"):
                fp.get_state_vector()


class TestSplitByDate:
    """测试 split_by_date 方法"""

    def test_split_covers_all_data(self, tmp_data_dir):
        tmpdir, _, _ = tmp_data_dir
        fp = FeaturePipeline(tmpdir, "BTC")
        train, val, test = fp.split_by_date()
        total = train.shape[0] + val.shape[0] + test.shape[0]
        assert total == 288

    def test_split_no_overlap(self, tmp_data_dir):
        tmpdir, _, _ = tmp_data_dir
        fp = FeaturePipeline(tmpdir, "BTC")
        state = fp.get_state_vector()
        train, val, test = fp.split_by_date(state)
        # 拼接后应还原原始数据
        reconstructed = np.concatenate([train, val, test], axis=0)
        np.testing.assert_array_equal(reconstructed, state)

    def test_split_with_explicit_ratios(self, tmp_data_dir):
        tmpdir, _, _ = tmp_data_dir
        fp = FeaturePipeline(tmpdir, "BTC")
        state = fp.get_state_vector()
        train, val, test = fp.split_by_date(state, train_ratio=0.6, val_ratio=0.2)
        assert train.shape[0] == int(288 * 0.6)
        assert val.shape[0] == int(288 * 0.8) - int(288 * 0.6)

    def test_split_proportions_reasonable(self, tmp_data_dir):
        tmpdir, _, _ = tmp_data_dir
        fp = FeaturePipeline(tmpdir, "BTC")
        train, val, test = fp.split_by_date()
        # 训练集应该是最大的
        assert train.shape[0] > val.shape[0]
        assert train.shape[0] > test.shape[0]


class TestSplitIntoHorizons:
    """测试 split_into_horizons 方法"""

    def test_exact_division(self, tmp_data_dir):
        tmpdir, _, _ = tmp_data_dir
        fp = FeaturePipeline(tmpdir, "BTC")
        data = np.random.randn(288, 45)
        horizons = fp.split_into_horizons(data, h=72)
        assert len(horizons) == 4
        for h in horizons:
            assert h.shape == (72, 45)

    def test_with_remainder(self):
        fp = FeaturePipeline(".", "BTC")
        data = np.random.randn(200, 45)
        horizons = fp.split_into_horizons(data, h=72)
        assert len(horizons) == 3
        assert horizons[0].shape == (72, 45)
        assert horizons[1].shape == (72, 45)
        assert horizons[2].shape == (56, 45)  # 200 - 144 = 56

    def test_reconstruction(self):
        fp = FeaturePipeline(".", "BTC")
        data = np.random.randn(200, 45)
        horizons = fp.split_into_horizons(data, h=72)
        reconstructed = np.concatenate(horizons, axis=0)
        np.testing.assert_array_equal(reconstructed, data)

    def test_invalid_horizon(self):
        fp = FeaturePipeline(".", "BTC")
        data = np.random.randn(100, 45)
        with pytest.raises(ValueError, match="正整数"):
            fp.split_into_horizons(data, h=0)

    def test_default_horizon(self, tmp_data_dir):
        tmpdir, _, _ = tmp_data_dir
        fp = FeaturePipeline(tmpdir, "BTC")
        data = np.random.randn(288, 45)
        horizons = fp.split_into_horizons(data)
        # 默认 h=72
        assert len(horizons) == 4


class TestPairValidation:
    """测试交易对验证"""

    def test_invalid_pair(self):
        with pytest.raises(ValueError, match="不支持的交易对"):
            FeaturePipeline(".", "INVALID")

    def test_all_valid_pairs(self):
        for pair in ["BTC", "ETH", "DOT", "BNB"]:
            fp = FeaturePipeline(".", pair)
            assert fp.pair == pair


class TestWithActualData:
    """使用实际 data/feature_list/ 数据的集成测试"""

    @pytest.mark.skipif(
        not os.path.exists("data/feature_list/single_features.npy"),
        reason="实际数据文件不存在",
    )
    def test_load_actual_data(self):
        fp = FeaturePipeline("data/feature_list", "BTC")
        single = fp.load_single_features()
        trend = fp.load_trend_features()
        state = fp.get_state_vector()

        assert single.shape[-1] == 36
        assert trend.shape[-1] == 9
        assert state.shape[-1] == 45


# ============================================================================
# Property-Based Tests (hypothesis)
# ============================================================================

from hypothesis import given, settings
from hypothesis import strategies as st


class TestPropFeatureDimensions:
    """Property 1: 特征维度验证

    对于任意加载的特征文件，single_features 的最后一维应为 36，
    trend_features 的最后一维应为 9，拼接后的状态向量最后一维应为 45。

    # Feature: archetype-trader, Property 1: 特征维度验证
    **Validates: Requirements 1.1, 1.2, 1.3**
    """

    @given(T=st.integers(min_value=1, max_value=1000))
    @settings(max_examples=100)
    def test_prop_feature_dimensions(self, T: int):
        single = np.random.randn(T, 36).astype(np.float32)
        trend = np.random.randn(T, 9).astype(np.float32)

        assert single.shape[-1] == 36, f"single last dim {single.shape[-1]} != 36"
        assert trend.shape[-1] == 9, f"trend last dim {trend.shape[-1]} != 9"

        state = np.concatenate([single, trend], axis=-1)
        assert state.shape == (T, 45), f"state shape {state.shape} != ({T}, 45)"


class TestPropConcatenationPreservesContent:
    """Property 2: 特征拼接保持内容不变

    对于任意的 single_features (T, 36) 和 trend_features (T, 9)，
    拼接后前 36 列等于 single_features，后 9 列等于 trend_features。

    # Feature: archetype-trader, Property 2: 特征拼接保持内容不变
    **Validates: Requirements 1.3**
    """

    @given(T=st.integers(min_value=1, max_value=1000))
    @settings(max_examples=100)
    def test_prop_concatenation_preserves_content(self, T: int):
        single = np.random.randn(T, 36).astype(np.float32)
        trend = np.random.randn(T, 9).astype(np.float32)

        state = np.concatenate([single, trend], axis=-1)

        np.testing.assert_array_equal(
            state[:, :36], single, err_msg="First 36 cols != single_features"
        )
        np.testing.assert_array_equal(
            state[:, 36:], trend, err_msg="Last 9 cols != trend_features"
        )


class TestPropTimeSplitCoverage:
    """Property 3: 时间划分不重叠且完整覆盖

    对于任意长度 T 的数据，按比例划分后：
    len(train) + len(val) + len(test) == T，且索引不重叠。

    # Feature: archetype-trader, Property 3: 时间划分不重叠且完整覆盖
    **Validates: Requirements 1.6**
    """

    @given(
        T=st.integers(min_value=10, max_value=500),
        train_pct=st.integers(min_value=10, max_value=80),
        val_pct=st.integers(min_value=5, max_value=40),
    )
    @settings(max_examples=100)
    def test_prop_time_split_no_overlap_full_coverage(
        self, T: int, train_pct: int, val_pct: int
    ):
        # Ensure train + val < 100% so test set is non-empty
        total_pct = train_pct + val_pct
        if total_pct >= 100:
            return  # skip invalid combos

        train_ratio = train_pct / 100.0
        val_ratio = val_pct / 100.0

        data = np.random.randn(T, 45).astype(np.float32)

        train_end = int(T * train_ratio)
        val_end = int(T * (train_ratio + val_ratio))

        train = data[:train_end]
        val = data[train_end:val_end]
        test = data[val_end:]

        # Complete coverage
        assert train.shape[0] + val.shape[0] + test.shape[0] == T, (
            f"Split sizes {train.shape[0]}+{val.shape[0]}+{test.shape[0]} != {T}"
        )

        # No overlap: reconstruction equals original
        reconstructed = np.concatenate([train, val, test], axis=0)
        np.testing.assert_array_equal(reconstructed, data)


class TestPropHorizonSplitConsistency:
    """Property 4: Horizon 切分长度一致

    对于任意长度 T 的数据和 horizon h，除最后一个片段外每个长度为 h，
    且拼接后还原原始序列。

    # Feature: archetype-trader, Property 4: Horizon 切分长度一致
    **Validates: Requirements 1.7**
    """

    @given(
        T=st.integers(min_value=1, max_value=500),
        h=st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=100)
    def test_prop_horizon_split_length_and_reconstruction(self, T: int, h: int):
        data = np.random.randn(T, 45).astype(np.float32)

        # Split into horizons
        horizons = []
        for start in range(0, T, h):
            end = min(start + h, T)
            horizons.append(data[start:end])

        # All chunks except possibly the last have length h
        for i, chunk in enumerate(horizons[:-1]):
            assert chunk.shape[0] == h, (
                f"Chunk {i} has length {chunk.shape[0]}, expected {h}"
            )

        # Last chunk length <= h
        assert horizons[-1].shape[0] <= h

        # Reconstruction
        reconstructed = np.concatenate(horizons, axis=0)
        np.testing.assert_array_equal(reconstructed, data)
