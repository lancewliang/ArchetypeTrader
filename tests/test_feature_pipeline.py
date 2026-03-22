"""Feature Pipeline 单元测试

测试特征加载、维度校验、拼接和 horizon 切分。
"""

import os
import tempfile

import polars as pl
import pyarrow.feather as pa_feather
import pytest

from src.data.feature_pipeline import FeaturePipeline, SINGLE_FEATURES, TREND_FEATURES


class TestGetStateVector:
    """测试 get_state_vector 方法"""

    @pytest.fixture
    def tmp_feather_dir(self):
        """创建临时目录，包含有效的 feather 文件。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_cols = SINGLE_FEATURES + TREND_FEATURES + ["close"]
            n_rows = 1000
            data = {col: list(range(n_rows)) for col in feature_cols}
            train_df = pl.DataFrame(data)
            val_df = pl.DataFrame({col: list(range(1000, 1500)) for col in feature_cols})
            test_df = pl.DataFrame({col: list(range(1500, 2000)) for col in feature_cols})

            pa_feather.write_feather(train_df.to_arrow(), f"{tmpdir}/df_train.feather")
            pa_feather.write_feather(val_df.to_arrow(), f"{tmpdir}/df_val.feather")
            pa_feather.write_feather(test_df.to_arrow(), f"{tmpdir}/df_test.feather")

            yield tmpdir, train_df, val_df, test_df

    def test_loads_three_datasets(self, tmp_feather_dir):
        tmpdir, _, _, _ = tmp_feather_dir
        fp = FeaturePipeline(tmpdir, "ETH")
        train, val, test = fp.get_state_vector()

        assert train.shape == (1000, 45)
        assert val.shape == (500, 45)
        assert test.shape == (500, 45)

    def test_returns_polars_dataframes(self, tmp_feather_dir):
        tmpdir, _, _, _ = tmp_feather_dir
        fp = FeaturePipeline(tmpdir, "ETH")
        train, val, test = fp.get_state_vector()

        assert isinstance(train, pl.DataFrame)
        assert isinstance(val, pl.DataFrame)
        assert isinstance(test, pl.DataFrame)

    def test_data_correctness(self, tmp_feather_dir):
        tmpdir, _, _, _ = tmp_feather_dir
        fp = FeaturePipeline(tmpdir, "ETH")
        train, val, test = fp.get_state_vector()

        assert train[SINGLE_FEATURES[0]][0] == 0
        assert val[SINGLE_FEATURES[0]][0] == 1000
        assert test[SINGLE_FEATURES[0]][0] == 1500

    def test_train_largest(self, tmp_feather_dir):
        tmpdir, _, _, _ = tmp_feather_dir
        fp = FeaturePipeline(tmpdir, "ETH")
        train, val, test = fp.get_state_vector()

        assert train.height > val.height
        assert train.height > test.height


class TestSplitIntoHorizons:
    """测试 split_into_horizons 方法"""

    @pytest.fixture
    def sample_df(self):
        return pl.DataFrame({"feature": range(200)})

    def test_exact_division(self, sample_df):
        fp = FeaturePipeline(".", "ETH")
        horizons = fp.split_into_horizons(sample_df, h=72)
        assert len(horizons) == 3
        assert horizons[0].height == 72
        assert horizons[1].height == 72
        assert horizons[2].height == 56

    def test_with_remainder(self):
        fp = FeaturePipeline(".", "ETH")
        df = pl.DataFrame({"feature": range(200)})
        horizons = fp.split_into_horizons(df, h=72)
        assert len(horizons) == 3
        assert horizons[0].height == 72
        assert horizons[1].height == 72
        assert horizons[2].height == 56

    def test_invalid_horizon(self):
        fp = FeaturePipeline(".", "ETH")
        df = pl.DataFrame({"feature": range(100)})
        with pytest.raises(ValueError, match="正整数"):
            fp.split_into_horizons(df, h=0)

    def test_default_horizon(self, sample_df):
        fp = FeaturePipeline(".", "ETH")
        horizons = fp.split_into_horizons(sample_df)
        assert len(horizons) == 3


class TestWithActualData:
    """使用实际 data/ETH/ 数据的集成测试"""

    @pytest.mark.skipif(
        not os.path.exists("data/ETH/df_train.feather"),
        reason="实际 feather 文件不存在",
    )
    def test_load_actual_data(self):
        fp = FeaturePipeline("data/ETH", "ETH")
        train, val, test = fp.get_state_vector()

        assert isinstance(train, pl.DataFrame)
        assert isinstance(val, pl.DataFrame)
        assert isinstance(test, pl.DataFrame)
        assert train.height > 0
        assert val.height > 0
        assert test.height > 0


class TestGetPrices:
    """测试 get_prices 方法"""

    @pytest.mark.skipif(
        not os.path.exists("data/ETH/df_train.feather"),
        reason="实际 feather 文件不存在",
    )
    def test_get_prices_returns_three_datasets(self):
        fp = FeaturePipeline("data/ETH", "ETH")
        train_p, val_p, test_p = fp.get_prices()

        assert "close" in train_p.columns
        assert "close" in val_p.columns
        assert "close" in test_p.columns
        assert train_p.height > 0
        assert val_p.height > 0
        assert test_p.height > 0
