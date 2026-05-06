"""Phase II dataset 单元测试。

测试用例:
- state 维度与 feature_columns + position_encoding + optional extensions 一致。
- position_continuity=true 时 prev_terminal_position 必须进入状态。
- 数据集不重写 Phase I horizon slicing 语义。
- phase2_dataset 不调用 DP。
"""
import json

import pytest
import numpy as np

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

from src.config.phase2_config import Phase2Config
from src.data.phase2_dataset import Phase2Dataset
from src.data.phase2_horizon_index import Phase2HorizonEntry
from src.phase1.data.state_normalizer import StateNormalizer


def _frame(num_rows=8):
    if not HAS_POLARS:
        pytest.skip("polars not installed")
    data = {
        "timestamp": list(range(num_rows)),
        "close": np.linspace(100, 101, num_rows).tolist(),
        "feature_return_1": np.zeros(num_rows).tolist(),
    }
    for i in range(1, 6):
        data[f"ask{i}_price"] = (np.linspace(100, 101, num_rows) + 0.01 * i).tolist()
        data[f"ask{i}_size"] = np.ones(num_rows).tolist()
        data[f"bid{i}_price"] = (np.linspace(100, 101, num_rows) - 0.01 * i).tolist()
        data[f"bid{i}_size"] = np.ones(num_rows).tolist()
    return pl.DataFrame(data)


def _schema():
    return {"feature_columns": ["feature_return_1"], "price_column": "close"}


class TestPhase2Dataset:
    """Phase II dataset 测试。"""

    def test_state_dim_matches_spec(self):
        """state 维度与 feature_columns + position_encoding 一致。"""
        config = Phase2Config(horizon=3, max_position=1)
        entries = [Phase2HorizonEntry("s0", 0, 2, "train")]
        ds = Phase2Dataset(_frame(), entries, _schema(), config)
        assert ds.state_spec().total_dim == 2
        assert ds.get_selector_state(0, 1).shape[0] == 2

    def test_position_continuity_includes_prev_position(self):
        """position_continuity=true 时 prev_terminal_position 进入状态。"""
        config = Phase2Config(horizon=3, max_position=2)
        entries = [Phase2HorizonEntry("s0", 0, 2, "train")]
        ds = Phase2Dataset(_frame(), entries, _schema(), config)
        state = ds.get_selector_state(0, prev_terminal_position=1)
        assert state[-1] == pytest.approx(0.5)

    def test_no_horizon_slicing_rewrite(self):
        """数据集不重写 Phase I horizon slicing 语义。"""
        config = Phase2Config(horizon=3)
        entries = [Phase2HorizonEntry("s0", 1, 3, "train")]
        ds = Phase2Dataset(_frame(), entries, _schema(), config)
        states = ds.get_horizon_states(0)
        assert states.shape == (3, 1)

    def test_no_dp_call(self):
        """phase2_dataset 不调用 DP。"""
        config = Phase2Config(horizon=3)
        entries = [Phase2HorizonEntry("s0", 0, 2, "train")]
        ds = Phase2Dataset(_frame(), entries, _schema(), config)
        assert len(ds) == 1

    def test_get_horizon_inputs_raises_on_execution_row_overflow(self):
        """execution_row 越界时不再 silent fallback 到最后一行。"""
        config = Phase2Config(horizon=3)
        entries = [Phase2HorizonEntry("s0", 3, 5, "train")]
        ds = Phase2Dataset(
            _frame(6),
            entries,
            _schema(),
            config,
            reward_alignment="next_row_execution",
        )
        with pytest.raises(IndexError):
            ds.get_horizon_inputs(0)

    def test_applies_phase1_state_normalizer(self, tmp_path):
        """Phase II 必须复用 Phase I 持久化的 state normalizer。"""
        config = Phase2Config(
            pair="TEST",
            phase1_batch_id="batch_001",
            artifact_root=str(tmp_path),
            horizon=3,
        )
        p1_dir = config.phase1_dir()
        p1_dir.mkdir(parents=True)
        normalizer = StateNormalizer.fit_matrix(
            np.array([[0.0], [10.0], [20.0], [30.0]], dtype="float32"),
            feature_columns=["feature_return_1"],
        )
        (p1_dir / "state_normalizer.json").write_text(
            json.dumps(normalizer.to_dict()),
            encoding="utf-8",
        )

        entries = [Phase2HorizonEntry("s0", 0, 2, "train")]
        ds = Phase2Dataset(_frame(), entries, _schema(), config)

        raw = _frame()["feature_return_1"].to_numpy()[:3].reshape(-1, 1)
        np.testing.assert_allclose(
            ds.get_horizon_states(0),
            normalizer.transform_array(raw),
        )
