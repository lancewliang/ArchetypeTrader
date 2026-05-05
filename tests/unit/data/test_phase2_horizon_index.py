"""Phase II horizon index 单元测试。

测试用例:
- non_overlap 生成的 horizons 不重叠。
- stride 模式按给定 stride 生效。
- phase1_index 模式下 phase1_sample_id 与 sample_id 对齐。
- markout 越界 horizon 被裁掉。
- gap horizon 被标记并按配置裁掉。
- test index 默认不包含 code_label。
"""
import pytest
import numpy as np
from dataclasses import replace

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

from src.config.phase2_config import Phase2Config, HorizonScheduleConfig
from src.data.phase2_horizon_index import (
    Phase1ArtifactValidator,
    Phase1ArtifactValidationError,
    Phase2HorizonIndexer,
)


def _make_frame(num_rows: int):
    """创建最小 market frame。"""
    if not HAS_POLARS:
        pytest.skip("polars not installed")
    rng = np.random.RandomState(42)
    data = {
        "timestamp": list(range(num_rows)),
        "close": (100 + rng.randn(num_rows).cumsum()).tolist(),
    }
    for i in range(1, 6):
        data[f"ask{i}_price"] = (np.array(data["close"]) + 0.1 * i).tolist()
        data[f"ask{i}_size"] = rng.uniform(10, 100, num_rows).tolist()
        data[f"bid{i}_price"] = (np.array(data["close"]) - 0.1 * i).tolist()
        data[f"bid{i}_size"] = rng.uniform(10, 100, num_rows).tolist()
    data["feature_return_1"] = rng.randn(num_rows).tolist()
    data["feature_vol_4"] = rng.uniform(0, 1, num_rows).tolist()
    data["feature_momentum_8"] = rng.randn(num_rows).tolist()
    return pl.DataFrame(data)


def _make_config(tmp_path, horizon=8, mode="non_overlap", stride=1, exclude_gap=True):
    """创建最小 Phase2Config，带 smoke Phase I 产物。"""
    # 创建最小 Phase I 产物
    p1_dir = tmp_path / "artifacts" / "TEST" / "smoke" / "phase1"
    p1_dir.mkdir(parents=True, exist_ok=True)
    import json, yaml
    (p1_dir / "decoder.pt").touch()
    (p1_dir / "codebook.pt").touch()
    (p1_dir / "input_schema.json").write_text(json.dumps({
        "feature_columns": ["feature_return_1", "feature_vol_4", "feature_momentum_8"],
        "price_column": "close",
    }))
    (p1_dir / "state_normalizer.json").write_text(json.dumps({
        "method": "train_state_robust_v1",
        "feature_columns": ["feature_return_1", "feature_vol_4", "feature_momentum_8"],
        "transform_kinds": ["identity", "identity", "identity"],
        "center": [0.0, 0.0, 0.0],
        "scale": [1.0, 1.0, 1.0],
        "clip_value": 8.0,
        "scale_floor": 1.0e-6,
        "max_abs_before": 1.0,
        "max_abs_after_fit": 1.0,
        "fallback_to_standard_count": 0,
    }))
    (p1_dir / "phase1_report.json").write_text(json.dumps({
        "fatal_collapse": False,
        "code_assignment_drift_warning": False,
        "hindsight_bias_warning": "ok",
    }))
    (p1_dir / "phase1_config.yaml").write_text(yaml.safe_dump({
        "dp": {"cost_config": {"reward_alignment": "paper_formula"}},
    }))
    (p1_dir / "feature_provenance.json").write_text("{}")
    (p1_dir / "checkpoint_manifest.json").write_text("[]")

    return Phase2Config(
        pair="TEST",
        phase1_batch_id="smoke",
        phase2_batch_id="test_p2",
        artifact_root=str(tmp_path / "artifacts"),
        horizon=horizon,
        horizon_schedule=HorizonScheduleConfig(
            mode=mode,
            stride=stride,
            exclude_gap_horizons=exclude_gap,
        ),
    )


@pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
class TestPhase2HorizonIndexer:

    def test_non_overlap_no_overlap(self, tmp_path):
        """non_overlap 模式生成的 horizons 不重叠。"""
        config = _make_config(tmp_path, horizon=8)
        indexer = Phase2HorizonIndexer(config)
        frame = _make_frame(96)
        entries = indexer.build_index(frame, "train", 8)
        # 检查不重叠
        for i in range(1, len(entries)):
            assert entries[i].horizon_start >= entries[i - 1].horizon_end + 1

    def test_stride_mode(self, tmp_path):
        """stride 模式按给定 stride 生效。"""
        config = _make_config(tmp_path, horizon=8, mode="stride", stride=4)
        indexer = Phase2HorizonIndexer(config)
        frame = _make_frame(96)
        entries = indexer.build_index(frame, "train", 8)
        if len(entries) > 1:
            assert entries[1].horizon_start - entries[0].horizon_start == 4

    def test_markout_overflow_trimmed(self, tmp_path):
        """markout 越界 horizon 被裁掉。"""
        config = _make_config(tmp_path, horizon=8)
        indexer = Phase2HorizonIndexer(config)
        frame = _make_frame(20)  # 很短的数据
        entries = indexer.build_index(frame, "train", 8)
        for e in entries:
            # horizon_end + lookahead 不应超过 frame 行数
            assert e.horizon_end < 20

    def test_test_index_no_code_label(self, tmp_path):
        """test index 默认不包含 code_label。"""
        config = _make_config(tmp_path, horizon=8)
        indexer = Phase2HorizonIndexer(config)
        frame = _make_frame(48)
        entries = indexer.build_index(frame, "test", 8)
        for e in entries:
            assert e.code_label is None
            assert e.is_labeled is False

    def test_write_index(self, tmp_path):
        """write_index 生成 feather 文件。"""
        config = _make_config(tmp_path, horizon=8)
        indexer = Phase2HorizonIndexer(config)
        frame = _make_frame(96)
        entries = indexer.build_index(frame, "train", 8)
        path = indexer.write_index(entries, tmp_path / "test_index.feather")
        assert path.exists()
        df = pl.read_ipc(path)
        assert "sample_id" in df.columns
        assert "horizon_start" in df.columns

    def test_cost_alignment_mismatch_rejects(self, tmp_path):
        """Phase I/II max_position 不一致时 fail-fast。"""
        config = _make_config(tmp_path, horizon=8)
        import yaml
        p1_cfg = config.phase1_dir() / "phase1_config.yaml"
        p1_cfg.write_text(yaml.safe_dump({
            "dp": {
                "max_position": 2,
                "cost_config": {"reward_alignment": "paper_formula"},
            }
        }))
        with pytest.raises(Phase1ArtifactValidationError):
            Phase1ArtifactValidator(config).validate()

    def test_missing_feature_provenance_warns_without_blocking_experiment(self, tmp_path):
        """缺少 provenance 时允许实验运行，但 no-leakage signoff=false。"""
        config = _make_config(tmp_path, horizon=8)
        (config.phase1_dir() / "feature_provenance.json").unlink()

        result = Phase1ArtifactValidator(config).validate()

        assert result.valid is True
        assert result.no_leakage_signoff is False
        assert result.no_leakage_signoff_blockers

    def test_missing_state_normalizer_rejects(self, tmp_path):
        """Phase II 必须拒绝缺少 Phase I state normalizer 的产物。"""
        config = _make_config(tmp_path, horizon=8)
        (config.phase1_dir() / "state_normalizer.json").unlink()

        with pytest.raises(Phase1ArtifactValidationError):
            Phase1ArtifactValidator(config).validate()

    def test_gap_bars_uses_bar_units_when_gap_check_disabled(self, tmp_path):
        """gap_bars 记录 bar 数，不把分钟阈值和 bar 阈值混用。"""
        config = _make_config(tmp_path, horizon=4, exclude_gap=False)
        config = replace(
            config,
            horizon_schedule=replace(
                config.horizon_schedule,
                data_gap_check_enabled=False,
                gap_threshold_bars=2,
                exclude_gap_horizons=False,
            ),
        )
        frame = _make_frame(32)
        timestamps = [i * 5 for i in range(32)]
        timestamps[3] = 30  # 10 -> 30 means 20 minutes, with 5-minute bars.
        for i in range(4, 32):
            timestamps[i] = timestamps[i - 1] + 5
        frame = frame.with_columns(pl.Series("timestamp", timestamps))
        entries = Phase2HorizonIndexer(config).build_index(frame, "train", 4)
        assert entries[0].max_timestamp_gap_minutes == pytest.approx(20.0)
        assert entries[0].gap_bars == 3
        assert entries[0].is_gap is True
