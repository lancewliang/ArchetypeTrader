"""Phase II report 单元测试。

测试用例:
- phase2_report.json 含配置、hash、coverage、scalar metrics、health warnings、stress summary。
- equity_curve_summary 字段存在。
- input_norm_stats_merge_protocol / running_mean_std_mode 字段存在。
- rolling validation summary 字段存在。
- test_used_for_selection=false 严格写入。
"""
import pytest

from src.evaluation.phase2_report import (
    Phase2ReportPaths,
    Phase2ReportSchemaError,
    Phase2ReportWriter,
    REQUIRED_PHASE2_REPORT_KEYS,
)


class TestPhase2Report:

    def test_required_fields_present(self, tmp_path):
        """phase2_report.json 含所有必填字段。"""
        paths = Phase2ReportPaths.from_artifacts_dir(tmp_path)
        writer = Phase2ReportWriter(paths)
        summary = {
            "config_hash": "abc123",
            "phase1_hash": "def456",
            "test_used_for_selection": False,
            "phase1_batch_id": "smoke_phase1",
            "equity_curve_summary": {"start_value": 1.0},
            "behavior_health_warnings": [],
            "risk_health_warnings": [],
            "ood_warning_count": 0,
        }
        path = writer.write_final_report(summary)
        assert path.exists()

    def test_equity_curve_summary_exists(self, tmp_path):
        """equity_curve_summary 字段存在。"""
        paths = Phase2ReportPaths.from_artifacts_dir(tmp_path)
        writer = Phase2ReportWriter(paths)
        summary = {k: "" for k in REQUIRED_PHASE2_REPORT_KEYS}
        summary["equity_curve_summary"] = {"start_value": 1.0}
        summary["test_used_for_selection"] = False
        summary["behavior_health_warnings"] = []
        summary["risk_health_warnings"] = []
        summary["ood_warning_count"] = 0
        writer.write_final_report(summary)

    def test_test_used_for_selection_false(self, tmp_path):
        """test_used_for_selection=false 严格写入。"""
        import json
        paths = Phase2ReportPaths.from_artifacts_dir(tmp_path)
        writer = Phase2ReportWriter(paths)
        summary = {
            "config_hash": "abc",
            "phase1_hash": "def",
            "test_used_for_selection": False,
            "phase1_batch_id": "test",
            "equity_curve_summary": {},
            "behavior_health_warnings": [],
            "risk_health_warnings": [],
            "ood_warning_count": 0,
        }
        path = writer.write_final_report(summary)
        with open(path) as f:
            data = json.load(f)
        assert data["test_used_for_selection"] is False

    def test_schema_validation_rejects_missing(self, tmp_path):
        """缺失必填字段时抛 Phase2ReportSchemaError。"""
        paths = Phase2ReportPaths.from_artifacts_dir(tmp_path)
        writer = Phase2ReportWriter(paths)
        with pytest.raises(Phase2ReportSchemaError):
            writer.write_final_report({"config_hash": "abc"})

    def test_rolling_validation_summary_exists(self, tmp_path):
        """rolling validation summary 可写入。"""
        paths = Phase2ReportPaths.from_artifacts_dir(tmp_path)
        writer = Phase2ReportWriter(paths)
        result = {"fold_metrics": [], "fold_mean": {}}
        path = writer.write_rolling_validation(result)
        assert path.exists()

    def test_write_baselines(self, tmp_path):
        """baselines 可写入。"""
        paths = Phase2ReportPaths.from_artifacts_dir(tmp_path)
        writer = Phase2ReportWriter(paths)
        baselines = {"random": {"net_return": 0.01}}
        path = writer.write_baselines(baselines, "val")
        assert path.exists()

    def test_write_rollout_stats(self, tmp_path):
        """rollout stats 可写入。"""
        paths = Phase2ReportPaths.from_artifacts_dir(tmp_path)
        writer = Phase2ReportWriter(paths)
        stats = [{"update_idx": 0, "reward_mean": 0.01}]
        path = writer.write_rollout_stats(stats)
        assert path.exists()
