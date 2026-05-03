"""Phase II report 写入: phase2_report.json / baselines / sensitivity / rolling validation。

设计文档锚点: Phase II 执行计划 §Step 7。

职责:
- 写出 phase2_report.json / phase2_baselines_{val,test}.json /
  composite_score_sensitivity_phase2.json / phase2_rolling_validation.json。
- phase2_report.json 至少包含: 配置 hash / Phase I hash / schema hash /
  horizon 覆盖 / label 覆盖 / PPO 健康 / train/val scalar 指标 /
  equity_curve_summary / behavior_health_warnings / risk_health_warnings /
  ood_warning_count / rolling validation summary 等。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl

from src.config.phase2_config import Phase2Config
from src.utils.feather_io import atomic_write_json, write_ipc


REQUIRED_PHASE2_REPORT_KEYS = (
    "config_hash",
    "phase1_hash",
    "test_used_for_selection",
    "phase1_batch_id",
    "equity_curve_summary",
    "behavior_health_warnings",
    "risk_health_warnings",
    "ood_warning_count",
)

PHASE2_AUDIT_REPORT_KEYS = (
    "horizon_schedule",
    "data_gap_filter",
    "input_norm",
    "env_shards",
    "reward_scaling",
    "cost_config_inherited",
    "baselines_val",
    "baselines_test",
    "rolling_validation_summary",
    "execution_stress_summary",
    "distribution_shift_warning_count",
    "resume_ready",
    "guardrails_pass",
    "val_guardrails_pass",
    "test_guardrails_pass_report_only",
)


class Phase2ReportSchemaError(ValueError):
    """phase2_report.json 缺失必填字段。"""


@dataclass
class Phase2ReportPaths:
    """Phase II report 文件路径集合。"""
    artifacts_dir: Path
    phase2_report: Path
    baselines_val: Path
    baselines_test: Path
    sensitivity: Path
    rolling_validation: Path
    rolling_validation_records: Path
    rollout_stats: Path
    ablation_kl_demo: Path
    ablation_summary_csv: Path

    @classmethod
    def from_artifacts_dir(cls, artifacts_dir: Path) -> "Phase2ReportPaths":
        d = Path(artifacts_dir)
        return cls(
            artifacts_dir=d,
            phase2_report=d / "phase2_report.json",
            baselines_val=d / "phase2_baselines_val.json",
            baselines_test=d / "phase2_baselines_test.json",
            sensitivity=d / "composite_score_sensitivity_phase2.json",
            rolling_validation=d / "phase2_rolling_validation.json",
            rolling_validation_records=d / "phase2_rolling_validation_records.feather",
            rollout_stats=d / "phase2_rollout_stats.feather",
            ablation_kl_demo=d / "phase2_ablation_kl_demo.json",
            ablation_summary_csv=d / "phase2_ablation_summary.csv",
        )


class Phase2ReportWriter:
    """Phase II report 写入器。

    边界:
    - 只负责序列化与 schema 校验，不重新计算指标。
    """

    def __init__(self, paths: Phase2ReportPaths) -> None:
        self.paths = paths
        self.paths.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def write_final_report(self, summary: Dict[str, Any]) -> Path:
        """写 phase2_report.json。"""
        self.validate_schema(summary)
        return atomic_write_json(summary, self.paths.phase2_report)

    def write_baselines(
        self, baselines: Dict[str, Any], split: str
    ) -> Path:
        """写 phase2_baselines_{split}.json。"""
        path = self.paths.baselines_val if split == "val" else self.paths.baselines_test
        return atomic_write_json(baselines, path)

    def write_sensitivity(self, sensitivity: Dict[str, Any]) -> Path:
        """写 composite_score_sensitivity_phase2.json。"""
        return atomic_write_json(sensitivity, self.paths.sensitivity)

    def write_rolling_validation(
        self,
        result: Dict[str, Any],
        records: Optional[List[Dict[str, Any]]] = None,
    ) -> Path:
        """写 phase2_rolling_validation.json，并可选写 per-fold records。"""
        path = atomic_write_json(result, self.paths.rolling_validation)
        if records is not None:
            if records:
                flat_records = [
                    {k: v for k, v in r.items() if not isinstance(v, (list, dict))}
                    for r in records
                ]
                df = pl.DataFrame(flat_records)
            else:
                df = pl.DataFrame({"sample_id": [], "fold_id": []})
            write_ipc(df, self.paths.rolling_validation_records)
        return path

    def write_rollout_stats(self, stats_records: List[Dict[str, Any]]) -> Path:
        """写 phase2_rollout_stats.feather。"""
        if not stats_records:
            df = pl.DataFrame({"update_idx": [], "reward_mean": []})
        else:
            df = pl.DataFrame(stats_records)
        return write_ipc(df, self.paths.rollout_stats)

    def write_per_horizon_records(
        self, records: List[Dict[str, Any]], split: str
    ) -> Path:
        """写 phase2_per_horizon_records_{split}.feather。"""
        path = self.paths.artifacts_dir / f"phase2_per_horizon_records_{split}.feather"
        if not records:
            df = pl.DataFrame({"sample_id": []})
        else:
            # 过滤掉 list 类型的列（step_returns 等）
            flat_records = []
            for r in records:
                flat = {k: v for k, v in r.items() if not isinstance(v, (list, dict))}
                flat_records.append(flat)
            df = pl.DataFrame(flat_records)
        return write_ipc(df, path)

    def validate_schema(self, report: Dict[str, Any]) -> None:
        """校验 phase2_report.json 必填字段。"""
        missing = [k for k in REQUIRED_PHASE2_REPORT_KEYS if k not in report]
        if missing:
            raise Phase2ReportSchemaError(
                f"phase2_report.json 缺失必填字段: {missing}"
            )
