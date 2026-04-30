"""``phase1_report.json`` 与诊断 JSON/Feather 写入.

设计文档锚点: §4.16 与 §8。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

from src.utils.feather_io import atomic_write_json


# ``phase1_report.json`` 必填字段。新增指标时必须同步本列表。
REQUIRED_REPORT_KEYS = (
    "reconstruction_accuracy",
    "weighted_reconstruction_accuracy",
    "non_flat_accuracy",
    "code_usage",
    "perplexity",
    "single_trade_consistency_rate",
    "no_trade_ratio",
    "reward_alignment",
    "reward_normalization_resolved",
    "reward_norm_clip_ratio",
    "dataset_reject_rate",
    "stratification_mode",
    "is_hindsight_stratification",
    "prospective_diagnostic_required",
    "diagnostic_pair_batch_id",
    "phase1_composite_score",
    "best_epoch",
    "best_checkpoint_path",
    "selection_metric",
    "composite_score_sensitivity",
)


class ReportSchemaError(ValueError):
    """``phase1_report.json`` 缺失必填字段。"""


@dataclass
class ReportPaths:
    artifacts_dir: Path
    phase1_report: Path
    action_diagnostics: Path
    risk_diagnostics: Path
    archetype_separation: Path
    archetype_behavior_diagnostics: Path
    horizon_boundary_diagnostics: Path
    code_stability_diagnostics: Path
    sampling_leakage_diagnostics: Path
    composite_score_sensitivity: Path
    epoch_metrics_dir: Path

    @classmethod
    def from_artifacts_dir(cls, artifacts_dir: Path) -> "ReportPaths":
        artifacts_dir = Path(artifacts_dir)
        return cls(
            artifacts_dir=artifacts_dir,
            phase1_report=artifacts_dir / "phase1_report.json",
            action_diagnostics=artifacts_dir / "action_diagnostics.json",
            risk_diagnostics=artifacts_dir / "risk_diagnostics.json",
            archetype_separation=artifacts_dir / "archetype_separation.json",
            archetype_behavior_diagnostics=artifacts_dir / "archetype_behavior_diagnostics.json",
            horizon_boundary_diagnostics=artifacts_dir / "horizon_boundary_diagnostics.json",
            code_stability_diagnostics=artifacts_dir / "code_stability_diagnostics.json",
            sampling_leakage_diagnostics=artifacts_dir / "sampling_leakage_diagnostics.json",
            composite_score_sensitivity=artifacts_dir / "composite_score_sensitivity.json",
            epoch_metrics_dir=artifacts_dir / "epoch_metrics",
        )


class Phase1ReportWriter:
    """统一 report 写入接口。

    边界
    ----
    - 只负责序列化与 schema 校验，不重新计算指标。
    - ``checkpoint_manifest.json`` 由 ``Phase1CheckpointManager`` 维护，
      ``selection_policy`` 决定 best；本类不参与决策。
    - 所有 JSON 通过 ``atomic_write_json`` 原子写。
    """

    def __init__(self, paths: ReportPaths) -> None:
        self.paths = paths
        self.paths.epoch_metrics_dir.mkdir(parents=True, exist_ok=True)

    def write_epoch_metrics(self, metrics: dict, epoch: int) -> Path:
        """写 ``epoch_metrics/epoch_{epoch:04d}.json``，便于审计每个 epoch 的指标。"""
        path = self.paths.epoch_metrics_dir / f"epoch_{epoch:04d}.json"
        atomic_write_json(metrics, path)
        return path

    def write_final_report(self, summary: dict) -> Path:
        """写 ``phase1_report.json``；落盘前调 ``validate_schema`` 校验必填字段。

        Raises
        ------
        ReportSchemaError : 缺失必填 key (``REQUIRED_REPORT_KEYS``)。
        """
        self.validate_schema(summary)
        atomic_write_json(summary, self.paths.phase1_report)
        return self.paths.phase1_report

    def write_diagnostics(self, diagnostics: dict) -> List[Path]:
        """根据 key 写各诊断 JSON。

        ``diagnostics`` 中允许包含 ``action / risk / archetype_separation /
        archetype_behavior / horizon_boundary / code_stability / sampling_leakage /
        composite_score_sensitivity`` 等键；未识别的 key 静默跳过，避免错配 path。
        """
        out: List[Path] = []
        mapping = {
            "action": self.paths.action_diagnostics,
            "risk": self.paths.risk_diagnostics,
            "archetype_separation": self.paths.archetype_separation,
            "archetype_behavior": self.paths.archetype_behavior_diagnostics,
            "horizon_boundary": self.paths.horizon_boundary_diagnostics,
            "code_stability": self.paths.code_stability_diagnostics,
            "sampling_leakage": self.paths.sampling_leakage_diagnostics,
            "composite_score_sensitivity": self.paths.composite_score_sensitivity,
        }
        for key, payload in diagnostics.items():
            if key in mapping:
                atomic_write_json(payload, mapping[key])
                out.append(mapping[key])
        return out

    def validate_schema(self, report: dict) -> None:
        """检查 ``REQUIRED_REPORT_KEYS`` 全部存在；缺失抛 ``ReportSchemaError``。

        - 只校验必填字段；未知字段（自定义扩展）默认放行。
        - 该方法是新增指标"必须同步必填列表"约束的执行者；任何在 ``REQUIRED_REPORT_KEYS``
          中追加的指标必须由 trainer 实际写入，否则 sign-off 阶段会被拦下。
        """
        missing = [k for k in REQUIRED_REPORT_KEYS if k not in report]
        if missing:
            raise ReportSchemaError(
                f"phase1_report.json 缺失必填字段: {missing}"
            )
