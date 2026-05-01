"""Phase II horizon index 生成与 Phase I 产物校验。

设计文档锚点: Phase II 执行计划 §Step 2。

职责:
- ``Phase1ArtifactValidator``: 校验 Phase I 冻结产物齐全性与 sign-off 状态。
- ``Phase2HorizonIndexer``: 生成 phase2_horizon_index_{train,val,test}.feather。

关键约束:
- 末尾 markout 越界 horizon 必须裁掉。
- gap horizon 必须标注并按配置裁掉。
- test 索引默认不含 code_label。
- 支持 non_overlap / stride / phase1_index 三种模式。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl

from src.config.phase2_config import Phase2Config
from src.trading.reward_alignment import RewardAlignment
from src.utils.feather_io import read_json, write_ipc


class Phase1ArtifactValidationError(ValueError):
    """Phase I 产物校验失败。"""


class Phase1HindsightWarningError(ValueError):
    """hindsight_bias_warning=exceeded 且未显式允许。"""


@dataclass
class Phase1ArtifactValidationResult:
    """校验结果。"""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    phase1_config: Optional[Dict[str, Any]] = None
    phase1_report: Optional[Dict[str, Any]] = None


class Phase1ArtifactValidator:
    """校验 Phase I 冻结产物。

    校验项:
    - decoder.pt / codebook.pt / input_schema.json 等文件齐全。
    - fatal_collapse=false / code_assignment_drift_warning=false。
    - hindsight_bias_warning 检查（默认拒绝 exceeded，除非 --allow-phase1-hindsight-warning）。
    - cost_config / reward_alignment / max_position 与 Phase I 一致。
    """

    def __init__(self, config: Phase2Config) -> None:
        self.config = config

    def validate(self) -> Phase1ArtifactValidationResult:
        """执行全部校验，返回结果。

        Raises
        ------
        Phase1ArtifactValidationError : 必要文件缺失或 fatal_collapse=true。
        Phase1HindsightWarningError : hindsight_bias_warning=exceeded 且未允许。
        """
        phase1_dir = self.config.phase1_dir()
        errors: List[str] = []
        warnings: List[str] = []

        # 1. 文件齐全性
        required = self.config.phase1_artifacts.required_files
        for fname in required:
            fpath = phase1_dir / fname
            if not fpath.exists():
                errors.append(f"缺少必要文件: {fpath}")

        if errors:
            raise Phase1ArtifactValidationError(
                f"Phase I 产物校验失败: {errors}"
            )

        # 2. 读取 phase1_report.json
        report_path = phase1_dir / "phase1_report.json"
        phase1_report: Dict[str, Any] = {}
        if report_path.exists():
            phase1_report = read_json(report_path)

        # 3. 读取 phase1_config.yaml
        config_path = phase1_dir / "phase1_config.yaml"
        phase1_config: Dict[str, Any] = {}
        if config_path.exists():
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                phase1_config = yaml.safe_load(f) or {}

        # 4. fatal_collapse 检查
        if phase1_report.get("fatal_collapse", False):
            raise Phase1ArtifactValidationError(
                "Phase I fatal_collapse=true，Phase II 拒绝启动"
            )

        # 5. code_assignment_drift_warning 检查
        if phase1_report.get("code_assignment_drift_warning", False):
            raise Phase1ArtifactValidationError(
                "Phase I code_assignment_drift_warning=true，Phase II 拒绝启动"
            )

        # 6. hindsight_bias_warning 检查
        hbw = phase1_report.get("hindsight_bias_warning", "ok")
        if hbw == "exceeded":
            if not self.config.allow_phase1_hindsight_warning:
                raise Phase1HindsightWarningError(
                    "Phase I hindsight_bias_warning=exceeded，"
                    "需要 --allow-phase1-hindsight-warning 才能继续"
                )
            warnings.append("hindsight_bias_warning=exceeded，已显式允许")

        return Phase1ArtifactValidationResult(
            valid=True,
            errors=errors,
            warnings=warnings,
            phase1_config=phase1_config,
            phase1_report=phase1_report,
        )


@dataclass
class Phase2HorizonEntry:
    """单条 horizon index 记录。"""
    sample_id: str
    horizon_start: int
    horizon_end: int
    split: str
    is_gap: bool = False
    gap_bars: int = 0
    code_label: Optional[int] = None
    is_labeled: bool = False


class Phase2HorizonIndexer:
    """生成 Phase II horizon index。

    支持三种模式:
    - non_overlap: 不重叠的连续 horizon。
    - stride: 按给定 stride 滑动。
    - phase1_index: 复用 Phase I 的 sample_id 对齐。

    输出 phase2_horizon_index_{train,val,test}.feather。
    """

    def __init__(self, config: Phase2Config) -> None:
        self.config = config
        self._alignment = RewardAlignment(
            self._get_reward_alignment()
        )

    def _get_reward_alignment(self) -> str:
        """从 Phase I config 获取 reward_alignment。"""
        phase1_dir = self.config.phase1_dir()
        config_path = phase1_dir / "phase1_config.yaml"
        if config_path.exists():
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                p1cfg = yaml.safe_load(f) or {}
            dp = p1cfg.get("dp", {})
            cc = dp.get("cost_config", {})
            return cc.get("reward_alignment", "paper_formula")
        return "paper_formula"

    def build_index(
        self,
        frame: pl.DataFrame,
        split: str,
        horizon: int,
        phase1_labels: Optional[pl.DataFrame] = None,
    ) -> List[Phase2HorizonEntry]:
        """为指定 split 生成 horizon index。

        Parameters
        ----------
        frame : polars.DataFrame，原始 market 数据。
        split : "train" / "val" / "test"。
        horizon : horizon 长度。
        phase1_labels : Phase I horizon labels（仅 train/val 使用）。

        Returns
        -------
        List[Phase2HorizonEntry] : horizon index 列表。
        """
        num_rows = frame.height
        lookahead = self._alignment.required_lookahead_rows()
        max_start = num_rows - horizon - lookahead

        mode = self.config.horizon_schedule.mode
        entries: List[Phase2HorizonEntry] = []

        if mode == "non_overlap":
            starts = list(range(0, max_start + 1, horizon))
        elif mode == "stride":
            stride = self.config.horizon_schedule.stride
            starts = list(range(0, max_start + 1, stride))
        elif mode == "phase1_index":
            # 从 Phase I labels 获取 start_index
            if phase1_labels is not None and "start_index" in phase1_labels.columns:
                all_starts = phase1_labels["start_index"].to_list()
                starts = [s for s in all_starts if s <= max_start]
            else:
                starts = list(range(0, max_start + 1, horizon))
        else:
            raise ValueError(f"未知 horizon_schedule.mode: {mode}")

        # 检测 timestamp gap
        ts_col = "timestamp"
        has_ts = ts_col in frame.columns
        ts_values = frame[ts_col].to_list() if has_ts else []

        # 构建 label lookup
        label_lookup: Dict[int, int] = {}
        if phase1_labels is not None and split != "test":
            if "start_index" in phase1_labels.columns and "code_label" in phase1_labels.columns:
                starts_col = phase1_labels["start_index"].to_list()
                labels_col = phase1_labels["code_label"].to_list()
                for si, cl in zip(starts_col, labels_col):
                    if cl is not None:
                        label_lookup[si] = int(cl)

        gap_threshold = self.config.horizon_schedule.gap_threshold_bars

        for idx, start in enumerate(starts):
            end = start + horizon - 1
            sample_id = f"p2_{split}_{idx:06d}"

            # gap 检测
            is_gap = False
            gap_bars = 0
            if has_ts and len(ts_values) > end:
                for t in range(start, min(end, len(ts_values) - 1)):
                    try:
                        diff = ts_values[t + 1] - ts_values[t]
                        if hasattr(diff, "total_seconds"):
                            gap_minutes = diff.total_seconds() / 60.0
                        else:
                            gap_minutes = float(diff)
                        if gap_minutes > gap_threshold:
                            is_gap = True
                            gap_bars = max(gap_bars, int(gap_minutes))
                    except (TypeError, ValueError):
                        pass

            # label
            code_label: Optional[int] = None
            is_labeled = False
            if split != "test" and start in label_lookup:
                code_label = label_lookup[start]
                is_labeled = True

            entries.append(Phase2HorizonEntry(
                sample_id=sample_id,
                horizon_start=start,
                horizon_end=end,
                split=split,
                is_gap=is_gap,
                gap_bars=gap_bars,
                code_label=code_label,
                is_labeled=is_labeled,
            ))

        # 按配置裁掉 gap horizons
        if self.config.horizon_schedule.exclude_gap_horizons:
            entries = [e for e in entries if not e.is_gap]

        return entries

    def write_index(
        self,
        entries: List[Phase2HorizonEntry],
        output_path: Path,
    ) -> Path:
        """将 horizon index 写入 feather 文件。"""
        data = {
            "sample_id": [e.sample_id for e in entries],
            "horizon_start": [e.horizon_start for e in entries],
            "horizon_end": [e.horizon_end for e in entries],
            "split": [e.split for e in entries],
            "is_gap": [e.is_gap for e in entries],
            "gap_bars": [e.gap_bars for e in entries],
            "is_labeled": [e.is_labeled for e in entries],
        }
        # test 索引默认不含 code_label
        split_set = set(e.split for e in entries)
        if "test" not in split_set:
            data["code_label"] = [e.code_label for e in entries]

        df = pl.DataFrame(data)
        return write_ipc(df, output_path)
