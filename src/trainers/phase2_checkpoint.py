"""Phase II checkpoint 管理: last_selector.pt / best_selector.pt / manifest。

设计文档锚点: Phase II 执行计划 §Step 6。

职责:
- 保存 last_selector.pt。
- 根据 verdict promote best_selector.pt。
- 写 phase2_checkpoint_manifest.json。
- 保存 replay_log_last_complete_checkpoint.feather。

镜像 Phase I 的 Phase1CheckpointManager 风格。
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl

from src.trainers.phase2_selection_policy import Phase2SelectionVerdict
from src.utils.feather_io import atomic_write_json, read_json, write_ipc


@dataclass
class Phase2CheckpointEntry:
    """checkpoint_manifest.json 中单条记录。"""
    update_idx: int
    path: str
    file_hash: str
    verdict: str
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    is_best: bool = False


def _file_sha256(path: Path) -> str:
    """计算文件 sha256。"""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class Phase2CheckpointManager:
    """Phase II checkpoint IO 管理器。

    边界:
    - 不嵌入 best 选择规则（由 Phase2SelectionPolicy 决定）。
    - 只做 IO: 保存 / 加载 / manifest 维护。
    """

    def __init__(self, artifacts_dir: Path) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.last_path = self.artifacts_dir / "last_selector.pt"
        self.best_path = self.artifacts_dir / "best_selector.pt"
        self.manifest_path = self.artifacts_dir / "phase2_checkpoint_manifest.json"
        self.replay_log_path = (
            self.artifacts_dir / "replay_log_last_complete_checkpoint.feather"
        )
        self._entries: List[Phase2CheckpointEntry] = []
        if self.manifest_path.exists():
            try:
                raw = read_json(self.manifest_path)
                self._entries = [Phase2CheckpointEntry(**e) for e in raw]
            except Exception:
                self._entries = []

    def _save_torch_state(self, state: Dict[str, Any], target: Path) -> Path:
        """原子写 torch state_dict。"""
        import torch
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            delete=False, dir=target.parent, suffix=".tmp"
        ) as tmp:
            tmp_path = Path(tmp.name)
        torch.save(state, tmp_path)
        os.replace(tmp_path, target)
        return target

    def _flush_manifest(self) -> None:
        atomic_write_json([asdict(e) for e in self._entries], self.manifest_path)

    def save_last(self, state: Dict[str, Any], update_idx: int) -> Path:
        """原子写 last_selector.pt。"""
        return self._save_torch_state(state, self.last_path)

    def commit_verdict(
        self,
        state: Dict[str, Any],
        verdict: Phase2SelectionVerdict,
        update_idx: int,
        metrics: Dict[str, float],
    ) -> Phase2CheckpointEntry:
        """根据 verdict 决定是否 promote best；写 manifest。"""
        # 确保 last 已保存
        if not self.last_path.exists():
            self._save_torch_state(state, self.last_path)

        is_best = False
        if verdict.decision == "promote_to_best":
            shutil.copyfile(self.last_path, self.best_path)
            for e in self._entries:
                e.is_best = False
            is_best = True

        entry = Phase2CheckpointEntry(
            update_idx=update_idx,
            path=str(self.best_path if is_best else self.last_path),
            file_hash=_file_sha256(self.best_path if is_best else self.last_path),
            verdict=verdict.decision,
            reasons=list(verdict.reasons),
            metrics={k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
            is_best=is_best,
        )
        self._entries.append(entry)
        self._flush_manifest()
        return entry

    def save_replay_log(self, replay_records: List[Dict[str, Any]]) -> Path:
        """保存 replay_log_last_complete_checkpoint.feather。

        最小字段 schema:
        update_idx / env_id / sample_id / timestamp_start / chosen_code /
        final_position / reward_raw / boundary_cost / risk_triggered。
        """
        if not replay_records:
            df = pl.DataFrame({
                "update_idx": [],
                "env_id": [],
                "sample_id": [],
                "chosen_code": [],
                "final_position": [],
                "reward_raw": [],
                "boundary_cost": [],
                "risk_triggered": [],
            })
        else:
            flat = []
            for r in replay_records:
                flat.append({
                    "update_idx": r.get("update_idx", 0),
                    "env_id": r.get("env_id", 0),
                    "sample_id": r.get("sample_id", ""),
                    "chosen_code": r.get("chosen_code", 0),
                    "final_position": r.get("final_position", 0),
                    "reward_raw": r.get("reward_raw", 0.0),
                    "boundary_cost": r.get("boundary_cost", 0.0),
                    "risk_triggered": r.get("risk_triggered", False),
                })
            df = pl.DataFrame(flat)
        return write_ipc(df, self.replay_log_path)

    def load(self, path: Path) -> Dict[str, Any]:
        """加载 checkpoint。"""
        import torch
        return torch.load(path, map_location="cpu")
