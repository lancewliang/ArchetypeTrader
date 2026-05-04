"""Phase I checkpoint 持久化（仅 IO）.

设计文档锚点: §4.14。
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from src.utils.feather_io import atomic_write_json, read_json

from .phase1_selection_policy import SelectionVerdict


@dataclass
class CheckpointEntry:
    """``checkpoint_manifest.json`` 中单条记录。

    Attributes
    ----------
    epoch : 该条记录对应的 epoch。
    path : checkpoint 文件路径（last 或 best 或 periodic）。
    file_hash : 该文件的 sha256，便于审计文件未被改写。
    metrics_path : 配套写入的 ``epoch_metrics/epoch_*.json`` 路径。
    verdict : ``selection_policy`` 给出的决策（promote_to_best / reject /
              keep_as_periodic / fatal）。
    reasons : guardrail 原因列表；空表示无阻塞。
    composite_score : 该 epoch 的 composite score。
    is_best : 当前是否为 best；promote 时设 True，并把同 manifest 中老的 best 清零。
    is_periodic : 是否是周期性 checkpoint（save_every）。
    """
    epoch: int
    path: str
    file_hash: str
    metrics_path: str
    verdict: str
    reasons: List[str] = field(default_factory=list)
    composite_score: float = 0.0
    is_best: bool = False
    is_periodic: bool = False


class Phase1FatalCollapse(RuntimeError):
    """``selection_policy`` 判定 fatal 时由 checkpoint manager 抛出。

    trainer 主循环捕获后会转抛 ``Phase1FatalError``，CLI 入口转换为非零退出码，
    避免静默产出无效 codebook。
    """


class Phase1CheckpointManager:
    """checkpoint IO 管理器；不嵌入 best 选择规则。

    职责
    ----
    - 原子写 ``last_vq_model.pt``、``best_vq_model.pt`` 与 ``checkpoints/epoch_*.pt``。
    - 接收 ``SelectionVerdict`` 决定是否 promote best。
    - 维护 ``checkpoint_manifest.json``（含 verdict reasons / 文件 hash / metrics 引用）。

    边界
    ----
    - 不直接判断 ``code_usage_ratio``、``val_max_drawdown`` 等业务阈值；这些规则全部
      由 ``selection_policy`` 决定。
    - 不写 ``phase1_report.json``；那由 ``Phase1ReportWriter`` 负责。
    - manifest 损坏时启动会重置为空列表，避免一次坏文件让训练永远无法恢复。
    """

    def __init__(self, artifacts_dir) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.last_path = self.artifacts_dir / "last_vq_model.pt"
        self.best_path = self.artifacts_dir / "best_vq_model.pt"
        self.periodic_dir = self.artifacts_dir / "checkpoints"
        self.periodic_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.artifacts_dir / "checkpoint_manifest.json"
        self.metrics_dir = self.artifacts_dir / "epoch_metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self._entries: List[CheckpointEntry] = []
        if self.manifest_path.exists():
            try:
                self._entries = [
                    CheckpointEntry(**e) for e in read_json(self.manifest_path)
                ]
            except Exception:
                # manifest 损坏不阻止训练；新写入会覆盖。
                self._entries = []

    # ---------- 内部 ----------

    def _save_torch_state(self, state: dict, target: Path) -> Path:
        """原子写 torch state_dict。"""
        try:
            import torch
        except ImportError:  # pragma: no cover
            raise RuntimeError("checkpoint 保存需要 torch")
        with tempfile.NamedTemporaryFile(
            delete=False, dir=target.parent, suffix=".tmp"
        ) as tmp:
            tmp_path = Path(tmp.name)
        torch.save(state, tmp_path)
        os.replace(tmp_path, target)
        return target

    def _write_metrics(self, metrics: dict, epoch: int) -> Path:
        path = self.metrics_dir / f"epoch_{epoch:04d}.json"
        atomic_write_json(metrics, path)
        return path

    def _flush_manifest(self) -> None:
        atomic_write_json([asdict(e) for e in self._entries], self.manifest_path)

    # ---------- 公共 API ----------

    def save_last(self, state: dict, metrics: dict, epoch: int) -> Path:
        """原子写 ``last_vq_model.pt`` + 当前 epoch metrics。

        每个 epoch 训练完都应调用（无论 verdict 如何），保证恢复训练时拿得到
        最近一次模型状态。
        """
        self._save_torch_state(state, self.last_path)
        self._write_metrics(metrics, epoch)
        return self.last_path

    def save_periodic(
        self, state: dict, metrics: dict, epoch: int, save_every: int
    ) -> Optional[Path]:
        """如果 ``epoch % save_every == 0`` 才写 ``checkpoints/epoch_{epoch:04d}.pt``。

        Returns
        -------
        Path or None : 已写入的 path；非触发周期时返回 ``None``。
        """
        if save_every <= 0 or epoch % save_every != 0:
            return None
        target = self.periodic_dir / f"epoch_{epoch:04d}.pt"
        self._save_torch_state(state, target)
        self._write_metrics(metrics, epoch)
        return target

    def commit_verdict(
        self,
        state: dict,
        metrics: dict,
        verdict: SelectionVerdict,
        epoch: int,
    ) -> CheckpointEntry:
        """根据 verdict 决定是否 promote best；写 manifest。

        实现要点
        --------
        - 若 last_path 不存在（异常路径），先把 state 写到 last_path，
          保证 best 的来源文件存在。
        - ``promote_to_best``: ``shutil.copyfile(last, best)``，并把所有旧 entry 的
          ``is_best`` 清零，保证同时只有一条 ``is_best=True``。
        - ``reject`` / ``keep_as_periodic``: 不动 best 文件。
        - ``fatal``: 把 verdict 写入 manifest，再抛 ``Phase1FatalCollapse``，
          让 trainer 退出。
        - 每条 entry 都计算文件 sha256，便于事后审计 checkpoint 是否被改写。

        Parameters
        ----------
        state : 当前模型 state（``{"model": state_dict, "epoch": ...}``）。
        metrics : 该 epoch 的指标 dict，已写入 metrics_dir。
        verdict : ``selection_policy`` 给出的决策。
        epoch : 当前 epoch index。

        Returns
        -------
        CheckpointEntry : 本次 commit 的记录（已经写入 manifest）。

        Raises
        ------
        Phase1FatalCollapse : ``verdict.decision == "fatal"``。
        """
        # 保证 last 已经存在
        if not self.last_path.exists():
            self._save_torch_state(state, self.last_path)
        metrics_path = self._write_metrics(metrics, epoch)

        is_best = False
        if verdict.decision == "promote_to_best":
            shutil.copyfile(self.last_path, self.best_path)
            # 旧 entries 清除 is_best
            for e in self._entries:
                e.is_best = False
            is_best = True
        elif verdict.decision == "fatal":
            entry = CheckpointEntry(
                epoch=epoch,
                path=str(self.last_path),
                file_hash=file_sha256(self.last_path),
                metrics_path=str(metrics_path),
                verdict="fatal",
                reasons=list(verdict.reasons),
                composite_score=verdict.composite_score,
                is_best=False,
                is_periodic=False,
            )
            self._entries.append(entry)
            self._flush_manifest()
            raise Phase1FatalCollapse(
                "Phase I 训练触发 fatal: " + "; ".join(verdict.reasons)
            )

        entry = CheckpointEntry(
            epoch=epoch,
            path=str(self.best_path if is_best else self.last_path),
            file_hash=file_sha256(self.best_path if is_best else self.last_path),
            metrics_path=str(metrics_path),
            verdict=verdict.decision,
            reasons=list(verdict.reasons),
            composite_score=verdict.composite_score,
            is_best=is_best,
            is_periodic=False,
        )
        self._entries.append(entry)
        self._flush_manifest()
        return entry

    def load(self, path):
        """加载 checkpoint 字典: ``{state_dict, optimizer, epoch, metrics, ...}``。

        约定 ``map_location="cpu"`` 让加载与设备无关；调用方再决定 ``.to(device)``。
        """
        try:
            import torch
        except ImportError:  # pragma: no cover
            raise RuntimeError("checkpoint 加载需要 torch")
        return torch.load(path, map_location="cpu")

    def load_manifest(self) -> List[CheckpointEntry]:
        """返回 manifest 当前内存副本（按写入顺序）。"""
        return list(self._entries)


def file_sha256(path: Path) -> str:
    """计算文件 sha256；checkpoint manifest 用于审计文件未被改写。"""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
