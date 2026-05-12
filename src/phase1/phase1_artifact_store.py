"""Phase I 训练产出物读写类的接口骨架。"""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, List, Mapping, cast

import polars as pl
import torch

from .checkpoint import (
    Phase1Checkpoint,
    Phase1CheckpointConfig,
    Phase1CheckpointMetrics,
    Phase1CheckpointStage,
    Phase1StateDict,
    Phase1ValidationCheckpoint,
)
from .metrics import Phase1ValidationResult
from ..store.artifact_store import DataFileStore


class Phase1ArtifactStore(DataFileStore):
    """负责 Phase I 训练产物目录、checkpoint、模型导出和 label 读写。

    为什么需要这个类:
        Phase I 训练产物包含 checkpoint、encoder/decoder/codebook 导出、horizon
        labels 和训练报告。它们与通用数据准备产物生命周期不同，单独拆分可以让
        ``DataFileStore`` 聚焦 horizon/trajectory 数据集读写，同时保留 Phase I
        对基础数据集读取接口的复用能力。
    """

    def initialize_phase1_artifact_dirs(self) -> None:
        """初始化 Phase I 训练产出物目录，并返回标准产物路径。

        参数:
            pair: 交易标的，例如 ``BTC``、``ETH`` 或当前工程中的 ``AL``。
            batch_id: Phase I 训练批次 ID，例如 ``batch_001``。
            artifacts_root: 全阶段产物根目录，默认 ``artifacts``。

        输出:
            返回 Phase I 标准产物路径字典。当前方法只创建目录和规划路径，
            不写入任何 checkpoint、模型、label 或报告文件。

        方法作用:
            将 Phase I 目录结构集中到 store 层管理，避免训练主流程、入口脚本
            或评估模块各自拼接路径。后续保存 checkpoint、导出 model 和写报告
            都应复用这里返回的 key。

        目录骨架:
            ``output_dir``:
                Phase I 根目录，保存 config、normalizer、导出模型、label 和报告。
            ``checkpoints``:
                周期性 epoch checkpoint 与 best checkpoint 的目录。
            ``diagnostics``:
                action/risk/archetype/boundary 等诊断 JSON 或图表目录。
            ``tensorboard``:
                TensorBoard 事件文件目录。
            ``latent_snapshots``:
                latent/codebook 快照目录，供离线可视化和稳定性复盘使用。
            ``failure_cases``:
                失败样本、bad reconstruction 或 guardrail reject 复盘目录。

        设计边界:
            本方法只负责文件系统目录和路径契约，不初始化日志、不设置随机种子、
            不加载数据，也不创建空产物文件。
        """

        root = Path(self.artifacts_root) / self.pair / self.batchid / "phase1"
        checkpoint_dir = root / "checkpoints"
        diagnostics_dir = root / "diagnostics"
        tensorboard_dir = root / "tensorboard"
        latent_snapshot_dir = root / "latent_snapshots"
        failure_case_dir = root / "failure_cases"
        label_dir = root / "labels"
        metrics_dir = root / "metrics"
        report_dir = root / "reports"
        validation_dir = root / "validation"
        best_checkpoint_path = checkpoint_dir / "best_checkpoint.pt"

        for directory in (
            root,
            checkpoint_dir,
            diagnostics_dir,
            tensorboard_dir,
            latent_snapshot_dir,
            failure_case_dir,
            label_dir,
            metrics_dir,
            report_dir,
            validation_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        self.artifact_paths = {
            "output_dir": root,
            "checkpoints": checkpoint_dir,
            "diagnostics": diagnostics_dir,
            "tensorboard": tensorboard_dir,
            "latent_snapshots": latent_snapshot_dir,
            "failure_cases": failure_case_dir,
            "labels": label_dir,
            "metrics": metrics_dir,
            "reports": report_dir,
            "validation_results": validation_dir,
            "horizon_train_labels": label_dir / "sampled_horizon_labels_train.feather",
            "best_checkpoint": best_checkpoint_path,
            "encoder": root / "encoder.pt",
            "decoder": root / "decoder.pt",
            "codebook": root / "codebook.pt",
            "phase1_report": root / "phase1_report.json",
            "phase1_codebook_validation_json": root / "phase1_codebook_validation.json",
            "phase1_codebook_validation_html": report_dir / "phase1_codebook_validation.html",
            "phase1_checkpoint_selection_html": report_dir / "phase1_checkpoint_selection.html",
        }

        return None

    def save_phase1_checkpoint(
        self,
        *,
        stage: Phase1CheckpointStage,
        epoch: int,
        config: Phase1CheckpointConfig,
        model_state_dict: Phase1StateDict,
        optimizer_state_dict: Phase1StateDict      
    ) -> None:
        """保存 Phase I checkpoint。

        参数:
            stage: checkpoint 所属阶段，例如 ``pretrain`` 或 ``vq``。
            epoch: checkpoint 所属 epoch，从 1 开始。
            config: 训练配置快照。
            model_state_dict: 模型参数状态。
            optimizer_state_dict: 优化器状态。
            metrics: 按 split 组织的 epoch 指标。

        方法作用:
            统一封装 Phase I checkpoint 持久化。正式实现应使用原子写入，并
            记录 sha256 供审计。
        """

        checkpoint = Phase1Checkpoint(
            stage=stage,
            epoch=epoch,
            is_best=False,
            config=config,
            model_state_dict=model_state_dict,
            optimizer_state_dict=optimizer_state_dict
        )
        checkpoint_path = self._phase1_checkpoint_path(stage=stage, epoch=epoch)
        self._save_phase1_checkpoint_payload(checkpoint, checkpoint_path)

    def load_phase1_checkpoint(
        self,
        *,
        stage: Phase1CheckpointStage | None = None,
        epoch: int | None = None     
    ) -> Phase1Checkpoint:
        """读取 Phase I checkpoint。

        参数:
            stage: checkpoint 所属阶段；读取非 best checkpoint 时必填。
            epoch: checkpoint 所属 epoch；读取非 best checkpoint 时必填。
            best: 是否读取当前 best checkpoint。为 ``True`` 时忽略
                ``stage`` 和 ``epoch``。

        输出:
            返回 checkpoint 内容，供恢复训练、best 选择和模型导出使用。
        """
      
        path = self._phase1_checkpoint_path(stage=stage, epoch=epoch)
        return self._load_phase1_checkpoint_payload(path)

    def save_best_checkpoint(
        self,
        checkpoint: Phase1Checkpoint,
    ) -> None:
        """保存 Phase I best checkpoint。

        参数:
            checkpoint: ``Phase1CheckpointSelector`` 选出的 best checkpoint payload。

        方法作用:
            为 ``Phase1MainFlow.export_phase2_artifacts`` 提供单独的 best
            checkpoint 固化入口。当前方法只定义 store 层调用契约，正式实现
            后续再补齐原子写入、索引更新、校验和审计元数据。
        """

        best_checkpoint = replace(checkpoint, is_best=True)
        self._save_phase1_checkpoint_payload(
            best_checkpoint,
            self._phase1_best_checkpoint_path(),
        )

    def save_phase1_horizon_labels(
        self,
        labels: Any,
        split_name: str = "train",
    ) -> None:
        """保存 Phase I horizon-level archetype labels。

        参数:
            labels: horizon label 表。后续实现可使用 polars/pandas DataFrame，
                至少应包含 ``sample_id`` 和 ``code_label``。
            split_name: 数据 split 名称，例如 ``train``、``val`` 或 ``test``。
                文件路径由当前 ``Phase1ArtifactStore`` 的 Phase I 产物路径决定。

        方法作用:
            保存由 best checkpoint 生成的离线 archetype label，供 Phase II/III
            训练和审计复用。
        """

        path = self._phase1_horizon_label_path(split_name)
        path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(labels, pl.DataFrame):
            label_frame = labels
        else:
            label_frame = pl.DataFrame(labels)

        suffix = path.suffix.lower()
        if suffix in {".feather", ".ipc", ".arrow"}:
            label_frame.write_ipc(path)
        elif suffix == ".parquet":
            label_frame.write_parquet(path)
        elif suffix == ".csv":
            label_frame.write_csv(path)
        else:
            raise ValueError(
                "unsupported horizon label format; use .feather, .ipc, .parquet, or .csv"
            )

    def load_phase1_horizon_labels(
        self,
        split_name: str = "train",
    ) -> Any:
        """读取 Phase I horizon-level archetype labels。

        参数:
            split_name: 数据 split 名称，例如 ``train``、``val`` 或 ``test``。
                文件路径由当前 ``Phase1ArtifactStore`` 的 Phase I 产物路径决定。

        输出:
            返回 label 表，供 Phase II/III 训练、评估或审计使用。
        """

        path = self._phase1_horizon_label_path(split_name)
        suffix = path.suffix.lower()
        if suffix in {".feather", ".ipc", ".arrow"}:
            return pl.read_ipc(path)
        if suffix == ".parquet":
            return pl.read_parquet(path)
        if suffix == ".csv":
            return pl.read_csv(path)
        raise ValueError(
            "unsupported horizon label format; use .feather, .ipc, .parquet, or .csv"
        )

    def save_phase1_validation_result(
        self,
        validation_result: Phase1ValidationResult,
    ) -> Path:
        """保存单个 Phase I codebook validation 结果。

        validation result 是 checkpoint 级五层验证的完整机器可读 payload。这里按
        stage/epoch 保存一份不可变结果，供 best 选择、报告和后续阶段审计读取。
        """

        path = self._phase1_validation_result_path(
            stage=validation_result.stage,
            epoch=validation_result.epoch,
        )
        payload = validation_result.to_dict()
        self._save_json_payload(payload, path)
        return path

    def save_phase1_epoch_metrics(
        self,
        *,
        metrics: Phase1ValidationCheckpoint | Mapping[str, Any],
        stage: str | None = None,
        epoch: int | None = None,
    ) -> Path:
        """保存单个 Phase I epoch 的训练/评估指标 JSON。"""

        if isinstance(metrics, Phase1ValidationCheckpoint):
            path = self._phase1_epoch_metrics_path(
                stage=metrics.stage,
                epoch=metrics.epoch,
            )
            payload = metrics.to_dict()
        else:
            if stage is None or epoch is None:
                raise ValueError(
                    "stage and epoch are required when saving raw phase1 metrics"
                )
            path = self._phase1_epoch_metrics_path(stage=stage, epoch=epoch)
            payload = {"stage": stage, "epoch": epoch, **dict(metrics)}
        self._save_json_payload(payload, path)
        return path

    def save_phase1_codebook_validation_html(
        self,
        *,
        validation_result: Phase1ValidationResult,
        html: str,
    ) -> Path:
        """保存单个 Phase I codebook validation HTML 报告。"""

        path = self._phase1_codebook_validation_html_path(
            stage=validation_result.stage,
            epoch=validation_result.epoch,
        )
        self._save_text_payload(html, path)
        return path

    def save_phase1_checkpoint_selection_html(
        self,
        *,
        html: str,
    ) -> Path:
        """保存 Phase I checkpoint selection HTML 报告。"""

        path = self._phase1_checkpoint_selection_html_path()
        self._save_text_payload(html, path)
        return path

    def load_phase1_all_epoch_metrics(
        self        
    ) -> List[Phase1ValidationCheckpoint]:
        """读取 Phase I epoch metrics。

        传入 ``stage`` 和 ``epoch`` 时读取单个 JSON payload；不传时读取 metrics
        目录下所有 validation checkpoint payload，供 selector 使用。
        """
        metrics_dir = self._phase1_artifact_path("metrics")
        checkpoints: list[Phase1ValidationCheckpoint] = []
        for path in sorted(metrics_dir.glob("*_metrics.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                raise ValueError(f"invalid phase1 epoch metrics payload: {path}")
            checkpoints.append(Phase1ValidationCheckpoint.from_dict(payload))
        return checkpoints

    def load_phase1_validation_result(
        self,
        *,
        stage: str,
        epoch: int,
    ) -> Phase1ValidationResult:
        """读取 Phase I codebook validation 结果。"""

        path = self._phase1_validation_result_path(stage=stage, epoch=epoch)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"invalid phase1 validation result payload: {path}")
        return Phase1ValidationResult.from_dict(payload)

    def _phase1_horizon_label_path(self, split_name: str = "train") -> Path:
        """返回 Phase I horizon label 的标准路径。"""

        if split_name == "train":
            train_labels = self.artifact_paths.get("horizon_train_labels")
            if train_labels is not None:
                return Path(train_labels)

        label_dir = self.artifact_paths.get("labels")
        if label_dir is not None:
            return Path(label_dir) / f"sampled_horizon_labels_{split_name}.feather"

        output_dir = self.artifact_paths.get("output_dir")
        if output_dir is not None:
            return Path(output_dir) / f"sampled_horizon_labels_{split_name}.feather"

        raise ValueError("data_store must be initialized with phase1 artifact paths")

    def save_phase1_report(
        self,
        report: dict[str, Any],
    ) -> None:
        """保存 Phase I 训练报告。

        参数:
            report: 训练报告内容，包含配置摘要、best checkpoint、评估指标、
                guardrail 结果、warning 和 collapse 诊断。

        方法作用:
            统一封装 Phase I report 写入。正式实现应在写入前校验 report
            schema，并使用稳定 JSON 格式。
        """

        _ = self._phase1_artifact_path("phase1_report")
        ...

    def load_phase1_report(
        self,
    ) -> dict[str, Any]:
        """读取 Phase I 训练报告。

        输出:
            返回训练报告字典，供复查、resume 和后续阶段读取。
        """

        _ = self._phase1_artifact_path("phase1_report")
        ...

    def _phase1_artifact_path(self, key: str) -> Path:
        """返回 Phase I 标准产物路径。"""

        path = self.artifact_paths.get(key)
        if path is None:
            raise ValueError(
                "data_store must be initialized with phase1 artifact paths"
            )
        return Path(path)

    def _phase1_checkpoint_path(
        self,
        *,
        stage: Phase1CheckpointStage,
        epoch: int,
    ) -> Path:
        """返回指定阶段和 epoch 的 Phase I checkpoint 标准路径。"""

        checkpoint_dir = self._phase1_artifact_path("checkpoints")
        return checkpoint_dir / f"{stage}_epoch_{epoch:04d}.pt"

    def _phase1_best_checkpoint_path(self) -> Path:
        """返回 Phase I best checkpoint 标准路径。"""

        return self._phase1_artifact_path("best_checkpoint")

    def _phase1_validation_result_path(
        self,
        *,
        stage: str,
        epoch: int,
    ) -> Path:
        """返回指定阶段和 epoch 的 validation result 标准路径。"""

        validation_dir = self._phase1_artifact_path("validation_results")
        return validation_dir / f"{stage}_epoch_{epoch:04d}_validation.json"

    def _phase1_epoch_metrics_path(
        self,
        *,
        stage: str,
        epoch: int,
    ) -> Path:
        """返回指定阶段和 epoch 的基础指标 JSON 标准路径。"""

        metrics_dir = self._phase1_artifact_path("metrics")
        return metrics_dir / f"{stage}_epoch_{epoch:04d}_metrics.json"

    def _phase1_codebook_validation_html_path(
        self,
        *,
        stage: str,
        epoch: int,
    ) -> Path:
        """返回指定阶段和 epoch 的 codebook validation HTML 报告路径。"""

        report_dir = self._phase1_artifact_path("reports")
        return report_dir / f"{stage}_epoch_{epoch:04d}_codebook_validation.html"

    def _phase1_checkpoint_selection_html_path(self) -> Path:
        """返回 checkpoint selection HTML 报告路径。"""

        return self._phase1_artifact_path("phase1_checkpoint_selection_html")

    def _save_phase1_checkpoint_payload(
        self,
        checkpoint: Phase1Checkpoint,
        path: Path,
    ) -> None:
        """以原子替换方式保存 checkpoint payload 并写出 sha256 sidecar。"""

        path.parent.mkdir(parents=True, exist_ok=True)
        payload = checkpoint.to_dict()

        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            torch.save(payload, temp_path)
            digest = self._sha256_file(temp_path)
            temp_path.replace(path)
            self._write_sha256_sidecar(path, digest)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

    def _load_phase1_checkpoint_payload(self, path: Path) -> Phase1Checkpoint:
        """读取 checkpoint 文件并恢复为 ``Phase1Checkpoint``。"""

        if not path.exists():
            raise FileNotFoundError(f"phase1 checkpoint not found: {path}")

        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")

        if isinstance(payload, Phase1Checkpoint):
            return payload
        if not isinstance(payload, Mapping):
            raise ValueError(f"invalid phase1 checkpoint payload: {path}")
        return self._phase1_checkpoint_from_mapping(payload, path)

    def _phase1_checkpoint_from_mapping(
        self,
        payload: Mapping[str, object],
        path: Path,
    ) -> Phase1Checkpoint:
        """校验 checkpoint 字段并恢复强类型 payload。"""

        required_keys = {
            "stage",
            "epoch",
            "is_best",
            "config",
            "model_state_dict",
            "optimizer_state_dict",
            "metrics",
        }
        missing_keys = required_keys.difference(payload)
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise ValueError(f"invalid phase1 checkpoint {path}: missing {missing}")

        stage = payload["stage"]
        if stage not in {"pretrain", "vq"}:
            raise ValueError(f"invalid phase1 checkpoint {path}: unsupported stage")
        stage = cast(Phase1CheckpointStage, stage)

        config = payload["config"]
        model_state_dict = payload["model_state_dict"]
        optimizer_state_dict = payload["optimizer_state_dict"]
        metrics = payload["metrics"]
        if not isinstance(config, Mapping):
            raise ValueError(
                f"invalid phase1 checkpoint {path}: config must be a mapping"
            )
        if not isinstance(model_state_dict, Mapping):
            raise ValueError(
                f"invalid phase1 checkpoint {path}: model_state_dict must be a mapping"
            )
        if not isinstance(optimizer_state_dict, Mapping):
            raise ValueError(
                f"invalid phase1 checkpoint {path}: optimizer_state_dict must be a mapping"
            )
        if not isinstance(metrics, Mapping):
            raise ValueError(
                f"invalid phase1 checkpoint {path}: metrics must be a mapping"
            )

        try:
            epoch = int(payload["epoch"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid phase1 checkpoint {path}: epoch must be an integer"
            ) from exc

        is_best = payload["is_best"]
        if not isinstance(is_best, bool):
            raise ValueError(f"invalid phase1 checkpoint {path}: is_best must be a bool")

        restored_metrics: dict[str, dict[str, object]] = {}
        for split, split_metrics in metrics.items():
            if not isinstance(split_metrics, Mapping):
                raise ValueError(
                    f"invalid phase1 checkpoint {path}: "
                    f"metrics[{split!r}] must be a mapping"
                )
            restored_metrics[str(split)] = dict(split_metrics)

        return Phase1Checkpoint(
            stage=stage,
            epoch=epoch,
            is_best=is_best,
            config=dict(config),
            model_state_dict=dict(model_state_dict),
            optimizer_state_dict=dict(optimizer_state_dict),
            metrics=restored_metrics,
        )

    def _write_sha256_sidecar(self, path: Path, digest: str) -> None:
        """写出与 checkpoint 同名的 sha256 审计文件。"""

        sidecar_path = path.with_suffix(f"{path.suffix}.sha256")
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=sidecar_path.parent,
            prefix=f".{sidecar_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(f"{digest}  {path.name}\n")

        try:
            temp_path.replace(sidecar_path)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

    @staticmethod
    def _sha256_file(path: Path) -> str:
        """计算文件 sha256。"""

        digest = hashlib.sha256()
        with path.open("rb") as payload_file:
            for chunk in iter(lambda: payload_file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _save_json_payload(self, payload: Mapping[str, Any], path: Path) -> None:
        """以原子替换方式保存 JSON payload 并写出 sha256 sidecar。"""

        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            json.dump(
                self._json_safe(payload),
                temp_file,
                ensure_ascii=False,
                indent=2,
            )
            temp_file.write("\n")

        try:
            digest = self._sha256_file(temp_path)
            temp_path.replace(path)
            self._write_sha256_sidecar(path, digest)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

    def _save_text_payload(self, payload: str, path: Path) -> None:
        """以原子替换方式保存文本 payload 并写出 sha256 sidecar。"""

        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(payload)

        try:
            digest = self._sha256_file(temp_path)
            temp_path.replace(path)
            self._write_sha256_sidecar(path, digest)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

    @classmethod
    def _json_safe(cls, value: Any) -> Any:
        """把 Path、numpy scalar 等对象转换成 JSON 友好值。"""

        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Mapping):
            return {str(key): cls._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._json_safe(item) for item in value]
        if hasattr(value, "item") and callable(value.item):
            try:
                return value.item()
            except (TypeError, ValueError):
                return value
        return value


Phase1Store = Phase1ArtifactStore
