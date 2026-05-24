"""Phase II 训练产物 store 骨架。

本文件只定义 Phase II artifact store 的职责边界、方法签名和路径语义注释。
不实现目录创建、文件读写、序列化、反序列化、checkpoint 保存或指标保存逻辑。
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from ..store.artifact_store import DataFileStore
from .checkpoint.phase2_checkpoint import Phase2Checkpoint
from .metrics.phase2_metric_results import Phase2ValidationResult
from .phase2_config import Phase2TrainConfig


class Phase2ArtifactStore(DataFileStore):
    """Phase II 产物路径与读写入口骨架。

    适用场景:
        由 ``Phase2MainFlow`` 持有，用于集中管理 Phase II selector 训练期间的
        dataset cache、checkpoint、metrics 和 report 路径。

    设计边界:
        本类复用 ``DataFileStore`` 的基础数据读写能力，但 Phase II 自有产物
        必须放在 ``artifacts/{pair}/{train_batch_id}/phase2/`` 命名空间下。
        本骨架不执行真实 I/O，只固定后续实现应提供的接口。
    """

    def __init__(
        self,
        pair: str,
        train_batch_id: str,
        artifacts_root: Path,
    ) -> None:
        """初始化 Phase II artifact store。

        适用场景:
            在 ``Phase2MainFlow`` 初始化时创建，绑定一次 Phase II 训练的交易
            标的、训练批次和产物根目录。

        参数:
            pair: 交易标的或数据域名称，例如 ``BTCUSDT``。
            train_batch_id: 当前训练批次 ID。
            artifacts_root: 全阶段产物根目录。

        字段解释:
            后续实现应将 ``train_batch_id`` 映射到 ``DataFileStore.batchid``，
            并把 Phase II 的标准路径写入 ``artifact_paths``。
        """

        super().__init__(
            pair=pair,
            batchid=train_batch_id,
            artifacts_root=artifacts_root,
        )

    def initialize_phase2_artifact_dirs(self) -> None:
        """初始化 Phase II 标准产物目录骨架。

        适用场景:
            在训练开始前调用，规划 checkpoint、dataset cache、metrics 和 report
            等路径，供训练、评估、checkpoint selector 和 report 复用。

        目录语义:
            ``output_dir``:
                Phase II 根目录，推荐为 ``artifacts/{pair}/{batch_id}/phase2``。
            ``dataset_cache``:
                保存按 split 生成的 ``Phase2SelectionDataset`` cache。
            ``checkpoints``:
                保存周期性 epoch checkpoint 和 best checkpoint。
            ``metrics``:
                保存 train/validation/test selection metrics JSON。
            ``reports``:
                保存 Phase II selection report 的 JSON/HTML 输出。
            ``diagnostics``:
                保存 code usage、label consistency、reward distribution 等诊断产物。

        设计边界:
            本方法正式实现时只负责目录和路径契约，不负责训练、不计算指标、
            不生成空 checkpoint 或 report。
        """

        root = Path(self.artifacts_root) / str(self.pair) / str(self.batchid) / "phase2"
        dataset_cache_dir = root / "dataset_cache"
        checkpoint_dir = root / "checkpoints"
        metrics_dir = root / "metrics"
        report_dir = root / "reports"
        diagnostics_dir = root / "diagnostics"
        validation_dir = root / "validation"

        for directory in (
            root,
            dataset_cache_dir,
            checkpoint_dir,
            metrics_dir,
            report_dir,
            diagnostics_dir,
            validation_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        self.artifact_paths = {
            "output_dir": root,
            "dataset_cache": dataset_cache_dir,
            "checkpoints": checkpoint_dir,
            "metrics": metrics_dir,
            "reports": report_dir,
            "diagnostics": diagnostics_dir,
            "validation_results": validation_dir,
            "best_checkpoint": checkpoint_dir / "best_checkpoint.pt",
            "best_validation_result": validation_dir / "best_validation_result.json",
            "phase2_selector_validation_html": (
                report_dir / "phase2_selector_validation.html"
            ),
        }

    def save_phase2_checkpoint(
        self,
        checkpoint: Phase2Checkpoint,
        *,
        checkpoint_name: str | None = None,
    ) -> Path:
        """保存 Phase II model checkpoint 的骨架入口。

        功能说明:
            后续实现应将 ``Phase2Checkpoint`` 转换为 torch/save 友好的 payload，
            并保存到 Phase II model checkpoint 目录。该 checkpoint 只包含
            Q-network、optimizer 和恢复训练所需配置，不包含 validation metrics。

        适用场景:
            ``Phase2DoubleDqnTrainer`` 在 checkpoint interval 或训练结束时调用，
            保存某个 epoch 的 selector 模型状态。

        参数:
            checkpoint: 待保存的 Phase II model checkpoint payload。
            checkpoint_name: 可选文件名或稳定 ID；为空时后续实现可按 epoch 生成。

        返回:
            保存后的 checkpoint 路径。

        设计边界:
            当前方法只固定接口，不创建目录、不序列化、不写文件。
        """

        path = self._phase2_checkpoint_path(
            epoch=checkpoint.epoch,
            checkpoint_name=checkpoint_name,
        )
        self._save_torch_payload(self._checkpoint_to_dict(checkpoint), path)
        return path

    def load_phase2_checkpoint(
        self,
        *,
        checkpoint_path: Path | None = None,
        epoch: int | None = None,
        best: bool = False,
    ) -> Phase2Checkpoint:
        """读取 Phase II model checkpoint 的骨架入口。

        功能说明:
            后续实现应从指定路径、epoch 或 best 标记解析 Phase II model checkpoint
            文件，并恢复为 ``Phase2Checkpoint`` 强类型对象。

        适用场景:
            恢复训练、加载 best selector 做 test evaluation，或后续阶段加载
            Phase II selector 时调用。

        参数:
            checkpoint_path: 显式 checkpoint 文件路径。
            epoch: 按训练 epoch 读取 checkpoint。
            best: 为 True 时读取 best model checkpoint。

        返回:
            ``Phase2Checkpoint``。

        设计边界:
            当前方法只固定接口，不读取文件、不反序列化、不校验路径。
        """

        if best:
            path = self._phase2_artifact_path("best_checkpoint")
        elif checkpoint_path is not None:
            path = Path(checkpoint_path)
        elif epoch is not None:
            path = self._phase2_checkpoint_path(epoch=epoch)
        else:
            raise ValueError("checkpoint_path, epoch, or best=True is required")

        if not path.exists():
            raise FileNotFoundError(f"phase2 checkpoint not found: {path}")
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        if isinstance(payload, Phase2Checkpoint):
            return payload
        if not isinstance(payload, Mapping):
            raise ValueError(f"invalid phase2 checkpoint payload: {path}")
        return self._checkpoint_from_dict(payload)

    def save_best_checkpoint(self, checkpoint: Phase2Checkpoint) -> Path:
        """保存 Phase II best model checkpoint。"""

        path = self._phase2_artifact_path("best_checkpoint")
        self._save_torch_payload(self._checkpoint_to_dict(checkpoint), path)
        return path

    def save_phase2_validation_result(
        self,
        validation_result: Phase2ValidationResult,
        *,
        split_name: str = "validation",
        epoch: int | None = None,
    ) -> Path:
        """保存 Phase II validation result 的骨架入口。

        功能说明:
            后续实现应把 ``Phase2ValidationResult.metrics``、``layers``、
            ``layer_computations`` 和 ``payloads`` 保存为 JSON 友好的评估结果文件。
            它只保存评估结果，不保存模型权重。

        适用场景:
            ``Phase2Evaluator.evaluate()`` 产出 validation/test 结果后调用，供
            report、checkpoint selector 和审计流程复用。

        参数:
            validation_result: 待保存的 validation/test 评估结果。
            split_name: 数据 split 名称，例如 ``"validation"`` 或 ``"test"``。
            epoch: 结果对应的训练 epoch；离线 test 或未知 epoch 时可为 None。

        返回:
            保存后的 validation result 路径。

        设计边界:
            当前方法只固定接口，不创建目录、不转 dict、不写 JSON。
        """

        path = self._phase2_validation_result_path(
            split_name=split_name,
            epoch=epoch,
        )
        self._save_json_payload(self._validation_result_to_dict(validation_result), path)
        return path

    def load_phase2_validation_result(
        self,
        *,
        result_path: Path | None = None,
        split_name: str = "validation",
        epoch: int | None = None,
        best: bool = False,
    ) -> Phase2ValidationResult:
        """读取 Phase II validation result 的骨架入口。

        功能说明:
            后续实现应从指定路径、split/epoch 或 best 标记读取评估结果文件，并恢复
            为 ``Phase2ValidationResult`` 强类型对象。

        适用场景:
            checkpoint selector 读取历史 validation 结果做排序，report 读取
            validation/test 结果构建展示上下文，或主流程恢复已完成评估结果时调用。

        参数:
            result_path: 显式 validation result 文件路径。
            split_name: 数据 split 名称。
            epoch: 结果对应的训练 epoch。
            best: 为 True 时读取 best checkpoint 对应的 validation result。

        返回:
            ``Phase2ValidationResult``。

        设计边界:
            当前方法只固定接口，不读取文件、不解析 JSON、不应用选择逻辑。
        """

        if best:
            path = self._phase2_artifact_path("best_validation_result")
        elif result_path is not None:
            path = Path(result_path)
        else:
            path = self._phase2_validation_result_path(
                split_name=split_name,
                epoch=epoch,
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"invalid phase2 validation result payload: {path}")
        return self._validation_result_from_dict(payload)

    def save_best_validation_result(
        self,
        validation_result: Phase2ValidationResult,
    ) -> Path:
        """保存被 selector 选中的 validation result 摘要。"""

        path = self._phase2_artifact_path("best_validation_result")
        self._save_json_payload(self._validation_result_to_dict(validation_result), path)
        return path

    def save_phase2_selector_validation_html(
        self,
        *,
        validation_result: Phase2ValidationResult,
        html: str,
        split_name: str | None = None,
        epoch: int | None = None,
    ) -> Path:
        """保存单个 Phase II selector validation HTML 报告。"""
        path = self._phase2_selector_validation_html_path(
            split_name=split_name,
            epoch=epoch,
        )
        self._save_text_payload(html, path)
        return path

    def _phase2_artifact_path(self, key: str) -> Path:
        path = self.artifact_paths.get(key)
        if path is None:
            raise ValueError("artifact store must be initialized with phase2 paths")
        return Path(path)

    def _phase2_checkpoint_path(
        self,
        *,
        epoch: int,
        checkpoint_name: str | None = None,
    ) -> Path:
        checkpoint_dir = self._phase2_artifact_path("checkpoints")
        if checkpoint_name is not None:
            path = Path(checkpoint_name)
            return path if path.is_absolute() else checkpoint_dir / path
        return checkpoint_dir / f"phase2_epoch_{epoch:04d}.pt"

    def _phase2_validation_result_path(
        self,
        *,
        split_name: str,
        epoch: int | None,
    ) -> Path:
        validation_dir = self._phase2_artifact_path("validation_results")
        if epoch is None:
            return validation_dir / f"{split_name}_validation_result.json"
        return validation_dir / f"{split_name}_epoch_{epoch:04d}_validation.json"

    def _phase2_selector_validation_html_path(
        self,
        *,
        split_name: str,
        epoch: int | None,
    ) -> Path:
        report_dir = self._phase2_artifact_path("reports")
        if epoch is None:
            return report_dir / f"{split_name}_selector_validation.html"
        return report_dir / f"{split_name}_epoch_{epoch:04d}_selector_validation.html"

    def _save_torch_payload(self, payload: Mapping[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
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

    def _save_json_payload(self, payload: Mapping[str, Any], path: Path) -> None:
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

    def _write_sha256_sidecar(self, path: Path, digest: str) -> None:
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
        digest = hashlib.sha256()
        with path.open("rb") as payload_file:
            for chunk in iter(lambda: payload_file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _checkpoint_to_dict(checkpoint: Phase2Checkpoint) -> dict[str, Any]:
        return {
            "epoch": checkpoint.epoch,
            "config": asdict(checkpoint.config)
            if is_dataclass(checkpoint.config)
            else dict(checkpoint.config),
            "q_network_state_dict": dict(checkpoint.q_network_state_dict),
            "optimizer_state_dict": dict(checkpoint.optimizer_state_dict),
        }

    @staticmethod
    def _checkpoint_from_dict(payload: Mapping[str, Any]) -> Phase2Checkpoint:
        config_payload = payload.get("config", {})
        if isinstance(config_payload, Phase2TrainConfig):
            config = config_payload
        elif isinstance(config_payload, Mapping):
            config_fields = {field.name for field in fields(Phase2TrainConfig)}
            config = Phase2TrainConfig(
                **{
                    key: value
                    for key, value in config_payload.items()
                    if key in config_fields
                }
            )
        else:
            raise ValueError("invalid phase2 checkpoint payload: config must be a mapping")

        return Phase2Checkpoint(
            epoch=int(payload["epoch"]),
            config=config,
            q_network_state_dict=dict(payload["q_network_state_dict"]),
            optimizer_state_dict=dict(payload["optimizer_state_dict"]),
        )

    @staticmethod
    def _validation_result_to_dict(
        validation_result: Phase2ValidationResult,
    ) -> dict[str, Any]:
        return validation_result.to_dict()

    @staticmethod
    def _validation_result_from_dict(
        payload: Mapping[str, Any],
    ) -> Phase2ValidationResult:
        return Phase2ValidationResult.from_dict(payload)

    @classmethod
    def _json_safe(cls, value: Any) -> Any:
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
