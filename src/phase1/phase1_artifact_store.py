"""Phase I 训练产出物读写类的接口骨架。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from ..model.vq_archetype import (
    ArchetypeActionDecoder,
    ArchetypeTrajectoryEncoder,
    VectorQuantizer,
)
from .checkpoint import (
    Phase1Checkpoint,
    Phase1CheckpointConfig,
    Phase1CheckpointMetrics,
    Phase1CheckpointStage,
    Phase1StateDict,
)
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
                周期性 epoch checkpoint 与 last/best checkpoint 的目录。
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
        best_checkpoint_path = checkpoint_dir / "best_checkpoint.pt"
        last_checkpoint_path = checkpoint_dir / "last_checkpoint.pt"

        for directory in (
            root,
            checkpoint_dir,
            diagnostics_dir,
            tensorboard_dir,
            latent_snapshot_dir,
            failure_case_dir,
            label_dir,
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
            "horizon_train_labels": label_dir / "sampled_horizon_labels_train.feather",
            "best_checkpoint": best_checkpoint_path,
            "last_checkpoint": last_checkpoint_path,
            "encoder": root / "encoder.pt",
            "decoder": root / "decoder.pt",
            "codebook": root / "codebook.pt",
            "phase1_report": root / "phase1_report.json",
        }

        return None

    def save_phase1_checkpoint(
        self,
        *,
        stage: Phase1CheckpointStage,
        epoch: int,
        config: Phase1CheckpointConfig,
        model_state_dict: Phase1StateDict,
        optimizer_state_dict: Phase1StateDict,
        metrics: Phase1CheckpointMetrics,
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
            optimizer_state_dict=optimizer_state_dict,
            metrics=metrics,
        )
        _ = self._phase1_checkpoint_path(stage=stage, epoch=epoch)
        checkpoint.to_dict()
        ...

    def load_phase1_checkpoint(
        self,
        *,
        stage: Phase1CheckpointStage | None = None,
        epoch: int | None = None,
        best: bool = False,
    ) -> Any:
        """读取 Phase I checkpoint。

        参数:
            stage: checkpoint 所属阶段；读取非 best checkpoint 时必填。
            epoch: checkpoint 所属 epoch；读取非 best checkpoint 时必填。
            best: 是否读取当前 best checkpoint。为 ``True`` 时忽略
                ``stage`` 和 ``epoch``。

        输出:
            返回 checkpoint 内容，供恢复训练、best 选择和模型导出使用。
        """

        if best:
            _ = self._phase1_best_checkpoint_path()
        else:
            if stage is None or epoch is None:
                raise ValueError(
                    "stage and epoch are required when loading a non-best checkpoint"
                )
            _ = self._phase1_checkpoint_path(stage=stage, epoch=epoch)
        ...

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

        _ = self._phase1_best_checkpoint_path()
        ...

    def save_phase1_encoder(
        self,
        encoder: ArchetypeTrajectoryEncoder,
    ) -> None:
        """保存 Phase I encoder 导出产物。

        参数:
            encoder: 训练完成后从 best checkpoint 导出的 trajectory encoder。

        方法作用:
            将 Phase II 离线 label 生成需要复用的 encoder 以统一方式写出。
        """

        _ = self._phase1_artifact_path("encoder")
        ...

    def load_phase1_encoder(
        self,
    ) -> ArchetypeTrajectoryEncoder:
        """读取 Phase I encoder 导出产物。

        输出:
            返回 trajectory encoder，供 Phase II label 生成或离线诊断加载。
        """

        _ = self._phase1_artifact_path("encoder")
        ...

    def save_phase1_decoder(
        self,
        decoder: ArchetypeActionDecoder,
    ) -> None:
        """保存 Phase I decoder 导出产物。

        参数:
            decoder: 训练完成后从 best checkpoint 导出的 causal action decoder。

        方法作用:
            将 Phase II/III 在线生成基础动作需要复用的 decoder 以统一方式写出。
        """

        _ = self._phase1_artifact_path("decoder")
        ...

    def load_phase1_decoder(
        self,
    ) -> ArchetypeActionDecoder:
        """读取 Phase I decoder 导出产物。

        输出:
            返回 causal action decoder，供 Phase II/III 或离线诊断加载。
        """

        _ = self._phase1_artifact_path("decoder")
        ...

    def save_phase1_codebook(
        self,
        codebook: VectorQuantizer,
    ) -> None:
        """保存 Phase I codebook 导出产物。

        参数:
            codebook: 训练完成后从 best checkpoint 导出的 VQ codebook。

        方法作用:
            将 Phase II/III 离散 archetype 选择和解码需要复用的 codebook
            以统一方式写出。
        """

        _ = self._phase1_artifact_path("codebook")
        ...

    def load_phase1_codebook(
        self,
    ) -> VectorQuantizer:
        """读取 Phase I codebook 导出产物。

        输出:
            返回 VQ codebook，供 Phase II/III 或离线诊断加载。
        """

        _ = self._phase1_artifact_path("codebook")
        ...

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


Phase1Store = Phase1ArtifactStore
