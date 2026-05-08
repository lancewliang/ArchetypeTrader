"""Phase I Archetype Discovery 主流程骨架。

这个模块只负责第一阶段训练的流程编排，不直接实现数据采样、DP teacher
生成或模型细节。Phase I 训练应消费已经固化的数据准备产物，训练 VQ
archetype encoder-decoder，并导出 Phase II/III 需要的 codebook、decoder
和 horizon labels。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..model.data_types import ArtifactPaths
from ..store.artifact_store import DataFileStore


class Phase1FatalError(RuntimeError):
    """Phase I 训练遇到不可恢复错误。"""


@dataclass(frozen=True)
class Phase1MainConfig:
    """Phase I 主流程配置。

    这里保留主流程必需的最小配置。更细的模型、loss、checkpoint 和评估配置
    后续可以拆到独立 config 模块，再由入口脚本组装后传入。
    """

    pair: str
    train_batch_id: str
    output_dir: Path
    data_process_manifest: Path | None = None
    train_file: Path | None = None
    val_file: Path | None = None
    test_file: Path | None = None
    epochs: int = 100
    pretrain_epochs: int = 10
    batch_size: int = 256
    learning_rate: float = 1e-3
    device: str = "cpu"
    seed: int = 42
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Phase1TrainerArtifacts:
    """Phase I 主流程输出产物路径。"""

    output_dir: Path
    checkpoint_dir: Path
    best_checkpoint_path: Path | None
    encoder_path: Path | None
    decoder_path: Path | None
    codebook_path: Path | None
    report_path: Path | None
    label_paths: dict[str, Path] = field(default_factory=dict)


class Phase1MainFlow:
    """第一阶段训练主编排器。

    推荐调用顺序:
        1. ``prepare()``
        2. ``load_inputs()``
        3. ``build_components()``
        4. ``train()``
        5. ``select_best_checkpoint()``
        6. ``export_phase2_artifacts()``
        7. ``export_horizon_labels()``
        8. ``write_report()``

    ``run()`` 会按以上顺序执行，并返回所有关键产物路径。
    """

    def __init__(self, config: Phase1MainConfig) -> None:
        self.config = config
        self.data_store = DataFileStore() 
     
        self._inputs: dict[str, Any] = {}
        self._components: dict[str, Any] = {}
        self._metrics: dict[str, Any] = {}
        self._best_checkpoint_path: Path | None = None
        self._exported_paths: dict[str, Path] = {}
        self._label_paths: dict[str, Path] = {}

    def run(self) -> None:
        """执行 Phase I 主流程并返回产物路径。"""

        try:
            self.prepare()
            self.load_inputs()
            self.build_components()
            self.train()
            self.select_best_checkpoint()
            self.export_phase2_artifacts()
            self.export_horizon_labels()
            self.write_report()
        except Phase1FatalError:
            raise
        except Exception as exc:  # pragma: no cover - 顶层错误包装
            raise Phase1FatalError("phase1 main flow failed") from exc

      

    def prepare(self) -> None:
        """创建输出目录、初始化随机种子和运行上下文。"""

        self.data_store.ensure_artifact_dirs(self.artifact_paths)
        self._metrics["seed"] = self.config.seed

    def load_inputs(self) -> None:
        """加载 Phase I 训练输入。

        后续实现应优先消费 ``data_process_manifest``，并校验 train/val/test
        split、schema hash、sample_id 对齐和 actions/rewards 长度。兼容旧
        ``train_file``/``val_file``/``test_file`` 时，也应只在这里完成适配。
        """

        if self.config.data_process_manifest is None and self.config.train_file is None:
            raise Phase1FatalError(
                "either data_process_manifest or train_file must be provided"
            )

        self._inputs = {
            "manifest": self.config.data_process_manifest,
            "train_file": self.config.train_file,
            "val_file": self.config.val_file,
            "test_file": self.config.test_file,
        }

    def build_components(self) -> None:
        """构建 dataset、dataloader、model、optimizer、scheduler 和 evaluator。"""

        raise NotImplementedError(
            "build Phase I datasets, model and training components here"
        )

    def train(self) -> None:
        """执行 Phase A 预训练和 Phase B VQ 训练。"""

        raise NotImplementedError(
            "implement pretrain loop, VQ training loop and checkpoint saving here"
        )

    def select_best_checkpoint(self) -> None:
        """根据评估指标和 guardrail 选择 best checkpoint。"""

        raise NotImplementedError(
            "implement Phase I checkpoint selection policy here"
        )

    def export_phase2_artifacts(self) -> None:
        """导出 Phase II/III 复用的 encoder、decoder 和 codebook。"""

        self._exported_paths = {
            "encoder": self.artifact_paths["encoder"],
            "decoder": self.artifact_paths["decoder"],
            "codebook": self.artifact_paths["codebook"],
        }
        raise NotImplementedError(
            "export encoder.pt, decoder.pt and codebook.pt from best checkpoint here"
        )

    def export_horizon_labels(self) -> None:
        """使用 best checkpoint 为各 split 导出 horizon archetype labels。"""

        self._label_paths = {
            "train": self.artifact_paths["sampled_horizon_labels_train"],
            "val": self.artifact_paths["sampled_horizon_labels_val"],
            "test": self.artifact_paths["sampled_horizon_labels_test"],
        }
        raise NotImplementedError(
            "generate sampled_horizon_labels_{split}.feather files here"
        )

    def write_report(self) -> None:
        """写入 Phase I 训练报告。"""

        raise NotImplementedError("write phase1_report.json here")
