from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..model.data_types import ArtifactPaths
from ..store.artifact_store import DataFileStore


class Phase1FatalError(RuntimeError):
    pass


@dataclass(frozen=True)
class Phase1MainConfig:
    pair: str
    train_batch_id: str
    train_file: Path | None = None
    val_file: Path | None = None
    test_file: Path | None = None
    epochs: int = 100
    pretrain_epochs: int = 10
    batch_size: int = 256
    learning_rate: float = 1e-3
    device: str = "cuda"


class Phase1MainFlow:
    def __init__(self, config: Phase1MainConfig) -> None:
        self.config = config
        self.data_store = DataFileStore(
            pair=config.pair,
            batchid=config.train_batch_id,
        )

    def run(self) -> None:
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
        except Exception as exc:
            raise Phase1FatalError("phase1 main flow failed") from exc

    def prepare(self) -> None:
        self.data_store.initialize_phase1_artifact_dirs()

    def load_inputs(self) -> None:
        raise NotImplementedError(
            "build Phase I datasets, model and training components here"
        )

    def build_components(self) -> None:
        raise NotImplementedError(
            "build Phase I datasets, model and training components here"
        )

    def train(self) -> None:
        raise NotImplementedError(
            "implement pretrain loop, VQ training loop and checkpoint saving here"
        )

    def select_best_checkpoint(self) -> None:
        raise NotImplementedError(
            "implement Phase I checkpoint selection policy here"
        )

    def export_phase2_artifacts(self) -> None:
        raise NotImplementedError(
            "export encoder.pt, decoder.pt and codebook.pt from best checkpoint here"
        )

    def export_horizon_labels(self) -> None:
        raise NotImplementedError(
            "generate sampled_horizon_labels_{split}.feather files here"
        )

    def write_report(self) -> None:
        raise NotImplementedError("write phase1_report.json here")
