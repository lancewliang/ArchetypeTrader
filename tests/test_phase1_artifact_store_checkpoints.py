from __future__ import annotations

import hashlib

import torch

from src.phase1.checkpoint import Phase1Checkpoint
from src.phase1.phase1_artifact_store import Phase1ArtifactStore


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as payload_file:
        for chunk in iter(lambda: payload_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint(is_best: bool = False) -> Phase1Checkpoint:
    return Phase1Checkpoint(
        stage="vq",
        epoch=3,
        is_best=is_best,
        config={"learning_rate": 0.001},
        model_state_dict={"weight": torch.tensor([1.0, 2.0])},
        optimizer_state_dict={"step": 3},
        metrics={"val": {"total_loss": 0.25}},
    )


def test_save_and_load_phase1_checkpoint_updates_epoch_and_last(tmp_path) -> None:
    store = Phase1ArtifactStore(
        pair="BTC",
        batchid="batch_001",
        artifacts_root=tmp_path,
    )
    store.initialize_phase1_artifact_dirs()

    store.save_phase1_checkpoint(
        stage="vq",
        epoch=3,
        config={"learning_rate": 0.001},
        model_state_dict={"weight": torch.tensor([1.0, 2.0])},
        optimizer_state_dict={"step": 3},
        metrics={"val": {"total_loss": 0.25}},
    )

    checkpoint_path = store.artifact_paths["checkpoints"] / "vq_epoch_0003.pt"
    last_checkpoint_path = store.artifact_paths["last_checkpoint"]

    loaded = store.load_phase1_checkpoint(stage="vq", epoch=3)
    last_loaded = torch.load(
        last_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    assert checkpoint_path.exists()
    assert loaded.stage == "vq"
    assert loaded.epoch == 3
    assert loaded.is_best is False
    assert loaded.metrics["val"]["total_loss"] == 0.25
    assert torch.equal(loaded.model_state_dict["weight"], torch.tensor([1.0, 2.0]))
    assert last_loaded["epoch"] == 3

    sidecar_path = checkpoint_path.with_suffix(".pt.sha256")
    assert sidecar_path.read_text(encoding="utf-8").startswith(_sha256(checkpoint_path))


def test_save_best_checkpoint_marks_payload_as_best(tmp_path) -> None:
    store = Phase1ArtifactStore(
        pair="BTC",
        batchid="batch_001",
        artifacts_root=tmp_path,
    )
    store.initialize_phase1_artifact_dirs()

    checkpoint = _checkpoint(is_best=False)
    store.save_best_checkpoint(checkpoint)

    loaded = store.load_phase1_checkpoint(best=True)
    best_checkpoint_path = store.artifact_paths["best_checkpoint"]

    assert checkpoint.is_best is False
    assert loaded.is_best is True
    assert loaded.stage == "vq"
    assert loaded.epoch == 3
    assert best_checkpoint_path.exists()
    assert best_checkpoint_path.with_suffix(".pt.sha256").exists()
