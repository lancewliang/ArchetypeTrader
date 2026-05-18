import numpy as np
import torch

from src.model.vq_archetype import ArchetypeVQModel
from src.phase1.losses import (
    Phase1LossConfig,
    build_phase1_auxiliary_label_store,
    compute_phase1_vq_training_loss,
)


def _toy_datasets():
    states = np.asarray(
        [
            [[0.0, 0.0], [0.1, 0.0], [0.2, 0.1], [0.3, 0.1], [0.4, 0.2], [0.5, 0.2]],
            [[0.0, 0.1], [0.0, 0.2], [0.1, 0.2], [0.1, 0.3], [0.2, 0.3], [0.2, 0.4]],
            [[0.5, 0.0], [0.4, 0.0], [0.3, 0.1], [0.2, 0.1], [0.1, 0.2], [0.0, 0.2]],
            [[0.2, 0.2], [0.3, 0.1], [0.2, 0.3], [0.3, 0.2], [0.2, 0.4], [0.3, 0.3]],
        ],
        dtype=np.float32,
    )
    prices = np.asarray(
        [
            [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            [100.0, 100.2, 100.4, 100.5, 100.6, 100.7],
            [105.0, 104.0, 103.0, 102.0, 101.0, 100.0],
            [100.0, 102.0, 99.0, 103.0, 98.0, 101.0],
        ],
        dtype=np.float32,
    )
    actions = np.asarray(
        [
            [1, 2, 2, 2, 2, 2],
            [1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0],
            [1, 2, 0, 2, 0, 1],
        ],
        dtype=np.int64,
    )
    rewards = np.zeros((states.shape[0], states.shape[1], 1), dtype=np.float32)
    trajectory_dataset = [
        (states[index], actions[index], rewards[index])
        for index in range(states.shape[0])
    ]
    horizon_dataset = (
        states,
        prices[..., np.newaxis],
        np.zeros((states.shape[0], states.shape[1], 20), dtype=np.float32),
    )
    return trajectory_dataset, horizon_dataset


def test_auxiliary_label_store_builds_pair_vocab_and_labels():
    trajectory_dataset, horizon_dataset = _toy_datasets()

    label_store = build_phase1_auxiliary_label_store(
        trajectory_dataset=trajectory_dataset,
        horizon_dataset=horizon_dataset,
    )

    assert label_store.morphology_labels.shape == (4,)
    assert label_store.pair_labels.shape == (4,)
    assert len(label_store.morphology_vocab) >= 1
    assert len(label_store.pair_vocab) >= 1
    assert all("with-recent-move" not in label for label in label_store.pair_vocab)
    assert all("against-recent-move" not in label for label in label_store.pair_vocab)
    assert all(" + " not in label for label in label_store.pair_vocab)


def test_phase1_vq_training_loss_supports_auxiliary_heads_and_backward():
    trajectory_dataset, horizon_dataset = _toy_datasets()
    label_store = build_phase1_auxiliary_label_store(
        trajectory_dataset=trajectory_dataset,
        horizon_dataset=horizon_dataset,
    )
    states = torch.as_tensor(np.stack([item[0] for item in trajectory_dataset]))
    actions = torch.as_tensor(np.stack([item[1] for item in trajectory_dataset]))
    rewards = torch.as_tensor(np.stack([item[2] for item in trajectory_dataset]))
    batch = (states, actions, rewards)
    model = ArchetypeVQModel(
        state_dim=2,
        action_dim=3,
        hidden_dim=8,
        latent_dim=4,
        num_archetypes=3,
        num_morphology_classes=len(label_store.morphology_vocab),
        num_pair_classes=len(label_store.pair_vocab),
    )

    outputs = model(batch)
    losses = compute_phase1_vq_training_loss(
        model=model,
        outputs=outputs,
        batch=batch,
        aux_labels=label_store.get(torch.arange(states.shape[0]), device="cpu"),
        config=Phase1LossConfig(
            morphology_aux_weight=0.1,
            pair_aux_weight=0.2,
            codebook_diversity_weight=0.001,
            prototype_diversity_weight=0.01,
            prototype_diversity_ref_samples=2,
        ),
    )

    assert outputs.morphology_logits is not None
    assert outputs.pair_logits is not None
    assert outputs.morphology_logits.shape == (4, len(label_store.morphology_vocab))
    assert outputs.pair_logits.shape == (4, len(label_store.pair_vocab))
    assert torch.isfinite(losses.total_loss)
    assert losses.total_loss >= losses.base_loss
    losses.total_loss.backward()
