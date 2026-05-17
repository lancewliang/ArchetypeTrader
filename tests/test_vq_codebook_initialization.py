import torch

from src.model.codebook import (
    VectorQuantizer,
    classify_trajectory_directions,
)


def test_classify_trajectory_directions() -> None:
    actions = torch.tensor(
        [
            [0, 0, 1, 1],
            [1, 1, 1, 1],
            [2, 2, 1, 1],
            [0, 2, 1, 1],
        ]
    )

    directions = classify_trajectory_directions(actions)

    assert directions.tolist() == [0, 1, 2, 3]


def test_directional_kmeans_initializes_each_present_direction() -> None:
    generator = torch.Generator().manual_seed(7)
    cluster_means = {
        0: torch.tensor([-5.0, 0.0]),
        1: torch.tensor([0.0, 5.0]),
        2: torch.tensor([5.0, 0.0]),
        3: torch.tensor([0.0, -5.0]),
    }
    latents = []
    directions = []
    for direction, mean in cluster_means.items():
        points = mean + 0.05 * torch.randn(12, 2, generator=generator)
        latents.append(points)
        directions.extend([direction] * points.shape[0])

    quantizer = VectorQuantizer(num_archetypes=8, latent_dim=2)
    result = quantizer.initialize_from_directional_kmeans(
        torch.cat(latents, dim=0),
        torch.tensor(directions),
        random_state=11,
        n_init=2,
        max_iter=20,
    )

    centers = quantizer.embedding.weight.detach()
    for direction, mean in cluster_means.items():
        distances = torch.linalg.norm(centers - mean, dim=1)
        assert torch.min(distances).item() < 0.25
        assert result.direction_quotas[direction] >= 1

    assert centers.shape == (8, 2)
    assert sum(result.direction_quotas.values()) == 8


def test_dead_code_reset_uses_high_error_latents() -> None:
    quantizer = VectorQuantizer(num_archetypes=4, latent_dim=2)
    with torch.no_grad():
        quantizer.embedding.weight.zero_()

    latents = torch.tensor(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 20.0],
            [30.0, 0.0],
        ]
    )

    result = quantizer.reset_dead_codes(
        latents,
        min_occupancy=0.25,
        max_resets=2,
        random_state=17,
        jitter_scale=0.0,
    )

    assert result.dead_code_indices == (1, 2, 3)
    assert result.reset_code_indices == (1, 2)
    assert result.source_sample_indices == (3, 2)
    assert torch.allclose(quantizer.embedding.weight[1], latents[3])
    assert torch.allclose(quantizer.embedding.weight[2], latents[2])
    assert torch.allclose(quantizer.embedding.weight[3], torch.zeros(2))
