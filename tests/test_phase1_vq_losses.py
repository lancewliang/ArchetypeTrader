import torch

from src.model.vq_archetype import ArchetypeVQModel
from src.phase1.metrics.phase1_metrics import Phase1Metrics


def _batch() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(7)
    batch_size = 4
    horizon = 6
    return (
        torch.randn(batch_size, horizon, 5),
        torch.randn(batch_size, horizon, 3),
        torch.randn(batch_size, horizon, 2),
        torch.tensor(
            [
                [1, 2, 2, 2, 1, 1],
                [1, 0, 0, 1, 1, 1],
                [1, 1, 2, 1, 0, 1],
                [1, 1, 1, 1, 1, 1],
            ],
            dtype=torch.long,
        ),
        torch.tensor(
            [
                [0.0, 0.2, 1.0, 0.3, 0.0, 0.0],
                [0.0, -0.4, 0.8, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.6, -0.2, 0.5, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        torch.arange(batch_size, dtype=torch.long),
    )


def _model(**kwargs: float) -> ArchetypeVQModel:
    torch.manual_seed(11)
    return ArchetypeVQModel(
        state_dim=5,
        relative_state_dim=3,
        trend_state_dim=2,
        action_dim=3,
        hidden_dim=8,
        latent_dim=4,
        num_archetypes=3,
        **kwargs,
    )


def test_phase1_vq_total_loss_includes_execution_regularizers() -> None:
    model = _model(
        return_weighted_ce_weight=0.25,
        turnover_smooth_loss_weight=0.05,
        turnover_return_alignment_loss_weight=0.03,
    )

    outputs = model(_batch())

    expected = (
        outputs.reconstruction_loss
        + outputs.vq_loss
        + 0.25 * outputs.return_weighted_ce_loss
        + 0.05 * outputs.turnover_smooth_loss
        + 0.03 * outputs.turnover_return_alignment_loss
    )
    torch.testing.assert_close(outputs.total_loss, expected)
    assert outputs.return_weighted_ce_loss.item() > 0.0
    assert outputs.turnover_smooth_loss.item() >= 0.0
    assert outputs.turnover_return_alignment_loss.item() >= 0.0


def test_phase1_vq_loss_regularizers_can_be_disabled() -> None:
    model = _model(
        return_weighted_ce_weight=0.0,
        turnover_smooth_loss_weight=0.0,
        turnover_return_alignment_loss_weight=0.0,
    )

    outputs = model(_batch())

    torch.testing.assert_close(outputs.return_weighted_ce_loss, torch.zeros(()))
    torch.testing.assert_close(outputs.turnover_smooth_loss, torch.zeros(()))
    torch.testing.assert_close(outputs.turnover_return_alignment_loss, torch.zeros(()))
    torch.testing.assert_close(
        outputs.total_loss,
        outputs.reconstruction_loss + outputs.vq_loss,
    )


def test_phase1_metrics_records_execution_regularizers() -> None:
    model = _model(
        return_weighted_ce_weight=0.25,
        turnover_smooth_loss_weight=0.05,
        turnover_return_alignment_loss_weight=0.03,
    )
    batch = _batch()
    outputs = model(batch)
    metrics = Phase1Metrics(stage="vq", split="train", epoch=1)

    metrics.add_batch(batch_size=batch[0].shape[0], outputs=outputs, actions=batch[3])
    payload = metrics.averaged().to_dict()

    assert payload["return_weighted_ce_loss"] > 0.0
    assert payload["turnover_smooth_loss"] >= 0.0
    assert payload["turnover_return_alignment_loss"] >= 0.0
