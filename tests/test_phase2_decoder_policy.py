import torch

from src.phase2.model.phase2_decoder_policy import FrozenArchetypeDecoderPolicy


class _FakeQuantizer:
    def embedding_from_code(self, selected_code_ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros(selected_code_ids.shape[0], 2, device=selected_code_ids.device)


class _RecordingDecoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.prefix_lengths: list[int] = []

    def forward(
        self,
        states: torch.Tensor,
        relative_states: torch.Tensor,
        trend_states: torch.Tensor,
        z_q: torch.Tensor,
    ) -> torch.Tensor:
        self.prefix_lengths.append(int(states.shape[1]))
        batch_size, prefix_length, _ = states.shape
        logits = torch.zeros(
            batch_size,
            prefix_length,
            3,
            device=states.device,
        )
        logits[:, -1, prefix_length % 3] = 1.0
        return logits


class _FakePhase1Model(torch.nn.Module):
    num_archetypes = 4

    def __init__(self) -> None:
        super().__init__()
        self.decoder = _RecordingDecoder()
        self.quantizer = _FakeQuantizer()


def test_decode_actions_generates_with_state_prefixes_only() -> None:
    model = _FakePhase1Model()
    policy = FrozenArchetypeDecoderPolicy(model, device="cpu")

    actions = policy.decode_actions(
        horizon_states=torch.zeros(2, 4, 3),
        horizon_relative_states=torch.zeros(2, 4, 2),
        horizon_trend_states=torch.zeros(2, 4, 1),
        selected_code_ids=torch.tensor([1, 2]),
    )

    assert model.decoder.prefix_lengths == [1, 2, 3, 4]
    torch.testing.assert_close(
        actions,
        torch.tensor(
            [
                [1, 2, 0, 1],
                [1, 2, 0, 1],
            ]
        ),
    )


def test_decode_all_codes_uses_stepwise_prefixes_for_expanded_codes() -> None:
    model = _FakePhase1Model()
    policy = FrozenArchetypeDecoderPolicy(model, device="cpu")

    actions = policy.decode_all_codes(
        horizon_states=torch.zeros(2, 3, 3),
        horizon_relative_states=torch.zeros(2, 3, 2),
        horizon_trend_states=torch.zeros(2, 3, 1),
    )

    assert model.decoder.prefix_lengths == [1, 2, 3]
    assert tuple(actions.shape) == (2, 4, 3)
